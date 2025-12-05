"""
RegenNexus UAP - Base Transport

Abstract base class for all transport implementations.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import time
import logging

from regennexus.core.message import Message

logger = logging.getLogger(__name__)


def find_available_port(start: int, end: int, protocol: str = "udp") -> Optional[int]:
    """
    Find an available port in the given range.

    Args:
        start: Start of port range
        end: End of port range
        protocol: 'udp' or 'tcp'

    Returns:
        Available port number, or None if no port available
    """
    sock_type = socket.SOCK_DGRAM if protocol == "udp" else socket.SOCK_STREAM

    for port in range(start, end + 1):
        try:
            sock = socket.socket(socket.AF_INET, sock_type)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            continue

    return None


def is_port_available(port: int, protocol: str = "udp") -> bool:
    """
    Check if a specific port is available.

    Args:
        port: Port number to check
        protocol: 'udp' or 'tcp'

    Returns:
        True if port is available
    """
    sock_type = socket.SOCK_DGRAM if protocol == "udp" else socket.SOCK_STREAM

    try:
        sock = socket.socket(socket.AF_INET, sock_type)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
        sock.close()
        return True
    except OSError:
        return False


class TransportState(Enum):
    """Transport connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()
    CLOSED = auto()


class TransportType(Enum):
    """Types of transport available."""
    IPC = "ipc"              # Local inter-process (<0.1ms)
    UDP = "udp"              # UDP multicast (1-5ms)
    WEBSOCKET = "websocket"  # WebSocket (10-50ms)
    QUEUE = "queue"          # Message queue (reliable)
    AUTO = "auto"            # Auto-select best


@dataclass
class TransportConfig:
    """Configuration for a transport."""

    type: TransportType = TransportType.AUTO
    
    # Transport enable flags (add these)
    udp_enabled: bool = True
    websocket_enabled: bool = True
    ipc_enabled: bool = True


    # IPC settings
    ipc_socket_path: str = "/tmp/regennexus.sock"
    ipc_method: str = "unix_socket"  # unix_socket, shared_memory, named_pipe

    # UDP settings
    udp_multicast_group: str = "224.0.0.251"
    udp_port: int = 5454  # Changed from 5353 to avoid mDNS conflicts
    udp_port_range: tuple = (5454, 5500)  # Range for auto-detection
    udp_broadcast_interval: float = 5.0

    # WebSocket settings
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    ws_port_range: tuple = (8765, 8800)  # Range for auto-detection
    ws_ssl: bool = False
    ws_ssl_cert: str = ""
    ws_ssl_key: str = ""
    ws_ping_interval: float = 30.0
    ws_ping_timeout: float = 10.0

    # Queue settings
    queue_max_size: int = 1000
    queue_retry_attempts: int = 3
    queue_retry_delay: float = 5.0
    queue_persist: bool = False
    queue_persist_path: str = "./regennexus_queue"

    # General settings
    connect_timeout: float = 10.0
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    max_reconnect_attempts: int = 0  # 0 = unlimited
    buffer_size: int = 65536


@dataclass
class TransportStats:
    """Statistics for a transport connection."""

    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connect_time: Optional[float] = None
    last_activity: Optional[float] = None
    reconnect_count: int = 0
    errors: int = 0
    latency_samples: List[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples) * 1000

    @property
    def uptime(self) -> float:
        """Get connection uptime in seconds."""
        if self.connect_time is None:
            return 0.0
        return time.time() - self.connect_time


# Type for message handlers
MessageHandler = Callable[[Message], None]
AsyncMessageHandler = Callable[[Message], asyncio.Future]


class Transport(ABC):
    """
    Abstract base class for RegenNexus transports.

    All transport implementations must inherit from this class and
    implement the abstract methods.
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        """
        Initialize transport.

        Args:
            config: Transport configuration
        """
        self.config = config or TransportConfig()
        self._state = TransportState.DISCONNECTED
        self._stats = TransportStats()
        self._handlers: List[AsyncMessageHandler] = []
        self._connected_peers: Set[str] = set()
        self._lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> TransportState:
        """Get current transport state."""
        return self._state

    @property
    def stats(self) -> TransportStats:
        """Get transport statistics."""
        return self._stats

    @property
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self._state == TransportState.CONNECTED

    @property
    def peers(self) -> Set[str]:
        """Get set of connected peer IDs."""
        return self._connected_peers.copy()

    def add_handler(self, handler: AsyncMessageHandler) -> None:
        """
        Add a message handler.

        Args:
            handler: Async function to handle incoming messages
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: AsyncMessageHandler) -> None:
        """
        Remove a message handler.

        Args:
            handler: Handler to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def _dispatch_message(self, message: Message) -> None:
        """
        Dispatch a message to all handlers.

        Args:
            message: Message to dispatch
        """
        self._stats.messages_received += 1
        self._stats.last_activity = time.time()

        for handler in self._handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Handler error: {e}")
                self._stats.errors += 1

    def _update_send_stats(self, bytes_count: int) -> None:
        """Update statistics after sending."""
        self._stats.messages_sent += 1
        self._stats.bytes_sent += bytes_count
        self._stats.last_activity = time.time()

    def _update_receive_stats(self, bytes_count: int) -> None:
        """Update statistics after receiving."""
        self._stats.bytes_received += bytes_count

    def _record_latency(self, latency: float) -> None:
        """
        Record a latency sample.

        Args:
            latency: Latency in seconds
        """
        self._stats.latency_samples.append(latency)
        # Keep only last 100 samples
        if len(self._stats.latency_samples) > 100:
            self._stats.latency_samples = self._stats.latency_samples[-100:]

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect the transport.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect the transport."""
        pass

    @abstractmethod
    async def send(self, message: Message, target: Optional[str] = None) -> bool:
        """
        Send a message.

        Args:
            message: Message to send
            target: Optional target peer ID (None for broadcast)

        Returns:
            True if send successful
        """
        pass

    @abstractmethod
    async def broadcast(self, message: Message) -> int:
        """
        Broadcast a message to all peers.

        Args:
            message: Message to broadcast

        Returns:
            Number of peers message was sent to
        """
        pass

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect.

        Returns:
            True if reconnection successful
        """
        if self._state == TransportState.RECONNECTING:
            return False

        self._state = TransportState.RECONNECTING
        delay = self.config.reconnect_delay
        attempts = 0

        while True:
            attempts += 1
            self._stats.reconnect_count += 1

            logger.info(f"Reconnection attempt {attempts}...")

            if await self.connect():
                logger.info("Reconnected successfully")
                return True

            if (self.config.max_reconnect_attempts > 0 and
                attempts >= self.config.max_reconnect_attempts):
                logger.error("Max reconnection attempts reached")
                self._state = TransportState.ERROR
                return False

            # Exponential backoff
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.config.max_reconnect_delay)

    async def __aenter__(self) -> "Transport":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class TransportError(Exception):
    """Base exception for transport errors."""
    pass


class ConnectionError(TransportError):
    """Connection-related errors."""
    pass


class SendError(TransportError):
    """Send operation errors."""
    pass


class TimeoutError(TransportError):
    """Timeout errors."""
    pass
