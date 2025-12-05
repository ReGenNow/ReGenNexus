"""
RegenNexus UAP - Auto Transport Selection

Automatically selects the best transport based on target location
and network conditions.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import socket
import time
import logging
from typing import Dict, List, Optional, Tuple

from regennexus.core.message import Message
from regennexus.transport.base import (
    Transport,
    TransportConfig,
    TransportState,
    TransportType,
)
from regennexus.transport.ipc import IPCTransport
from regennexus.transport.udp import UDPTransport
from regennexus.transport.websocket import WebSocketTransport
from regennexus.transport.queue import MessageQueueTransport

logger = logging.getLogger(__name__)


def is_local_address(host: str) -> bool:
    """
    Check if an address is local.

    Args:
        host: Hostname or IP address

    Returns:
        True if the address is local
    """
    try:
        # Check common local addresses
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True

        # Check if it's the same machine
        local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        if host in local_ips:
            return True

        # Resolve hostname
        resolved = socket.gethostbyname(host)
        if resolved in ("127.0.0.1", "::1") or resolved in local_ips:
            return True

        return False

    except socket.error:
        return False


def is_lan_address(host: str) -> bool:
    """
    Check if an address is on the local network.

    Args:
        host: Hostname or IP address

    Returns:
        True if the address is on the LAN
    """
    try:
        resolved = socket.gethostbyname(host)

        # Check private IP ranges
        octets = [int(x) for x in resolved.split(".")]

        # 10.0.0.0/8
        if octets[0] == 10:
            return True

        # 172.16.0.0/12
        if octets[0] == 172 and 16 <= octets[1] <= 31:
            return True

        # 192.168.0.0/16
        if octets[0] == 192 and octets[1] == 168:
            return True

        # 169.254.0.0/16 (link-local)
        if octets[0] == 169 and octets[1] == 254:
            return True

        return False

    except (socket.error, ValueError):
        return False


def select_best_transport(
    target: Optional[str] = None,
    config: Optional[TransportConfig] = None,
    require_reliable: bool = False
) -> TransportType:
    """
    Select the best transport for a given target.

    Args:
        target: Target address or None for local
        config: Transport configuration
        require_reliable: If True, prefer reliable transports

    Returns:
        Best TransportType for the situation
    """
    if require_reliable:
        return TransportType.QUEUE

    if target is None:
        # No target - use IPC for same-machine
        return TransportType.IPC

    if is_local_address(target):
        return TransportType.IPC
    elif is_lan_address(target):
        return TransportType.UDP
    else:
        return TransportType.WEBSOCKET


class AutoTransport(Transport):
    """
    Automatic Transport Selection.

    Manages multiple transports and automatically selects
    the best one for each message based on:
    - Target location (local, LAN, internet)
    - Reliability requirements
    - Latency preferences
    - Current network conditions
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        super().__init__(config)
        self._transports: Dict[TransportType, Transport] = {}
        self._peer_transports: Dict[str, TransportType] = {}
        self._local_id: Optional[str] = None
        self._local_info: Dict = {}

    async def connect(self) -> bool:
        """
        Initialize and connect all configured transports.

        Returns:
            True if at least one transport connected
        """
        async with self._lock:
            if self._state == TransportState.CONNECTED:
                return True

            self._state = TransportState.CONNECTING

            # Initialize transports based on config
            transports_to_init = []

            # IPC transport
            if self.config.ipc_enabled:
                ipc = IPCTransport(self.config)
                transports_to_init.append((TransportType.IPC, ipc))

            # UDP transport
            if self.config.udp_enabled:
                udp = UDPTransport(self.config)
                transports_to_init.append((TransportType.UDP, udp))

            # WebSocket transport
            if self.config.websocket_enabled:
                try:
                    ws = WebSocketTransport(
                        self.config,
                        server_mode=True
                    )
                    transports_to_init.append((TransportType.WEBSOCKET, ws))
                except ImportError:
                    logger.warning("WebSocket transport unavailable (install websockets)")

            # Set local_id and local_info on all transports BEFORE connecting (needed for mDNS and peer announce)
            if self._local_id:
                for _, transport in transports_to_init:
                    if hasattr(transport, "set_local_id"):
                        transport.set_local_id(self._local_id)
                    if hasattr(transport, "set_local_info") and self._local_info:
                        transport.set_local_info(self._local_info)

            # Connect transports in parallel
            connect_tasks = []
            for transport_type, transport in transports_to_init:
                connect_tasks.append(
                    self._connect_transport(transport_type, transport)
                )

            results = await asyncio.gather(*connect_tasks, return_exceptions=True)

            # Check results
            success_count = sum(1 for r in results if r is True)

            if success_count > 0:
                self._state = TransportState.CONNECTED
                self._stats.connect_time = time.time()
                logger.info(
                    f"Auto transport ready ({success_count} transports active)"
                )
                return True
            else:
                self._state = TransportState.ERROR
                logger.error("No transports available")
                return False

    async def _connect_transport(
        self,
        transport_type: TransportType,
        transport: Transport
    ) -> bool:
        """Connect a single transport."""
        try:
            if await transport.connect():
                self._transports[transport_type] = transport
                # Register message handler
                transport.add_handler(self._dispatch_message)
                return True
            return False
        except Exception as e:
            logger.warning(f"Transport {transport_type.value} failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect all transports."""
        async with self._lock:
            self._state = TransportState.DISCONNECTED

            # Disconnect all transports in parallel
            tasks = [t.disconnect() for t in self._transports.values()]
            await asyncio.gather(*tasks, return_exceptions=True)

            self._transports.clear()
            self._peer_transports.clear()
            self._connected_peers.clear()

            logger.info("Auto transport disconnected")

    def set_local_id(self, entity_id: str) -> None:
        """
        Set the local entity ID.

        Args:
            entity_id: Local entity's ID
        """
        self._local_id = entity_id

        # Propagate to transports
        for transport in self._transports.values():
            if hasattr(transport, "set_local_id"):
                transport.set_local_id(entity_id)

    def set_local_info(self, info: dict) -> None:
        """
        Set local node info for peer announcements.

        Args:
            info: Dict with entity_type, capabilities, etc.
        """
        self._local_info = info
        # Propagate to transports that support it
        for transport in self._transports.values():
            if hasattr(transport, "set_local_info"):
                transport.set_local_info(info)

    async def connect_to_peer(self, url: str) -> bool:
        """
        Connect to a remote peer via WebSocket.

        Args:
            url: WebSocket URL (ws://host:port)

        Returns:
            True if connection successful
        """
        ws_transport = self._transports.get(TransportType.WEBSOCKET)
        if ws_transport and hasattr(ws_transport, "connect_to_peer"):
            return await ws_transport.connect_to_peer(url)
        return False

    def _select_transport(
        self,
        target: Optional[str] = None,
        require_reliable: bool = False
    ) -> Optional[Transport]:
        """
        Select the best transport for a target.

        Args:
            target: Target peer ID or address
            require_reliable: If True, prefer reliable transport

        Returns:
            Best available transport or None
        """
        # Check if we know the best transport for this peer
        if target and target in self._peer_transports:
            preferred = self._peer_transports[target]
            if preferred in self._transports:
                return self._transports[preferred]

        # Check which transport actually knows this peer
        if target:
            for ttype, transport in self._transports.items():
                if target in transport.peers:
                    self._peer_transports[target] = ttype
                    return transport

        # Determine best transport type
        best_type = select_best_transport(
            target,
            self.config,
            require_reliable
        )

        # Fall back through available transports
        priority = [
            best_type,
            TransportType.WEBSOCKET,
            TransportType.UDP,
            TransportType.IPC,
        ]

        for transport_type in priority:
            if transport_type in self._transports:
                transport = self._transports[transport_type]
                if transport.is_connected:
                    return transport

        return None

    async def send(self, message: Message, target: Optional[str] = None) -> bool:
        """
        Send a message using the best transport.

        Args:
            message: Message to send
            target: Target peer ID

        Returns:
            True if send successful
        """
        if self._state != TransportState.CONNECTED:
            return False

        # Check if message requires reliable delivery
        require_reliable = message.metadata.get("reliable", False)

        # Select transport
        transport = self._select_transport(target, require_reliable)
        if not transport:
            logger.error("No transport available")
            return False

        # Send message
        start_time = time.time()
        success = await transport.send(message, target)

        if success:
            self._stats.messages_sent += 1
            self._record_latency(time.time() - start_time)

            # Remember which transport worked for this peer
            if target:
                for ttype, t in self._transports.items():
                    if t is transport:
                        self._peer_transports[target] = ttype
                        break

        return success

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message via all transports.

        Args:
            message: Message to broadcast

        Returns:
            Total number of peers reached
        """
        if self._state != TransportState.CONNECTED:
            return 0

        total = 0
        for transport in self._transports.values():
            if transport.is_connected:
                try:
                    count = await transport.broadcast(message)
                    total += count
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")

        self._stats.messages_sent += 1
        return total

    @property
    def peers(self) -> set:
        """Get all known peers from all transports."""
        all_peers = set()
        for transport in self._transports.values():
            all_peers.update(transport.peers)
        return all_peers

    def get_transport(self, transport_type: TransportType) -> Optional[Transport]:
        """
        Get a specific transport instance.

        Args:
            transport_type: Type of transport to get

        Returns:
            Transport instance or None
        """
        return self._transports.get(transport_type)

    def get_transport_stats(self) -> Dict[str, dict]:
        """
        Get statistics for all transports.

        Returns:
            Dictionary of transport stats
        """
        stats = {}
        for ttype, transport in self._transports.items():
            s = transport.stats
            stats[ttype.value] = {
                "connected": transport.is_connected,
                "messages_sent": s.messages_sent,
                "messages_received": s.messages_received,
                "bytes_sent": s.bytes_sent,
                "bytes_received": s.bytes_received,
                "avg_latency_ms": s.avg_latency,
                "errors": s.errors,
                "peers": len(transport.peers),
            }
        return stats