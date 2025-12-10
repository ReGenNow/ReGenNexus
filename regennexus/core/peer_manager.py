"""
RegenNexus UAP - Peer Connection Manager

Manages persistent WebSocket connections to all mesh peers.
Uses PRESENCE-BASED approach - no continuous ping-pong flooding.

How it works:
1. When peer starts: sends "peer_announce" to all known peers
2. Receiving peer sends "peer_announce_ack" back
3. Connection is maintained by WebSocket protocol-level pings (built-in)
4. If connection drops, WebSocket detects it and we reconnect

NO continuous application-level ping-pong - that causes network flooding!

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import websockets
    from websockets.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from regennexus.core.message import Message

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Peer connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class PeerConnection:
    """Represents a connection to a peer."""
    peer_id: str
    ws_url: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    websocket: Optional[Any] = None  # WebSocket connection
    last_seen: float = 0.0  # Last time we received any message
    last_rtt: float = 0.0  # Round-trip time in ms (from on-demand pings)
    consecutive_failures: int = 0
    reconnect_attempts: int = 0
    entity_type: str = "unknown"
    capabilities: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.state == ConnectionState.CONNECTED and self.websocket is not None

    @property
    def latency_ms(self) -> float:
        """Get last measured latency in milliseconds."""
        return self.last_rtt


class PeerManager:
    """
    Manages persistent connections to all mesh peers.

    Uses PRESENCE-BASED discovery:
    - Peer sends announcement on connect
    - Other peers acknowledge
    - WebSocket protocol handles keepalive (ping_interval in ws_connect)
    - NO application-level ping flooding

    On-demand pings only for:
    - CLI benchmark command
    - Explicit ping requests

    Example:
        pm = PeerManager(local_id="Mac")
        await pm.start()

        # Add peer when discovered
        await pm.add_peer("pi-hole", "ws://192.168.68.93:8765")

        # Send message (uses persistent connection)
        await pm.send_to_peer("pi-hole", message)

        # On-demand ping (for benchmark)
        rtt = await pm.ping_peer("pi-hole")
    """

    def __init__(
        self,
        local_id: str,
        local_info: Optional[Dict] = None,
        reconnect_delays: Optional[List[float]] = None,
        on_peer_connected: Optional[Callable] = None,
        on_peer_disconnected: Optional[Callable] = None,
        on_message: Optional[Callable] = None,
    ):
        """
        Initialize PeerManager.

        Args:
            local_id: This node's ID
            local_info: Local node info (entity_type, capabilities)
            reconnect_delays: List of delays for reconnection backoff
            on_peer_connected: Callback when peer connects
            on_peer_disconnected: Callback when peer disconnects
            on_message: Callback for received messages
        """
        self.local_id = local_id
        self.local_info = local_info or {}
        self.reconnect_delays = reconnect_delays or [1, 2, 5, 10, 30, 60]

        # Callbacks
        self._on_peer_connected = on_peer_connected
        self._on_peer_disconnected = on_peer_disconnected
        self._on_message = on_message

        # Peer connections
        self._peers: Dict[str, PeerConnection] = {}
        self._peer_lock = asyncio.Lock()

        # Pending pings awaiting pong (for on-demand pings only)
        self._pending_pings: Dict[str, asyncio.Future] = {}

        # Tasks
        self._receive_tasks: Dict[str, asyncio.Task] = {}
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}

        # Running state
        self._running = False

    async def start(self) -> None:
        """Start the peer manager."""
        if self._running:
            return

        self._running = True
        logger.info(f"PeerManager started for {self.local_id} (presence-based, no ping flooding)")

    async def stop(self) -> None:
        """Stop the peer manager and close all connections."""
        self._running = False

        # Cancel all receive tasks (copy to list to avoid dict modification during iteration)
        for task in list(self._receive_tasks.values()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cancel all reconnect tasks (copy to list to avoid dict modification during iteration)
        for task in list(self._reconnect_tasks.values()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._peer_lock:
            for peer in self._peers.values():
                if peer.websocket:
                    try:
                        await peer.websocket.close()
                    except Exception:
                        pass
            self._peers.clear()

        logger.info("PeerManager stopped")

    async def add_peer(
        self,
        peer_id: str,
        ws_url: str,
        entity_type: str = "unknown",
        capabilities: Optional[List[str]] = None,
    ) -> bool:
        """
        Add a peer and establish persistent connection.

        Args:
            peer_id: Peer's node ID
            ws_url: WebSocket URL (ws://host:port)
            entity_type: Peer's entity type
            capabilities: Peer's capabilities

        Returns:
            True if connection established
        """
        async with self._peer_lock:
            # Check if already connected
            if peer_id in self._peers:
                existing = self._peers[peer_id]
                if existing.state == ConnectionState.CONNECTED:
                    # Update info but keep connection
                    existing.entity_type = entity_type
                    existing.capabilities = capabilities or []
                    return True

            # Create new peer connection
            peer = PeerConnection(
                peer_id=peer_id,
                ws_url=ws_url,
                entity_type=entity_type,
                capabilities=capabilities or [],
            )
            self._peers[peer_id] = peer

        # Connect in background
        asyncio.create_task(self._connect_to_peer(peer_id))
        return True

    async def remove_peer(self, peer_id: str) -> None:
        """Remove a peer and close its connection."""
        async with self._peer_lock:
            if peer_id not in self._peers:
                return

            peer = self._peers.pop(peer_id)

            # Cancel receive task
            if peer_id in self._receive_tasks:
                self._receive_tasks[peer_id].cancel()
                del self._receive_tasks[peer_id]

            # Cancel reconnect task
            if peer_id in self._reconnect_tasks:
                self._reconnect_tasks[peer_id].cancel()
                del self._reconnect_tasks[peer_id]

            # Close connection
            if peer.websocket:
                try:
                    await peer.websocket.close()
                except Exception:
                    pass

        logger.info(f"Removed peer: {peer_id}")

    async def send_to_peer(self, peer_id: str, message: Message) -> bool:
        """
        Send message to peer using persistent connection.

        Args:
            peer_id: Target peer ID
            message: Message to send

        Returns:
            True if sent successfully
        """
        async with self._peer_lock:
            if peer_id not in self._peers:
                logger.warning(f"Unknown peer: {peer_id}")
                return False

            peer = self._peers[peer_id]

            if peer.state != ConnectionState.CONNECTED or not peer.websocket:
                logger.warning(f"Peer not connected: {peer_id} (state={peer.state.value})")
                return False

            try:
                data = json.dumps(message.to_dict())
                await peer.websocket.send(data)
                return True
            except Exception as e:
                logger.error(f"Send to {peer_id} failed: {e}")
                # Mark for reconnection
                peer.state = ConnectionState.RECONNECTING
                peer.consecutive_failures += 1
                asyncio.create_task(self._reconnect_peer(peer_id))
                return False

    async def ping_peer(self, peer_id: str, timeout: float = 5.0) -> Optional[float]:
        """
        Ping peer and measure round-trip time (ON-DEMAND only, not continuous).

        This is used for:
        - CLI benchmark command
        - Explicit ping requests

        NOT for keepalive - WebSocket protocol handles that.

        Args:
            peer_id: Peer to ping
            timeout: Timeout in seconds

        Returns:
            RTT in milliseconds, or None if failed
        """
        async with self._peer_lock:
            if peer_id not in self._peers:
                return None
            peer = self._peers[peer_id]
            if peer.state != ConnectionState.CONNECTED:
                return None

        # Create ping message
        ping_time = time.time()
        ping_id = f"{self.local_id}_{ping_time}"

        ping_msg = Message(
            sender_id=self.local_id,
            recipient_id=peer_id,
            intent="mesh.ping",
            content={"ping_id": ping_id, "timestamp": ping_time},
        )

        # Create future for pong
        future = asyncio.get_event_loop().create_future()
        self._pending_pings[ping_id] = future

        try:
            # Send ping
            if not await self.send_to_peer(peer_id, ping_msg):
                return None

            # Wait for pong
            await asyncio.wait_for(future, timeout=timeout)
            rtt = (time.time() - ping_time) * 1000

            # Update peer stats
            async with self._peer_lock:
                if peer_id in self._peers:
                    self._peers[peer_id].last_rtt = rtt
                    self._peers[peer_id].last_seen = time.time()
                    self._peers[peer_id].consecutive_failures = 0

            return rtt

        except asyncio.TimeoutError:
            async with self._peer_lock:
                if peer_id in self._peers:
                    self._peers[peer_id].consecutive_failures += 1
            return None
        finally:
            self._pending_pings.pop(ping_id, None)

    def get_peer_status(self, peer_id: str) -> Optional[Dict]:
        """Get status of a peer connection."""
        if peer_id not in self._peers:
            return None

        peer = self._peers[peer_id]
        return {
            "peer_id": peer.peer_id,
            "ws_url": peer.ws_url,
            "state": peer.state.value,
            "is_healthy": peer.is_healthy,
            "latency_ms": peer.latency_ms,
            "last_seen": peer.last_seen,
            "consecutive_failures": peer.consecutive_failures,
            "entity_type": peer.entity_type,
            "capabilities": peer.capabilities,
        }

    def get_all_peers(self) -> List[Dict]:
        """Get status of all peer connections."""
        return [self.get_peer_status(pid) for pid in self._peers]

    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        return [
            pid for pid, peer in self._peers.items()
            if peer.state == ConnectionState.CONNECTED
        ]

    def get_healthy_peers(self) -> List[str]:
        """Get list of healthy peer IDs."""
        return [
            pid for pid, peer in self._peers.items()
            if peer.is_healthy
        ]

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _connect_to_peer(self, peer_id: str) -> bool:
        """Establish WebSocket connection to peer."""
        async with self._peer_lock:
            if peer_id not in self._peers:
                return False
            peer = self._peers[peer_id]
            peer.state = ConnectionState.CONNECTING

        try:
            logger.info(f"Connecting to peer {peer_id} at {peer.ws_url}")

            # WebSocket protocol-level ping handles keepalive (ping_interval=20)
            # No need for application-level ping flooding
            ws = await asyncio.wait_for(
                ws_connect(peer.ws_url, ping_interval=20, ping_timeout=10),
                timeout=10.0
            )

            # Wait for welcome message
            try:
                welcome_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                welcome_data = json.loads(welcome_raw)
                remote_id = welcome_data.get("content", {}).get("node_id")
                if remote_id and remote_id != peer_id:
                    logger.warning(f"Peer ID mismatch: expected {peer_id}, got {remote_id}")
            except Exception as e:
                logger.warning(f"No welcome from {peer_id}: {e}")

            # Send our announcement (this is our "hello")
            announce = Message(
                sender_id=self.local_id,
                recipient_id="*",
                intent="peer_announce",
                content={
                    "node_id": self.local_id,
                    **self.local_info
                }
            )
            await ws.send(json.dumps(announce.to_dict()))

            # Update peer state
            async with self._peer_lock:
                if peer_id in self._peers:
                    self._peers[peer_id].websocket = ws
                    self._peers[peer_id].state = ConnectionState.CONNECTED
                    self._peers[peer_id].last_seen = time.time()
                    self._peers[peer_id].consecutive_failures = 0
                    self._peers[peer_id].reconnect_attempts = 0

            # Start receive loop
            self._receive_tasks[peer_id] = asyncio.create_task(
                self._receive_loop(peer_id, ws)
            )

            logger.info(f"Connected to peer: {peer_id}")

            # Notify callback
            if self._on_peer_connected:
                try:
                    result = self._on_peer_connected(peer_id)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"on_peer_connected callback error: {e}")

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout to {peer_id}")
        except Exception as e:
            logger.warning(f"Connection to {peer_id} failed: {e}")

        # Mark for reconnection
        async with self._peer_lock:
            if peer_id in self._peers:
                self._peers[peer_id].state = ConnectionState.RECONNECTING
                self._peers[peer_id].consecutive_failures += 1

        # Schedule reconnect
        asyncio.create_task(self._reconnect_peer(peer_id))
        return False

    async def _reconnect_peer(self, peer_id: str) -> None:
        """Reconnect to peer with exponential backoff."""
        async with self._peer_lock:
            if peer_id not in self._peers:
                return
            peer = self._peers[peer_id]

            # Get backoff delay
            attempt = peer.reconnect_attempts
            delay_idx = min(attempt, len(self.reconnect_delays) - 1)
            delay = self.reconnect_delays[delay_idx]
            peer.reconnect_attempts += 1

        logger.info(f"Reconnecting to {peer_id} in {delay}s (attempt {attempt + 1})")
        await asyncio.sleep(delay)

        if not self._running:
            return

        # Try to connect
        await self._connect_to_peer(peer_id)

    async def _receive_loop(self, peer_id: str, ws) -> None:
        """Receive messages from peer."""
        try:
            async for raw_message in ws:
                try:
                    data = json.loads(raw_message)
                    msg = Message.from_dict(data)

                    # Update last_seen on ANY message
                    async with self._peer_lock:
                        if peer_id in self._peers:
                            self._peers[peer_id].last_seen = time.time()

                    # Handle pong response (for on-demand pings only)
                    if msg.intent in ("mesh.pong", "pong"):
                        ping_id = msg.content.get("ping_id") if msg.content else None

                        # Try to match by ping_id
                        if ping_id and ping_id in self._pending_pings:
                            try:
                                self._pending_pings[ping_id].set_result(True)
                            except asyncio.InvalidStateError:
                                pass

                        # Fallback for old nodes without ping_id
                        elif not ping_id:
                            for pid, future in list(self._pending_pings.items()):
                                if not future.done():
                                    try:
                                        future.set_result(True)
                                    except asyncio.InvalidStateError:
                                        pass
                                    break

                        # Don't forward pong to callback - it's handled above
                        continue

                    # Forward other messages to callback
                    if self._on_message:
                        try:
                            result = self._on_message(msg, peer_id)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.error(f"on_message callback error: {e}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {peer_id}")
                except Exception as e:
                    logger.error(f"Message handling error from {peer_id}: {e}")

        except ConnectionClosed:
            logger.info(f"Connection to {peer_id} closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error for {peer_id}: {e}")
        finally:
            # Mark disconnected
            async with self._peer_lock:
                if peer_id in self._peers:
                    self._peers[peer_id].state = ConnectionState.DISCONNECTED
                    self._peers[peer_id].websocket = None

            # Notify callback
            if self._on_peer_disconnected:
                try:
                    result = self._on_peer_disconnected(peer_id)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"on_peer_disconnected callback error: {e}")

            # Schedule reconnect if still running
            if self._running and peer_id in self._peers:
                asyncio.create_task(self._reconnect_peer(peer_id))

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all connected peers.

        Returns:
            Number of peers message was sent to
        """
        count = 0
        for peer_id in self.get_connected_peers():
            if await self.send_to_peer(peer_id, message):
                count += 1
        return count
