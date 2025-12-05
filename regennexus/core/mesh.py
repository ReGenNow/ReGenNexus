"""
RegenNexus UAP - Mesh Network Discovery

Automatic plug-and-play mesh networking. Devices find each other
on LAN via UDP multicast, across networks via WebSocket, and
communicate using the best available transport.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from regennexus.core.message import Message
from regennexus.transport.base import TransportConfig, TransportType
from regennexus.transport.auto import AutoTransport

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Node state in the mesh."""
    OFFLINE = "offline"
    DISCOVERING = "discovering"
    ONLINE = "online"
    ERROR = "error"


@dataclass
class MeshNode:
    """A node in the mesh network."""
    node_id: str
    entity_type: str
    capabilities: List[str] = field(default_factory=list)
    address: Optional[str] = None
    port: int = 0
    transport: TransportType = TransportType.UDP
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stale(self) -> bool:
        """Check if node hasn't been seen recently."""
        return time.time() - self.last_seen > 60.0  # 60 second timeout


@dataclass
class MeshConfig:
    """Mesh network configuration."""
    node_id: Optional[str] = None  # Auto-generated if None
    entity_type: str = "device"
    capabilities: List[str] = field(default_factory=list)

    # Discovery settings
    discovery_enabled: bool = True
    discovery_interval: float = 5.0  # seconds
    node_timeout: float = 60.0  # seconds before node considered offline

    # Transport settings
    auto_select_transport: bool = True
    udp_enabled: bool = True
    udp_port: int = 5353
    udp_multicast_group: str = "239.255.255.250"

    websocket_enabled: bool = True
    websocket_port: int = 8765

    ipc_enabled: bool = True

    # Hub settings (for internet connectivity)
    hub_url: Optional[str] = None  # Central hub for cross-network discovery


class MeshNetwork:
    """
    Automatic mesh network for RegenNexus UAP.

    Provides plug-and-play device discovery and communication:
    - UDP multicast for LAN discovery
    - WebSocket for remote/internet connections
    - Auto-selection of best transport per peer
    - Message routing between nodes

    Example:
        mesh = MeshNetwork(MeshConfig(
            node_id="raspi-001",
            entity_type="device",
            capabilities=["gpio", "camera"]
        ))

        await mesh.start()

        # Nodes auto-discover each other
        peers = mesh.get_peers()

        # Send message (transport auto-selected)
        await mesh.send("jetson-001", {"command": "capture"})

        # Broadcast to all
        await mesh.broadcast({"event": "sensor_update", "value": 23.5})
    """

    def __init__(self, config: Optional[MeshConfig] = None):
        """
        Initialize mesh network.

        Args:
            config: Mesh configuration (uses defaults if None)
        """
        self.config = config or MeshConfig()

        # Generate node ID if not provided
        if not self.config.node_id:
            self.config.node_id = f"node-{uuid.uuid4().hex[:8]}"

        self.node_id = self.config.node_id
        self.state = NodeState.OFFLINE

        # Known peers
        self._peers: Dict[str, MeshNode] = {}
        self._peer_lock = asyncio.Lock()

        # Transport
        self._transport: Optional[AutoTransport] = None

        # Tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Message handlers
        self._message_handlers: List[Callable] = []
        self._peer_handlers: List[Callable] = []

    async def start(self) -> bool:
        """
        Start mesh network.

        Initializes transport, starts discovery, and begins
        listening for peer announcements.

        Returns:
            True if started successfully
        """
        if self.state == NodeState.ONLINE:
            return True

        self.state = NodeState.DISCOVERING
        logger.info(f"Starting mesh network as {self.node_id}")

        try:
            # Create transport config
            transport_config = TransportConfig(
                udp_enabled=self.config.udp_enabled,
                udp_port=self.config.udp_port,
                udp_multicast_group=self.config.udp_multicast_group,
                udp_broadcast_interval=self.config.discovery_interval,
                websocket_enabled=self.config.websocket_enabled,
                ws_port=self.config.websocket_port,
                ipc_enabled=self.config.ipc_enabled,
            )

            # Create and connect transport
            self._transport = AutoTransport(transport_config)
            self._transport.set_local_id(self.node_id)
            self._transport.set_local_info({
                "entity_type": self.config.entity_type,
                "capabilities": self.config.capabilities,
            })
            self._transport.add_handler(self._handle_message)

            if not await self._transport.connect():
                logger.error("Failed to connect transport")
                self.state = NodeState.ERROR
                return False

            # Start discovery loop
            if self.config.discovery_enabled:
                self._discovery_task = asyncio.create_task(self._discovery_loop())

            # Start cleanup loop
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.state = NodeState.ONLINE
            logger.info(f"Mesh network online: {self.node_id}")

            # Announce ourselves
            await self._announce()

            return True

        except Exception as e:
            logger.error(f"Mesh start error: {e}")
            self.state = NodeState.ERROR
            return False

    async def stop(self) -> None:
        """Stop mesh network."""
        if self.state == NodeState.OFFLINE:
            return

        logger.info(f"Stopping mesh network: {self.node_id}")

        # Cancel tasks
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Send goodbye
        await self._announce(leaving=True)

        # Disconnect transport
        if self._transport:
            await self._transport.disconnect()
            self._transport = None

        self._peers.clear()
        self.state = NodeState.OFFLINE

        logger.info("Mesh network stopped")

    async def _discovery_loop(self) -> None:
        """Periodically announce presence."""
        try:
            while self.state == NodeState.ONLINE:
                await self._announce()
                await asyncio.sleep(self.config.discovery_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Discovery loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up stale peers."""
        try:
            while self.state == NodeState.ONLINE:
                await asyncio.sleep(30)
                await self._cleanup_stale_peers()
        except asyncio.CancelledError:
            pass

    async def _announce(self, leaving: bool = False) -> None:
        """
        Announce presence to the network.

        Args:
            leaving: If True, announce that we're leaving
        """
        if not self._transport:
            return

        announcement = Message(
            sender_id=self.node_id,
            recipient_id="*",
            intent="mesh.announce" if not leaving else "mesh.goodbye",
            content={
                "node_id": self.node_id,
                "entity_type": self.config.entity_type,
                "capabilities": self.config.capabilities,
                "timestamp": time.time(),
            },
        )

        await self._transport.broadcast(announcement)

    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming message.

        Args:
            message: Received message
        """
        try:
            # Handle mesh protocol messages
            if message.intent == "mesh.announce":
                await self._handle_announcement(message)
            elif message.intent == "mesh.goodbye":
                await self._handle_goodbye(message)
            elif message.intent == "mesh.ping":
                await self._handle_ping(message)
            elif message.intent == "mesh.pong":
                await self._handle_pong(message)
            else:
                # Regular message - dispatch to handlers
                for handler in self._message_handlers:
                    try:
                        result = handler(message)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")

        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _handle_announcement(self, message: Message) -> None:
        """Handle peer announcement."""
        content = message.content
        if not isinstance(content, dict):
            return

        node_id = content.get("node_id")
        if not node_id or node_id == self.node_id:
            return

        async with self._peer_lock:
            is_new = node_id not in self._peers

            # Create or update peer
            peer = MeshNode(
                node_id=node_id,
                entity_type=content.get("entity_type", "unknown"),
                capabilities=content.get("capabilities", []),
                last_seen=time.time(),
                metadata=content.get("metadata", {}),
            )

            # Get transport info if available
            if self._transport:
                transport_type = TransportType.UDP  # Default
                for ttype, t in self._transport._transports.items():
                    if node_id in t.peers:
                        transport_type = ttype
                        break
                peer.transport = transport_type

            self._peers[node_id] = peer

            if is_new:
                logger.info(f"Discovered peer: {node_id} ({peer.entity_type})")

                # Auto-connect via WebSocket if ws_url is provided (plug-and-play)
                ws_url = content.get("ws_url")
                if ws_url and self._transport:
                    # Check if we're not already connected via WebSocket
                    ws_transport = self._transport._transports.get(TransportType.WEBSOCKET)
                    if ws_transport and node_id not in ws_transport.peers:
                        logger.info(f"Auto-connecting to peer WebSocket: {ws_url}")
                        asyncio.create_task(self._auto_connect_websocket(ws_url, node_id))

                # Notify handlers
                for handler in self._peer_handlers:
                    try:
                        result = handler(peer, "connected")
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Peer handler error: {e}")

    async def _auto_connect_websocket(self, ws_url: str, peer_id: str) -> None:
        """
        Auto-connect to a peer's WebSocket for plug-and-play connectivity.

        Args:
            ws_url: WebSocket URL (ws://host:port)
            peer_id: Peer's node ID
        """
        try:
            if self._transport:
                success = await self._transport.connect_to_peer(ws_url)
                if success:
                    logger.info(f"Auto-connected to {peer_id} via WebSocket: {ws_url}")
                else:
                    logger.debug(f"Could not auto-connect to {peer_id} via WebSocket")
        except Exception as e:
            logger.debug(f"Auto-connect to {peer_id} failed: {e}")

    async def _handle_goodbye(self, message: Message) -> None:
        """Handle peer leaving."""
        content = message.content
        if not isinstance(content, dict):
            return

        node_id = content.get("node_id")
        if not node_id:
            return

        async with self._peer_lock:
            if node_id in self._peers:
                peer = self._peers.pop(node_id)
                logger.info(f"Peer left: {node_id}")

                for handler in self._peer_handlers:
                    try:
                        result = handler(peer, "disconnected")
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Peer handler error: {e}")

    async def _handle_ping(self, message: Message) -> None:
        """Handle ping request."""
        if not self._transport:
            return

        pong = Message(
            sender_id=self.node_id,
            recipient_id=message.sender_id,
            intent="mesh.pong",
            content={"timestamp": time.time()},
        )
        await self._transport.send(pong, message.sender_id)

    async def _handle_pong(self, message: Message) -> None:
        """Handle pong response."""
        # Update peer last_seen
        async with self._peer_lock:
            if message.sender_id in self._peers:
                self._peers[message.sender_id].last_seen = time.time()

    async def _cleanup_stale_peers(self) -> None:
        """Remove stale peers."""
        async with self._peer_lock:
            stale = [
                node_id for node_id, peer in self._peers.items()
                if time.time() - peer.last_seen > self.config.node_timeout
            ]

            for node_id in stale:
                peer = self._peers.pop(node_id)
                logger.info(f"Peer timed out: {node_id}")

                for handler in self._peer_handlers:
                    try:
                        result = handler(peer, "timeout")
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Peer handler error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    async def send(
        self,
        recipient_id: str,
        content: Any,
        intent: str = "message",
    ) -> bool:
        """
        Send message to a peer.

        Args:
            recipient_id: Target node ID
            content: Message content
            intent: Message intent

        Returns:
            True if sent successfully
        """
        if not self._transport or self.state != NodeState.ONLINE:
            return False

        message = Message(
            sender_id=self.node_id,
            recipient_id=recipient_id,
            intent=intent,
            content=content,
        )

        return await self._transport.send(message, recipient_id)

    async def broadcast(
        self,
        content: Any,
        intent: str = "broadcast",
    ) -> int:
        """
        Broadcast message to all peers.

        Args:
            content: Message content
            intent: Message intent

        Returns:
            Number of peers reached
        """
        if not self._transport or self.state != NodeState.ONLINE:
            return 0

        message = Message(
            sender_id=self.node_id,
            recipient_id="*",
            intent=intent,
            content=content,
        )

        return await self._transport.broadcast(message)

    async def ping(self, node_id: str) -> Optional[float]:
        """
        Ping a peer and measure latency.

        Args:
            node_id: Target node ID

        Returns:
            Round-trip time in ms, or None if failed
        """
        if not self._transport or node_id not in self._peers:
            return None

        start = time.time()
        ping_msg = Message(
            sender_id=self.node_id,
            recipient_id=node_id,
            intent="mesh.ping",
            content={"timestamp": start},
        )

        if await self._transport.send(ping_msg, node_id):
            # Wait for pong (simplified - real impl would use future)
            await asyncio.sleep(0.1)
            return (time.time() - start) * 1000
        return None

    def get_peers(self) -> List[MeshNode]:
        """Get list of known peers."""
        return list(self._peers.values())

    def get_peer(self, node_id: str) -> Optional[MeshNode]:
        """Get a specific peer."""
        return self._peers.get(node_id)

    def find_by_capability(self, capability: str) -> List[MeshNode]:
        """Find peers with a specific capability."""
        return [
            peer for peer in self._peers.values()
            if capability in peer.capabilities
        ]

    def find_by_type(self, entity_type: str) -> List[MeshNode]:
        """Find peers of a specific type."""
        return [
            peer for peer in self._peers.values()
            if peer.entity_type == entity_type
        ]

    def on_message(self, handler: Callable) -> None:
        """
        Register message handler.

        Args:
            handler: async def handler(message: Message)
        """
        self._message_handlers.append(handler)

    def on_peer(self, handler: Callable) -> None:
        """
        Register peer event handler.

        Args:
            handler: async def handler(peer: MeshNode, event: str)
                     event is "connected", "disconnected", or "timeout"
        """
        self._peer_handlers.append(handler)

    @property
    def peer_count(self) -> int:
        """Number of known peers."""
        return len(self._peers)

    @property
    def is_online(self) -> bool:
        """Check if mesh is online."""
        return self.state == NodeState.ONLINE

    def get_stats(self) -> Dict[str, Any]:
        """Get mesh network statistics."""
        stats = {
            "node_id": self.node_id,
            "state": self.state.value,
            "peer_count": len(self._peers),
            "peers": [
                {
                    "id": p.node_id,
                    "type": p.entity_type,
                    "capabilities": p.capabilities,
                    "transport": p.transport.value,
                    "last_seen": p.last_seen,
                }
                for p in self._peers.values()
            ],
        }

        if self._transport:
            stats["transport_stats"] = self._transport.get_transport_stats()

        return stats
