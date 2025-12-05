"""
RegenNexus UAP - UDP Transport

UDP multicast for LAN discovery and fast local messaging (1-5ms).
Ideal for device discovery on the same network.

Fallback chain:
1. Multicast (239.255.255.250) - standard LAN discovery
2. Broadcast (255.255.255.255) - works when multicast blocked
3. Direct peers (known_peers) - works across subnets

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import socket
import struct
import time
import logging
from typing import Dict, List, Optional, Tuple

from regennexus.core.message import Message
from regennexus.transport.base import (
    Transport,
    TransportConfig,
    TransportState,
)

logger = logging.getLogger(__name__)


class UDPProtocol(asyncio.DatagramProtocol):
    """Asyncio protocol for UDP communication."""

    def __init__(self, transport_instance: "UDPTransport"):
        self.transport_instance = transport_instance
        self.transport = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Called when connection is established."""
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Called when a datagram is received."""
        asyncio.create_task(
            self.transport_instance._handle_datagram(data, addr)
        )

    def error_received(self, exc: Exception) -> None:
        """Called when an error occurs."""
        logger.error(f"UDP error: {exc}")
        self.transport_instance._stats.errors += 1

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Called when connection is lost."""
        if exc:
            logger.error(f"UDP connection lost: {exc}")


class UDPTransport(Transport):
    """
    UDP Transport for LAN communication.

    Uses multiple discovery methods simultaneously:
    - Multicast for standard LAN discovery
    - Broadcast as fallback when multicast is blocked
    - Direct peer IPs for cross-subnet discovery

    Features:
    - Multi-method discovery (multicast + broadcast + direct)
    - Automatic fallback
    - Low latency (1-5ms)
    - No connection overhead
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        super().__init__(config)
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[UDPProtocol] = None
        self._peers: Dict[str, Tuple[str, int]] = {}  # peer_id -> (host, port)
        self._discovery_task: Optional[asyncio.Task] = None
        self._local_id: Optional[str] = None
        
        # Discovery method flags
        self._multicast_enabled: bool = True
        self._broadcast_enabled: bool = True
        self._direct_peers_enabled: bool = True
        
        # Known peers for direct discovery (list of "host:port" or just "host")
        self._known_peers: List[Tuple[str, int]] = []
        
        # Track which methods are working
        self._multicast_working: bool = False
        self._broadcast_working: bool = False

    def add_known_peer(self, host: str, port: Optional[int] = None) -> None:
        """
        Add a known peer IP for direct discovery.
        
        Args:
            host: Peer IP address
            port: Peer port (uses config port if not specified)
        """
        peer_port = port or self.config.udp_port
        peer_addr = (host, peer_port)
        if peer_addr not in self._known_peers:
            self._known_peers.append(peer_addr)
            logger.debug(f"Added known peer: {host}:{peer_port}")

    def set_discovery_methods(
        self,
        multicast: bool = True,
        broadcast: bool = True,
        direct_peers: bool = True
    ) -> None:
        """
        Enable/disable discovery methods.
        
        Args:
            multicast: Enable multicast discovery
            broadcast: Enable broadcast discovery
            direct_peers: Enable direct peer discovery
        """
        self._multicast_enabled = multicast
        self._broadcast_enabled = broadcast
        self._direct_peers_enabled = direct_peers

    async def connect(self) -> bool:
        """
        Start UDP transport.

        Creates a UDP socket and joins the multicast group
        for discovery.

        Returns:
            True if started successfully
        """
        async with self._lock:
            if self._state == TransportState.CONNECTED:
                return True

            self._state = TransportState.CONNECTING

            try:
                loop = asyncio.get_event_loop()

                # Create UDP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                # Enable broadcast
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

                # Bind to port
                sock.bind(("", self.config.udp_port))

                # Try to join multicast group (may fail on some networks)
                if self._multicast_enabled:
                    try:
                        group = socket.inet_aton(self.config.udp_multicast_group)
                        mreq = struct.pack("4sL", group, socket.INADDR_ANY)
                        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                        self._multicast_working = True
                        logger.debug("Multicast enabled")
                    except Exception as e:
                        logger.warning(f"Multicast setup failed (will use broadcast): {e}")
                        self._multicast_working = False

                # Create asyncio transport
                self._transport, self._protocol = await loop.create_datagram_endpoint(
                    lambda: UDPProtocol(self),
                    sock=sock
                )

                self._state = TransportState.CONNECTED
                self._stats.connect_time = time.time()

                # Start discovery broadcast
                self._discovery_task = asyncio.create_task(self._discovery_loop())

                methods = []
                if self._multicast_enabled:
                    methods.append(f"multicast({self.config.udp_multicast_group})")
                if self._broadcast_enabled:
                    methods.append("broadcast")
                if self._direct_peers_enabled and self._known_peers:
                    methods.append(f"direct({len(self._known_peers)} peers)")

                logger.info(
                    f"UDP transport started on port {self.config.udp_port}, "
                    f"discovery: {', '.join(methods) or 'none'}"
                )
                return True

            except Exception as e:
                logger.error(f"UDP start error: {e}")
                self._state = TransportState.ERROR
                self._stats.errors += 1
                return False

    async def disconnect(self) -> None:
        """Stop UDP transport."""
        async with self._lock:
            self._state = TransportState.DISCONNECTED

            # Cancel discovery
            if self._discovery_task:
                self._discovery_task.cancel()
                try:
                    await self._discovery_task
                except asyncio.CancelledError:
                    pass

            # Close transport
            if self._transport:
                self._transport.close()
                self._transport = None
                self._protocol = None

            self._peers.clear()
            self._connected_peers.clear()

            logger.info("UDP transport stopped")

    async def _discovery_loop(self) -> None:
        """Periodically broadcast discovery messages via all methods."""
        try:
            while self._state == TransportState.CONNECTED:
                # Send discovery announcement via all enabled methods
                if self._local_id:
                    await self._send_discovery()

                await asyncio.sleep(self.config.udp_broadcast_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Discovery loop error: {e}")

    async def _send_discovery(self) -> None:
        """Send discovery announcement via all enabled methods."""
        if not self._transport or not self._local_id:
            return

        discovery_msg = {
            "type": "discovery",
            "id": self._local_id,
            "timestamp": time.time(),
        }

        data = json.dumps(discovery_msg).encode("utf-8")

        # Method 1: Multicast
        if self._multicast_enabled and self._multicast_working:
            try:
                addr = (self.config.udp_multicast_group, self.config.udp_port)
                self._transport.sendto(data, addr)
                logger.debug(f"Discovery sent via multicast to {addr}")
            except Exception as e:
                logger.debug(f"Multicast send failed: {e}")
                self._multicast_working = False

        # Method 2: Broadcast (255.255.255.255)
        if self._broadcast_enabled:
            try:
                addr = ("255.255.255.255", self.config.udp_port)
                self._transport.sendto(data, addr)
                logger.debug(f"Discovery sent via broadcast to {addr}")
            except Exception as e:
                logger.debug(f"Broadcast send failed: {e}")

        # Method 3: Direct peers
        if self._direct_peers_enabled and self._known_peers:
            for peer_addr in self._known_peers:
                try:
                    self._transport.sendto(data, peer_addr)
                    logger.debug(f"Discovery sent directly to {peer_addr}")
                except Exception as e:
                    logger.debug(f"Direct send to {peer_addr} failed: {e}")

    async def _handle_datagram(
        self,
        data: bytes,
        addr: Tuple[str, int]
    ) -> None:
        """
        Handle received datagram.

        Args:
            data: Raw datagram data
            addr: Sender address (host, port)
        """
        try:
            decoded = data.decode("utf-8")
            payload = json.loads(decoded)

            # Handle discovery messages
            if payload.get("type") == "discovery":
                peer_id = payload.get("id")
                if peer_id and peer_id != self._local_id:
                    # Add to known peers for direct communication
                    self._peers[peer_id] = addr
                    self._connected_peers.add(peer_id)
                    
                    # Also add to known_peers list for future direct discovery
                    if addr not in self._known_peers:
                        self._known_peers.append(addr)
                    
                    logger.debug(f"Discovered peer: {peer_id} at {addr}")
                return

            # Handle regular messages
            msg = Message.from_dict(payload)

            # Update peer info
            if msg.sender_id:
                self._peers[msg.sender_id] = addr
                self._connected_peers.add(msg.sender_id)
                
                # Auto-add to known peers
                if addr not in self._known_peers:
                    self._known_peers.append(addr)

            self._update_receive_stats(len(data))
            await self._dispatch_message(msg)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {addr}")
        except Exception as e:
            logger.error(f"Datagram handling error: {e}")
            self._stats.errors += 1

    def set_local_id(self, entity_id: str) -> None:
        """
        Set the local entity ID for discovery.

        Args:
            entity_id: Local entity's ID
        """
        self._local_id = entity_id

    async def send(self, message: Message, target: Optional[str] = None) -> bool:
        """
        Send a message via UDP.

        Args:
            message: Message to send
            target: Target peer ID

        Returns:
            True if send successful
        """
        if self._state != TransportState.CONNECTED or not self._transport:
            return False

        start_time = time.time()
        data = json.dumps(message.to_dict()).encode("utf-8")

        try:
            if target:
                # Send to specific peer
                if target in self._peers:
                    addr = self._peers[target]
                    self._transport.sendto(data, addr)
                else:
                    logger.warning(f"Unknown peer: {target}")
                    return False
            else:
                # Broadcast via all methods
                sent = False
                
                # Multicast
                if self._multicast_enabled and self._multicast_working:
                    try:
                        addr = (self.config.udp_multicast_group, self.config.udp_port)
                        self._transport.sendto(data, addr)
                        sent = True
                    except Exception as e:
                        logger.debug(f"Multicast send failed: {e}")
                
                # Broadcast
                if self._broadcast_enabled:
                    try:
                        addr = ("255.255.255.255", self.config.udp_port)
                        self._transport.sendto(data, addr)
                        sent = True
                    except Exception as e:
                        logger.debug(f"Broadcast send failed: {e}")
                
                # Direct to all known peers
                if self._direct_peers_enabled:
                    for peer_addr in self._known_peers:
                        try:
                            self._transport.sendto(data, peer_addr)
                            sent = True
                        except Exception:
                            pass
                
                if not sent:
                    return False

            self._update_send_stats(len(data))
            self._record_latency(time.time() - start_time)
            return True

        except Exception as e:
            logger.error(f"UDP send error: {e}")
            self._stats.errors += 1
            return False

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all peers via all methods.

        Args:
            message: Message to broadcast

        Returns:
            Number of known peers (actual delivery is best-effort)
        """
        if await self.send(message):
            return len(self._peers)
        return 0

    def get_peer_address(self, peer_id: str) -> Optional[Tuple[str, int]]:
        """
        Get the address of a known peer.

        Args:
            peer_id: Peer's entity ID

        Returns:
            (host, port) tuple or None if unknown
        """
        return self._peers.get(peer_id)