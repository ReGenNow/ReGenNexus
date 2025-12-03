"""
RegenNexus UAP - UDP Transport

UDP multicast for LAN discovery and fast local messaging (1-5ms).
Ideal for device discovery on the same network.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import socket
import struct
import time
import logging
from typing import Dict, Optional, Tuple

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

    Uses multicast for device discovery and direct UDP for
    point-to-point messaging. Latency is typically 1-5ms.

    Features:
    - Multicast discovery
    - Broadcast messaging
    - Low latency
    - No connection overhead
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        super().__init__(config)
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[UDPProtocol] = None
        self._peers: Dict[str, Tuple[str, int]] = {}  # peer_id -> (host, port)
        self._discovery_task: Optional[asyncio.Task] = None
        self._local_id: Optional[str] = None

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

                # Join multicast group
                group = socket.inet_aton(self.config.udp_multicast_group)
                mreq = struct.pack("4sL", group, socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

                # Set multicast TTL (1 = local network only)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)

                # Create asyncio transport
                self._transport, self._protocol = await loop.create_datagram_endpoint(
                    lambda: UDPProtocol(self),
                    sock=sock
                )

                self._state = TransportState.CONNECTED
                self._stats.connect_time = time.time()

                # Start discovery broadcast
                self._discovery_task = asyncio.create_task(self._discovery_loop())

                logger.info(
                    f"UDP transport started on port {self.config.udp_port}, "
                    f"multicast group {self.config.udp_multicast_group}"
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
        """Periodically broadcast discovery messages."""
        try:
            while self._state == TransportState.CONNECTED:
                # Send discovery announcement
                if self._local_id:
                    await self._send_discovery()

                await asyncio.sleep(self.config.udp_broadcast_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Discovery loop error: {e}")

    async def _send_discovery(self) -> None:
        """Send discovery announcement."""
        if not self._transport or not self._local_id:
            return

        discovery_msg = {
            "type": "discovery",
            "id": self._local_id,
            "timestamp": time.time(),
        }

        data = json.dumps(discovery_msg).encode("utf-8")
        addr = (self.config.udp_multicast_group, self.config.udp_port)

        try:
            self._transport.sendto(data, addr)
        except Exception as e:
            logger.error(f"Discovery send error: {e}")

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
                    self._peers[peer_id] = addr
                    self._connected_peers.add(peer_id)
                    logger.debug(f"Discovered peer: {peer_id} at {addr}")
                return

            # Handle regular messages
            msg = Message.from_dict(payload)

            # Update peer info
            if msg.source:
                self._peers[msg.source] = addr
                self._connected_peers.add(msg.source)

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
                # Send to multicast group
                addr = (self.config.udp_multicast_group, self.config.udp_port)
                self._transport.sendto(data, addr)

            self._update_send_stats(len(data))
            self._record_latency(time.time() - start_time)
            return True

        except Exception as e:
            logger.error(f"UDP send error: {e}")
            self._stats.errors += 1
            return False

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all peers via multicast.

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
