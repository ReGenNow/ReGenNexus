"""
RegenNexus UAP - UDP Transport

UDP multicast for LAN discovery and fast local messaging (1-5ms).
Ideal for device discovery on the same network.

Fallback chain:
1. mDNS/Zeroconf - works on mesh routers (plug-and-play)
2. Multicast (239.255.255.250) - standard LAN discovery
3. Broadcast (255.255.255.255) - works when multicast blocked
4. Direct peers (known_peers) - works across subnets

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

# mDNS/Zeroconf support (optional dependency)
try:
    from zeroconf import ServiceInfo, ServiceBrowser
    from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    logger.debug("zeroconf not installed - mDNS discovery disabled")


# mDNS service type for RegenNexus
MDNS_SERVICE_TYPE = "_regennexus._udp.local."


class RegenNexusServiceListener:
    """Listener for mDNS service discovery."""

    def __init__(self, transport: "UDPTransport", loop: asyncio.AbstractEventLoop):
        self.transport = transport
        self._loop = loop  # Store event loop reference for thread-safe scheduling

    def add_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:
        """Called when a service is discovered (from zeroconf thread)."""
        logger.debug(f"mDNS add_service callback: {name}")
        # Use run_coroutine_threadsafe to schedule on the main event loop
        try:
            asyncio.run_coroutine_threadsafe(
                self._handle_service(zc, service_type, name, "add"),
                self._loop
            )
        except Exception as e:
            logger.error(f"Error scheduling mDNS callback: {e}")

    def remove_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:
        """Called when a service is removed (from zeroconf thread)."""
        asyncio.run_coroutine_threadsafe(
            self._handle_service(zc, service_type, name, "remove"),
            self._loop
        )

    def update_service(self, zc: "Zeroconf", service_type: str, name: str) -> None:
        """Called when a service is updated (from zeroconf thread)."""
        asyncio.run_coroutine_threadsafe(
            self._handle_service(zc, service_type, name, "update"),
            self._loop
        )

    async def _handle_service(self, zc: "Zeroconf", service_type: str, name: str, event: str) -> None:
        """Handle service discovery event."""
        try:
            # Use AsyncServiceInfo for async-compatible service info lookup
            from zeroconf.asyncio import AsyncServiceInfo
            info = AsyncServiceInfo(service_type, name)
            await info.async_request(zc, 3000)  # 3 second timeout
            if info and info.addresses:
                # Get peer info from mDNS
                peer_id = info.properties.get(b"id", b"").decode("utf-8")
                entity_type = info.properties.get(b"type", b"device").decode("utf-8")
                if not peer_id or peer_id == self.transport._local_id:
                    return

                # Get IP address
                ip = socket.inet_ntoa(info.addresses[0])
                port = info.port
                addr = (ip, port)

                # Get WebSocket URL if available (for auto-connect)
                ws_url = None
                ws_port_bytes = info.properties.get(b"ws_port")
                ws_url_bytes = info.properties.get(b"ws_url")
                if ws_url_bytes:
                    ws_url = ws_url_bytes.decode("utf-8")
                elif ws_port_bytes:
                    ws_port = ws_port_bytes.decode("utf-8")
                    ws_url = f"ws://{ip}:{ws_port}"

                if event in ("add", "update"):
                    # Add peer
                    self.transport._peers[peer_id] = addr
                    self.transport._connected_peers.add(peer_id)
                    if addr not in self.transport._known_peers:
                        self.transport._known_peers.append(addr)
                    logger.info(f"mDNS discovered peer: {peer_id} at {ip}:{port}" + (f" (ws: {ws_url})" if ws_url else ""))

                    # Create synthetic announcement message to notify mesh layer
                    # Include WebSocket URL so mesh can auto-connect
                    import time
                    announce_msg = Message(
                        sender_id=peer_id,
                        recipient_id="*",
                        intent="mesh.announce",
                        content={
                            "node_id": peer_id,
                            "entity_type": entity_type,
                            "capabilities": [],
                            "timestamp": time.time(),
                            "discovery": "mdns",
                            "ws_url": ws_url,  # Include WebSocket URL for auto-connect
                            "ip": ip,
                            "udp_port": port,
                        },
                    )
                    # Dispatch to handlers so mesh layer knows about this peer
                    await self.transport._dispatch_message(announce_msg)

                elif event == "remove":
                    # Remove peer
                    self.transport._peers.pop(peer_id, None)
                    self.transport._connected_peers.discard(peer_id)
                    logger.info(f"mDNS peer removed: {peer_id}")

                    # Create synthetic goodbye message
                    import time
                    goodbye_msg = Message(
                        sender_id=peer_id,
                        recipient_id="*",
                        intent="mesh.goodbye",
                        content={
                            "node_id": peer_id,
                            "timestamp": time.time(),
                        },
                    )
                    await self.transport._dispatch_message(goodbye_msg)

        except Exception as e:
            import traceback
            logger.warning(f"mDNS service handling error: {e}")
            logger.debug(f"mDNS traceback: {traceback.format_exc()}")


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
        self._local_info: Dict = {}  # Info to share via mDNS (ws_port, etc.)
        self._bound_port: int = 0  # Actual port bound (after auto-detect)

        # Discovery method flags
        self._mdns_enabled: bool = True  # mDNS is preferred method
        self._multicast_enabled: bool = True
        self._broadcast_enabled: bool = True
        self._direct_peers_enabled: bool = True

        # Known peers for direct discovery (list of "host:port" or just "host")
        self._known_peers: List[Tuple[str, int]] = []

        # Track which methods are working
        self._mdns_working: bool = False
        self._multicast_working: bool = False
        self._broadcast_working: bool = False

        # mDNS/Zeroconf instances
        self._async_zeroconf: Optional["AsyncZeroconf"] = None
        self._zeroconf = None  # Reference to sync zeroconf from AsyncZeroconf
        self._mdns_service_info: Optional["ServiceInfo"] = None
        self._mdns_browser: Optional["AsyncServiceBrowser"] = None
        self._mdns_listener: Optional["RegenNexusServiceListener"] = None

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
        for discovery. Auto-detects available port if default is busy.

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

                # Try to bind to configured port, fall back to auto-detect
                bound_port = self.config.udp_port
                try:
                    sock.bind(("", self.config.udp_port))
                except OSError as e:
                    logger.warning(f"Port {self.config.udp_port} busy, auto-detecting...")
                    # Find available port - try multiple ranges for cross-platform compatibility
                    # 5454-5500: preferred range (avoids mDNS port 5353 on Linux)
                    # 5353: fallback for Windows where 5454 may conflict
                    # 5354-5400: additional fallback range
                    fallback_ports = (
                        list(range(5454, 5501)) +  # Primary range
                        [5353] +                    # mDNS port (works on Windows)
                        list(range(5354, 5401))     # Secondary range
                    )
                    for port in fallback_ports:
                        if port == self.config.udp_port:
                            continue  # Already tried this one
                        try:
                            sock.bind(("", port))
                            bound_port = port
                            logger.info(f"Auto-selected UDP port: {bound_port}")
                            break
                        except OSError:
                            continue
                    else:
                        raise OSError(f"No available UDP port found (tried {len(fallback_ports)} ports)")

                # Store the actual bound port for mDNS and peer discovery
                self._bound_port = bound_port

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

                # Start mDNS discovery (runs in background)
                if self._mdns_enabled:
                    await self._start_mdns()

                methods = []
                if self._mdns_enabled and self._mdns_working:
                    methods.append("mDNS")
                if self._multicast_enabled:
                    methods.append(f"multicast({self.config.udp_multicast_group})")
                if self._broadcast_enabled:
                    methods.append("broadcast")
                if self._direct_peers_enabled and self._known_peers:
                    methods.append(f"direct({len(self._known_peers)} peers)")

                logger.info(
                    f"UDP transport started on port {self._bound_port}, "
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

            # Stop mDNS
            await self._stop_mdns()

            logger.info("UDP transport stopped")

    async def _start_mdns(self) -> None:
        """Start mDNS service registration and discovery."""
        if not ZEROCONF_AVAILABLE:
            logger.debug("zeroconf not available - mDNS disabled")
            return

        if not self._local_id:
            logger.debug("No local ID set - mDNS registration skipped")
            return

        try:
            # Get local IP address
            local_ip = self._get_local_ip()
            if not local_ip:
                logger.warning("Could not determine local IP - mDNS disabled")
                return

            # Create async Zeroconf instance with specific interface
            # This is critical for Linux systems with multiple interfaces (e.g., docker0, br0, etc.)
            # Without specifying the interface, Zeroconf may bind to the wrong one
            self._async_zeroconf = AsyncZeroconf(interfaces=[local_ip])
            self._zeroconf = self._async_zeroconf.zeroconf

            # Create service info for registration
            # Service name format: nodeid._regennexus._udp.local.
            service_name = f"{self._local_id}.{MDNS_SERVICE_TYPE}"

            # Build mDNS properties - include WebSocket port for auto-discovery
            mdns_properties = {
                "id": self._local_id.encode("utf-8"),
                "type": getattr(self, "_entity_type", "device").encode("utf-8"),
            }

            # Add WebSocket port from local_info if available
            ws_port = self._local_info.get("ws_port") or self.config.ws_port
            if ws_port:
                mdns_properties["ws_port"] = str(ws_port).encode("utf-8")
                mdns_properties["ws_url"] = f"ws://{local_ip}:{ws_port}".encode("utf-8")

            self._mdns_service_info = ServiceInfo(
                MDNS_SERVICE_TYPE,
                service_name,
                addresses=[socket.inet_aton(local_ip)],
                port=self._bound_port or self.config.udp_port,
                properties=mdns_properties,
                server=f"{self._local_id}.local.",
            )

            # Register our service (async)
            await self._async_zeroconf.async_register_service(self._mdns_service_info)
            logger.info(f"mDNS service registered: {service_name} at {local_ip}:{self._bound_port} (ws:{ws_port})")

            # Start browsing for other services
            # Use sync ServiceBrowser (works better cross-platform than AsyncServiceBrowser)
            # The callbacks are called from zeroconf's internal thread, so we use
            # run_coroutine_threadsafe to schedule onto the asyncio event loop
            # IMPORTANT: Use get_running_loop() not get_event_loop() - in Python 3.10+,
            # get_event_loop() may return a different loop when called inside asyncio.run()
            loop = asyncio.get_running_loop()
            self._mdns_listener = RegenNexusServiceListener(self, loop)
            self._mdns_browser = ServiceBrowser(
                self._zeroconf,
                MDNS_SERVICE_TYPE,
                self._mdns_listener
            )

            self._mdns_working = True
            logger.info("mDNS discovery started")

        except Exception as e:
            import traceback
            logger.warning(f"mDNS setup failed: {type(e).__name__}: {e}")
            logger.debug(f"mDNS traceback: {traceback.format_exc()}")
            self._mdns_working = False
            # Clean up on failure
            await self._stop_mdns()

    async def _stop_mdns(self) -> None:
        """Stop mDNS service and cleanup."""
        if not ZEROCONF_AVAILABLE:
            return

        try:
            # Unregister service
            if hasattr(self, '_async_zeroconf') and self._async_zeroconf and self._mdns_service_info:
                await self._async_zeroconf.async_unregister_service(self._mdns_service_info)
                logger.debug("mDNS service unregistered")

            # Close browser (sync ServiceBrowser uses cancel(), not async_cancel())
            if self._mdns_browser:
                self._mdns_browser.cancel()
                self._mdns_browser = None

            # Close async zeroconf
            if hasattr(self, '_async_zeroconf') and self._async_zeroconf:
                await self._async_zeroconf.async_close()
                self._async_zeroconf = None
                self._zeroconf = None

            self._mdns_service_info = None
            self._mdns_working = False

        except Exception as e:
            logger.debug(f"mDNS cleanup error: {e}")

    def _get_local_ip(self) -> Optional[str]:
        """Get the local IP address for mDNS registration.

        Prefers LAN IPs (192.168.x.x, 10.x.x.x, 172.16-31.x.x) over VPN/tunnel IPs.
        """
        def is_lan_ip(ip: str) -> bool:
            """Check if IP is a typical LAN address."""
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            try:
                first = int(parts[0])
                second = int(parts[1])
                # 192.168.x.x
                if first == 192 and second == 168:
                    return True
                # 10.x.x.x
                if first == 10:
                    return True
                # 172.16.x.x - 172.31.x.x
                if first == 172 and 16 <= second <= 31:
                    return True
                return False
            except ValueError:
                return False

        try:
            # First, try to find a LAN IP by connecting to the local gateway/broadcast
            # Try common LAN broadcast addresses
            lan_targets = [
                ("192.168.68.1", 80),   # Common gateway
                ("192.168.1.1", 80),    # Common gateway
                ("192.168.0.1", 80),    # Common gateway
                ("10.0.0.1", 80),       # Common gateway
            ]

            for target_ip, port in lan_targets:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(0.1)
                    s.connect((target_ip, port))
                    local_ip = s.getsockname()[0]
                    s.close()
                    if is_lan_ip(local_ip):
                        logger.debug(f"Found LAN IP via {target_ip}: {local_ip}")
                        return local_ip
                except Exception:
                    continue

            # Fallback: Connect to public IP and check if it's a LAN address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            try:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            finally:
                s.close()

            # If we got a LAN IP, use it
            if is_lan_ip(local_ip):
                return local_ip

            # Otherwise, log warning but still return it (might be VPN-only network)
            logger.debug(f"Detected non-LAN IP: {local_ip} (VPN/tunnel?)")
            return local_ip

        except Exception:
            # Fallback: try to get from hostname
            try:
                hostname = socket.gethostname()
                return socket.gethostbyname(hostname)
            except Exception:
                return None

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
        # Skip non-JSON packets (mDNS binary protocol, etc.)
        # JSON always starts with { or [ in ASCII (0x7b or 0x5b)
        if not data or data[0] not in (0x7b, 0x5b):  # '{' or '['
            return  # Silently ignore non-JSON packets

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

    def set_local_info(self, info: Dict) -> None:
        """
        Set local node info for mDNS announcements.

        Args:
            info: Dict with ws_port, entity_type, capabilities, etc.
        """
        self._local_info = info

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