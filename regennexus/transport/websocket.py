"""
RegenNexus UAP - WebSocket Transport

WebSocket transport for internet and remote communication (10-50ms).
Supports both server and client modes with SSL/TLS.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import ssl
import time
import logging
from typing import Dict, Optional, Set

try:
    import websockets
    from websockets.server import serve as ws_serve
    from websockets.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from regennexus.core.message import Message
from regennexus.transport.base import (
    Transport,
    TransportConfig,
    TransportState,
)

logger = logging.getLogger(__name__)


class WebSocketTransport(Transport):
    """
    WebSocket Transport for remote communication.

    Provides reliable bidirectional communication over the internet.
    Latency is typically 10-50ms depending on network.

    Features:
    - Server and client modes
    - SSL/TLS encryption
    - Automatic reconnection
    - Ping/pong heartbeat
    - Bidirectional peer discovery
    """

    def __init__(
        self,
        config: Optional[TransportConfig] = None,
        server_mode: bool = True,
        server_url: Optional[str] = None
    ):
        """
        Initialize WebSocket transport.

        Args:
            config: Transport configuration
            server_mode: True to run as server, False as client
            server_url: Server URL for client mode (ws://host:port)
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        super().__init__(config)
        self._server_mode = server_mode
        self._server_url = server_url
        self._server = None
        self._client_ws = None
        self._clients: Dict[str, "websockets.WebSocketServerProtocol"] = {}
        self._outbound_clients: Dict[str, "websockets.WebSocketClientProtocol"] = {}
        self._peer_id_to_url: Dict[str, str] = {}  # Map peer node_id to outbound URL
        self._receive_task: Optional[asyncio.Task] = None
        self._outbound_tasks: Dict[str, asyncio.Task] = {}
        self._local_id: Optional[str] = None
        self._local_info: Dict = {}
        self._known_peer_urls: Set[str] = set()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: int = 30  # seconds
        self._peer_lookup_callback = None  # Callback to get real peer list from mesh

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context if configured."""
        if not self.config.ws_ssl:
            return None

        if not self.config.ws_ssl_cert or not self.config.ws_ssl_key:
            logger.warning("SSL enabled but cert/key not provided")
            return None

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if self._server_mode
                            else ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_cert_chain(
            self.config.ws_ssl_cert,
            self.config.ws_ssl_key
        )
        return ctx

    async def connect(self) -> bool:
        """
        Connect or start WebSocket transport.

        Returns:
            True if successful
        """
        async with self._lock:
            if self._state == TransportState.CONNECTED:
                return True

            self._state = TransportState.CONNECTING

            try:
                if self._server_mode:
                    return await self._start_server()
                else:
                    return await self._connect_client()

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self._state = TransportState.ERROR
                self._stats.errors += 1
                return False

    async def _start_server(self) -> bool:
        """Start WebSocket server."""
        try:
            ssl_ctx = self._create_ssl_context()

            self._server = await ws_serve(
                self._handle_client,
                self.config.ws_host,
                self.config.ws_port,
                ssl=ssl_ctx,
                ping_interval=self.config.ws_ping_interval,
                ping_timeout=self.config.ws_ping_timeout,
            )

            self._state = TransportState.CONNECTED
            self._stats.connect_time = time.time()

            protocol = "wss" if ssl_ctx else "ws"
            logger.info(
                f"WebSocket server started on "
                f"{protocol}://{self.config.ws_host}:{self.config.ws_port}"
            )
            return True

        except Exception as e:
            logger.error(f"WebSocket server start error: {e}")
            return False

    async def _connect_client(self) -> bool:
        """Connect as WebSocket client."""
        if not self._server_url:
            # Construct URL from config
            protocol = "wss" if self.config.ws_ssl else "ws"
            self._server_url = (
                f"{protocol}://{self.config.ws_host}:{self.config.ws_port}"
            )

        try:
            ssl_ctx = self._create_ssl_context() if self.config.ws_ssl else None

            self._client_ws = await asyncio.wait_for(
                ws_connect(
                    self._server_url,
                    ssl=ssl_ctx,
                    ping_interval=self.config.ws_ping_interval,
                    ping_timeout=self.config.ws_ping_timeout,
                ),
                timeout=self.config.connect_timeout
            )

            self._state = TransportState.CONNECTED
            self._stats.connect_time = time.time()

            # Send registration message
            if self._local_id:
                reg_msg = Message(
                    sender_id=self._local_id,
                    recipient_id="server",
                    intent="register",
                    content={"id": self._local_id}
                )
                await self._client_ws.send(json.dumps(reg_msg.to_dict()))

            # Start receive loop
            self._receive_task = asyncio.create_task(self._client_receive_loop())

            logger.info(f"Connected to WebSocket server: {self._server_url}")
            return True

        except asyncio.TimeoutError:
            logger.error("WebSocket connection timeout")
            return False
        except Exception as e:
            logger.error(f"WebSocket client connection error: {e}")
            return False

    async def _handle_client(self, websocket) -> None:
        """
        Handle a client connection (server mode).

        Args:
            websocket: Client WebSocket connection
        """
        peer_id = None
        peer_ws_url = None
        remote_addr = websocket.remote_address if hasattr(websocket, 'remote_address') else None

        try:
            # Send welcome message with our info (this is our "hello")
            welcome = Message(
                sender_id=self._local_id or "server",
                recipient_id="*",
                intent="welcome",
                content={
                    "node_id": self._local_id,
                    "ws_url": f"ws://{self.config.ws_host}:{self.config.ws_port}",
                    "message": "Connected to mesh network",
                    **self._local_info
                }
            )
            await websocket.send(json.dumps(welcome.to_dict()))
            logger.info(f"New peer connected from {remote_addr}, sent welcome")

            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)

                    # Handle mesh.peers request early (before Message parsing)
                    if data.get("intent") == "mesh.peers":
                        logger.info(f"Received mesh.peers request from {data.get('sender_id', 'unknown')}")

                        # Use peer lookup callback if available (gets real mesh peers)
                        if self._peer_lookup_callback:
                            peers_data = self._peer_lookup_callback()
                        else:
                            # Fallback to WebSocket-only connected peers
                            peers_data = [{"node_id": pid} for pid in self._connected_peers]

                        response = Message(
                            sender_id=self._local_id or "mesh",
                            recipient_id=data.get("sender_id", "*"),
                            intent="mesh.peers.response",
                            content={"peers": peers_data, "count": len(peers_data)},
                        )
                        await websocket.send(response.to_json())
                        logger.info(f"Sent mesh.peers.response with {len(peers_data)} peers")
                        continue

                    msg = Message.from_dict(data)

                    # Track peer from any message with sender_id
                    if msg.sender_id and msg.sender_id != "cli":
                        if msg.sender_id not in self._clients:
                            self._clients[msg.sender_id] = websocket
                            self._connected_peers.add(msg.sender_id)
                            logger.info(f"Peer registered: {msg.sender_id}")
                        peer_id = msg.sender_id

                    self._update_receive_stats(len(raw_message))

                    # Handle peer announcement - this is the "hello" from connecting peer
                    # We respond with "peer_announce_ack" (our "hi" back)
                    if msg.intent == "peer_announce" and msg.content:
                        peer_node_id = msg.content.get("node_id")
                        peer_ws_url = msg.content.get("ws_url")

                        # Register this peer
                        if peer_node_id:
                            self._clients[peer_node_id] = websocket
                            self._connected_peers.add(peer_node_id)
                            peer_id = peer_node_id
                            logger.info(f"Peer announced: {peer_node_id} at {peer_ws_url}")

                        # Send acknowledgment back with our info (the "hi")
                        ack = Message(
                            sender_id=self._local_id or "server",
                            recipient_id=peer_node_id or "*",
                            intent="peer_announce_ack",
                            content={
                                "node_id": self._local_id,
                                "ws_url": f"ws://{self.config.ws_host}:{self.config.ws_port}",
                                "message": "Peer acknowledged",
                                **self._local_info
                            }
                        )
                        await websocket.send(json.dumps(ack.to_dict()))
                        logger.info(f"Sent peer_announce_ack to {peer_node_id}")

                        # Dispatch the announce so mesh can register the peer
                        await self._dispatch_message(msg)
                        continue

                    # Handle heartbeat - peer is still alive, update tracking
                    if msg.intent == "heartbeat" and msg.content:
                        heartbeat_node_id = msg.content.get("node_id")
                        if heartbeat_node_id and heartbeat_node_id not in self._connected_peers:
                            self._connected_peers.add(heartbeat_node_id)
                            logger.info(f"Peer re-discovered via heartbeat: {heartbeat_node_id}")
                        continue

                    # Handle registration
                    if msg.intent == "register":
                        continue

                    # Handle mesh.ping directly - respond on same socket connection
                    # Respond immediately, do NOT dispatch to mesh (prevents echo loop)
                    if msg.intent == "mesh.ping":
                        ping_id = msg.content.get("ping_id") if msg.content else None
                        pong = Message(
                            sender_id=self._local_id or "server",
                            recipient_id=msg.sender_id,
                            intent="mesh.pong",
                            content={
                                "pong": True,
                                "from": self._local_id,
                                "time": __import__("time").time(),
                                "ping_id": ping_id
                            }
                        )
                        await websocket.send(json.dumps(pong.to_dict()))
                        logger.debug(f"Sent mesh.pong to {msg.sender_id} (ping_id={ping_id})")
                        # Do NOT dispatch - this prevents echo loops
                        continue

                    await self._dispatch_message(msg)

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
                    self._stats.errors += 1

        except ConnectionClosed:
            logger.info(f"Peer disconnected: {peer_id or remote_addr}")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            if peer_id:
                self._clients.pop(peer_id, None)
                self._connected_peers.discard(peer_id)
                logger.info(f"Peer removed from cache: {peer_id}")

    async def _client_receive_loop(self) -> None:
        """Receive loop for client mode."""
        try:
            async for raw_message in self._client_ws:
                try:
                    data = json.loads(raw_message)
                    msg = Message.from_dict(data)
                    self._update_receive_stats(len(raw_message))
                    await self._dispatch_message(msg)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
                    self._stats.errors += 1

        except ConnectionClosed:
            logger.info("WebSocket connection closed")
            self._state = TransportState.DISCONNECTED
            # Attempt reconnection
            asyncio.create_task(self.reconnect())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket transport."""
        async with self._lock:
            self._state = TransportState.DISCONNECTED

            # Cancel heartbeat task
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

            # Cancel receive task
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            # Cancel outbound receive tasks
            for url, task in list(self._outbound_tasks.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._outbound_tasks.clear()

            # Close outbound peer connections
            for url, ws in list(self._outbound_clients.items()):
                try:
                    await ws.close()
                except Exception:
                    pass
            self._outbound_clients.clear()

            # Close inbound client connections (server mode)
            for peer_id, ws in list(self._clients.items()):
                try:
                    await ws.close()
                except Exception:
                    pass
            self._clients.clear()
            self._connected_peers.clear()

            # Close client connection
            if self._client_ws:
                try:
                    await self._client_ws.close()
                except Exception:
                    pass
                self._client_ws = None

            # Close server
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                self._server = None

            logger.info("WebSocket transport disconnected")

    def set_local_id(self, entity_id: str) -> None:
        """
        Set the local entity ID.

        Args:
            entity_id: Local entity's ID
        """
        self._local_id = entity_id

    def set_local_info(self, info: Dict) -> None:
        """
        Set local node info for peer announcements.

        Args:
            info: Dict with node_id, entity_type, capabilities, etc.
        """
        self._local_info = info

    def set_peer_lookup(self, callback) -> None:
        """
        Set callback to lookup peers from mesh.

        Args:
            callback: Function that returns list of peer dicts with node_id, etc.
        """
        self._peer_lookup_callback = callback

    def add_known_peer(self, url: str) -> None:
        """
        Add a known peer URL to connect to.

        Args:
            url: WebSocket URL (ws://host:port)
        """
        self._known_peer_urls.add(url)

    async def connect_to_peer(self, url: str) -> bool:
        """
        Connect to a remote peer's WebSocket server.

        Args:
            url: WebSocket URL (ws://host:port)

        Returns:
            True if connection successful
        """
        if url in self._outbound_clients:
            return True

        try:
            logger.info(f"Connecting to peer at {url}...")
            ws = await asyncio.wait_for(
                ws_connect(url, ping_interval=20, ping_timeout=10),
                timeout=self.config.connect_timeout
            )

            # Wait for welcome message from peer
            peer_node_id = None
            try:
                welcome_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                welcome_data = json.loads(welcome_raw)
                peer_node_id = welcome_data.get("content", {}).get("node_id")
                if peer_node_id:
                    self._connected_peers.add(peer_node_id)
                    self._peer_id_to_url[peer_node_id] = url  # Map node_id to URL
                    logger.info(f"Peer identified from welcome: {peer_node_id}")
            except Exception as e:
                logger.warning(f"Failed to get welcome from {url}: {e}")

            # Send our announcement
            announce = Message(
                sender_id=self._local_id or "unknown",
                recipient_id="*",
                intent="peer_announce",
                content={
                    "node_id": self._local_id,
                    "ws_url": f"ws://{self.config.ws_host}:{self.config.ws_port}",
                    **self._local_info
                }
            )
            await ws.send(json.dumps(announce.to_dict()))

            self._outbound_clients[url] = ws
            self._known_peer_urls.add(url)

            # Start receive loop for this connection
            task = asyncio.create_task(self._outbound_receive_loop(url, ws))
            self._outbound_tasks[url] = task

            logger.info(f"Connected to peer: {url}")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to peer: {url}")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to peer {url}: {e}")
            return False

    async def _outbound_receive_loop(self, url: str, ws) -> None:
        """Receive loop for outbound peer connections."""
        peer_node_id = None
        try:
            async for raw_message in ws:
                try:
                    data = json.loads(raw_message)
                    msg = Message.from_dict(data)
                    self._update_receive_stats(len(raw_message))

                    # Track peer from sender_id (but not "cli" or "server")
                    if msg.sender_id and msg.sender_id not in ("cli", "server"):
                        if msg.sender_id not in self._connected_peers:
                            self._connected_peers.add(msg.sender_id)
                            peer_node_id = msg.sender_id
                            logger.info(f"Peer identified: {msg.sender_id}")

                    # Handle welcome message - server's initial "hello"
                    if msg.intent == "welcome" and msg.content:
                        node_id = msg.content.get("node_id")
                        if node_id and node_id not in self._connected_peers:
                            self._connected_peers.add(node_id)
                            peer_node_id = node_id
                            logger.info(f"Peer identified from welcome: {node_id}")

                    # Handle peer_announce_ack - server's "hi" response to our "hello"
                    if msg.intent == "peer_announce_ack" and msg.content:
                        node_id = msg.content.get("node_id")
                        if node_id and node_id not in self._connected_peers:
                            self._connected_peers.add(node_id)
                            peer_node_id = node_id
                            logger.info(f"Peer acknowledged us: {node_id}")

                    # Handle heartbeat - peer is still alive
                    if msg.intent == "heartbeat" and msg.content:
                        node_id = msg.content.get("node_id")
                        if node_id and node_id not in self._connected_peers:
                            self._connected_peers.add(node_id)
                            peer_node_id = node_id
                            logger.info(f"Peer re-discovered via heartbeat: {node_id}")

                    await self._dispatch_message(msg)

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from peer")
                except Exception as e:
                    logger.error(f"Peer message handling error: {e}")

        except ConnectionClosed:
            logger.info(f"Peer connection closed: {url}")
            # Remove peer from connected list on disconnect
            if peer_node_id:
                self._connected_peers.discard(peer_node_id)
                self._peer_id_to_url.pop(peer_node_id, None)  # Clean up mapping
                logger.info(f"Peer removed from cache: {peer_node_id}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Peer receive loop error: {e}")
        finally:
            self._outbound_clients.pop(url, None)
            self._outbound_tasks.pop(url, None)

    async def send(self, message: Message, target: Optional[str] = None) -> bool:
        """
        Send a message via WebSocket.

        Args:
            message: Message to send
            target: Target peer ID (server mode only)

        Returns:
            True if send successful
        """
        if self._state != TransportState.CONNECTED:
            return False

        start_time = time.time()
        data = json.dumps(message.to_dict())

        try:
            if self._server_mode:
                # Server mode - send to specific client
                if target and target in self._clients:
                    # Inbound client (they connected to us)
                    await self._clients[target].send(data)
                elif target and target in self._peer_id_to_url:
                    # Outbound client (we connected to them)
                    url = self._peer_id_to_url[target]
                    if url in self._outbound_clients:
                        await self._outbound_clients[url].send(data)
                    else:
                        logger.warning(f"Outbound connection lost for: {target}")
                        return False
                elif target:
                    logger.warning(f"Unknown client: {target}")
                    return False
                else:
                    # No target - broadcast
                    return await self.broadcast(message) > 0
            else:
                # Client mode - send to server
                if self._client_ws:
                    await self._client_ws.send(data)
                else:
                    return False

            self._update_send_stats(len(data))
            self._record_latency(time.time() - start_time)
            return True

        except ConnectionClosed:
            logger.error("Connection closed during send")
            self._state = TransportState.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            self._stats.errors += 1
            return False

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all connected peers (inbound and outbound).

        Args:
            message: Message to broadcast

        Returns:
            Number of peers message was sent to
        """
        if self._state != TransportState.CONNECTED:
            return 0

        data = json.dumps(message.to_dict())
        count = 0

        # Send to inbound clients (peers that connected to us)
        for peer_id, ws in list(self._clients.items()):
            try:
                await ws.send(data)
                count += 1
                self._update_send_stats(len(data))
            except Exception as e:
                logger.error(f"Broadcast to {peer_id} failed: {e}")
                self._stats.errors += 1

        # Send to outbound connections (peers we connected to)
        for url, ws in list(self._outbound_clients.items()):
            try:
                await ws.send(data)
                count += 1
                self._update_send_stats(len(data))
            except Exception as e:
                logger.error(f"Broadcast to {url} failed: {e}")
                self._stats.errors += 1

        return count

    async def broadcast_to_local_clients(self, message: Message) -> int:
        """
        Broadcast message only to local inbound clients (like CLI).

        This is used to forward pong responses to CLI clients
        that are waiting for ping results.

        IMPORTANT: Only sends to clients with 'cli-' prefix to avoid
        sending to remote peer connections which would create echo loops.

        Args:
            message: Message to broadcast

        Returns:
            Number of local clients message was sent to
        """
        if self._state != TransportState.CONNECTED:
            return 0

        data = json.dumps(message.to_dict())
        count = 0

        # Send ONLY to local CLI clients (peer_id starts with "cli-")
        # Do NOT send to remote peer connections (would create echo loops)
        for peer_id, ws in list(self._clients.items()):
            # Only send to CLI clients, not to remote peers
            if not peer_id.startswith("cli-"):
                continue
            try:
                await ws.send(data)
                count += 1
            except Exception as e:
                logger.debug(f"Broadcast to local client {peer_id} failed: {e}")

        return count

    def start_heartbeat(self, interval: int = 30) -> None:
        """
        Start periodic heartbeat to maintain peer awareness.

        This sends periodic announcements to all peers and attempts
        to reconnect to known peers that may have disconnected.

        Args:
            interval: Seconds between heartbeats (default 30)
        """
        self._heartbeat_interval = interval
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info(f"Heartbeat started (every {interval}s)")

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat loop."""
        while self._state == TransportState.CONNECTED:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                # Send heartbeat to all connected peers
                heartbeat = Message(
                    sender_id=self._local_id or "unknown",
                    recipient_id="*",
                    intent="heartbeat",
                    content={
                        "node_id": self._local_id,
                        "ws_url": f"ws://{self.config.ws_host}:{self.config.ws_port}",
                        "timestamp": time.time(),
                        **self._local_info
                    }
                )

                sent_count = await self.broadcast(heartbeat)
                if sent_count > 0:
                    logger.debug(f"Heartbeat sent to {sent_count} peers")

                # Try to reconnect to known peers that we've lost connection to
                for url in list(self._known_peer_urls):
                    if url not in self._outbound_clients:
                        logger.info(f"Attempting to reconnect to {url}")
                        asyncio.create_task(self.connect_to_peer(url))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

        logger.info("Heartbeat stopped")
