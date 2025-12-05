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
        self._receive_task: Optional[asyncio.Task] = None
        self._outbound_tasks: Dict[str, asyncio.Task] = {}
        self._local_id: Optional[str] = None
        self._local_info: Dict = {}
        self._known_peer_urls: Set[str] = set()

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
                    source=self._local_id,
                    target="server",
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
            # Send welcome message with our info
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
                    msg = Message.from_dict(data)

                    # Track peer
                    if msg.sender_id:
                        if msg.sender_id not in self._clients:
                            self._clients[msg.sender_id] = websocket
                            self._connected_peers.add(msg.sender_id)
                            logger.info(f"Peer registered: {msg.sender_id}")
                        peer_id = msg.sender_id

                    self._update_receive_stats(len(raw_message))

                    # Handle peer announcement - connect back to them
                    if msg.intent == "peer_announce" and msg.content:
                        peer_ws_url = msg.content.get("ws_url")
                        if peer_ws_url and peer_ws_url not in self._outbound_clients:
                            # Connect back to the peer for bidirectional communication
                            logger.info(f"Peer announced at {peer_ws_url}, connecting back...")
                            asyncio.create_task(self.connect_to_peer(peer_ws_url))
                        # Dispatch the announce so mesh can register the peer
                        await self._dispatch_message(msg)
                        continue

                    # Handle registration
                    if msg.intent == "register":
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
        try:
            async for raw_message in ws:
                try:
                    data = json.loads(raw_message)
                    msg = Message.from_dict(data)
                    self._update_receive_stats(len(raw_message))

                    # Track peer
                    if msg.sender_id and msg.sender_id not in self._connected_peers:
                        self._connected_peers.add(msg.sender_id)
                        logger.info(f"Peer identified: {msg.sender_id}")

                    await self._dispatch_message(msg)

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from peer")
                except Exception as e:
                    logger.error(f"Peer message handling error: {e}")

        except ConnectionClosed:
            logger.info(f"Peer connection closed: {url}")
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
                    await self._clients[target].send(data)
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
