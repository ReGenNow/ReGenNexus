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
        self._receive_task: Optional[asyncio.Task] = None
        self._local_id: Optional[str] = None

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
        try:
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                    msg = Message.from_dict(data)

                    # Track peer
                    if msg.source:
                        if msg.source not in self._clients:
                            self._clients[msg.source] = websocket
                            self._connected_peers.add(msg.source)
                            logger.debug(f"WebSocket client connected: {msg.source}")
                        peer_id = msg.source

                    self._update_receive_stats(len(raw_message))

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
            logger.debug(f"WebSocket client disconnected: {peer_id}")
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

            # Close client connections (server mode)
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
        Broadcast message to all connected clients.

        Args:
            message: Message to broadcast

        Returns:
            Number of clients message was sent to
        """
        if self._state != TransportState.CONNECTED:
            return 0

        if not self._server_mode:
            # Client can't broadcast
            return 0

        data = json.dumps(message.to_dict())
        count = 0

        for peer_id, ws in list(self._clients.items()):
            try:
                await ws.send(data)
                count += 1
                self._update_send_stats(len(data))
            except Exception as e:
                logger.error(f"Broadcast to {peer_id} failed: {e}")
                self._stats.errors += 1

        return count
