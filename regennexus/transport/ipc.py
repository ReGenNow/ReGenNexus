"""
RegenNexus UAP - IPC Transport

Local inter-process communication for fastest messaging (<0.1ms).
Supports Unix sockets, named pipes, and shared memory.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import os
import sys
import struct
import time
import logging
from typing import Dict, Optional, Set

from regennexus.core.message import Message
from regennexus.transport.base import (
    Transport,
    TransportConfig,
    TransportState,
    TransportError,
)

logger = logging.getLogger(__name__)


class IPCTransport(Transport):
    """
    IPC (Inter-Process Communication) Transport.

    Provides the fastest communication method for processes on the
    same machine. Latency is typically <0.1ms.

    Supports:
    - Unix domain sockets (Linux/macOS)
    - Named pipes (Windows)
    - Shared memory (experimental)
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        super().__init__(config)
        self._server = None
        self._clients: Dict[str, asyncio.StreamWriter] = {}
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._is_server = False
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def socket_path(self) -> str:
        """Get the socket path, adjusted for Windows if needed."""
        if sys.platform == "win32":
            # Windows uses named pipes
            path = self.config.ipc_socket_path
            if not path.startswith("\\\\.\\pipe\\"):
                # Convert Unix-style path to Windows named pipe
                name = os.path.basename(path).replace(".sock", "")
                return f"\\\\.\\pipe\\regennexus_{name}"
            return path
        return self.config.ipc_socket_path

    async def connect(self) -> bool:
        """
        Connect to IPC endpoint.

        First tries to connect as client. If no server exists,
        starts as server.

        Returns:
            True if connection successful
        """
        async with self._lock:
            if self._state == TransportState.CONNECTED:
                return True

            self._state = TransportState.CONNECTING

            try:
                # Try to connect as client first
                if await self._connect_as_client():
                    self._state = TransportState.CONNECTED
                    self._stats.connect_time = time.time()
                    logger.info(f"Connected as IPC client to {self.socket_path}")
                    return True

                # If client connection fails, start as server
                if await self._start_server():
                    self._state = TransportState.CONNECTED
                    self._stats.connect_time = time.time()
                    self._is_server = True
                    logger.info(f"Started IPC server on {self.socket_path}")
                    return True

                self._state = TransportState.ERROR
                return False

            except Exception as e:
                logger.error(f"IPC connection error: {e}")
                self._state = TransportState.ERROR
                self._stats.errors += 1
                return False

    async def _connect_as_client(self) -> bool:
        """Try to connect as a client."""
        try:
            if sys.platform == "win32":
                # Windows named pipe client
                return await self._connect_windows_pipe()
            else:
                # Unix socket client
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(self.socket_path),
                    timeout=self.config.connect_timeout
                )

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            return True

        except (FileNotFoundError, ConnectionRefusedError):
            # Server not running
            return False
        except asyncio.TimeoutError:
            logger.debug("IPC client connection timeout")
            return False
        except Exception as e:
            logger.debug(f"IPC client connection failed: {e}")
            return False

    async def _connect_windows_pipe(self) -> bool:
        """Connect to Windows named pipe."""
        try:
            # Use asyncio's Windows-specific pipe support
            loop = asyncio.get_event_loop()
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.socket_path),
                timeout=self.config.connect_timeout
            )
            return True
        except Exception as e:
            logger.debug(f"Windows pipe connection failed: {e}")
            return False

    async def _start_server(self) -> bool:
        """Start as IPC server."""
        try:
            # Remove existing socket file
            if sys.platform != "win32" and os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            if sys.platform == "win32":
                # Windows named pipe server
                return await self._start_windows_server()
            else:
                # Unix socket server
                self._server = await asyncio.start_unix_server(
                    self._handle_client,
                    path=self.socket_path
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start IPC server: {e}")
            return False

    async def _start_windows_server(self) -> bool:
        """Start Windows named pipe server."""
        try:
            # For Windows, we use standard TCP on localhost as fallback
            # True named pipe support would require additional libraries
            self._server = await asyncio.start_server(
                self._handle_client,
                host="127.0.0.1",
                port=0  # Let OS assign port
            )
            # Store the assigned port for clients to find
            port = self._server.sockets[0].getsockname()[1]
            logger.info(f"Windows IPC server on localhost:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Windows server: {e}")
            return False

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new client connection."""
        peer_id = None
        try:
            # Receive peer ID as first message
            data = await self._read_message(reader)
            if data:
                msg = Message.from_dict(json.loads(data))
                peer_id = msg.sender_id
                self._clients[peer_id] = writer
                self._connected_peers.add(peer_id)
                logger.debug(f"IPC client connected: {peer_id}")

            # Handle messages from this client
            while True:
                data = await self._read_message(reader)
                if not data:
                    break

                msg = Message.from_dict(json.loads(data))
                await self._dispatch_message(msg)
                self._update_receive_stats(len(data))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            self._stats.errors += 1
        finally:
            if peer_id:
                self._clients.pop(peer_id, None)
                self._connected_peers.discard(peer_id)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _receive_loop(self) -> None:
        """Receive loop for client mode."""
        try:
            while self._reader and not self._reader.at_eof():
                data = await self._read_message(self._reader)
                if not data:
                    break

                msg = Message.from_dict(json.loads(data))
                await self._dispatch_message(msg)
                self._update_receive_stats(len(data))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            self._stats.errors += 1

    async def _read_message(self, reader: asyncio.StreamReader) -> Optional[str]:
        """
        Read a length-prefixed message.

        Format: 4-byte length (big-endian) + JSON data

        Args:
            reader: Stream reader

        Returns:
            Message data as string, or None on EOF
        """
        try:
            # Read 4-byte length header
            header = await reader.readexactly(4)
            length = struct.unpack(">I", header)[0]

            if length > self.config.buffer_size:
                logger.error(f"Message too large: {length}")
                return None

            # Read message body
            data = await reader.readexactly(length)
            return data.decode("utf-8")

        except asyncio.IncompleteReadError:
            return None
        except Exception as e:
            logger.debug(f"Read error: {e}")
            return None

    async def _write_message(
        self,
        writer: asyncio.StreamWriter,
        data: str
    ) -> bool:
        """
        Write a length-prefixed message.

        Args:
            writer: Stream writer
            data: Message data as string

        Returns:
            True if write successful
        """
        try:
            encoded = data.encode("utf-8")
            header = struct.pack(">I", len(encoded))
            writer.write(header + encoded)
            await writer.drain()
            return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect IPC transport."""
        async with self._lock:
            self._state = TransportState.DISCONNECTED

            # Cancel receive task
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            # Close client connections
            for peer_id, writer in list(self._clients.items()):
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            self._clients.clear()
            self._connected_peers.clear()

            # Close own connection (client mode)
            if self._writer:
                try:
                    self._writer.close()
                    await self._writer.wait_closed()
                except Exception:
                    pass
                self._writer = None
                self._reader = None

            # Close server
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                self._server = None

                # Remove socket file
                if sys.platform != "win32" and os.path.exists(self.socket_path):
                    try:
                        os.unlink(self.socket_path)
                    except Exception:
                        pass

            self._is_server = False
            logger.info("IPC transport disconnected")

    async def send(self, message: Message, target: Optional[str] = None) -> bool:
        """
        Send a message via IPC.

        Args:
            message: Message to send
            target: Target peer ID (None for server in client mode)

        Returns:
            True if send successful
        """
        if self._state != TransportState.CONNECTED:
            return False

        start_time = time.time()
        data = json.dumps(message.to_dict())

        try:
            if self._is_server:
                # Server mode - send to specific client
                if target and target in self._clients:
                    success = await self._write_message(
                        self._clients[target], data
                    )
                else:
                    logger.warning(f"Target not found: {target}")
                    return False
            else:
                # Client mode - send to server
                if self._writer:
                    success = await self._write_message(self._writer, data)
                else:
                    return False

            if success:
                self._update_send_stats(len(data))
                self._record_latency(time.time() - start_time)

            return success

        except Exception as e:
            logger.error(f"Send error: {e}")
            self._stats.errors += 1
            return False

    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all connected peers.

        Args:
            message: Message to broadcast

        Returns:
            Number of peers message was sent to
        """
        if self._state != TransportState.CONNECTED:
            return 0

        if not self._is_server:
            # Client can't broadcast, send to server instead
            success = await self.send(message)
            return 1 if success else 0

        # Server broadcasts to all clients
        data = json.dumps(message.to_dict())
        count = 0

        for peer_id, writer in list(self._clients.items()):
            try:
                if await self._write_message(writer, data):
                    count += 1
                    self._update_send_stats(len(data))
            except Exception as e:
                logger.error(f"Broadcast to {peer_id} failed: {e}")
                self._stats.errors += 1

        return count
