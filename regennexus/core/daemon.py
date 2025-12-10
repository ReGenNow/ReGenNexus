"""
RegenNexus UAP - Mesh Daemon

Background service for always-on mesh networking.
Runs as a daemon and automatically discovers and maintains
connections with all RegenNexus peers on the network.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from regennexus.core.mesh import MeshNetwork, MeshConfig, MeshNode, NodeState
from regennexus.core.message import Message

logger = logging.getLogger(__name__)

# Default paths for different platforms
if sys.platform == "darwin":
    DEFAULT_PID_FILE = Path.home() / "Library/Application Support/RegenNexus/mesh.pid"
    DEFAULT_LOG_FILE = Path.home() / "Library/Logs/RegenNexus/mesh.log"
    DEFAULT_CONFIG_FILE = Path.home() / "Library/Application Support/RegenNexus/mesh.yaml"
elif sys.platform == "win32":
    DEFAULT_PID_FILE = Path(os.environ.get("APPDATA", "")) / "RegenNexus/mesh.pid"
    DEFAULT_LOG_FILE = Path(os.environ.get("APPDATA", "")) / "RegenNexus/mesh.log"
    DEFAULT_CONFIG_FILE = Path(os.environ.get("APPDATA", "")) / "RegenNexus/mesh.yaml"
else:  # Linux and others
    DEFAULT_PID_FILE = Path.home() / ".local/share/regennexus/mesh.pid"
    DEFAULT_LOG_FILE = Path.home() / ".local/share/regennexus/mesh.log"
    DEFAULT_CONFIG_FILE = Path.home() / ".config/regennexus/mesh.yaml"


class MeshDaemon:
    """
    Always-on mesh network daemon.

    Provides:
    - Background mesh service that runs forever
    - Auto-restart on crash
    - Graceful shutdown on SIGTERM/SIGINT
    - Status reporting via IPC
    - Idle mode when no activity (low CPU usage)

    Example:
        daemon = MeshDaemon(node_id="my-device")
        daemon.run()  # Blocks forever

    Or programmatically:
        daemon = MeshDaemon(node_id="my-device")
        await daemon.start()
        # ... do other things ...
        await daemon.stop()
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        entity_type: str = "device",
        capabilities: Optional[List[str]] = None,
        known_peers: Optional[List[str]] = None,
        config_file: Optional[Path] = None,
        pid_file: Optional[Path] = None,
        log_file: Optional[Path] = None,
        udp_port: Optional[int] = None,
        websocket_port: Optional[int] = None,
    ):
        """
        Initialize mesh daemon.

        Args:
            node_id: Unique node identifier (auto-generated if None)
            entity_type: Type of this node (device, agent, service, etc.)
            capabilities: List of capabilities this node provides
            known_peers: List of peer URLs to connect to (ws://host:port)
            config_file: Path to configuration file
            pid_file: Path to PID file
            log_file: Path to log file
            udp_port: UDP port for discovery (auto-detect if None)
            websocket_port: WebSocket port (auto-detect if None)
        """
        self.node_id = node_id
        self.entity_type = entity_type
        self.capabilities = capabilities or []
        self.known_peers = known_peers or []
        self.udp_port = udp_port
        self.websocket_port = websocket_port

        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.pid_file = pid_file or DEFAULT_PID_FILE
        self.log_file = log_file or DEFAULT_LOG_FILE

        self._mesh: Optional[MeshNetwork] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Message handlers registered by external code
        self._message_handlers: List[Callable] = []
        self._peer_handlers: List[Callable] = []

        # Stats
        self._start_time: Optional[float] = None
        self._restart_count = 0

    def _setup_logging(self) -> None:
        """Setup logging to file."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        # Add to root logger
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        # Also log to console if interactive
        if sys.stdout.isatty():
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            root.addHandler(console)

    def _write_pid(self) -> None:
        """Write PID file."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass

    def _load_config(self) -> MeshConfig:
        """Load configuration from file or use defaults."""
        config = MeshConfig(
            node_id=self.node_id,
            entity_type=self.entity_type,
            capabilities=self.capabilities,
        )

        # Apply command-line port overrides first
        if self.udp_port is not None:
            config.udp_port = self.udp_port
        if self.websocket_port is not None:
            config.websocket_port = self.websocket_port

        if self.config_file.exists():
            try:
                import yaml
                with open(self.config_file) as f:
                    data = yaml.safe_load(f)

                if data:
                    config.node_id = data.get("node_id", config.node_id)
                    config.entity_type = data.get("entity_type", config.entity_type)
                    config.capabilities = data.get("capabilities", config.capabilities)
                    config.discovery_interval = data.get("discovery_interval", config.discovery_interval)
                    # Only use file config if not overridden by command line
                    if self.udp_port is None:
                        config.udp_port = data.get("udp_port", config.udp_port)
                    if self.websocket_port is None:
                        config.websocket_port = data.get("websocket_port", config.websocket_port)

                logger.info(f"Loaded config from {self.config_file}")

            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        return config

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: self._signal_handler())

    def _signal_handler(self) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal")
        self._shutdown_event.set()

    async def start(self) -> bool:
        """
        Start the mesh daemon.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        self._setup_logging()
        logger.info("Starting RegenNexus mesh daemon...")

        self._write_pid()
        self._start_time = time.time()

        # Load config
        config = self._load_config()

        # Create mesh network
        self._mesh = MeshNetwork(config)

        # Register handlers
        self._mesh.on_message(self._handle_message)
        self._mesh.on_peer(self._handle_peer_event)

        # Start mesh
        if not await self._mesh.start():
            logger.error("Failed to start mesh network")
            self._remove_pid()
            return False

        self._running = True
        logger.info(f"Mesh daemon running as {self._mesh.node_id}")

        # Connect to known peers
        if self.known_peers:
            logger.info(f"Connecting to {len(self.known_peers)} known peer(s)...")
            for peer_url in self.known_peers:
                try:
                    await self._mesh._transport.connect_to_peer(peer_url)
                except Exception as e:
                    logger.warning(f"Failed to connect to peer {peer_url}: {e}")

        return True

    async def stop(self) -> None:
        """Stop the mesh daemon."""
        if not self._running:
            return

        logger.info("Stopping mesh daemon...")
        self._running = False

        if self._mesh:
            await self._mesh.stop()
            self._mesh = None

        self._remove_pid()
        logger.info("Mesh daemon stopped")

    async def run_forever(self) -> None:
        """
        Run the daemon forever with auto-restart.

        This is the main entry point for running as a service.
        """
        self._setup_signals()

        while not self._shutdown_event.is_set():
            try:
                if not await self.start():
                    logger.error("Failed to start, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    self._restart_count += 1
                    continue

                # Run until shutdown
                await self._shutdown_event.wait()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daemon error: {e}")
                self._restart_count += 1
                await asyncio.sleep(5)

        await self.stop()

    def run(self) -> None:
        """
        Run the daemon (blocking).

        This is the entry point for CLI usage.
        """
        try:
            asyncio.run(self.run_forever())
        except KeyboardInterrupt:
            pass

    async def _handle_message(self, message: Message) -> None:
        """Handle incoming messages."""
        logger.info(f"Message from {message.sender_id}: intent={message.intent}")

        # Built-in intent handlers
        if message.intent == "execute" and message.content:
            await self._handle_execute(message)
            return
        elif message.intent == "system_info":
            await self._handle_system_info(message)
            return
        elif message.intent == "ping":
            await self._handle_ping(message)
            return
        # CLI intent handlers (thin client commands via daemon)
        elif message.intent == "cli.ping":
            await self._handle_cli_ping(message)
            return
        elif message.intent == "cli.benchmark":
            await self._handle_cli_benchmark(message)
            return
        elif message.intent == "cli.peers":
            await self._handle_cli_peers(message)
            return
        elif message.intent == "cli.status":
            await self._handle_cli_status(message)
            return
        elif message.intent == "cli.send":
            await self._handle_cli_send(message)
            return

        # Dispatch to registered handlers
        for handler in self._message_handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Message handler error: {e}")

    async def _handle_execute(self, message: Message) -> None:
        """Execute a command and send result back."""
        import subprocess

        cmd = message.content.get("command", "")
        if not cmd:
            return

        logger.info(f"Executing command from {message.sender_id}: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            response = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": cmd
            }
        except subprocess.TimeoutExpired:
            response = {"error": "Command timed out", "command": cmd}
        except Exception as e:
            response = {"error": str(e), "command": cmd}

        # Send response back
        if self._mesh and message.sender_id:
            await self._mesh.send(message.sender_id, response, intent="execute_result")
            logger.info(f"Sent result to {message.sender_id}")

    async def _handle_system_info(self, message: Message) -> None:
        """Send system info back to requester."""
        import platform
        import os

        info = {
            "hostname": platform.node(),
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "pid": os.getpid(),
            "node_id": self._mesh.node_id if self._mesh else None,
        }

        # Try to get memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
            info["memory_used_percent"] = mem.percent
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except ImportError:
            pass

        # Try to get GPU info (NVIDIA via nvidia-smi)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpus.append({
                            "name": parts[0],
                            "vram_total_mb": int(parts[1]),
                            "vram_used_mb": int(parts[2]),
                            "vram_free_mb": int(parts[3]),
                            "gpu_utilization_percent": int(parts[4]),
                            "temperature_c": int(parts[5])
                        })
                if gpus:
                    info["gpus"] = gpus
        except Exception:
            pass

        if self._mesh and message.sender_id:
            await self._mesh.send(message.sender_id, info, intent="system_info_result")

    async def _handle_ping(self, message: Message) -> None:
        """Respond to ping with pong."""
        if self._mesh and message.sender_id:
            # Include ping_id if present for round-trip correlation
            ping_id = message.content.get("ping_id") if message.content else None
            response = {
                "pong": True,
                "from": self._mesh.node_id,
                "time": time.time()
            }
            if ping_id:
                response["ping_id"] = ping_id
            await self._mesh.send(
                message.sender_id,
                response,
                intent="pong"
            )

    # =========================================================================
    # CLI Intent Handlers (thin client commands via daemon)
    # =========================================================================

    async def _handle_cli_ping(self, message: Message) -> None:
        """
        Handle CLI ping request - ping a peer using daemon's persistent connection.

        Request content: {target: str, count: int, timeout: float}
        Response: {results: [...], success: bool}
        """
        if not self._mesh or not message.sender_id:
            return

        content = message.content or {}
        target = content.get("target")
        count = content.get("count", 1)
        timeout = content.get("timeout", 5.0)

        if not target:
            await self._mesh.send(
                message.sender_id,
                {"error": "No target specified", "success": False},
                intent="cli.ping.result"
            )
            return

        # Find target peer by ID or partial match
        target_id = None
        peers = self._mesh.get_peers()
        for peer in peers:
            if peer.node_id == target or target.lower() in peer.node_id.lower():
                target_id = peer.node_id
                break

        if not target_id:
            await self._mesh.send(
                message.sender_id,
                {"error": f"Peer '{target}' not found", "success": False, "available_peers": [p.node_id for p in peers]},
                intent="cli.ping.result"
            )
            return

        # Perform pings using PeerManager's persistent connection
        results = []
        for i in range(count):
            rtt = await self._mesh.ping(target_id, timeout=timeout)
            results.append({
                "seq": i + 1,
                "target": target_id,
                "rtt_ms": rtt,
                "success": rtt is not None
            })
            if i < count - 1:
                await asyncio.sleep(0.5)  # Small delay between pings

        # Calculate summary
        successful = [r for r in results if r["success"]]
        summary = {
            "target": target_id,
            "sent": count,
            "received": len(successful),
            "loss_percent": ((count - len(successful)) / count) * 100 if count > 0 else 100,
            "avg_rtt_ms": sum(r["rtt_ms"] for r in successful) / len(successful) if successful else None,
            "min_rtt_ms": min(r["rtt_ms"] for r in successful) if successful else None,
            "max_rtt_ms": max(r["rtt_ms"] for r in successful) if successful else None,
        }

        await self._mesh.send(
            message.sender_id,
            {"results": results, "summary": summary, "success": len(successful) > 0},
            intent="cli.ping.result"
        )

    async def _handle_cli_benchmark(self, message: Message) -> None:
        """
        Handle CLI benchmark request - benchmark all peers using daemon's connections.

        Request content: {count: int, timeout: float}
        Response: {results: {...}, success: bool}
        """
        if not self._mesh or not message.sender_id:
            return

        content = message.content or {}
        count = content.get("count", 5)
        timeout = content.get("timeout", 5.0)

        peers = self._mesh.get_peers()
        if not peers:
            await self._mesh.send(
                message.sender_id,
                {"error": "No peers available", "success": False},
                intent="cli.benchmark.result"
            )
            return

        # Benchmark each peer
        results = {}
        for peer in peers:
            peer_results = []
            for i in range(count):
                rtt = await self._mesh.ping(peer.node_id, timeout=timeout)
                peer_results.append(rtt)
                if i < count - 1:
                    await asyncio.sleep(0.2)

            successful = [r for r in peer_results if r is not None]
            results[peer.node_id] = {
                "entity_type": peer.entity_type,
                "connection_state": getattr(peer, 'connection_state', 'unknown'),
                "sent": count,
                "received": len(successful),
                "loss_percent": ((count - len(successful)) / count) * 100,
                "avg_rtt_ms": round(sum(successful) / len(successful), 2) if successful else None,
                "min_rtt_ms": round(min(successful), 2) if successful else None,
                "max_rtt_ms": round(max(successful), 2) if successful else None,
                "rtts": [round(r, 2) if r else None for r in peer_results]
            }

        await self._mesh.send(
            message.sender_id,
            {"results": results, "peer_count": len(peers), "success": True},
            intent="cli.benchmark.result"
        )

    async def _handle_cli_peers(self, message: Message) -> None:
        """
        Handle CLI peers request - return detailed peer information.

        Response includes connection state, RTT, and capabilities.
        """
        if not self._mesh or not message.sender_id:
            return

        peers = self._mesh.get_peers()
        peer_list = []

        for peer in peers:
            peer_info = {
                "node_id": peer.node_id,
                "entity_type": peer.entity_type,
                "capabilities": peer.capabilities,
                "address": peer.address,
                "port": peer.port,
                "ws_url": getattr(peer, 'ws_url', None),
                "transport": peer.transport.value if hasattr(peer.transport, 'value') else str(peer.transport),
                "connection_state": getattr(peer, 'connection_state', 'unknown'),
                "last_rtt_ms": getattr(peer, 'last_rtt', None),
                "last_seen": peer.last_seen,
                "is_connected": getattr(peer, 'is_connected', False),
            }
            peer_list.append(peer_info)

        await self._mesh.send(
            message.sender_id,
            {"peers": peer_list, "count": len(peer_list), "local_id": self._mesh.node_id},
            intent="cli.peers.result"
        )

    async def _handle_cli_status(self, message: Message) -> None:
        """
        Handle CLI status request - return full daemon status.
        """
        if not self._mesh or not message.sender_id:
            return

        status = self.get_status()

        # Add PeerManager stats if available
        if hasattr(self._mesh, '_peer_manager') and self._mesh._peer_manager:
            pm = self._mesh._peer_manager
            status["peer_manager"] = {
                "active_connections": len([p for p in pm._peers.values() if p.state.value == "connected"]),
                "total_peers": len(pm._peers),
                "keepalive_interval": pm._keepalive_interval,
            }

        await self._mesh.send(
            message.sender_id,
            status,
            intent="cli.status.result"
        )

    async def _handle_cli_send(self, message: Message) -> None:
        """
        Handle CLI send request - send a message to a peer via daemon.

        Request content: {target: str, payload: any, intent: str}
        """
        if not self._mesh or not message.sender_id:
            return

        content = message.content or {}
        target = content.get("target")
        payload = content.get("payload", {})
        intent = content.get("intent", "message")

        if not target:
            await self._mesh.send(
                message.sender_id,
                {"error": "No target specified", "success": False},
                intent="cli.send.result"
            )
            return

        # Find target peer
        target_id = None
        peers = self._mesh.get_peers()
        for peer in peers:
            if peer.node_id == target or target.lower() in peer.node_id.lower():
                target_id = peer.node_id
                break

        if not target_id:
            await self._mesh.send(
                message.sender_id,
                {"error": f"Peer '{target}' not found", "success": False},
                intent="cli.send.result"
            )
            return

        # Send via daemon's persistent connection
        success = await self._mesh.send(target_id, payload, intent=intent)

        await self._mesh.send(
            message.sender_id,
            {"success": success, "target": target_id, "intent": intent},
            intent="cli.send.result"
        )

    async def _handle_peer_event(self, peer: MeshNode, event: str) -> None:
        """Handle peer events."""
        logger.info(f"Peer {event}: {peer.node_id} ({peer.entity_type})")

        # Dispatch to registered handlers
        for handler in self._peer_handlers:
            try:
                result = handler(peer, event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Peer handler error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def on_message(self, handler: Callable) -> None:
        """Register message handler."""
        self._message_handlers.append(handler)

    def on_peer(self, handler: Callable) -> None:
        """Register peer event handler."""
        self._peer_handlers.append(handler)

    async def send(self, target: str, content: Any, intent: str = "message") -> bool:
        """Send message to a peer."""
        if not self._mesh:
            return False
        return await self._mesh.send(target, content, intent)

    async def broadcast(self, content: Any, intent: str = "broadcast") -> int:
        """Broadcast message to all peers."""
        if not self._mesh:
            return 0
        return await self._mesh.broadcast(content, intent)

    def get_peers(self) -> List[MeshNode]:
        """Get list of connected peers."""
        if not self._mesh:
            return []
        return self._mesh.get_peers()

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        status = {
            "running": self._running,
            "node_id": self._mesh.node_id if self._mesh else None,
            "state": self._mesh.state.value if self._mesh else "offline",
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "restart_count": self._restart_count,
            "peer_count": len(self.get_peers()),
            "peers": [
                {
                    "id": p.node_id,
                    "type": p.entity_type,
                    "capabilities": p.capabilities,
                }
                for p in self.get_peers()
            ],
        }

        if self._mesh:
            status["transport_stats"] = self._mesh.get_stats().get("transport_stats", {})

        return status

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    @property
    def mesh(self) -> Optional[MeshNetwork]:
        """Get underlying mesh network."""
        return self._mesh


# =============================================================================
# Singleton for auto-start on import
# =============================================================================

_daemon_instance: Optional[MeshDaemon] = None
_daemon_task: Optional[asyncio.Task] = None


def get_daemon() -> MeshDaemon:
    """
    Get or create the global mesh daemon instance.

    This allows code to access the mesh without explicitly starting it:

        from regennexus.core.daemon import get_daemon

        daemon = get_daemon()
        await daemon.send("other-node", {"hello": "world"})
    """
    global _daemon_instance

    if _daemon_instance is None:
        _daemon_instance = MeshDaemon()

    return _daemon_instance


async def ensure_running() -> MeshDaemon:
    """
    Ensure the mesh daemon is running.

    This can be called from anywhere to make sure the mesh is up:

        from regennexus.core.daemon import ensure_running

        daemon = await ensure_running()
        await daemon.send("target", {"data": 123})
    """
    global _daemon_instance, _daemon_task

    daemon = get_daemon()

    if not daemon.is_running:
        await daemon.start()

    return daemon


def is_daemon_running() -> bool:
    """Check if a daemon is already running (via PID file)."""
    pid_file = DEFAULT_PID_FILE

    if not pid_file.exists():
        return False

    try:
        pid = int(pid_file.read_text().strip())

        # Check if process is running
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)  # Doesn't kill, just checks
            return True

    except (ValueError, ProcessLookupError, PermissionError):
        # PID file exists but process is not running
        pid_file.unlink(missing_ok=True)
        return False


def get_daemon_pid() -> Optional[int]:
    """Get PID of running daemon, if any."""
    if not is_daemon_running():
        return None

    try:
        return int(DEFAULT_PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        return None


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for running as daemon."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="regennexus-daemon",
        description="RegenNexus Mesh Network Daemon"
    )

    parser.add_argument(
        "--node-id", "-n",
        help="Unique node identifier"
    )
    parser.add_argument(
        "--type", "-t",
        default="device",
        help="Entity type (device, agent, service)"
    )
    parser.add_argument(
        "--capabilities", "-c",
        nargs="*",
        default=[],
        help="Node capabilities"
    )
    parser.add_argument(
        "--config", "-f",
        type=Path,
        help="Config file path"
    )
    parser.add_argument(
        "--foreground", "-F",
        action="store_true",
        help="Run in foreground (don't daemonize)"
    )

    args = parser.parse_args()

    daemon = MeshDaemon(
        node_id=args.node_id,
        entity_type=args.type,
        capabilities=args.capabilities,
        config_file=args.config,
    )

    print(f"Starting RegenNexus mesh daemon...")
    daemon.run()


if __name__ == "__main__":
    main()
