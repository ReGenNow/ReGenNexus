"""
RegenNexus UAP - Command Line Interface

CLI for RegenNexus UAP.

Usage:
    regen start [--config CONFIG]
    regen stop
    regen status
    regen devices list
    regen devices info <device_id>
    regen send <target> <message>
    regen --version

Copyright (c) 2024 ReGen Designs LLC
"""

import asyncio
import json
import sys
from typing import Optional

# Try to import click, fall back to argparse
try:
    import click

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False
    import argparse

from regennexus.__version__ import __version__, __title__


def print_banner():
    """Print RegenNexus banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   RegenNexus UAP - Universal Adapter Protocol            ║
║   Version {__version__:<47} ║
║   Copyright (c) 2024 ReGen Designs LLC                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


HELP_TEXT = """
RegenNexus UAP - Universal Adapter Protocol
============================================

A plug-and-play mesh networking system for connecting devices, robots,
AI agents, and applications across LAN and internet.

QUICK START
-----------
  regen mesh start                  Start mesh daemon (background)
  regen mesh start -f               Start in foreground (see logs)
  regen mesh status                 Check if mesh is running
  regen mesh stop                   Stop the mesh daemon

MESH NETWORKING
---------------
The mesh network automatically discovers devices on your LAN using:
  - mDNS/Zeroconf (works on all routers, plug-and-play)
  - UDP multicast (fast, low latency)
  - UDP broadcast (fallback when multicast blocked)
  - Direct peer IPs (cross-subnet communication)

Examples:
  # Start mesh on Mac
  regen mesh start -n mac-studio -t controller -c audio -c display

  # Start mesh on Windows
  regen mesh start -n win11-pc -t workstation -c gpu

  # Start with debug logging
  regen mesh start -f -d

PYTHON API
----------
  from regennexus.core.mesh import MeshNetwork, MeshConfig

  mesh = MeshNetwork(MeshConfig(node_id="my-device"))
  await mesh.start()

  # Auto-discovered peers
  for peer in mesh.get_peers():
      print(f"Found: {peer.node_id}")

  # Send message
  await mesh.send("other-device", {"hello": "world"})

CONFIGURATION
-------------
Config files are stored at:
  macOS:   ~/Library/Application Support/RegenNexus/mesh.yaml
  Windows: %APPDATA%/RegenNexus/mesh.yaml
  Linux:   ~/.config/regennexus/mesh.yaml

PID files (for daemon management):
  macOS:   ~/Library/Application Support/RegenNexus/mesh.pid
  Windows: %APPDATA%/RegenNexus/mesh.pid
  Linux:   /var/run/regennexus/mesh.pid

TROUBLESHOOTING
---------------
  regen doctor                      Check dependencies
  regen mesh start -f -d            Run with debug output

  If devices don't discover each other:
  1. Ensure both are on the same subnet (192.168.x.x)
  2. Check firewall allows UDP port 5353
  3. On Windows, run: netsh advfirewall firewall add rule \\
       name="RegenNexus" dir=in action=allow protocol=UDP localport=5353

MORE INFO
---------
  GitHub:  https://github.com/ReGenNow/ReGenNexus
  Docs:    https://github.com/ReGenNow/ReGenNexus/tree/main/docs
  Issues:  https://github.com/ReGenNow/ReGenNexus/issues
"""


if HAS_CLICK:
    # =========================================================================
    # Click-based CLI (preferred)
    # =========================================================================

    @click.group(invoke_without_command=True)
    @click.version_option(version=__version__, prog_name=__title__)
    @click.option("--man", "-m", is_flag=True, help="Show detailed manual/help")
    @click.pass_context
    def cli(ctx, man: bool):
        """RegenNexus UAP - Universal Adapter Protocol

        Fast, reliable mesh networking for devices, robots, and AI agents.

        \b
        Quick start:
          regen mesh start     Start mesh network daemon
          regen mesh status    Check status and peers
          regen doctor         Check installation

        \b
        For detailed documentation:
          regen --man          Show full manual
          regen mesh --help    Mesh network commands
        """
        if man:
            click.echo(HELP_TEXT)
            ctx.exit(0)
        elif ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    @cli.command()
    @click.option(
        "--config", "-c", default=None, help="Path to configuration file"
    )
    @click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
    @click.option("--port", "-p", default=8765, type=int, help="Port to listen on")
    @click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
    def start(config: Optional[str], host: str, port: int, debug: bool):
        """Start RegenNexus UAP server."""
        print_banner()

        from regennexus.core.protocol import RegenNexus

        click.echo(f"Starting RegenNexus UAP on {host}:{port}...")

        if config:
            click.echo(f"Using config: {config}")

        regen = RegenNexus(config=config)

        try:
            asyncio.run(regen.start())
            click.echo("RegenNexus UAP is running. Press Ctrl+C to stop.")

            # Keep running
            asyncio.get_event_loop().run_forever()

        except KeyboardInterrupt:
            click.echo("\nShutting down...")
            asyncio.run(regen.stop())
            click.echo("RegenNexus UAP stopped.")

    @cli.command()
    def stop():
        """Stop RegenNexus UAP server."""
        click.echo("Stopping RegenNexus UAP...")
        # TODO: Implement remote stop
        click.echo("Stop command sent.")

    @cli.command()
    def status():
        """Show RegenNexus UAP status."""
        click.echo("RegenNexus UAP Status")
        click.echo("=" * 40)
        # TODO: Implement status check
        click.echo("Status: Not implemented yet")

    @cli.group()
    def devices():
        """Device management commands."""
        pass

    @devices.command("list")
    def devices_list():
        """List all registered devices."""
        click.echo("Registered Devices:")
        click.echo("=" * 40)
        # TODO: Implement device listing
        click.echo("No devices registered.")

    @devices.command("info")
    @click.argument("device_id")
    def devices_info(device_id: str):
        """Show device information."""
        click.echo(f"Device Info: {device_id}")
        click.echo("=" * 40)
        # TODO: Implement device info
        click.echo("Device not found.")

    @cli.command()
    @click.argument("target")
    @click.argument("message")
    def send(target: str, message: str):
        """Send a message to a device or entity."""
        click.echo(f"Sending to {target}: {message}")

        try:
            # Try to parse as JSON
            content = json.loads(message)
        except json.JSONDecodeError:
            content = message

        # TODO: Implement message sending
        click.echo("Message sent (not implemented yet).")

    @cli.command()
    def benchmark():
        """Run performance benchmarks."""
        click.echo("Running benchmarks...")
        click.echo("=" * 40)
        # TODO: Implement benchmarks
        click.echo("Benchmarks not implemented yet.")

    # =========================================================================
    # Mesh Network Commands
    # =========================================================================

    @cli.group()
    def mesh():
        """Mesh network commands for LAN device discovery.

        \b
        The mesh network enables automatic peer discovery across your LAN.
        Devices find each other using mDNS, multicast, and broadcast UDP.

        \b
        Commands:
          start   Start the mesh daemon (background or foreground)
          stop    Stop the running daemon
          status  Check if daemon is running and show peers
          peers   List discovered peers
          send    Send a message to a peer

        \b
        Examples:
          regen mesh start                    # Background daemon
          regen mesh start -f                 # Foreground with logs
          regen mesh start -n mynode -t agent # Custom node ID and type
          regen mesh status                   # Check status
          regen mesh stop                     # Stop daemon
        """
        pass

    @mesh.command("start")
    @click.option("--node-id", "-n", default=None, help="Unique node identifier")
    @click.option("--type", "-t", "entity_type", default="device", help="Entity type (device, agent, service)")
    @click.option("--caps", "-c", multiple=True, help="Node capabilities (can specify multiple)")
    @click.option("--port", "-p", default=5353, type=int, help="UDP discovery port")
    @click.option("--peer", multiple=True, help="Known peer to connect to (ws://host:port)")
    @click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't daemonize)")
    @click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
    def mesh_start(node_id: Optional[str], entity_type: str, caps, port: int, peer, foreground: bool, debug: bool):
        """Start the mesh network daemon.

        Examples:
            regen mesh start
            regen mesh start -n my-mac -t controller -c audio -c display
            regen mesh start --peer ws://192.168.1.100:8765
            regen mesh start --foreground --debug
        """
        import logging
        from regennexus.core.daemon import MeshDaemon, is_daemon_running, get_daemon_pid

        # Check if already running
        if is_daemon_running():
            pid = get_daemon_pid()
            click.echo(f"Mesh daemon is already running (PID: {pid})")
            click.echo("Use 'regen mesh stop' to stop it first.")
            return

        # Setup logging
        if debug:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        elif foreground:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Create daemon
        daemon = MeshDaemon(
            node_id=node_id,
            entity_type=entity_type,
            capabilities=list(caps) if caps else [],
            known_peers=list(peer) if peer else [],
            udp_port=port if port != 5353 else None,  # Only override if not default
        )

        if foreground:
            click.echo("Starting mesh daemon in foreground...")
            click.echo("Press Ctrl+C to stop.")
            click.echo("=" * 50)
            daemon.run()  # Blocks until Ctrl+C
        else:
            # Start in background
            import subprocess
            import sys

            # Build command to run in background
            udp_port_arg = f"udp_port={port}," if port != 5353 else ""
            cmd = [sys.executable, "-c", f"""
import asyncio
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from regennexus.core.daemon import MeshDaemon
daemon = MeshDaemon(
    node_id={repr(node_id)},
    entity_type={repr(entity_type)},
    capabilities={list(caps) if caps else []},
    known_peers={list(peer) if peer else []},
    {udp_port_arg}
)
daemon.run()
"""]

            # Start detached process
            if sys.platform == "win32":
                # Windows: use CREATE_NEW_PROCESS_GROUP
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # Unix: use nohup-like behavior
                subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Wait a moment and check if it started
            import time
            time.sleep(1)

            if is_daemon_running():
                pid = get_daemon_pid()
                click.echo(f"Mesh daemon started (PID: {pid})")
                click.echo(f"Node ID: {node_id or '(auto-generated)'}")
                click.echo(f"Type: {entity_type}")
                if caps:
                    click.echo(f"Capabilities: {', '.join(caps)}")
                if peer:
                    click.echo(f"Connecting to peers: {', '.join(peer)}")
                click.echo("\nUse 'regen mesh status' to check peers.")
                click.echo("Use 'regen mesh stop' to stop the daemon.")
            else:
                click.echo("Failed to start mesh daemon. Try running with --foreground --debug")

    @mesh.command("stop")
    def mesh_stop():
        """Stop the mesh network daemon."""
        import os
        import signal
        from regennexus.core.daemon import is_daemon_running, get_daemon_pid, DEFAULT_PID_FILE

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            return

        pid = get_daemon_pid()
        if pid:
            click.echo(f"Stopping mesh daemon (PID: {pid})...")
            try:
                if sys.platform == "win32":
                    # Windows: use taskkill
                    import subprocess
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                else:
                    # Unix: send SIGTERM
                    os.kill(pid, signal.SIGTERM)

                # Wait for it to stop
                import time
                for _ in range(10):
                    time.sleep(0.5)
                    if not is_daemon_running():
                        break

                if is_daemon_running():
                    click.echo("Daemon didn't stop gracefully, forcing...")
                    if sys.platform != "win32":
                        os.kill(pid, signal.SIGKILL)
                    # Clean up PID file
                    try:
                        DEFAULT_PID_FILE.unlink()
                    except:
                        pass

                click.echo("Mesh daemon stopped.")

            except ProcessLookupError:
                click.echo("Process not found. Cleaning up PID file...")
                try:
                    DEFAULT_PID_FILE.unlink()
                except:
                    pass
            except PermissionError:
                click.echo(f"Permission denied. Try: sudo kill {pid}")
        else:
            click.echo("Could not determine daemon PID.")

    @mesh.command("status")
    @click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON")
    def mesh_status(as_json: bool):
        """Show mesh network status and connected peers."""
        from regennexus.core.daemon import is_daemon_running, get_daemon_pid

        running = is_daemon_running()
        pid = get_daemon_pid() if running else None

        if as_json:
            import json as json_module
            status = {
                "running": running,
                "pid": pid,
            }
            click.echo(json_module.dumps(status, indent=2))
        else:
            click.echo("Mesh Network Status")
            click.echo("=" * 50)

            if running:
                click.echo(f"Status: Running (PID: {pid})")
                click.echo("\nTo see peers, run the daemon in foreground:")
                click.echo("  regen mesh stop && regen mesh start --foreground")
            else:
                click.echo("Status: Not running")
                click.echo("\nStart with: regen mesh start")

    @mesh.command("peers")
    def mesh_peers():
        """List discovered peers (requires foreground daemon or IPC)."""
        from regennexus.core.daemon import is_daemon_running

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        click.echo("Peer listing requires IPC connection to daemon.")
        click.echo("For now, run daemon in foreground to see peers:")
        click.echo("  regen mesh start --foreground")

    @mesh.command("send")
    @click.argument("target")
    @click.argument("message")
    def mesh_send(target: str, message: str):
        """Send a message to a peer on the mesh network.

        Examples:
            regen mesh send win11-pc '{"hello": "world"}'
            regen mesh send raspi-001 "ping"
        """
        click.echo(f"Sending to {target}: {message}")
        click.echo("Note: Direct send requires IPC to running daemon.")
        click.echo("For now, use the Python API or run in foreground.")

    @cli.command()
    def doctor():
        """Check installation, dependencies, and mesh network status.

        Verifies that all required and optional dependencies are installed,
        checks network configuration, and reports mesh daemon status.
        """
        click.echo("RegenNexus UAP Doctor")
        click.echo("=" * 50)

        # Check Python version
        py_version = sys.version_info
        py_ok = py_version >= (3, 8)
        status = "✓" if py_ok else "✗"
        click.echo(f"\n{status} Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
        if not py_ok:
            click.echo("  (Python 3.8+ required)")

        # Check core dependencies
        click.echo("\nCore Dependencies:")
        dependencies = [
            ("pyyaml", "yaml", "Configuration files"),
            ("websockets", "websockets", "WebSocket transport"),
            ("click", "click", "CLI interface"),
        ]

        for name, module, desc in dependencies:
            try:
                __import__(module)
                click.echo(f"  ✓ {name} - {desc}")
            except ImportError:
                click.echo(f"  ✗ {name} - {desc} (pip install {name})")

        # Check mesh networking dependencies
        click.echo("\nMesh Networking:")
        mesh_deps = [
            ("zeroconf", "zeroconf", "mDNS discovery (recommended)"),
        ]

        for name, module, desc in mesh_deps:
            try:
                __import__(module)
                click.echo(f"  ✓ {name} - {desc}")
            except ImportError:
                click.echo(f"  - {name} - {desc} (pip install {name})")

        # Check mesh daemon status
        click.echo("\nMesh Daemon:")
        try:
            from regennexus.core.daemon import is_daemon_running, get_daemon_pid
            if is_daemon_running():
                pid = get_daemon_pid()
                click.echo(f"  ✓ Running (PID: {pid})")
            else:
                click.echo("  - Not running (regen mesh start)")
        except Exception as e:
            click.echo(f"  ✗ Error checking status: {e}")

        # Check network
        click.echo("\nNetwork:")
        try:
            import socket
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            try:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                click.echo(f"  ✓ Local IP: {local_ip}")
            finally:
                s.close()

            # Check if UDP port 5353 is available
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_sock.bind(("", 5353))
                test_sock.close()
                click.echo("  ✓ UDP port 5353 available")
            except OSError:
                click.echo("  - UDP port 5353 in use (mesh may already be running)")

        except Exception as e:
            click.echo(f"  ✗ Network check failed: {e}")

        # Check optional dependencies
        click.echo("\nOptional Dependencies:")
        optional = [
            ("pyserial", "serial", "Arduino support"),
            ("RPi.GPIO", "RPi.GPIO", "Raspberry Pi support"),
            ("rclpy", "rclpy", "ROS 2 support"),
        ]

        for name, module, desc in optional:
            try:
                __import__(module)
                click.echo(f"  ✓ {name} ({desc})")
            except ImportError:
                click.echo(f"  - {name} ({desc}) - not installed")

        click.echo("\nDoctor check complete.")

    def main():
        """Main entry point for CLI."""
        cli()

else:
    # =========================================================================
    # Argparse-based CLI (fallback)
    # =========================================================================

    def main():
        """Main entry point for CLI (argparse fallback)."""
        parser = argparse.ArgumentParser(
            prog="regen",
            description="RegenNexus UAP - Universal Adapter Protocol",
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"{__title__} {__version__}",
        )

        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # Start command
        start_parser = subparsers.add_parser("start", help="Start RegenNexus")
        start_parser.add_argument("--config", "-c", help="Config file path")
        start_parser.add_argument(
            "--host", "-h", default="0.0.0.0", help="Host to bind"
        )
        start_parser.add_argument(
            "--port", "-p", type=int, default=8765, help="Port"
        )

        # Status command
        subparsers.add_parser("status", help="Show status")

        # Devices command
        devices_parser = subparsers.add_parser("devices", help="Device commands")
        devices_sub = devices_parser.add_subparsers(dest="devices_command")
        devices_sub.add_parser("list", help="List devices")

        # Doctor command
        subparsers.add_parser("doctor", help="Check installation")

        args = parser.parse_args()

        if args.command == "start":
            print_banner()
            print(f"Starting RegenNexus UAP on {args.host}:{args.port}...")
            print("Note: Install 'click' for better CLI: pip install click")

        elif args.command == "status":
            print("RegenNexus UAP Status: Not implemented")

        elif args.command == "devices":
            if args.devices_command == "list":
                print("Devices: None registered")

        elif args.command == "doctor":
            print("Doctor: Run 'pip install click' for full CLI")

        else:
            parser.print_help()


if __name__ == "__main__":
    main()
