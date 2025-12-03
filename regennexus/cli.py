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


if HAS_CLICK:
    # =========================================================================
    # Click-based CLI (preferred)
    # =========================================================================

    @click.group()
    @click.version_option(version=__version__, prog_name=__title__)
    def cli():
        """RegenNexus UAP - Universal Adapter Protocol

        Fast, reliable communication for devices, robots, and applications.
        """
        pass

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

    @cli.command()
    def doctor():
        """Check installation and dependencies."""
        click.echo("RegenNexus UAP Doctor")
        click.echo("=" * 40)

        # Check Python version
        py_version = sys.version_info
        click.echo(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")

        # Check dependencies
        dependencies = [
            ("pyyaml", "yaml"),
            ("websockets", "websockets"),
            ("aiohttp", "aiohttp"),
            ("pycryptodome", "Crypto"),
            ("fastapi", "fastapi"),
        ]

        click.echo("\nDependencies:")
        for name, module in dependencies:
            try:
                __import__(module)
                click.echo(f"  ✓ {name}")
            except ImportError:
                click.echo(f"  ✗ {name} (not installed)")

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
