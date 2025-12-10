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

ARCHITECTURE
------------
The mesh daemon maintains persistent WebSocket connections to all peers:
  - 15-second keepalive pings maintain connection health
  - Auto-reconnect with exponential backoff (1, 2, 5, 10, 30 seconds)
  - CLI commands use daemon's connections for reliable messaging
  - No more connection drops or 100% packet loss!

QUICK START
-----------
  regen mesh start                  Start mesh daemon (background)
  regen mesh start -f               Start in foreground (see logs)
  regen mesh status                 Check if mesh is running
  regen mesh peers                  List connected peers
  regen mesh peers -v               Show connection state & RTT
  regen mesh stop                   Stop the mesh daemon

SHORTCUT COMMANDS
-----------------
  regen peers                       Same as 'regen mesh peers'
  regen peers -v                    Show detailed connection info
  regen doctor                      Check installation and dependencies

MESSAGING & COMMUNICATION
-------------------------
Send messages between mesh peers (routed via daemon):

  regen send <target> <message>     Send message to a peer
  regen send Server "Hello!"        Send text message
  regen send pi-hole "ping"         Ping a peer (gets pong response)
  regen send Server '{"cmd":"x"}'   Send JSON message

Options:
  -i, --intent TEXT     Message intent (default: message)
  -w, --wait            Wait for response from target
  -t, --timeout FLOAT   Response timeout in seconds (default: 5.0)

Examples:
  regen send Xavier-AGX "ping"                    # Test connectivity
  regen send Server "status" --wait --timeout 10  # Wait for response
  regen send pi-hole '{"action":"query"}' -i device.query

DEVICE MANAGEMENT
-----------------
View and manage devices on the mesh network:

  regen devices list                List all devices with capabilities
  regen devices list --json         Output as JSON
  regen devices info <device>       Show detailed info about a device

Examples:
  regen devices list                # Shows all peers + local node
  regen devices info pi-hole        # Shows type, capabilities, latency

BENCHMARKING & DIAGNOSTICS
--------------------------
Test mesh network performance using daemon's persistent connections:

  regen benchmark                   Run latency tests to all peers
  regen benchmark -c 20             Send 20 pings per peer
  regen benchmark -t 3.0            Set timeout to 3 seconds

Output shows:
  - Connection state (connected/reconnecting/disconnected)
  - Average, min, max latency per peer
  - Packet loss percentage
  - Overall network summary

CONNECTION STATES
-----------------
Peers have these connection states (shown with regen peers -v):
  ● connected     - Persistent WebSocket connection active
  ○ reconnecting  - Connection lost, auto-reconnecting
  ○ disconnected  - Not connected (discovery only)

MESH NETWORKING
---------------
The mesh network automatically discovers devices on your LAN using:
  - mDNS/Zeroconf (works on all routers, plug-and-play)
  - UDP multicast (fast, low latency)
  - UDP broadcast (fallback when multicast blocked)
  - Direct peer connections (--peer ws://host:8765)

Once discovered, the daemon establishes persistent WebSocket connections
that are maintained 24/7 with keepalive pings.

Start Options:
  -n, --node-id TEXT    Set custom node name (e.g., -n Mac)
  -t, --type TEXT       Entity type: device, agent, service
  -c, --caps TEXT       Capabilities (can repeat: -c audio -c gpu)
  --peer TEXT           Connect to peer directly (can repeat)
  -f, --foreground      Run with visible logs
  -d, --debug           Enable debug output

Examples:
  # Start mesh on Mac with custom name
  regen mesh start -n Mac -t device

  # Start with explicit peer connections (bypasses mDNS)
  regen mesh start -n server-ubuntu --peer ws://192.168.68.94:8765

  # Start with debug logging
  regen mesh start -f -d

PYTHON API
----------
  from regennexus.core.mesh import MeshNetwork, MeshConfig

  mesh = MeshNetwork(MeshConfig(node_id="my-device"))
  await mesh.start()

  # Auto-discovered peers
  for peer in mesh.get_peers():
      print(f"Found: {peer.node_id} ({peer.connection_state})")

  # Send message via persistent connection
  await mesh.send("other-device", {"hello": "world"})

  # Ping with accurate RTT via persistent connection
  rtt = await mesh.ping("other-device")
  print(f"RTT: {rtt}ms")

CONFIGURATION
-------------
Config files are stored at:
  macOS:   ~/Library/Application Support/RegenNexus/mesh.yaml
  Windows: %APPDATA%/RegenNexus/mesh.yaml
  Linux:   ~/.config/regennexus/mesh.yaml

Example mesh.yaml:
  node_id: Mac
  entity_type: device
  capabilities:
    - compute
    - storage
  keepalive_interval: 15.0   # Seconds between keepalive pings
  reconnect_delays: [1, 2, 5, 10, 30]  # Exponential backoff

PID files (for daemon management):
  macOS:   ~/Library/Application Support/RegenNexus/mesh.pid
  Windows: %APPDATA%/RegenNexus/mesh.pid
  Linux:   /var/run/regennexus/mesh.pid

TROUBLESHOOTING
---------------
  regen doctor                      Check dependencies
  regen mesh start -f -d            Run with debug output
  regen benchmark                   Test network connectivity
  regen peers -v                    Check connection states

  If devices don't discover each other:
  1. Ensure both are on the same subnet (192.168.x.x)
  2. Check firewall allows UDP port 5353 (mDNS) and 5454 (mesh)
  3. Check firewall allows TCP port 8765 (WebSocket)
  4. Use --peer ws://IP:8765 for direct connection
  5. On Windows, run: netsh advfirewall firewall add rule \\
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
          regen mesh peers     List connected peers
          regen peers          Shortcut for 'mesh peers'
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
        """Show RegenNexus UAP status (same as 'mesh status')."""
        from regennexus.core.daemon import is_daemon_running, get_daemon_pid

        click.echo("RegenNexus UAP Status")
        click.echo("=" * 50)

        running = is_daemon_running()
        pid = get_daemon_pid() if running else None

        if running:
            click.echo(f"Mesh Daemon: Running (PID: {pid})")
            click.echo("\nUse 'regen peers' to see connected peers.")
        else:
            click.echo("Mesh Daemon: Not running")
            click.echo("\nStart with: regen mesh start")

    @cli.group()
    def devices():
        """Device management commands.

        \b
        Lists devices on the mesh network and their capabilities.
        Devices include all mesh peers (computers, servers, IoT devices).
        """
        pass

    @devices.command("list")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    def devices_list(as_json: bool):
        """List all devices on the mesh network."""
        from regennexus.core.daemon import is_daemon_running

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        async def list_devices():
            try:
                import websockets
                import json as json_mod

                async with websockets.connect('ws://localhost:8765', ping_interval=None) as ws:
                    # Get welcome
                    welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    welcome_data = json_mod.loads(welcome)
                    local_node = welcome_data.get('content', {}).get('node_id', 'unknown')
                    local_type = welcome_data.get('content', {}).get('entity_type', 'device')
                    local_caps = welcome_data.get('content', {}).get('capabilities', [])

                    # Query peers
                    request = json_mod.dumps({
                        'sender_id': 'cli',
                        'recipient_id': local_node,
                        'intent': 'mesh.peers',
                        'content': {},
                        'metadata': {}
                    })
                    await ws.send(request)
                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    response_data = json_mod.loads(response)
                    peers_list = response_data.get('content', {}).get('peers', [])

                    # Build device list (local + peers)
                    devices = [{
                        'node_id': local_node,
                        'type': local_type,
                        'capabilities': local_caps,
                        'status': 'local'
                    }]

                    for peer in peers_list:
                        if isinstance(peer, dict):
                            devices.append({
                                'node_id': peer.get('node_id', 'unknown'),
                                'type': peer.get('entity_type', 'device'),
                                'capabilities': peer.get('capabilities', []),
                                'status': 'connected'
                            })
                        else:
                            devices.append({
                                'node_id': peer,
                                'type': 'device',
                                'capabilities': [],
                                'status': 'connected'
                            })

                    if as_json:
                        click.echo(json_mod.dumps(devices, indent=2))
                    else:
                        click.echo("Mesh Network Devices")
                        click.echo("=" * 60)
                        click.echo(f"{'Device ID':<20} {'Type':<10} {'Status':<12} {'Capabilities'}")
                        click.echo("-" * 60)

                        for dev in devices:
                            caps = ', '.join(dev['capabilities']) if dev['capabilities'] else '-'
                            status = dev['status']
                            if status == 'local':
                                status = 'local (self)'
                            click.echo(f"{dev['node_id']:<20} {dev['type']:<10} {status:<12} {caps}")

                        click.echo("-" * 60)
                        click.echo(f"Total: {len(devices)} devices")

            except ImportError:
                click.echo("Error: websockets package required.")
            except asyncio.TimeoutError:
                click.echo("Timeout connecting to mesh daemon.")
            except ConnectionRefusedError:
                click.echo("Cannot connect to mesh daemon.")
            except Exception as e:
                click.echo(f"Error listing devices: {e}")

        asyncio.run(list_devices())

    @devices.command("info")
    @click.argument("device_id")
    def devices_info(device_id: str):
        """Show detailed information about a device."""
        from regennexus.core.daemon import is_daemon_running

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        async def get_device_info():
            try:
                import websockets
                import json as json_mod

                async with websockets.connect('ws://localhost:8765', ping_interval=None) as ws:
                    # Get welcome
                    welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    welcome_data = json_mod.loads(welcome)
                    local_node = welcome_data.get('content', {}).get('node_id', 'unknown')

                    # Check if asking about local node
                    if device_id == local_node:
                        click.echo(f"Device: {local_node}")
                        click.echo("=" * 40)
                        click.echo(f"  Status: local (self)")
                        click.echo(f"  Type: {welcome_data.get('content', {}).get('entity_type', 'device')}")
                        caps = welcome_data.get('content', {}).get('capabilities', [])
                        click.echo(f"  Capabilities: {', '.join(caps) if caps else 'none'}")
                        return

                    # Query peers to find the device
                    request = json_mod.dumps({
                        'sender_id': 'cli',
                        'recipient_id': local_node,
                        'intent': 'mesh.peers',
                        'content': {},
                        'metadata': {}
                    })
                    await ws.send(request)
                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    response_data = json_mod.loads(response)
                    peers_list = response_data.get('content', {}).get('peers', [])

                    # Find the device
                    found = None
                    for peer in peers_list:
                        peer_id = peer.get('node_id', peer) if isinstance(peer, dict) else peer
                        if peer_id == device_id:
                            found = peer if isinstance(peer, dict) else {'node_id': peer}
                            break

                    if found:
                        click.echo(f"Device: {device_id}")
                        click.echo("=" * 40)
                        click.echo(f"  Status: connected")
                        click.echo(f"  Type: {found.get('entity_type', 'device')}")
                        caps = found.get('capabilities', [])
                        click.echo(f"  Capabilities: {', '.join(caps) if caps else 'none'}")

                        # Try to ping for latency
                        import time
                        start = time.perf_counter()
                        ping_msg = json_mod.dumps({
                            'sender_id': local_node,
                            'recipient_id': device_id,
                            'intent': 'mesh.ping',
                            'content': {},
                            'metadata': {}
                        })
                        await ws.send(ping_msg)
                        try:
                            await asyncio.wait_for(ws.recv(), timeout=2.0)
                            latency = (time.perf_counter() - start) * 1000
                            click.echo(f"  Latency: {latency:.1f}ms")
                        except asyncio.TimeoutError:
                            click.echo(f"  Latency: timeout")
                    else:
                        click.echo(f"Device '{device_id}' not found.")
                        click.echo("\nAvailable devices:")
                        for peer in peers_list:
                            peer_id = peer.get('node_id', peer) if isinstance(peer, dict) else peer
                            click.echo(f"  • {peer_id}")

            except ImportError:
                click.echo("Error: websockets package required.")
            except asyncio.TimeoutError:
                click.echo("Timeout connecting to mesh daemon.")
            except ConnectionRefusedError:
                click.echo("Cannot connect to mesh daemon.")
            except Exception as e:
                click.echo(f"Error getting device info: {e}")

        asyncio.run(get_device_info())

    @cli.command()
    @click.argument("target")
    @click.argument("message")
    @click.option("--intent", "-i", default="message", help="Message intent (default: message)")
    @click.option("--wait", "-w", is_flag=True, help="Wait for response")
    @click.option("--timeout", "-t", default=5.0, help="Response timeout in seconds")
    def send(target: str, message: str, intent: str, wait: bool, timeout: float):
        """Send a message to a device or entity.

        \b
        Examples:
          regen send Server "Hello!"
          regen send pi-hole "ping"              # Uses mesh.ping, gets pong back
          regen send Xavier-AGX '{"cmd": "x"}' --intent device.query

        \b
        Note: Use "ping" as message to test connectivity (gets pong response).
        Other messages are delivered but response depends on target having a handler.
        """
        from regennexus.core.daemon import is_daemon_running
        import time

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        # Special case: "ping" message uses mesh.ping intent for guaranteed response
        use_ping = message.lower() == "ping" and intent == "message"

        try:
            # Try to parse as JSON
            content = json.loads(message)
        except json.JSONDecodeError:
            content = {"text": message}

        async def send_message():
            try:
                import websockets
                import json as json_mod
                import uuid

                async with websockets.connect('ws://localhost:8765', ping_interval=None) as ws:
                    # Get welcome message to know our node ID
                    welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    welcome_data = json_mod.loads(welcome)
                    local_node = welcome_data.get('content', {}).get('node_id', 'cli')

                    # Create message
                    msg_id = str(uuid.uuid4())[:8]

                    if use_ping:
                        # Use mesh.ping for guaranteed pong response
                        actual_intent = "mesh.ping"
                        actual_content = {"seq": 0, "timestamp": time.time()}
                    else:
                        actual_intent = intent
                        actual_content = content

                    request = json_mod.dumps({
                        'sender_id': local_node,
                        'recipient_id': target,
                        'intent': actual_intent,
                        'content': actual_content,
                        'metadata': {'msg_id': msg_id, 'cli': True}
                    })

                    start_time = time.perf_counter()
                    await ws.send(request)
                    click.echo(f"✓ Message sent to {target}" + (" (ping)" if use_ping else ""))

                    if wait or use_ping:
                        if not use_ping:
                            click.echo(f"Waiting for response (timeout: {timeout}s)...")

                        # Wait for response from the target
                        deadline = time.perf_counter() + timeout
                        while time.perf_counter() < deadline:
                            try:
                                remaining = deadline - time.perf_counter()
                                response = await asyncio.wait_for(ws.recv(), timeout=remaining)
                                response_data = json_mod.loads(response)

                                resp_sender = response_data.get('sender_id', 'unknown')
                                resp_intent = response_data.get('intent', 'unknown')

                                # Check if this response is from our target
                                if resp_sender == target:
                                    elapsed = (time.perf_counter() - start_time) * 1000

                                    if use_ping and resp_intent == "mesh.pong":
                                        click.echo(f"✓ Pong from {target} ({elapsed:.1f}ms)")
                                    else:
                                        resp_content = response_data.get('content', {})
                                        click.echo(f"\nResponse from {resp_sender} ({elapsed:.1f}ms):")
                                        click.echo(f"  Intent: {resp_intent}")
                                        if isinstance(resp_content, dict):
                                            click.echo(f"  Content: {json_mod.dumps(resp_content, indent=4)}")
                                        else:
                                            click.echo(f"  Content: {resp_content}")
                                    return
                                # else: ignore messages from other senders, keep waiting

                            except asyncio.TimeoutError:
                                break

                        click.echo(f"No response from {target} (timeout)")

            except ImportError:
                click.echo("Error: websockets package required.")
            except asyncio.TimeoutError:
                click.echo("Timeout connecting to mesh daemon.")
            except ConnectionRefusedError:
                click.echo("Cannot connect to mesh daemon.")
            except Exception as e:
                click.echo(f"Error sending message: {e}")

        asyncio.run(send_message())

    @cli.command()
    @click.option("--count", "-c", default=10, help="Number of pings per peer")
    @click.option("--size", "-s", default=64, help="Payload size in bytes")
    @click.option("--timeout", "-t", default=5.0, help="Timeout per ping in seconds")
    def benchmark(count: int, size: int, timeout: float):
        """Run performance benchmarks on mesh network.

        \b
        Uses daemon's persistent connections for accurate latency measurements.
        The daemon maintains WebSocket connections to all peers with keepalive.

        \b
        Tests:
          - Latency: Round-trip time to each peer via daemon
          - Packet loss: Percentage of failed pings
          - Connection state: Whether peer is connected/reconnecting

        \b
        Examples:
          regen benchmark
          regen benchmark --count 20 --timeout 3.0
        """
        from regennexus.core.daemon import is_daemon_running
        import time

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        click.echo("RegenNexus Mesh Benchmark")
        click.echo("=" * 60)
        click.echo("Using daemon's persistent connections for accurate measurements")
        click.echo("-" * 60)

        async def run_benchmarks():
            try:
                import websockets
                import json as json_mod

                async with websockets.connect('ws://localhost:8765', ping_interval=None) as ws:
                    # Get welcome
                    welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    welcome_data = json_mod.loads(welcome)
                    local_node = welcome_data.get('content', {}).get('node_id', 'unknown')

                    click.echo(f"Local Node: {local_node}")

                    # Send benchmark request to daemon
                    request = json_mod.dumps({
                        'sender_id': 'cli-benchmark',
                        'recipient_id': local_node,
                        'intent': 'cli.benchmark',
                        'content': {'count': count, 'timeout': timeout},
                        'metadata': {}
                    })
                    await ws.send(request)

                    # Wait for benchmark results (may take a while)
                    benchmark_timeout = (count * timeout * 5) + 10  # Allow time for all pings
                    click.echo(f"Running benchmark ({count} pings per peer, timeout: {timeout}s)...")
                    click.echo("")

                    # Keep reading until we get cli.benchmark.result
                    deadline = time.perf_counter() + benchmark_timeout
                    while time.perf_counter() < deadline:
                        try:
                            remaining = deadline - time.perf_counter()
                            response = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30))
                            response_data = json_mod.loads(response)

                            if response_data.get('intent') == 'cli.benchmark.result':
                                content = response_data.get('content', {})

                                if content.get('error'):
                                    click.echo(f"Error: {content['error']}")
                                    return

                                results = content.get('results', {})
                                peer_count = content.get('peer_count', 0)

                                click.echo(f"{'Peer':<25} {'State':<12} {'Sent':<6} {'Recv':<6} {'Loss':<8} {'Avg':<10} {'Min':<10} {'Max':<10}")
                                click.echo("-" * 90)

                                all_rtts = []
                                total_sent = 0
                                total_recv = 0

                                for peer_id, stats in results.items():
                                    state = stats.get('connection_state', 'unknown')[:10]
                                    sent = stats.get('sent', 0)
                                    recv = stats.get('received', 0)
                                    loss = stats.get('loss_percent', 100)
                                    avg_rtt = stats.get('avg_rtt_ms')
                                    min_rtt = stats.get('min_rtt_ms')
                                    max_rtt = stats.get('max_rtt_ms')

                                    avg_str = f"{avg_rtt:.1f}ms" if avg_rtt else "-"
                                    min_str = f"{min_rtt:.1f}ms" if min_rtt else "-"
                                    max_str = f"{max_rtt:.1f}ms" if max_rtt else "-"
                                    loss_str = f"{loss:.0f}%"

                                    click.echo(f"{peer_id:<25} {state:<12} {sent:<6} {recv:<6} {loss_str:<8} {avg_str:<10} {min_str:<10} {max_str:<10}")

                                    total_sent += sent
                                    total_recv += recv
                                    if stats.get('rtts'):
                                        all_rtts.extend([r for r in stats['rtts'] if r is not None])

                                # Summary
                                click.echo("-" * 90)
                                click.echo(f"\nSummary:")
                                click.echo(f"  Peers tested: {peer_count}")
                                click.echo(f"  Total pings: {total_recv}/{total_sent} ({(total_recv/total_sent*100) if total_sent else 0:.0f}% success)")
                                if all_rtts:
                                    import statistics
                                    click.echo(f"  Overall latency: avg={statistics.mean(all_rtts):.1f}ms  min={min(all_rtts):.1f}ms  max={max(all_rtts):.1f}ms")
                                return

                        except asyncio.TimeoutError:
                            continue

                    click.echo("Benchmark timed out waiting for results")

            except ImportError:
                click.echo("Error: websockets package required.")
            except asyncio.TimeoutError:
                click.echo("Timeout connecting to mesh daemon.")
            except ConnectionRefusedError:
                click.echo("Cannot connect to mesh daemon.")
            except Exception as e:
                click.echo(f"Error running benchmark: {e}")

        asyncio.run(run_benchmarks())

    @cli.command("peers")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    @click.option("--verbose", "-v", is_flag=True, help="Show detailed peer info (connection state, RTT)")
    def peers_shortcut(as_json: bool, verbose: bool):
        """List mesh network peers with connection status.

        \b
        Shows peers connected via daemon's persistent WebSocket connections.
        Use --verbose for connection state and last RTT measurements.
        """
        from regennexus.core.daemon import is_daemon_running
        import asyncio

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        async def query_peers():
            try:
                import websockets
                import json as json_mod
                import time

                async with websockets.connect('ws://localhost:8765', ping_interval=None) as ws:
                    welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    welcome_data = json_mod.loads(welcome)
                    local_node = welcome_data.get('content', {}).get('node_id', 'unknown')

                    # Use cli.peers intent for detailed peer info
                    request = json_mod.dumps({
                        'sender_id': 'cli-peers',
                        'recipient_id': local_node,
                        'intent': 'cli.peers',
                        'content': {},
                        'metadata': {}
                    })
                    await ws.send(request)

                    # Wait for cli.peers.result
                    deadline = time.perf_counter() + 5.0
                    while time.perf_counter() < deadline:
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            response_data = json_mod.loads(response)

                            if response_data.get('intent') == 'cli.peers.result':
                                content = response_data.get('content', {})
                                peers_list = content.get('peers', [])
                                local_id = content.get('local_id', local_node)

                                if as_json:
                                    click.echo(json_mod.dumps({
                                        'local_node': local_id,
                                        'peers': peers_list
                                    }, indent=2))
                                else:
                                    click.echo("Mesh Network Peers")
                                    click.echo("=" * 70)
                                    click.echo(f"Local Node: {local_id}")
                                    click.echo("-" * 70)

                                    if peers_list:
                                        if verbose:
                                            click.echo(f"\n{'Peer ID':<20} {'Type':<10} {'State':<12} {'RTT':<10} {'Transport'}")
                                            click.echo("-" * 70)
                                            for peer in peers_list:
                                                node_id = peer.get('node_id', 'unknown')[:18]
                                                entity_type = peer.get('entity_type', 'device')[:8]
                                                state = peer.get('connection_state', 'unknown')[:10]
                                                rtt = peer.get('last_rtt_ms')
                                                rtt_str = f"{rtt:.1f}ms" if rtt else "-"
                                                transport = peer.get('transport', 'unknown')[:10]
                                                click.echo(f"  {node_id:<18} {entity_type:<10} {state:<12} {rtt_str:<10} {transport}")
                                        else:
                                            click.echo(f"\nConnected Peers ({len(peers_list)}):")
                                            for peer in peers_list:
                                                node_id = peer.get('node_id', 'unknown')
                                                state = peer.get('connection_state', '')
                                                state_icon = "●" if state == "connected" else "○" if state == "reconnecting" else "○"
                                                click.echo(f"  {state_icon} {node_id}")
                                            click.echo("\nUse --verbose for detailed connection info.")
                                    else:
                                        click.echo("\nNo peers discovered yet.")
                                        click.echo("Other devices need 'regen mesh start' running.")
                                return

                        except asyncio.TimeoutError:
                            continue

                    click.echo("Timeout waiting for peer list")

            except ImportError:
                click.echo("Error: websockets package required.")
            except asyncio.TimeoutError:
                click.echo("Timeout connecting to mesh daemon.")
            except ConnectionRefusedError:
                click.echo("Cannot connect to mesh daemon.")
            except Exception as e:
                click.echo(f"Error querying peers: {e}")

        asyncio.run(query_peers())

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
    @click.option("--port", "-p", default=5454, type=int, help="UDP discovery port")
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
            udp_port=port if port != 5454 else None,  # Only override if not default
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
            udp_port_arg = f"udp_port={port}," if port != 5454 else ""
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
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    @click.option("--verbose", "-v", is_flag=True, help="Show detailed peer info")
    def mesh_peers(as_json: bool, verbose: bool):
        """List discovered peers on the mesh network with connection status."""
        from regennexus.core.daemon import is_daemon_running
        import asyncio

        if not is_daemon_running():
            click.echo("Mesh daemon is not running.")
            click.echo("Start with: regen mesh start")
            return

        async def query_peers():
            try:
                import websockets
                import json as json_mod
                import time

                async with websockets.connect('ws://localhost:8765', ping_interval=None) as ws:
                    welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    welcome_data = json_mod.loads(welcome)
                    local_node = welcome_data.get('content', {}).get('node_id', 'unknown')

                    # Use cli.peers intent for detailed peer info
                    request = json_mod.dumps({
                        'sender_id': 'cli-mesh-peers',
                        'recipient_id': local_node,
                        'intent': 'cli.peers',
                        'content': {},
                        'metadata': {}
                    })
                    await ws.send(request)

                    # Wait for cli.peers.result
                    deadline = time.perf_counter() + 5.0
                    while time.perf_counter() < deadline:
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            response_data = json_mod.loads(response)

                            if response_data.get('intent') == 'cli.peers.result':
                                content = response_data.get('content', {})
                                peers_list = content.get('peers', [])
                                local_id = content.get('local_id', local_node)

                                if as_json:
                                    click.echo(json_mod.dumps({
                                        'local_node': local_id,
                                        'peers': peers_list
                                    }, indent=2))
                                else:
                                    click.echo("Mesh Network Peers")
                                    click.echo("=" * 70)
                                    click.echo(f"Local Node: {local_id}")
                                    click.echo("-" * 70)

                                    if peers_list:
                                        if verbose:
                                            click.echo(f"\n{'Peer ID':<20} {'Type':<10} {'State':<12} {'RTT':<10} {'Transport'}")
                                            click.echo("-" * 70)
                                            for peer in peers_list:
                                                node_id = peer.get('node_id', 'unknown')[:18]
                                                entity_type = peer.get('entity_type', 'device')[:8]
                                                state = peer.get('connection_state', 'unknown')[:10]
                                                rtt = peer.get('last_rtt_ms')
                                                rtt_str = f"{rtt:.1f}ms" if rtt else "-"
                                                transport = peer.get('transport', 'unknown')[:10]
                                                click.echo(f"  {node_id:<18} {entity_type:<10} {state:<12} {rtt_str:<10} {transport}")
                                        else:
                                            click.echo(f"\nConnected Peers ({len(peers_list)}):")
                                            for peer in peers_list:
                                                node_id = peer.get('node_id', 'unknown')
                                                state = peer.get('connection_state', '')
                                                state_icon = "●" if state == "connected" else "○" if state == "reconnecting" else "○"
                                                click.echo(f"  {state_icon} {node_id}")
                                            click.echo("\nUse --verbose for detailed connection info.")
                                    else:
                                        click.echo("\nNo peers discovered yet.")
                                        click.echo("Other devices need 'regen mesh start' running.")
                                return

                        except asyncio.TimeoutError:
                            continue

                    click.echo("Timeout waiting for peer list")

            except ImportError:
                click.echo("Error: websockets package required. Install with:")
                click.echo("  pip install websockets")
            except asyncio.TimeoutError:
                click.echo("Timeout connecting to mesh daemon.")
            except ConnectionRefusedError:
                click.echo("Cannot connect to mesh daemon on ws://localhost:8765")
                click.echo("Daemon may be starting up. Try again in a moment.")
            except Exception as e:
                click.echo(f"Error querying peers: {e}")

        asyncio.run(query_peers())

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

            # Check if UDP port 5454 is available (RegenNexus UDP messaging port)
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_sock.bind(("", 5454))
                test_sock.close()
                click.echo("  ✓ UDP port 5454 available")
            except OSError:
                click.echo("  - UDP port 5454 in use (mesh may already be running)")

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
            "--host", "-H", default="0.0.0.0", help="Host to bind"
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
