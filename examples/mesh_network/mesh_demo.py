#!/usr/bin/env python3
"""
RegenNexus UAP - Mesh Network Demo

Demonstrates plug-and-play device discovery on a network.

Run this script on multiple devices (Raspberry Pi, Jetson, PC, etc.)
and they will automatically discover each other.

Usage:
    # On Raspberry Pi
    python mesh_demo.py --id raspi-001 --type device --caps gpio,camera

    # On Jetson
    python mesh_demo.py --id jetson-001 --type device --caps gpu,camera

    # On PC (controller)
    python mesh_demo.py --id controller --type controller --caps command

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import argparse
import asyncio
import sys

# Add parent directory for development
sys.path.insert(0, "../../")

from regennexus.core.mesh import MeshNetwork, MeshConfig, MeshNode
from regennexus.core.message import Message


async def main():
    parser = argparse.ArgumentParser(description="RegenNexus Mesh Network Demo")
    parser.add_argument("--id", default=None, help="Node ID (auto-generated if not set)")
    parser.add_argument("--type", default="device", help="Entity type")
    parser.add_argument("--caps", default="", help="Comma-separated capabilities")
    parser.add_argument("--port", type=int, default=5353, help="UDP discovery port")
    args = parser.parse_args()

    # Parse capabilities
    capabilities = [c.strip() for c in args.caps.split(",") if c.strip()]

    # Create mesh config
    config = MeshConfig(
        node_id=args.id,
        entity_type=args.type,
        capabilities=capabilities,
        udp_port=args.port,
    )

    # Create mesh network
    mesh = MeshNetwork(config)

    # Register handlers
    @mesh.on_peer
    async def on_peer(peer: MeshNode, event: str):
        if event == "connected":
            print(f"\n[+] New peer discovered: {peer.node_id}")
            print(f"    Type: {peer.entity_type}")
            print(f"    Capabilities: {peer.capabilities}")
            print(f"    Transport: {peer.transport.value}")
        elif event == "disconnected":
            print(f"\n[-] Peer left: {peer.node_id}")
        elif event == "timeout":
            print(f"\n[!] Peer timed out: {peer.node_id}")

    @mesh.on_message
    async def on_message(message: Message):
        print(f"\n[MSG] From {message.sender_id}: {message.content}")

    # Start mesh
    print("=" * 60)
    print("RegenNexus UAP - Mesh Network Demo")
    print("=" * 60)

    if not await mesh.start():
        print("Failed to start mesh network")
        return

    print(f"\nNode ID: {mesh.node_id}")
    print(f"Type: {config.entity_type}")
    print(f"Capabilities: {capabilities}")
    print(f"\nListening for peers on UDP port {args.port}...")
    print("Press Ctrl+C to exit\n")

    # Main loop - show status and handle commands
    try:
        while True:
            await asyncio.sleep(10)

            # Show current peers
            peers = mesh.get_peers()
            print(f"\n[STATUS] Online peers: {len(peers)}")
            for peer in peers:
                print(f"  - {peer.node_id} ({peer.entity_type})")

            # Broadcast heartbeat
            await mesh.broadcast({
                "event": "heartbeat",
                "node": mesh.node_id,
            })

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    await mesh.stop()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
