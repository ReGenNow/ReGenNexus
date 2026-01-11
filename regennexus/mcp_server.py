#!/usr/bin/env python3
"""
RegenNexus UAP - MCP Server Entry Point

Run RegenNexus as an MCP server for Claude Desktop or other MCP clients.

Usage (stdio for Claude Desktop):
    python -m regennexus.mcp_server

Usage (WebSocket for LAN):
    python -m regennexus.mcp_server --mode websocket --port 8765

Configure in Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "regennexus": {
                "command": "python",
                "args": ["-m", "regennexus.mcp_server"]
            }
        }
    }

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import argparse
import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr  # Log to stderr, keep stdout for MCP
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="RegenNexus UAP MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run as stdio server (for Claude Desktop)
    python -m regennexus.mcp_server

    # Run as WebSocket server (for LAN access)
    python -m regennexus.mcp_server --mode websocket --port 8765

    # Connect to mesh network for device discovery
    python -m regennexus.mcp_server --mesh --mesh-port 5353
        """
    )

    parser.add_argument(
        "--mode",
        choices=["stdio", "websocket"],
        default="stdio",
        help="Server mode (default: stdio for Claude Desktop)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="WebSocket bind host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket port"
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Connect to UAP mesh for device discovery"
    )
    parser.add_argument(
        "--mesh-port",
        type=int,
        default=5353,
        help="Mesh UDP discovery port"
    )
    parser.add_argument(
        "--node-id",
        default="mcp-server",
        help="Mesh node ID"
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from regennexus.bridges.mcp_bridge import create_hardware_mcp_server

    # Create MCP server with pre-configured hardware tools
    server = create_hardware_mcp_server()

    # Connect to mesh if requested
    if args.mesh:
        from regennexus.core.mesh import MeshConfig

        mesh_config = MeshConfig(
            node_id=args.node_id,
            entity_type="mcp_server",
            capabilities=["mcp", "hardware_control", "ai_bridge"],
            udp_port=args.mesh_port,
        )

        if await server.connect_mesh(mesh_config):
            logger.info("Connected to UAP mesh network")
        else:
            logger.warning("Failed to connect to mesh (continuing without discovery)")

    # Run server
    if args.mode == "stdio":
        logger.info("Starting MCP server on stdio (for Claude Desktop)")
        await server.run_stdio()
    else:
        logger.info(f"Starting MCP WebSocket server on ws://{args.host}:{args.port}")
        await server.run_websocket(args.host, args.port)


if __name__ == "__main__":
    asyncio.run(main())
