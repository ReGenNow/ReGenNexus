#!/usr/bin/env python3
"""
RegenNexus UAP - MCP Hardware Control Demo

Demonstrates LLM (Claude) controlling hardware through MCP protocol.

This example shows:
1. Starting the MCP server with hardware tools
2. Claude Desktop calling hardware tools
3. Reading sensor data as MCP resources

Usage:
    # Start MCP server for Claude Desktop
    python -m regennexus.mcp_server

    # Or run this demo directly with WebSocket
    python mcp_hardware_demo.py --mode websocket

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import sys

sys.path.insert(0, "../../")

from regennexus.bridges.mcp_bridge import MCPServer, MCPTool, MCPResource


# Simulated hardware state
hardware_state = {
    "arm-001": {
        "type": "amber_b1",
        "positions": [0, 0, 0, 0, 0, 0, 0],
        "gripper": "open",
        "enabled": True,
    },
    "raspi-001": {
        "type": "raspberry_pi",
        "gpio": {17: 0, 18: 0, 27: 0},
        "sensors": {"temperature": 22.5, "humidity": 45.0},
    },
}


async def gpio_write_handler(args):
    """Handle GPIO write requests."""
    device_id = args.get("device_id")
    pin = args.get("pin")
    value = args.get("value")

    if device_id not in hardware_state:
        return {"error": f"Device {device_id} not found"}

    device = hardware_state[device_id]
    if "gpio" not in device:
        return {"error": f"Device {device_id} has no GPIO"}

    device["gpio"][pin] = value
    return {
        "success": True,
        "device": device_id,
        "pin": pin,
        "value": value,
        "message": f"GPIO {pin} set to {'HIGH' if value else 'LOW'}",
    }


async def arm_move_handler(args):
    """Handle robot arm movement requests."""
    device_id = args.get("device_id")
    positions = args.get("positions", [])
    duration = args.get("duration", 2.0)

    if device_id not in hardware_state:
        return {"error": f"Device {device_id} not found"}

    device = hardware_state[device_id]
    if device["type"] not in ["amber_b1", "lucid_one"]:
        return {"error": f"Device {device_id} is not a robot arm"}

    device["positions"] = positions
    return {
        "success": True,
        "device": device_id,
        "positions": positions,
        "duration": duration,
        "message": f"Arm moving to {positions} over {duration}s",
    }


async def gripper_handler(args):
    """Handle gripper control requests."""
    device_id = args.get("device_id")
    action = args.get("action")
    force = args.get("force", 10.0)

    if device_id not in hardware_state:
        return {"error": f"Device {device_id} not found"}

    device = hardware_state[device_id]
    device["gripper"] = action
    return {
        "success": True,
        "device": device_id,
        "action": action,
        "force": force if action == "close" else None,
        "message": f"Gripper {'opened' if action == 'open' else f'closed with {force}N'}",
    }


async def read_sensor_handler(args):
    """Handle sensor read requests."""
    device_id = args.get("device_id")
    sensor_type = args.get("sensor_type")

    if device_id not in hardware_state:
        return {"error": f"Device {device_id} not found"}

    device = hardware_state[device_id]
    if "sensors" not in device:
        return {"error": f"Device {device_id} has no sensors"}

    value = device["sensors"].get(sensor_type)
    if value is None:
        return {"error": f"Sensor {sensor_type} not found"}

    return {
        "success": True,
        "device": device_id,
        "sensor": sensor_type,
        "value": value,
        "unit": "°C" if sensor_type == "temperature" else "%",
    }


async def list_devices_handler(args):
    """List all connected devices."""
    devices = []
    for device_id, info in hardware_state.items():
        devices.append({
            "id": device_id,
            "type": info["type"],
            "capabilities": list(info.keys()),
        })
    return {"devices": devices, "count": len(devices)}


def create_demo_server():
    """Create MCP server with demo handlers."""
    server = MCPServer(name="regennexus-demo", version="0.2.0")

    # Register tools with handlers
    server.register_tool(MCPTool(
        name="gpio_write",
        description="Set a GPIO pin to HIGH (1) or LOW (0)",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "Device ID"},
                "pin": {"type": "integer", "description": "GPIO pin number"},
                "value": {"type": "integer", "enum": [0, 1], "description": "Pin value"},
            },
            "required": ["device_id", "pin", "value"],
        },
        handler=gpio_write_handler,
    ))

    server.register_tool(MCPTool(
        name="robot_arm_move",
        description="Move a robotic arm to specified joint positions",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "positions": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Joint positions in degrees (7 values)",
                },
                "duration": {"type": "number", "description": "Move duration in seconds"},
            },
            "required": ["device_id", "positions"],
        },
        handler=arm_move_handler,
    ))

    server.register_tool(MCPTool(
        name="gripper_control",
        description="Open or close a robotic gripper",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "action": {"type": "string", "enum": ["open", "close"]},
                "force": {"type": "number", "description": "Grip force in Newtons"},
            },
            "required": ["device_id", "action"],
        },
        handler=gripper_handler,
    ))

    server.register_tool(MCPTool(
        name="read_sensor",
        description="Read value from a sensor",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "sensor_type": {
                    "type": "string",
                    "enum": ["temperature", "humidity", "distance", "light"],
                },
            },
            "required": ["device_id", "sensor_type"],
        },
        handler=read_sensor_handler,
    ))

    server.register_tool(MCPTool(
        name="list_devices",
        description="List all connected hardware devices",
        input_schema={"type": "object", "properties": {}},
        handler=list_devices_handler,
    ))

    # Register resources (sensors)
    server.register_resource(MCPResource(
        uri="sensor://raspi-001/temperature",
        name="Temperature Sensor",
        description="Current temperature reading from Raspberry Pi",
        handler=lambda: {"value": 22.5, "unit": "°C"},
    ))

    server.register_resource(MCPResource(
        uri="device://arm-001/state",
        name="Robot Arm State",
        description="Current state of Amber B1 arm",
        handler=lambda: hardware_state["arm-001"],
    ))

    return server


async def run_demo():
    """Run the MCP demo server."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Hardware Demo")
    parser.add_argument("--mode", choices=["stdio", "websocket"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = create_demo_server()

    print("=" * 60)
    print("RegenNexus UAP - MCP Hardware Demo")
    print("=" * 60)
    print()
    print("Registered tools:")
    print("  - gpio_write: Control GPIO pins")
    print("  - robot_arm_move: Move robotic arm")
    print("  - gripper_control: Open/close gripper")
    print("  - read_sensor: Read sensor values")
    print("  - list_devices: List connected devices")
    print()
    print("Registered resources:")
    print("  - sensor://raspi-001/temperature")
    print("  - device://arm-001/state")
    print()

    if args.mode == "stdio":
        print("Starting in stdio mode (for Claude Desktop)")
        print("Configure claude_desktop_config.json:")
        print(json.dumps({
            "mcpServers": {
                "regennexus": {
                    "command": "python",
                    "args": ["-m", "regennexus.mcp_server"],
                }
            }
        }, indent=2))
        await server.run_stdio()
    else:
        print(f"Starting WebSocket server on ws://localhost:{args.port}")
        print("Connect from LLM applications using MCP over WebSocket")
        await server.run_websocket("0.0.0.0", args.port)


if __name__ == "__main__":
    asyncio.run(run_demo())
