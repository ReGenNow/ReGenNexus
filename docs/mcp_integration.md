# MCP Integration Guide

RegenNexus UAP can be used as an MCP (Model Context Protocol) server, allowing AI assistants like Claude to directly control hardware devices.

## What is MCP?

MCP (Model Context Protocol) is a standard protocol that allows AI applications to connect to external tools and data sources. RegenNexus implements MCP to expose hardware as AI-accessible tools.

```
[Claude Desktop] <--MCP--> [RegenNexus Server] <--UAP--> [Hardware]
```

## Quick Start

### 1. Configure Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "regennexus": {
            "command": "python",
            "args": ["-m", "regennexus.mcp_server"]
        }
    }
}
```

### 2. Restart Claude Desktop

The RegenNexus tools will now be available to Claude.

### 3. Ask Claude to Control Hardware

Example prompts:
- "List all connected devices"
- "Turn on GPIO pin 17 on raspi-001"
- "Move the robot arm to pick position"
- "Read the temperature sensor"

## Available MCP Tools

### gpio_write
Control GPIO pins on connected devices.

```json
{
    "device_id": "raspi-001",
    "pin": 17,
    "value": 1
}
```

### robot_arm_move
Move a robotic arm to specified positions.

```json
{
    "device_id": "arm-001",
    "positions": [0, 30, -45, 0, 90, 0, 0],
    "duration": 2.0
}
```

### gripper_control
Open or close a robotic gripper.

```json
{
    "device_id": "arm-001",
    "action": "close",
    "force": 15.0
}
```

### read_sensor
Read sensor values from devices.

```json
{
    "device_id": "raspi-001",
    "sensor_type": "temperature"
}
```

### list_devices
List all connected hardware devices.

```json
{}
```

## WebSocket Mode

For LAN access (multiple machines), run as WebSocket server:

```bash
python -m regennexus.mcp_server --mode websocket --port 8765
```

Connect from other machines using:
```
ws://server-ip:8765
```

## Mesh Network Integration

Enable auto-discovery of devices across the network:

```bash
python -m regennexus.mcp_server --mesh --mesh-port 5353
```

With mesh enabled:
- Devices automatically appear as MCP tools
- Claude can control any discovered device
- Hot-plug support (devices can join/leave)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Desktop                         │
│                    (MCP Client)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ MCP Protocol (stdio/WebSocket)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  RegenNexus MCP Server                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  MCP Tools  │  │  Resources  │  │   Prompts   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ UAP Protocol
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Mesh Network                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Raspi Pi │  │  Jetson  │  │  Arduino │  │ Robot Arm│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Example: Pick and Place with Claude

User prompt:
> "Pick up the object in front of the robot and place it on the right side"

Claude's actions:
1. Calls `list_devices` to find the arm
2. Calls `robot_arm_move` to position over object
3. Calls `gripper_control` with action="close"
4. Calls `robot_arm_move` to move to right position
5. Calls `gripper_control` with action="open"

## Custom Tool Registration

Add your own hardware tools:

```python
from regennexus.bridges.mcp_bridge import MCPServer, MCPTool

server = MCPServer()

server.register_tool(MCPTool(
    name="my_device_control",
    description="Control my custom device",
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "value": {"type": "number"}
        },
        "required": ["action"]
    },
    handler=my_handler_function
))

await server.run_stdio()
```

## Resource Subscriptions

Register sensor data as MCP resources:

```python
from regennexus.bridges.mcp_bridge import MCPResource

server.register_resource(MCPResource(
    uri="sensor://living-room/temperature",
    name="Living Room Temperature",
    description="Current temperature reading",
    handler=get_temperature_reading
))
```

Claude can then read these resources to get sensor data.

## Ollama / LM Studio Integration

For local LLM control (without Claude Desktop):

```python
from regennexus.bridges.llm_bridge import LLMBridge, LLMConfig

config = LLMConfig(
    provider="ollama",
    model="llama3",
    host="localhost",
    port=11434
)

llm = LLMBridge(config)
response = await llm.chat("Turn on the lights")
```

See `examples/mcp_integration/full_ai_hardware_demo.py` for a complete example.

## Troubleshooting

### Claude doesn't see the tools
1. Check that `claude_desktop_config.json` is valid JSON
2. Restart Claude Desktop after changes
3. Check server logs: `python -m regennexus.mcp_server 2>&1`

### WebSocket connection refused
1. Check firewall settings
2. Verify the port is not in use
3. Use `--host 0.0.0.0` for network access

### Devices not discovered
1. Ensure mesh is enabled: `--mesh`
2. Check that devices are on same network
3. Verify UDP port 5353 is open
