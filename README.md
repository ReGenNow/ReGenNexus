# RegenNexus UAP

**Universal Adapter Protocol** - Connect devices, robots, apps, and AI agents with minimal latency and maximum security. MCP-compatible for seamless AI integration.

![ReGenNexus Logo](https://raw.githubusercontent.com/ReGenNow/ReGenNexus/main/images/rguapa.png)

## Installation

```bash
pip install regennexus
```

Or with all features:
```bash
pip install regennexus[full]
```

## Quick Start

```python
import asyncio
from regennexus import RegenNexusProtocol

async def main():
    protocol = RegenNexusProtocol()
    await protocol.initialize()

    # Register entities
    await protocol.registry.register_entity(
        entity_id="sensor_01",
        entity_type="device",
        capabilities=["temperature", "humidity"]
    )

    # Send messages
    await protocol.send_message(
        sender="controller",
        recipient="sensor_01",
        intent="read",
        payload={"sensors": ["temperature"]}
    )

    await protocol.shutdown()

asyncio.run(main())
```

## Features

### Device Support
- **Raspberry Pi** - GPIO, PWM, camera, sensors
- **Arduino** - Digital/analog I/O, serial commands
- **NVIDIA Jetson** - GPU, CUDA, camera, inference
- **IoT Devices** - MQTT, HTTP, CoAP protocols

### Robotic Arms
- **Amber B1** - 7-DOF control, gripper, trajectory
- **Lucid One** - Cartesian control, force sensing, teach mode

```python
from regennexus.plugins import get_amber_b1_plugin

AmberB1 = get_amber_b1_plugin()
arm = AmberB1(entity_id="arm_001", mock_mode=True)
await arm.initialize()

# Move joints
await arm.move_to([0, 45, -30, 0, 90, 0, 0], duration=2.0)

# Gripper control
await arm.open_gripper()
await arm.close_gripper(force=15.0)
```

### Transport Layers
| Transport | Latency | Use Case |
|-----------|---------|----------|
| IPC | < 0.1ms | Local processes |
| UDP Multicast | 1-5ms | LAN discovery |
| WebSocket | 10-50ms | Remote/internet |
| Message Queue | Variable | Reliable delivery |

### Security
- **Encryption**: AES-128/256-GCM
- **Key Exchange**: ECDH-384
- **Authentication**: Tokens, API keys
- **Rate Limiting**: Adaptive throttling

### AI Integration (MCP)

Control hardware directly from Claude Desktop or any MCP-compatible AI:

```bash
# Start MCP server for Claude Desktop
python -m regennexus.mcp_server
```

Configure in `claude_desktop_config.json`:
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

Now ask Claude:
- "Move the robot arm to pick position"
- "Turn on GPIO pin 17"
- "What's the temperature sensor reading?"

### LLM Bridge (Ollama, LM Studio)

Connect local LLMs to hardware:

```python
from regennexus.bridges import LLMBridge, LLMConfig

llm = LLMBridge(LLMConfig(provider="ollama", model="llama3"))
response = await llm.chat("Turn on the lights")
```

### Mesh Networking

Auto-discovery across devices on the network:

```python
from regennexus.core import MeshNetwork, MeshConfig

mesh = MeshNetwork(MeshConfig(
    node_id="controller",
    capabilities=["command"]
))
await mesh.start()

# Devices auto-discovered
for peer in mesh.get_peers():
    print(f"Found: {peer.node_id} ({peer.capabilities})")
```

## Interactive Demos

Try RegenNexus in Google Colab:

- [Basic Demo](https://colab.research.google.com/github/ReGenNow/ReGenNexus/blob/main/examples/binder/colab_basic_demo.ipynb) - Core communication
- [Security Demo](https://colab.research.google.com/github/ReGenNow/ReGenNexus/blob/main/examples/binder/colab_security_demo.ipynb) - Encryption & auth

## CLI Usage

```bash
# Start server
regen server --host 0.0.0.0 --port 8080

# Run example
regen run examples/robotic_arms/arm_demo.py

# Version info
regen version
```

## Optional Dependencies

```bash
pip install regennexus[api]        # FastAPI server
pip install regennexus[mqtt]       # MQTT support
pip install regennexus[robotics]   # Robotic arms
pip install regennexus[arduino]    # Arduino support
pip install regennexus[dev]        # Development tools
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [MCP Integration](docs/mcp_integration.md)
- [Device Integration](docs/device_integration.md)
- [Robotic Arms Guide](docs/robotic_arms.md)
- [Security Guide](docs/security.md)
- [API Reference](docs/api_reference.md)
- [ROS Integration](docs/ros_integration.md)

## Examples

```
examples/
├── simple_connection/    # Basic protocol usage
├── mcp_integration/      # Claude Desktop & LLM demos
├── mesh_network/         # Device auto-discovery
├── llm_integration/      # Ollama/LM Studio demos
├── robotic_arms/         # Amber B1 & Lucid One demos
├── ros_integration/      # ROS 2 bridge examples
├── security/             # Encryption & auth
└── binder/               # Jupyter notebooks
```

## Docker

```bash
docker-compose up
```

See [Docker Deployment](docs/docker_deployment.md) for details.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**RegenNexus UAP** - Connect Everything, Securely.

Copyright (c) 2024-2025 ReGen Designs LLC

