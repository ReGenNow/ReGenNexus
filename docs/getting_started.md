# Getting Started with RegenNexus UAP

RegenNexus is a **Universal Adapter Protocol** - an open-source framework for fast, secure communication between devices, robots, apps, and AI agents. Think of it as "MCP for hardware" - connect anything to anything with minimal latency and maximum security.

## Installation

```bash
pip install regennexus
```

Or install from source:
```bash
git clone https://github.com/ReGenNow/ReGenNexus.git
cd ReGenNexus
pip install -e .
```

## Quick Start

### 1. Basic Communication (< 5 minutes)

Connect two entities and send a message:

```python
import asyncio
from regennexus import RegenNexusProtocol

async def main():
    # Create protocol instance
    protocol = RegenNexusProtocol()
    await protocol.initialize()

    # Register two entities
    await protocol.registry.register_entity(
        entity_id="sensor_01",
        entity_type="device",
        capabilities=["temperature", "humidity"]
    )

    await protocol.registry.register_entity(
        entity_id="controller_01",
        entity_type="controller",
        capabilities=["command", "monitor"]
    )

    # Send a message
    await protocol.send_message(
        sender="controller_01",
        recipient="sensor_01",
        intent="read",
        payload={"sensors": ["temperature", "humidity"]}
    )

    print("Message sent!")
    await protocol.shutdown()

asyncio.run(main())
```

### 2. Using Device Plugins

Control real hardware with mock mode for development:

```python
import asyncio
from regennexus.plugins import get_raspberry_pi_plugin

async def main():
    # Get the plugin class
    RaspberryPiPlugin = get_raspberry_pi_plugin()

    # Create plugin in mock mode (no hardware needed)
    rpi = RaspberryPiPlugin(
        entity_id="rpi_001",
        mock_mode=True  # Set False for real hardware
    )

    # Initialize
    await rpi.initialize()

    # Control GPIO
    result = await rpi.execute_command("gpio.write", {
        "pin": 18,
        "value": 1
    })
    print(f"GPIO write: {result}")

    # Read GPIO
    result = await rpi.execute_command("gpio.read", {"pin": 17})
    print(f"GPIO read: {result}")

    await rpi.shutdown()

asyncio.run(main())
```

### 3. Control Robotic Arms

```python
import asyncio
from regennexus.plugins import get_amber_b1_plugin, get_lucid_one_plugin

async def main():
    # Amber B1 arm
    AmberB1 = get_amber_b1_plugin()
    arm = AmberB1(entity_id="arm_001", mock_mode=True)
    await arm.initialize()

    # Move to position
    await arm.move_to([0, 45, -30, 0, 90, 0, 0], duration=2.0)

    # Control gripper
    await arm.open_gripper()
    await asyncio.sleep(1)
    await arm.close_gripper(force=15.0)

    # Home the arm
    await arm.home()

    await arm.shutdown()

asyncio.run(main())
```

## What You Can Do

### Device Communication
- **Raspberry Pi**: GPIO, PWM, camera, sensors
- **Arduino**: Digital/analog I/O, serial commands
- **NVIDIA Jetson**: GPU, CUDA, camera, inference
- **IoT Devices**: MQTT, HTTP, CoAP protocols

### Robotic Arms
- **Amber B1**: 7-DOF control, gripper, trajectory
- **Lucid One**: Cartesian control, force sensing, teach mode

### Transport Layers
- **IPC**: < 0.1ms latency (local processes)
- **UDP Multicast**: 1-5ms (LAN discovery)
- **WebSocket**: 10-50ms (remote/internet)
- **Message Queue**: Reliable with persistence

### Security Features
- **Encryption**: AES-128/256-GCM
- **Key Exchange**: ECDH-384
- **Authentication**: Tokens, API keys
- **Rate Limiting**: Adaptive throttling

## Configuration

Create `regennexus-config.yaml`:

```yaml
version: "0.2"

security:
  encryption_enabled: true
  algorithm: "AES-256-GCM"
  authentication:
    method: "token"
    token_expiry: 3600

communication:
  transport: "auto"
  default_timeout: 30.0
  retry_attempts: 3

registry:
  discovery_enabled: true
  heartbeat_interval: 30

logging:
  level: "INFO"
  file: "regennexus.log"
```

Load configuration:

```python
from regennexus.config import load_config

config = load_config("regennexus-config.yaml")
protocol = RegenNexusProtocol(config=config)
```

## CLI Usage

```bash
# Start the registry server
regen server --host 0.0.0.0 --port 8080

# Run an example
regen run examples/simple_connection/basic_protocol_example.py

# Check version
regen version
```

## Examples

### Sensor Network
```python
# Multiple sensors reporting to a controller
sensors = []
for i in range(5):
    sensor = TemperatureSensor(entity_id=f"temp_{i}", mock_mode=True)
    await sensor.initialize()
    sensors.append(sensor)

# Controller aggregates data
controller = DataController(entity_id="aggregator")
for sensor in sensors:
    data = await sensor.read()
    controller.process(data)
```

### Pick and Place Robot
```python
async def pick_and_place(arm, pick_pos, place_pos):
    """Simple pick and place operation."""
    # Move above pick position
    await arm.move_to_pose(pick_pos[0], pick_pos[1], pick_pos[2] + 50)

    # Open gripper
    await arm.open_gripper()

    # Move down to pick
    await arm.move_to_pose(*pick_pos)

    # Close gripper
    await arm.close_gripper(force=20.0)

    # Lift
    await arm.move_to_pose(pick_pos[0], pick_pos[1], pick_pos[2] + 50)

    # Move to place position
    await arm.move_to_pose(place_pos[0], place_pos[1], place_pos[2] + 50)

    # Lower and release
    await arm.move_to_pose(*place_pos)
    await arm.open_gripper()
```

### Secure Communication
```python
from regennexus.security import AESEncryptor, TokenAuth

# Create encryptor
encryptor = AESEncryptor()
key = encryptor.generate_key()

# Encrypt message
encrypted = encryptor.encrypt(b"secret data", key)

# Create authenticator
auth = TokenAuth(secret_key="your-secret-key")
token = auth.generate_token("user_001", expires_in=3600)

# Validate
result = auth.validate_token(token)
print(f"Valid: {result.valid}, User: {result.entity_id}")
```

## Next Steps

- [API Reference](api_reference.md) - Complete API documentation
- [Device Integration](device_integration.md) - Hardware guides
- [Security Guide](security.md) - Security best practices
- [ROS Integration](ros_integration.md) - Robot Operating System
- [Examples](../examples/) - More code examples

## Support

- GitHub Issues: https://github.com/ReGenNow/ReGenNexus/issues
- Documentation: https://regennexus.readthedocs.io

---

**RegenNexus UAP** - Connect Everything, Securely.

Copyright (c) 2024-2025 ReGen Designs LLC | MIT License
