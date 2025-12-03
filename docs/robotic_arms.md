# Robotic Arm Integration Guide

RegenNexus UAP provides native support for professional robotic arms from AMBER Robotics. Both the Amber B1 and Lucid One 7-DOF arms are fully supported with gripper control.

## Plugin Locations

```
regennexus/plugins/
├── base.py          # Base plugin class (DevicePlugin, MockDeviceMixin)
├── amber_b1.py      # Amber B1 7-DOF arm
├── lucid_one.py     # Lucid One 7-DOF arm
├── raspberry_pi.py  # Raspberry Pi GPIO/camera
├── arduino.py       # Arduino serial/I/O
├── jetson.py        # NVIDIA Jetson GPU/camera
└── iot.py           # MQTT/HTTP IoT devices
```

## Supported Arms

| Arm | DOF | Payload | Reach | Protocol | Features |
|-----|-----|---------|-------|----------|----------|
| Amber B1 | 7 | 5kg | 700mm | UDP | Joint control, gripper |
| Lucid One | 7 | 3kg | 850mm | UDP | Cartesian, force sensing, teach mode |

## Quick Start

### Amber B1

```python
import asyncio
from regennexus.plugins import get_amber_b1_plugin

async def main():
    # Get plugin class
    AmberB1 = get_amber_b1_plugin()

    # Create arm (mock_mode=True for development)
    arm = AmberB1(
        entity_id="amber_arm",
        mock_mode=True  # Set False for real hardware
    )

    await arm.initialize()
    print(f"Arm state: {arm.get_state()}")

    # Move joints
    positions = [0, 45, -30, 0, 90, 0, 0]  # 7 joint angles in degrees
    await arm.move_to(positions, duration=2.0)

    # Control gripper
    await arm.open_gripper()
    await asyncio.sleep(1)
    await arm.close_gripper(force=15.0)

    # Return home
    await arm.home()

    await arm.shutdown()

asyncio.run(main())
```

### Lucid One

```python
import asyncio
from regennexus.plugins import get_lucid_one_plugin

async def main():
    # Get plugin class
    LucidOne = get_lucid_one_plugin()

    # Create arm
    arm = LucidOne(
        entity_id="lucid_arm",
        mock_mode=True
    )

    await arm.initialize()

    # Move to Cartesian pose [x, y, z, rx, ry, rz] in mm and degrees
    await arm.move_to_pose(
        x=400, y=100, z=300,
        rx=0, ry=180, rz=45,
        velocity=0.1  # m/s
    )

    # Read force/torque sensor
    result = await arm.execute_command("get_force_torque", {})
    print(f"Force: Fx={result['fx']:.2f}, Fy={result['fy']:.2f}, Fz={result['fz']:.2f}")

    # Enable teach mode (gravity compensation)
    await arm.execute_command("teach_mode", {"enable": True})

    await arm.shutdown()

asyncio.run(main())
```

## Configuration

### Amber B1 Configuration

```python
from regennexus.plugins.amber_b1 import AmberB1Plugin, ArmConfig, JointConfig

# Custom configuration
config = ArmConfig(
    base_ip="10.0.0.10",      # First joint IP
    port=25001,               # UDP port
    num_joints=7,
    has_gripper=True,
    gripper_max_width=85.0,   # mm
)

arm = AmberB1Plugin(
    entity_id="custom_arm",
    config=config,
    mock_mode=False
)
```

### Lucid One Configuration

```python
from regennexus.plugins.lucid_one import LucidOnePlugin, LucidConfig

config = LucidConfig(
    ip_address="10.0.0.100",
    port=25001,
    num_joints=7,
    has_gripper=True,
    gripper_max_width=85.0,
    max_velocity=1.0,         # m/s
    max_acceleration=2.0,     # m/s²
    payload=3.0,              # kg
    reach=850.0,              # mm
)

arm = LucidOnePlugin(
    entity_id="lucid_custom",
    config=config,
    mock_mode=False
)
```

## Commands Reference

### Joint Control

```python
# Move single joint
await arm.execute_command("move_joint", {
    "joint_id": 3,      # Joint 1-7
    "position": 45.0,   # degrees
    "velocity": 30.0    # deg/s (optional)
})

# Move all joints
await arm.execute_command("move_joints", {
    "positions": [0, 45, -30, 0, 90, 0, 0],
    "duration": 2.0     # seconds
})

# Get joint positions
result = await arm.execute_command("get_joint_positions", {})
print(f"Positions: {result['positions']}")
print(f"Velocities: {result['velocities']}")
print(f"Torques: {result['torques']}")
```

### Cartesian Control (Lucid One)

```python
# Move to Cartesian pose
await arm.execute_command("move_cartesian", {
    "pose": [400, 100, 300, 0, 180, 45],  # x, y, z, rx, ry, rz
    "velocity": 0.1  # m/s
})

# Linear motion (same as move_cartesian but explicitly linear)
await arm.execute_command("move_linear", {
    "pose": [450, 100, 250, 0, 180, 45],
    "velocity": 0.05
})

# Get current pose
result = await arm.execute_command("get_cartesian_pose", {})
print(f"Position: x={result['x']}, y={result['y']}, z={result['z']}")
print(f"Orientation: rx={result['rx']}, ry={result['ry']}, rz={result['rz']}")
```

### Gripper Control

```python
# Open gripper fully
await arm.execute_command("gripper.open", {})

# Close gripper with force
await arm.execute_command("gripper.close", {
    "force": 20.0  # Newtons
})

# Set specific width
await arm.execute_command("gripper.set_position", {
    "position": 40.0  # mm
})

# Get gripper status
result = await arm.execute_command("gripper.get_position", {})
print(f"Width: {result['position']}mm, Max: {result['max_width']}mm")
```

### Motion Control

```python
# Home the arm (all joints to 0)
await arm.execute_command("home", {})

# Stop current motion
await arm.execute_command("stop", {})

# Emergency stop (immediate halt)
await arm.execute_command("emergency_stop", {})

# Get arm state
result = await arm.execute_command("get_state", {})
print(f"State: {result['state']}")
print(f"Mode: {result['control_mode']}")
```

### Trajectory Recording (Lucid One)

```python
# Start recording
await arm.execute_command("record_trajectory", {"action": "start"})

# Manually move arm in teach mode, then add waypoints
await arm.execute_command("teach_mode", {"enable": True})

# Add current position as waypoint
await arm.execute_command("record_trajectory", {"action": "add_point"})

# Stop recording
await arm.execute_command("record_trajectory", {"action": "stop"})

# Play back trajectory
await arm.execute_command("play_trajectory", {
    "speed": 0.5  # 50% speed
})

# Clear trajectory
await arm.execute_command("record_trajectory", {"action": "clear"})
```

## Application Examples

### Pick and Place

```python
async def pick_and_place(arm, pick_pos, place_pos, height=50):
    """
    Pick object from one location and place at another.

    Args:
        arm: Arm plugin instance
        pick_pos: [x, y, z] pick position in mm
        place_pos: [x, y, z] place position in mm
        height: Approach height in mm
    """
    # Approach pick position from above
    await arm.move_to_pose(
        pick_pos[0], pick_pos[1], pick_pos[2] + height,
        rx=0, ry=180, rz=0
    )

    # Open gripper
    await arm.open_gripper()

    # Descend to pick
    await arm.move_to_pose(
        pick_pos[0], pick_pos[1], pick_pos[2],
        rx=0, ry=180, rz=0,
        velocity=0.05
    )

    # Grasp object
    await arm.close_gripper(force=15.0)
    await asyncio.sleep(0.5)

    # Lift object
    await arm.move_to_pose(
        pick_pos[0], pick_pos[1], pick_pos[2] + height,
        rx=0, ry=180, rz=0
    )

    # Move to place position
    await arm.move_to_pose(
        place_pos[0], place_pos[1], place_pos[2] + height,
        rx=0, ry=180, rz=0
    )

    # Descend to place
    await arm.move_to_pose(
        place_pos[0], place_pos[1], place_pos[2],
        rx=0, ry=180, rz=0,
        velocity=0.05
    )

    # Release object
    await arm.open_gripper()

    # Retract
    await arm.move_to_pose(
        place_pos[0], place_pos[1], place_pos[2] + height,
        rx=0, ry=180, rz=0
    )

# Usage
await pick_and_place(
    arm,
    pick_pos=[400, -100, 50],
    place_pos=[400, 100, 50]
)
```

### Palletizing Pattern

```python
async def palletize(arm, start_pos, rows, cols, spacing):
    """
    Create a palletizing pattern.

    Args:
        arm: Arm plugin instance
        start_pos: [x, y, z] starting corner
        rows: Number of rows
        cols: Number of columns
        spacing: Distance between positions in mm
    """
    positions = []

    for row in range(rows):
        for col in range(cols):
            x = start_pos[0] + col * spacing
            y = start_pos[1] + row * spacing
            z = start_pos[2]
            positions.append([x, y, z])

    for i, pos in enumerate(positions):
        print(f"Moving to position {i+1}/{len(positions)}")
        await arm.move_to_pose(
            pos[0], pos[1], pos[2],
            rx=0, ry=180, rz=0
        )
        await asyncio.sleep(0.5)

    return positions

# Create 3x4 grid with 80mm spacing
await palletize(
    arm,
    start_pos=[300, -150, 100],
    rows=3,
    cols=4,
    spacing=80
)
```

### Force-Controlled Insertion

```python
async def force_insert(arm, target_pos, max_force=10.0):
    """
    Insert object using force feedback.

    Args:
        arm: Arm plugin instance (Lucid One with force sensing)
        target_pos: Target position
        max_force: Maximum insertion force in N
    """
    current_z = target_pos[2] + 50  # Start above

    while True:
        # Move down slowly
        await arm.move_to_pose(
            target_pos[0], target_pos[1], current_z,
            rx=0, ry=180, rz=0,
            velocity=0.01
        )

        # Check force
        result = await arm.execute_command("get_force_torque", {})
        fz = abs(result['fz'])

        print(f"Z: {current_z:.1f}mm, Force: {fz:.2f}N")

        if fz >= max_force:
            print("Target force reached!")
            break

        if current_z <= target_pos[2]:
            print("Target depth reached!")
            break

        current_z -= 1  # Descend 1mm

    return current_z
```

## Events

Listen for arm events:

```python
async def on_emergency_stop(event_data):
    print(f"EMERGENCY STOP: {event_data['entity_id']}")
    print(f"Position at stop: {event_data['positions']}")

# Register event listener
arm.register_event_listener("arm.emergency_stop", on_emergency_stop)
```

## Safety Guidelines

1. **Always test in mock mode first** - Verify logic before running on real hardware
2. **Set appropriate velocity limits** - Start slow, increase gradually
3. **Implement emergency stop handling** - Have a way to stop immediately
4. **Check workspace limits** - Ensure positions are reachable
5. **Monitor force feedback** - Use force limits to prevent damage
6. **Keep humans clear** - Establish safety zones around the robot

## Troubleshooting

### Connection Issues
```python
# Check if arm is connected
result = await arm.execute_command("get_state", {})
if result['state'] == 'disconnected':
    print("Arm not connected. Check network settings.")
```

### Motion Errors
```python
# Arm in error state
if arm.get_state() == ArmState.ERROR:
    # Try to recover
    await arm.execute_command("stop", {})
    await asyncio.sleep(1)
    await arm.home()
```

### Joint Limits
```python
# Check limits before moving
from regennexus.plugins.lucid_one import LucidConfig

config = LucidConfig()
for i, (min_val, max_val) in enumerate(config.joint_limits):
    print(f"Joint {i+1}: {min_val}° to {max_val}°")
```

---

For more information, see the [API Reference](api_reference.md) or check the [examples](../examples/ros_integration/).
