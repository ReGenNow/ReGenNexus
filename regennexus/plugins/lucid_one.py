"""
RegenNexus UAP - Lucid One Robotic Arm Plugin

AMBER Robotics Lucid One 7-DOF robotic arm integration with UDP control.
Supports joint control, gripper operations, and Cartesian positioning.
Includes mock mode for development without hardware.

Specifications:
- 7 Degrees of Freedom (DOF)
- UDP communication protocol (port 25001)
- Payload capacity: 3kg
- Reach: 850mm
- Repeatability: ±0.1mm
- Built-in force/torque sensing
- Cartesian space control

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import logging
import math
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from regennexus.plugins.base import DevicePlugin, MockDeviceMixin

logger = logging.getLogger(__name__)


class ArmState(Enum):
    """Arm operational state."""
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MOVING = "moving"
    TEACHING = "teaching"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class ControlMode(Enum):
    """Control mode."""
    JOINT = "joint"
    CARTESIAN = "cartesian"
    IMPEDANCE = "impedance"


@dataclass
class CartesianPose:
    """Cartesian pose (position + orientation)."""
    x: float = 0.0  # mm
    y: float = 0.0  # mm
    z: float = 0.0  # mm
    rx: float = 0.0  # degrees (roll)
    ry: float = 0.0  # degrees (pitch)
    rz: float = 0.0  # degrees (yaw)

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_list(cls, values: List[float]) -> "CartesianPose":
        return cls(*values[:6])


@dataclass
class LucidConfig:
    """Configuration for Lucid One arm."""
    ip_address: str = "10.0.0.100"
    port: int = 25001
    num_joints: int = 7
    has_gripper: bool = True
    gripper_max_width: float = 85.0  # mm
    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 2.0  # m/s²
    payload: float = 3.0  # kg
    reach: float = 850.0  # mm

    # Joint limits (degrees)
    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-170, 170),   # J1
        (-120, 120),   # J2
        (-170, 170),   # J3
        (-120, 120),   # J4
        (-170, 170),   # J5
        (-120, 120),   # J6
        (-270, 270),   # J7
    ])


class LucidOnePlugin(DevicePlugin, MockDeviceMixin):
    """
    Lucid One 7-DOF robotic arm plugin for RegenNexus.

    Supports:
    - Joint space control
    - Cartesian space control
    - Force/torque feedback
    - Gripper control
    - Teach mode
    - Trajectory recording/playback
    - Mock mode for development
    """

    def __init__(
        self,
        entity_id: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False,
        config: Optional[LucidConfig] = None
    ):
        """
        Initialize Lucid One plugin.

        Args:
            entity_id: Unique identifier
            protocol: Protocol instance
            mock_mode: If True, simulate without hardware
            config: Arm configuration (uses defaults if None)
        """
        DevicePlugin.__init__(self, entity_id, "lucid_one", protocol, mock_mode)
        MockDeviceMixin.__init__(self)

        self.config = config or LucidConfig()
        self.state = ArmState.DISCONNECTED
        self.control_mode = ControlMode.JOINT

        # Joint states
        self.joint_positions: List[float] = [0.0] * self.config.num_joints
        self.joint_velocities: List[float] = [0.0] * self.config.num_joints
        self.joint_torques: List[float] = [0.0] * self.config.num_joints

        # Cartesian pose
        self.cartesian_pose = CartesianPose()

        # Force/torque sensing
        self.force_torque: List[float] = [0.0] * 6  # Fx, Fy, Fz, Tx, Ty, Tz

        # Gripper state
        self.gripper_position: float = 0.0
        self.gripper_force: float = 0.0

        # UDP socket
        self._socket: Optional[socket.socket] = None

        # Motion tracking
        self._motion_task: Optional[asyncio.Task] = None

        # Trajectory recording
        self._recorded_trajectory: List[Dict] = []
        self._recording = False

    async def _device_init(self) -> bool:
        """Initialize Lucid One hardware."""
        try:
            if self.mock_mode:
                logger.info("Lucid One arm in mock mode")
                self.state = ArmState.IDLE
                # Initialize mock pose
                self.cartesian_pose = CartesianPose(x=400, y=0, z=300, rx=0, ry=180, rz=0)
            else:
                # Create UDP socket
                try:
                    self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self._socket.setblocking(False)
                    self._socket.settimeout(1.0)

                    # Test connection
                    if await self._ping_arm():
                        self.state = ArmState.IDLE
                        logger.info(f"Lucid One arm connected at {self.config.ip_address}")
                    else:
                        logger.warning("Lucid One arm not responding")
                        self.state = ArmState.DISCONNECTED

                except Exception as e:
                    logger.error(f"Failed to create socket: {e}")
                    self.state = ArmState.DISCONNECTED

            # Add capabilities
            self.capabilities.extend([
                "arm.move_joints",
                "arm.move_cartesian",
                "arm.move_linear",
                "arm.get_joint_positions",
                "arm.get_cartesian_pose",
                "arm.get_force_torque",
                "arm.get_state",
                "arm.home",
                "arm.stop",
                "arm.emergency_stop",
                "arm.teach_mode",
                "arm.record_trajectory",
                "arm.play_trajectory",
            ])

            if self.config.has_gripper:
                self.capabilities.extend([
                    "gripper.open",
                    "gripper.close",
                    "gripper.set_position",
                    "gripper.get_position",
                ])

            # Register command handlers
            self.register_command_handler("move_joints", self._handle_move_joints)
            self.register_command_handler("move_cartesian", self._handle_move_cartesian)
            self.register_command_handler("move_linear", self._handle_move_linear)
            self.register_command_handler("get_joint_positions", self._handle_get_joints)
            self.register_command_handler("get_cartesian_pose", self._handle_get_cartesian)
            self.register_command_handler("get_force_torque", self._handle_get_force_torque)
            self.register_command_handler("get_state", self._handle_get_state)
            self.register_command_handler("home", self._handle_home)
            self.register_command_handler("stop", self._handle_stop)
            self.register_command_handler("emergency_stop", self._handle_emergency_stop)
            self.register_command_handler("teach_mode", self._handle_teach_mode)
            self.register_command_handler("record_trajectory", self._handle_record)
            self.register_command_handler("play_trajectory", self._handle_playback)
            self.register_command_handler("gripper.open", self._handle_gripper_open)
            self.register_command_handler("gripper.close", self._handle_gripper_close)
            self.register_command_handler("gripper.set_position", self._handle_gripper_set)
            self.register_command_handler("gripper.get_position", self._handle_gripper_get)

            # Update metadata
            self.metadata.update({
                "arm_type": "Lucid One",
                "manufacturer": "AMBER Robotics",
                "num_joints": self.config.num_joints,
                "has_gripper": self.config.has_gripper,
                "payload_kg": self.config.payload,
                "reach_mm": self.config.reach,
                "state": self.state.value,
                "control_mode": self.control_mode.value,
            })

            return True

        except Exception as e:
            logger.error(f"Lucid One init error: {e}")
            return False

    async def _device_shutdown(self) -> None:
        """Clean up Lucid One resources."""
        # Stop any motion
        if self._motion_task:
            self._motion_task.cancel()
            try:
                await self._motion_task
            except asyncio.CancelledError:
                pass
            self._motion_task = None

        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        self.state = ArmState.DISCONNECTED

    async def _ping_arm(self) -> bool:
        """Ping arm to check connection."""
        if self.mock_mode:
            return True

        try:
            # Send ping packet
            packet = struct.pack("<BB", 0x00, 0x01)  # Type: ping, ID: 1
            self._socket.sendto(packet, (self.config.ip_address, self.config.port))

            # Wait for response
            try:
                data, addr = self._socket.recvfrom(1024)
                return len(data) > 0
            except socket.timeout:
                return False

        except Exception:
            return False

    def _send_command(self, cmd_type: int, data: bytes) -> bool:
        """Send command to arm via UDP."""
        if self.mock_mode:
            return True

        if not self._socket:
            return False

        try:
            packet = struct.pack("<BB", cmd_type, len(data)) + data
            self._socket.sendto(packet, (self.config.ip_address, self.config.port))
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    async def _execute_joint_motion(
        self,
        target_positions: List[float],
        velocity: float = 0.5,
        duration: Optional[float] = None
    ) -> bool:
        """Execute joint space motion."""
        if len(target_positions) != self.config.num_joints:
            return False

        self.state = ArmState.MOVING
        start_positions = self.joint_positions.copy()

        # Calculate duration if not specified
        if duration is None:
            max_distance = max(
                abs(target_positions[i] - start_positions[i])
                for i in range(self.config.num_joints)
            )
            duration = max(1.0, max_distance / (velocity * 60))  # velocity in deg/s

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                t = min(elapsed / duration, 1.0)

                # S-curve interpolation for smooth motion
                t = t * t * t * (t * (t * 6 - 15) + 10)

                for i in range(self.config.num_joints):
                    self.joint_positions[i] = (
                        start_positions[i] + t * (target_positions[i] - start_positions[i])
                    )

                if not self.mock_mode:
                    # Send position command
                    data = struct.pack(f"<{self.config.num_joints}f", *self.joint_positions)
                    self._send_command(0x10, data)  # Joint position command

                # Update Cartesian pose (simplified forward kinematics mock)
                if self.mock_mode:
                    self._update_mock_cartesian()

                await asyncio.sleep(0.01)  # 100Hz update rate

            # Set final positions
            for i, pos in enumerate(target_positions):
                self.joint_positions[i] = pos

            self.state = ArmState.IDLE
            return True

        except asyncio.CancelledError:
            self.state = ArmState.IDLE
            raise
        except Exception as e:
            logger.error(f"Joint motion error: {e}")
            self.state = ArmState.ERROR
            return False

    async def _execute_linear_motion(
        self,
        target_pose: CartesianPose,
        velocity: float = 0.1  # m/s
    ) -> bool:
        """Execute linear Cartesian motion."""
        self.state = ArmState.MOVING
        start_pose = CartesianPose(
            self.cartesian_pose.x, self.cartesian_pose.y, self.cartesian_pose.z,
            self.cartesian_pose.rx, self.cartesian_pose.ry, self.cartesian_pose.rz
        )

        # Calculate distance and duration
        distance = math.sqrt(
            (target_pose.x - start_pose.x) ** 2 +
            (target_pose.y - start_pose.y) ** 2 +
            (target_pose.z - start_pose.z) ** 2
        )
        duration = max(0.5, distance / (velocity * 1000))  # velocity in m/s, distance in mm

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                t = min(elapsed / duration, 1.0)

                # Linear interpolation with smoothing
                t = t * t * (3 - 2 * t)

                self.cartesian_pose.x = start_pose.x + t * (target_pose.x - start_pose.x)
                self.cartesian_pose.y = start_pose.y + t * (target_pose.y - start_pose.y)
                self.cartesian_pose.z = start_pose.z + t * (target_pose.z - start_pose.z)
                self.cartesian_pose.rx = start_pose.rx + t * (target_pose.rx - start_pose.rx)
                self.cartesian_pose.ry = start_pose.ry + t * (target_pose.ry - start_pose.ry)
                self.cartesian_pose.rz = start_pose.rz + t * (target_pose.rz - start_pose.rz)

                if not self.mock_mode:
                    data = struct.pack("<6f", *self.cartesian_pose.to_list())
                    self._send_command(0x20, data)  # Cartesian position command

                await asyncio.sleep(0.01)

            # Set final pose
            self.cartesian_pose = target_pose
            self.state = ArmState.IDLE
            return True

        except asyncio.CancelledError:
            self.state = ArmState.IDLE
            raise
        except Exception as e:
            logger.error(f"Linear motion error: {e}")
            self.state = ArmState.ERROR
            return False

    def _update_mock_cartesian(self):
        """Update mock Cartesian pose from joint positions (simplified)."""
        # Simplified forward kinematics for mock mode
        # Real implementation would use actual kinematic model
        j1, j2, j3, j4, j5, j6, j7 = self.joint_positions

        # Very simplified positioning
        self.cartesian_pose.x = 400 + 100 * math.sin(math.radians(j1))
        self.cartesian_pose.y = 100 * math.cos(math.radians(j1))
        self.cartesian_pose.z = 300 + 50 * math.sin(math.radians(j2))
        self.cartesian_pose.rz = j1
        self.cartesian_pose.ry = 180 + j5
        self.cartesian_pose.rx = j7

    async def _handle_move_joints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move joints command."""
        positions = params.get("positions")
        velocity = params.get("velocity", 0.5)
        duration = params.get("duration")

        if not positions or len(positions) != self.config.num_joints:
            return {
                "success": False,
                "error": f"Expected {self.config.num_joints} joint positions",
            }

        # Clamp to joint limits
        clamped = []
        for i, pos in enumerate(positions):
            min_val, max_val = self.config.joint_limits[i]
            clamped.append(max(min_val, min(max_val, pos)))

        try:
            await self._execute_joint_motion(clamped, velocity, duration)
            return {
                "success": True,
                "positions": clamped,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_move_cartesian(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move to Cartesian pose command."""
        pose = params.get("pose")
        velocity = params.get("velocity", 0.1)

        if not pose or len(pose) != 6:
            return {"success": False, "error": "Expected pose [x, y, z, rx, ry, rz]"}

        try:
            target = CartesianPose.from_list(pose)
            await self._execute_linear_motion(target, velocity)
            return {
                "success": True,
                "pose": target.to_list(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_move_linear(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle linear motion command."""
        # Same as move_cartesian but explicitly linear
        return await self._handle_move_cartesian(params)

    async def _handle_get_joints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get joint positions command."""
        return {
            "success": True,
            "positions": self.joint_positions.copy(),
            "velocities": self.joint_velocities.copy(),
            "torques": self.joint_torques.copy(),
        }

    async def _handle_get_cartesian(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get Cartesian pose command."""
        return {
            "success": True,
            "pose": self.cartesian_pose.to_list(),
            "x": self.cartesian_pose.x,
            "y": self.cartesian_pose.y,
            "z": self.cartesian_pose.z,
            "rx": self.cartesian_pose.rx,
            "ry": self.cartesian_pose.ry,
            "rz": self.cartesian_pose.rz,
        }

    async def _handle_get_force_torque(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get force/torque sensor data command."""
        if self.mock_mode:
            # Simulate some force readings
            import random
            self.force_torque = [random.uniform(-0.5, 0.5) for _ in range(6)]

        return {
            "success": True,
            "force_torque": self.force_torque.copy(),
            "fx": self.force_torque[0],
            "fy": self.force_torque[1],
            "fz": self.force_torque[2],
            "tx": self.force_torque[3],
            "ty": self.force_torque[4],
            "tz": self.force_torque[5],
        }

    async def _handle_get_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get arm state command."""
        return {
            "success": True,
            "state": self.state.value,
            "control_mode": self.control_mode.value,
            "joint_positions": self.joint_positions.copy(),
            "cartesian_pose": self.cartesian_pose.to_list(),
            "gripper_position": self.gripper_position,
            "recording": self._recording,
        }

    async def _handle_home(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle home command."""
        try:
            home_positions = [0.0] * self.config.num_joints
            await self._execute_joint_motion(home_positions, 0.3, 5.0)
            return {"success": True, "message": "Arm homed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop command."""
        if self._motion_task:
            self._motion_task.cancel()
            try:
                await self._motion_task
            except asyncio.CancelledError:
                pass
            self._motion_task = None

        self.state = ArmState.IDLE
        return {"success": True, "message": "Motion stopped"}

    async def _handle_emergency_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency stop command."""
        if self._motion_task:
            self._motion_task.cancel()
            try:
                await self._motion_task
            except asyncio.CancelledError:
                pass
            self._motion_task = None

        self.state = ArmState.EMERGENCY_STOP
        self._recording = False

        await self.emit_event("arm.emergency_stop", {
            "entity_id": self.entity_id,
            "positions": self.joint_positions.copy(),
            "pose": self.cartesian_pose.to_list(),
        })

        return {"success": True, "message": "Emergency stop activated"}

    async def _handle_teach_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle teach mode command."""
        enable = params.get("enable", True)

        if enable:
            self.state = ArmState.TEACHING
            self.control_mode = ControlMode.IMPEDANCE
        else:
            self.state = ArmState.IDLE
            self.control_mode = ControlMode.JOINT

        return {
            "success": True,
            "teach_mode": enable,
            "message": "Teach mode " + ("enabled" if enable else "disabled"),
        }

    async def _handle_record(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trajectory recording command."""
        action = params.get("action", "start")

        if action == "start":
            self._recorded_trajectory = []
            self._recording = True
            return {"success": True, "message": "Recording started"}

        elif action == "stop":
            self._recording = False
            return {
                "success": True,
                "message": "Recording stopped",
                "points": len(self._recorded_trajectory),
            }

        elif action == "add_point":
            if self._recording:
                self._recorded_trajectory.append({
                    "joints": self.joint_positions.copy(),
                    "pose": self.cartesian_pose.to_list(),
                    "gripper": self.gripper_position,
                    "timestamp": time.time(),
                })
                return {
                    "success": True,
                    "points": len(self._recorded_trajectory),
                }
            return {"success": False, "error": "Not recording"}

        elif action == "clear":
            self._recorded_trajectory = []
            self._recording = False
            return {"success": True, "message": "Trajectory cleared"}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _handle_playback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trajectory playback command."""
        if not self._recorded_trajectory:
            return {"success": False, "error": "No trajectory recorded"}

        speed_factor = params.get("speed", 1.0)

        try:
            self.state = ArmState.MOVING

            for i, point in enumerate(self._recorded_trajectory):
                await self._execute_joint_motion(
                    point["joints"],
                    velocity=0.5 * speed_factor,
                    duration=1.0 / speed_factor
                )

                if self.config.has_gripper:
                    self.gripper_position = point.get("gripper", self.gripper_position)

            self.state = ArmState.IDLE
            return {
                "success": True,
                "message": "Playback complete",
                "points_played": len(self._recorded_trajectory),
            }

        except Exception as e:
            self.state = ArmState.ERROR
            return {"success": False, "error": str(e)}

    async def _handle_gripper_open(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper open command."""
        if not self.config.has_gripper:
            return {"success": False, "error": "No gripper configured"}

        if self.mock_mode:
            self.gripper_position = self.config.gripper_max_width

        return {
            "success": True,
            "position": self.gripper_position,
            "state": "open",
        }

    async def _handle_gripper_close(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper close command."""
        if not self.config.has_gripper:
            return {"success": False, "error": "No gripper configured"}

        force = params.get("force", 10.0)

        if self.mock_mode:
            self.gripper_position = 0.0
            self.gripper_force = force

        return {
            "success": True,
            "position": self.gripper_position,
            "force": force,
            "state": "closed",
        }

    async def _handle_gripper_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper set position command."""
        if not self.config.has_gripper:
            return {"success": False, "error": "No gripper configured"}

        position = params.get("position")
        if position is None:
            return {"success": False, "error": "Missing position parameter"}

        position = max(0.0, min(self.config.gripper_max_width, position))

        if self.mock_mode:
            self.gripper_position = position

        return {"success": True, "position": self.gripper_position}

    async def _handle_gripper_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper get position command."""
        return {
            "success": True,
            "position": self.gripper_position,
            "max_width": self.config.gripper_max_width,
            "force": self.gripper_force,
        }

    # Convenience methods
    async def move_to_joints(self, positions: List[float], velocity: float = 0.5) -> bool:
        """Move arm to joint positions."""
        result = await self._handle_move_joints({
            "positions": positions,
            "velocity": velocity,
        })
        return result.get("success", False)

    async def move_to_pose(
        self,
        x: float, y: float, z: float,
        rx: float = 0, ry: float = 180, rz: float = 0,
        velocity: float = 0.1
    ) -> bool:
        """Move arm to Cartesian pose."""
        result = await self._handle_move_cartesian({
            "pose": [x, y, z, rx, ry, rz],
            "velocity": velocity,
        })
        return result.get("success", False)

    async def home(self) -> bool:
        """Move arm to home position."""
        result = await self._handle_home({})
        return result.get("success", False)

    async def open_gripper(self) -> bool:
        """Open the gripper."""
        result = await self._handle_gripper_open({})
        return result.get("success", False)

    async def close_gripper(self, force: float = 10.0) -> bool:
        """Close the gripper."""
        result = await self._handle_gripper_close({"force": force})
        return result.get("success", False)

    def get_pose(self) -> CartesianPose:
        """Get current Cartesian pose."""
        return CartesianPose(
            self.cartesian_pose.x, self.cartesian_pose.y, self.cartesian_pose.z,
            self.cartesian_pose.rx, self.cartesian_pose.ry, self.cartesian_pose.rz
        )

    def get_joints(self) -> List[float]:
        """Get current joint positions."""
        return self.joint_positions.copy()

    def get_state(self) -> ArmState:
        """Get current arm state."""
        return self.state
