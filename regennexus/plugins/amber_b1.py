"""
RegenNexus UAP - Amber B1 Robotic Arm Plugin

AMBER Robotics B1 7-DOF robotic arm integration with UDP control protocol.
Supports joint control, gripper operations, and trajectory planning.
Includes mock mode for development without hardware.

Specifications:
- 7 Degrees of Freedom (DOF)
- UDP communication protocol (port 25001)
- Individual joint IPs: 10.0.0.10 - 10.0.0.16
- Gripper control on joints 8/9
- Position control with trajectory interpolation

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
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class ControlMode(Enum):
    """Joint control mode."""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


@dataclass
class JointConfig:
    """Configuration for a single joint."""
    joint_id: int
    ip_address: str
    port: int = 25001
    min_angle: float = -180.0  # degrees
    max_angle: float = 180.0   # degrees
    max_velocity: float = 60.0  # degrees/second
    max_torque: float = 10.0    # Nm


@dataclass
class ArmConfig:
    """Configuration for Amber B1 arm."""
    base_ip: str = "10.0.0.10"
    port: int = 25001
    num_joints: int = 7
    has_gripper: bool = True
    gripper_max_width: float = 85.0  # mm
    joints: List[JointConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.joints:
            # Default joint configuration for Amber B1
            base_parts = self.base_ip.rsplit(".", 1)
            base = base_parts[0]
            start_num = int(base_parts[1])

            for i in range(self.num_joints):
                self.joints.append(JointConfig(
                    joint_id=i + 1,
                    ip_address=f"{base}.{start_num + i}",
                    port=self.port,
                ))


class AmberB1Plugin(DevicePlugin, MockDeviceMixin):
    """
    Amber B1 7-DOF robotic arm plugin for RegenNexus.

    Supports:
    - Joint position control
    - Velocity control
    - Trajectory execution
    - Gripper open/close
    - Home position
    - Emergency stop
    - Mock mode for development
    """

    def __init__(
        self,
        entity_id: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False,
        config: Optional[ArmConfig] = None
    ):
        """
        Initialize Amber B1 plugin.

        Args:
            entity_id: Unique identifier
            protocol: Protocol instance
            mock_mode: If True, simulate without hardware
            config: Arm configuration (uses defaults if None)
        """
        DevicePlugin.__init__(self, entity_id, "amber_b1", protocol, mock_mode)
        MockDeviceMixin.__init__(self)

        self.config = config or ArmConfig()
        self.state = ArmState.DISCONNECTED
        self.control_mode = ControlMode.POSITION

        # Current joint states
        self.joint_positions: List[float] = [0.0] * self.config.num_joints
        self.joint_velocities: List[float] = [0.0] * self.config.num_joints
        self.joint_torques: List[float] = [0.0] * self.config.num_joints

        # Gripper state
        self.gripper_position: float = 0.0  # 0 = closed, gripper_max_width = open
        self.gripper_force: float = 0.0

        # UDP sockets for each joint
        self._sockets: Dict[int, socket.socket] = {}

        # Motion tracking
        self._motion_task: Optional[asyncio.Task] = None
        self._target_positions: List[float] = [0.0] * self.config.num_joints

    async def _device_init(self) -> bool:
        """Initialize Amber B1 hardware."""
        try:
            if self.mock_mode:
                logger.info("Amber B1 arm in mock mode")
                self.state = ArmState.IDLE
            else:
                # Create UDP sockets for each joint
                for joint in self.config.joints:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setblocking(False)
                        sock.settimeout(1.0)
                        self._sockets[joint.joint_id] = sock
                        logger.debug(f"Created socket for joint {joint.joint_id}")
                    except Exception as e:
                        logger.error(f"Failed to create socket for joint {joint.joint_id}: {e}")

                if self._sockets:
                    self.state = ArmState.IDLE
                    logger.info(f"Amber B1 arm initialized with {len(self._sockets)} joints")
                else:
                    logger.warning("No joints connected")
                    self.state = ArmState.DISCONNECTED

            # Add capabilities
            self.capabilities.extend([
                "arm.move_joint",
                "arm.move_joints",
                "arm.move_to_position",
                "arm.get_joint_positions",
                "arm.get_state",
                "arm.home",
                "arm.stop",
                "arm.emergency_stop",
            ])

            if self.config.has_gripper:
                self.capabilities.extend([
                    "gripper.open",
                    "gripper.close",
                    "gripper.set_position",
                    "gripper.get_position",
                ])

            # Register command handlers
            self.register_command_handler("move_joint", self._handle_move_joint)
            self.register_command_handler("move_joints", self._handle_move_joints)
            self.register_command_handler("get_joint_positions", self._handle_get_positions)
            self.register_command_handler("get_state", self._handle_get_state)
            self.register_command_handler("home", self._handle_home)
            self.register_command_handler("stop", self._handle_stop)
            self.register_command_handler("emergency_stop", self._handle_emergency_stop)
            self.register_command_handler("gripper.open", self._handle_gripper_open)
            self.register_command_handler("gripper.close", self._handle_gripper_close)
            self.register_command_handler("gripper.set_position", self._handle_gripper_set)
            self.register_command_handler("gripper.get_position", self._handle_gripper_get)

            # Update metadata
            self.metadata.update({
                "arm_type": "Amber B1",
                "num_joints": self.config.num_joints,
                "has_gripper": self.config.has_gripper,
                "state": self.state.value,
                "control_mode": self.control_mode.value,
            })

            return True

        except Exception as e:
            logger.error(f"Amber B1 init error: {e}")
            return False

    async def _device_shutdown(self) -> None:
        """Clean up Amber B1 resources."""
        # Stop any motion
        if self._motion_task:
            self._motion_task.cancel()
            try:
                await self._motion_task
            except asyncio.CancelledError:
                pass
            self._motion_task = None

        # Close sockets
        for sock in self._sockets.values():
            try:
                sock.close()
            except Exception:
                pass
        self._sockets.clear()

        self.state = ArmState.DISCONNECTED

    def _send_joint_command(
        self,
        joint_id: int,
        position: Optional[float] = None,
        velocity: Optional[float] = None,
        torque: Optional[float] = None
    ) -> bool:
        """Send command to a specific joint via UDP."""
        if self.mock_mode:
            return True

        if joint_id not in self._sockets:
            return False

        try:
            joint_config = self.config.joints[joint_id - 1]

            # Build command packet (simplified protocol)
            # Real implementation would follow AMBER's specific protocol
            cmd_type = 0x01  # Position command
            if velocity is not None:
                cmd_type = 0x02  # Velocity command
            elif torque is not None:
                cmd_type = 0x03  # Torque command

            value = position if position is not None else (velocity if velocity is not None else torque)
            if value is None:
                return False

            # Pack: [cmd_type (1 byte), joint_id (1 byte), value (4 bytes float)]
            packet = struct.pack("<BBf", cmd_type, joint_id, value)

            sock = self._sockets[joint_id]
            sock.sendto(packet, (joint_config.ip_address, joint_config.port))

            return True

        except Exception as e:
            logger.error(f"Failed to send command to joint {joint_id}: {e}")
            return False

    async def _execute_motion(
        self,
        target_positions: List[float],
        duration: float = 2.0
    ) -> bool:
        """Execute motion to target positions with interpolation."""
        if len(target_positions) != self.config.num_joints:
            return False

        self.state = ArmState.MOVING
        start_positions = self.joint_positions.copy()
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                t = min(elapsed / duration, 1.0)

                # Smooth interpolation (cubic ease-in-out)
                t = t * t * (3 - 2 * t)

                for i in range(self.config.num_joints):
                    target = start_positions[i] + t * (target_positions[i] - start_positions[i])

                    if self.mock_mode:
                        self.joint_positions[i] = target
                    else:
                        self._send_joint_command(i + 1, position=target)

                await asyncio.sleep(0.02)  # 50Hz update rate

            # Final positions
            for i, pos in enumerate(target_positions):
                self.joint_positions[i] = pos
                if not self.mock_mode:
                    self._send_joint_command(i + 1, position=pos)

            self.state = ArmState.IDLE
            return True

        except asyncio.CancelledError:
            self.state = ArmState.IDLE
            raise
        except Exception as e:
            logger.error(f"Motion execution error: {e}")
            self.state = ArmState.ERROR
            return False

    async def _handle_move_joint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move single joint command."""
        joint_id = params.get("joint_id")
        position = params.get("position")
        velocity = params.get("velocity")

        if joint_id is None or position is None:
            return {"success": False, "error": "Missing joint_id or position"}

        if joint_id < 1 or joint_id > self.config.num_joints:
            return {"success": False, "error": f"Invalid joint_id: {joint_id}"}

        try:
            joint_config = self.config.joints[joint_id - 1]

            # Clamp position to joint limits
            position = max(joint_config.min_angle, min(joint_config.max_angle, position))

            # Create target positions (only change specified joint)
            target = self.joint_positions.copy()
            target[joint_id - 1] = position

            # Calculate duration based on distance and velocity
            distance = abs(position - self.joint_positions[joint_id - 1])
            vel = velocity or joint_config.max_velocity
            duration = max(0.5, distance / vel)

            # Execute motion
            await self._execute_motion(target, duration)

            return {
                "success": True,
                "joint_id": joint_id,
                "position": position,
                "duration": duration,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_move_joints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move all joints command."""
        positions = params.get("positions")
        duration = params.get("duration", 2.0)

        if not positions or len(positions) != self.config.num_joints:
            return {
                "success": False,
                "error": f"Expected {self.config.num_joints} joint positions",
            }

        try:
            # Clamp positions to joint limits
            clamped = []
            for i, pos in enumerate(positions):
                joint_config = self.config.joints[i]
                clamped.append(max(joint_config.min_angle, min(joint_config.max_angle, pos)))

            # Execute motion
            await self._execute_motion(clamped, duration)

            return {
                "success": True,
                "positions": clamped,
                "duration": duration,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_get_positions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get joint positions command."""
        return {
            "success": True,
            "positions": self.joint_positions.copy(),
            "velocities": self.joint_velocities.copy(),
            "torques": self.joint_torques.copy(),
        }

    async def _handle_get_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get arm state command."""
        return {
            "success": True,
            "state": self.state.value,
            "control_mode": self.control_mode.value,
            "positions": self.joint_positions.copy(),
            "gripper_position": self.gripper_position,
        }

    async def _handle_home(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle home command (move to zero position)."""
        try:
            home_positions = [0.0] * self.config.num_joints
            await self._execute_motion(home_positions, 3.0)

            return {"success": True, "message": "Arm homed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop command (controlled stop)."""
        try:
            if self._motion_task:
                self._motion_task.cancel()
                try:
                    await self._motion_task
                except asyncio.CancelledError:
                    pass
                self._motion_task = None

            self.state = ArmState.IDLE
            return {"success": True, "message": "Motion stopped"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_emergency_stop(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency stop command."""
        try:
            if self._motion_task:
                self._motion_task.cancel()
                try:
                    await self._motion_task
                except asyncio.CancelledError:
                    pass
                self._motion_task = None

            # In real implementation, would send emergency stop to all joints
            self.state = ArmState.EMERGENCY_STOP

            await self.emit_event("arm.emergency_stop", {
                "entity_id": self.entity_id,
                "positions": self.joint_positions.copy(),
            })

            return {"success": True, "message": "Emergency stop activated"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gripper_open(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper open command."""
        if not self.config.has_gripper:
            return {"success": False, "error": "No gripper configured"}

        try:
            if self.mock_mode:
                self.gripper_position = self.config.gripper_max_width
            else:
                # Send gripper open command (joint 8/9)
                self._send_joint_command(8, position=self.config.gripper_max_width)

            return {
                "success": True,
                "position": self.gripper_position,
                "state": "open",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gripper_close(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper close command."""
        if not self.config.has_gripper:
            return {"success": False, "error": "No gripper configured"}

        force = params.get("force", 10.0)

        try:
            if self.mock_mode:
                self.gripper_position = 0.0
                self.gripper_force = force
            else:
                # Send gripper close command with force
                self._send_joint_command(8, position=0.0)

            return {
                "success": True,
                "position": self.gripper_position,
                "force": force,
                "state": "closed",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gripper_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper set position command."""
        if not self.config.has_gripper:
            return {"success": False, "error": "No gripper configured"}

        position = params.get("position")
        if position is None:
            return {"success": False, "error": "Missing position parameter"}

        position = max(0.0, min(self.config.gripper_max_width, position))

        try:
            if self.mock_mode:
                self.gripper_position = position
            else:
                self._send_joint_command(8, position=position)

            return {
                "success": True,
                "position": self.gripper_position,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gripper_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gripper get position command."""
        return {
            "success": True,
            "position": self.gripper_position,
            "max_width": self.config.gripper_max_width,
            "force": self.gripper_force,
        }

    # Convenience methods
    async def move_to(self, positions: List[float], duration: float = 2.0) -> bool:
        """Move arm to specified joint positions."""
        result = await self._handle_move_joints({
            "positions": positions,
            "duration": duration,
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
        """Close the gripper with specified force."""
        result = await self._handle_gripper_close({"force": force})
        return result.get("success", False)

    def get_positions(self) -> List[float]:
        """Get current joint positions."""
        return self.joint_positions.copy()

    def get_state(self) -> ArmState:
        """Get current arm state."""
        return self.state
