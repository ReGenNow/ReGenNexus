#!/usr/bin/env python3
"""
Amber B1 Robotic Arm ROS2 Integration Example

This example demonstrates how to bridge ReGenNexus Core with a ROS 2-based
Amber B1 robotic arm. It maps the /joint_states topic into a ReGenNexus entity
and publishes joint trajectory commands to the arm.

Requirements:
  - ROS 2 (e.g., Foxy, Humble) installed and sourced
  - ROS packages: sensor_msgs, trajectory_msgs
  - rclpy Python client library
"""
import asyncio
import logging

from regennexus.bridges.ros_bridge import ROSBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def jointstate_to_dict(msg):
    """Convert sensor_msgs/JointState to a simple dict."""
    return {
        'name': list(msg.name),
        'position': list(msg.position),
        'velocity': list(msg.velocity) if hasattr(msg, 'velocity') else [],
        'effort': list(msg.effort) if hasattr(msg, 'effort') else []
    }

def dict_to_trajectory(msg_dict):
    """Convert a dict to trajectory_msgs/JointTrajectory message."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    traj = JointTrajectory()
    traj.joint_names = msg_dict.get('joint_names', [])
    point = JointTrajectoryPoint()
    point.positions = msg_dict.get('positions', [])
    point.velocities = msg_dict.get('velocities', [])
    # time_from_start expects a builtin_interfaces/Duration
    from builtin_interfaces.msg import Duration
    point.time_from_start = Duration(sec=int(msg_dict.get('time_from_start', 1)), nanosec=0)
    traj.points = [point]
    return traj

async def async_main():
    bridge = ROSBridge(node_name='amber_b1_bridge')
    await bridge.initialize()

    if not bridge.ros_initialized:
        logger.error("ROS2 client library not available or failed to initialize.")
        return

    # Map the Amber B1's joint_states topic into a ReGenNexus entity
    await bridge.map_topic_to_entity(
        topic_name='/joint_states',
        entity_id='amber_b1_arm',
        direction='to_entity',
        message_type='sensor_msgs/JointState',
        topic_to_entity_transform=jointstate_to_dict
    )
    logger.info("Mapped /joint_states -> entity 'amber_b1_arm'.")

    # Map UAP messages from entity 'arm_controller' to the arm's command topic
    await bridge.map_topic_to_entity(
        topic_name='/arm_controller/command',
        entity_id='arm_controller',
        direction='to_topic',
        message_type='trajectory_msgs/JointTrajectory',
        entity_to_topic_transform=dict_to_trajectory
    )
    logger.info("Mapped entity 'arm_controller' -> /arm_controller/command.")

    # Example: send a trajectory command from UAP to the arm
    cmd = {
        'joint_names': ['joint1','joint2','joint3','joint4','joint5','joint6'],
        'positions': [0.0, 0.5, 1.0, -0.5, 0.2, -1.2],
        'time_from_start': 2
    }
    logger.info(f"Publishing trajectory command: {cmd}")
    await bridge.publish_to_ros('/arm_controller/command', cmd)

    # Allow bridge to process a few incoming joint_states messages
    await asyncio.sleep(5)

    await bridge.shutdown()
    logger.info("ROS bridge shut down.")

def main():
    asyncio.run(async_main())

if __name__ == '__main__':
    main()