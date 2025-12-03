#!/usr/bin/env python3
"""
Amber B1 MoveIt2 Planning Bridge Example

This example uses MoveIt2 to plan a joint-space trajectory for the Amber B1 arm
and publishes the resulting trajectory to the arm's command topic via the ROSBridge.

Requirements:
  - ROS 2 (e.g., Foxy, Humble) installed and sourced
  - MoveIt2 Python API (moveit_commander)
  - ROS packages: sensor_msgs, trajectory_msgs
  - rclpy, moveit_commander
  - regennexus with ROSBridge
"""
import asyncio
import argparse
import logging

try:
    import rclpy
    from moveit_commander import RobotCommander, MoveGroupCommander, roscpp_initialize, roscpp_shutdown
except ImportError:
    rclpy = None

from regennexus.bridges.ros_bridge import ROSBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Amber B1 MoveIt2 Bridge Example")
    p.add_argument("--node-name", default="amber_b1_moveit_bridge", help="ROS2 node name")
    p.add_argument("--planning-group", default="arm", help="MoveIt2 planning group name")
    p.add_argument("--joint-names", required=True,
                   help="Comma-separated joint names, e.g. 'joint1,joint2,...'")
    p.add_argument("--positions", required=True,
                   help="Comma-separated positions matching joint names")
    p.add_argument("--cmd-topic", default="/arm_controller/command",
                   help="ROS2 topic to publish trajectory")
    p.add_argument("--entity-id", default="arm_controller",
                   help="ReGenNexus entity ID for the arm controller")
    return p.parse_args()

async def async_main(args):
    if rclpy is None:
        logger.error("moveit_commander or rclpy not available; install MoveIt2 Python API.")
        return
    # Initialize ROS2 and MoveIt
    rclpy.init()
    roscpp_initialize()
    robot = RobotCommander()
    group = MoveGroupCommander(args.planning_group)

    # Set joint target
    names = args.joint_names.split(',')
    positions = [float(x) for x in args.positions.split(',')]
    logger.info(f"Planning to joints {names} -> {positions}")
    group.set_joint_value_target(dict(zip(names, positions)))
    plan = group.plan()
    traj = plan.joint_trajectory
    roscpp_shutdown()
    rclpy.shutdown()

    # Bridge publish
    bridge = ROSBridge(node_name=args.node_name)
    await bridge.initialize()
    msg = {
        'joint_names': traj.joint_names,
        'positions': traj.points[-1].positions,
        'time_from_start': traj.points[-1].time_from_start.sec
    }
    logger.info(f"Publishing planned trajectory: {msg}")
    await bridge.publish_to_ros(args.cmd_topic, msg)
    await asyncio.sleep(2)
    await bridge.shutdown()

def main():
    args = parse_args()
    asyncio.run(async_main(args))

if __name__ == '__main__':
    main()