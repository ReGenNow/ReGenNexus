#!/usr/bin/env python3
"""
RegenNexus UAP - Robotic Arm Demo

Demonstrates Amber B1 and Lucid One robotic arm control.
Uses mock mode by default - set mock_mode=False for real hardware.

Usage:
    python arm_demo.py --arm amber     # Test Amber B1
    python arm_demo.py --arm lucid     # Test Lucid One
    python arm_demo.py --arm both      # Test both (default)

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import argparse
import asyncio
import sys

# Add parent directory to path for development
sys.path.insert(0, "../../")

from regennexus.plugins import get_amber_b1_plugin, get_lucid_one_plugin


async def demo_amber_b1():
    """Demonstrate Amber B1 robotic arm capabilities."""
    print("\n" + "=" * 60)
    print("AMBER B1 ROBOTIC ARM DEMO")
    print("=" * 60)

    # Get plugin class and create instance
    AmberB1 = get_amber_b1_plugin()
    arm = AmberB1(
        entity_id="amber_demo",
        mock_mode=True  # Set False for real hardware
    )

    try:
        # Initialize
        print("\n[1] Initializing arm...")
        await arm.initialize()
        print(f"    State: {arm.get_state()}")
        print(f"    Joints: {arm.config.num_joints}")
        print(f"    Has gripper: {arm.config.has_gripper}")

        # Get initial positions
        print("\n[2] Current joint positions:")
        positions = arm.get_positions()
        for i, pos in enumerate(positions):
            print(f"    Joint {i+1}: {pos:.1f}°")

        # Move to a position
        print("\n[3] Moving to test position...")
        target = [30, 45, -20, 0, 60, 0, 0]
        success = await arm.move_to(target, duration=1.0)
        print(f"    Move completed: {success}")

        positions = arm.get_positions()
        print("    New positions:")
        for i, pos in enumerate(positions):
            print(f"    Joint {i+1}: {pos:.1f}°")

        # Gripper demo
        print("\n[4] Gripper operations...")
        print("    Opening gripper...")
        await arm.open_gripper()
        result = await arm.execute_command("gripper.get_position", {})
        print(f"    Gripper width: {result['position']:.1f}mm")

        print("    Closing gripper with 15N force...")
        await arm.close_gripper(force=15.0)
        result = await arm.execute_command("gripper.get_position", {})
        print(f"    Gripper width: {result['position']:.1f}mm")

        # Home the arm
        print("\n[5] Homing arm...")
        await arm.home()
        print("    Arm homed successfully")

        # Get final state
        print("\n[6] Final state:")
        result = await arm.execute_command("get_state", {})
        print(f"    State: {result['state']}")
        print(f"    Positions: {result['positions']}")

    finally:
        print("\n[7] Shutting down...")
        await arm.shutdown()
        print("    Shutdown complete")


async def demo_lucid_one():
    """Demonstrate Lucid One robotic arm capabilities."""
    print("\n" + "=" * 60)
    print("LUCID ONE ROBOTIC ARM DEMO")
    print("=" * 60)

    # Get plugin class and create instance
    LucidOne = get_lucid_one_plugin()
    arm = LucidOne(
        entity_id="lucid_demo",
        mock_mode=True  # Set False for real hardware
    )

    try:
        # Initialize
        print("\n[1] Initializing arm...")
        await arm.initialize()
        print(f"    State: {arm.get_state()}")
        print(f"    Model: {arm.metadata['arm_type']}")
        print(f"    Payload: {arm.config.payload}kg")
        print(f"    Reach: {arm.config.reach}mm")

        # Get Cartesian pose
        print("\n[2] Current Cartesian pose:")
        pose = arm.get_pose()
        print(f"    Position: X={pose.x:.1f}, Y={pose.y:.1f}, Z={pose.z:.1f} mm")
        print(f"    Rotation: RX={pose.rx:.1f}, RY={pose.ry:.1f}, RZ={pose.rz:.1f}°")

        # Move in Cartesian space
        print("\n[3] Moving to Cartesian position...")
        await arm.move_to_pose(
            x=450, y=100, z=250,
            rx=0, ry=180, rz=30,
            velocity=0.2
        )

        pose = arm.get_pose()
        print(f"    New position: X={pose.x:.1f}, Y={pose.y:.1f}, Z={pose.z:.1f} mm")

        # Force/torque sensing
        print("\n[4] Force/torque sensor:")
        result = await arm.execute_command("get_force_torque", {})
        print(f"    Force:  Fx={result['fx']:.2f}, Fy={result['fy']:.2f}, Fz={result['fz']:.2f} N")
        print(f"    Torque: Tx={result['tx']:.2f}, Ty={result['ty']:.2f}, Tz={result['tz']:.2f} Nm")

        # Teach mode
        print("\n[5] Teach mode demo...")
        print("    Enabling teach mode...")
        await arm.execute_command("teach_mode", {"enable": True})
        print(f"    State: {arm.get_state()}")

        print("    Disabling teach mode...")
        await arm.execute_command("teach_mode", {"enable": False})

        # Trajectory recording
        print("\n[6] Trajectory recording demo...")
        print("    Starting recording...")
        await arm.execute_command("record_trajectory", {"action": "start"})

        print("    Adding waypoints...")
        for i in range(3):
            await arm.move_to_joints([i * 10, 0, 0, 0, 0, 0, 0], velocity=2.0)
            await arm.execute_command("record_trajectory", {"action": "add_point"})
            print(f"    Point {i+1} added")

        result = await arm.execute_command("record_trajectory", {"action": "stop"})
        print(f"    Recording stopped: {result['points']} points recorded")

        # Gripper demo
        print("\n[7] Gripper operations...")
        await arm.open_gripper()
        print("    Gripper opened")

        await arm.close_gripper(force=10.0)
        print("    Gripper closed")

        # Home
        print("\n[8] Homing arm...")
        await arm.home()

        # Final state
        print("\n[9] Final state:")
        result = await arm.execute_command("get_state", {})
        print(f"    State: {result['state']}")
        print(f"    Control mode: {result['control_mode']}")

    finally:
        print("\n[10] Shutting down...")
        await arm.shutdown()
        print("     Shutdown complete")


async def demo_pick_and_place():
    """Demonstrate pick and place operation."""
    print("\n" + "=" * 60)
    print("PICK AND PLACE DEMO")
    print("=" * 60)

    LucidOne = get_lucid_one_plugin()
    arm = LucidOne(entity_id="pick_place_demo", mock_mode=True)

    try:
        await arm.initialize()
        print("\nArm initialized")

        # Define positions
        pick_pos = (400, -100, 50)
        place_pos = (400, 100, 50)
        safe_height = 100

        print(f"\nPick position:  {pick_pos}")
        print(f"Place position: {place_pos}")

        # Pick sequence
        print("\n--- PICK SEQUENCE ---")

        print("1. Moving above pick position...")
        await arm.move_to_pose(pick_pos[0], pick_pos[1], pick_pos[2] + safe_height,
                               ry=180, velocity=0.1)

        print("2. Opening gripper...")
        await arm.open_gripper()

        print("3. Descending to pick...")
        await arm.move_to_pose(pick_pos[0], pick_pos[1], pick_pos[2],
                               ry=180, velocity=0.05)

        print("4. Closing gripper...")
        await arm.close_gripper(force=15.0)
        await asyncio.sleep(0.3)

        print("5. Lifting object...")
        await arm.move_to_pose(pick_pos[0], pick_pos[1], pick_pos[2] + safe_height,
                               ry=180, velocity=0.1)

        # Place sequence
        print("\n--- PLACE SEQUENCE ---")

        print("6. Moving to place position...")
        await arm.move_to_pose(place_pos[0], place_pos[1], place_pos[2] + safe_height,
                               ry=180, velocity=0.1)

        print("7. Descending to place...")
        await arm.move_to_pose(place_pos[0], place_pos[1], place_pos[2],
                               ry=180, velocity=0.05)

        print("8. Releasing object...")
        await arm.open_gripper()
        await asyncio.sleep(0.3)

        print("9. Retracting...")
        await arm.move_to_pose(place_pos[0], place_pos[1], place_pos[2] + safe_height,
                               ry=180, velocity=0.1)

        print("\n✓ Pick and place complete!")

        # Return home
        print("\nReturning to home...")
        await arm.home()

    finally:
        await arm.shutdown()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RegenNexus Robotic Arm Demo")
    parser.add_argument(
        "--arm",
        choices=["amber", "lucid", "both", "pickplace"],
        default="both",
        help="Which arm to demo (default: both)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RegenNexus UAP - Robotic Arm Demo")
    print("Running in MOCK MODE (no hardware required)")
    print("=" * 60)

    try:
        if args.arm == "amber":
            await demo_amber_b1()
        elif args.arm == "lucid":
            await demo_lucid_one()
        elif args.arm == "pickplace":
            await demo_pick_and_place()
        else:
            await demo_amber_b1()
            await demo_lucid_one()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
