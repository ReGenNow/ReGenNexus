#!/usr/bin/env python3
"""
RegenNexus UAP - Full AI-to-Hardware Demo

End-to-end demonstration of LLM controlling hardware through:
    [User] -> [LLM/Claude] -> [MCP] -> [RegenNexus UAP] -> [Hardware]

This example shows the complete flow:
1. LLM receives natural language command
2. LLM uses MCP tools to control hardware
3. Hardware executes and returns status
4. LLM confirms action to user

Usage:
    # Run with Ollama
    python full_ai_hardware_demo.py --provider ollama --model llama3

    # Run with LM Studio
    python full_ai_hardware_demo.py --provider lmstudio

    # Simulate without actual LLM
    python full_ai_hardware_demo.py --simulate

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import argparse
import asyncio
import json
import sys

sys.path.insert(0, "../../")

from regennexus.bridges.mcp_bridge import MCPServer, MCPTool, MCPResource
from regennexus.bridges.llm_bridge import LLMBridge, LLMConfig


# Simulated hardware
class SimulatedHardware:
    """Simulated hardware for demo purposes."""

    def __init__(self):
        self.arm_position = [0, 0, 0, 0, 0, 0, 0]
        self.gripper_state = "open"
        self.gpio_pins = {17: 0, 18: 0, 27: 0, 22: 0}
        self.sensors = {
            "temperature": 23.5,
            "humidity": 48.0,
            "distance": 15.2,
        }

    def move_arm(self, positions, duration=2.0):
        self.arm_position = positions
        return {"success": True, "positions": positions, "duration": duration}

    def control_gripper(self, action, force=10.0):
        self.gripper_state = action
        return {"success": True, "state": action, "force": force if action == "close" else None}

    def set_gpio(self, pin, value):
        self.gpio_pins[pin] = value
        return {"success": True, "pin": pin, "value": value}

    def read_sensor(self, sensor_type):
        value = self.sensors.get(sensor_type)
        return {"value": value, "unit": "°C" if sensor_type == "temperature" else "%"}


class AIHardwareController:
    """
    Combines LLM intelligence with hardware control via MCP.

    This is the "brain" that connects AI decision-making with
    physical hardware execution.
    """

    def __init__(self, llm_config: LLMConfig = None, simulate: bool = False):
        """
        Initialize the AI hardware controller.

        Args:
            llm_config: LLM configuration (Ollama, LM Studio, etc.)
            simulate: If True, simulate LLM responses
        """
        self.simulate = simulate
        self.hardware = SimulatedHardware()
        self.mcp_server = self._create_mcp_server()

        if not simulate and llm_config:
            self.llm = LLMBridge(llm_config)
        else:
            self.llm = None

        # Available tools for AI
        self.tools_schema = self._get_tools_schema()

    def _create_mcp_server(self):
        """Create MCP server with hardware tools."""
        server = MCPServer()

        # Arm movement
        server.register_tool(MCPTool(
            name="move_arm",
            description="Move the robotic arm to specified joint positions (7 joints, degrees)",
            input_schema={
                "type": "object",
                "properties": {
                    "positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "7 joint angles in degrees",
                    },
                    "duration": {"type": "number", "description": "Move time in seconds"},
                },
                "required": ["positions"],
            },
            handler=lambda args: self.hardware.move_arm(
                args["positions"], args.get("duration", 2.0)
            ),
        ))

        # Gripper control
        server.register_tool(MCPTool(
            name="control_gripper",
            description="Open or close the robot gripper",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["open", "close"]},
                    "force": {"type": "number", "description": "Grip force in Newtons"},
                },
                "required": ["action"],
            },
            handler=lambda args: self.hardware.control_gripper(
                args["action"], args.get("force", 10.0)
            ),
        ))

        # GPIO control
        server.register_tool(MCPTool(
            name="set_gpio",
            description="Set a GPIO pin to HIGH (1) or LOW (0)",
            input_schema={
                "type": "object",
                "properties": {
                    "pin": {"type": "integer", "description": "GPIO pin number"},
                    "value": {"type": "integer", "enum": [0, 1]},
                },
                "required": ["pin", "value"],
            },
            handler=lambda args: self.hardware.set_gpio(args["pin"], args["value"]),
        ))

        # Sensor reading
        server.register_tool(MCPTool(
            name="read_sensor",
            description="Read a sensor value (temperature, humidity, distance)",
            input_schema={
                "type": "object",
                "properties": {
                    "sensor_type": {
                        "type": "string",
                        "enum": ["temperature", "humidity", "distance"],
                    },
                },
                "required": ["sensor_type"],
            },
            handler=lambda args: self.hardware.read_sensor(args["sensor_type"]),
        ))

        return server

    def _get_tools_schema(self):
        """Get tools schema for LLM."""
        return [
            {
                "name": "move_arm",
                "description": "Move the robotic arm to joint positions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positions": {
                            "type": "array",
                            "description": "7 joint angles in degrees (e.g., [0, 30, -45, 0, 90, 0, 0])",
                        },
                    },
                    "required": ["positions"],
                },
            },
            {
                "name": "control_gripper",
                "description": "Open or close the gripper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["open", "close"]},
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "read_sensor",
                "description": "Read a sensor value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor_type": {"type": "string", "enum": ["temperature", "humidity", "distance"]},
                    },
                    "required": ["sensor_type"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict):
        """Execute an MCP tool."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        response = await self.mcp_server.handle_message(request)

        if "result" in response:
            content = response["result"]["content"][0]["text"]
            return json.loads(content)
        else:
            return {"error": response.get("error", {}).get("message", "Unknown error")}

    async def process_command(self, user_input: str) -> str:
        """
        Process a natural language command.

        The LLM interprets the command and decides which
        hardware actions to take.
        """
        print(f"\n[User]: {user_input}")

        if self.simulate:
            # Simulate LLM responses
            return await self._simulate_response(user_input)

        # Build system prompt with available tools
        system_prompt = """You are an AI assistant that controls robotic hardware.

Available tools:
- move_arm(positions): Move arm to 7 joint positions in degrees
- control_gripper(action): Open or close gripper
- read_sensor(sensor_type): Read temperature, humidity, or distance

When given a command, output JSON with:
{
    "tool": "tool_name",
    "arguments": {...},
    "explanation": "What you're doing"
}

For multi-step tasks, output an array of actions."""

        # Get LLM response
        response = await self.llm.chat(user_input, system_prompt=system_prompt)

        if not response:
            return "Failed to get LLM response"

        # Parse and execute actions
        try:
            actions = json.loads(response.content)
            if not isinstance(actions, list):
                actions = [actions]

            results = []
            for action in actions:
                tool = action.get("tool")
                args = action.get("arguments", {})
                explanation = action.get("explanation", "")

                print(f"[AI]: {explanation}")
                result = await self.execute_tool(tool, args)
                results.append({"action": tool, "result": result})
                print(f"[Hardware]: {result}")

            return json.dumps(results, indent=2)

        except json.JSONDecodeError:
            return response.content

    async def _simulate_response(self, user_input: str) -> str:
        """Simulate LLM response for demo without actual LLM."""
        user_lower = user_input.lower()

        if "pick" in user_lower or "grab" in user_lower:
            # Simulate pick operation
            print("[AI]: Moving arm to object position...")
            result1 = await self.execute_tool("move_arm", {"positions": [0, 45, -30, 0, 60, 0, 0]})
            print(f"[Hardware]: {result1}")

            print("[AI]: Closing gripper to grab object...")
            result2 = await self.execute_tool("control_gripper", {"action": "close", "force": 15})
            print(f"[Hardware]: {result2}")

            print("[AI]: Lifting object...")
            result3 = await self.execute_tool("move_arm", {"positions": [0, 30, -15, 0, 45, 0, 0]})
            print(f"[Hardware]: {result3}")

            return "Pick operation completed successfully"

        elif "place" in user_lower or "drop" in user_lower:
            print("[AI]: Moving arm to place position...")
            result1 = await self.execute_tool("move_arm", {"positions": [90, 45, -30, 0, 60, 0, 0]})
            print(f"[Hardware]: {result1}")

            print("[AI]: Opening gripper to release object...")
            result2 = await self.execute_tool("control_gripper", {"action": "open"})
            print(f"[Hardware]: {result2}")

            return "Place operation completed successfully"

        elif "temperature" in user_lower or "temp" in user_lower:
            print("[AI]: Reading temperature sensor...")
            result = await self.execute_tool("read_sensor", {"sensor_type": "temperature"})
            print(f"[Hardware]: {result}")
            return f"Temperature is {result['value']}{result['unit']}"

        elif "home" in user_lower or "reset" in user_lower:
            print("[AI]: Moving arm to home position...")
            result = await self.execute_tool("move_arm", {"positions": [0, 0, 0, 0, 0, 0, 0]})
            print(f"[Hardware]: {result}")
            return "Arm returned to home position"

        elif "light" in user_lower or "led" in user_lower:
            action = "on" if "on" in user_lower else "off"
            print(f"[AI]: Turning LED {action}...")
            result = await self.execute_tool("set_gpio", {"pin": 17, "value": 1 if action == "on" else 0})
            print(f"[Hardware]: {result}")
            return f"LED turned {action}"

        else:
            return "I understand. What would you like me to do with the hardware?"


async def interactive_demo(controller: AIHardwareController):
    """Run interactive demo."""
    print("\n" + "=" * 60)
    print("RegenNexus UAP - AI Hardware Control Demo")
    print("=" * 60)
    print()
    print("This demo shows LLM-controlled robotics through MCP.")
    print()
    print("Example commands:")
    print("  'Pick up the object'")
    print("  'Place it on the right side'")
    print("  'What is the temperature?'")
    print("  'Turn on the LED'")
    print("  'Go to home position'")
    print()
    print("Type 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            response = await controller.process_command(user_input)
            print(f"\n[Result]: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


async def main():
    parser = argparse.ArgumentParser(description="AI Hardware Control Demo")
    parser.add_argument(
        "--provider",
        choices=["ollama", "lmstudio", "openai"],
        default="ollama",
        help="LLM provider",
    )
    parser.add_argument("--model", default="llama3", help="Model name")
    parser.add_argument("--host", default="localhost", help="LLM API host")
    parser.add_argument("--port", type=int, default=11434, help="LLM API port")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate LLM responses (no actual LLM needed)",
    )
    args = parser.parse_args()

    # Configure LLM
    llm_config = None
    if not args.simulate:
        if args.provider == "ollama":
            llm_config = LLMConfig(
                provider="ollama",
                model=args.model,
                host=args.host,
                port=args.port,
            )
        elif args.provider == "lmstudio":
            llm_config = LLMConfig(
                provider="lmstudio",
                model=args.model,
                host=args.host,
                port=1234,
            )

    controller = AIHardwareController(llm_config=llm_config, simulate=args.simulate)
    await interactive_demo(controller)


if __name__ == "__main__":
    asyncio.run(main())
