#!/usr/bin/env python3
"""
RegenNexus UAP - LLM Hardware Control Demo

Demonstrates connecting to a local LLM (Ollama, LM Studio) and
using natural language to control hardware devices.

This is "MCP for hardware" - AI meets physical devices.

Prerequisites:
    - Ollama or LM Studio running on LAN
    - A model loaded (e.g., llama2, mistral)

Usage:
    # Connect to Ollama on localhost
    python llm_hardware_demo.py

    # Connect to Ollama on another machine
    python llm_hardware_demo.py --host 192.168.1.100

    # Use LM Studio instead
    python llm_hardware_demo.py --provider lmstudio --port 1234

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import argparse
import asyncio
import sys

sys.path.insert(0, "../../")

from regennexus.bridges.llm_bridge import (
    LLMBridge,
    LLMConfig,
    LLMProvider,
    UAP_LLM_Agent,
)
from regennexus.core.mesh import MeshConfig


async def demo_basic_llm():
    """Basic LLM connection demo."""
    print("\n" + "=" * 60)
    print("BASIC LLM CONNECTION")
    print("=" * 60)

    # Configure LLM
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        host="localhost",
        port=11434,
        model="llama2",  # Change to your model
    )

    bridge = LLMBridge(config)

    print(f"\nConnecting to {config.provider.value} at {config.base_url}...")

    if not await bridge.connect():
        print("Failed to connect. Is Ollama/LM Studio running?")
        print("Start with: ollama serve")
        return

    print(f"Connected! Available models: {bridge.available_models[:3]}")

    # Simple chat
    print("\n--- Simple Chat ---")
    response = await bridge.chat("Hello! What can you help me with?")
    if response:
        print(f"AI: {response.content[:200]}...")

    # Chat with hardware context
    print("\n--- Hardware-Aware Chat ---")
    response = await bridge.chat_with_context(
        "What devices do I have and what are their states?",
        devices={
            "raspi-001": {"type": "raspberry_pi", "gpio": {"pin_18": 1}},
            "sensor-temp": {"type": "sensor", "temperature": 23.5, "humidity": 45},
            "arm-001": {"type": "robot_arm", "state": "idle", "position": [0, 45, -30]},
        },
        capabilities=["gpio", "camera", "temperature", "arm.move", "gripper"]
    )
    if response:
        print(f"AI: {response.content[:500]}")

    await bridge.disconnect()


async def demo_llm_agent(host: str, port: int, provider: str):
    """Full LLM Agent with mesh networking."""
    print("\n" + "=" * 60)
    print("LLM HARDWARE AGENT")
    print("=" * 60)

    # Configure LLM
    llm_config = LLMConfig(
        provider=LLMProvider(provider),
        host=host,
        port=port,
        model="llama2",
        system_prompt=(
            "You are a hardware control assistant. "
            "You can see connected devices and help control them. "
            "Be concise and helpful."
        ),
    )

    # Configure mesh (for device discovery)
    mesh_config = MeshConfig(
        node_id="ai-controller",
        entity_type="controller",
        capabilities=["llm", "command"],
    )

    # Create agent
    agent = UAP_LLM_Agent(
        llm_config=llm_config,
        mesh_config=mesh_config,
    )

    print(f"\nStarting AI agent...")
    print(f"LLM: {llm_config.provider.value} at {llm_config.base_url}")

    if not await agent.start():
        print("Failed to start agent")
        return

    print("Agent started! Listening for devices on the network...")
    print("\nYou can now chat with the AI to control hardware.")
    print("Type 'quit' to exit.\n")

    # Interactive loop
    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input:
                continue

            response = await agent.process(user_input)
            print(f"AI: {response}\n")

    except KeyboardInterrupt:
        print("\n")

    await agent.stop()
    print("Agent stopped.")


async def demo_ollama_direct():
    """Direct Ollama API demo without mesh."""
    print("\n" + "=" * 60)
    print("DIRECT OLLAMA DEMO")
    print("=" * 60)

    bridge = LLMBridge(LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama2",
    ))

    if not await bridge.connect():
        print("Cannot connect to Ollama. Make sure it's running:")
        print("  ollama serve")
        print("  ollama pull llama2")
        return

    print("Connected to Ollama!\n")

    # Example: Generate device control commands
    prompt = """
You are a hardware controller. Convert this natural language to a JSON command:

User request: "Turn on the kitchen lights and set them to 50% brightness"

Respond with ONLY a JSON object like:
{"device": "kitchen_lights", "action": "set", "params": {"power": true, "brightness": 50}}
"""

    print("Asking LLM to generate hardware command...")
    response = await bridge.generate(prompt)

    if response:
        print(f"\nLLM Response:\n{response.content}")

    await bridge.disconnect()


async def main():
    parser = argparse.ArgumentParser(description="LLM Hardware Control Demo")
    parser.add_argument("--host", default="localhost", help="LLM server host")
    parser.add_argument("--port", type=int, default=None, help="LLM server port")
    parser.add_argument(
        "--provider",
        choices=["ollama", "lmstudio", "openai"],
        default="ollama",
        help="LLM provider"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "agent", "direct"],
        default="basic",
        help="Demo mode"
    )
    args = parser.parse_args()

    # Set default port based on provider
    if args.port is None:
        args.port = 11434 if args.provider == "ollama" else 1234

    print("=" * 60)
    print("RegenNexus UAP - LLM Hardware Integration Demo")
    print("=" * 60)
    print(f"\nProvider: {args.provider}")
    print(f"Server: {args.host}:{args.port}")

    if args.mode == "basic":
        await demo_basic_llm()
    elif args.mode == "agent":
        await demo_llm_agent(args.host, args.port, args.provider)
    elif args.mode == "direct":
        await demo_ollama_direct()

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
