"""
RegenNexus UAP - LLM Bridge

Bridge for connecting to local LLMs (Ollama, LM Studio, etc.) over LAN.
Enables AI-to-hardware communication through the UAP protocol.

This makes RegenNexus a hardware bridge for the MCP ecosystem -
LLMs can control physical devices through natural language.

Supported LLM Backends:
- Ollama (localhost:11434 or LAN)
- LM Studio (localhost:1234 or LAN)
- Any OpenAI-compatible API
- vLLM, text-generation-webui, LocalAI

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import aiohttp
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    OPENAI_COMPATIBLE = "openai"
    VLLM = "vllm"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """LLM connection configuration."""
    provider: LLMProvider = LLMProvider.OLLAMA
    host: str = "localhost"
    port: int = 11434  # Ollama default
    model: str = "llama2"
    api_key: Optional[str] = None  # For OpenAI-compatible APIs
    timeout: float = 120.0  # LLM can be slow

    # Endpoint paths (auto-configured per provider)
    chat_endpoint: str = "/api/chat"
    generate_endpoint: str = "/api/generate"
    models_endpoint: str = "/api/tags"

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None

    def __post_init__(self):
        """Configure endpoints based on provider."""
        if self.provider == LLMProvider.OLLAMA:
            self.port = self.port or 11434
            self.chat_endpoint = "/api/chat"
            self.generate_endpoint = "/api/generate"
            self.models_endpoint = "/api/tags"
        elif self.provider == LLMProvider.LMSTUDIO:
            self.port = self.port or 1234
            self.chat_endpoint = "/v1/chat/completions"
            self.generate_endpoint = "/v1/completions"
            self.models_endpoint = "/v1/models"
        elif self.provider in (LLMProvider.OPENAI_COMPATIBLE, LLMProvider.VLLM):
            self.chat_endpoint = "/v1/chat/completions"
            self.generate_endpoint = "/v1/completions"
            self.models_endpoint = "/v1/models"

    @property
    def base_url(self) -> str:
        """Get base URL."""
        return f"http://{self.host}:{self.port}"


@dataclass
class LLMMessage:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int = 0
    finish_reason: str = "stop"
    raw_response: Dict[str, Any] = field(default_factory=dict)


class LLMBridge:
    """
    Bridge for LLM communication over LAN.

    Connects RegenNexus UAP to local LLMs, enabling:
    - Natural language device control
    - AI-powered automation
    - Hardware-aware AI assistants

    Example:
        # Connect to Ollama on another machine
        bridge = LLMBridge(LLMConfig(
            provider=LLMProvider.OLLAMA,
            host="192.168.1.100",  # LAN IP of Ollama server
            model="llama2"
        ))
        await bridge.connect()

        # Chat with the LLM
        response = await bridge.chat("Turn on the kitchen lights")

        # Or use with device context
        response = await bridge.chat_with_context(
            "What's the temperature?",
            devices={"sensor_01": {"temp": 23.5, "humidity": 45}}
        )
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM bridge.

        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._available_models: List[str] = []

        # Conversation history
        self._history: List[LLMMessage] = []

        # Tool/function definitions for hardware control
        self._tools: List[Dict[str, Any]] = []
        self._tool_handlers: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """
        Connect to the LLM server.

        Returns:
            True if connected successfully
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp required for LLM bridge")
            return False

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

            # Test connection by listing models
            self._available_models = await self._list_models()

            if self._available_models:
                self._connected = True
                logger.info(
                    f"Connected to {self.config.provider.value} at "
                    f"{self.config.base_url}, models: {self._available_models[:3]}"
                )
                return True
            else:
                logger.warning("Connected but no models available")
                self._connected = True
                return True

        except Exception as e:
            logger.error(f"LLM connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from LLM server."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        self._history.clear()

    async def _list_models(self) -> List[str]:
        """List available models."""
        if not self._session:
            return []

        try:
            url = f"{self.config.base_url}{self.config.models_endpoint}"

            async with self._session.get(url) as response:
                if response.status != 200:
                    return []

                data = await response.json()

                # Parse based on provider
                if self.config.provider == LLMProvider.OLLAMA:
                    models = data.get("models", [])
                    return [m.get("name", "") for m in models]
                else:
                    # OpenAI-compatible format
                    models = data.get("data", [])
                    return [m.get("id", "") for m in models]

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def generate(
        self,
        prompt: str,
        stream: bool = False
    ) -> Optional[LLMResponse]:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            stream: Whether to stream response

        Returns:
            LLMResponse or None if failed
        """
        if not self._connected or not self._session:
            return None

        try:
            url = f"{self.config.base_url}{self.config.generate_endpoint}"

            if self.config.provider == LLMProvider.OLLAMA:
                payload = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                }
            else:
                # OpenAI-compatible
                payload = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stream": False,
                }

            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"LLM error: {error}")
                    return None

                data = await response.json()

                # Parse response based on provider
                if self.config.provider == LLMProvider.OLLAMA:
                    content = data.get("response", "")
                else:
                    choices = data.get("choices", [])
                    content = choices[0].get("text", "") if choices else ""

                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider=self.config.provider,
                    raw_response=data,
                )

        except Exception as e:
            logger.error(f"Generate error: {e}")
            return None

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        keep_history: bool = True
    ) -> Optional[LLMResponse]:
        """
        Send a chat message.

        Args:
            message: User message
            system_prompt: Override system prompt
            keep_history: Whether to maintain conversation history

        Returns:
            LLMResponse or None if failed
        """
        if not self._connected or not self._session:
            return None

        try:
            url = f"{self.config.base_url}{self.config.chat_endpoint}"

            # Build messages
            messages = []

            # System prompt
            sys_prompt = system_prompt or self.config.system_prompt
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})

            # History
            if keep_history:
                messages.extend([m.to_dict() for m in self._history])

            # Current message
            messages.append({"role": "user", "content": message})

            if self.config.provider == LLMProvider.OLLAMA:
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                }
            else:
                # OpenAI-compatible
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stream": False,
                }

                # Add tools if defined
                if self._tools:
                    payload["tools"] = self._tools

            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"LLM chat error: {error}")
                    return None

                data = await response.json()

                # Parse response
                if self.config.provider == LLMProvider.OLLAMA:
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                else:
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content", "")
                    else:
                        content = ""

                # Update history
                if keep_history:
                    self._history.append(LLMMessage("user", message))
                    self._history.append(LLMMessage("assistant", content))

                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider=self.config.provider,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    raw_response=data,
                )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return None

    async def chat_with_context(
        self,
        message: str,
        devices: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
    ) -> Optional[LLMResponse]:
        """
        Chat with device context - makes LLM aware of available hardware.

        Args:
            message: User message
            devices: Dict of device_id -> device_state
            capabilities: List of available capabilities

        Returns:
            LLMResponse with hardware-aware answer
        """
        # Build context-aware system prompt
        context_parts = [
            "You are a hardware control assistant for RegenNexus UAP.",
            "You can help control connected devices and sensors.",
        ]

        if devices:
            context_parts.append("\nConnected devices:")
            for device_id, state in devices.items():
                context_parts.append(f"  - {device_id}: {json.dumps(state)}")

        if capabilities:
            context_parts.append(f"\nAvailable capabilities: {', '.join(capabilities)}")

        context_parts.append(
            "\nWhen asked to control devices, respond with a JSON action block like:"
            '\n```json\n{"action": "command", "device": "device_id", "params": {...}}\n```'
        )

        system_prompt = "\n".join(context_parts)

        return await self.chat(message, system_prompt=system_prompt)

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> None:
        """
        Register a tool/function for the LLM to call.

        This enables function calling where the LLM can request
        hardware actions.

        Args:
            name: Tool name
            description: What it does
            parameters: JSON schema for parameters
            handler: async def handler(**params) -> result
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }
        self._tools.append(tool_def)
        self._tool_handlers[name] = handler

        logger.info(f"Registered LLM tool: {name}")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @property
    def available_models(self) -> List[str]:
        """Get available models."""
        return self._available_models.copy()

    @property
    def history(self) -> List[LLMMessage]:
        """Get conversation history."""
        return self._history.copy()


class UAP_LLM_Agent:
    """
    Combines LLM with UAP mesh for AI-controlled hardware.

    This is the "MCP for hardware" - connects AI to physical devices.

    Example:
        agent = UAP_LLM_Agent(
            llm_config=LLMConfig(host="192.168.1.100", model="llama2"),
            mesh_config=MeshConfig(node_id="ai-controller")
        )
        await agent.start()

        # AI can now see and control all devices on the network
        response = await agent.process("Turn on the lights in the kitchen")
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mesh_config: Optional[Any] = None,  # MeshConfig
    ):
        """
        Initialize UAP LLM Agent.

        Args:
            llm_config: LLM connection config
            mesh_config: Mesh network config
        """
        self.llm = LLMBridge(llm_config)
        self._mesh = None
        self._mesh_config = mesh_config
        self._device_states: Dict[str, Dict] = {}

    async def start(self) -> bool:
        """Start the agent."""
        # Connect to LLM
        if not await self.llm.connect():
            logger.error("Failed to connect to LLM")
            return False

        # Start mesh if configured
        if self._mesh_config:
            from regennexus.core.mesh import MeshNetwork
            self._mesh = MeshNetwork(self._mesh_config)

            # Register handlers
            self._mesh.on_message(self._handle_device_message)
            self._mesh.on_peer(self._handle_peer_event)

            if not await self._mesh.start():
                logger.warning("Mesh network failed to start")

        # Register hardware control tools
        self._register_hardware_tools()

        logger.info("UAP LLM Agent started")
        return True

    async def stop(self) -> None:
        """Stop the agent."""
        await self.llm.disconnect()
        if self._mesh:
            await self._mesh.stop()

    def _register_hardware_tools(self) -> None:
        """Register tools for hardware control."""
        # List devices tool
        self.llm.register_tool(
            name="list_devices",
            description="List all connected devices and their current state",
            parameters={"type": "object", "properties": {}},
            handler=self._tool_list_devices
        )

        # Send command tool
        self.llm.register_tool(
            name="send_command",
            description="Send a command to a specific device",
            parameters={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Target device ID"},
                    "command": {"type": "string", "description": "Command to send"},
                    "params": {"type": "object", "description": "Command parameters"},
                },
                "required": ["device_id", "command"]
            },
            handler=self._tool_send_command
        )

    async def _tool_list_devices(self) -> Dict[str, Any]:
        """Tool: List connected devices."""
        if self._mesh:
            peers = self._mesh.get_peers()
            return {
                "devices": [
                    {
                        "id": p.node_id,
                        "type": p.entity_type,
                        "capabilities": p.capabilities,
                        "state": self._device_states.get(p.node_id, {}),
                    }
                    for p in peers
                ]
            }
        return {"devices": []}

    async def _tool_send_command(
        self,
        device_id: str,
        command: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Tool: Send command to device."""
        if not self._mesh:
            return {"error": "Mesh not connected"}

        success = await self._mesh.send(device_id, {
            "command": command,
            "params": params or {},
        }, intent="command")

        return {"success": success, "device": device_id, "command": command}

    async def _handle_device_message(self, message: Any) -> None:
        """Handle messages from devices."""
        if hasattr(message, 'content') and isinstance(message.content, dict):
            # Update device state
            if message.sender_id:
                self._device_states[message.sender_id] = message.content

    async def _handle_peer_event(self, peer: Any, event: str) -> None:
        """Handle peer connect/disconnect."""
        if event == "disconnected" or event == "timeout":
            self._device_states.pop(peer.node_id, None)

    async def process(self, user_input: str) -> str:
        """
        Process natural language input and control devices.

        Args:
            user_input: User's natural language request

        Returns:
            AI response
        """
        response = await self.llm.chat_with_context(
            user_input,
            devices=self._device_states,
            capabilities=self._get_all_capabilities(),
        )

        if response:
            return response.content
        return "I'm sorry, I couldn't process that request."

    def _get_all_capabilities(self) -> List[str]:
        """Get all capabilities from connected devices."""
        caps = set()
        if self._mesh:
            for peer in self._mesh.get_peers():
                caps.update(peer.capabilities)
        return list(caps)
