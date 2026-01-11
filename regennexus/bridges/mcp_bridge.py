"""
RegenNexus UAP - MCP (Model Context Protocol) Bridge

Full MCP server implementation that exposes hardware as AI tools.
Any MCP-compatible client (Claude Desktop, etc.) can control devices.

This bridge makes RegenNexus a "hardware provider" for the MCP ecosystem:
- Devices become MCP tools
- Sensors become MCP resources
- Events become MCP notifications

Architecture:
    [Claude/LLM] <--MCP--> [RegenNexus MCP Server] <--UAP--> [Hardware]

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import sys

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types."""
    # Requests
    INITIALIZE = "initialize"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"

    # Notifications
    INITIALIZED = "notifications/initialized"
    PROGRESS = "notifications/progress"
    RESOURCE_UPDATED = "notifications/resources/updated"

    # Responses
    RESULT = "result"
    ERROR = "error"


@dataclass
class MCPTool:
    """MCP Tool definition (hardware action)."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """MCP Resource definition (sensor/state)."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    handler: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPrompt:
    """MCP Prompt template."""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


class MCPServer:
    """
    MCP Server for RegenNexus UAP.

    Exposes hardware devices as MCP tools and resources, allowing
    any MCP-compatible AI client to control physical devices.

    Example:
        server = MCPServer()

        # Register a device as a tool
        server.register_tool(MCPTool(
            name="control_light",
            description="Turn a light on or off",
            input_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "state": {"type": "boolean"}
                },
                "required": ["device_id", "state"]
            },
            handler=light_control_handler
        ))

        # Register a sensor as a resource
        server.register_resource(MCPResource(
            uri="sensor://temperature/living_room",
            name="Living Room Temperature",
            description="Current temperature reading",
            handler=get_temperature
        ))

        # Run server (stdio for Claude Desktop)
        await server.run_stdio()
    """

    def __init__(
        self,
        name: str = "regennexus-hardware",
        version: str = "0.2.5",
    ):
        """
        Initialize MCP server.

        Args:
            name: Server name
            version: Server version
        """
        self.name = name
        self.version = version

        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}

        self._initialized = False
        self._request_id = 0

        # Mesh connection for device discovery
        self._mesh = None

    def register_tool(self, tool: MCPTool) -> None:
        """Register a hardware control tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")

    def register_resource(self, resource: MCPResource) -> None:
        """Register a sensor/state resource."""
        self._resources[resource.uri] = resource
        logger.info(f"Registered MCP resource: {resource.uri}")

    def register_prompt(self, prompt: MCPPrompt) -> None:
        """Register a prompt template."""
        self._prompts[prompt.name] = prompt

    async def connect_mesh(self, mesh_config: Any = None) -> bool:
        """
        Connect to UAP mesh network for device discovery.

        Args:
            mesh_config: MeshConfig instance

        Returns:
            True if connected
        """
        try:
            from regennexus.core.mesh import MeshNetwork, MeshConfig

            config = mesh_config or MeshConfig(
                node_id="mcp-server",
                entity_type="mcp_server",
                capabilities=["mcp", "hardware_control"],
            )

            self._mesh = MeshNetwork(config)

            # Auto-register devices as tools/resources
            self._mesh.on_peer(self._handle_peer_event)
            self._mesh.on_message(self._handle_device_message)

            if await self._mesh.start():
                logger.info("MCP Server connected to mesh network")
                return True

            return False

        except Exception as e:
            logger.error(f"Mesh connection error: {e}")
            return False

    async def _handle_peer_event(self, peer: Any, event: str) -> None:
        """Handle device connect/disconnect."""
        if event == "connected":
            # Auto-register device capabilities as tools
            for cap in peer.capabilities:
                tool_name = f"{peer.node_id}_{cap}"
                if tool_name not in self._tools:
                    self.register_tool(MCPTool(
                        name=tool_name,
                        description=f"{cap} on {peer.node_id}",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "params": {"type": "object"}
                            }
                        },
                        handler=lambda params, node=peer.node_id, c=cap: self._send_to_device(node, c, params)
                    ))

            # Register device state as resource
            self.register_resource(MCPResource(
                uri=f"device://{peer.node_id}/state",
                name=f"{peer.node_id} State",
                description=f"Current state of {peer.entity_type} device",
                handler=lambda node=peer.node_id: self._get_device_state(node)
            ))

    async def _handle_device_message(self, message: Any) -> None:
        """Handle messages from devices."""
        # Could emit MCP notifications here
        pass

    async def _send_to_device(self, node_id: str, capability: str, params: Dict) -> Dict:
        """Send command to device via mesh."""
        if not self._mesh:
            return {"error": "Mesh not connected"}

        await self._mesh.send(node_id, {
            "capability": capability,
            "action": params.get("action"),
            "params": params.get("params", {}),
        }, intent="command")

        return {"success": True, "device": node_id}

    async def _get_device_state(self, node_id: str) -> Dict:
        """Get device state."""
        if self._mesh:
            peer = self._mesh.get_peer(node_id)
            if peer:
                return {
                    "device_id": peer.node_id,
                    "type": peer.entity_type,
                    "capabilities": peer.capabilities,
                    "last_seen": peer.last_seen,
                }
        return {"error": "Device not found"}

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming MCP message.

        Args:
            message: MCP JSON-RPC message

        Returns:
            Response message or None (for notifications)
        """
        method = message.get("method", "")
        params = message.get("params", {})
        msg_id = message.get("id")

        # MCP notifications don't have an id - don't respond to them
        # Per JSON-RPC 2.0 spec: notifications have no id field
        is_notification = "id" not in message

        try:
            # Handle notifications (no response expected)
            if method == "notifications/initialized":
                logger.info("Client initialized notification received")
                return None  # Don't respond to notifications

            if method == "notifications/cancelled":
                logger.info("Request cancelled notification received")
                return None

            # If it's a notification we don't recognize, just ignore it
            if is_notification:
                logger.debug(f"Ignoring unknown notification: {method}")
                return None

            # Handle requests (response expected)
            if method == "initialize":
                return await self._handle_initialize(msg_id, params)

            elif method == "tools/list":
                return await self._handle_list_tools(msg_id)

            elif method == "tools/call":
                return await self._handle_call_tool(msg_id, params)

            elif method == "resources/list":
                return await self._handle_list_resources(msg_id)

            elif method == "resources/read":
                return await self._handle_read_resource(msg_id, params)

            elif method == "prompts/list":
                return await self._handle_list_prompts(msg_id)

            elif method == "prompts/get":
                return await self._handle_get_prompt(msg_id, params)

            # Handle ping for keepalive
            elif method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            else:
                logger.warning(f"Unknown method: {method}")
                return self._error_response(msg_id, -32601, f"Method not found: {method}")

        except Exception as e:
            logger.error(f"Error handling {method}: {e}")
            if is_notification:
                return None  # Don't respond to failed notifications
            return self._error_response(msg_id, -32603, str(e))

    async def _handle_initialize(self, msg_id: Any, params: Dict) -> Dict:
        """Handle initialize request."""
        self._initialized = True

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {},
                    "resources": {"subscribe": True},
                    "prompts": {},
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                }
            }
        }

    async def _handle_list_tools(self, msg_id: Any) -> Dict:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [tool.to_dict() for tool in self._tools.values()]
            }
        }

    async def _handle_call_tool(self, msg_id: Any, params: Dict) -> Dict:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            return self._error_response(msg_id, -32602, f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]

        try:
            if tool.handler:
                result = tool.handler(arguments)
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                result = {"error": "No handler for tool"}

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result) if isinstance(result, dict) else str(result)
                        }
                    ]
                }
            }

        except Exception as e:
            return self._error_response(msg_id, -32603, str(e))

    async def _handle_list_resources(self, msg_id: Any) -> Dict:
        """Handle resources/list request."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "resources": [res.to_dict() for res in self._resources.values()]
            }
        }

    async def _handle_read_resource(self, msg_id: Any, params: Dict) -> Dict:
        """Handle resources/read request."""
        uri = params.get("uri")

        if uri not in self._resources:
            return self._error_response(msg_id, -32602, f"Unknown resource: {uri}")

        resource = self._resources[uri]

        try:
            if resource.handler:
                content = resource.handler()
                if asyncio.iscoroutine(content):
                    content = await content
            else:
                content = {}

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": resource.mime_type,
                            "text": json.dumps(content) if isinstance(content, dict) else str(content)
                        }
                    ]
                }
            }

        except Exception as e:
            return self._error_response(msg_id, -32603, str(e))

    async def _handle_list_prompts(self, msg_id: Any) -> Dict:
        """Handle prompts/list request."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "prompts": [p.to_dict() for p in self._prompts.values()]
            }
        }

    async def _handle_get_prompt(self, msg_id: Any, params: Dict) -> Dict:
        """Handle prompts/get request."""
        name = params.get("name")

        if name not in self._prompts:
            return self._error_response(msg_id, -32602, f"Unknown prompt: {name}")

        prompt = self._prompts[name]

        # Build prompt messages
        messages = [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"Use the {name} prompt for hardware control."
                }
            }
        ]

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "description": prompt.description,
                "messages": messages
            }
        }

    def _error_response(self, msg_id: Any, code: int, message: str) -> Dict:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    async def run_stdio(self) -> None:
        """
        Run MCP server over stdio (for Claude Desktop integration).

        This is the standard way to run an MCP server that can be
        configured in Claude Desktop's settings.

        Uses thread-based I/O for Windows compatibility.
        """
        logger.info("Starting MCP server on stdio")

        import threading
        import queue

        # Use thread-based stdin reading for Windows compatibility
        input_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()

        def stdin_reader():
            """Read stdin in a separate thread (Windows compatible)."""
            try:
                # Try to set stdin to binary mode on Windows (may fail with IPC)
                if sys.platform == 'win32':
                    try:
                        import msvcrt
                        msvcrt.setmode(sys.stdin.fileno(), 0)  # O_BINARY = 0
                    except OSError:
                        pass  # Skip if handle doesn't support binary mode

                while not stop_event.is_set():
                    try:
                        line = sys.stdin.readline()
                        if line:
                            input_queue.put(line)
                        elif line == '':
                            # EOF
                            input_queue.put(None)
                            break
                    except Exception as e:
                        logger.error(f"Stdin read error: {e}")
                        input_queue.put(None)
                        break
            except Exception as e:
                logger.error(f"Stdin reader thread error: {e}")
                input_queue.put(None)

        # Start stdin reader thread
        reader_thread = threading.Thread(target=stdin_reader, daemon=True)
        reader_thread.start()

        # Try to set stdout to binary mode on Windows (may fail with IPC)
        if sys.platform == 'win32':
            try:
                import msvcrt
                msvcrt.setmode(sys.stdout.fileno(), 0)  # O_BINARY = 0
            except OSError:
                pass  # Skip if handle doesn't support binary mode

        try:
            while True:
                try:
                    # Check for input with timeout to allow clean shutdown
                    try:
                        line = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: input_queue.get(timeout=0.1)
                        )
                    except queue.Empty:
                        continue

                    if line is None:
                        # EOF received
                        break

                    line = line.strip()
                    if not line:
                        continue

                    message = json.loads(line)
                    response = await self.handle_message(message)

                    if response:
                        response_line = json.dumps(response) + "\n"
                        sys.stdout.write(response_line)
                        sys.stdout.flush()

                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Stdio error: {e}")
                    break
        finally:
            stop_event.set()

    async def run_websocket(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Run MCP server over WebSocket (for LAN access).

        This allows LLMs on other machines to connect.

        Args:
            host: Bind host
            port: Bind port
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets required for WebSocket server")
            return

        async def handle_connection(websocket, path):
            logger.info(f"MCP client connected from {websocket.remote_address}")

            try:
                async for message_str in websocket:
                    message = json.loads(message_str)
                    response = await self.handle_message(message)

                    if response:
                        await websocket.send(json.dumps(response))

            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        logger.info(f"Starting MCP WebSocket server on ws://{host}:{port}")
        async with websockets.serve(handle_connection, host, port):
            await asyncio.Future()  # Run forever


def create_hardware_mcp_server() -> MCPServer:
    """
    Create an MCP server pre-configured with common hardware tools.

    Returns:
        Configured MCPServer instance
    """
    server = MCPServer()

    # GPIO control tool
    server.register_tool(MCPTool(
        name="gpio_write",
        description="Set a GPIO pin to HIGH (1) or LOW (0)",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (e.g., 'raspi-001')"
                },
                "pin": {
                    "type": "integer",
                    "description": "GPIO pin number"
                },
                "value": {
                    "type": "integer",
                    "enum": [0, 1],
                    "description": "Pin value (0=LOW, 1=HIGH)"
                }
            },
            "required": ["device_id", "pin", "value"]
        }
    ))

    # Arm control tool
    server.register_tool(MCPTool(
        name="robot_arm_move",
        description="Move a robotic arm to specified joint positions",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Arm device ID"
                },
                "positions": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Joint positions in degrees (7 values for 7-DOF)"
                },
                "duration": {
                    "type": "number",
                    "description": "Move duration in seconds"
                }
            },
            "required": ["device_id", "positions"]
        }
    ))

    # Gripper control
    server.register_tool(MCPTool(
        name="gripper_control",
        description="Open or close a robotic gripper",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["open", "close"],
                    "description": "Gripper action"
                },
                "force": {
                    "type": "number",
                    "description": "Grip force in Newtons (for close)"
                }
            },
            "required": ["device_id", "action"]
        }
    ))

    # Sensor read tool
    server.register_tool(MCPTool(
        name="read_sensor",
        description="Read value from a sensor",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "sensor_type": {
                    "type": "string",
                    "enum": ["temperature", "humidity", "distance", "light", "pressure"],
                    "description": "Type of sensor to read"
                }
            },
            "required": ["device_id", "sensor_type"]
        }
    ))

    # List devices tool
    server.register_tool(MCPTool(
        name="list_devices",
        description="List all connected hardware devices",
        input_schema={
            "type": "object",
            "properties": {}
        }
    ))

    # =========================================================================
    # GPIO Extended Tools
    # =========================================================================

    # GPIO read tool
    server.register_tool(MCPTool(
        name="gpio_read",
        description="Read the current state of a GPIO pin (HIGH=1 or LOW=0)",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (e.g., 'raspi-001')"
                },
                "pin": {
                    "type": "integer",
                    "description": "GPIO pin number to read"
                }
            },
            "required": ["device_id", "pin"]
        }
    ))

    # PWM write tool
    server.register_tool(MCPTool(
        name="pwm_write",
        description="Set PWM duty cycle on a pin (0-100%) for motors, LEDs, servos",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (e.g., 'raspi-001')"
                },
                "pin": {
                    "type": "integer",
                    "description": "PWM-capable pin number"
                },
                "duty_cycle": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Duty cycle percentage (0-100)"
                },
                "frequency": {
                    "type": "number",
                    "description": "PWM frequency in Hz (optional, default varies by device)"
                }
            },
            "required": ["device_id", "pin", "duty_cycle"]
        }
    ))

    # =========================================================================
    # I2C/Serial Tools
    # =========================================================================

    # I2C scan tool
    server.register_tool(MCPTool(
        name="i2c_scan",
        description="Scan I2C bus for connected devices and return their addresses",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (e.g., 'raspi-001')"
                },
                "bus": {
                    "type": "integer",
                    "description": "I2C bus number (default: 1)"
                }
            },
            "required": ["device_id"]
        }
    ))

    # Serial send tool
    server.register_tool(MCPTool(
        name="serial_send",
        description="Send data over serial port (UART)",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID"
                },
                "port": {
                    "type": "string",
                    "description": "Serial port (e.g., '/dev/ttyUSB0', 'COM3')"
                },
                "data": {
                    "type": "string",
                    "description": "Data to send (string or hex bytes)"
                },
                "baudrate": {
                    "type": "integer",
                    "description": "Baud rate (default: 9600)"
                }
            },
            "required": ["device_id", "port", "data"]
        }
    ))

    # Serial read tool
    server.register_tool(MCPTool(
        name="serial_read",
        description="Read data from serial port (UART)",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID"
                },
                "port": {
                    "type": "string",
                    "description": "Serial port (e.g., '/dev/ttyUSB0', 'COM3')"
                },
                "bytes": {
                    "type": "integer",
                    "description": "Number of bytes to read (default: 256)"
                },
                "timeout": {
                    "type": "number",
                    "description": "Read timeout in seconds (default: 1.0)"
                },
                "baudrate": {
                    "type": "integer",
                    "description": "Baud rate (default: 9600)"
                }
            },
            "required": ["device_id", "port"]
        }
    ))

    # =========================================================================
    # Device Info Tools
    # =========================================================================

    # Device info tool
    server.register_tool(MCPTool(
        name="device_info",
        description="Get detailed device information (CPU, memory, IP, temperature)",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID (e.g., 'raspi-001', 'jetson-001')"
                }
            },
            "required": ["device_id"]
        }
    ))

    # =========================================================================
    # Camera Tools
    # =========================================================================

    # Camera capture tool
    server.register_tool(MCPTool(
        name="camera_capture",
        description="Capture a single image from a camera",
        input_schema={
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Device ID with camera"
                },
                "camera_id": {
                    "type": "integer",
                    "description": "Camera index (default: 0)"
                },
                "resolution": {
                    "type": "string",
                    "enum": ["640x480", "1280x720", "1920x1080"],
                    "description": "Image resolution"
                },
                "format": {
                    "type": "string",
                    "enum": ["jpeg", "png", "base64"],
                    "description": "Output format (default: jpeg)"
                }
            },
            "required": ["device_id"]
        }
    ))

    # =========================================================================
    # Mesh Network Tools
    # =========================================================================

    # List mesh nodes tool
    server.register_tool(MCPTool(
        name="list_nodes",
        description="List all nodes in the mesh network (Raspberry Pi, Jetson, robots, etc.)",
        input_schema={
            "type": "object",
            "properties": {
                "filter_type": {
                    "type": "string",
                    "description": "Filter by entity type (e.g., 'device', 'robot', 'sensor')"
                }
            }
        }
    ))

    # Ping mesh node tool
    server.register_tool(MCPTool(
        name="ping_node",
        description="Ping a mesh node and measure network latency",
        input_schema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Target node ID (e.g., 'raspi-001', 'jetson-001')"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 5.0)"
                }
            },
            "required": ["node_id"]
        }
    ))

    # Send to node tool
    server.register_tool(MCPTool(
        name="send_to_node",
        description="Send a message/command to a specific mesh node",
        input_schema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Target node ID"
                },
                "intent": {
                    "type": "string",
                    "description": "Message intent (e.g., 'command', 'query', 'event')"
                },
                "content": {
                    "type": "object",
                    "description": "Message content/payload"
                }
            },
            "required": ["node_id", "intent", "content"]
        }
    ))

    # Broadcast message tool
    server.register_tool(MCPTool(
        name="broadcast_message",
        description="Broadcast a message to all nodes in the mesh network",
        input_schema={
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Message intent (e.g., 'announcement', 'sync', 'alert')"
                },
                "content": {
                    "type": "object",
                    "description": "Message content/payload"
                }
            },
            "required": ["intent", "content"]
        }
    ))

    # Find by capability tool
    server.register_tool(MCPTool(
        name="find_by_capability",
        description="Find mesh nodes that have a specific capability",
        input_schema={
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "Capability to search for (e.g., 'gpio', 'camera', 'arm', 'gripper')"
                }
            },
            "required": ["capability"]
        }
    ))

    # Hardware control prompt
    server.register_prompt(MCPPrompt(
        name="hardware_assistant",
        description="A prompt for AI hardware control assistant",
        arguments=[
            {
                "name": "task",
                "description": "What you want to accomplish with the hardware",
                "required": True
            }
        ]
    ))

    return server
