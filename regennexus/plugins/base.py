"""
RegenNexus UAP - Plugin Base Interface

Base interface for device plugins. All device plugins should inherit
from DevicePlugin class.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set

from regennexus.__version__ import __version__

logger = logging.getLogger(__name__)


class DevicePlugin(ABC):
    """
    Base class for RegenNexus device plugins.

    All device plugins should inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        entity_id: str,
        device_type: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the device plugin.

        Args:
            entity_id: Unique identifier for this device entity
            device_type: Type of device (e.g., 'raspberry_pi', 'arduino')
            protocol: Optional protocol instance for message handling
            mock_mode: If True, simulate device without real hardware
        """
        self.entity_id = entity_id
        self.device_type = device_type
        self.protocol = protocol
        self.mock_mode = mock_mode
        self.initialized = False
        self.capabilities: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.command_handlers: Dict[str, Callable] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}
        self._running = False

    async def initialize(self) -> bool:
        """
        Initialize the device plugin.

        Returns:
            True if initialization successful
        """
        try:
            # Register basic capabilities
            self.capabilities = [
                "device",
                self.device_type,
                "status",
                "command",
            ]

            # Set basic metadata
            self.metadata = {
                "device_type": self.device_type,
                "version": __version__,
                "status": "initializing",
                "mock_mode": self.mock_mode,
            }

            # Register default command handlers
            self.register_command_handler("status", self._handle_status_command)
            self.register_command_handler("capabilities", self._handle_capabilities_command)
            self.register_command_handler("ping", self._handle_ping_command)

            # Device-specific initialization
            if not await self._device_init():
                return False

            # Register with protocol if available
            if self.protocol:
                self.protocol.register_message_handler(
                    self.entity_id,
                    self._handle_message
                )
                if hasattr(self.protocol, 'registry') and self.protocol.registry:
                    await self.protocol.registry.register_entity(
                        entity_id=self.entity_id,
                        entity_type="device",
                        capabilities=self.capabilities,
                        metadata=self.metadata,
                    )

            self.initialized = True
            self._running = True
            self.metadata["status"] = "ready"
            logger.info(
                f"Initialized {self.device_type} plugin: {self.entity_id}"
                f"{' (mock mode)' if self.mock_mode else ''}"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.device_type} plugin: {e}")
            self.metadata["status"] = "error"
            self.metadata["error"] = str(e)
            return False

    async def shutdown(self) -> bool:
        """
        Shut down the device plugin.

        Returns:
            True if shutdown successful
        """
        try:
            self._running = False
            self.metadata["status"] = "shutting_down"

            # Device-specific shutdown
            await self._device_shutdown()

            # Unregister from protocol
            if self.protocol:
                self.protocol.unregister_message_handler(
                    self.entity_id,
                    self._handle_message
                )
                if hasattr(self.protocol, 'registry') and self.protocol.registry:
                    await self.protocol.registry.unregister_entity(self.entity_id)

            self.initialized = False
            logger.info(f"Shut down {self.device_type} plugin: {self.entity_id}")
            return True

        except Exception as e:
            logger.error(f"Error shutting down {self.device_type} plugin: {e}")
            return False

    @abstractmethod
    async def _device_init(self) -> bool:
        """
        Device-specific initialization.

        Implement this method in subclasses to handle device-specific
        initialization (GPIO setup, serial connection, etc.).

        Returns:
            True if initialization successful
        """
        pass

    async def _device_shutdown(self) -> None:
        """
        Device-specific shutdown.

        Override this method in subclasses to handle device-specific
        cleanup (GPIO cleanup, close connections, etc.).
        """
        pass

    def register_command_handler(self, command: str, handler: Callable) -> None:
        """
        Register a handler for a specific command.

        Args:
            command: Command name
            handler: Async function that handles the command
        """
        self.command_handlers[command] = handler

        # Add to capabilities
        capability = f"command.{command}"
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def unregister_command_handler(self, command: str) -> bool:
        """Remove a command handler."""
        if command in self.command_handlers:
            del self.command_handlers[command]
            capability = f"command.{command}"
            if capability in self.capabilities:
                self.capabilities.remove(capability)
            return True
        return False

    def register_event_listener(self, event_type: str, listener: Callable) -> None:
        """Register a listener for an event type."""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)

        capability = f"event.{event_type}"
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def unregister_event_listener(self, event_type: str, listener: Callable) -> bool:
        """Remove an event listener."""
        if event_type in self.event_listeners and listener in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(listener)
            if not self.event_listeners[event_type]:
                del self.event_listeners[event_type]
                capability = f"event.{event_type}"
                if capability in self.capabilities:
                    self.capabilities.remove(capability)
            return True
        return False

    async def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Emit an event to listeners and via protocol.

        Args:
            event_type: Event type name
            event_data: Event payload

        Returns:
            True if event emitted successfully
        """
        try:
            # Call local listeners
            if event_type in self.event_listeners:
                for listener in self.event_listeners[event_type]:
                    try:
                        result = listener(event_data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Event listener error: {e}")

            # Send via protocol
            if self.protocol:
                await self.protocol.send_message(
                    sender=self.entity_id,
                    recipient="*",
                    intent=f"event.{event_type}",
                    payload=event_data,
                )

            return True

        except Exception as e:
            logger.error(f"Error emitting event: {e}")
            return False

    async def execute_command(
        self,
        command: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a command on the device.

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Command result dictionary
        """
        params = params or {}

        if command not in self.command_handlers:
            return {
                "success": False,
                "error": f"Unsupported command: {command}",
            }

        try:
            handler = self.command_handlers[command]
            result = handler(params)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming messages from protocol."""
        try:
            recipient = message.get("recipient", message.get("target"))
            if recipient != self.entity_id and recipient != "*":
                return

            intent = message.get("intent", "")
            payload = message.get("payload", message.get("content", {}))

            if intent == "command":
                command = payload.get("command", "")
                params = payload.get("params", {})
                result = await self.execute_command(command, params)

                # Send response
                if self.protocol and recipient != "*":
                    sender = message.get("sender", message.get("source"))
                    if sender:
                        await self.protocol.send_message(
                            sender=self.entity_id,
                            recipient=sender,
                            intent="command_result",
                            payload={"command": command, "result": result},
                        )

        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _handle_status_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status command."""
        return {
            "success": True,
            "entity_id": self.entity_id,
            "device_type": self.device_type,
            "status": self.metadata.get("status", "unknown"),
            "mock_mode": self.mock_mode,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }

    async def _handle_capabilities_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle capabilities command."""
        return {
            "success": True,
            "capabilities": self.capabilities,
        }

    async def _handle_ping_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping command."""
        return {
            "success": True,
            "pong": True,
            "entity_id": self.entity_id,
        }

    @property
    def is_running(self) -> bool:
        """Check if plugin is running."""
        return self._running and self.initialized


class MockDeviceMixin:
    """
    Mixin class for mock device support.

    Provides common mock functionality for device plugins.
    """

    def __init__(self):
        self._mock_state: Dict[str, Any] = {}
        self._mock_pins: Dict[int, int] = {}

    def mock_set_pin(self, pin: int, value: int) -> None:
        """Set mock pin value."""
        self._mock_pins[pin] = value

    def mock_get_pin(self, pin: int) -> int:
        """Get mock pin value."""
        return self._mock_pins.get(pin, 0)

    def mock_set_state(self, key: str, value: Any) -> None:
        """Set mock state value."""
        self._mock_state[key] = value

    def mock_get_state(self, key: str, default: Any = None) -> Any:
        """Get mock state value."""
        return self._mock_state.get(key, default)
