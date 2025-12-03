"""
RegenNexus UAP - Entity Module

Defines the Entity base class for all protocol participants.

Copyright (c) 2024 ReGen Designs LLC
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from abc import ABC, abstractmethod

from regennexus.core.message import Message, Intent

logger = logging.getLogger(__name__)


class Entity:
    """
    Base class for entities in the RegenNexus UAP protocol.

    Entities are the primary actors in the system. They can send/receive
    messages and have capabilities that define what they can do.

    Attributes:
        id: Unique entity identifier
        name: Human-readable name
        entity_type: Type of entity (e.g., "device", "service", "agent")
        capabilities: List of capabilities this entity supports
        metadata: Additional entity information

    Example:
        class MyDevice(Entity):
            def __init__(self):
                super().__init__(
                    entity_id="my-device-001",
                    name="My Smart Device",
                    entity_type="device"
                )
                self.add_capability("sensor.temperature")
                self.add_capability("control.switch")

            async def on_message(self, message: Message) -> Optional[Message]:
                if message.intent == Intent.QUERY:
                    return message.create_response({"temperature": 23.5})
                return None
    """

    def __init__(
        self,
        entity_id: str,
        name: Optional[str] = None,
        entity_type: str = "entity",
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new entity.

        Args:
            entity_id: Unique identifier for this entity
            name: Human-readable name (defaults to entity_id)
            entity_type: Type of entity
            capabilities: Initial list of capabilities
            metadata: Additional metadata
        """
        self.id = entity_id
        self.name = name or entity_id
        self.entity_type = entity_type
        self.capabilities: Set[str] = set(capabilities or [])
        self.metadata: Dict[str, Any] = metadata or {}

        # Internal state
        self._message_handlers: List[Callable] = []
        self._running = False
        self._registry = None

    # =========================================================================
    # Capability Management
    # =========================================================================

    def add_capability(self, capability: str) -> None:
        """
        Add a capability to this entity.

        Args:
            capability: Capability identifier (e.g., "sensor.temperature")
        """
        self.capabilities.add(capability)
        logger.debug(f"Entity {self.id}: Added capability '{capability}'")

    def remove_capability(self, capability: str) -> None:
        """
        Remove a capability from this entity.

        Args:
            capability: Capability identifier
        """
        self.capabilities.discard(capability)
        logger.debug(f"Entity {self.id}: Removed capability '{capability}'")

    def has_capability(self, capability: str) -> bool:
        """
        Check if this entity has a specific capability.

        Args:
            capability: Capability identifier

        Returns:
            True if entity has the capability
        """
        # Check for exact match
        if capability in self.capabilities:
            return True

        # Check for wildcard match (e.g., "sensor.*" matches "sensor.temperature")
        for cap in self.capabilities:
            if cap.endswith(".*"):
                prefix = cap[:-2]
                if capability.startswith(prefix):
                    return True

        return False

    def get_capabilities(self) -> List[str]:
        """
        Get all capabilities of this entity.

        Returns:
            List of capability identifiers
        """
        return list(self.capabilities)

    # =========================================================================
    # Message Handling
    # =========================================================================

    def register_message_handler(
        self, handler: Callable[[Message], Optional[Message]]
    ) -> None:
        """
        Register a message handler function.

        Args:
            handler: Async function that processes messages
        """
        self._message_handlers.append(handler)

    def unregister_message_handler(
        self, handler: Callable[[Message], Optional[Message]]
    ) -> None:
        """
        Unregister a message handler function.

        Args:
            handler: Handler function to remove
        """
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)

    async def process_message(
        self, message: Message, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Message]:
        """
        Process an incoming message.

        This method first calls all registered handlers, then calls
        the on_message method for subclass handling.

        Args:
            message: The message to process
            context: Optional context information

        Returns:
            Optional response message
        """
        ctx = context or {}

        # Call registered handlers
        for handler in self._message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message, ctx)
                else:
                    result = handler(message, ctx)

                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"Entity {self.id}: Error in message handler: {e}")

        # Call subclass handler
        try:
            return await self.on_message(message, ctx)
        except Exception as e:
            logger.error(f"Entity {self.id}: Error in on_message: {e}")
            return None

    async def on_message(
        self, message: Message, context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Handle an incoming message. Override in subclass.

        Args:
            message: The message to handle
            context: Context information

        Returns:
            Optional response message
        """
        # Default implementation does nothing
        return None

    # =========================================================================
    # Message Sending
    # =========================================================================

    async def send_message(
        self,
        recipient_id: str,
        content: Any,
        intent: str = Intent.MESSAGE,
        context_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Create and send a message.

        Args:
            recipient_id: Target entity ID (or '*' for broadcast)
            content: Message content
            intent: Message intent
            context_id: Context identifier
            metadata: Additional metadata

        Returns:
            The created message
        """
        message = Message(
            sender_id=self.id,
            recipient_id=recipient_id,
            content=content,
            intent=intent,
            context_id=context_id,
            metadata=metadata or {},
        )

        # If registered with a registry, route through it
        if self._registry:
            await self._registry.route_message(message)

        return message

    async def broadcast(
        self,
        content: Any,
        intent: str = Intent.EVENT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Broadcast a message to all entities.

        Args:
            content: Message content
            intent: Message intent
            metadata: Additional metadata

        Returns:
            The broadcast message
        """
        return await self.send_message(
            recipient_id="*",
            content=content,
            intent=intent,
            metadata=metadata,
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the entity.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning(f"Entity {self.id}: Already running")
            return True

        try:
            await self.on_start()
            self._running = True
            logger.info(f"Entity {self.id}: Started")
            return True
        except Exception as e:
            logger.error(f"Entity {self.id}: Failed to start: {e}")
            return False

    async def stop(self) -> bool:
        """
        Stop the entity.

        Returns:
            True if stopped successfully
        """
        if not self._running:
            logger.warning(f"Entity {self.id}: Not running")
            return True

        try:
            await self.on_stop()
            self._running = False
            logger.info(f"Entity {self.id}: Stopped")
            return True
        except Exception as e:
            logger.error(f"Entity {self.id}: Failed to stop: {e}")
            return False

    async def on_start(self) -> None:
        """Called when entity starts. Override in subclass."""
        pass

    async def on_stop(self) -> None:
        """Called when entity stops. Override in subclass."""
        pass

    @property
    def is_running(self) -> bool:
        """Check if entity is running."""
        return self._running

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary representation.

        Returns:
            Dictionary containing entity info
        """
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "capabilities": list(self.capabilities),
            "metadata": self.metadata,
            "running": self._running,
        }

    def __str__(self) -> str:
        """String representation."""
        return f"Entity({self.id}, type={self.entity_type})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Entity(id={self.id!r}, name={self.name!r}, "
            f"type={self.entity_type!r}, capabilities={list(self.capabilities)!r})"
        )
