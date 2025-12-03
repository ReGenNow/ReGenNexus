"""
RegenNexus UAP - Registry Module

Entity registry for managing and routing between entities.

Copyright (c) 2024 ReGen Designs LLC
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Set

from regennexus.core.message import Message, Intent
from regennexus.core.entity import Entity
from regennexus.core.context import ContextManager, get_context_manager

logger = logging.getLogger(__name__)


class Registry:
    """
    Entity registry for RegenNexus UAP.

    The registry manages all registered entities and routes messages between them.
    It supports both central (server-based) and local (in-process) modes.

    Example:
        registry = Registry()

        # Register entities
        await registry.register(my_device)
        await registry.register(my_service)

        # Find entities
        devices = await registry.find_by_type("device")
        sensors = await registry.find_by_capability("sensor.temperature")

        # Route messages
        await registry.route_message(message)
    """

    def __init__(self):
        """Initialize the registry."""
        self._entities: Dict[str, Entity] = {}
        self._context_manager = get_context_manager()
        self._message_handlers: List[Callable] = []
        self._connection_handlers: List[Callable] = []
        self._disconnection_handlers: List[Callable] = []

    # =========================================================================
    # Entity Registration
    # =========================================================================

    async def register(self, entity: Entity) -> bool:
        """
        Register an entity with the registry.

        Args:
            entity: Entity to register

        Returns:
            True if registered successfully
        """
        if entity.id in self._entities:
            logger.warning(f"Entity {entity.id} already registered")
            return False

        self._entities[entity.id] = entity
        entity._registry = self

        # Notify handlers
        for handler in self._connection_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(entity.id, entity)
                else:
                    handler(entity.id, entity)
            except Exception as e:
                logger.error(f"Error in connection handler: {e}")

        logger.info(f"Registered entity: {entity.id}")
        return True

    async def unregister(self, entity_id: str) -> bool:
        """
        Unregister an entity from the registry.

        Args:
            entity_id: Entity identifier

        Returns:
            True if unregistered successfully
        """
        if entity_id not in self._entities:
            logger.warning(f"Entity {entity_id} not found")
            return False

        entity = self._entities[entity_id]
        entity._registry = None
        del self._entities[entity_id]

        # Notify handlers
        for handler in self._disconnection_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(entity_id)
                else:
                    handler(entity_id)
            except Exception as e:
                logger.error(f"Error in disconnection handler: {e}")

        logger.info(f"Unregistered entity: {entity_id}")
        return True

    # =========================================================================
    # Entity Lookup
    # =========================================================================

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity or None if not found
        """
        return self._entities.get(entity_id)

    async def find_by_type(self, entity_type: str) -> List[Entity]:
        """
        Find entities by type.

        Args:
            entity_type: Type to search for

        Returns:
            List of matching entities
        """
        return [
            entity
            for entity in self._entities.values()
            if entity.entity_type == entity_type
        ]

    async def find_by_capability(self, capability: str) -> List[Entity]:
        """
        Find entities by capability.

        Args:
            capability: Capability to search for

        Returns:
            List of entities with the capability
        """
        return [
            entity
            for entity in self._entities.values()
            if entity.has_capability(capability)
        ]

    async def find_entities(
        self,
        entity_type: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> List[Entity]:
        """
        Find entities matching criteria.

        Args:
            entity_type: Optional type filter
            capability: Optional capability filter

        Returns:
            List of matching entities
        """
        results = list(self._entities.values())

        if entity_type:
            results = [e for e in results if e.entity_type == entity_type]

        if capability:
            results = [e for e in results if e.has_capability(capability)]

        return results

    async def list_entities(self) -> List[Dict[str, Any]]:
        """
        List all registered entities.

        Returns:
            List of entity info dictionaries
        """
        return [entity.to_dict() for entity in self._entities.values()]

    def get_entity_count(self) -> int:
        """Get the number of registered entities."""
        return len(self._entities)

    def has_entity(self, entity_id: str) -> bool:
        """Check if an entity is registered."""
        return entity_id in self._entities

    # =========================================================================
    # Message Routing
    # =========================================================================

    async def route_message(
        self,
        message: Message,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """
        Route a message to its recipient(s).

        Args:
            message: Message to route
            context: Optional context data

        Returns:
            Response message if any
        """
        ctx = context or {}

        # Add to context manager
        if message.context_id:
            await self._context_manager.add_message(message.context_id, message)

        # Check if message is expired
        if message.is_expired():
            logger.warning(f"Message {message.id} has expired, dropping")
            return None

        # Handle broadcast
        if message.is_broadcast():
            await self._broadcast_message(message, ctx)
            return None

        # Find recipient
        recipient = self._entities.get(message.recipient_id)
        if recipient is None:
            logger.warning(f"Recipient not found: {message.recipient_id}")
            return self._create_error_response(
                message, f"Recipient not found: {message.recipient_id}"
            )

        # Deliver message
        try:
            response = await recipient.process_message(message, ctx)

            # Add response to context
            if response and response.context_id:
                await self._context_manager.add_message(
                    response.context_id, response
                )

            return response

        except Exception as e:
            logger.error(f"Error delivering message to {message.recipient_id}: {e}")
            return self._create_error_response(message, str(e))

    async def _broadcast_message(
        self,
        message: Message,
        context: Dict[str, Any],
    ) -> None:
        """
        Broadcast a message to all entities except sender.

        Args:
            message: Message to broadcast
            context: Context data
        """
        tasks = []
        for entity_id, entity in self._entities.items():
            if entity_id != message.sender_id:
                tasks.append(entity.process_message(message, context))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _create_error_response(self, message: Message, error: str) -> Message:
        """Create an error response message."""
        return Message(
            sender_id="system",
            recipient_id=message.sender_id,
            content={"error": error},
            intent=Intent.ERROR,
            context_id=message.context_id,
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_connection(self, handler: Callable) -> None:
        """
        Register a handler for entity connections.

        Args:
            handler: Function(entity_id, entity) to call
        """
        self._connection_handlers.append(handler)

    def on_disconnection(self, handler: Callable) -> None:
        """
        Register a handler for entity disconnections.

        Args:
            handler: Function(entity_id) to call
        """
        self._disconnection_handlers.append(handler)

    def on_message(self, handler: Callable) -> None:
        """
        Register a global message handler.

        Args:
            handler: Function(message) to call for all messages
        """
        self._message_handlers.append(handler)

    # =========================================================================
    # Utility
    # =========================================================================

    async def clear(self) -> None:
        """Unregister all entities."""
        entity_ids = list(self._entities.keys())
        for entity_id in entity_ids:
            await self.unregister(entity_id)

    def __len__(self) -> int:
        """Return number of registered entities."""
        return len(self._entities)

    def __contains__(self, entity_id: str) -> bool:
        """Check if entity is registered."""
        return entity_id in self._entities


# Global registry instance
_registry: Optional[Registry] = None


def get_registry() -> Registry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
