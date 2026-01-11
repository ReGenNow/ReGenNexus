"""
RegenNexus UAP - Context Manager

Manages conversation contexts and message history.

Copyright (c) 2024 ReGen Designs LLC
"""

import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

from regennexus.core.message import Message

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """
    Represents a conversation context.

    Contexts track message history and state for a conversation.

    Attributes:
        id: Unique context identifier
        metadata: Additional context information
        messages: Message history
        created_at: Creation timestamp
        updated_at: Last update timestamp
        max_messages: Maximum messages to keep (0 = unlimited)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    max_messages: int = 1000

    def add_message(self, message: Message) -> None:
        """
        Add a message to the context.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.updated_at = time.time()

        # Trim if exceeds max
        if self.max_messages > 0 and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(
        self, start: int = 0, limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages from context.

        Args:
            start: Starting index
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        msgs = self.messages[start:]
        if limit is not None:
            msgs = msgs[:limit]
        return msgs

    def get_last_message(self) -> Optional[Message]:
        """
        Get the last message in context.

        Returns:
            Last message or None
        """
        return self.messages[-1] if self.messages else None

    def clear(self) -> None:
        """Clear all messages from context."""
        self.messages.clear()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "id": self.id,
            "metadata": self.metadata,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }


class ContextManager:
    """
    Manages conversation contexts and message history.

    This is a singleton class that tracks all active contexts.

    Example:
        manager = ContextManager()

        # Create a new context
        ctx = await manager.create_context(metadata={"topic": "device-control"})

        # Add messages
        await manager.add_message(ctx.id, message)

        # Get messages
        messages = await manager.get_messages(ctx.id)

        # Delete context
        await manager.delete_context(ctx.id)
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_contexts: int = 10000, context_ttl: int = 3600):
        """
        Initialize context manager.

        Args:
            max_contexts: Maximum number of contexts to keep
            context_ttl: Time-to-live for contexts in seconds (0 = no expiry)
        """
        if self._initialized:
            return

        self._contexts: Dict[str, Context] = {}
        self._max_contexts = max_contexts
        self._context_ttl = context_ttl
        self._initialized = True

    async def create_context(
        self,
        context_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_messages: int = 1000,
    ) -> Context:
        """
        Create a new conversation context.

        Args:
            context_id: Optional custom context ID
            metadata: Optional context metadata
            max_messages: Maximum messages to keep

        Returns:
            New Context instance
        """
        ctx_id = context_id or str(uuid.uuid4())
        ctx = Context(
            id=ctx_id,
            metadata=metadata or {},
            max_messages=max_messages,
        )
        self._contexts[ctx_id] = ctx

        # Clean up old contexts if needed
        await self._cleanup_contexts()

        logger.debug(f"Created context: {ctx_id}")
        return ctx

    async def get_context(self, context_id: str) -> Optional[Context]:
        """
        Get a context by ID.

        Args:
            context_id: Context identifier

        Returns:
            Context or None if not found
        """
        return self._contexts.get(context_id)

    async def get_or_create_context(
        self,
        context_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Context:
        """
        Get existing context or create new one.

        Args:
            context_id: Context identifier
            metadata: Metadata for new context

        Returns:
            Context instance
        """
        ctx = await self.get_context(context_id)
        if ctx is None:
            ctx = await self.create_context(context_id=context_id, metadata=metadata)
        return ctx

    async def delete_context(self, context_id: str) -> bool:
        """
        Delete a context.

        Args:
            context_id: Context identifier

        Returns:
            True if deleted, False if not found
        """
        if context_id in self._contexts:
            del self._contexts[context_id]
            logger.debug(f"Deleted context: {context_id}")
            return True
        return False

    async def add_message(self, context_id: str, message: Message) -> bool:
        """
        Add a message to a context.

        Args:
            context_id: Context identifier
            message: Message to add

        Returns:
            True if added, False if context not found
        """
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return False

        ctx.add_message(message)
        return True

    async def get_messages(
        self,
        context_id: str,
        start: int = 0,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        Get messages from a context.

        Args:
            context_id: Context identifier
            start: Starting index
            limit: Maximum messages

        Returns:
            List of messages (empty if context not found)
        """
        ctx = self._contexts.get(context_id)
        if ctx is None:
            return []

        return ctx.get_messages(start=start, limit=limit)

    async def list_contexts(self) -> List[Dict[str, Any]]:
        """
        List all contexts.

        Returns:
            List of context info dictionaries
        """
        return [
            {
                "id": ctx.id,
                "metadata": ctx.metadata,
                "message_count": len(ctx.messages),
                "created_at": ctx.created_at,
                "updated_at": ctx.updated_at,
            }
            for ctx in self._contexts.values()
        ]

    async def _cleanup_contexts(self) -> None:
        """Clean up expired or excess contexts."""
        current_time = time.time()

        # Remove expired contexts
        if self._context_ttl > 0:
            expired = [
                ctx_id
                for ctx_id, ctx in self._contexts.items()
                if (current_time - ctx.updated_at) > self._context_ttl
            ]
            for ctx_id in expired:
                del self._contexts[ctx_id]
                logger.debug(f"Expired context: {ctx_id}")

        # Remove oldest if exceeds max
        if len(self._contexts) > self._max_contexts:
            # Sort by updated_at and remove oldest
            sorted_contexts = sorted(
                self._contexts.items(),
                key=lambda x: x[1].updated_at,
            )
            to_remove = len(self._contexts) - self._max_contexts
            for ctx_id, _ in sorted_contexts[:to_remove]:
                del self._contexts[ctx_id]
                logger.debug(f"Removed old context: {ctx_id}")

    def clear_all(self) -> None:
        """Clear all contexts."""
        self._contexts.clear()


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
