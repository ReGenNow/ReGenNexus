"""
ReGenNexus Core - Context Manager

This module provides a simple in-memory context manager for tracking
conversation contexts and message history.
"""
import uuid

__all__ = ["ContextManager"]

class Context:
    """Represents a conversation context."""
    def __init__(self, context_id, metadata=None):
        self.id = context_id
        self.metadata = metadata or {}
        self.messages = []

class ContextManager:
    """Manages conversation contexts and message history."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ContextManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_contexts"):
            self._contexts = {}

    async def create_context(self, metadata=None):
        """Create a new conversation context."""
        context_id = str(uuid.uuid4())
        ctx = Context(context_id, metadata)
        self._contexts[context_id] = ctx
        return ctx

    async def get_context(self, context_id):
        """Retrieve a context by its ID."""
        return self._contexts.get(context_id)

    async def add_message(self, context_id, message):
        """Add a message to the specified context."""
        ctx = self._contexts.get(context_id)
        if not ctx:
            return False
        ctx.messages.append(message)
        return True

    async def get_messages(self, context_id, start=0, limit=None):
        """Get messages from a context."""
        ctx = self._contexts.get(context_id)
        if not ctx:
            return []
        msgs = ctx.messages[start:]
        if limit is not None:
            msgs = msgs[:limit]
        return msgs