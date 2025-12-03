"""
RegenNexus UAP - Message Module

Defines the Message class for communication between entities.

Copyright (c) 2024 ReGen Designs LLC
"""

import uuid
import time
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Represents a message in the RegenNexus UAP protocol.

    Messages are the primary means of communication between entities.
    They contain sender/recipient info, content, intent, and metadata.

    Attributes:
        sender_id: Identifier of the sending entity
        recipient_id: Identifier of the receiving entity (or '*' for broadcast)
        content: Message content (any serializable object)
        intent: Purpose of the message (e.g., "query", "command", "response")
        context_id: Conversation context identifier
        id: Unique message identifier (auto-generated)
        timestamp: Message creation time (auto-generated)
        metadata: Additional message metadata
        encrypted: Whether the message content is encrypted
        ttl: Time-to-live in seconds (0 = no expiry)

    Example:
        # Create a simple message
        msg = Message(
            sender_id="device-1",
            recipient_id="device-2",
            content={"action": "turn_on"},
            intent="command"
        )

        # Serialize to JSON
        json_str = msg.to_json()

        # Deserialize from JSON
        msg2 = Message.from_json(json_str)
    """

    sender_id: str
    recipient_id: str
    content: Any
    intent: str = "message"
    context_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted: bool = False
    ttl: int = 0

    def __post_init__(self):
        """Post-initialization processing."""
        # Generate context_id if not provided
        if self.context_id is None:
            self.context_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary representation.

        Returns:
            Dictionary containing all message fields
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create message from dictionary representation.

        Args:
            data: Dictionary containing message fields

        Returns:
            Message instance
        """
        return cls(
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            content=data["content"],
            intent=data.get("intent", "message"),
            context_id=data.get("context_id"),
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            encrypted=data.get("encrypted", False),
            ttl=data.get("ttl", 0),
        )

    def to_json(self) -> str:
        """
        Serialize message to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """
        Deserialize message from JSON string.

        Args:
            json_str: JSON string

        Returns:
            Message instance
        """
        return cls.from_dict(json.loads(json_str))

    def to_bytes(self) -> bytes:
        """
        Serialize message to bytes.

        Returns:
            Bytes representation
        """
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        """
        Deserialize message from bytes.

        Args:
            data: Bytes data

        Returns:
            Message instance
        """
        return cls.from_json(data.decode("utf-8"))

    def is_expired(self) -> bool:
        """
        Check if message has expired based on TTL.

        Returns:
            True if expired, False otherwise
        """
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def is_broadcast(self) -> bool:
        """
        Check if this is a broadcast message.

        Returns:
            True if broadcast, False otherwise
        """
        return self.recipient_id == "*"

    def create_response(
        self,
        content: Any,
        intent: str = "response",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Message":
        """
        Create a response message to this message.

        Args:
            content: Response content
            intent: Response intent (default: "response")
            metadata: Additional metadata

        Returns:
            New Message instance as response
        """
        return Message(
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            content=content,
            intent=intent,
            context_id=self.context_id,
            metadata=metadata or {},
        )

    def __str__(self) -> str:
        """String representation of message."""
        return f"Message({self.sender_id} -> {self.recipient_id}: {self.intent})"

    def __repr__(self) -> str:
        """Detailed representation of message."""
        return (
            f"Message(id={self.id!r}, sender_id={self.sender_id!r}, "
            f"recipient_id={self.recipient_id!r}, intent={self.intent!r}, "
            f"content={self.content!r})"
        )


class Intent:
    """Standard message intents."""

    # Basic intents
    MESSAGE = "message"
    QUERY = "query"
    RESPONSE = "response"
    COMMAND = "command"
    EVENT = "event"
    ERROR = "error"

    # System intents
    REGISTER = "register"
    UNREGISTER = "unregister"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"

    # Device intents
    STATUS = "status"
    CONTROL = "control"
    SENSOR_DATA = "sensor_data"
    CONFIG = "config"
