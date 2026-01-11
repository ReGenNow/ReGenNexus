"""
ReGenNexus Core - Protocol Core Module

This module implements the core protocol functionality for the ReGenNexus Core,
providing message handling, entity management, and secure communication.
"""

import asyncio
import uuid
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple

# Import security components
from regennexus.security.security import SecurityManager

logger = logging.getLogger(__name__)

class Message:
    """
    Represents a message in the ReGenNexus Core protocol.
    
    Messages are the primary means of communication between entities.
    """
    
    def __init__(self, 
                 sender_id: str, 
                 recipient_id: str, 
                 content: Any, 
                 intent: str = "message",
                 context_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new message.
        
        Args:
            sender_id: Identifier of the sending entity
            recipient_id: Identifier of the receiving entity
            content: Message content (can be any serializable object)
            intent: Purpose of the message (e.g., "greeting", "command", "response")
            context_id: Optional identifier for the conversation context
            metadata: Optional additional information about the message
        """
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.content = content
        self.intent = intent
        self.context_id = context_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.timestamp = self.metadata.get("timestamp") or asyncio.get_event_loop().time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "intent": self.intent,
            "context_id": self.context_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary representation."""
        msg = cls(
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            content=data["content"],
            intent=data["intent"],
            context_id=data["context_id"],
            metadata=data["metadata"]
        )
        msg.id = data["id"]
        msg.timestamp = data["timestamp"]
        return msg
    
    def serialize(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: str) -> 'Message':
        """Deserialize message from JSON string."""
        return cls.from_dict(json.loads(data))


class Entity:
    """
    Base class for entities in the ReGenNexus Core protocol.
    
    Entities are the primary actors in the system and can send/receive messages.
    """
    
    def __init__(self, entity_id: str):
        """
        Initialize a new entity.
        
        Args:
            entity_id: Unique identifier for this entity
        """
        self.id = entity_id
        self.capabilities = []
        self.security_manager = SecurityManager()
        self._message_handlers = []
        
    async def process_message(self, message: Message, context: Dict[str, Any]) -> Optional[Message]:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
            context: The conversation context
            
        Returns:
            Optional response message
        """
        for handler in self._message_handlers:
            try:
                result = await handler(message, context)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
        
        return None
    
    def register_message_handler(self, handler: Callable[[Message, Dict[str, Any]], Optional[Message]]):
        """
        Register a message handler function.
        
        Args:
            handler: Function that processes messages
        """
        self._message_handlers.append(handler)
        
    async def send_message(self, 
                          recipient_id: str, 
                          content: Any, 
                          intent: str = "message",
                          context_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Create a new message from this entity.
        
        Args:
            recipient_id: Identifier of the receiving entity
            content: Message content
            intent: Purpose of the message
            context_id: Optional identifier for the conversation context
            metadata: Optional additional information about the message
            
        Returns:
            The created message
        """
        return Message(
            sender_id=self.id,
            recipient_id=recipient_id,
            content=content,
            intent=intent,
            context_id=context_id,
            metadata=metadata
        )
    
    async def encrypt_message(self, message: Message, recipient_public_key: bytes) -> bytes:
        """
        Encrypt a message for secure transmission.
        
        Args:
            message: The message to encrypt
            recipient_public_key: Public key of the recipient
            
        Returns:
            Encrypted message data
        """
        # Use the security manager to encrypt the message
        message_data = message.serialize().encode('utf-8')
        
        # Use ECDH-384 if available, fall back to RSA for backward compatibility
        if self.security_manager.supports_ecdh():
            return await self.security_manager.encrypt_message_ecdh(
                message_data, 
                recipient_public_key
            )
        else:
            return await self.security_manager.encrypt_message_rsa(
                message_data, 
                recipient_public_key
            )
    
    async def decrypt_message(self, encrypted_data: bytes) -> Message:
        """
        Decrypt a received message.
        
        Args:
            encrypted_data: The encrypted message data
            
        Returns:
            Decrypted message
        """
        # Use the security manager to decrypt the message
        decrypted_data = await self.security_manager.decrypt_message(encrypted_data)
        return Message.deserialize(decrypted_data.decode('utf-8'))
    
class Intent:
    """Placeholder for message intents."""
    QUERY = "query"
    RESPONSE = "response"

# ProtocolCore singleton factory
_protocol_core_instance = None

def get_instance(security_level: int = 2) -> "ProtocolCore":
    """
    Get a singleton instance of the ProtocolCore.

    Args:
        security_level: Security level parameter (1=basic,2=enhanced,3=maximum)

    Returns:
        ProtocolCore instance
    """
    global _protocol_core_instance
    if _protocol_core_instance is None:
        _protocol_core_instance = ProtocolCore(security_level=security_level)
    return _protocol_core_instance
    
    def get_public_key(self) -> bytes:
        """
        Get the public key for this entity.
        
        Returns:
            Public key bytes
        """
        return self.security_manager.get_public_key()
    
    def add_capability(self, capability: str):
        """
        Add a capability to this entity.
        
        Args:
            capability: Capability identifier
        """
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def has_capability(self, capability: str) -> bool:
        """
        Check if this entity has a specific capability.
        
        Args:
            capability: Capability identifier
            
        Returns:
            True if the entity has the capability, False otherwise
        """
        return capability in self.capabilities


class ProtocolCore:
    """
    Core protocol implementation for ReGenNexus Core.
    
    This class provides the main protocol functionality, including message
    routing, entity management, and security features.
    """
    
    def __init__(self, security_level: int = 2):
        """
        Initialize the protocol core.
        
        Args:
            security_level: Security level (1=basic, 2=enhanced, 3=maximum)
        """
        self.entities = {}
        self.security_manager = SecurityManager(security_level=security_level)
        
    async def register_entity(self, entity: Entity):
        """
        Register an entity with the protocol.
        
        Args:
            entity: The entity to register
        """
        self.entities[entity.id] = entity
        logger.info(f"Entity registered: {entity.id}")
        
    async def unregister_entity(self, entity_id: str):
        """
        Unregister an entity from the protocol.
        
        Args:
            entity_id: Identifier of the entity to unregister
        """
        if entity_id in self.entities:
            del self.entities[entity_id]
            logger.info(f"Entity unregistered: {entity_id}")
        
    async def route_message(self, message: Message, context: Optional[Dict[str, Any]] = None) -> Optional[Message]:
        """
        Route a message to its recipient.
        
        Args:
            message: The message to route
            context: Optional conversation context
            
        Returns:
            Optional response message
        """
        if message.recipient_id not in self.entities:
            logger.warning(f"Recipient not found: {message.recipient_id}")
            return None
        
        recipient = self.entities[message.recipient_id]
        ctx = context or {}
        
        logger.debug(f"Routing message: {message.id} from {message.sender_id} to {message.recipient_id}")
        return await recipient.process_message(message, ctx)
    
    async def encrypt_message(self, message: Message, recipient_id: str) -> bytes:
        """
        Encrypt a message for secure transmission.
        
        Args:
            message: The message to encrypt
            recipient_id: Identifier of the recipient
            
        Returns:
            Encrypted message data
        """
        if recipient_id not in self.entities:
            raise ValueError(f"Recipient not found: {recipient_id}")
        
        recipient = self.entities[recipient_id]
        recipient_public_key = recipient.get_public_key()
        
        # Use ECDH-384 if available, fall back to RSA for backward compatibility
        if self.security_manager.supports_ecdh():
            return await self.security_manager.encrypt_message_ecdh(
                message.serialize().encode('utf-8'), 
                recipient_public_key
            )
        else:
            return await self.security_manager.encrypt_message_rsa(
                message.serialize().encode('utf-8'), 
                recipient_public_key
            )
    
    async def decrypt_message(self, encrypted_data: bytes, entity_id: str) -> Message:
        """
        Decrypt a received message.
        
        Args:
            encrypted_data: The encrypted message data
            entity_id: Identifier of the receiving entity
            
        Returns:
            Decrypted message
        """
        if entity_id not in self.entities:
            raise ValueError(f"Entity not found: {entity_id}")
        
        entity = self.entities[entity_id]
        decrypted_data = await entity.security_manager.decrypt_message(encrypted_data)
        return Message.deserialize(decrypted_data.decode('utf-8'))
