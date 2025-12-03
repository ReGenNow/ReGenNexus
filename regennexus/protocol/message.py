"""
ReGenNexus Core - Message Definition

This module provides the message definition for the ReGenNexus Core protocol.
It defines the structure and validation of messages exchanged between entities.
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UAP_Message:
    """Message for the ReGenNexus Core protocol."""
    
    def __init__(self, sender: str, recipient: str, intent: str, 
                payload: Dict[str, Any], message_id: Optional[str] = None,
                timestamp: Optional[float] = None, encrypted: bool = False,
                signature: Optional[str] = None, ttl: Optional[int] = None):
        """
        Initialize a UAP message.
        
        Args:
            sender: Entity ID of the sender
            recipient: Entity ID of the recipient (or '*' for broadcast)
            intent: Intent of the message
            payload: Message payload data
            message_id: Optional unique identifier (generated if not provided)
            timestamp: Optional timestamp (current time if not provided)
            encrypted: Whether the message is encrypted
            signature: Optional cryptographic signature
            ttl: Optional time-to-live in seconds
        """
        self.sender = sender
        self.recipient = recipient
        self.intent = intent
        self.payload = payload
        self.id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.encrypted = encrypted
        self.signature = signature
        self.ttl = ttl
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UAP_Message':
        """
        Create a message from a dictionary.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            UAP_Message instance
        """
        # Validate required fields
        required_fields = ['sender', 'recipient', 'intent']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Create message
        return cls(
            sender=data['sender'],
            recipient=data['recipient'],
            intent=data['intent'],
            payload=data.get('payload', {}),
            message_id=data.get('id'),
            timestamp=data.get('timestamp'),
            encrypted=data.get('encrypted', False),
            signature=data.get('signature'),
            ttl=data.get('ttl')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        result = {
            'id': self.id,
            'sender': self.sender,
            'recipient': self.recipient,
            'intent': self.intent,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
        
        if self.encrypted:
            result['encrypted'] = True
            
        if self.signature:
            result['signature'] = self.signature
            
        if self.ttl is not None:
            result['ttl'] = self.ttl
            
        return result
    
    def to_json(self) -> str:
        """
        Convert the message to a JSON string.
        
        Returns:
            JSON string representation of the message
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UAP_Message':
        """
        Create a message from a JSON string.
        
        Args:
            json_str: JSON string containing message data
            
        Returns:
            UAP_Message instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_expired(self) -> bool:
        """
        Check if the message has expired.
        
        Returns:
            Boolean indicating whether the message has expired
        """
        if self.ttl is None:
            return False
            
        return time.time() > self.timestamp + self.ttl
    
    def is_broadcast(self) -> bool:
        """
        Check if the message is a broadcast.
        
        Returns:
            Boolean indicating whether the message is a broadcast
        """
        return self.recipient == '*'
    
    def validate(self) -> bool:
        """
        Validate the message structure.
        
        Returns:
            Boolean indicating whether the message is valid
        """
        # Check required fields
        if not self.sender or not self.recipient or not self.intent:
            return False
            
        # Check if expired
        if self.is_expired():
            return False
            
        return True
    
    def __str__(self) -> str:
        """
        Get a string representation of the message.
        
        Returns:
            String representation
        """
        return f"UAP_Message(id={self.id}, sender={self.sender}, recipient={self.recipient}, intent={self.intent})"
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the message.
        
        Returns:
            Detailed string representation
        """
        return (f"UAP_Message(id={self.id}, sender={self.sender}, recipient={self.recipient}, "
                f"intent={self.intent}, timestamp={self.timestamp}, encrypted={self.encrypted})")

def create_response(request: UAP_Message, intent: str, payload: Dict[str, Any]) -> UAP_Message:
    """
    Create a response message to a request.
    
    Args:
        request: Original request message
        intent: Intent of the response
        payload: Response payload data
        
    Returns:
        Response message
    """
    return UAP_Message(
        sender=request.recipient,
        recipient=request.sender,
        intent=intent,
        payload=payload,
        message_id=f"response-{request.id}"
    )

def create_error_response(request: UAP_Message, error_code: str, error_message: str) -> UAP_Message:
    """
    Create an error response message.
    
    Args:
        request: Original request message
        error_code: Error code
        error_message: Error message
        
    Returns:
        Error response message
    """
    return create_response(
        request=request,
        intent="error",
        payload={
            "error_code": error_code,
            "error_message": error_message,
            "original_intent": request.intent
        }
    )

def create_ack_response(request: UAP_Message) -> UAP_Message:
    """
    Create an acknowledgment response message.
    
    Args:
        request: Original request message
        
    Returns:
        Acknowledgment response message
    """
    return create_response(
        request=request,
        intent="ack",
        payload={
            "original_intent": request.intent,
            "timestamp": time.time()
        }
    )
