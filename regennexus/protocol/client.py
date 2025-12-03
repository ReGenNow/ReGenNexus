"""
ReGenNexus Core - Protocol Client

This module provides the client implementation for the ReGenNexus Core protocol.
It handles message sending, receiving, and entity registration.
"""

import asyncio
import json
import time
import logging
import uuid
import aiohttp
import websockets
from typing import Dict, Any, List, Optional, Callable, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UAP_Client:
    """Client for the ReGenNexus Core protocol."""
    
    def __init__(self, entity_id: str, registry_url: str = "ws://localhost:8000", 
                 security_enabled: bool = True, auto_reconnect: bool = True):
        """
        Initialize the UAP client.
        
        Args:
            entity_id: Unique identifier for this entity
            registry_url: URL of the registry service (e.g., ws://localhost:8000)
            security_enabled: Whether to enable security features
            auto_reconnect: Whether to automatically reconnect on connection loss
        """
        self.entity_id = entity_id
        self.registry_url = registry_url
        self.security_enabled = security_enabled
        self.auto_reconnect = auto_reconnect
        self.connected = False
        self.websocket = None
        self.message_handlers = []
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.receive_task = None
    
    async def connect(self) -> bool:
        """
        Connect to the registry and initialize the client.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Connect to the WebSocket registry
            self.websocket = await websockets.connect(self.registry_url)
            
            # Send registration message
            registration = {
                "entity_id": self.entity_id,
                "auth": {}  # Add authentication data if required
            }
            await self.websocket.send(json.dumps(registration))
            
            # Wait for acknowledgment
            ack = await self.websocket.recv()
            ack_data = json.loads(ack)
            if ack_data.get("status") != "success":
                logger.error(f"Registration failed: {ack_data}")
                return False
            
            # Initialize security if enabled
            if self.security_enabled:
                from regennexus.security.security import get_instance as get_security
                self.security_manager = get_security()
            
            # Start message processing
            self.processing_task = asyncio.create_task(self._process_messages())
            self.receive_task = asyncio.create_task(self._receive_messages())
            
            self.connected = True
            logger.info(f"Client connected: {self.entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting client: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the registry and clean up resources.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.connected = False
            
            # Stop message processing
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
                self.processing_task = None
            
            # Stop receiving messages
            if self.receive_task:
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass
                self.receive_task = None
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            logger.info(f"Client disconnected: {self.entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")
            return False
    
    def register_message_handler(self, handler: Callable) -> None:
        """
        Register a message handler function.
        
        Args:
            handler: Async function that takes a message as parameter
        """
        self.message_handlers.append(handler)
        logger.debug(f"Registered message handler for {self.entity_id}")
    
    def unregister_message_handler(self, handler: Callable) -> bool:
        """
        Unregister a message handler function.
        
        Args:
            handler: Handler function to unregister
            
        Returns:
            Boolean indicating success
        """
        if handler in self.message_handlers:
            self.message_handlers.remove(handler)
            logger.debug(f"Unregistered message handler for {self.entity_id}")
            return True
        return False
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to another entity.
        
        Args:
            message: Message to send
            
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            logger.error(f"Cannot send message: client not connected")
            return False
        
        try:
            # Ensure message has required fields
            if not isinstance(message, dict):
                raise ValueError("Message must be a dictionary")
                
            if 'sender' not in message:
                message['sender'] = self.entity_id
                
            if 'recipient' not in message:
                raise ValueError("Message must have a recipient")
                
            if 'intent' not in message:
                raise ValueError("Message must have an intent")
                
            if 'payload' not in message:
                message['payload'] = {}
                
            # Add message ID and timestamp if not present
            if 'id' not in message:
                message['id'] = str(uuid.uuid4())
                
            if 'timestamp' not in message:
                message['timestamp'] = time.time()
            
            # Apply security if enabled
            if self.security_enabled and self.security_manager:
                # Encrypt message if recipient is not a broadcast
                if message['recipient'] != '*':
                    message = await self.security_manager.encrypt_message(
                        sender_id=self.entity_id,
                        recipient_id=message['recipient'],
                        message=message
                    )
            
            # Send message over WebSocket
            await self.websocket.send(json.dumps(message))
            
            logger.debug(f"Sent message: {message['id']} to {message['recipient']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def receive_message(self, message: Dict[str, Any]) -> bool:
        """
        Receive a message from another entity.
        
        Args:
            message: Message received
            
        Returns:
            Boolean indicating success
        """
        try:
            # Apply security if enabled
            if self.security_enabled and self.security_manager:
                # Decrypt message if it's encrypted
                if 'encrypted' in message and message['encrypted']:
                    message = await self.security_manager.decrypt_message(
                        recipient_id=self.entity_id,
                        encrypted_message=message
                    )
            
            # Add message to processing queue
            await self.message_queue.put(message)
            return True
            
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return False
    
    async def _receive_messages(self) -> None:
        """Receive messages from the WebSocket and dispatch to handlers."""
        try:
            async for message in self.websocket:
                try:
                    msg_data = json.loads(message)
                    await self.receive_message(msg_data)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for {self.entity_id}")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
    
    async def _process_messages(self) -> None:
        """Process messages from the queue and dispatch to handlers."""
        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Process message with all handlers
                for handler in self.message_handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")
                
                # Mark message as processed
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                # Task was cancelled, exit
                break
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def register_capabilities(self, capabilities: List[str], 
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register entity capabilities with the registry.
        
        Args:
            capabilities: List of capabilities this entity provides
            metadata: Additional metadata about the entity
            
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            logger.error(f"Cannot register capabilities: client not connected")
            return False
        
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Send capabilities registration message
            message = {
                "recipient": "registry",
                "intent": "register_capabilities",
                "payload": {
                    "capabilities": capabilities,
                    "metadata": metadata
                }
            }
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error registering capabilities: {e}")
            return False
    
    async def find_entities(self, entity_type: Optional[str] = None,
                           capabilities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find entities matching criteria.
        
        Args:
            entity_type: Optional entity type to filter by
            capabilities: Optional list of required capabilities
            
        Returns:
            List of matching entity information
        """
        if not self.connected:
            logger.error(f"Cannot find entities: client not connected")
            return []
        
        try:
            # Send find entities message
            message = {
                "recipient": "registry",
                "intent": "find_entities",
                "payload": {
                    "entity_type": entity_type,
                    "capabilities": capabilities
                }
            }
            await self.send_message(message)
            # Note: This requires the registry to send a response, which isn't implemented here
            # For now, return an empty list
            return []
            
        except Exception as e:
            logger.error(f"Error finding entities: {e}")
            return []
    
    async def heartbeat(self) -> bool:
        """
        Send a heartbeat to the registry to keep registration active.
        
        Returns:
            Boolean indicating success
        """
        if not self.connected:
            logger.error(f"Cannot send heartbeat: client not connected")
            return False
        
        try:
            # Send heartbeat message
            message = {
                "recipient": "registry",
                "intent": "heartbeat",
                "payload": {}
            }
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False