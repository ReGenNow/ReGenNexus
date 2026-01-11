import asyncio
import json
import logging
import ssl
import uuid
import websockets
from typing import Dict, List, Optional, Callable, Any, Union

logger = logging.getLogger(__name__)

class UAP_Registry:
    """
    Universal Agent Protocol Registry Server
    
    This class implements a registry server that manages connections between UAP clients.
    It handles client registration, message routing, and connection management.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, 
                 ssl_context: Optional[ssl.SSLContext] = None,
                 auth_required: bool = True):
        """
        Initialize the UAP Registry server.
        
        Args:
            host: Host address to bind the server to
            port: Port number to listen on
            ssl_context: SSL context for secure connections
            auth_required: Whether authentication is required for clients
        """
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.auth_required = auth_required
        
        # Client connections
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Connection event handlers
        self.connection_handlers: List[Callable] = []
        self.disconnection_handlers: List[Callable] = []
        
        # Server state
        self.server = None
        self.running = False
    
    async def start(self):
        """Start the registry server."""
        self.running = True
        logger.info(f"Starting UAP Registry server on {self.host}:{self.port}")
        
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ssl=self.ssl_context
            )
            
            logger.info("UAP Registry server started successfully")
            
            # Keep the server running
            await self.server.wait_closed()
        except Exception as e:
            logger.error(f"Failed to start UAP Registry server: {e}")
            self.running = False
            raise
    
    async def stop(self):
        """Stop the registry server."""
        if self.server:
            logger.info("Stopping UAP Registry server")
            self.server.close()
            await self.server.wait_closed()
            self.running = False
            logger.info("UAP Registry server stopped")
    
    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """
        Handle a new client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        client_id = None
        
        try:
            # Wait for registration message
            registration = await websocket.recv()
            reg_data = json.loads(registration)
            
            # Validate registration
            if "entity_id" not in reg_data:
                logger.warning("Registration missing entity_id, closing connection")
                await websocket.close(1008, "Missing entity_id")
                return
            
            client_id = reg_data["entity_id"]
            
            # Check if client ID already exists
            if client_id in self.clients:
                logger.warning(f"Client ID {client_id} already registered, closing connection")
                await websocket.close(1008, "Entity ID already registered")
                return
            
            # Authenticate client if required
            if self.auth_required:
                if "auth" not in reg_data:
                    logger.warning(f"Authentication required but not provided for {client_id}")
                    await websocket.close(1008, "Authentication required")
                    return
                
                # Implement your authentication logic here
                auth_result = self._authenticate_client(client_id, reg_data["auth"])
                if not auth_result:
                    logger.warning(f"Authentication failed for {client_id}")
                    await websocket.close(1008, "Authentication failed")
                    return
            
            # Register the client
            logger.info(f"Client {client_id} registered")
            self.clients[client_id] = websocket
            
            # Notify connection handlers
            for handler in self.connection_handlers:
                try:
                    await handler(client_id)
                except Exception as e:
                    logger.error(f"Error in connection handler: {e}")
            
            # Send acknowledgment
            await websocket.send(json.dumps({
                "type": "registration_ack",
                "status": "success",
                "registry_id": str(uuid.uuid4())
            }))
            
            # Handle messages from this client
            await self._handle_client_messages(client_id, websocket)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for client {client_id}")
        except json.JSONDecodeError:
            logger.warning("Received invalid JSON during registration")
            await websocket.close(1008, "Invalid registration format")
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            # Clean up when the connection is closed
            if client_id and client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Client {client_id} unregistered")
                
                # Notify disconnection handlers
                for handler in self.disconnection_handlers:
                    try:
                        await handler(client_id)
                    except Exception as e:
                        logger.error(f"Error in disconnection handler: {e}")
    
    async def _handle_client_messages(self, client_id: str, websocket: websockets.WebSocketServerProtocol):
        """
        Handle messages from a connected client.
        
        Args:
            client_id: Client identifier
            websocket: WebSocket connection
        """
        try:
            async for message in websocket:
                try:
                    # Parse the message
                    msg_data = json.loads(message)
                    
                    # Check if it's a routing message
                    if "recipient" in msg_data:
                        await self._route_message(client_id, msg_data)
                    else:
                        # Handle system messages
                        await self._handle_system_message(client_id, msg_data)
                        
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {client_id}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for client {client_id}")
        except Exception as e:
            logger.error(f"Error in message handling loop for {client_id}: {e}")
    
    async def _route_message(self, sender_id: str, message: Dict[str, Any]):
        """
        Route a message to its intended recipient.
        
        Args:
            sender_id: Sender's client ID
            message: Message to route
        """
        recipient_id = message.get("recipient")
        
        if not recipient_id:
            logger.warning(f"Message from {sender_id} missing recipient")
            return
        
        # Add sender information if not present
        if "sender" not in message:
            message["sender"] = sender_id
        
        # Check if recipient exists
        if recipient_id not in self.clients:
            logger.warning(f"Recipient {recipient_id} not found for message from {sender_id}")
            
            # Send delivery failure notification to sender
            try:
                await self.clients[sender_id].send(json.dumps({
                    "type": "delivery_failure",
                    "recipient": recipient_id,
                    "reason": "Recipient not found",
                    "message_id": message.get("id", "unknown")
                }))
            except Exception as e:
                logger.error(f"Error sending delivery failure to {sender_id}: {e}")
            
            return
        
        # Send the message to the recipient
        try:
            await self.clients[recipient_id].send(json.dumps(message))
            logger.debug(f"Routed message from {sender_id} to {recipient_id}")
        except Exception as e:
            logger.error(f"Error routing message to {recipient_id}: {e}")
            
            # Send delivery failure notification to sender
            try:
                await self.clients[sender_id].send(json.dumps({
                    "type": "delivery_failure",
                    "recipient": recipient_id,
                    "reason": "Delivery failed",
                    "message_id": message.get("id", "unknown")
                }))
            except Exception as e2:
                logger.error(f"Error sending delivery failure to {sender_id}: {e2}")
    
    async def _handle_system_message(self, client_id: str, message: Dict[str, Any]):
        """
        Handle system messages from clients.
        
        Args:
            client_id: Client identifier
            message: System message
        """
        msg_type = message.get("type", "unknown")
        
        # Check if we have handlers for this message type
        if msg_type in self.message_handlers:
            for handler in self.message_handlers[msg_type]:
                try:
                    await handler(client_id, message)
                except Exception as e:
                    logger.error(f"Error in message handler for {msg_type}: {e}")
        else:
            logger.debug(f"No handlers for message type {msg_type} from {client_id}")
    
    def _authenticate_client(self, client_id: str, auth_data: Dict[str, Any]) -> bool:
        """
        Authenticate a client.
        
        Args:
            client_id: Client identifier
            auth_data: Authentication data
            
        Returns:
            True if authentication succeeds, False otherwise
        """
        # Implement your authentication logic here
        # This is a placeholder that always returns True
        logger.info(f"Authenticating client {client_id}")
        return True
    
    def on_connection(self, handler: Callable[[str], Any]):
        """
        Register a handler for client connections.
        
        Args:
            handler: Function to call when a client connects
        """
        self.connection_handlers.append(handler)
    
    def on_disconnection(self, handler: Callable[[str], Any]):
        """
        Register a handler for client disconnections.
        
        Args:
            handler: Function to call when a client disconnects
        """
        self.disconnection_handlers.append(handler)
    
    def on_message(self, message_type: str, handler: Callable[[str, Dict[str, Any]], Any]):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to call when a message of this type is received
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
    
    def broadcast(self, message: Dict[str, Any], exclude: Optional[Union[str, List[str]]] = None):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
            exclude: Client ID(s) to exclude from the broadcast
        """
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
        
        async def _do_broadcast():
            for client_id, websocket in self.clients.items():
                if client_id not in exclude:
                    try:
                        await websocket.send(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Error broadcasting to {client_id}: {e}")
        
        asyncio.create_task(_do_broadcast())

# In-process registry for local use
from regennexus.context.context_manager import ContextManager

class Registry:
    """In-process registry for local use."""
    def __init__(self):
        from regennexus.protocol.protocol_core import get_instance as get_protocol
        self._protocol = get_protocol()
        self._context_manager = ContextManager()

    async def register_entity(self, entity):
        return await self._protocol.register_entity(entity)

    async def unregister_entity(self, entity_id):
        return await self._protocol.unregister_entity(entity_id)

    async def get_entity(self, entity_id):
        return await self._protocol.get_entity(entity_id)

    async def find_entities(self, capability=None, entity_type=None):
        return await self._protocol.find_entities(capability=capability, entity_type=entity_type)

    async def route_message(self, message):
        cid = getattr(message, 'context_id', None)
        if cid:
            await self._context_manager.add_message(cid, message)
        response = await self._protocol.route_message(message)
        if response and getattr(response, 'context_id', None):
            await self._context_manager.add_message(response.context_id, response)
        return response

# Singleton server instance for UAP_Registry
_registry_server_instance = None

def get_instance() -> UAP_Registry:
    """Get singleton UAP_Registry server instance."""
    global _registry_server_instance
    if _registry_server_instance is None:
        _registry_server_instance = UAP_Registry()
    return _registry_server_instance
