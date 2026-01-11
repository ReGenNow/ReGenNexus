# API Reference

This document provides a comprehensive reference for the ReGenNexus Core API.

## Protocol Module

### UAP_Client

The main client class for interacting with the ReGenNexus protocol.

```python
class UAP_Client:
    def __init__(self, entity_id, registry_url, private_key=None, certificate=None):
        """
        Initialize a new UAP client.
        
        Args:
            entity_id (str): Unique identifier for this entity
            registry_url (str): URL of the registry server
            private_key (str, optional): Private key for secure communication
            certificate (str, optional): Certificate for authentication
        """
        
    async def connect(self):
        """
        Connect to the registry server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        
    async def send_message(self, message):
        """
        Send a message to another entity.
        
        Args:
            message (UAP_Message): Message to send
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        
    def register_message_handler(self, handler_func):
        """
        Register a function to handle incoming messages.
        
        Args:
            handler_func (callable): Async function that takes a UAP_Message parameter
        """
        
    async def run(self):
        """
        Run the client's message loop.
        """
        
    async def disconnect(self):
        """
        Disconnect from the registry server.
        """
        
    async def execute_plugin_action(self, plugin_id, action, parameters=None):
        """
        Execute an action on a registered plugin.
        
        Args:
            plugin_id (str): ID of the plugin
            action (str): Name of the action to execute
            parameters (dict, optional): Parameters for the action
            
        Returns:
            dict: Result of the action
        """
```

### UAP_Message

Represents a message in the ReGenNexus protocol.

```python
class UAP_Message:
    def __init__(self, sender, recipient, intent, payload, context_id=None, timestamp=None):
        """
        Initialize a new UAP message.
        
        Args:
            sender (str): Entity ID of the sender
            recipient (str): Entity ID of the recipient
            intent (str): Purpose or type of the message
            payload (dict): Content of the message
            context_id (str, optional): ID of the conversation context
            timestamp (float, optional): Message creation time
        """
        
    def to_dict(self):
        """
        Convert the message to a dictionary.
        
        Returns:
            dict: Dictionary representation of the message
        """
        
    @classmethod
    def from_dict(cls, data):
        """
        Create a message from a dictionary.
        
        Args:
            data (dict): Dictionary representation of a message
            
        Returns:
            UAP_Message: New message instance
        """
```

## Registry Module

### Registry

Manages entity registration and message routing.

```python
class Registry:
    def __init__(self):
        """
        Initialize a new registry.
        """
        
    async def register_entity(self, entity):
        """
        Register an entity with the registry.
        
        Args:
            entity (Entity): Entity to register
            
        Returns:
            str: Registration ID
        """
        
    async def unregister_entity(self, entity_id):
        """
        Unregister an entity from the registry.
        
        Args:
            entity_id (str): ID of the entity to unregister
            
        Returns:
            bool: True if successful, False otherwise
        """
        
    async def get_entity(self, entity_id):
        """
        Get an entity by ID.
        
        Args:
            entity_id (str): ID of the entity to retrieve
            
        Returns:
            Entity: The requested entity or None
        """
        
    async def find_entities(self, capability=None, entity_type=None):
        """
        Find entities by capability or type.
        
        Args:
            capability (str, optional): Capability to search for
            entity_type (str, optional): Entity type to search for
            
        Returns:
            list: List of matching entities
        """
        
    async def route_message(self, message):
        """
        Route a message to its recipient.
        
        Args:
            message (UAP_Message): Message to route
            
        Returns:
            bool: True if message routed successfully, False otherwise
        """
```

## Context Module

### ContextManager

Manages conversation contexts.

```python
class ContextManager:
    def __init__(self):
        """
        Initialize a new context manager.
        """
        
    async def create_context(self, metadata=None):
        """
        Create a new conversation context.
        
        Args:
            metadata (dict, optional): Additional information about the context
            
        Returns:
            Context: Newly created context
        """
        
    async def get_context(self, context_id):
        """
        Get a context by ID.
        
        Args:
            context_id (str): ID of the context to retrieve
            
        Returns:
            Context: The requested context or None
        """
        
    async def add_message(self, context_id, message):
        """
        Add a message to a context.
        
        Args:
            context_id (str): ID of the context
            message (UAP_Message): Message to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        
    async def get_messages(self, context_id, start=0, limit=None):
        """
        Get messages from a context.
        
        Args:
            context_id (str): ID of the context
            start (int, optional): Index to start from
            limit (int, optional): Maximum number of messages to return
            
        Returns:
            list: List of messages
        """
```

## Security Module

### Crypto

Cryptographic functions for secure communication.

```python
def generate_keypair():
    """
    Generate a new ECDH-384 keypair.
    
    Returns:
        tuple: (private_key, public_key)
    """
    
def encrypt(data, public_key):
    """
    Encrypt data using ECDH-384 and AES-256-GCM.
    
    Args:
        data (bytes): Data to encrypt
        public_key (str): Recipient's public key
        
    Returns:
        bytes: Encrypted data
    """
    
def decrypt(encrypted_data, private_key):
    """
    Decrypt data using ECDH-384 and AES-256-GCM.
    
    Args:
        encrypted_data (bytes): Data to decrypt
        private_key (str): Recipient's private key
        
    Returns:
        bytes: Decrypted data
    """
```

### Auth

Authentication functions.

```python
def create_certificate(entity_id, public_key, valid_days=365):
    """
    Create a certificate for entity authentication.
    
    Args:
        entity_id (str): ID of the entity
        public_key (str): Public key of the entity
        valid_days (int, optional): Certificate validity period in days
        
    Returns:
        str: Certificate in PEM format
    """
    
def verify_certificate(certificate):
    """
    Verify a certificate.
    
    Args:
        certificate (str): Certificate to verify
        
    Returns:
        dict: Certificate information if valid, None otherwise
    """
```

## Plugins Module

### Base Plugin

Base class for all plugins.

```python
class Plugin:
    def __init__(self, plugin_id):
        """
        Initialize a new plugin.
        
        Args:
            plugin_id (str): Unique identifier for this plugin
        """
        
    async def initialize(self):
        """
        Initialize the plugin.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        
    async def execute_action(self, action, parameters=None):
        """
        Execute an action.
        
        Args:
            action (str): Name of the action to execute
            parameters (dict, optional): Parameters for the action
            
        Returns:
            dict: Result of the action
        """
        
    async def shutdown(self):
        """
        Shut down the plugin.
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
```

## Bridges Module

### ROS Bridge

Bridge for Robot Operating System integration.

```python
class ROSBridge:
    def __init__(self, node_name="regennexus_bridge"):
        """
        Initialize a new ROS bridge.
        
        Args:
            node_name (str, optional): Name of the ROS node
        """
        
    async def initialize(self):
        """
        Initialize the bridge.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        
    async def create_topic_subscription(self, topic_name, message_type, callback):
        """
        Create a subscription to a ROS topic.
        
        Args:
            topic_name (str): Name of the topic
            message_type (type): ROS message type
            callback (callable): Function to call when a message is received
            
        Returns:
            Subscription: ROS subscription object
        """
        
    async def create_topic_publisher(self, topic_name, message_type):
        """
        Create a publisher for a ROS topic.
        
        Args:
            topic_name (str): Name of the topic
            message_type (type): ROS message type
            
        Returns:
            Publisher: ROS publisher object
        """
        
    async def call_service(self, service_name, service_type, request):
        """
        Call a ROS service.
        
        Args:
            service_name (str): Name of the service
            service_type (type): ROS service type
            request: Service request
            
        Returns:
            Response: Service response
        """
```

### Azure Bridge

Bridge for Azure IoT Hub integration.

```python
class AzureBridge:
    def __init__(self, connection_string):
        """
        Initialize a new Azure bridge.
        
        Args:
            connection_string (str): Azure IoT Hub connection string
        """
        
    async def initialize(self):
        """
        Initialize the bridge.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        
    async def send_telemetry(self, data):
        """
        Send telemetry data to Azure IoT Hub.
        
        Args:
            data (dict): Telemetry data
            
        Returns:
            bool: True if successful, False otherwise
        """
        
    async def receive_command(self, command_name, callback):
        """
        Register a callback for a direct method command.
        
        Args:
            command_name (str): Name of the command
            callback (callable): Function to call when the command is received
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        
    async def update_twin(self, properties):
        """
        Update device twin reported properties.
        
        Args:
            properties (dict): Properties to update
            
        Returns:
            bool: True if update successful, False otherwise
        """
```
