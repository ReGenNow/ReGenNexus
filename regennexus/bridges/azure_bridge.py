"""
ReGenNexus Core - Azure Bridge Module

This module implements a standalone bridge between ReGenNexus Core and Azure IoT services,
enabling communication with Azure IoT Hub and related services.
"""

import asyncio
import json
import base64
import hmac
import hashlib
import urllib.parse
import logging
import time
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

class AzureBridge:
    """
    Bridge between ReGenNexus Core and Azure IoT services.
    
    Provides connectivity to Azure IoT Hub and related services.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the Azure bridge.
        
        Args:
            connection_string: Azure IoT Hub connection string
        """
        self.connection_string = connection_string
        self.device_mappings = {}
        self.azure_initialized = False
        self.client = None
        
    async def initialize(self):
        """Initialize the Azure bridge."""
        if not self.connection_string:
            logger.warning("Azure IoT Hub connection string not provided. Azure bridge will be disabled.")
            return
        
        try:
            # Try to import Azure IoT SDK
            # Note: This is done at runtime to avoid hard dependency on Azure
            import azure.iot.device
            from azure.iot.device.aio import IoTHubDeviceClient
            
            # Parse connection string
            cs_args = self._parse_connection_string(self.connection_string)
            
            # Create client
            self.client = IoTHubDeviceClient.create_from_connection_string(self.connection_string)
            await self.client.connect()
            
            # Set up message handlers
            self.client.on_message_received = lambda message: asyncio.create_task(
                self._handle_cloud_to_device_message(message)
            )
            
            self.azure_initialized = True
            logger.info(f"Azure IoT Hub bridge initialized for host: {cs_args.get('HostName')}")
            
        except ImportError:
            logger.warning("Azure IoT Device SDK not found. Azure bridge will be disabled.")
            self.azure_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize Azure IoT Hub bridge: {e}")
            self.azure_initialized = False
    
    async def shutdown(self):
        """Shut down the Azure bridge."""
        if not self.azure_initialized or not self.client:
            return
        
        try:
            await self.client.disconnect()
            logger.info("Azure IoT Hub bridge shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down Azure IoT Hub bridge: {e}")
    
    def _parse_connection_string(self, connection_string: str) -> Dict[str, str]:
        """
        Parse an Azure IoT Hub connection string.
        
        Args:
            connection_string: Connection string to parse
            
        Returns:
            Dictionary of connection string components
        """
        result = {}
        parts = connection_string.split(';')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                result[key] = value
        return result
    
    async def map_device_to_entity(self, device_id: str, entity_id: str,
                                 device_to_entity_transform: Optional[Callable] = None,
                                 entity_to_device_transform: Optional[Callable] = None):
        """
        Map an Azure IoT device to a ReGenNexus entity.
        
        Args:
            device_id: Azure IoT device ID
            entity_id: ReGenNexus entity ID
            device_to_entity_transform: Function to transform device messages to entity messages
            entity_to_device_transform: Function to transform entity messages to device messages
        """
        if not self.azure_initialized:
            logger.warning("Azure bridge not initialized. Cannot map device.")
            return
        
        self.device_mappings[device_id] = {
            "entity_id": entity_id,
            "device_to_entity_transform": device_to_entity_transform,
            "entity_to_device_transform": entity_to_device_transform
        }
        
        logger.info(f"Mapped Azure IoT device {device_id} to entity {entity_id}")
    
    async def _handle_cloud_to_device_message(self, message):
        """
        Handle a message received from Azure IoT Hub.
        
        Args:
            message: Azure IoT Hub message
        """
        # Extract device ID from message properties
        device_id = message.custom_properties.get("deviceId")
        if not device_id or device_id not in self.device_mappings:
            logger.warning(f"Received message for unknown device: {device_id}")
            return
        
        mapping = self.device_mappings[device_id]
        entity_id = mapping["entity_id"]
        
        # Parse message content
        try:
            content = message.data.decode('utf-8')
            data = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse message content: {e}")
            return
        
        # Transform the message if a transform function is provided
        transform_func = mapping.get("device_to_entity_transform")
        if transform_func:
            entity_msg = transform_func(data)
        else:
            entity_msg = data
        
        # Forward the message to the entity
        # This would typically be done through the ReGenNexus Core protocol
        # For now, we just log it
        logger.info(f"Forwarding message from Azure IoT device {device_id} to entity {entity_id}: {entity_msg}")
        
        # In a real implementation, you would send the message to the entity
        # await protocol.send_message(entity_id, entity_msg, intent="azure_message")
    
    async def send_device_to_cloud_message(self, device_id: str, message: Any):
        """
        Send a device-to-cloud message to Azure IoT Hub.
        
        Args:
            device_id: Azure IoT device ID
            message: Message to send
        """
        if not self.azure_initialized or not self.client:
            logger.warning("Azure bridge not initialized. Cannot send message.")
            return
        
        if device_id not in self.device_mappings:
            logger.warning(f"Unknown device ID: {device_id}")
            return
        
        mapping = self.device_mappings[device_id]
        
        # Transform the message if a transform function is provided
        transform_func = mapping.get("entity_to_device_transform")
        if transform_func:
            device_msg = transform_func(message)
        else:
            device_msg = message
        
        # Convert message to JSON
        msg_json = json.dumps(device_msg)
        
        # Send the message
        try:
            await self.client.send_message(msg_json)
            logger.debug(f"Sent message to Azure IoT Hub for device {device_id}")
        except Exception as e:
            logger.error(f"Failed to send message to Azure IoT Hub: {e}")
    
    async def update_device_twin(self, device_id: str, properties: Dict[str, Any]):
        """
        Update device twin reported properties.
        
        Args:
            device_id: Azure IoT device ID
            properties: Properties to update
        """
        if not self.azure_initialized or not self.client:
            logger.warning("Azure bridge not initialized. Cannot update device twin.")
            return
        
        if device_id not in self.device_mappings:
            logger.warning(f"Unknown device ID: {device_id}")
            return
        
        try:
            await self.client.patch_twin_reported_properties(properties)
            logger.debug(f"Updated device twin for device {device_id}")
        except Exception as e:
            logger.error(f"Failed to update device twin: {e}")
    
    async def get_device_twin(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get device twin.
        
        Args:
            device_id: Azure IoT device ID
            
        Returns:
            Device twin properties or None if not available
        """
        if not self.azure_initialized or not self.client:
            logger.warning("Azure bridge not initialized. Cannot get device twin.")
            return None
        
        if device_id not in self.device_mappings:
            logger.warning(f"Unknown device ID: {device_id}")
            return None
        
        try:
            twin = await self.client.get_twin()
            return twin
        except Exception as e:
            logger.error(f"Failed to get device twin: {e}")
            return None
    
    async def invoke_direct_method(self, target_device_id: str, method_name: str,
                                 payload: Any, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Invoke a direct method on another device.
        
        Note: This requires a service client, which is not implemented in this basic bridge.
        This method is included for completeness but will log a warning.
        
        Args:
            target_device_id: Target device ID
            method_name: Method name
            payload: Method payload
            timeout: Timeout in seconds
            
        Returns:
            Method response or None if not available
        """
        logger.warning("Direct method invocation requires a service client, which is not implemented in this basic bridge.")
        return None
    
    def generate_sas_token(self, uri: str, key: str, policy_name: Optional[str] = None,
                         expiry: Optional[int] = None) -> str:
        """
        Generate a SAS token for authentication.
        
        Args:
            uri: Resource URI
            key: Shared access key
            policy_name: Shared access policy name
            expiry: Token expiry time in seconds since epoch
            
        Returns:
            SAS token
        """
        if not expiry:
            expiry = int(time.time() + 3600)  # Default to 1 hour
        
        encoded_uri = urllib.parse.quote(uri, safe='')
        ttl = int(expiry)
        sign_key = f"{encoded_uri}\n{ttl}"
        
        key = base64.b64decode(key)
        signature = hmac.HMAC(key, sign_key.encode('utf-8'), hashlib.sha256).digest()
        signature = base64.b64encode(signature).decode('utf-8')
        
        token = {
            'sr': uri,
            'sig': signature,
            'se': str(ttl)
        }
        
        if policy_name:
            token['skn'] = policy_name
        
        return 'SharedAccessSignature ' + '&'.join([f"{k}={urllib.parse.quote(v, safe='')}" for k, v in token.items()])
