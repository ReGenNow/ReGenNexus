"""
ReGenNexus Core - ROS Bridge Module

This module implements a bridge between ReGenNexus Core and ROS 2,
enabling seamless communication between ROS topics/services and ReGenNexus entities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

class ROSBridge:
    """
    Bridge between ReGenNexus Core and ROS 2.
    
    Enables bidirectional communication between ROS topics/services and
    ReGenNexus entities.
    """
    
    def __init__(self, node_name: str = "regennexus_bridge"):
        """
        Initialize the ROS bridge.
        
        Args:
            node_name: Name of the ROS node
        """
        self.node_name = node_name
        self.topic_mappings = {}
        self.service_mappings = {}
        self.action_mappings = {}
        self.parameter_mappings = {}
        self.ros_initialized = False
        self.subscribers = {}
        self.publishers = {}
        self.service_clients = {}
        self.service_servers = {}
        self.action_clients = {}
        self.action_servers = {}
        
    async def initialize(self):
        """Initialize the ROS bridge."""
        try:
            # Import ROS 2 Python client library
            # Note: This is done at runtime to avoid hard dependency on ROS
            import rclpy
            from rclpy.node import Node
            
            # Initialize ROS 2
            rclpy.init()
            self.node = Node(self.node_name)
            self.executor = rclpy.executors.MultiThreadedExecutor()
            self.executor.add_node(self.node)
            
            # Start the executor in a separate thread
            import threading
            self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.executor_thread.start()
            
            self.ros_initialized = True
            logger.info(f"ROS 2 bridge initialized with node name: {self.node_name}")
            
        except ImportError:
            logger.warning("ROS 2 Python client library (rclpy) not found. ROS bridge will be disabled.")
            self.ros_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize ROS 2 bridge: {e}")
            self.ros_initialized = False
    
    async def shutdown(self):
        """Shut down the ROS bridge."""
        if not self.ros_initialized:
            return
        
        try:
            import rclpy
            
            # Clean up subscribers and publishers
            for topic, sub in self.subscribers.items():
                self.node.destroy_subscription(sub)
            
            for topic, pub in self.publishers.items():
                self.node.destroy_publisher(pub)
            
            # Clean up service clients and servers
            for service, client in self.service_clients.items():
                self.node.destroy_client(client)
            
            for service, server in self.service_servers.items():
                self.node.destroy_service(server)
            
            # Clean up action clients and servers
            for action, client in self.action_clients.items():
                client.destroy()
            
            for action, server in self.action_servers.items():
                server.destroy()
            
            # Shut down ROS 2
            self.node.destroy_node()
            rclpy.shutdown()
            
            logger.info("ROS 2 bridge shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down ROS 2 bridge: {e}")
    
    async def map_topic_to_entity(self, topic_name: str, entity_id: str, 
                                direction: str = "bidirectional",
                                message_type: str = "std_msgs/String",
                                topic_to_entity_transform: Optional[Callable] = None,
                                entity_to_topic_transform: Optional[Callable] = None):
        """
        Map a ROS topic to a ReGenNexus entity.
        
        Args:
            topic_name: Name of the ROS topic
            entity_id: Identifier of the ReGenNexus entity
            direction: Direction of communication ("to_entity", "to_topic", or "bidirectional")
            message_type: ROS message type
            topic_to_entity_transform: Function to transform ROS messages to entity messages
            entity_to_topic_transform: Function to transform entity messages to ROS messages
        """
        if not self.ros_initialized:
            logger.warning("ROS bridge not initialized. Cannot map topic.")
            return
        
        self.topic_mappings[topic_name] = {
            "entity_id": entity_id,
            "direction": direction,
            "message_type": message_type,
            "topic_to_entity_transform": topic_to_entity_transform,
            "entity_to_topic_transform": entity_to_topic_transform
        }
        
        # Set up ROS subscriber if needed
        if direction in ["to_entity", "bidirectional"]:
            await self._create_subscriber(topic_name, message_type)
        
        # Set up ROS publisher if needed
        if direction in ["to_topic", "bidirectional"]:
            await self._create_publisher(topic_name, message_type)
        
        logger.info(f"Mapped ROS topic {topic_name} to entity {entity_id} ({direction})")
    
    async def _create_subscriber(self, topic_name: str, message_type: str):
        """
        Create a ROS subscriber for a topic.
        
        Args:
            topic_name: Name of the ROS topic
            message_type: ROS message type
        """
        if topic_name in self.subscribers:
            return
        
        try:
            # Import ROS message type dynamically
            msg_module, msg_class = message_type.split('/')
            exec(f"from {msg_module}.msg import {msg_class}")
            msg_type = eval(f"{msg_class}")
            
            # Create subscriber
            self.subscribers[topic_name] = self.node.create_subscription(
                msg_type,
                topic_name,
                lambda msg: asyncio.run(self._handle_ros_message(topic_name, msg)),
                10  # QoS profile depth
            )
            
            logger.debug(f"Created ROS subscriber for topic: {topic_name}")
            
        except Exception as e:
            logger.error(f"Failed to create ROS subscriber for topic {topic_name}: {e}")
    
    async def _create_publisher(self, topic_name: str, message_type: str):
        """
        Create a ROS publisher for a topic.
        
        Args:
            topic_name: Name of the ROS topic
            message_type: ROS message type
        """
        if topic_name in self.publishers:
            return
        
        try:
            # Import ROS message type dynamically
            msg_module, msg_class = message_type.split('/')
            exec(f"from {msg_module}.msg import {msg_class}")
            msg_type = eval(f"{msg_class}")
            
            # Create publisher
            self.publishers[topic_name] = self.node.create_publisher(
                msg_type,
                topic_name,
                10  # QoS profile depth
            )
            
            logger.debug(f"Created ROS publisher for topic: {topic_name}")
            
        except Exception as e:
            logger.error(f"Failed to create ROS publisher for topic {topic_name}: {e}")
    
    async def _handle_ros_message(self, topic_name: str, ros_msg):
        """
        Handle a message received from a ROS topic.
        
        Args:
            topic_name: Name of the ROS topic
            ros_msg: ROS message
        """
        if topic_name not in self.topic_mappings:
            return
        
        mapping = self.topic_mappings[topic_name]
        entity_id = mapping["entity_id"]
        
        # Transform the message if a transform function is provided
        transform_func = mapping.get("topic_to_entity_transform")
        if transform_func:
            entity_msg = transform_func(ros_msg)
        else:
            # Default transformation: convert ROS message to dictionary
            entity_msg = self._ros_msg_to_dict(ros_msg)
        
        # Forward the message to the entity
        # This would typically be done through the ReGenNexus Core protocol
        # For now, we just log it
        logger.info(f"Forwarding message from ROS topic {topic_name} to entity {entity_id}: {entity_msg}")
        
        # In a real implementation, you would send the message to the entity
        # await protocol.send_message(entity_id, entity_msg, intent="ros_message")
    
    async def publish_to_ros(self, topic_name: str, message: Any):
        """
        Publish a message to a ROS topic.
        
        Args:
            topic_name: Name of the ROS topic
            message: Message to publish
        """
        if not self.ros_initialized or topic_name not in self.publishers:
            logger.warning(f"Cannot publish to ROS topic {topic_name}: topic not mapped or bridge not initialized")
            return
        
        mapping = self.topic_mappings[topic_name]
        
        # Transform the message if a transform function is provided
        transform_func = mapping.get("entity_to_topic_transform")
        if transform_func:
            ros_msg = transform_func(message)
        else:
            # Default transformation: convert dictionary to ROS message
            msg_type = self.publishers[topic_name].msg_type
            ros_msg = self._dict_to_ros_msg(message, msg_type)
        
        # Publish the message
        self.publishers[topic_name].publish(ros_msg)
        logger.debug(f"Published message to ROS topic {topic_name}")
    
    async def map_service_to_entity(self, service_name: str, entity_id: str,
                                  service_type: str,
                                  request_transform: Optional[Callable] = None,
                                  response_transform: Optional[Callable] = None):
        """
        Map a ROS service to a ReGenNexus entity.
        
        Args:
            service_name: Name of the ROS service
            entity_id: Identifier of the ReGenNexus entity
            service_type: ROS service type
            request_transform: Function to transform ROS requests to entity messages
            response_transform: Function to transform entity messages to ROS responses
        """
        if not self.ros_initialized:
            logger.warning("ROS bridge not initialized. Cannot map service.")
            return
        
        self.service_mappings[service_name] = {
            "entity_id": entity_id,
            "service_type": service_type,
            "request_transform": request_transform,
            "response_transform": response_transform
        }
        
        # Set up ROS service server
        await self._create_service_server(service_name, service_type)
        
        logger.info(f"Mapped ROS service {service_name} to entity {entity_id}")
    
    async def _create_service_server(self, service_name: str, service_type: str):
        """
        Create a ROS service server.
        
        Args:
            service_name: Name of the ROS service
            service_type: ROS service type
        """
        if service_name in self.service_servers:
            return
        
        try:
            # Import ROS service type dynamically
            srv_module, srv_class = service_type.split('/')
            exec(f"from {srv_module}.srv import {srv_class}")
            srv_type = eval(f"{srv_class}")
            
            # Create service server
            self.service_servers[service_name] = self.node.create_service(
                srv_type,
                service_name,
                lambda request, response: asyncio.run(self._handle_ros_service_request(service_name, request, response))
            )
            
            logger.debug(f"Created ROS service server for service: {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to create ROS service server for service {service_name}: {e}")
    
    async def _handle_ros_service_request(self, service_name: str, request, response):
        """
        Handle a request received from a ROS service.
        
        Args:
            service_name: Name of the ROS service
            request: ROS service request
            response: ROS service response
        """
        if service_name not in self.service_mappings:
            return
        
        mapping = self.service_mappings[service_name]
        entity_id = mapping["entity_id"]
        
        # Transform the request if a transform function is provided
        transform_func = mapping.get("request_transform")
        if transform_func:
            entity_msg = transform_func(request)
        else:
            # Default transformation: convert ROS request to dictionary
            entity_msg = self._ros_msg_to_dict(request)
        
        # Forward the request to the entity
        # This would typically be done through the ReGenNexus Core protocol
        # For now, we just log it and return a dummy response
        logger.info(f"Forwarding service request from ROS service {service_name} to entity {entity_id}: {entity_msg}")
        
        # In a real implementation, you would send the request to the entity and wait for a response
        # entity_response = await protocol.send_request(entity_id, entity_msg, intent="ros_service_request")
        
        # For now, just create a dummy response
        entity_response = {"success": True, "message": "Dummy response"}
        
        # Transform the response if a transform function is provided
        transform_func = mapping.get("response_transform")
        if transform_func:
            transform_func(entity_response, response)
        else:
            # Default transformation: copy fields from dictionary to ROS response
            self._dict_to_ros_msg(entity_response, response)
        
        return response
    
    def _ros_msg_to_dict(self, ros_msg) -> Dict[str, Any]:
        """
        Convert a ROS message to a dictionary.
        
        Args:
            ros_msg: ROS message
            
        Returns:
            Dictionary representation of the message
        """
        result = {}
        
        # Get all fields in the message
        for field_name in dir(ros_msg):
            # Skip private and special fields
            if field_name.startswith('_'):
                continue
            
            # Skip methods and callables
            if callable(getattr(ros_msg, field_name)):
                continue
            
            # Get the field value
            value = getattr(ros_msg, field_name)
            
            # Handle nested messages
            if hasattr(value, '__slots__'):
                result[field_name] = self._ros_msg_to_dict(value)
            else:
                result[field_name] = value
        
        return result
    
    def _dict_to_ros_msg(self, data: Dict[str, Any], ros_msg):
        """
        Copy data from a dictionary to a ROS message.
        
        Args:
            data: Source dictionary
            ros_msg: Target ROS message
        """
        for field_name, value in data.items():
            if hasattr(ros_msg, field_name):
                field = getattr(ros_msg, field_name)
                
                # Handle nested messages
                if hasattr(field, '__slots__'):
                    self._dict_to_ros_msg(value, field)
                else:
                    setattr(ros_msg, field_name, value)
        
        return ros_msg
