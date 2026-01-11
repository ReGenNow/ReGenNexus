"""
ReGenNexus Core - ROS 2 Integration Example

This example demonstrates how to integrate ROS 2 with ReGenNexus Core.
It creates a bridge between ROS 2 topics and ReGenNexus messages.
"""

import asyncio
import json
import time
import logging
import sys
import os
from typing import Dict, Any, List, Optional, Callable

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ReGenNexus Core modules

from regennexus.bridges.ros_bridge import ROSBridge
from regennexus.core.protocol import RegenNexus, get_registry
from regennexus.core.entity import Entity

async def main():
    """Main function to demonstrate ROS 2 integration."""
    logger.info("Starting ROS 2 integration example")
    
    # Initialize registry
    registry = Registry()
    await registry.start()
    logger.info("Registry started")
    
    # Initialize protocol
    protocol = UAP_Protocol(registry=registry)
    await protocol.initialize()
    logger.info("Protocol initialized")
    
    # Initialize ROS 2 bridge
    ros_bridge = ROSBridge(protocol=protocol)
    await ros_bridge.initialize()
    logger.info("ROS 2 bridge initialized")
    
    # Create a ReGenNexus entity for the ROS bridge
    entity_id = "ros2_bridge"
    await registry.register_entity(
        entity_id=entity_id,
        entity_type="bridge",
        capabilities=["ros2", "topic_bridge", "service_bridge"],
        metadata={
            "description": "ROS 2 Bridge",
            "ros_domain_id": 0
        }
    )
    logger.info(f"Registered entity: {entity_id}")
    
    # Subscribe to a ROS 2 topic and bridge to ReGenNexus
    ros_topic = "/chatter"
    ros_type = "std_msgs/String"
    
    # Define callback for messages from ROS
    async def ros_message_callback(msg_data):
        logger.info(f"Received message from ROS topic {ros_topic}: {msg_data}")
        
        # Create a ReGenNexus message
        await protocol.send_message(
            sender=entity_id,
            recipient="*",  # Broadcast to all entities
            intent="ros_topic_data",
            payload={
                "topic": ros_topic,
                "type": ros_type,
                "data": msg_data
            }
        )
    
    # Subscribe to the ROS topic
    subscription_id = await ros_bridge.subscribe_topic(
        topic_name=ros_topic,
        message_type=ros_type,
        callback=ros_message_callback
    )
    logger.info(f"Subscribed to ROS topic: {ros_topic}")
    
    # Create a ROS 2 publisher and bridge from ReGenNexus
    pub_topic = "/regennexus_out"
    pub_type = "std_msgs/String"
    
    # Create the publisher
    publisher_id = await ros_bridge.create_publisher(
        topic_name=pub_topic,
        message_type=pub_type
    )
    logger.info(f"Created ROS publisher: {pub_topic}")
    
    # Define a message handler for ReGenNexus messages
    async def message_handler(message):
        if message.intent == "publish_to_ros":
            # Extract data from the message
            topic = message.payload.get("topic", pub_topic)
            data = message.payload.get("data", {})
            
            # Publish to ROS
            await ros_bridge.publish_message(
                publisher_id=publisher_id,
                message_data=data
            )
            logger.info(f"Published message to ROS topic {topic}: {data}")
    
    # Register the message handler
    protocol.register_message_handler(entity_id, message_handler)
    logger.info("Registered message handler")
    
    # Simulate sending a message to ROS
    logger.info("Simulating a message from ReGenNexus to ROS")
    test_message = {
        "sender": "test_entity",
        "recipient": entity_id,
        "intent": "publish_to_ros",
        "payload": {
            "topic": pub_topic,
            "data": {
                "data": "Hello from ReGenNexus!"
            }
        }
    }
    
    # Process the test message
    await protocol.process_message(test_message)
    
    # Keep the example running
    logger.info("ROS 2 bridge is running. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Clean up
        await ros_bridge.unsubscribe_topic(subscription_id)
        await ros_bridge.destroy_publisher(publisher_id)
        await ros_bridge.shutdown()
        await protocol.shutdown()
        await registry.stop()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
