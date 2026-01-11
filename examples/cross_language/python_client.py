#!/usr/bin/env python3
"""
Python Client for Cross-Language Communication

This example demonstrates how to use the ReGenNexus Core protocol from Python
to communicate with clients written in other programming languages.

This file serves as the Python side of the cross-language demonstration.
"""

import asyncio
import logging
import json
import time
import uuid
# UAP_Client not available - this example needs updating for new API
from regennexus.core.message import Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Create a client
    client = UAP_Client(entity_id="python_client", registry_url="localhost:8000")
    
    # Connect to the registry
    await client.connect()
    logger.info("Python client connected to registry")
    
    # Register message handlers for different languages
    @client.message_handler(intent="js_message")
    async def handle_js_message(message):
        logger.info(f"Received message from JavaScript client: {message.payload}")
        
        # Process the message
        response_data = {
            "received": message.payload,
            "processed_by": "Python",
            "timestamp": time.time(),
            "message": "Hello from Python to JavaScript!"
        }
        
        # Send response back to JavaScript client
        await client.send_message(
            recipient=message.sender,
            intent="python_response",
            payload=response_data
        )
        
        logger.info(f"Sent response to JavaScript client")
    
    @client.message_handler(intent="cpp_message")
    async def handle_cpp_message(message):
        logger.info(f"Received message from C++ client: {message.payload}")
        
        # Process the message
        response_data = {
            "received": message.payload,
            "processed_by": "Python",
            "timestamp": time.time(),
            "message": "Hello from Python to C++!"
        }
        
        # Send response back to C++ client
        await client.send_message(
            recipient=message.sender,
            intent="python_response",
            payload=response_data
        )
        
        logger.info(f"Sent response to C++ client")
    
    # Periodic ping to other language clients
    async def ping_other_clients():
        while True:
            # Create a unique request ID
            request_id = str(uuid.uuid4())
            
            # Ping JavaScript client
            logger.info("Pinging JavaScript client...")
            await client.send_message(
                recipient="js_client",
                intent="python_message",
                payload={
                    "message": "Ping from Python!",
                    "timestamp": time.time(),
                    "request_id": request_id
                }
            )
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Ping C++ client
            logger.info("Pinging C++ client...")
            await client.send_message(
                recipient="cpp_client",
                intent="python_message",
                payload={
                    "message": "Ping from Python!",
                    "timestamp": time.time(),
                    "request_id": request_id
                }
            )
            
            # Wait before next ping cycle
            await asyncio.sleep(10)
    
    # Start pinging other clients
    asyncio.create_task(ping_other_clients())
    
    # Run the client
    try:
        logger.info("Python client is running...")
        await client.run()
    except KeyboardInterrupt:
        logger.info("Python client terminated by user")
    except Exception as e:
        logger.error(f"Python client error: {e}")
    finally:
        await client.disconnect()
        logger.info("Python client disconnected")

if __name__ == "__main__":
    asyncio.run(main())
