# regennexus/protocol/cli.py
import asyncio
import websockets
import json
from .client import UAP_Client

def main():
    # Create a client instance pointing to the WebSocket registry
    client = UAP_Client(entity_id="cli_test_client", registry_url="ws://localhost:8000")
    
    # Run the client in an async context
    async def run_client():
        # Connect to the registry
        connected = await client.connect()
        if not connected:
            print("ReGenNexus Client CLI: Failed to connect to registry")
            return
        
        print("ReGenNexus Client CLI: Connected to registry")
        
        # Register some test capabilities
        capabilities = ["test_capability"]
        await client.register_capabilities(capabilities)
        print(f"ReGenNexus Client CLI: Registered capabilities: {capabilities}")
        
        # Send a test message (broadcast)
        message = {
            "recipient": "*",  # Broadcast to all entities
            "intent": "test",
            "payload": {"message": "Hello from CLI"}
        }
        await client.send_message(message)
        print("ReGenNexus Client CLI: Sent test message")
        
        # Disconnect
        await client.disconnect()
        print("ReGenNexus Client CLI: Disconnected")

    # Run the async function
    asyncio.run(run_client())

if __name__ == "__main__":
    main()