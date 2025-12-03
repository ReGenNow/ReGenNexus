"""
Basic Protocol Example - Simple Message Exchange

This example demonstrates the core functionality of the ReGenNexus UAP
without any LLM integration or advanced features. It shows how to:
1. Create protocol entities
2. Register them with the registry
3. Send and receive messages between entities
4. Handle basic context management
"""

import asyncio
from regennexus.protocol.protocol_core import Message, Entity, Intent
from regennexus.registry.registry import Registry
from regennexus.context.context_manager import ContextManager

# Create a simple entity class
class SimpleEntity(Entity):
    def __init__(self, entity_id, name):
        super().__init__(entity_id)
        self.name = name
        self.received_messages = []
        
    async def process_message(self, message, context):
        """Process an incoming message"""
        print(f"{self.name} received: {message.content}")
        self.received_messages.append(message)
        
        # If this is a query, send a response
        if message.intent == "query":
            response = Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content=f"Response from {self.name}: I received your query",
                intent="response",
                context_id=message.context_id
            )
            return response
        return None

async def main():
    # Create the registry and context manager
    registry = Registry()
    context_manager = ContextManager()
    
    # Create two simple entities
    entity_a = SimpleEntity("entity-a", "Entity A")
    entity_b = SimpleEntity("entity-b", "Entity B")
    
    # Register the entities
    await registry.register_entity(entity_a)
    await registry.register_entity(entity_b)
    
    # Create a new context
    context = await context_manager.create_context()
    
    # Entity A sends a message to Entity B
    message = Message(
        sender_id=entity_a.id,
        recipient_id=entity_b.id,
        content="Hello from Entity A!",
        intent="query",
        context_id=context.id
    )
    
    # Process the message through the registry
    response = await registry.route_message(message)
    
    # If there's a response, process it
    if response:
        await registry.route_message(response)
    
    # Print the conversation history from the context
    conversation = await context_manager.get_context(context.id)
    print("\nConversation History:")
    for msg in conversation.messages:
        print(f"From {msg.sender_id} to {msg.recipient_id}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())



