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
from regennexus.core.message import Message
from regennexus.core.entity import Entity

# Simple context store (in-memory for demo)
context_store = {}

class SimpleEntity(Entity):
    def __init__(self, entity_id, name):
        super().__init__(entity_id)
        self.name = name

    async def process_message(self, message, context):
        print(f"{self.name} received: {message.content}")
        context.setdefault("messages", []).append(message)

        # Auto-response to "query"
        if message.intent == "query":
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content=f"Reply from {self.name}",
                intent="response",
                context_id=message.context_id
            )

async def main():
    # Setup fake context and entities
    context_id = "demo-context"
    context = {"id": context_id, "messages": []}
    context_store[context_id] = context

    entity_a = SimpleEntity("entity-a", "Entity A")
    entity_b = SimpleEntity("entity-b", "Entity B")

    # Entity A sends a query to Entity B
    msg = await entity_a.send_message(
        recipient_id="entity-b",
        content="Hello from A",
        intent="query",
        context_id=context_id
    )

    # Route message manually
    response = await entity_b.process_message(msg, context)

    if response:
        await entity_a.process_message(response, context)

    print("\nConversation History:")
    for m in context["messages"]:
        print(f"{m.sender_id} → {m.recipient_id}: {m.content}")

if __name__ == "__main__":
    asyncio.run(main())