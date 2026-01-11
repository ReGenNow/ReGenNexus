"""
Protocol Integration Example

This example demonstrates how to integrate the ReGenNexus UAP core protocol
with an existing application. It shows:
1. Creating custom entity types for your application
2. Handling protocol messages within your application logic
3. Using the context manager to maintain conversation state
4. Implementing custom message intents
"""

import asyncio
import json
from regennexus.core.protocol import RegenNexus
from regennexus.core.message import Message, Entity, Intent
from regennexus.core.registry import get_registry
# ContextManager not available

# Example application class that will integrate with ReGenNexus
class WeatherApplication:
    def __init__(self):
        self.locations = {
            "new_york": {"temp": 22, "conditions": "sunny"},
            "london": {"temp": 15, "conditions": "rainy"},
            "tokyo": {"temp": 28, "conditions": "cloudy"}
        }
    
    def get_weather(self, location):
        """Get weather for a specific location"""
        return self.locations.get(location.lower(), {"temp": 0, "conditions": "unknown"})
    
    def update_weather(self, location, data):
        """Update weather for a location"""
        if location.lower() in self.locations:
            self.locations[location.lower()].update(data)
            return True
        return False

# Create a ReGenNexus entity that wraps our application
class WeatherServiceEntity(Entity):
    def __init__(self, entity_id, weather_app):
        super().__init__(entity_id)
        self.weather_app = weather_app
        
    async def process_message(self, message, context):
        """Process incoming protocol messages"""
        if message.intent == "query.weather":
            # Extract location from message
            location = message.content.get("location", "")
            if not location:
                return self._create_error_response(message, "Location is required")
            
            # Get weather data from our application
            weather_data = self.weather_app.get_weather(location)
            
            # Create and return a response message
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={
                    "location": location,
                    "temperature": weather_data["temp"],
                    "conditions": weather_data["conditions"]
                },
                intent="response.weather",
                context_id=message.context_id
            )
        
        elif message.intent == "command.update_weather":
            # Extract data from message
            location = message.content.get("location", "")
            data = {
                "temp": message.content.get("temperature"),
                "conditions": message.content.get("conditions")
            }
            
            # Update weather in our application
            success = self.weather_app.update_weather(location, data)
            
            # Create and return a response message
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"success": success, "location": location},
                intent="response.update_result",
                context_id=message.context_id
            )
        
        return None
    
    def _create_error_response(self, original_message, error_text):
        """Helper to create error response messages"""
        return Message(
            sender_id=self.id,
            recipient_id=original_message.sender_id,
            content={"error": error_text},
            intent="response.error",
            context_id=original_message.context_id
        )

# Client entity that will communicate with our weather service
class WeatherClientEntity(Entity):
    def __init__(self, entity_id, name):
        super().__init__(entity_id)
        self.name = name
        self.responses = []
        
    async def process_message(self, message, context):
        """Process responses from the weather service"""
        print(f"{self.name} received: {json.dumps(message.content, indent=2)}")
        self.responses.append(message)
        return None

async def main():
    # Create our application
    weather_app = WeatherApplication()
    
    # Create ReGenNexus components
    registry = Registry()
    context_manager = ContextManager()
    
    # Create and register entities
    weather_service = WeatherServiceEntity("weather-service", weather_app)
    client = WeatherClientEntity("weather-client", "Weather Client")
    
    await registry.register_entity(weather_service)
    await registry.register_entity(client)
    
    # Create a context for our conversation
    context = await context_manager.create_context()
    
    # Query weather for New York
    query_message = Message(
        sender_id=client.id,
        recipient_id=weather_service.id,
        content={"location": "New York"},
        intent="query.weather",
        context_id=context.id
    )
    
    response = await registry.route_message(query_message)
    if response:
        await registry.route_message(response)
    
    # Update weather for London
    update_message = Message(
        sender_id=client.id,
        recipient_id=weather_service.id,
        content={
            "location": "London",
            "temperature": 18,
            "conditions": "partly cloudy"
        },
        intent="command.update_weather",
        context_id=context.id
    )
    
    response = await registry.route_message(update_message)
    if response:
        await registry.route_message(response)
    
    # Query updated weather for London
    query_message = Message(
        sender_id=client.id,
        recipient_id=weather_service.id,
        content={"location": "London"},
        intent="query.weather",
        context_id=context.id
    )
    
    response = await registry.route_message(query_message)
    if response:
        await registry.route_message(response)
    
    # Print conversation history
    conversation = await context_manager.get_context(context.id)
    print("\nConversation History:")
    for i, msg in enumerate(conversation.messages):
        print(f"Message {i+1}: {msg.intent} from {msg.sender_id} to {msg.recipient_id}")

if __name__ == "__main__":
    asyncio.run(main())
