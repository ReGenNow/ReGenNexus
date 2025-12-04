"""
Multi-Entity Communication Example

This example demonstrates how to set up multiple entities that can communicate
with each other using the ReGenNexus UAP core protocol. This example shows:
1. Creating multiple entities with different capabilities
2. Setting up a communication network
3. Broadcasting messages to multiple recipients
4. Handling different message intents
"""

import asyncio
from regennexus.core.protocol import RegenNexus
from regennexus.core.message import Message, Entity, Intent
from regennexus.core.registry import get_registry
# ContextManager not available

# Define different entity types
class SensorEntity(Entity):
    def __init__(self, entity_id, sensor_type):
        super().__init__(entity_id)
        self.sensor_type = sensor_type
        self.value = 0
        
    async def process_message(self, message, context):
        if message.intent == "query.sensor_reading":
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"sensor_type": self.sensor_type, "value": self.value},
                intent="response.sensor_reading",
                context_id=message.context_id
            )
        elif message.intent == "command.update_value":
            self.value = message.content.get("value", 0)
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"status": "updated", "new_value": self.value},
                intent="response.update_confirmation",
                context_id=message.context_id
            )
        return None

class ControllerEntity(Entity):
    def __init__(self, entity_id, name):
        super().__init__(entity_id)
        self.name = name
        self.sensor_readings = {}
        
    async def process_message(self, message, context):
        if message.intent == "response.sensor_reading":
            sensor_data = message.content
            self.sensor_readings[message.sender_id] = sensor_data
            print(f"{self.name} received sensor reading from {message.sender_id}: {sensor_data}")
        elif message.intent == "response.update_confirmation":
            print(f"{self.name} received update confirmation: {message.content}")
        return None

async def main():
    # Create registry and context manager
    registry = Registry()
    context_manager = ContextManager()
    
    # Create entities
    temp_sensor = SensorEntity("temp-sensor-1", "temperature")
    humidity_sensor = SensorEntity("humidity-sensor-1", "humidity")
    controller = ControllerEntity("controller-1", "Main Controller")
    
    # Register entities
    await registry.register_entity(temp_sensor)
    await registry.register_entity(humidity_sensor)
    await registry.register_entity(controller)
    
    # Create context
    context = await context_manager.create_context()
    
    # Controller queries all sensors
    for sensor_id in ["temp-sensor-1", "humidity-sensor-1"]:
        query_message = Message(
            sender_id=controller.id,
            recipient_id=sensor_id,
            content={},
            intent="query.sensor_reading",
            context_id=context.id
        )
        response = await registry.route_message(query_message)
        if response:
            await registry.route_message(response)
    
    # Controller updates sensor values
    update_temp = Message(
        sender_id=controller.id,
        recipient_id="temp-sensor-1",
        content={"value": 25.5},
        intent="command.update_value",
        context_id=context.id
    )
    response = await registry.route_message(update_temp)
    if response:
        await registry.route_message(response)
    
    update_humidity = Message(
        sender_id=controller.id,
        recipient_id="humidity-sensor-1",
        content={"value": 60},
        intent="command.update_value",
        context_id=context.id
    )
    response = await registry.route_message(update_humidity)
    if response:
        await registry.route_message(response)
    
    # Query again to see updated values
    for sensor_id in ["temp-sensor-1", "humidity-sensor-1"]:
        query_message = Message(
            sender_id=controller.id,
            recipient_id=sensor_id,
            content={},
            intent="query.sensor_reading",
            context_id=context.id
        )
        response = await registry.route_message(query_message)
        if response:
            await registry.route_message(response)
    
    # Print final sensor readings
    print("\nFinal sensor readings:")
    for sensor_id, reading in controller.sensor_readings.items():
        print(f"{sensor_id}: {reading}")

if __name__ == "__main__":
    asyncio.run(main())
