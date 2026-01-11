"""
Event-Driven Communication Example - Publish/Subscribe Pattern
This example demonstrates how to implement an event-driven communication pattern
using the ReGenNexus UAP core protocol. It shows how to:

1. Create publisher entities that broadcast events
2. Create subscriber entities that listen for specific events
3. Implement a publish/subscribe pattern without direct entity references
4. Handle event filtering and routing

This example showcases how the protocol can be used for event-driven architectures
without requiring LLM integration.
"""

import asyncio
import json
import random
from regennexus.core.protocol import RegenNexus
from regennexus.core.message import Message, Entity, Intent
from regennexus.core.registry import get_registry
# ContextManager not available - using basic pattern

# Event topics we'll use in this example
TOPICS = [
    "system.status",
    "sensor.temperature",
    "sensor.humidity",
    "user.login",
    "user.logout"
]

# Event broker entity that manages subscriptions and routes events
class EventBrokerEntity(Entity):
    def __init__(self, entity_id):
        super().__init__(entity_id)
        # Dictionary mapping topics to sets of subscriber IDs
        self.subscriptions = {topic: set() for topic in TOPICS}
        self.event_count = 0
        
    async def process_message(self, message, context):
        """Process subscription requests and events"""
        if message.intent == "broker.subscribe":
            return await self._handle_subscribe(message)
        elif message.intent == "broker.unsubscribe":
            return await self._handle_unsubscribe(message)
        elif message.intent == "broker.publish":
            return await self._handle_publish(message, context)
        elif message.intent == "broker.list_topics":
            return self._handle_list_topics(message)
        return None
    
    async def _handle_subscribe(self, message):
        """Handle a subscription request"""
        topic = message.content.get("topic")
        if not topic or topic not in self.subscriptions:
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"error": f"Invalid topic: {topic}"},
                intent="broker.error",
                context_id=message.context_id
            )
        
        # Add the subscriber to the topic
        self.subscriptions[topic].add(message.sender_id)
        print(f"Entity {message.sender_id} subscribed to topic {topic}")
        
        return Message(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content={"status": "subscribed", "topic": topic},
            intent="broker.subscribe.success",
            context_id=message.context_id
        )
    
    async def _handle_unsubscribe(self, message):
        """Handle an unsubscription request"""
        topic = message.content.get("topic")
        if not topic or topic not in self.subscriptions:
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"error": f"Invalid topic: {topic}"},
                intent="broker.error",
                context_id=message.context_id
            )
        
        # Remove the subscriber from the topic
        if message.sender_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(message.sender_id)
            print(f"Entity {message.sender_id} unsubscribed from topic {topic}")
        
        return Message(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content={"status": "unsubscribed", "topic": topic},
            intent="broker.unsubscribe.success",
            context_id=message.context_id
        )
    
    async def _handle_publish(self, message, context):
        """Handle an event publication"""
        topic = message.content.get("topic")
        event_data = message.content.get("data", {})
        
        if not topic or topic not in self.subscriptions:
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"error": f"Invalid topic: {topic}"},
                intent="broker.error",
                context_id=message.context_id
            )
        
        # Get subscribers for this topic
        subscribers = self.subscriptions[topic]
        if not subscribers:
            print(f"No subscribers for topic {topic}")
            return Message(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content={"status": "published", "subscribers": 0},
                intent="broker.publish.success",
                context_id=message.context_id
            )
        
        # Create event message
        event_id = f"event-{self.event_count}"
        self.event_count += 1
        
        event_message = Message(
            sender_id=self.id,
            # This will be set for each subscriber
            recipient_id="",
            content={
                "topic": topic,
                "data": event_data,
                "publisher": message.sender_id,
                "event_id": event_id,
                "timestamp": message.timestamp
            },
            intent="event.notification",
            context_id=message.context_id
        )
        
        # Broadcast to all subscribers
        for subscriber_id in subscribers:
            # Set the recipient for this copy of the message
            event_message.recipient_id = subscriber_id
            
            # In a real implementation, we would use a more efficient
            # broadcasting mechanism, but for this example we'll just
            # create individual messages
            await context.registry.route_message(event_message)
        
        print(f"Published event on topic {topic} to {len(subscribers)} subscribers")
        
        return Message(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content={
                "status": "published", 
                "subscribers": len(subscribers),
                "event_id": event_id
            },
            intent="broker.publish.success",
            context_id=message.context_id
        )
    
    def _handle_list_topics(self, message):
        """Handle a request to list available topics"""
        topic_info = {}
        for topic in self.subscriptions:
            topic_info[topic] = len(self.subscriptions[topic])
        
        return Message(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content={"topics": topic_info},
            intent="broker.topics.list",
            context_id=message.context_id
        )

# Publisher entity that generates events
class PublisherEntity(Entity):
    def __init__(self, entity_id, name, topics):
        super().__init__(entity_id)
        self.name = name
        self.topics = topics
        self.broker_id = None
        self.responses = []
    
    async def connect_to_broker(self, broker_id, registry, context_id):
        """Connect to the event broker"""
        self.broker_id = broker_id
        
        # Get the list of available topics
        list_topics_msg = Message(
            sender_id=self.id,
            recipient_id=broker_id,
            content={},
            intent="broker.list_topics",
            context_id=context_id
        )
        
        response = await registry.route_message(list_topics_msg)
        if response:
            self.responses.append(response)
            print(f"{self.name} received topics: {response.content.get('topics', {})}")
    
    async def publish_event(self, topic, data, registry, context_id):
        """Publish an event to a topic"""
        if not self.broker_id:
            print(f"{self.name} is not connected to a broker")
            return False
        
        if topic not in self.topics:
            print(f"{self.name} is not configured to publish to topic {topic}")
            return False
        
        publish_msg = Message(
            sender_id=self.id,
            recipient_id=self.broker_id,
            content={"topic": topic, "data": data},
            intent="broker.publish",
            context_id=context_id
        )
        
        response = await registry.route_message(publish_msg)
        if response:
            self.responses.append(response)
            if response.intent == "broker.publish.success":
                print(f"{self.name} published to {topic}, reached {response.content.get('subscribers', 0)} subscribers")
                return True
            else:
                print(f"{self.name} failed to publish: {response.content.get('error', 'Unknown error')}")
        
        return False
    
    async def process_message(self, message, context):
        """Process broker responses"""
        self.responses.append(message)
        return None

# Subscriber entity that receives events
class SubscriberEntity(Entity):
    def __init__(self, entity_id, name, interests):
        super().__init__(entity_id)
        self.name = name
        self.interests = interests  # Topics this subscriber is interested in
        self.broker_id = None
        self.received_events = []
        self.responses = []
    
    async def connect_to_broker(self, broker_id, registry, context_id):
        """Connect to the event broker and subscribe to topics"""
        self.broker_id = broker_id
        
        # Subscribe to all topics of interest
        for topic in self.interests:
            subscribe_msg = Message(
                sender_id=self.id,
                recipient_id=broker_id,
                content={"topic": topic},
                intent="broker.subscribe",
                context_id=context_id
            )
            
            response = await registry.route_message(subscribe_msg)
            if response:
                self.responses.append(response)
                if response.intent == "broker.subscribe.success":
                    print(f"{self.name} subscribed to {topic}")
                else:
                    print(f"{self.name} failed to subscribe to {topic}: {response.content.get('error', 'Unknown error')}")
    
    async def unsubscribe(self, topic, registry, context_id):
        """Unsubscribe from a topic"""
        if not self.broker_id:
            print(f"{self.name} is not connected to a broker")
            return False
        
        unsubscribe_msg = Message(
            sender_id=self.id,
            recipient_id=self.broker_id,
            content={"topic": topic},
            intent="broker.unsubscribe",
            context_id=context_id
        )
        
        response = await registry.route_message(unsubscribe_msg)
        if response:
            self.responses.append(response)
            if response.intent == "broker.unsubscribe.success":
                print(f"{self.name} unsubscribed from {topic}")
                if topic in self.interests:
                    self.interests.remove(topic)
                return True
            else:
                print(f"{self.name} failed to unsubscribe from {topic}: {response.content.get('error', 'Unknown error')}")
        
        return False
    
    async def process_message(self, message, context):
        """Process incoming events"""
        if message.intent == "event.notification":
            event_data = message.content
            print(f"{self.name} received event on topic {event_data.get('topic')}: {event_data.get('data')}")
            self.received_events.append(event_data)
        else:
            self.responses.append(message)
        return None

async def main():
    # Create ReGenNexus components
    registry = Registry()
    context_manager = ContextManager()
    
    # Create a context for our event-driven communication
    context = await context_manager.create_context()
    
    # Add registry to context for event broadcasting
    context.registry = registry
    
    # Create the event broker
    broker = EventBrokerEntity("event-broker")
    
    # Create publishers
    system_monitor = PublisherEntity("system-monitor", "System Monitor", ["system.status"])
    temp_sensor = PublisherEntity("temp-sensor", "Temperature Sensor", ["sensor.temperature"])
    humidity_sensor = PublisherEntity("humidity-sensor", "Humidity Sensor", ["sensor.humidity"])
    auth_service = PublisherEntity("auth-service", "Auth Service", ["user.login", "user.logout"])
    
    # Create subscribers
    dashboard = SubscriberEntity("dashboard", "Dashboard", ["system.status", "sensor.temperature", "sensor.humidity"])
    logger = SubscriberEntity("logger", "System Logger", ["system.status", "user.login", "user.logout"])
    alert_system = SubscriberEntity("alert-system", "Alert System", ["sensor.temperature"])
    
    # Register all entities
    await registry.register_entity(broker)
    await registry.register_entity(system_monitor)
    await registry.register_entity(temp_sensor)
    await registry.register_entity(humidity_sensor)
    await registry.register_entity(auth_service)
    await registry.register_entity(dashboard)
    await registry.register_entity(logger)
    await registry.register_entity(alert_system)
    
    # Connect publishers to broker
    print("\n=== Publishers connecting to broker ===")
    await system_monitor.connect_to_broker(broker.id, registry, context.id)
    await temp_sensor.connect_to_broker(broker.id, registry, context.id)
    await humidity_sensor.connect_to_broker(broker.id, registry, context.id)
    await auth_service.connect_to_broker(broker.id, registry, context.id)
    
    # Connect subscribers to broker and subscribe to topics
    print("\n=== Subscribers connecting to broker ===")
    await dashboard.connect_to_broker(broker.id, registry, context.id)
    await logger.connect_to_broker(broker.id, registry, context.id)
    await alert_system.connect_to_broker(broker.id, registry, context.id)
    
    # Publish some events
    print("\n=== Publishing events ===")
    await system_monitor.publish_event("system.status", {
        "cpu": 23.5,
        "memory": 45.2,
        "disk": 78.1,
        "status": "healthy"
    }, registry, context.id)
    
    await temp_sensor.publish_event("sensor.temperature", {
        "value": 22.5,
        "unit": "celsius",
        "location": "server-room"
    }, registry, context.id)
    
    await humidity_sensor.publish_event("sensor.humidity", {
        "value": 35.8,
        "unit": "percent",
        "location": "server-room"
    }, registry, context.id)
    
    await auth_service.publish_event("user.login", {
        "user_id": "user123",
        "timestamp": 1617293932,
        "ip": "192.168.1.100"
    }, registry, context.id)
    
    # Unsubscribe from a topic
    print("\n=== Unsubscribing from topics ===")
    await dashboard.unsubscribe("sensor.humidity", registry, context.id)
    
    # Publish another event to see the effect of unsubscribing
    print("\n=== Publishing after unsubscribe ===")
    await humidity_sensor.publish_event("sensor.humidity", {
        "value": 36.2,
        "unit": "percent",
        "location": "server-room"
    }, registry, context.id)
    
    # Print event counts for each subscriber
    print("\n=== Event Reception Summary ===")
    print(f"Dashboard received {len(dashboard.received_events)} events")
    print(f"Logger received {len(logger.received_events)} events")
    print(f"Alert System received {len(alert_system.received_events)} events")
    
    # Print detailed event information for one subscriber
    print("\n=== Dashboard Events ===")
    for i, event in enumerate(dashboard.received_events):
        print(f"Event {i+1}:")
        print(f"  Topic: {event.get('topic')}")
        print(f"  Publisher: {event.get('publisher')}")
        print(f"  Data: {json.dumps(event.get('data'), indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
