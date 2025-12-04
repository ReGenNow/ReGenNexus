#!/usr/bin/env python3
"""
Multi-Agent Sensor Network Example

This example demonstrates how multiple agents can form a sensor network
where each agent manages a different sensor and they collaborate to
provide a comprehensive environmental monitoring system.

Requirements:
- ReGenNexus Core installed
- Registry server running
"""

import asyncio
import logging
import random
import time
import json
# UAP_Client not available - this example needs updating for new API

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulated sensor data generators
def generate_temperature_data():
    """Generate simulated temperature data"""
    return round(20.0 + random.uniform(-5.0, 5.0), 1)  # 15-25°C

def generate_humidity_data():
    """Generate simulated humidity data"""
    return round(60.0 + random.uniform(-20.0, 20.0), 1)  # 40-80%

def generate_pressure_data():
    """Generate simulated pressure data"""
    return round(1013.0 + random.uniform(-10.0, 10.0), 1)  # 1003-1023 hPa

def generate_light_data():
    """Generate simulated light data"""
    return round(random.uniform(0, 1000), 1)  # 0-1000 lux

def generate_air_quality_data():
    """Generate simulated air quality data (CO2 in ppm)"""
    return round(400.0 + random.uniform(0, 200.0), 1)  # 400-600 ppm

# Temperature Sensor Agent
async def run_temperature_sensor(registry_url, sensor_id="temp_sensor"):
    # Create a client for the temperature sensor
    client = UAP_Client(entity_id=sensor_id, registry_url=registry_url)
    
    # Connect to the registry
    await client.connect()
    logger.info(f"{sensor_id} connected to registry")
    
    # Register with the aggregator
    await client.send_message(
        recipient="data_aggregator",
        intent="sensor_registration",
        payload={
            "sensor_id": sensor_id,
            "sensor_type": "temperature",
            "unit": "celsius",
            "update_frequency": 10,  # seconds
            "location": "living_room"
        }
    )
    
    # Register message handlers
    @client.message_handler(intent="data_request")
    async def handle_data_request(message):
        # Generate temperature data
        temperature = generate_temperature_data()
        
        # Send data to the requester
        await client.send_message(
            recipient=message.sender,
            intent="sensor_data",
            payload={
                "sensor_id": sensor_id,
                "sensor_type": "temperature",
                "value": temperature,
                "unit": "celsius",
                "timestamp": time.time(),
                "request_id": message.payload.get("request_id")
            }
        )
        
        logger.info(f"{sensor_id} sent temperature data: {temperature}°C")
    
    # Periodic data reporting
    async def report_data_periodically():
        while True:
            # Generate temperature data
            temperature = generate_temperature_data()
            
            # Send data to the aggregator
            await client.send_message(
                recipient="data_aggregator",
                intent="sensor_data",
                payload={
                    "sensor_id": sensor_id,
                    "sensor_type": "temperature",
                    "value": temperature,
                    "unit": "celsius",
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"{sensor_id} reported temperature: {temperature}°C")
            
            # Wait before next report
            await asyncio.sleep(10)
    
    # Start periodic reporting
    asyncio.create_task(report_data_periodically())
    
    # Run the sensor
    try:
        logger.info(f"{sensor_id} is running...")
        await client.run()
    except Exception as e:
        logger.error(f"{sensor_id} error: {e}")
    finally:
        await client.disconnect()

# Humidity Sensor Agent
async def run_humidity_sensor(registry_url, sensor_id="humidity_sensor"):
    # Create a client for the humidity sensor
    client = UAP_Client(entity_id=sensor_id, registry_url=registry_url)
    
    # Connect to the registry
    await client.connect()
    logger.info(f"{sensor_id} connected to registry")
    
    # Register with the aggregator
    await client.send_message(
        recipient="data_aggregator",
        intent="sensor_registration",
        payload={
            "sensor_id": sensor_id,
            "sensor_type": "humidity",
            "unit": "percent",
            "update_frequency": 15,  # seconds
            "location": "living_room"
        }
    )
    
    # Register message handlers
    @client.message_handler(intent="data_request")
    async def handle_data_request(message):
        # Generate humidity data
        humidity = generate_humidity_data()
        
        # Send data to the requester
        await client.send_message(
            recipient=message.sender,
            intent="sensor_data",
            payload={
                "sensor_id": sensor_id,
                "sensor_type": "humidity",
                "value": humidity,
                "unit": "percent",
                "timestamp": time.time(),
                "request_id": message.payload.get("request_id")
            }
        )
        
        logger.info(f"{sensor_id} sent humidity data: {humidity}%")
    
    # Periodic data reporting
    async def report_data_periodically():
        while True:
            # Generate humidity data
            humidity = generate_humidity_data()
            
            # Send data to the aggregator
            await client.send_message(
                recipient="data_aggregator",
                intent="sensor_data",
                payload={
                    "sensor_id": sensor_id,
                    "sensor_type": "humidity",
                    "value": humidity,
                    "unit": "percent",
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"{sensor_id} reported humidity: {humidity}%")
            
            # Wait before next report
            await asyncio.sleep(15)
    
    # Start periodic reporting
    asyncio.create_task(report_data_periodically())
    
    # Run the sensor
    try:
        logger.info(f"{sensor_id} is running...")
        await client.run()
    except Exception as e:
        logger.error(f"{sensor_id} error: {e}")
    finally:
        await client.disconnect()

# Pressure Sensor Agent
async def run_pressure_sensor(registry_url, sensor_id="pressure_sensor"):
    # Create a client for the pressure sensor
    client = UAP_Client(entity_id=sensor_id, registry_url=registry_url)
    
    # Connect to the registry
    await client.connect()
    logger.info(f"{sensor_id} connected to registry")
    
    # Register with the aggregator
    await client.send_message(
        recipient="data_aggregator",
        intent="sensor_registration",
        payload={
            "sensor_id": sensor_id,
            "sensor_type": "pressure",
            "unit": "hPa",
            "update_frequency": 20,  # seconds
            "location": "living_room"
        }
    )
    
    # Register message handlers
    @client.message_handler(intent="data_request")
    async def handle_data_request(message):
        # Generate pressure data
        pressure = generate_pressure_data()
        
        # Send data to the requester
        await client.send_message(
            recipient=message.sender,
            intent="sensor_data",
            payload={
                "sensor_id": sensor_id,
                "sensor_type": "pressure",
                "value": pressure,
                "unit": "hPa",
                "timestamp": time.time(),
                "request_id": message.payload.get("request_id")
            }
        )
        
        logger.info(f"{sensor_id} sent pressure data: {pressure} hPa")
    
    # Periodic data reporting
    async def report_data_periodically():
        while True:
            # Generate pressure data
            pressure = generate_pressure_data()
            
            # Send data to the aggregator
            await client.send_message(
                recipient="data_aggregator",
                intent="sensor_data",
                payload={
                    "sensor_id": sensor_id,
                    "sensor_type": "pressure",
                    "value": pressure,
                    "unit": "hPa",
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"{sensor_id} reported pressure: {pressure} hPa")
            
            # Wait before next report
            await asyncio.sleep(20)
    
    # Start periodic reporting
    asyncio.create_task(report_data_periodically())
    
    # Run the sensor
    try:
        logger.info(f"{sensor_id} is running...")
        await client.run()
    except Exception as e:
        logger.error(f"{sensor_id} error: {e}")
    finally:
        await client.disconnect()

# Data Aggregator Agent
async def run_data_aggregator(registry_url):
    # Create a client for the data aggregator
    client = UAP_Client(entity_id="data_aggregator", registry_url=registry_url)
    
    # Connect to the registry
    await client.connect()
    logger.info("Data Aggregator connected to registry")
    
    # Sensor registry and latest data
    sensors = {}
    latest_data = {}
    
    # Register message handlers
    @client.message_handler(intent="sensor_registration")
    async def handle_sensor_registration(message):
        sensor_id = message.payload.get("sensor_id")
        sensor_info = {
            "sensor_type": message.payload.get("sensor_type"),
            "unit": message.payload.get("unit"),
            "update_frequency": message.payload.get("update_frequency"),
            "location": message.payload.get("location"),
            "last_seen": time.time()
        }
        
        sensors[sensor_id] = sensor_info
        logger.info(f"Registered sensor: {sensor_id} ({sensor_info['sensor_type']})")
        
        # Acknowledge registration
        await client.send_message(
            recipient=sensor_id,
            intent="registration_confirmation",
            payload={
                "status": "success",
                "message": "Sensor registered successfully"
            }
        )
    
    @client.message_handler(intent="sensor_data")
    async def handle_sensor_data(message):
        sensor_id = message.payload.get("sensor_id")
        sensor_type = message.payload.get("sensor_type")
        value = message.payload.get("value")
        unit = message.payload.get("unit")
        timestamp = message.payload.get("timestamp")
        
        # Update sensor last seen time
        if sensor_id in sensors:
            sensors[sensor_id]["last_seen"] = time.time()
        
        # Store latest data
        if sensor_id not in latest_data:
            latest_data[sensor_id] = {}
        
        latest_data[sensor_id] = {
            "value": value,
            "unit": unit,
            "timestamp": timestamp,
            "sensor_type": sensor_type
        }
        
        # Log data receipt
        logger.info(f"Received {sensor_type} data from {sensor_id}: {value} {unit}")
        
        # Forward data to any subscribers
        # (In a real system, you might have a list of subscribers)
        await client.send_message(
            recipient="data_visualizer",
            intent="aggregated_sensor_data",
            payload={
                "sensor_id": sensor_id,
                "sensor_type": sensor_type,
                "value": value,
                "unit": unit,
                "timestamp": timestamp
            }
        )
    
    @client.message_handler(intent="get_all_sensor_data")
    async def handle_get_all_data(message):
        # Prepare current snapshot of all sensor data
        snapshot = {
            "timestamp": time.time(),
            "sensors": latest_data,
            "request_id": message.payload.get("request_id")
        }
        
        # Send response
        await client.send_message(
            recipient=message.sender,
            intent="all_sensor_data",
            payload=snapshot
        )
        
        logger.info(f"Sent all sensor data to {message.sender}")
    
    # Periodic sensor health check
    async def check_sensor_health():
        while True:
            current_time = time.time()
            for sensor_id, info in list(sensors.items()):
                # Check if sensor hasn't reported in 3x its update frequency
                max_silence = info["update_frequency"] * 3
                if current_time - info["last_seen"] > max_silence:
                    logger.warning(f"Sensor {sensor_id} may be offline (last seen {current_time - info['last_seen']} seconds ago)")
                    
                    # In a real system, you might take action here
                    # such as sending an alert or trying to reconnect
            
            # Wait before next check
            await asyncio.sleep(30)
    
    # Start health check
    asyncio.create_task(check_sensor_health())
    
    # Run the aggregator
    try:
        logger.info("Data Aggregator is running...")
        await client.run()
    except Exception as e:
        logger.error(f"Data Aggregator error: {e}")
    finally:
        await client.disconnect()

# Data Visualizer Agent
async def run_data_visualizer(registry_url):
    # Create a client for the data visualizer
    client = UAP_Client(entity_id="data_visualizer", registry_url=registry_url)
    
    # Connect to the registry
    await client.connect()
    logger.info("Data Visualizer connected to registry")
    
    # Data storage for visualization
    visualization_data = {
        "temperature": [],
        "humidity": [],
        "pressure": []
    }
    
    # Register message handlers
    @client.message_handler(intent="aggregated_sensor_data")
    async def handle_aggregated_data(message):
        sensor_type = message.payload.get("sensor_type")
        value = message.payload.get("value")
        timestamp = message.payload.get("timestamp")
        
        # Store data for visualization
        if sensor_type in visualization_data:
            # Keep only the last 100 readings
            if len(visualization_data[sensor_type]) >= 100:
                visualization_data[sensor_type].pop(0)
            
            visualization_data[sensor_type].append({
                "value": value,
                "timestamp": timestamp
            })
            
            logger.info(f"Added {sensor_type} data to visualization: {value}")
        
        # In a real system, this might update a web dashboard or plot
    
    @client.message_handler(intent="get_visualization_data")
    async def handle_get_visualization(message):
        # Prepare visualization data
        response = {
            "data": visualization_data,
            "timestamp": time.time(),
            "request_id": message.payload.get("request_id")
        }
        
        # Send response
        await client.send_message(
            recipient=message.sender,
            intent="visualization_data",
            payload=response
        )
        
        logger.info(f"Sent visualization data to {message.sender}")
    
    # Periodic data request (to ensure we have the latest data)
    async def request_all_data_periodically():
        while True:
            # Request all current sensor data
            await client.send_message(
                recipient="data_aggregator",
                intent="get_all_sensor_data",
                payload={
                    "request_id": f"vis_{int(time.time())}"
                }
            )
            
            # Wait before next request
            await asyncio.sleep(30)
    
    # Start periodic data requests
    asyncio.create_task(request_all_data_periodically())
    
    # Simulate dashboard updates
    async def update_dashboard():
        while True:
            # In a real system, this would update a web dashboard
            # Here we just log the current state
            logger.info("Dashboard update:")
            
            for sensor_type, readings in visualization_data.items():
                if readings:
                    latest = readings[-1]["value"]
                    logger.info(f"  - Latest {sensor_type}: {latest}")
                    
                    # Calculate average over last 5 readings
                    recent = readings[-5:] if len(readings) >= 5 else readings
                    avg = sum(r["value"] for r in recent) / len(recent)
                    logger.info(f"  - Average {sensor_type} (last 5): {avg:.1f}")
            
            # Wait before next update
            await asyncio.sleep(15)
    
    # Start dashboard updates
    asyncio.create_task(update_dashboard())
    
    # Run the visualizer
    try:
        logger.info("Data Visualizer is running...")
        await client.run()
    except Exception as e:
        logger.error(f"Data Visualizer error: {e}")
    finally:
        await client.disconnect()

# Main function
async def main():
    # Registry URL
    registry_url = "localhost:8000"
    
    # Start the data aggregator
    aggregator_task = asyncio.create_task(run_data_aggregator(registry_url))
    
    # Start the data visualizer
    visualizer_task = asyncio.create_task(run_data_visualizer(registry_url))
    
    # Give the aggregator and visualizer time to initialize
    await asyncio.sleep(2)
    
    # Start the sensor agents
    sensor_tasks = [
        asyncio.create_task(run_temperature_sensor(registry_url)),
        asyncio.create_task(run_humidity_sensor(registry_url)),
        asyncio.create_task(run_pressure_sensor(registry_url)),
    ]
    
    # Wait for all tasks to complete
    await asyncio.gather(aggregator_task, visualizer_task, *sensor_tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
