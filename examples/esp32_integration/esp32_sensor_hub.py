#!/usr/bin/env python3
"""
ESP32 IoT Sensor Hub Example

This example demonstrates how to use the ESP32 plugin with ReGenNexus Core
to create an IoT sensor hub that collects data from multiple sensors
and publishes it to both local clients and cloud services.

Requirements:
- ESP32 device connected via USB
- ESP32 running MicroPython or Arduino firmware with appropriate sensors
"""

import asyncio
import logging
import json
import time
# UAP_Client not available - this example needs updating for new API
# ESP32Plugin not available - example for ESP32 MicroPython
from regennexus.bridges.azure_bridge import AzureBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample sensor data storage
sensor_data_history = []

async def main():
    # Create an ESP32 plugin
    esp32_plugin = ESP32Plugin(
        port="/dev/ttyUSB0",  # Update this to match your system
        baud_rate=115200,
        plugin_id="esp32_sensor_hub"
    )
    
    # Create an Azure bridge (optional - for cloud connectivity)
    # Uncomment and configure if you want to send data to Azure IoT Hub
    """
    azure_bridge = AzureBridge(
        connection_string="HostName=your-hub.azure-devices.net;DeviceId=your-device;SharedAccessKey=your-key",
        bridge_id="azure_iot"
    )
    """
    
    # Create a ReGenNexus client
    client = UAP_Client(entity_id="sensor_hub_controller", registry_url="localhost:8000")
    
    # Register the ESP32 plugin with the client
    client.register_plugin(esp32_plugin)
    
    # Register the Azure bridge (if using)
    # client.register_bridge(azure_bridge)
    
    # Connect to the registry
    logger.info("Connecting to registry...")
    await client.connect()
    logger.info("Connected to registry")
    
    # Initialize the ESP32 plugin
    logger.info("Initializing ESP32 plugin...")
    await client.execute_plugin_action(
        plugin_id="esp32_sensor_hub",
        action="initialize"
    )
    
    # Initialize the Azure bridge (if using)
    """
    logger.info("Initializing Azure bridge...")
    await azure_bridge.initialize()
    """
    
    # Configure WiFi on the ESP32
    logger.info("Configuring WiFi...")
    await client.execute_plugin_action(
        plugin_id="esp32_sensor_hub",
        action="configure_wifi",
        parameters={
            "ssid": "YourWiFiNetwork",
            "password": "YourWiFiPassword"
        }
    )
    
    # Register message handler for sensor data requests
    @client.message_handler(intent="request_sensor_data")
    async def handle_sensor_data_request(message):
        logger.info(f"Received sensor data request from {message.sender}")
        
        # Collect latest sensor data
        latest_data = await collect_sensor_data(client)
        
        # Send response with sensor data
        response = {
            "timestamp": time.time(),
            "sensor_data": latest_data,
            "device_id": "esp32_sensor_hub"
        }
        
        await client.send_message(
            recipient=message.sender,
            intent="sensor_data_response",
            payload=response,
            context_id=message.context_id
        )
    
    # Register message handler for sensor history requests
    @client.message_handler(intent="request_sensor_history")
    async def handle_history_request(message):
        logger.info(f"Received sensor history request from {message.sender}")
        
        # Send response with sensor history
        response = {
            "timestamp": time.time(),
            "sensor_history": sensor_data_history,
            "device_id": "esp32_sensor_hub"
        }
        
        await client.send_message(
            recipient=message.sender,
            intent="sensor_history_response",
            payload=response,
            context_id=message.context_id
        )
    
    # Main sensor monitoring loop
    try:
        logger.info("Starting sensor monitoring loop...")
        while True:
            # Collect sensor data
            sensor_data = await collect_sensor_data(client)
            
            # Store in history (limited to last 100 readings)
            sensor_data_history.append(sensor_data)
            if len(sensor_data_history) > 100:
                sensor_data_history.pop(0)
            
            # Log the data
            logger.info(f"Sensor readings: {json.dumps(sensor_data, indent=2)}")
            
            # Send to Azure IoT Hub (if configured)
            """
            await client.execute_bridge_action(
                bridge_id="azure_iot",
                action="send_telemetry",
                parameters={"data": sensor_data}
            )
            """
            
            # Wait before next reading
            await asyncio.sleep(60)  # Collect data every minute
    
    except KeyboardInterrupt:
        logger.info("Sensor monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Error in sensor monitoring loop: {e}")
    finally:
        # Clean up and disconnect
        logger.info("Shutting down ESP32 plugin...")
        await client.execute_plugin_action(
            plugin_id="esp32_sensor_hub",
            action="shutdown"
        )
        
        # Disconnect from the registry
        logger.info("Disconnecting from registry...")
        await client.disconnect()
        logger.info("Disconnected from registry")

async def collect_sensor_data(client):
    """Collect data from all sensors connected to the ESP32"""
    sensor_data = {}
    
    try:
        # Read temperature and humidity from DHT22 sensor
        dht_result = await client.execute_plugin_action(
            plugin_id="esp32_sensor_hub",
            action="read_dht",
            parameters={"pin": 4, "sensor_type": "DHT22"}
        )
        sensor_data["temperature"] = dht_result["temperature"]
        sensor_data["humidity"] = dht_result["humidity"]
    except Exception as e:
        logger.warning(f"Could not read DHT sensor: {e}")
    
    try:
        # Read barometric pressure from BMP280 sensor
        bmp_result = await client.execute_plugin_action(
            plugin_id="esp32_sensor_hub",
            action="read_bmp280",
            parameters={"sda_pin": 21, "scl_pin": 22}
        )
        sensor_data["pressure"] = bmp_result["pressure"]
        sensor_data["altitude"] = bmp_result["altitude"]
    except Exception as e:
        logger.warning(f"Could not read BMP280 sensor: {e}")
    
    try:
        # Read light level from analog pin
        light_result = await client.execute_plugin_action(
            plugin_id="esp32_sensor_hub",
            action="analog_read",
            parameters={"pin": 34}
        )
        sensor_data["light_level"] = light_result["value"]
    except Exception as e:
        logger.warning(f"Could not read light sensor: {e}")
    
    try:
        # Read soil moisture from analog pin
        moisture_result = await client.execute_plugin_action(
            plugin_id="esp32_sensor_hub",
            action="analog_read",
            parameters={"pin": 35}
        )
        sensor_data["soil_moisture"] = moisture_result["value"]
    except Exception as e:
        logger.warning(f"Could not read soil moisture sensor: {e}")
    
    # Add timestamp
    sensor_data["timestamp"] = time.time()
    
    return sensor_data

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
