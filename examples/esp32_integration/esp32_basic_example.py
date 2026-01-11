#!/usr/bin/env python3
"""
ESP32 Basic Integration Example

This example demonstrates how to use the ESP32 plugin with ReGenNexus Core
to communicate with an ESP32 device over serial connection.

Requirements:
- ESP32 device connected via USB
- ESP32 running MicroPython or Arduino firmware
"""

import asyncio
import logging
# UAP_Client not available - this example needs updating for new API
# ESP32Plugin not available - example for ESP32 MicroPython

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Create an ESP32 plugin
    # Note: Update the port to match your ESP32 device
    # Common ports:
    # - Linux: /dev/ttyUSB0 or /dev/ttyACM0
    # - Windows: COM3, COM4, etc.
    # - macOS: /dev/cu.SLAB_USBtoUART
    esp32_plugin = ESP32Plugin(
        port="/dev/ttyUSB0",  # Update this to match your system
        baud_rate=115200,
        plugin_id="esp32_device"
    )
    
    # Create a ReGenNexus client
    client = UAP_Client(entity_id="esp32_controller", registry_url="localhost:8000")
    
    # Register the ESP32 plugin with the client
    client.register_plugin(esp32_plugin)
    
    # Connect to the registry
    logger.info("Connecting to registry...")
    await client.connect()
    logger.info("Connected to registry")
    
    # Initialize the ESP32 plugin
    logger.info("Initializing ESP32 plugin...")
    init_result = await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="initialize"
    )
    logger.info(f"ESP32 initialization result: {init_result}")
    
    # Configure WiFi (if needed)
    # Uncomment and modify the following code to configure WiFi
    """
    logger.info("Configuring WiFi...")
    wifi_result = await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="configure_wifi",
        parameters={
            "ssid": "YourWiFiNetwork",
            "password": "YourWiFiPassword"
        }
    )
    logger.info(f"WiFi configuration result: {wifi_result}")
    """
    
    # Basic device information
    logger.info("Getting device information...")
    info_result = await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="get_device_info"
    )
    logger.info(f"Device information: {info_result}")
    
    # GPIO operations
    logger.info("Performing GPIO operations...")
    
    # Set GPIO pin 2 (built-in LED on many ESP32 boards) as output
    await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="set_pin_mode",
        parameters={"pin": 2, "mode": "OUTPUT"}
    )
    
    # Blink the LED 5 times
    for i in range(5):
        # Turn LED on
        logger.info(f"LED ON (cycle {i+1}/5)")
        await client.execute_plugin_action(
            plugin_id="esp32_device",
            action="digital_write",
            parameters={"pin": 2, "value": 1}
        )
        await asyncio.sleep(0.5)
        
        # Turn LED off
        logger.info(f"LED OFF (cycle {i+1}/5)")
        await client.execute_plugin_action(
            plugin_id="esp32_device",
            action="digital_write",
            parameters={"pin": 2, "value": 0}
        )
        await asyncio.sleep(0.5)
    
    # Read analog value from pin 34 (if available)
    try:
        analog_result = await client.execute_plugin_action(
            plugin_id="esp32_device",
            action="analog_read",
            parameters={"pin": 34}
        )
        logger.info(f"Analog reading from pin 34: {analog_result}")
    except Exception as e:
        logger.warning(f"Could not read analog value: {e}")
    
    # Read temperature from DHT sensor (if connected)
    try:
        dht_result = await client.execute_plugin_action(
            plugin_id="esp32_device",
            action="read_dht",
            parameters={"pin": 4, "sensor_type": "DHT22"}
        )
        logger.info(f"Temperature: {dht_result['temperature']}°C, Humidity: {dht_result['humidity']}%")
    except Exception as e:
        logger.warning(f"Could not read DHT sensor: {e}")
    
    # Send custom command to ESP32
    logger.info("Sending custom command...")
    command_result = await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="send_command",
        parameters={"command": "get_status"}
    )
    logger.info(f"Command result: {command_result}")
    
    # Deep sleep example (commented out to avoid disconnecting the device)
    """
    logger.info("Putting ESP32 into deep sleep for 10 seconds...")
    sleep_result = await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="deep_sleep",
        parameters={"sleep_time": 10}
    )
    logger.info(f"Deep sleep result: {sleep_result}")
    """
    
    # Clean up and disconnect
    logger.info("Shutting down ESP32 plugin...")
    shutdown_result = await client.execute_plugin_action(
        plugin_id="esp32_device",
        action="shutdown"
    )
    logger.info(f"Shutdown result: {shutdown_result}")
    
    # Disconnect from the registry
    logger.info("Disconnecting from registry...")
    await client.disconnect()
    logger.info("Disconnected from registry")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
