#!/usr/bin/env python3
"""
ESP32 Web Server Example

This example demonstrates how to use the ESP32 plugin with ReGenNexus Core
to create a web server that exposes device functionality through a REST API.

Requirements:
- ESP32 device connected via USB
- ESP32 running MicroPython or Arduino firmware
- FastAPI and uvicorn installed (pip install fastapi uvicorn)
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
# UAP_Client not available - this example needs updating for new API
# ESP32Plugin not available - example for ESP32 MicroPython

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="ESP32 Web Server", description="REST API for ESP32 device control")

# Global variables
client = None
esp32_plugin = None

# Data models
class PinConfig(BaseModel):
    pin: int
    mode: str  # "INPUT", "OUTPUT", "INPUT_PULLUP"

class PinValue(BaseModel):
    pin: int
    value: int  # 0 or 1 for digital, 0-4095 for analog

class WiFiConfig(BaseModel):
    ssid: str
    password: str

class SensorReading(BaseModel):
    sensor_type: str
    pin: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None

class DeepSleepConfig(BaseModel):
    sleep_time: int  # seconds

# Initialize the ESP32 plugin and client
async def initialize():
    global client, esp32_plugin
    
    # Create an ESP32 plugin
    esp32_plugin = ESP32Plugin(
        port="/dev/ttyUSB0",  # Update this to match your system
        baud_rate=115200,
        plugin_id="esp32_webserver"
    )
    
    # Create a ReGenNexus client
    client = UAP_Client(entity_id="esp32_web_controller", registry_url="localhost:8000")
    
    # Register the ESP32 plugin with the client
    client.register_plugin(esp32_plugin)
    
    # Connect to the registry
    logger.info("Connecting to registry...")
    await client.connect()
    logger.info("Connected to registry")
    
    # Initialize the ESP32 plugin
    logger.info("Initializing ESP32 plugin...")
    init_result = await client.execute_plugin_action(
        plugin_id="esp32_webserver",
        action="initialize"
    )
    logger.info(f"ESP32 initialization result: {init_result}")
    
    return init_result

# Shutdown the ESP32 plugin and client
async def shutdown():
    global client, esp32_plugin
    
    if client:
        # Shutdown the ESP32 plugin
        logger.info("Shutting down ESP32 plugin...")
        await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="shutdown"
        )
        
        # Disconnect from the registry
        logger.info("Disconnecting from registry...")
        await client.disconnect()
        logger.info("Disconnected from registry")

# API routes
@app.get("/")
async def root():
    return {"message": "ESP32 Web Server API", "status": "running"}

@app.get("/device/info")
async def get_device_info():
    try:
        info_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="get_device_info"
        )
        return info_result
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/device/wifi")
async def configure_wifi(config: WiFiConfig):
    try:
        wifi_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="configure_wifi",
            parameters={
                "ssid": config.ssid,
                "password": config.password
            }
        )
        return wifi_result
    except Exception as e:
        logger.error(f"Error configuring WiFi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gpio/mode")
async def set_pin_mode(config: PinConfig):
    try:
        mode_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="set_pin_mode",
            parameters={"pin": config.pin, "mode": config.mode}
        )
        return mode_result
    except Exception as e:
        logger.error(f"Error setting pin mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gpio/digital/write")
async def digital_write(pin_value: PinValue):
    try:
        write_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="digital_write",
            parameters={"pin": pin_value.pin, "value": pin_value.value}
        )
        return write_result
    except Exception as e:
        logger.error(f"Error writing to digital pin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpio/digital/read/{pin}")
async def digital_read(pin: int):
    try:
        read_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="digital_read",
            parameters={"pin": pin}
        )
        return read_result
    except Exception as e:
        logger.error(f"Error reading from digital pin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpio/analog/read/{pin}")
async def analog_read(pin: int):
    try:
        read_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="analog_read",
            parameters={"pin": pin}
        )
        return read_result
    except Exception as e:
        logger.error(f"Error reading from analog pin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sensor/read")
async def read_sensor(sensor: SensorReading):
    try:
        if sensor.sensor_type == "dht":
            # Read DHT temperature/humidity sensor
            sensor_result = await client.execute_plugin_action(
                plugin_id="esp32_webserver",
                action="read_dht",
                parameters={
                    "pin": sensor.pin,
                    "sensor_type": sensor.parameters.get("sensor_type", "DHT22")
                }
            )
        elif sensor.sensor_type == "bmp280":
            # Read BMP280 pressure sensor
            sensor_result = await client.execute_plugin_action(
                plugin_id="esp32_webserver",
                action="read_bmp280",
                parameters={
                    "sda_pin": sensor.parameters.get("sda_pin", 21),
                    "scl_pin": sensor.parameters.get("scl_pin", 22)
                }
            )
        elif sensor.sensor_type == "ds18b20":
            # Read DS18B20 temperature sensor
            sensor_result = await client.execute_plugin_action(
                plugin_id="esp32_webserver",
                action="read_ds18b20",
                parameters={"pin": sensor.pin}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported sensor type: {sensor.sensor_type}")
        
        return sensor_result
    except Exception as e:
        logger.error(f"Error reading sensor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/device/deep_sleep")
async def deep_sleep(config: DeepSleepConfig):
    try:
        sleep_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="deep_sleep",
            parameters={"sleep_time": config.sleep_time}
        )
        return sleep_result
    except Exception as e:
        logger.error(f"Error setting deep sleep: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/device/command")
async def send_command(command: Dict[str, Any]):
    try:
        command_result = await client.execute_plugin_action(
            plugin_id="esp32_webserver",
            action="send_command",
            parameters={"command": command.get("command"), "parameters": command.get("parameters")}
        )
        return command_result
    except Exception as e:
        logger.error(f"Error sending command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown()

# Main function to run the server
def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
