"""
ReGenNexus Core - Raspberry Pi GPIO Example

This example demonstrates how to use ReGenNexus Core with a Raspberry Pi
to control GPIO pins and interact with sensors.

Requirements:
- Raspberry Pi (any model)
- RPi.GPIO library installed
- ReGenNexus Core installed
- Optional: LED connected to GPIO pin 18
- Optional: Button connected to GPIO pin 17
- Optional: DHT22 temperature/humidity sensor on GPIO pin 4
"""

import asyncio
import json
import time
import argparse
from datetime import datetime
import logging

# ReGenNexus imports
# UAP_Client not available - this example needs updating for new API
from regennexus.core.message import Message

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Raspberry Pi specific libraries
try:
    import RPi.GPIO as GPIO
    import Adafruit_DHT
    RPI_AVAILABLE = True
except ImportError:
    logger.warning("RPi.GPIO or Adafruit_DHT not available. Running in simulation mode.")
    RPI_AVAILABLE = False

class RaspberryPiGPIOAgent:
    """Agent that controls GPIO pins on a Raspberry Pi."""
    
    def __init__(self, entity_id, registry_url, 
                 led_pin=18, button_pin=17, dht_pin=4):
        """
        Initialize the Raspberry Pi GPIO agent.
        
        Args:
            entity_id: Unique identifier for this agent
            registry_url: URL of the ReGenNexus registry
            led_pin: GPIO pin for LED
            button_pin: GPIO pin for button
            dht_pin: GPIO pin for DHT22 sensor
        """
        self.entity_id = entity_id
        self.registry_url = registry_url
        self.led_pin = led_pin
        self.button_pin = button_pin
        self.dht_pin = dht_pin
        
        self.client = None
        self.running = False
        self.button_state = False
        self.last_button_state = False
        self.monitoring_enabled = False
        self.subscribers = set()
        
    async def initialize(self):
        """Initialize the agent and connect to the registry."""
        # Initialize ReGenNexus client
        self.client = UAP_Client(entity_id=self.entity_id, registry_url=self.registry_url)
        await self.client.connect()
        
        # Register message handler
        self.client.register_message_handler(self.handle_message)
        
        # Initialize GPIO if available
        if RPI_AVAILABLE:
            self.init_gpio()
        
        logger.info(f"Raspberry Pi GPIO Agent initialized with ID: {self.entity_id}")
        
        # Send device info to registry
        device_info = self.get_device_info()
        await self.client.send_message(UAP_Message(
            sender=self.entity_id,
            recipient="registry",
            intent="device_info",
            payload=device_info
        ))
    
    def init_gpio(self):
        """Initialize GPIO pins."""
        try:
            # Set up GPIO using BCM numbering
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Set up LED pin as output
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
            
            # Set up button pin as input with pull-up resistor
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Add event detection for button
            GPIO.add_event_detect(self.button_pin, GPIO.BOTH, 
                                 callback=self.button_callback, 
                                 bouncetime=300)
            
            logger.info("GPIO pins initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing GPIO: {e}")
    
    def button_callback(self, channel):
        """
        Callback function for button state changes.
        
        Args:
            channel: GPIO channel that triggered the callback
        """
        if not RPI_AVAILABLE:
            return
            
        # Read button state (inverted because of pull-up)
        self.button_state = not GPIO.input(self.button_pin)
        
        # Only process if state has changed
        if self.button_state != self.last_button_state:
            logger.info(f"Button state changed to: {self.button_state}")
            self.last_button_state = self.button_state
            
            # Schedule async notification
            asyncio.create_task(self.notify_button_state())
    
    async def notify_button_state(self):
        """Notify subscribers of button state change."""
        if not self.client:
            return
            
        # Create message
        message = UAP_Message(
            sender=self.entity_id,
            recipient="*",  # Will be replaced with actual recipient
            intent="button_state",
            payload={
                "pin": self.button_pin,
                "state": self.button_state,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Send to all subscribers
        for subscriber in self.subscribers:
            message.recipient = subscriber
            await self.client.send_message(message)
    
    def get_device_info(self):
        """
        Get information about the Raspberry Pi.
        
        Returns:
            Dictionary of device information
        """
        info = {
            "device_type": "raspberry_pi",
            "pins": {
                "led": self.led_pin,
                "button": self.button_pin,
                "dht": self.dht_pin
            },
            "capabilities": ["gpio", "sensors", "actuators"]
        }
        
        # Add model info if available
        if RPI_AVAILABLE:
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    info["model"] = f.read().strip('\0')
            except:
                info["model"] = "Unknown Raspberry Pi"
                
            # Add temperature info
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read().strip()) / 1000.0
                    info["cpu_temperature"] = temp
            except:
                pass
        else:
            info["model"] = "Simulated Raspberry Pi"
            info["simulation_mode"] = True
        
        return info
    
    async def handle_message(self, message):
        """
        Handle incoming messages.
        
        Args:
            message: UAP_Message object
        """
        logger.info(f"Received message: {message.intent}")
        
        if message.intent == "led_control":
            # Control LED
            state = message.payload.get("state", False)
            await self.set_led_state(state)
            
            # Send confirmation
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="led_state",
                payload={
                    "pin": self.led_pin,
                    "state": state,
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
        elif message.intent == "read_button":
            # Read button state
            state = self.read_button_state()
            
            # Send state
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="button_state",
                payload={
                    "pin": self.button_pin,
                    "state": state,
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
        elif message.intent == "read_sensor":
            # Read sensor data
            sensor_data = await self.read_sensor_data()
            
            # Send data
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="sensor_data",
                payload=sensor_data
            ))
            
        elif message.intent == "subscribe":
            # Subscribe to events
            self.subscribers.add(message.sender)
            logger.info(f"Added subscriber: {message.sender}")
            
            # Send confirmation
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="subscription_confirmed",
                payload={
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
        elif message.intent == "unsubscribe":
            # Unsubscribe from events
            if message.sender in self.subscribers:
                self.subscribers.remove(message.sender)
                logger.info(f"Removed subscriber: {message.sender}")
            
            # Send confirmation
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="unsubscription_confirmed",
                payload={
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
        elif message.intent == "start_monitoring":
            # Start sensor monitoring
            interval = message.payload.get("interval", 60)  # Default: 60 seconds
            self.monitoring_enabled = True
            asyncio.create_task(self.monitor_sensors(message.sender, interval))
            
        elif message.intent == "stop_monitoring":
            # Stop sensor monitoring
            self.monitoring_enabled = False
    
    async def set_led_state(self, state):
        """
        Set the state of the LED.
        
        Args:
            state: Boolean indicating whether LED should be on or off
        """
        if RPI_AVAILABLE:
            try:
                GPIO.output(self.led_pin, GPIO.HIGH if state else GPIO.LOW)
                logger.info(f"LED state set to: {state}")
            except Exception as e:
                logger.error(f"Error setting LED state: {e}")
        else:
            logger.info(f"Simulated LED state set to: {state}")
    
    def read_button_state(self):
        """
        Read the current state of the button.
        
        Returns:
            Boolean indicating whether button is pressed
        """
        if RPI_AVAILABLE:
            try:
                # Inverted because of pull-up resistor
                state = not GPIO.input(self.button_pin)
                return state
            except Exception as e:
                logger.error(f"Error reading button state: {e}")
                return False
        else:
            # In simulation mode, alternate the state
            self.button_state = not self.button_state
            return self.button_state
    
    async def read_sensor_data(self):
        """
        Read data from the DHT22 sensor.
        
        Returns:
            Dictionary of sensor data
        """
        data = {
            "timestamp": datetime.now().isoformat()
        }
        
        if RPI_AVAILABLE:
            try:
                # Read DHT22 sensor
                humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT22, self.dht_pin)
                
                if humidity is not None and temperature is not None:
                    data["temperature"] = round(temperature, 1)
                    data["humidity"] = round(humidity, 1)
                    logger.info(f"Sensor reading: {temperature}°C, {humidity}%")
                else:
                    logger.warning("Failed to read from DHT sensor")
                    data["error"] = "Failed to read from sensor"
            except Exception as e:
                logger.error(f"Error reading sensor: {e}")
                data["error"] = str(e)
        else:
            # Simulate sensor data
            import random
            data["temperature"] = round(20 + random.uniform(-5, 5), 1)
            data["humidity"] = round(50 + random.uniform(-10, 10), 1)
            logger.info(f"Simulated sensor reading: {data['temperature']}°C, {data['humidity']}%")
        
        # Add CPU temperature if available
        if RPI_AVAILABLE:
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read().strip()) / 1000.0
                    data["cpu_temperature"] = round(temp, 1)
            except:
                pass
        
        return data
    
    async def monitor_sensors(self, recipient, interval):
        """
        Periodically monitor sensors and send data.
        
        Args:
            recipient: Entity ID of the recipient
            interval: Time between readings in seconds
        """
        logger.info(f"Starting sensor monitoring with interval: {interval}s")
        
        while self.monitoring_enabled:
            # Read sensor data
            sensor_data = await self.read_sensor_data()
            
            # Send data
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=recipient,
                intent="sensor_data",
                payload=sensor_data
            ))
            
            # Wait for next interval
            await asyncio.sleep(interval)
        
        logger.info("Sensor monitoring stopped")
    
    async def run(self):
        """Run the agent."""
        try:
            await self.initialize()
            
            # Keep the agent running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Clean up resources
            if RPI_AVAILABLE:
                GPIO.cleanup()
            
            if self.client:
                await self.client.disconnect()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Raspberry Pi GPIO Agent')
    parser.add_argument('--entity-id', type=str, default='raspberry_pi_gpio',
                        help='Entity ID for this agent')
    parser.add_argument('--registry', type=str, default='localhost:8000',
                        help='URL of the ReGenNexus registry')
    parser.add_argument('--led-pin', type=int, default=18,
                        help='GPIO pin for LED')
    parser.add_argument('--button-pin', type=int, default=17,
                        help='GPIO pin for button')
    parser.add_argument('--dht-pin', type=int, default=4,
                        help='GPIO pin for DHT22 sensor')
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = RaspberryPiGPIOAgent(
        entity_id=args.entity_id,
        registry_url=args.registry,
        led_pin=args.led_pin,
        button_pin=args.button_pin,
        dht_pin=args.dht_pin
    )
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
