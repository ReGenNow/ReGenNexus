"""
ReGenNexus Core - Raspberry Pi Pico Example

This example demonstrates how to use ReGenNexus Core with a Raspberry Pi Pico
microcontroller via serial communication.

Requirements:
- Raspberry Pi (as the host)
- Raspberry Pi Pico connected via USB
- MicroPython installed on the Pico
- ReGenNexus Core installed on the host
- pyserial library installed on the host
"""

import asyncio
import json
import time
import argparse
from datetime import datetime
import logging
import serial
import os

# ReGenNexus imports
from regennexus.protocol.client import UAP_Client
from regennexus.protocol.message import UAP_Message

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PicoSerialAgent:
    """Agent that communicates with a Raspberry Pi Pico via serial."""
    
    def __init__(self, entity_id, registry_url, serial_port='/dev/ttyACM0', baud_rate=115200):
        """
        Initialize the Pico Serial agent.
        
        Args:
            entity_id: Unique identifier for this agent
            registry_url: URL of the ReGenNexus registry
            serial_port: Serial port for Pico communication
            baud_rate: Baud rate for serial communication
        """
        self.entity_id = entity_id
        self.registry_url = registry_url
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        self.client = None
        self.serial = None
        self.running = False
        self.subscribers = set()
        
        # Pico MicroPython script to upload
        self.micropython_script = """
# MicroPython script for Raspberry Pi Pico
# This script reads sensors and responds to commands from the host

import machine
import time
import json
import _thread
import sys

# Set up LED
led = machine.Pin(25, machine.Pin.OUT)  # Onboard LED

# Set up ADC for temperature sensor (internal)
temp_sensor = machine.ADC(4)
conversion_factor = 3.3 / 65535

# Set up a button on GP15 with pull-up
button = machine.Pin(15, machine.Pin.IN, machine.Pin.PULL_UP)

# Set up an external sensor on ADC0 (GP26)
adc0 = machine.ADC(26)

# Lock for thread synchronization
lock = _thread.allocate_lock()

# Global variables
button_state = False
last_button_state = False

def read_temperature():
    """Read the internal temperature sensor."""
    adc_value = temp_sensor.read_u16()
    voltage = adc_value * conversion_factor
    temperature = 27 - (voltage - 0.706) / 0.001721
    return round(temperature, 1)

def read_adc():
    """Read the external ADC sensor."""
    adc_value = adc0.read_u16()
    voltage = adc_value * conversion_factor
    return round(voltage, 3)

def button_monitor():
    """Monitor button state changes in a separate thread."""
    global button_state, last_button_state
    
    while True:
        # Read button state (inverted because of pull-up)
        current_state = not button.value()
        
        # Check if state changed
        if current_state != last_button_state:
            with lock:
                button_state = current_state
                last_button_state = current_state
            
            # Send button state change notification
            message = {
                "type": "event",
                "event": "button_state",
                "state": current_state,
                "timestamp": time.time()
            }
            print(json.dumps(message))
        
        time.sleep(0.1)

def process_command(command):
    """Process a command from the host."""
    try:
        cmd = json.loads(command)
        cmd_type = cmd.get("type", "")
        
        if cmd_type == "led":
            # Control LED
            state = cmd.get("state", False)
            led.value(1 if state else 0)
            
            # Send confirmation
            response = {
                "type": "response",
                "command": "led",
                "state": state,
                "success": True
            }
            print(json.dumps(response))
            
        elif cmd_type == "read":
            # Read sensors
            temp = read_temperature()
            adc = read_adc()
            
            with lock:
                btn = button_state
            
            # Send sensor data
            response = {
                "type": "response",
                "command": "read",
                "temperature": temp,
                "adc_voltage": adc,
                "button_state": btn,
                "timestamp": time.time()
            }
            print(json.dumps(response))
            
        elif cmd_type == "ping":
            # Simple ping command
            response = {
                "type": "response",
                "command": "ping",
                "timestamp": time.time()
            }
            print(json.dumps(response))
            
        else:
            # Unknown command
            response = {
                "type": "error",
                "error": "Unknown command",
                "command": cmd_type
            }
            print(json.dumps(response))
            
    except Exception as e:
        # Error processing command
        response = {
            "type": "error",
            "error": str(e),
            "command": command
        }
        print(json.dumps(response))

# Start button monitoring in a separate thread
_thread.start_new_thread(button_monitor, ())

# Main loop
print('{"type":"status","status":"ready"}')

while True:
    try:
        command = sys.stdin.readline().strip()
        if command:
            process_command(command)
    except Exception as e:
        print('{{"type":"error","error":"{}"}}'.format(str(e)))
        time.sleep(1)
"""
        
    async def initialize(self):
        """Initialize the agent and connect to the registry."""
        # Initialize ReGenNexus client
        self.client = UAP_Client(entity_id=self.entity_id, registry_url=self.registry_url)
        await self.client.connect()
        
        # Register message handler
        self.client.register_message_handler(self.handle_message)
        
        # Initialize serial connection
        await self.connect_to_pico()
        
        logger.info(f"Pico Serial Agent initialized with ID: {self.entity_id}")
        
        # Send device info to registry
        device_info = await self.get_device_info()
        await self.client.send_message(UAP_Message(
            sender=self.entity_id,
            recipient="registry",
            intent="device_info",
            payload=device_info
        ))
    
    async def connect_to_pico(self):
        """Connect to the Raspberry Pi Pico via serial."""
        try:
            # Check if serial port exists
            if not os.path.exists(self.serial_port):
                logger.warning(f"Serial port {self.serial_port} not found. Running in simulation mode.")
                return
            
            # Open serial connection
            self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            logger.info(f"Connected to Pico on {self.serial_port}")
            
            # Wait for device to initialize
            await asyncio.sleep(2)
            
            # Flush input buffer
            self.serial.reset_input_buffer()
            
            # Upload MicroPython script
            await self.upload_script()
            
            # Start serial monitoring
            self.running = True
            asyncio.create_task(self.monitor_serial())
            
        except Exception as e:
            logger.error(f"Error connecting to Pico: {e}")
            self.serial = None
    
    async def upload_script(self):
        """Upload MicroPython script to the Pico."""
        if not self.serial:
            return
            
        try:
            logger.info("Uploading MicroPython script to Pico...")
            
            # Enter REPL mode
            self.serial.write(b'\r\x03\x03')  # Ctrl+C to interrupt any running program
            await asyncio.sleep(0.5)
            
            # Flush input buffer
            self.serial.reset_input_buffer()
            
            # Create a new script file
            self.serial.write(b'\r\x01')  # Ctrl+A to enter raw REPL mode
            await asyncio.sleep(0.5)
            
            # Send the script
            self.serial.write(self.micropython_script.encode('utf-8'))
            self.serial.write(b'\x04')  # Ctrl+D to execute
            
            # Wait for script to start
            await asyncio.sleep(2)
            
            # Check for ready message
            ready = False
            timeout = time.time() + 10  # 10 second timeout
            
            while not ready and time.time() < timeout:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8').strip()
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'status' and data.get('status') == 'ready':
                            ready = True
                            logger.info("Pico is ready")
                    except:
                        pass
                await asyncio.sleep(0.1)
            
            if not ready:
                logger.warning("Timed out waiting for Pico to be ready")
            
        except Exception as e:
            logger.error(f"Error uploading script: {e}")
    
    async def monitor_serial(self):
        """Monitor serial output from the Pico."""
        if not self.serial:
            return
            
        logger.info("Starting serial monitoring")
        
        while self.running:
            try:
                # Check if there's data available
                if self.serial.in_waiting:
                    # Read a line
                    line = self.serial.readline().decode('utf-8').strip()
                    
                    # Process the line
                    if line:
                        await self.process_serial_data(line)
                
                # Short sleep to prevent CPU hogging
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error reading from serial: {e}")
                await asyncio.sleep(1)
    
    async def process_serial_data(self, data):
        """
        Process data received from the Pico.
        
        Args:
            data: String data from serial port
        """
        try:
            # Parse JSON data
            message = json.loads(data)
            message_type = message.get('type', '')
            
            logger.debug(f"Received from Pico: {message_type}")
            
            if message_type == 'event':
                # Handle event
                event_type = message.get('event', '')
                
                if event_type == 'button_state':
                    # Button state changed
                    await self.notify_button_state(message.get('state', False))
                    
            elif message_type == 'response':
                # Handle response
                command = message.get('command', '')
                
                if command == 'read':
                    # Sensor data
                    await self.notify_sensor_data(message)
                    
            elif message_type == 'error':
                # Handle error
                logger.error(f"Error from Pico: {message.get('error', 'Unknown error')}")
                
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON data: {data}")
        except Exception as e:
            logger.error(f"Error processing serial data: {e}")
    
    async def notify_button_state(self, state):
        """
        Notify subscribers of button state change.
        
        Args:
            state: Boolean indicating button state
        """
        if not self.client:
            return
            
        logger.info(f"Button state changed to: {state}")
            
        # Create message
        message = UAP_Message(
            sender=self.entity_id,
            recipient="*",  # Will be replaced with actual recipient
            intent="button_state",
            payload={
                "state": state,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Send to all subscribers
        for subscriber in self.subscribers:
            message.recipient = subscriber
            await self.client.send_message(message)
    
    async def notify_sensor_data(self, data):
        """
        Notify subscribers of sensor data.
        
        Args:
            data: Dictionary of sensor data
        """
        if not self.client:
            return
            
        # Format data for UAP message
        payload = {
            "temperature": data.get('temperature'),
            "adc_voltage": data.get('adc_voltage'),
            "button_state": data.get('button_state'),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"Sensor data: {payload}")
        
        # Create message
        message = UAP_Message(
            sender=self.entity_id,
            recipient="*",  # Will be replaced with actual recipient
            intent="sensor_data",
            payload=payload
        )
        
        # Send to all subscribers
        for subscriber in self.subscribers:
            message.recipient = subscriber
            await self.client.send_message(message)
    
    async def get_device_info(self):
        """
        Get information about the Pico.
        
        Returns:
            Dictionary of device information
        """
        info = {
            "device_type": "raspberry_pi_pico",
            "connection": {
                "type": "serial",
                "port": self.serial_port,
                "baud_rate": self.baud_rate
            },
            "capabilities": ["gpio", "adc", "sensors", "actuators"],
            "sensors": ["temperature", "button", "adc"],
            "actuators": ["led"]
        }
        
        # Add connection status
        info["connected"] = self.serial is not None
        
        # If not connected, add simulation mode flag
        if not info["connected"]:
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
            success = await self.set_led_state(state)
            
            # Send confirmation
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="led_state",
                payload={
                    "state": state,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            ))
            
        elif message.intent == "read_sensors":
            # Read sensor data
            success = await self.request_sensor_data(message.sender)
            
            # If failed, send error
            if not success:
                await self.client.send_message(UAP_Message(
                    sender=self.entity_id,
                    recipient=message.sender,
                    intent="error",
                    payload={
                        "error": "Failed to read sensors",
                        "timestamp": datetime.now().isoformat()
                    }
                ))
            
        elif message.intent == "ping":
            # Ping the Pico
            success = await self.ping_pico()
            
            # Send response
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="pong",
                payload={
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
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
            interval = message.payload.get("interval", 10)  # Default: 10 seconds
            asyncio.create_task(self.monitor_sensors(message.sender, interval))
    
    async def set_led_state(self, state):
        """
        Set the state of the LED on the Pico.
        
        Args:
            state: Boolean indicating whether LED should be on or off
            
        Returns:
            Boolean indicating success
        """
        if not self.serial:
            logger.warning("Not connected to Pico")
            return False
            
        try:
            # Create command
            command = {
                "type": "led",
                "state": state
            }
            
            # Send command
            self.serial.write((json.dumps(command) + '\r\n').encode('utf-8'))
            logger.info(f"Sent LED command: {state}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting LED state: {e}")
            return False
    
    async def request_sensor_data(self, recipient):
        """
        Request sensor data from the Pico.
        
        Args:
            recipient: Entity ID to send the data to
            
        Returns:
            Boolean indicating success
        """
        if not self.serial:
            logger.warning("Not connected to Pico")
            return False
            
        try:
            # Create command
            command = {
                "type": "read"
            }
            
            # Send command
            self.serial.write((json.dumps(command) + '\r\n').encode('utf-8'))
            logger.info("Sent sensor read command")
            return True
            
        except Exception as e:
            logger.error(f"Error requesting sensor data: {e}")
            return False
    
    async def ping_pico(self):
        """
        Ping the Pico to check if it's responsive.
        
        Returns:
            Boolean indicating success
        """
        if not self.serial:
            logger.warning("Not connected to Pico")
            return False
            
        try:
            # Create command
            command = {
                "type": "ping"
            }
            
            # Send command
            self.serial.write((json.dumps(command) + '\r\n').encode('utf-8'))
            logger.info("Sent ping command")
            return True
            
        except Exception as e:
            logger.error(f"Error pinging Pico: {e}")
            return False
    
    async def monitor_sensors(self, recipient, interval):
        """
        Periodically request sensor data.
        
        Args:
            recipient: Entity ID to send the data to
            interval: Time between readings in seconds
        """
        logger.info(f"Starting sensor monitoring with interval: {interval}s")
        
        while recipient in self.subscribers:
            # Request sensor data
            await self.request_sensor_data(recipient)
            
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
            self.running = False
            
            if self.serial:
                self.serial.close()
            
            if self.client:
                await self.client.disconnect()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Raspberry Pi Pico Serial Agent')
    parser.add_argument('--entity-id', type=str, default='raspberry_pi_pico',
                        help='Entity ID for this agent')
    parser.add_argument('--registry', type=str, default='localhost:8000',
                        help='URL of the ReGenNexus registry')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                        help='Serial port for Pico')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Baud rate for serial communication')
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = PicoSerialAgent(
        entity_id=args.entity_id,
        registry_url=args.registry,
        serial_port=args.port,
        baud_rate=args.baud
    )
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
