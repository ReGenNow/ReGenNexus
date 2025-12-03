"""
RegenNexus UAP - Arduino Plugin

Arduino integration via serial communication with support for
digital/analog I/O and custom commands. Includes mock mode for
development without hardware.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import logging
import json
import time
from typing import Any, Callable, Dict, List, Optional

from regennexus.plugins.base import DevicePlugin, MockDeviceMixin

logger = logging.getLogger(__name__)

# Try to import serial library
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    serial = None


class ArduinoPlugin(DevicePlugin, MockDeviceMixin):
    """
    Arduino plugin for RegenNexus.

    Supports:
    - Digital read/write
    - Analog read/write (PWM)
    - Custom serial commands
    - Auto port detection
    - Mock mode for development
    """

    def __init__(
        self,
        entity_id: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False,
        port: Optional[str] = None,
        baud_rate: int = 9600
    ):
        """
        Initialize Arduino plugin.

        Args:
            entity_id: Unique identifier
            protocol: Protocol instance
            mock_mode: If True, simulate without hardware
            port: Serial port (e.g., '/dev/ttyACM0', 'COM3')
            baud_rate: Serial baud rate
        """
        DevicePlugin.__init__(self, entity_id, "arduino", protocol, mock_mode)
        MockDeviceMixin.__init__(self)

        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.connected = False
        self.read_task = None
        self.response_queue: asyncio.Queue = asyncio.Queue()

        # Mock analog values (10-bit ADC = 0-1023)
        self._mock_analog: Dict[int, int] = {}

    async def _device_init(self) -> bool:
        """Initialize Arduino hardware."""
        try:
            if self.mock_mode:
                logger.info("Arduino in mock mode")
                self.connected = True
            elif HAS_SERIAL:
                # Auto-detect port if not specified
                if not self.port:
                    self.port = self._auto_detect_port()

                if self.port:
                    try:
                        self.serial_conn = serial.Serial(
                            self.port, self.baud_rate, timeout=1
                        )
                        await asyncio.sleep(2)  # Wait for Arduino reset
                        self.connected = True

                        # Start read task
                        self.read_task = asyncio.create_task(self._read_serial())

                        logger.info(f"Connected to Arduino on {self.port}")
                    except Exception as e:
                        logger.error(f"Error connecting to Arduino: {e}")
                        self.connected = False
                else:
                    logger.warning("No Arduino port found")
                    self.connected = False
            else:
                logger.warning("Serial library not available")
                self.connected = False

            # Add capabilities
            self.capabilities.extend([
                "arduino.digital_read",
                "arduino.digital_write",
                "arduino.analog_read",
                "arduino.analog_write",
                "arduino.send_command",
            ])

            # Register command handlers
            self.register_command_handler("digital_read", self._handle_digital_read)
            self.register_command_handler("digital_write", self._handle_digital_write)
            self.register_command_handler("analog_read", self._handle_analog_read)
            self.register_command_handler("analog_write", self._handle_analog_write)
            self.register_command_handler("send_command", self._handle_send_command)

            # Update metadata
            self.metadata.update({
                "serial_available": HAS_SERIAL or self.mock_mode,
                "port": self.port,
                "baud_rate": self.baud_rate,
                "connected": self.connected,
            })

            return True

        except Exception as e:
            logger.error(f"Arduino init error: {e}")
            return False

    async def _device_shutdown(self) -> None:
        """Clean up Arduino resources."""
        # Stop read task
        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass
            self.read_task = None

        # Close serial connection
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

        self.connected = False

    def _auto_detect_port(self) -> Optional[str]:
        """Auto-detect Arduino port."""
        if not HAS_SERIAL:
            return None

        try:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if "Arduino" in p.description or "CH340" in p.description:
                    return p.device

            # Fallback to first available port
            if ports:
                return ports[0].device

            return None
        except Exception:
            return None

    async def _read_serial(self) -> None:
        """Read data from serial port."""
        try:
            while self._running:
                if self.serial_conn and self.serial_conn.in_waiting:
                    try:
                        line = self.serial_conn.readline().decode("utf-8").strip()
                        if line:
                            logger.debug(f"Received from Arduino: {line}")

                            # Try to parse as JSON
                            try:
                                data = json.loads(line)

                                if "response" in data:
                                    await self.response_queue.put(data)
                                elif "event" in data:
                                    await self.emit_event("arduino", data)
                            except json.JSONDecodeError:
                                await self.emit_event("arduino.data", {"data": line})

                    except Exception as e:
                        logger.error(f"Error reading from Arduino: {e}")

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in serial read task: {e}")

    async def _send_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Send command to Arduino and wait for response."""
        if self.mock_mode:
            # Simulate response
            return {"response": True, "command": command}

        if not self.connected or not self.serial_conn:
            raise ValueError("Not connected to Arduino")

        try:
            self.serial_conn.write(f"{command}\n".encode("utf-8"))
            self.serial_conn.flush()

            response = await asyncio.wait_for(
                self.response_queue.get(), timeout=5.0
            )
            return response
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response: {command}")
            return None

    async def _handle_digital_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle digital read command."""
        pin = params.get("pin")
        if pin is None:
            return {"success": False, "error": "Missing pin parameter"}

        try:
            if self.mock_mode:
                value = self.mock_get_pin(pin)
            elif self.connected:
                response = await self._send_command(f"DR {pin}")
                if not response:
                    return {"success": False, "error": "No response from Arduino"}
                value = response.get("value", 0)
            else:
                return {"success": False, "error": "Not connected to Arduino"}

            return {"success": True, "pin": pin, "value": value}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_digital_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle digital write command."""
        pin = params.get("pin")
        value = params.get("value")

        if pin is None or value is None:
            return {"success": False, "error": "Missing pin or value parameter"}

        try:
            value = 1 if value else 0

            if self.mock_mode:
                self.mock_set_pin(pin, value)
            elif self.connected:
                response = await self._send_command(f"DW {pin} {value}")
                if not response:
                    return {"success": False, "error": "No response from Arduino"}
            else:
                return {"success": False, "error": "Not connected to Arduino"}

            return {"success": True, "pin": pin, "value": value}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_analog_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analog read command."""
        pin = params.get("pin")
        if pin is None:
            return {"success": False, "error": "Missing pin parameter"}

        try:
            if self.mock_mode:
                # Return mock analog value (0-1023)
                value = self._mock_analog.get(pin, 512)
            elif self.connected:
                response = await self._send_command(f"AR {pin}")
                if not response:
                    return {"success": False, "error": "No response from Arduino"}
                value = response.get("value", 0)
            else:
                return {"success": False, "error": "Not connected to Arduino"}

            return {"success": True, "pin": pin, "value": value}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_analog_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analog write (PWM) command."""
        pin = params.get("pin")
        value = params.get("value")

        if pin is None or value is None:
            return {"success": False, "error": "Missing pin or value parameter"}

        try:
            # Ensure value is in range 0-255
            value = max(0, min(255, int(value)))

            if self.mock_mode:
                self.mock_set_state(f"pwm_{pin}", value)
            elif self.connected:
                response = await self._send_command(f"AW {pin} {value}")
                if not response:
                    return {"success": False, "error": "No response from Arduino"}
            else:
                return {"success": False, "error": "Not connected to Arduino"}

            return {"success": True, "pin": pin, "value": value}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_send_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom serial command."""
        command = params.get("command")
        if not command:
            return {"success": False, "error": "Missing command parameter"}

        try:
            if self.mock_mode:
                return {
                    "success": True,
                    "command": command,
                    "response": {"mock": True},
                }

            if not self.connected:
                return {"success": False, "error": "Not connected to Arduino"}

            response = await self._send_command(command)
            if not response:
                return {"success": False, "error": "No response from Arduino"}

            return {"success": True, "command": command, "response": response}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # Convenience methods
    def digital_read(self, pin: int) -> int:
        """Synchronous digital read."""
        if self.mock_mode:
            return self.mock_get_pin(pin)
        return 0

    def digital_write(self, pin: int, value: int) -> None:
        """Synchronous digital write."""
        if self.mock_mode:
            self.mock_set_pin(pin, value)

    def analog_read(self, pin: int) -> int:
        """Synchronous analog read."""
        if self.mock_mode:
            return self._mock_analog.get(pin, 512)
        return 0

    def set_mock_analog(self, pin: int, value: int) -> None:
        """Set mock analog value for testing."""
        self._mock_analog[pin] = max(0, min(1023, value))
