"""
RegenNexus UAP - Raspberry Pi Plugin

Raspberry Pi integration with GPIO, camera, and sensor support.
Includes mock mode for development without hardware.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from regennexus.plugins.base import DevicePlugin, MockDeviceMixin

logger = logging.getLogger(__name__)

# Try to import RPi.GPIO
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    GPIO = None

# Try to import picamera
try:
    import picamera
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False
    picamera = None


class RaspberryPiPlugin(DevicePlugin, MockDeviceMixin):
    """
    Raspberry Pi plugin for RegenNexus.

    Supports:
    - GPIO read/write/PWM
    - Camera capture/record
    - Sensor integration
    - Mock mode for development
    """

    def __init__(
        self,
        entity_id: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False,
        gpio_mode: str = "BCM"
    ):
        """
        Initialize Raspberry Pi plugin.

        Args:
            entity_id: Unique identifier
            protocol: Protocol instance
            mock_mode: If True, simulate without hardware
            gpio_mode: GPIO numbering mode ("BCM" or "BOARD")
        """
        DevicePlugin.__init__(self, entity_id, "raspberry_pi", protocol, mock_mode)
        MockDeviceMixin.__init__(self)

        self.gpio_mode = gpio_mode
        self.gpio_state: Dict[int, Dict] = {}
        self.pwm_instances: Dict[int, Any] = {}
        self.camera = None
        self.camera_active = False
        self.sensors: Dict[str, Dict] = {}

    async def _device_init(self) -> bool:
        """Initialize Raspberry Pi hardware."""
        try:
            # Initialize GPIO
            if not self.mock_mode and HAS_GPIO:
                mode = GPIO.BCM if self.gpio_mode == "BCM" else GPIO.BOARD
                GPIO.setmode(mode)
                GPIO.setwarnings(False)
                logger.info(f"GPIO initialized in {self.gpio_mode} mode")
            elif self.mock_mode:
                logger.info("GPIO in mock mode")

            # Add GPIO capabilities
            self.capabilities.extend([
                "gpio.read",
                "gpio.write",
                "gpio.pwm",
            ])

            # Add camera capabilities if available
            if HAS_CAMERA or self.mock_mode:
                self.capabilities.extend([
                    "camera.capture",
                    "camera.record",
                ])

            # Register command handlers
            self.register_command_handler("gpio.read", self._handle_gpio_read)
            self.register_command_handler("gpio.write", self._handle_gpio_write)
            self.register_command_handler("gpio.pwm", self._handle_gpio_pwm)
            self.register_command_handler("gpio.cleanup", self._handle_gpio_cleanup)
            self.register_command_handler("camera.capture", self._handle_camera_capture)
            self.register_command_handler("sensor.read", self._handle_sensor_read)

            # Update metadata
            self.metadata.update({
                "gpio_available": HAS_GPIO or self.mock_mode,
                "camera_available": HAS_CAMERA or self.mock_mode,
                "gpio_mode": self.gpio_mode,
                "model": self._get_pi_model(),
            })

            return True

        except Exception as e:
            logger.error(f"Raspberry Pi init error: {e}")
            return False

    async def _device_shutdown(self) -> None:
        """Clean up Raspberry Pi resources."""
        # Stop all PWM
        for pin, pwm in self.pwm_instances.items():
            try:
                pwm.stop()
            except Exception:
                pass
        self.pwm_instances.clear()

        # Cleanup GPIO
        if not self.mock_mode and HAS_GPIO:
            try:
                GPIO.cleanup()
            except Exception:
                pass

        # Close camera
        if self.camera and self.camera_active:
            try:
                self.camera.close()
            except Exception:
                pass
            self.camera = None
            self.camera_active = False

    def _get_pi_model(self) -> str:
        """Get Raspberry Pi model information."""
        if self.mock_mode:
            return "Mock Raspberry Pi"

        try:
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "r") as f:
                    return f.read().strip("\0")
            return "Unknown Raspberry Pi"
        except Exception:
            return "Unknown Raspberry Pi"

    async def _handle_gpio_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GPIO read command."""
        pin = params.get("pin")
        if pin is None:
            return {"success": False, "error": "Missing pin parameter"}

        try:
            if self.mock_mode:
                value = self.mock_get_pin(pin)
            elif HAS_GPIO:
                # Set up as input if needed
                if pin not in self.gpio_state:
                    GPIO.setup(pin, GPIO.IN)
                    self.gpio_state[pin] = {"mode": "input"}
                value = GPIO.input(pin)
            else:
                return {"success": False, "error": "GPIO not available"}

            self.gpio_state[pin] = {"mode": "input", "value": value}

            return {"success": True, "pin": pin, "value": value}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gpio_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GPIO write command."""
        pin = params.get("pin")
        value = params.get("value")

        if pin is None or value is None:
            return {"success": False, "error": "Missing pin or value parameter"}

        try:
            # Normalize value
            value = 1 if value else 0

            if self.mock_mode:
                self.mock_set_pin(pin, value)
            elif HAS_GPIO:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, value)
            else:
                return {"success": False, "error": "GPIO not available"}

            self.gpio_state[pin] = {"mode": "output", "value": value}

            return {"success": True, "pin": pin, "value": value}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gpio_pwm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GPIO PWM command."""
        pin = params.get("pin")
        frequency = params.get("frequency", 1000)
        duty_cycle = params.get("duty_cycle", 50)
        action = params.get("action", "start")

        if pin is None:
            return {"success": False, "error": "Missing pin parameter"}

        try:
            if self.mock_mode:
                self.mock_set_state(f"pwm_{pin}", {
                    "frequency": frequency,
                    "duty_cycle": duty_cycle,
                    "running": action == "start",
                })
            elif HAS_GPIO:
                if action == "start":
                    if pin not in self.pwm_instances:
                        GPIO.setup(pin, GPIO.OUT)
                        self.pwm_instances[pin] = GPIO.PWM(pin, frequency)
                    self.pwm_instances[pin].start(duty_cycle)
                elif action == "stop" and pin in self.pwm_instances:
                    self.pwm_instances[pin].stop()
                elif action == "change" and pin in self.pwm_instances:
                    self.pwm_instances[pin].ChangeDutyCycle(duty_cycle)
            else:
                return {"success": False, "error": "GPIO not available"}

            self.gpio_state[pin] = {
                "mode": "pwm",
                "frequency": frequency,
                "duty_cycle": duty_cycle,
            }

            return {
                "success": True,
                "pin": pin,
                "frequency": frequency,
                "duty_cycle": duty_cycle,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_gpio_cleanup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GPIO cleanup command."""
        pin = params.get("pin")

        try:
            if self.mock_mode:
                if pin:
                    self._mock_pins.pop(pin, None)
                else:
                    self._mock_pins.clear()
            elif HAS_GPIO:
                if pin:
                    GPIO.cleanup(pin)
                    self.gpio_state.pop(pin, None)
                else:
                    GPIO.cleanup()
                    self.gpio_state.clear()

            return {"success": True, "pin": pin}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_camera_capture(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle camera capture command."""
        path = params.get("path", f"/tmp/capture_{int(time.time())}.jpg")
        resolution = params.get("resolution", (1280, 720))

        try:
            if self.mock_mode:
                # Simulate capture
                self.mock_set_state("last_capture", {
                    "path": path,
                    "resolution": resolution,
                    "timestamp": time.time(),
                })
                return {
                    "success": True,
                    "path": path,
                    "resolution": resolution,
                    "mock": True,
                }

            if not HAS_CAMERA:
                return {"success": False, "error": "Camera not available"}

            # Initialize camera
            if not self.camera_active:
                self.camera = picamera.PiCamera()
                self.camera_active = True

            self.camera.resolution = resolution
            self.camera.capture(path)

            return {"success": True, "path": path, "resolution": resolution}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_sensor_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sensor read command."""
        sensor_type = params.get("sensor_type")

        if not sensor_type:
            return {"success": False, "error": "Missing sensor_type parameter"}

        if sensor_type not in self.sensors:
            return {"success": False, "error": f"Sensor {sensor_type} not registered"}

        try:
            sensor = self.sensors[sensor_type]
            read_func = sensor["read_func"]
            data = read_func()
            if asyncio.iscoroutine(data):
                data = await data

            return {"success": True, "sensor_type": sensor_type, "data": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def register_sensor(
        self,
        sensor_type: str,
        read_func: Callable
    ) -> bool:
        """
        Register a sensor.

        Args:
            sensor_type: Sensor type name
            read_func: Function to read sensor data

        Returns:
            True if registration successful
        """
        self.sensors[sensor_type] = {"read_func": read_func}
        capability = f"sensor.{sensor_type}"
        if capability not in self.capabilities:
            self.capabilities.append(capability)
        return True

    # Convenience methods for common operations
    def digital_read(self, pin: int) -> int:
        """Synchronous digital read."""
        if self.mock_mode:
            return self.mock_get_pin(pin)
        if HAS_GPIO:
            if pin not in self.gpio_state:
                GPIO.setup(pin, GPIO.IN)
            return GPIO.input(pin)
        return 0

    def digital_write(self, pin: int, value: int) -> None:
        """Synchronous digital write."""
        if self.mock_mode:
            self.mock_set_pin(pin, value)
        elif HAS_GPIO:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, value)

    def set_pwm(self, pin: int, duty_cycle: float, frequency: int = 1000) -> None:
        """Set PWM on a pin."""
        if self.mock_mode:
            self.mock_set_state(f"pwm_{pin}", {
                "frequency": frequency,
                "duty_cycle": duty_cycle,
            })
        elif HAS_GPIO:
            if pin not in self.pwm_instances:
                GPIO.setup(pin, GPIO.OUT)
                self.pwm_instances[pin] = GPIO.PWM(pin, frequency)
                self.pwm_instances[pin].start(duty_cycle)
            else:
                self.pwm_instances[pin].ChangeDutyCycle(duty_cycle)
