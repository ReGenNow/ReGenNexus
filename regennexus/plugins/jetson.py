"""
RegenNexus UAP - NVIDIA Jetson Plugin

NVIDIA Jetson integration with GPU, CUDA, TensorRT, and camera support.
Includes mock mode for development without hardware.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import logging
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional

from regennexus.plugins.base import DevicePlugin, MockDeviceMixin

logger = logging.getLogger(__name__)

# Try to import Jetson GPIO
try:
    import Jetson.GPIO as GPIO
    HAS_JETSON_GPIO = True
except ImportError:
    HAS_JETSON_GPIO = False
    GPIO = None

# Try to import OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None


class JetsonPlugin(DevicePlugin, MockDeviceMixin):
    """
    NVIDIA Jetson plugin for RegenNexus.

    Supports:
    - GPIO read/write
    - Camera capture (CSI and USB)
    - CUDA detection and info
    - TensorRT inference (placeholder)
    - Mock mode for development
    """

    def __init__(
        self,
        entity_id: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False,
        gpio_mode: str = "BOARD"
    ):
        """
        Initialize Jetson plugin.

        Args:
            entity_id: Unique identifier
            protocol: Protocol instance
            mock_mode: If True, simulate without hardware
            gpio_mode: GPIO numbering mode ("BOARD" or "BCM")
        """
        DevicePlugin.__init__(self, entity_id, "jetson", protocol, mock_mode)
        MockDeviceMixin.__init__(self)

        self.gpio_mode = gpio_mode
        self.gpio_state: Dict[int, Dict] = {}
        self.jetson_model: str = "Unknown"
        self.cuda_available: bool = False
        self.camera_devices: Dict[str, Dict] = {}

    async def _device_init(self) -> bool:
        """Initialize Jetson hardware."""
        try:
            # Detect if running on Jetson
            is_jetson = os.path.exists("/etc/nv_tegra_release") or self.mock_mode

            if not is_jetson and not self.mock_mode:
                logger.warning("Not running on a Jetson device")

            # Detect Jetson model
            self.jetson_model = self._detect_model()

            # Check CUDA availability
            self.cuda_available = self._check_cuda()

            # Detect cameras
            self.camera_devices = self._detect_cameras()

            # Initialize GPIO
            if not self.mock_mode and HAS_JETSON_GPIO:
                mode = GPIO.BOARD if self.gpio_mode == "BOARD" else GPIO.BCM
                GPIO.setmode(mode)
                GPIO.setwarnings(False)
                logger.info(f"Jetson GPIO initialized in {self.gpio_mode} mode")
            elif self.mock_mode:
                logger.info("Jetson GPIO in mock mode")

            # Add capabilities
            self.capabilities.extend([
                "jetson.gpio.read",
                "jetson.gpio.write",
                "jetson.device_info",
            ])

            if HAS_OPENCV or self.mock_mode:
                self.capabilities.extend([
                    "jetson.camera.capture",
                    "jetson.camera.list",
                ])

            if self.cuda_available or self.mock_mode:
                self.capabilities.extend([
                    "jetson.cuda.info",
                    "jetson.inference",
                ])

            # Register command handlers
            self.register_command_handler("gpio.read", self._handle_gpio_read)
            self.register_command_handler("gpio.write", self._handle_gpio_write)
            self.register_command_handler("gpio.cleanup", self._handle_gpio_cleanup)
            self.register_command_handler("device_info", self._handle_device_info)
            self.register_command_handler("camera.capture", self._handle_camera_capture)
            self.register_command_handler("camera.list", self._handle_camera_list)
            self.register_command_handler("cuda.info", self._handle_cuda_info)

            # Update metadata
            self.metadata.update({
                "is_jetson": is_jetson or self.mock_mode,
                "model": self.jetson_model,
                "cuda_available": self.cuda_available,
                "gpio_available": HAS_JETSON_GPIO or self.mock_mode,
                "opencv_available": HAS_OPENCV or self.mock_mode,
                "gpio_mode": self.gpio_mode,
                "cameras": list(self.camera_devices.keys()),
            })

            return True

        except Exception as e:
            logger.error(f"Jetson init error: {e}")
            return False

    async def _device_shutdown(self) -> None:
        """Clean up Jetson resources."""
        # Cleanup GPIO
        if not self.mock_mode and HAS_JETSON_GPIO:
            try:
                GPIO.cleanup()
            except Exception:
                pass

        self.gpio_state.clear()

    def _detect_model(self) -> str:
        """Detect Jetson model."""
        if self.mock_mode:
            return "Mock Jetson Orin"

        try:
            result = subprocess.run(
                ["cat", "/proc/device-tree/model"],
                capture_output=True, text=True, check=True
            )
            model = result.stdout.strip().rstrip("\x00")

            # Map to common names
            model_map = {
                "Nano": "Jetson Nano",
                "Xavier NX": "Jetson Xavier NX",
                "AGX Xavier": "Jetson AGX Xavier",
                "Orin Nano": "Jetson Orin Nano",
                "Orin NX": "Jetson Orin NX",
                "AGX Orin": "Jetson AGX Orin",
            }

            for key, name in model_map.items():
                if key in model:
                    return name

            return model

        except Exception:
            return "Unknown Jetson"

    def _check_cuda(self) -> bool:
        """Check CUDA availability."""
        if self.mock_mode:
            return True

        try:
            if not os.path.exists("/usr/local/cuda"):
                return False

            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True
            )
            return result.returncode == 0

        except Exception:
            return False

    def _detect_cameras(self) -> Dict[str, Dict]:
        """Detect available cameras."""
        cameras = {}

        if self.mock_mode:
            return {
                "csi0": {"path": "/dev/video0", "type": "csi", "name": "Mock CSI Camera"},
                "usb0": {"path": "/dev/video1", "type": "usb", "name": "Mock USB Camera"},
            }

        try:
            for i in range(10):
                device_path = f"/dev/video{i}"
                if not os.path.exists(device_path):
                    continue

                camera_info = {
                    "path": device_path,
                    "type": "v4l2",
                    "name": f"Camera {i}",
                }

                try:
                    result = subprocess.run(
                        ["v4l2-ctl", "--device", device_path, "--all"],
                        capture_output=True, text=True
                    )

                    for line in result.stdout.split("\n"):
                        if "Card type" in line:
                            camera_info["name"] = line.split(":")[1].strip()
                            break

                except Exception:
                    pass

                cameras[f"camera{i}"] = camera_info

        except Exception as e:
            logger.error(f"Failed to detect cameras: {e}")

        return cameras

    async def _handle_gpio_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GPIO read command."""
        pin = params.get("pin")
        if pin is None:
            return {"success": False, "error": "Missing pin parameter"}

        try:
            if self.mock_mode:
                value = self.mock_get_pin(pin)
            elif HAS_JETSON_GPIO:
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
            value = 1 if value else 0

            if self.mock_mode:
                self.mock_set_pin(pin, value)
            elif HAS_JETSON_GPIO:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, value)
            else:
                return {"success": False, "error": "GPIO not available"}

            self.gpio_state[pin] = {"mode": "output", "value": value}

            return {"success": True, "pin": pin, "value": value}

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
            elif HAS_JETSON_GPIO:
                if pin:
                    GPIO.cleanup(pin)
                    self.gpio_state.pop(pin, None)
                else:
                    GPIO.cleanup()
                    self.gpio_state.clear()

            return {"success": True, "pin": pin}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_device_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle device info command."""
        info = {
            "success": True,
            "model": self.jetson_model,
            "cuda_available": self.cuda_available,
            "cameras": list(self.camera_devices.keys()),
            "gpio_pins": list(self.gpio_state.keys()),
        }

        if self.mock_mode:
            info.update({
                "cpu_count": 6,
                "memory_mb": 8192,
                "gpu": {"name": "Mock GPU", "memory": "8GB"},
            })
        else:
            try:
                # Get CPU count
                with open("/proc/cpuinfo", "r") as f:
                    info["cpu_count"] = f.read().count("processor")

                # Get memory
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            mem_kb = int(line.split(":")[1].strip().split()[0])
                            info["memory_mb"] = mem_kb // 1024
                            break

                # Get GPU info
                if self.cuda_available:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        gpu_info = result.stdout.strip().split(",")
                        info["gpu"] = {
                            "name": gpu_info[0].strip(),
                            "memory": gpu_info[1].strip() if len(gpu_info) > 1 else "Unknown",
                        }

            except Exception as e:
                logger.error(f"Failed to get system info: {e}")

        return info

    async def _handle_camera_capture(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle camera capture command."""
        camera_id = params.get("camera_id", "camera0")
        width = params.get("width", 640)
        height = params.get("height", 480)
        output_path = params.get("path")

        if self.mock_mode:
            return {
                "success": True,
                "camera_id": camera_id,
                "width": width,
                "height": height,
                "path": output_path,
                "mock": True,
            }

        if not HAS_OPENCV:
            return {"success": False, "error": "OpenCV not available"}

        if camera_id not in self.camera_devices:
            return {"success": False, "error": f"Camera {camera_id} not found"}

        try:
            camera_path = self.camera_devices[camera_id]["path"]
            cap = cv2.VideoCapture(camera_path)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return {"success": False, "error": "Failed to capture image"}

            if output_path:
                cv2.imwrite(output_path, frame)
                return {
                    "success": True,
                    "camera_id": camera_id,
                    "path": output_path,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
            else:
                _, img_encoded = cv2.imencode(".jpg", frame)
                return {
                    "success": True,
                    "camera_id": camera_id,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "data_size": len(img_encoded),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_camera_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle camera list command."""
        return {
            "success": True,
            "cameras": self.camera_devices,
        }

    async def _handle_cuda_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CUDA info command."""
        if self.mock_mode:
            return {
                "success": True,
                "cuda_available": True,
                "cuda_version": "11.4",
                "driver_version": "470.82.01",
                "device_count": 1,
                "devices": [
                    {
                        "name": "Mock Jetson GPU",
                        "compute_capability": "8.7",
                        "memory_total": "8192 MB",
                    }
                ],
            }

        if not self.cuda_available:
            return {"success": False, "error": "CUDA not available"}

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True
            )

            info = {
                "success": True,
                "cuda_available": True,
                "devices": [],
            }

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(",")
                    info["devices"].append({
                        "name": parts[0].strip(),
                        "memory_total": parts[1].strip() if len(parts) > 1 else "Unknown",
                    })
                    if len(parts) > 2:
                        info["driver_version"] = parts[2].strip()

            return info

        except Exception as e:
            return {"success": False, "error": str(e)}

    # Convenience methods
    def gpio_read(self, pin: int) -> int:
        """Synchronous GPIO read."""
        if self.mock_mode:
            return self.mock_get_pin(pin)
        if HAS_JETSON_GPIO:
            if pin not in self.gpio_state:
                GPIO.setup(pin, GPIO.IN)
            return GPIO.input(pin)
        return 0

    def gpio_write(self, pin: int, value: int) -> None:
        """Synchronous GPIO write."""
        if self.mock_mode:
            self.mock_set_pin(pin, value)
        elif HAS_JETSON_GPIO:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, value)
