"""
RegenNexus UAP - Device Plugins

Device plugins for various hardware platforms.

Supported devices:
- Raspberry Pi (GPIO, camera, sensors)
- Arduino (serial, digital/analog I/O)
- NVIDIA Jetson (GPU, CUDA, TensorRT)
- Generic IoT (MQTT, HTTP, CoAP)
- Amber B1 Robotic Arm (7-DOF, UDP control)
- Lucid One Robotic Arm (7-DOF, UDP control)

Copyright (c) 2024-2025 ReGen Designs LLC
"""

from regennexus.plugins.base import DevicePlugin, MockDeviceMixin

__all__ = [
    "DevicePlugin",
    "MockDeviceMixin",
    "get_raspberry_pi_plugin",
    "get_arduino_plugin",
    "get_jetson_plugin",
    "get_iot_plugin",
    "get_amber_b1_plugin",
    "get_lucid_one_plugin",
]

# Lazy imports for device-specific plugins
def get_raspberry_pi_plugin():
    """Get Raspberry Pi plugin (lazy import)."""
    from regennexus.plugins.raspberry_pi import RaspberryPiPlugin
    return RaspberryPiPlugin

def get_arduino_plugin():
    """Get Arduino plugin (lazy import)."""
    from regennexus.plugins.arduino import ArduinoPlugin
    return ArduinoPlugin

def get_jetson_plugin():
    """Get Jetson plugin (lazy import)."""
    from regennexus.plugins.jetson import JetsonPlugin
    return JetsonPlugin

def get_iot_plugin():
    """Get IoT plugin (lazy import)."""
    from regennexus.plugins.iot import IoTPlugin
    return IoTPlugin

def get_amber_b1_plugin():
    """Get Amber B1 robotic arm plugin (lazy import)."""
    from regennexus.plugins.amber_b1 import AmberB1Plugin
    return AmberB1Plugin

def get_lucid_one_plugin():
    """Get Lucid One robotic arm plugin (lazy import)."""
    from regennexus.plugins.lucid_one import LucidOnePlugin
    return LucidOnePlugin
