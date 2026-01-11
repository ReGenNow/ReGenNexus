"""
RegenNexus UAP - Devices Module

Device plugins for hardware integration.

Copyright (c) 2024 ReGen Designs LLC
"""

# Import device plugins (to be fully implemented in Phase 5)
try:
    from regennexus.devices.base import DevicePlugin
except ImportError:
    DevicePlugin = None

__all__ = [
    "DevicePlugin",
]
