"""
RegenNexus UAP - Universal Adapter Protocol

Fast, reliable, plug-and-play communication framework for devices, robots, and applications.

Copyright (c) 2024-2025 ReGen Designs LLC
Licensed under MIT License with Attribution

Quick Start:
    import regennexus

    # Start with default config
    regen = regennexus.start()

    # Send a message
    regen.send("device-id", {"action": "hello"})

    # Or with custom config
    regen = regennexus.start(config="my-config.yaml")

For more information, visit: https://github.com/ReGenNow/ReGenNexus
"""

from regennexus.__version__ import (
    __title__,
    __description__,
    __version__,
    __author__,
    __license__,
    __copyright__,
    VERSION,
)

# Convenience function to start RegenNexus
def start(config=None, **kwargs):
    """
    Start RegenNexus UAP.

    Args:
        config: Path to YAML config file, or dict of config options
        **kwargs: Override specific config options

    Returns:
        RegenNexus instance ready to use

    Example:
        # With defaults
        regen = regennexus.start()

        # With config file
        regen = regennexus.start(config="my-config.yaml")

        # With inline config
        regen = regennexus.start(
            security={"enabled": True},
            api={"port": 8080}
        )
    """
    from regennexus.core.protocol import RegenNexus
    instance = RegenNexus(config=config, **kwargs)
    return instance

# Config loading
from regennexus.config import load_config, RegenNexusConfig

# All public exports
__all__ = [
    # Version info
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",
    "__copyright__",
    "VERSION",
    # Functions
    "start",
    "load_config",
    # Config
    "RegenNexusConfig",
]
