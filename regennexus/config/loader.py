"""
RegenNexus UAP - Configuration Loader

Unified configuration loading from YAML files.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class SecurityConfig:
    """Security configuration."""
    enabled: bool = True
    level: int = 2
    encryption: str = "aes-256-gcm"
    token_enabled: bool = False
    token_secret: str = ""
    token_expire_hours: int = 24
    api_keys_enabled: bool = False
    api_keys: List[Dict[str, Any]] = field(default_factory=list)
    rate_limiting_enabled: bool = False
    requests_per_minute: int = 100
    burst_limit: int = 20


@dataclass
class CommunicationConfig:
    """Communication configuration."""
    auto_select: bool = True
    ipc_enabled: bool = True
    ipc_method: str = "unix_socket"
    ipc_socket_path: str = "/tmp/regennexus.sock"
    udp_enabled: bool = True
    udp_multicast_group: str = "224.0.0.251"
    udp_port: int = 5353
    ws_enabled: bool = True
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    ws_ssl: bool = False
    queue_enabled: bool = False
    queue_max_size: int = 1000
    queue_persist: bool = False


@dataclass
class RegistryConfig:
    """Registry configuration."""
    mode: str = "hybrid"
    server_host: str = "localhost"
    server_port: int = 8000
    auto_start: bool = True


@dataclass
class DeviceConfig:
    """Device plugin configuration."""
    raspberry_pi_enabled: bool = True
    raspberry_pi_gpio_mode: str = "BCM"
    raspberry_pi_mock: bool = False
    arduino_enabled: bool = True
    arduino_auto_detect: bool = True
    arduino_port: str = "auto"
    arduino_baud_rate: int = 115200
    arduino_mock: bool = False
    jetson_enabled: bool = True
    jetson_use_cuda: bool = True
    jetson_mock: bool = False
    generic_iot_enabled: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    docs_enabled: bool = True
    docs_path: str = "/docs"
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None
    max_size_mb: int = 10
    backup_count: int = 3
    console: bool = True


@dataclass
class RegenNexusConfig:
    """Complete RegenNexus configuration."""
    security: SecurityConfig = field(default_factory=SecurityConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: Optional[str] = None) -> RegenNexusConfig:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        logger.warning("PyYAML not installed, using defaults")
        return RegenNexusConfig()

    if path is None:
        for p in ["regennexus-config.yaml", "regennexus.yaml", "config.yaml"]:
            if os.path.exists(p):
                path = p
                break

    if path is None or not os.path.exists(path):
        return RegenNexusConfig()

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {path}")
        return _parse_config(data)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return RegenNexusConfig()


def _parse_config(data: Dict[str, Any]) -> RegenNexusConfig:
    """Parse config dictionary."""
    config = RegenNexusConfig()

    if "security" in data:
        sec = data["security"]
        config.security.enabled = sec.get("enabled", True)
        config.security.level = sec.get("level", 2)
        config.security.encryption = sec.get("encryption", "aes-256-gcm")

    if "communication" in data:
        comm = data["communication"]
        config.communication.auto_select = comm.get("auto_select", True)
        if "websocket" in comm:
            config.communication.ws_port = comm["websocket"].get("port", 8765)

    if "api" in data:
        api = data["api"]
        config.api.enabled = api.get("enabled", True)
        config.api.port = api.get("port", 8080)

    if "registry" in data:
        reg = data["registry"]
        config.registry.mode = reg.get("mode", "hybrid")

    return config


def save_config(config: RegenNexusConfig, path: str) -> bool:
    """Save configuration to YAML file."""
    if not HAS_YAML:
        return False

    try:
        data = {
            "security": {"enabled": config.security.enabled, "level": config.security.level},
            "api": {"enabled": config.api.enabled, "port": config.api.port},
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception:
        return False
