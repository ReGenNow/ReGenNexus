"""
RegenNexus UAP - Main Protocol Module

Main entry point and coordinator for RegenNexus UAP.

Copyright (c) 2024 ReGen Designs LLC
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from regennexus.core.message import Message, Intent
from regennexus.core.entity import Entity
from regennexus.core.registry import Registry, get_registry
from regennexus.core.context import ContextManager, get_context_manager

logger = logging.getLogger(__name__)


class RegenNexus:
    """
    Main RegenNexus UAP coordinator.

    This is the primary entry point for using RegenNexus. It coordinates
    the registry, transport, security, and API components.

    Example:
        # Simple usage
        import regennexus

        regen = regennexus.start()
        regen.send("device-1", {"action": "on"})

        # With config file
        regen = regennexus.start(config="my-config.yaml")

        # Access components
        registry = regen.registry
        await registry.list_entities()
    """

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Initialize RegenNexus.

        Args:
            config: Path to YAML config file, or dict of config options
            **kwargs: Override specific config options
        """
        # Load configuration
        self._config = self._load_config(config, **kwargs)

        # Initialize components
        self._registry = get_registry()
        self._context_manager = get_context_manager()

        # State
        self._running = False
        self._transport = None
        self._security = None
        self._api = None

        # Setup logging
        self._setup_logging()

        logger.info("RegenNexus UAP initialized")

    def _load_config(
        self,
        config: Optional[Union[str, Dict[str, Any]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load configuration from file or dict.

        Args:
            config: Config file path or dict
            **kwargs: Override options

        Returns:
            Configuration dictionary
        """
        # Start with defaults
        cfg = self._get_default_config()

        # Load from file if string
        if isinstance(config, str):
            cfg = self._load_config_file(config, cfg)
        elif isinstance(config, dict):
            cfg = self._merge_config(cfg, config)

        # Apply overrides
        if kwargs:
            cfg = self._merge_config(cfg, kwargs)

        return cfg

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "security": {
                "enabled": True,
                "level": 2,  # 1=basic, 2=medium, 3=maximum
                "encryption": "aes-256-gcm",
                "authentication": {
                    "token": {"enabled": False},
                    "api_keys": {"enabled": False},
                },
                "rate_limiting": {"enabled": False},
            },
            "communication": {
                "auto_select": True,
                "local_ipc": {"enabled": True},
                "udp_discovery": {"enabled": True, "port": 5353},
                "websocket": {"enabled": True, "host": "0.0.0.0", "port": 8765},
                "message_queue": {"enabled": False},
            },
            "registry": {
                "mode": "hybrid",  # central, p2p, hybrid
                "server": {"host": "localhost", "port": 8000, "auto_start": True},
            },
            "devices": {
                "raspberry_pi": {"enabled": True, "mock_mode": False},
                "arduino": {"enabled": True, "mock_mode": False},
                "jetson": {"enabled": True, "mock_mode": False},
                "esp32": {"enabled": True, "mock_mode": False},
                "generic_iot": {"enabled": True},
            },
            "api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8080,
                "docs": {"enabled": True, "path": "/docs"},
            },
            "logging": {
                "level": "INFO",
                "file": None,
                "console": True,
            },
        }

    def _load_config_file(
        self, path: str, defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load config from YAML file."""
        try:
            import yaml

            config_path = Path(path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {path}, using defaults")
                return defaults

            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}

            return self._merge_config(defaults, file_config)

        except ImportError:
            logger.warning("PyYAML not installed, cannot load config file")
            return defaults
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return defaults

    def _merge_config(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def _setup_logging(self) -> None:
        """Setup logging based on config."""
        log_config = self._config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        # Configure root logger
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Set regennexus logger level
        logging.getLogger("regennexus").setLevel(level)

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def registry(self) -> Registry:
        """Get the entity registry."""
        return self._registry

    @property
    def context_manager(self) -> ContextManager:
        """Get the context manager."""
        return self._context_manager

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if RegenNexus is running."""
        return self._running

    async def start(self) -> "RegenNexus":
        """
        Start RegenNexus services.

        Returns:
            Self for chaining
        """
        if self._running:
            logger.warning("RegenNexus already running")
            return self

        logger.info("Starting RegenNexus UAP...")

        # Start transport (will be implemented in Phase 2)
        # Start security (will be implemented in Phase 3)
        # Start API (will be implemented in Phase 4)

        self._running = True
        logger.info("RegenNexus UAP started")
        return self

    async def stop(self) -> None:
        """Stop RegenNexus services."""
        if not self._running:
            logger.warning("RegenNexus not running")
            return

        logger.info("Stopping RegenNexus UAP...")

        # Stop API
        # Stop transport
        # Clear registry
        await self._registry.clear()

        self._running = False
        logger.info("RegenNexus UAP stopped")

    # =========================================================================
    # Entity Management
    # =========================================================================

    async def register(self, entity: Entity) -> bool:
        """
        Register an entity.

        Args:
            entity: Entity to register

        Returns:
            True if registered
        """
        return await self._registry.register(entity)

    async def unregister(self, entity_id: str) -> bool:
        """
        Unregister an entity.

        Args:
            entity_id: Entity ID

        Returns:
            True if unregistered
        """
        return await self._registry.unregister(entity_id)

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity or None
        """
        return await self._registry.get_entity(entity_id)

    async def list_entities(self) -> List[Dict[str, Any]]:
        """
        List all entities.

        Returns:
            List of entity info
        """
        return await self._registry.list_entities()

    # =========================================================================
    # Messaging
    # =========================================================================

    async def send(
        self,
        recipient_id: str,
        content: Any,
        intent: str = Intent.MESSAGE,
        sender_id: str = "system",
        context_id: Optional[str] = None,
    ) -> Optional[Message]:
        """
        Send a message.

        Args:
            recipient_id: Target entity ID (or '*' for broadcast)
            content: Message content
            intent: Message intent
            sender_id: Sender ID
            context_id: Context ID

        Returns:
            Response message if any
        """
        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            intent=intent,
            context_id=context_id,
        )
        return await self._registry.route_message(message)

    async def broadcast(
        self,
        content: Any,
        intent: str = Intent.EVENT,
        sender_id: str = "system",
    ) -> None:
        """
        Broadcast a message to all entities.

        Args:
            content: Message content
            intent: Message intent
            sender_id: Sender ID
        """
        await self.send(
            recipient_id="*",
            content=content,
            intent=intent,
            sender_id=sender_id,
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def __enter__(self) -> "RegenNexus":
        """Context manager entry."""
        asyncio.get_event_loop().run_until_complete(self.start())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        asyncio.get_event_loop().run_until_complete(self.stop())

    async def __aenter__(self) -> "RegenNexus":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# Singleton instance
_instance: Optional[RegenNexus] = None


def get_instance(config: Optional[Union[str, Dict[str, Any]]] = None) -> RegenNexus:
    """
    Get the global RegenNexus instance.

    Args:
        config: Optional config (only used on first call)

    Returns:
        RegenNexus instance
    """
    global _instance
    if _instance is None:
        _instance = RegenNexus(config=config)
    return _instance
