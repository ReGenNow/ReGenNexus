"""
RegenNexus UAP - API Module

Provides REST and WebSocket APIs for external applications.

Features:
- REST API for device management and messaging
- WebSocket API for real-time communication
- OpenAPI documentation (Swagger)
- CORS support

Copyright (c) 2024-2025 ReGen Designs LLC
"""

from regennexus.api.server import APIServer, create_app

__all__ = [
    "APIServer",
    "create_app",
]
