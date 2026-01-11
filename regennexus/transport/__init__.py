"""
RegenNexus UAP - Transport Layer

Multiple transport options for different use cases:
- IPC: Local process communication (<0.1ms)
- UDP: LAN discovery and fast local messaging (1-5ms)
- WebSocket: Internet/remote communication (10-50ms)
- Queue: Reliable delivery with persistence

Copyright (c) 2024-2025 ReGen Designs LLC
"""

from regennexus.transport.base import Transport, TransportConfig, TransportState
from regennexus.transport.auto import AutoTransport, select_best_transport

__all__ = [
    "Transport",
    "TransportConfig",
    "TransportState",
    "AutoTransport",
    "select_best_transport",
]
