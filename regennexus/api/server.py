"""
RegenNexus UAP - API Server

REST and WebSocket API server using FastAPI.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    BaseModel = object

# Try to import uvicorn
try:
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False


# Pydantic models for API
if HAS_FASTAPI:

    class MessageRequest(BaseModel):
        """Request to send a message."""
        target: str = Field(..., description="Target entity ID")
        intent: str = Field(..., description="Message intent")
        content: Any = Field(default=None, description="Message content")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class MessageResponse(BaseModel):
        """Response for a message."""
        success: bool
        message_id: Optional[str] = None
        error: Optional[str] = None

    class EntityInfo(BaseModel):
        """Entity information."""
        id: str
        type: str
        capabilities: List[str] = []
        metadata: Dict[str, Any] = {}

    class StatusResponse(BaseModel):
        """Server status response."""
        status: str
        version: str
        uptime: float
        entities: int
        transports: Dict[str, bool] = {}


def create_app(
    regen_instance: Optional[Any] = None,
    title: str = "RegenNexus UAP API",
    cors_origins: Optional[List[str]] = None,
    docs_enabled: bool = True
) -> "FastAPI":
    """
    Create a FastAPI application for RegenNexus.

    Args:
        regen_instance: RegenNexus instance
        title: API title
        cors_origins: Allowed CORS origins
        docs_enabled: Enable Swagger docs

    Returns:
        FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI required for API server. "
            "Install with: pip install fastapi uvicorn"
        )

    from regennexus.__version__ import __version__

    app = FastAPI(
        title=title,
        description="Universal Adapter Protocol API",
        version=__version__,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
    )

    # CORS middleware
    if cors_origins is None:
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store RegenNexus instance
    app.state.regen = regen_instance
    app.state.start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
    app.state.websocket_clients: Dict[str, WebSocket] = {}

    # =========================================================================
    # Health & Status Endpoints
    # =========================================================================

    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "RegenNexus UAP API",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/health", tags=["Health"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/status", response_model=StatusResponse, tags=["Health"])
    async def status():
        """Get server status."""
        regen = app.state.regen
        uptime = 0
        if hasattr(app.state, 'start_time') and app.state.start_time:
            try:
                uptime = asyncio.get_event_loop().time() - app.state.start_time
            except Exception:
                pass

        entity_count = 0
        transports = {}

        if regen:
            if hasattr(regen, 'registry'):
                entity_count = len(regen.registry.entities) if regen.registry else 0
            if hasattr(regen, '_transport'):
                transports["auto"] = regen._transport.is_connected if regen._transport else False

        return StatusResponse(
            status="running",
            version=__version__,
            uptime=uptime,
            entities=entity_count,
            transports=transports,
        )

    # =========================================================================
    # Entity Endpoints
    # =========================================================================

    @app.get("/entities", response_model=List[EntityInfo], tags=["Entities"])
    async def list_entities():
        """List all registered entities."""
        regen = app.state.regen
        if not regen or not hasattr(regen, 'registry'):
            return []

        entities = []
        for entity_id, entity in regen.registry.entities.items():
            entities.append(EntityInfo(
                id=entity_id,
                type=entity.entity_type,
                capabilities=list(entity.capabilities),
                metadata=entity.metadata,
            ))
        return entities

    @app.get("/entities/{entity_id}", response_model=EntityInfo, tags=["Entities"])
    async def get_entity(entity_id: str):
        """Get entity information."""
        regen = app.state.regen
        if not regen or not hasattr(regen, 'registry'):
            raise HTTPException(status_code=404, detail="Entity not found")

        entity = regen.registry.get(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        return EntityInfo(
            id=entity_id,
            type=entity.entity_type,
            capabilities=list(entity.capabilities),
            metadata=entity.metadata,
        )

    # =========================================================================
    # Messaging Endpoints
    # =========================================================================

    @app.post("/send", response_model=MessageResponse, tags=["Messaging"])
    async def send_message(request: MessageRequest):
        """Send a message to an entity."""
        regen = app.state.regen
        if not regen:
            raise HTTPException(status_code=503, detail="RegenNexus not initialized")

        try:
            from regennexus.core.message import Message

            msg = Message(
                source="api",
                target=request.target,
                intent=request.intent,
                content=request.content,
                metadata=request.metadata,
            )

            success = await regen.send(msg)

            return MessageResponse(
                success=success,
                message_id=msg.id if success else None,
                error=None if success else "Failed to send message",
            )

        except Exception as e:
            logger.error(f"Send error: {e}")
            return MessageResponse(
                success=False,
                error=str(e),
            )

    @app.post("/broadcast", response_model=MessageResponse, tags=["Messaging"])
    async def broadcast_message(request: MessageRequest):
        """Broadcast a message to all entities."""
        regen = app.state.regen
        if not regen:
            raise HTTPException(status_code=503, detail="RegenNexus not initialized")

        try:
            from regennexus.core.message import Message

            msg = Message(
                source="api",
                target="*",
                intent=request.intent,
                content=request.content,
                metadata=request.metadata,
                is_broadcast=True,
            )

            count = await regen.broadcast(msg)

            return MessageResponse(
                success=count > 0,
                message_id=msg.id,
            )

        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            return MessageResponse(
                success=False,
                error=str(e),
            )

    # =========================================================================
    # WebSocket Endpoint
    # =========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time communication."""
        await websocket.accept()
        client_id = str(id(websocket))
        app.state.websocket_clients[client_id] = websocket

        try:
            while True:
                data = await websocket.receive_json()

                # Handle different message types
                msg_type = data.get("type", "message")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "subscribe":
                    # Subscribe to entity updates
                    entity_id = data.get("entity_id")
                    if entity_id:
                        # Store subscription (simplified)
                        await websocket.send_json({
                            "type": "subscribed",
                            "entity_id": entity_id,
                        })

                elif msg_type == "send":
                    # Send a message
                    regen = app.state.regen
                    if regen:
                        from regennexus.core.message import Message
                        msg = Message(
                            source=f"ws:{client_id}",
                            target=data.get("target", "*"),
                            intent=data.get("intent", "message"),
                            content=data.get("content"),
                        )
                        success = await regen.send(msg)
                        await websocket.send_json({
                            "type": "sent",
                            "success": success,
                            "message_id": msg.id,
                        })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}",
                    })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            app.state.websocket_clients.pop(client_id, None)

    return app


class APIServer:
    """
    RegenNexus API Server.

    Manages the FastAPI server lifecycle.
    """

    def __init__(
        self,
        regen_instance: Optional[Any] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        cors_origins: Optional[List[str]] = None,
        docs_enabled: bool = True
    ):
        """
        Initialize API server.

        Args:
            regen_instance: RegenNexus instance
            host: Host to bind to
            port: Port to listen on
            cors_origins: Allowed CORS origins
            docs_enabled: Enable API documentation
        """
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI required. Install with: pip install fastapi"
            )
        if not HAS_UVICORN:
            raise ImportError(
                "uvicorn required. Install with: pip install uvicorn"
            )

        self.host = host
        self.port = port
        self.app = create_app(
            regen_instance=regen_instance,
            cors_origins=cors_origins,
            docs_enabled=docs_enabled,
        )
        self._server = None
        self._server_task = None

    async def start(self) -> None:
        """Start the API server."""
        import time
        self.app.state.start_time = time.time()

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        # Run in background
        self._server_task = asyncio.create_task(self._server.serve())

        logger.info(f"API server started on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            self._server.should_exit = True

        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        logger.info("API server stopped")

    def run(self) -> None:
        """Run the API server (blocking)."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
        )

    async def broadcast_to_websockets(self, data: Dict) -> int:
        """
        Broadcast data to all WebSocket clients.

        Args:
            data: Data to broadcast

        Returns:
            Number of clients sent to
        """
        count = 0
        for client_id, ws in list(self.app.state.websocket_clients.items()):
            try:
                await ws.send_json(data)
                count += 1
            except Exception:
                # Remove dead connections
                self.app.state.websocket_clients.pop(client_id, None)
        return count

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server is not None and not self._server.should_exit
