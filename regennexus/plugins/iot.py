"""
RegenNexus UAP - IoT Plugin

Generic IoT device integration with MQTT, HTTP, and CoAP support.
Includes mock mode for development without hardware.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
import ssl
from typing import Any, Callable, Dict, List, Optional

from regennexus.plugins.base import DevicePlugin, MockDeviceMixin

logger = logging.getLogger(__name__)

# Try to import aiohttp
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None

# Try to import MQTT client (library was renamed from asyncio_mqtt to aiomqtt)
try:
    import aiomqtt as asyncio_mqtt
    HAS_MQTT = True
except ImportError:
    try:
        import asyncio_mqtt
        HAS_MQTT = True
    except ImportError:
        HAS_MQTT = False
        asyncio_mqtt = None


class IoTPlugin(DevicePlugin, MockDeviceMixin):
    """
    IoT plugin for RegenNexus.

    Supports:
    - MQTT publish/subscribe
    - HTTP GET/POST/PUT/DELETE
    - Mock mode for development
    """

    def __init__(
        self,
        entity_id: str,
        protocol: Optional[Any] = None,
        mock_mode: bool = False
    ):
        """
        Initialize IoT plugin.

        Args:
            entity_id: Unique identifier
            protocol: Protocol instance
            mock_mode: If True, simulate without network
        """
        DevicePlugin.__init__(self, entity_id, "iot", protocol, mock_mode)
        MockDeviceMixin.__init__(self)

        self.mqtt_client = None
        self.mqtt_connected = False
        self.mqtt_subscriptions: Dict[str, List[Callable]] = {}
        self.mqtt_task = None
        self.http_session = None

        # Mock storage
        self._mock_mqtt_messages: List[Dict] = []
        self._mock_http_responses: Dict[str, Dict] = {}

    async def _device_init(self) -> bool:
        """Initialize IoT resources."""
        try:
            # Initialize HTTP session
            if HAS_AIOHTTP and not self.mock_mode:
                self.http_session = aiohttp.ClientSession()
            elif self.mock_mode:
                logger.info("IoT HTTP in mock mode")

            # Add capabilities
            self.capabilities.extend([
                "iot.http.get",
                "iot.http.post",
                "iot.http.put",
                "iot.http.delete",
            ])

            if HAS_MQTT or self.mock_mode:
                self.capabilities.extend([
                    "iot.mqtt.connect",
                    "iot.mqtt.publish",
                    "iot.mqtt.subscribe",
                    "iot.mqtt.disconnect",
                ])

            # Register command handlers
            self.register_command_handler("mqtt.connect", self._handle_mqtt_connect)
            self.register_command_handler("mqtt.publish", self._handle_mqtt_publish)
            self.register_command_handler("mqtt.subscribe", self._handle_mqtt_subscribe)
            self.register_command_handler("mqtt.disconnect", self._handle_mqtt_disconnect)
            self.register_command_handler("http.get", self._handle_http_get)
            self.register_command_handler("http.post", self._handle_http_post)
            self.register_command_handler("http.put", self._handle_http_put)
            self.register_command_handler("http.delete", self._handle_http_delete)

            # Update metadata
            self.metadata.update({
                "mqtt_available": HAS_MQTT or self.mock_mode,
                "http_available": HAS_AIOHTTP or self.mock_mode,
                "mqtt_connected": False,
            })

            return True

        except Exception as e:
            logger.error(f"IoT init error: {e}")
            return False

    async def _device_shutdown(self) -> None:
        """Clean up IoT resources."""
        # Disconnect MQTT
        if self.mqtt_task:
            self.mqtt_task.cancel()
            try:
                await self.mqtt_task
            except asyncio.CancelledError:
                pass
            self.mqtt_task = None

        if self.mqtt_client and self.mqtt_connected:
            try:
                await self.mqtt_client.disconnect()
            except Exception:
                pass
            self.mqtt_connected = False

        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def _handle_mqtt_connect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MQTT connect command."""
        if self.mqtt_connected:
            return {"success": True, "message": "Already connected"}

        broker = params.get("broker")
        port = params.get("port", 1883)
        username = params.get("username")
        password = params.get("password")
        client_id = params.get("client_id", f"regennexus-{self.entity_id}")
        use_tls = params.get("use_tls", False)

        if not broker:
            return {"success": False, "error": "Missing broker parameter"}

        try:
            if self.mock_mode:
                self.mqtt_connected = True
                self.mock_set_state("mqtt_broker", broker)
                self.mock_set_state("mqtt_port", port)
                logger.info(f"Mock connected to MQTT broker: {broker}:{port}")
            elif HAS_MQTT:
                tls_context = None
                if use_tls:
                    tls_context = ssl.create_default_context()

                self.mqtt_client = asyncio_mqtt.Client(
                    hostname=broker,
                    port=port,
                    username=username,
                    password=password,
                    client_id=client_id,
                    tls_context=tls_context
                )
                await self.mqtt_client.connect()
                self.mqtt_connected = True

                # Start MQTT loop
                self.mqtt_task = asyncio.create_task(self._mqtt_loop())

                logger.info(f"Connected to MQTT broker: {broker}:{port}")
            else:
                return {"success": False, "error": "MQTT library not available"}

            self.metadata["mqtt_connected"] = True

            return {
                "success": True,
                "broker": broker,
                "port": port,
                "client_id": client_id,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _mqtt_loop(self) -> None:
        """MQTT message loop."""
        try:
            async with self.mqtt_client.filtered_messages("#") as messages:
                await self.mqtt_client.subscribe("#")
                async for message in messages:
                    topic = message.topic.value
                    payload = message.payload.decode()

                    logger.debug(f"MQTT message: {topic} - {payload}")

                    try:
                        payload_data = json.loads(payload)
                    except json.JSONDecodeError:
                        payload_data = payload

                    await self.emit_event("iot.mqtt.message", {
                        "topic": topic,
                        "payload": payload_data,
                    })

                    if topic in self.mqtt_subscriptions:
                        for handler in self.mqtt_subscriptions[topic]:
                            try:
                                result = handler(topic, payload_data)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.error(f"MQTT handler error: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MQTT loop error: {e}")
            self.mqtt_connected = False
            self.metadata["mqtt_connected"] = False

    async def _handle_mqtt_publish(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MQTT publish command."""
        if not self.mqtt_connected:
            return {"success": False, "error": "Not connected to MQTT broker"}

        topic = params.get("topic")
        payload = params.get("payload")
        qos = params.get("qos", 0)
        retain = params.get("retain", False)

        if not topic or payload is None:
            return {"success": False, "error": "Missing topic or payload"}

        try:
            if isinstance(payload, dict):
                payload = json.dumps(payload)

            if self.mock_mode:
                self._mock_mqtt_messages.append({
                    "topic": topic,
                    "payload": payload,
                    "qos": qos,
                    "retain": retain,
                })
                logger.debug(f"Mock MQTT publish: {topic}")
            else:
                await self.mqtt_client.publish(
                    topic=topic,
                    payload=payload,
                    qos=qos,
                    retain=retain
                )

            return {"success": True, "topic": topic, "qos": qos}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_mqtt_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MQTT subscribe command."""
        if not self.mqtt_connected:
            return {"success": False, "error": "Not connected to MQTT broker"}

        topic = params.get("topic")
        qos = params.get("qos", 0)

        if not topic:
            return {"success": False, "error": "Missing topic parameter"}

        try:
            if not self.mock_mode and self.mqtt_client:
                await self.mqtt_client.subscribe(topic=topic, qos=qos)

            if topic not in self.mqtt_subscriptions:
                self.mqtt_subscriptions[topic] = []

            return {"success": True, "topic": topic, "qos": qos}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_mqtt_disconnect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MQTT disconnect command."""
        try:
            if self.mqtt_task:
                self.mqtt_task.cancel()
                try:
                    await self.mqtt_task
                except asyncio.CancelledError:
                    pass
                self.mqtt_task = None

            if self.mqtt_client and not self.mock_mode:
                await self.mqtt_client.disconnect()

            self.mqtt_connected = False
            self.metadata["mqtt_connected"] = False

            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_http_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP GET command."""
        url = params.get("url")
        headers = params.get("headers", {})
        query_params = params.get("params", {})

        if not url:
            return {"success": False, "error": "Missing url parameter"}

        try:
            if self.mock_mode:
                mock_response = self._mock_http_responses.get(url, {
                    "status": 200,
                    "content": {"mock": True},
                })
                return {
                    "success": True,
                    "status": mock_response.get("status", 200),
                    "content": mock_response.get("content", {}),
                    "mock": True,
                }

            if not self.http_session:
                return {"success": False, "error": "HTTP session not available"}

            async with self.http_session.get(
                url=url, headers=headers, params=query_params
            ) as response:
                try:
                    content = await response.json()
                except Exception:
                    content = await response.text()

                return {
                    "success": response.status < 400,
                    "status": response.status,
                    "url": url,
                    "content": content,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_http_post(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP POST command."""
        url = params.get("url")
        headers = params.get("headers", {})
        data = params.get("data")
        json_data = params.get("json")

        if not url:
            return {"success": False, "error": "Missing url parameter"}

        try:
            if self.mock_mode:
                return {
                    "success": True,
                    "status": 200,
                    "content": {"mock": True, "received": json_data or data},
                    "mock": True,
                }

            if not self.http_session:
                return {"success": False, "error": "HTTP session not available"}

            async with self.http_session.post(
                url=url, headers=headers, data=data, json=json_data
            ) as response:
                try:
                    content = await response.json()
                except Exception:
                    content = await response.text()

                return {
                    "success": response.status < 400,
                    "status": response.status,
                    "url": url,
                    "content": content,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_http_put(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP PUT command."""
        url = params.get("url")
        headers = params.get("headers", {})
        data = params.get("data")
        json_data = params.get("json")

        if not url:
            return {"success": False, "error": "Missing url parameter"}

        try:
            if self.mock_mode:
                return {
                    "success": True,
                    "status": 200,
                    "content": {"mock": True, "received": json_data or data},
                    "mock": True,
                }

            if not self.http_session:
                return {"success": False, "error": "HTTP session not available"}

            async with self.http_session.put(
                url=url, headers=headers, data=data, json=json_data
            ) as response:
                try:
                    content = await response.json()
                except Exception:
                    content = await response.text()

                return {
                    "success": response.status < 400,
                    "status": response.status,
                    "url": url,
                    "content": content,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_http_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP DELETE command."""
        url = params.get("url")
        headers = params.get("headers", {})

        if not url:
            return {"success": False, "error": "Missing url parameter"}

        try:
            if self.mock_mode:
                return {
                    "success": True,
                    "status": 200,
                    "content": {"mock": True, "deleted": True},
                    "mock": True,
                }

            if not self.http_session:
                return {"success": False, "error": "HTTP session not available"}

            async with self.http_session.delete(
                url=url, headers=headers
            ) as response:
                try:
                    content = await response.json()
                except Exception:
                    content = await response.text()

                return {
                    "success": response.status < 400,
                    "status": response.status,
                    "url": url,
                    "content": content,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # Convenience methods for testing
    def set_mock_http_response(self, url: str, status: int, content: Any) -> None:
        """Set mock HTTP response for testing."""
        self._mock_http_responses[url] = {"status": status, "content": content}

    def get_mock_mqtt_messages(self) -> List[Dict]:
        """Get mock MQTT messages for testing."""
        return self._mock_mqtt_messages.copy()

    def clear_mock_mqtt_messages(self) -> None:
        """Clear mock MQTT messages."""
        self._mock_mqtt_messages.clear()
