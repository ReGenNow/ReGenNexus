"""
RegenNexus UAP - Plugin Tests

Tests for all device plugins using mock mode.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any


class TestRaspberryPiPlugin:
    """Tests for Raspberry Pi plugin."""

    @pytest_asyncio.fixture
    async def plugin(self):
        """Create plugin instance."""
        from regennexus.plugins import get_raspberry_pi_plugin
        RaspberryPiPlugin = get_raspberry_pi_plugin()
        plugin = RaspberryPiPlugin(entity_id="test_rpi", mock_mode=True)
        await plugin.initialize()
        yield plugin
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.initialized
        assert plugin.mock_mode
        assert "gpio.read" in plugin.capabilities
        assert "gpio.write" in plugin.capabilities

    @pytest.mark.asyncio
    async def test_gpio_read(self, plugin):
        """Test GPIO read command."""
        result = await plugin.execute_command("gpio.read", {"pin": 17})
        assert result["success"]
        assert "value" in result

    @pytest.mark.asyncio
    async def test_gpio_write(self, plugin):
        """Test GPIO write command."""
        result = await plugin.execute_command("gpio.write", {"pin": 18, "value": 1})
        assert result["success"]
        assert result["pin"] == 18
        assert result["value"] == 1

    @pytest.mark.asyncio
    async def test_gpio_pwm(self, plugin):
        """Test GPIO PWM command."""
        result = await plugin.execute_command("gpio.pwm", {
            "pin": 12,
            "frequency": 1000,
            "duty_cycle": 50,
            "action": "start"
        })
        assert result["success"]

    @pytest.mark.asyncio
    async def test_status_command(self, plugin):
        """Test status command."""
        result = await plugin.execute_command("status", {})
        assert result["success"]
        assert result["device_type"] == "raspberry_pi"
        assert result["mock_mode"]


class TestArduinoPlugin:
    """Tests for Arduino plugin."""

    @pytest_asyncio.fixture
    async def plugin(self):
        """Create plugin instance."""
        from regennexus.plugins import get_arduino_plugin
        ArduinoPlugin = get_arduino_plugin()
        plugin = ArduinoPlugin(entity_id="test_arduino", mock_mode=True)
        await plugin.initialize()
        yield plugin
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.initialized
        assert plugin.mock_mode
        assert plugin.connected  # In mock mode, always connected

    @pytest.mark.asyncio
    async def test_digital_read(self, plugin):
        """Test digital read."""
        result = await plugin.execute_command("digital_read", {"pin": 13})
        assert result["success"]
        assert "value" in result

    @pytest.mark.asyncio
    async def test_digital_write(self, plugin):
        """Test digital write."""
        result = await plugin.execute_command("digital_write", {"pin": 13, "value": 1})
        assert result["success"]
        assert result["value"] == 1

    @pytest.mark.asyncio
    async def test_analog_read(self, plugin):
        """Test analog read."""
        plugin.set_mock_analog(0, 512)
        result = await plugin.execute_command("analog_read", {"pin": 0})
        assert result["success"]
        assert result["value"] == 512

    @pytest.mark.asyncio
    async def test_analog_write(self, plugin):
        """Test analog write (PWM)."""
        result = await plugin.execute_command("analog_write", {"pin": 9, "value": 128})
        assert result["success"]
        assert result["value"] == 128


class TestJetsonPlugin:
    """Tests for Jetson plugin."""

    @pytest_asyncio.fixture
    async def plugin(self):
        """Create plugin instance."""
        from regennexus.plugins import get_jetson_plugin
        JetsonPlugin = get_jetson_plugin()
        plugin = JetsonPlugin(entity_id="test_jetson", mock_mode=True)
        await plugin.initialize()
        yield plugin
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.initialized
        assert plugin.mock_mode
        assert plugin.jetson_model == "Mock Jetson Orin"

    @pytest.mark.asyncio
    async def test_device_info(self, plugin):
        """Test device info command."""
        result = await plugin.execute_command("device_info", {})
        assert result["success"]
        assert "model" in result
        assert "cuda_available" in result

    @pytest.mark.asyncio
    async def test_gpio_operations(self, plugin):
        """Test GPIO operations."""
        # Write
        result = await plugin.execute_command("gpio.write", {"pin": 7, "value": 1})
        assert result["success"]

        # Read
        result = await plugin.execute_command("gpio.read", {"pin": 7})
        assert result["success"]
        assert result["value"] == 1

    @pytest.mark.asyncio
    async def test_camera_list(self, plugin):
        """Test camera list."""
        result = await plugin.execute_command("camera.list", {})
        assert result["success"]
        assert "cameras" in result


class TestIoTPlugin:
    """Tests for IoT plugin."""

    @pytest_asyncio.fixture
    async def plugin(self):
        """Create plugin instance."""
        from regennexus.plugins import get_iot_plugin
        IoTPlugin = get_iot_plugin()
        plugin = IoTPlugin(entity_id="test_iot", mock_mode=True)
        await plugin.initialize()
        yield plugin
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.initialized
        assert plugin.mock_mode

    @pytest.mark.asyncio
    async def test_mqtt_connect(self, plugin):
        """Test MQTT connect (mock)."""
        result = await plugin.execute_command("mqtt.connect", {
            "broker": "test.mosquitto.org",
            "port": 1883
        })
        assert result["success"]
        assert plugin.mqtt_connected

    @pytest.mark.asyncio
    async def test_mqtt_publish(self, plugin):
        """Test MQTT publish (mock)."""
        # Connect first
        await plugin.execute_command("mqtt.connect", {"broker": "test.mosquitto.org"})

        result = await plugin.execute_command("mqtt.publish", {
            "topic": "test/topic",
            "payload": {"value": 42}
        })
        assert result["success"]

        # Check mock messages
        messages = plugin.get_mock_mqtt_messages()
        assert len(messages) > 0
        assert messages[-1]["topic"] == "test/topic"

    @pytest.mark.asyncio
    async def test_http_get(self, plugin):
        """Test HTTP GET (mock)."""
        plugin.set_mock_http_response(
            "https://api.example.com/data",
            200,
            {"status": "ok", "value": 123}
        )

        result = await plugin.execute_command("http.get", {
            "url": "https://api.example.com/data"
        })
        assert result["success"]
        assert result["status"] == 200
        assert result["content"]["value"] == 123


class TestAmberB1Plugin:
    """Tests for Amber B1 robotic arm plugin."""

    @pytest_asyncio.fixture
    async def plugin(self):
        """Create plugin instance."""
        from regennexus.plugins import get_amber_b1_plugin
        AmberB1Plugin = get_amber_b1_plugin()
        plugin = AmberB1Plugin(entity_id="test_amber", mock_mode=True)
        await plugin.initialize()
        yield plugin
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.initialized
        assert plugin.mock_mode
        assert plugin.config.num_joints == 7
        assert plugin.config.has_gripper

    @pytest.mark.asyncio
    async def test_get_state(self, plugin):
        """Test get state command."""
        result = await plugin.execute_command("get_state", {})
        assert result["success"]
        assert result["state"] == "idle"
        assert len(result["positions"]) == 7

    @pytest.mark.asyncio
    async def test_move_joint(self, plugin):
        """Test move single joint."""
        result = await plugin.execute_command("move_joint", {
            "joint_id": 1,
            "position": 45.0
        })
        assert result["success"]
        assert result["joint_id"] == 1

    @pytest.mark.asyncio
    async def test_move_joints(self, plugin):
        """Test move all joints."""
        positions = [10, 20, 30, 40, 50, 60, 70]
        result = await plugin.execute_command("move_joints", {
            "positions": positions,
            "duration": 0.5
        })
        assert result["success"]

    @pytest.mark.asyncio
    async def test_home(self, plugin):
        """Test home command."""
        result = await plugin.execute_command("home", {})
        assert result["success"]

    @pytest.mark.asyncio
    async def test_gripper_operations(self, plugin):
        """Test gripper operations."""
        # Open
        result = await plugin.execute_command("gripper.open", {})
        assert result["success"]
        assert plugin.gripper_position == plugin.config.gripper_max_width

        # Close
        result = await plugin.execute_command("gripper.close", {"force": 15.0})
        assert result["success"]
        assert plugin.gripper_position == 0.0

        # Set position
        result = await plugin.execute_command("gripper.set_position", {"position": 40.0})
        assert result["success"]
        assert plugin.gripper_position == 40.0

    @pytest.mark.asyncio
    async def test_emergency_stop(self, plugin):
        """Test emergency stop."""
        result = await plugin.execute_command("emergency_stop", {})
        assert result["success"]
        assert plugin.state.value == "emergency_stop"


class TestLucidOnePlugin:
    """Tests for Lucid One robotic arm plugin."""

    @pytest_asyncio.fixture
    async def plugin(self):
        """Create plugin instance."""
        from regennexus.plugins import get_lucid_one_plugin
        LucidOnePlugin = get_lucid_one_plugin()
        plugin = LucidOnePlugin(entity_id="test_lucid", mock_mode=True)
        await plugin.initialize()
        yield plugin
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.initialized
        assert plugin.mock_mode
        assert plugin.config.num_joints == 7

    @pytest.mark.asyncio
    async def test_move_joints(self, plugin):
        """Test joint space motion."""
        positions = [10, 20, -30, 40, -50, 60, 0]
        result = await plugin.execute_command("move_joints", {
            "positions": positions,
            "velocity": 1.0,
            "duration": 0.5
        })
        assert result["success"]

    @pytest.mark.asyncio
    async def test_move_cartesian(self, plugin):
        """Test Cartesian space motion."""
        result = await plugin.execute_command("move_cartesian", {
            "pose": [400, 100, 300, 0, 180, 45],
            "velocity": 0.5
        })
        assert result["success"]

    @pytest.mark.asyncio
    async def test_get_cartesian_pose(self, plugin):
        """Test get Cartesian pose."""
        result = await plugin.execute_command("get_cartesian_pose", {})
        assert result["success"]
        assert "x" in result
        assert "y" in result
        assert "z" in result

    @pytest.mark.asyncio
    async def test_force_torque(self, plugin):
        """Test force/torque reading."""
        result = await plugin.execute_command("get_force_torque", {})
        assert result["success"]
        assert "fx" in result
        assert "fy" in result
        assert "fz" in result

    @pytest.mark.asyncio
    async def test_teach_mode(self, plugin):
        """Test teach mode toggle."""
        result = await plugin.execute_command("teach_mode", {"enable": True})
        assert result["success"]
        assert plugin.state.value == "teaching"

        result = await plugin.execute_command("teach_mode", {"enable": False})
        assert result["success"]
        assert plugin.state.value == "idle"

    @pytest.mark.asyncio
    async def test_trajectory_recording(self, plugin):
        """Test trajectory recording."""
        # Start recording
        result = await plugin.execute_command("record_trajectory", {"action": "start"})
        assert result["success"]
        assert plugin._recording

        # Add points
        for _ in range(3):
            result = await plugin.execute_command("record_trajectory", {"action": "add_point"})
            assert result["success"]

        # Stop recording
        result = await plugin.execute_command("record_trajectory", {"action": "stop"})
        assert result["success"]
        assert result["points"] == 3

    @pytest.mark.asyncio
    async def test_gripper(self, plugin):
        """Test gripper control."""
        result = await plugin.execute_command("gripper.open", {})
        assert result["success"]

        result = await plugin.execute_command("gripper.close", {"force": 10.0})
        assert result["success"]


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
