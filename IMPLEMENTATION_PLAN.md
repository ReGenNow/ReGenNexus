# RegenNexus UAP Implementation Plan

## Universal Adapter Protocol by ReGen Designs LLC

**Version Target**: 1.0.0
**Package Name**: `regennexus`
**CLI Command**: `regen`
**License**: MIT with Attribution

---

## Overview

RegenNexus UAP (Universal Adapter Protocol) is a fast, reliable, plug-and-play communication framework for devices, robots, and applications. Think of it like MCP (Model Context Protocol) but for hardware devices and IoT.

### Key Features
- Fast communication (< 0.1ms local, < 10ms network)
- Multiple transport options (IPC, UDP, WebSocket, Message Queue)
- Configurable security (encryption, authentication, rate limiting)
- REST & WebSocket API for external apps
- Device plugins (Raspberry Pi, Arduino, Jetson, ESP32, ROS2, Azure IoT)
- Single YAML config file for all settings
- Mock mode for testing without hardware

---

## Package Structure

```
E:\ReGenNexus-UAP\
│
├── regennexus/                      # Main package
│   ├── __init__.py                  # Package entry point
│   ├── __version__.py               # Version info
│   ├── cli.py                       # CLI entry point (regen command)
│   │
│   ├── core/                        # Core protocol
│   │   ├── __init__.py
│   │   ├── message.py               # Message class
│   │   ├── entity.py                # Entity base class
│   │   ├── registry.py              # Entity registry (central + P2P)
│   │   ├── context.py               # Context manager
│   │   └── protocol.py              # Main protocol coordinator
│   │
│   ├── transport/                   # Communication layer
│   │   ├── __init__.py
│   │   ├── base.py                  # Transport base class
│   │   ├── ipc.py                   # Local IPC (Unix sockets, shared memory)
│   │   ├── udp.py                   # UDP multicast (LAN discovery)
│   │   ├── websocket.py             # WebSocket (remote/internet)
│   │   ├── queue.py                 # Message queue (reliable delivery)
│   │   └── auto.py                  # Auto-select best transport
│   │
│   ├── security/                    # Security module
│   │   ├── __init__.py
│   │   ├── encryption.py            # ECDH-384, AES-256-GCM, AES-128
│   │   ├── authentication.py        # Token auth, API keys
│   │   ├── rate_limiter.py          # Rate limiting
│   │   ├── policy.py                # Access control policies
│   │   └── manager.py               # Security manager (coordinates all)
│   │
│   ├── api/                         # API server
│   │   ├── __init__.py
│   │   ├── rest.py                  # REST API (FastAPI)
│   │   ├── websocket_api.py         # WebSocket API
│   │   ├── routes/                  # API routes
│   │   │   ├── __init__.py
│   │   │   ├── devices.py           # /api/devices/*
│   │   │   ├── messages.py          # /api/messages/*
│   │   │   ├── registry.py          # /api/registry/*
│   │   │   └── health.py            # /api/health
│   │   └── docs.py                  # OpenAPI/Swagger docs
│   │
│   ├── devices/                     # Device plugins
│   │   ├── __init__.py
│   │   ├── base.py                  # Device plugin base class
│   │   ├── raspberry_pi.py          # Raspberry Pi (GPIO, camera, sensors)
│   │   ├── arduino.py               # Arduino (serial communication)
│   │   ├── jetson.py                # NVIDIA Jetson (CUDA, AI)
│   │   ├── esp32.py                 # ESP32 (WiFi, sensors)
│   │   ├── generic_iot.py           # Generic IoT (MQTT, HTTP, CoAP)
│   │   └── mock.py                  # Mock devices for testing
│   │
│   ├── bridges/                     # External system bridges
│   │   ├── __init__.py
│   │   ├── ros2.py                  # ROS 2 bridge
│   │   ├── azure_iot.py             # Azure IoT Hub
│   │   └── mqtt.py                  # MQTT broker bridge
│   │
│   ├── config/                      # Configuration
│   │   ├── __init__.py
│   │   ├── loader.py                # YAML config loader
│   │   ├── validator.py             # Config validation
│   │   └── defaults.py              # Default configuration
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logging.py               # Logging setup
│       ├── errors.py                # Custom exceptions
│       └── helpers.py               # Helper functions
│
├── docs/                            # Documentation
│   ├── index.md                     # Documentation home
│   ├── getting-started.md           # Quick start guide
│   ├── installation.md              # Installation instructions
│   ├── configuration.md             # Config file reference
│   ├── security.md                  # Security guide
│   ├── api/                         # API documentation
│   │   ├── rest-api.md              # REST API reference
│   │   ├── websocket-api.md         # WebSocket API reference
│   │   └── python-api.md            # Python API reference
│   ├── devices/                     # Device guides
│   │   ├── raspberry-pi.md
│   │   ├── arduino.md
│   │   ├── jetson.md
│   │   ├── esp32.md
│   │   └── ros2.md
│   ├── examples/                    # Example guides
│   │   ├── basic-usage.md
│   │   ├── device-communication.md
│   │   └── building-apps.md
│   └── api-reference/               # Auto-generated API docs
│
├── examples/                        # Example code
│   ├── basic/                       # Basic examples
│   │   ├── hello_world.py           # Simplest example
│   │   ├── send_receive.py          # Send/receive messages
│   │   └── multi_device.py          # Multiple devices
│   │
│   ├── devices/                     # Hardware examples
│   │   ├── raspberry_pi_gpio.py
│   │   ├── arduino_serial.py
│   │   ├── jetson_camera.py
│   │   └── esp32_sensor.py
│   │
│   ├── mock/                        # No-hardware examples
│   │   ├── mock_devices.py          # Simulated devices
│   │   ├── mock_network.py          # Simulated network
│   │   └── test_communication.py    # Test without hardware
│   │
│   ├── api/                         # API examples
│   │   ├── rest_client.py           # REST API client
│   │   ├── websocket_client.py      # WebSocket client
│   │   └── web_dashboard/           # Simple web dashboard
│   │
│   ├── ros2/                        # ROS 2 examples
│   │   ├── amber_b1_arm.py
│   │   ├── amber_lucid1_arm.py
│   │   └── generic_robot.py
│   │
│   └── colab/                       # Google Colab notebooks
│       ├── getting_started.ipynb
│       ├── security_demo.ipynb
│       └── mock_devices.ipynb
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_core/
│   │   ├── test_message.py
│   │   ├── test_entity.py
│   │   ├── test_registry.py
│   │   └── test_protocol.py
│   ├── test_transport/
│   │   ├── test_ipc.py
│   │   ├── test_udp.py
│   │   ├── test_websocket.py
│   │   └── test_queue.py
│   ├── test_security/
│   │   ├── test_encryption.py
│   │   ├── test_authentication.py
│   │   └── test_rate_limiter.py
│   ├── test_api/
│   │   ├── test_rest.py
│   │   └── test_websocket_api.py
│   ├── test_devices/
│   │   ├── test_mock_devices.py
│   │   └── test_device_base.py
│   └── test_integration/
│       └── test_full_flow.py
│
├── setup.py                         # Package setup (legacy)
├── pyproject.toml                   # Modern Python packaging
├── requirements.txt                 # Dependencies
├── requirements-dev.txt             # Development dependencies
├── README.md                        # Main README
├── LICENSE                          # MIT License
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Contribution guide
├── regennexus-config.example.yaml   # Example config
└── .github/                         # GitHub Actions
    └── workflows/
        ├── test.yml                 # Run tests on PR
        └── publish.yml              # Publish to PyPI
```

---

## Implementation Phases

### Phase 1: Bug Fixes & Restructure (Foundation)

**Goal**: Fix all critical bugs and set up new package structure

#### 1.1 Fix Critical Bugs
- [ ] Fix `protocol_core.py` - Move misplaced methods back into Entity class (lines 231-260)
- [ ] Fix `protocol_core.py` - Replace deprecated `asyncio.get_event_loop().time()` with `time.time()`
- [ ] Add missing `get_entity()` and `find_entities()` methods to ProtocolCore
- [ ] Fix `test_basic_protocol.py` import path
- [ ] Fix `jetson.py` - Make it inherit from DevicePlugin
- [ ] Fix `ros_bridge.py` - Replace `exec()`/`eval()` with `importlib`
- [ ] Fix `arduino.py` - Wrap serial import in try/except
- [ ] Fix `iot.py` - Update `asyncio_mqtt` to `aiomqtt`
- [ ] Fix `raspberry_pi.py` - Initialize `self.camera` in `__init__`
- [ ] Fix `policy.py` - Complete CIDR implementation

#### 1.2 Create New Package Structure
- [ ] Create `regennexus/` directory structure as shown above
- [ ] Move and refactor existing code into new structure
- [ ] Create `__init__.py` files with proper exports
- [ ] Create `__version__.py` with version info

#### 1.3 Setup Packaging
- [ ] Create `pyproject.toml` for modern packaging
- [ ] Update `setup.py` for legacy compatibility
- [ ] Create `requirements.txt` with core dependencies
- [ ] Create `requirements-dev.txt` with dev dependencies

**Deliverables**:
- All critical bugs fixed
- New package structure in place
- Package installable with `pip install -e .`

---

### Phase 2: Transport Layer (Speed)

**Goal**: Implement all communication methods for fast device communication

#### 2.1 Transport Base
- [ ] Create `transport/base.py` - Abstract base class for all transports
  - `connect()`, `disconnect()`, `send()`, `receive()`, `is_connected()`
  - Common message format

#### 2.2 Local IPC (Fastest - < 0.1ms)
- [ ] Create `transport/ipc.py`
  - Unix domain sockets (Linux/Mac)
  - Named pipes (Windows)
  - Shared memory option for ultra-fast
  - Auto-detect OS and use best method

#### 2.3 UDP Multicast (LAN Discovery - 1-5ms)
- [ ] Create `transport/udp.py`
  - Device discovery via multicast
  - Broadcast presence announcements
  - Direct UDP messaging for speed
  - Handle packet loss gracefully

#### 2.4 WebSocket (Remote - 10-50ms)
- [ ] Refactor existing `websocket` code into `transport/websocket.py`
  - Client and server modes
  - Auto-reconnection
  - SSL/TLS support
  - Heartbeat/ping-pong

#### 2.5 Message Queue (Reliable)
- [ ] Create `transport/queue.py`
  - In-memory queue (default)
  - Persistent queue option (file-based)
  - Retry logic with exponential backoff
  - Dead letter queue for failed messages

#### 2.6 Auto-Select Transport
- [ ] Create `transport/auto.py`
  - Detect if target is local → use IPC
  - Detect if target is on LAN → use UDP/WebSocket
  - Detect if target is remote → use WebSocket
  - Fallback chain: IPC → UDP → WebSocket → Queue

**Deliverables**:
- All 4 transport methods working
- Auto-selection working
- Benchmark tests showing speed

---

### Phase 3: Security Module (Protection)

**Goal**: Implement all security features, all optional/configurable

#### 3.1 Encryption
- [ ] Create `security/encryption.py`
  - ECDH-384 + AES-256-GCM (strongest)
  - AES-256-CBC (compatible)
  - AES-128 (lightweight for IoT)
  - No encryption option
  - Auto-negotiate based on capabilities

#### 3.2 Authentication
- [ ] Create `security/authentication.py`
  - Token-based auth (JWT)
  - API key auth
  - Certificate-based auth (for advanced users)
  - Anonymous mode (for testing)

#### 3.3 Rate Limiting
- [ ] Create `security/rate_limiter.py`
  - Token bucket algorithm
  - Per-client limits
  - Per-endpoint limits
  - Configurable thresholds

#### 3.4 Security Manager
- [ ] Create `security/manager.py`
  - Coordinates all security features
  - Easy enable/disable via config
  - Security level presets (none, basic, medium, maximum)

**Deliverables**:
- All security features working
- Security completely optional
- Security level presets for easy configuration

---

### Phase 4: API Server (Integration)

**Goal**: REST and WebSocket APIs for external applications

#### 4.1 REST API
- [ ] Create `api/rest.py` using FastAPI
  - `GET /api/health` - Health check
  - `GET /api/devices` - List devices
  - `GET /api/devices/{id}` - Get device info
  - `POST /api/devices/{id}/command` - Send command
  - `GET /api/messages` - Get message history
  - `POST /api/messages` - Send message
  - `GET /api/registry` - Get registered entities

#### 4.2 WebSocket API
- [ ] Create `api/websocket_api.py`
  - Real-time message streaming
  - Device status updates
  - Command/response pattern
  - Pub/sub for events

#### 4.3 API Documentation
- [ ] Create `api/docs.py`
  - Auto-generated OpenAPI/Swagger
  - Interactive API explorer at `/docs`
  - ReDoc alternative at `/redoc`

#### 4.4 API Security
- [ ] Integrate with security module
  - API key validation
  - Token validation
  - Rate limiting
  - CORS configuration

**Deliverables**:
- REST API working at configurable port
- WebSocket API working
- Swagger documentation at `/docs`

---

### Phase 5: Device Plugins (Hardware)

**Goal**: Fix and enhance all device plugins

#### 5.1 Base Plugin
- [ ] Refactor `devices/base.py`
  - Standard interface for all devices
  - Mock mode support
  - Auto-registration with registry
  - Event system for device events

#### 5.2 Raspberry Pi
- [ ] Fix and enhance `devices/raspberry_pi.py`
  - GPIO read/write
  - PWM support
  - Camera capture
  - Sensor support (DHT22, etc.)
  - Mock mode

#### 5.3 Arduino
- [ ] Fix and enhance `devices/arduino.py`
  - Serial communication
  - Digital/analog read/write
  - Auto-detect port
  - Mock mode

#### 5.4 Jetson
- [ ] Fix `devices/jetson.py` to inherit from base
  - GPIO support
  - CUDA detection
  - Camera support
  - TensorRT inference (basic)
  - Mock mode

#### 5.5 ESP32
- [ ] Create `devices/esp32.py`
  - WiFi communication
  - Sensor support
  - Deep sleep support
  - Mock mode

#### 5.6 Generic IoT
- [ ] Enhance `devices/generic_iot.py`
  - MQTT support
  - HTTP support
  - CoAP support
  - Mock mode

#### 5.7 Mock Devices
- [ ] Create `devices/mock.py`
  - Simulate any device type
  - Configurable responses
  - Latency simulation
  - Error simulation

**Deliverables**:
- All device plugins working
- Mock mode for all devices
- Consistent interface across all devices

---

### Phase 6: Configuration System

**Goal**: Single YAML config file for everything

#### 6.1 Config Loader
- [ ] Create `config/loader.py`
  - Load YAML config file
  - Environment variable overrides
  - Command-line argument overrides
  - Default config if none provided

#### 6.2 Config Validator
- [ ] Create `config/validator.py`
  - Validate config structure
  - Type checking
  - Required fields
  - Helpful error messages

#### 6.3 Default Config
- [ ] Create `config/defaults.py`
  - Sensible defaults for all options
  - Works out of the box
  - Easy to customize

#### 6.4 Example Config
- [ ] Create `regennexus-config.example.yaml`
  - Full example with all options
  - Detailed comments explaining each option

**Deliverables**:
- Config system working
- Example config file
- Config validation with helpful errors

---

### Phase 7: CLI (Command Line)

**Goal**: `regen` command for terminal usage

#### 7.1 Main CLI
- [ ] Create `cli.py` using Click or Typer
  - `regen start` - Start RegenNexus
  - `regen stop` - Stop RegenNexus
  - `regen status` - Show status
  - `regen config` - Show/edit config

#### 7.2 Device Commands
- [ ] Add device commands
  - `regen devices list` - List devices
  - `regen devices info <id>` - Show device info
  - `regen devices mock` - Start mock devices

#### 7.3 Message Commands
- [ ] Add message commands
  - `regen send <target> <message>` - Send message
  - `regen listen` - Listen for messages

#### 7.4 Debug Commands
- [ ] Add debug commands
  - `regen debug` - Debug mode
  - `regen benchmark` - Run benchmarks
  - `regen doctor` - Check installation

**Deliverables**:
- `regen` command working
- All subcommands working
- Help text for all commands

---

### Phase 8: Documentation

**Goal**: Complete documentation for all features

#### 8.1 User Documentation
- [ ] Write `docs/index.md` - Documentation home
- [ ] Write `docs/getting-started.md` - Quick start (5 minutes)
- [ ] Write `docs/installation.md` - Installation guide
- [ ] Write `docs/configuration.md` - Config reference

#### 8.2 Device Documentation
- [ ] Write `docs/devices/raspberry-pi.md`
- [ ] Write `docs/devices/arduino.md`
- [ ] Write `docs/devices/jetson.md`
- [ ] Write `docs/devices/esp32.md`
- [ ] Write `docs/devices/ros2.md`

#### 8.3 API Documentation
- [ ] Write `docs/api/rest-api.md`
- [ ] Write `docs/api/websocket-api.md`
- [ ] Write `docs/api/python-api.md`

#### 8.4 Auto-Generated Docs
- [ ] Setup pdoc or sphinx for API reference
- [ ] Generate from docstrings
- [ ] Include in `/docs/api-reference/`

#### 8.5 README
- [ ] Update `README.md` with:
  - Logo and badges
  - Quick start example
  - Feature list
  - Installation instructions
  - Links to documentation

**Deliverables**:
- Complete documentation
- Auto-generated API reference
- Updated README

---

### Phase 9: Examples & Tests

**Goal**: Working examples and comprehensive tests

#### 9.1 Basic Examples
- [ ] Create `examples/basic/hello_world.py`
- [ ] Create `examples/basic/send_receive.py`
- [ ] Create `examples/basic/multi_device.py`

#### 9.2 Mock Examples
- [ ] Create `examples/mock/mock_devices.py`
- [ ] Create `examples/mock/test_communication.py`

#### 9.3 Device Examples
- [ ] Create `examples/devices/raspberry_pi_gpio.py`
- [ ] Create `examples/devices/arduino_serial.py`
- [ ] Create `examples/devices/jetson_camera.py`

#### 9.4 Colab Notebooks
- [ ] Create `examples/colab/getting_started.ipynb`
- [ ] Create `examples/colab/security_demo.ipynb`
- [ ] Create `examples/colab/mock_devices.ipynb`

#### 9.5 Test Suite
- [ ] Create tests for core module
- [ ] Create tests for transport module
- [ ] Create tests for security module
- [ ] Create tests for API module
- [ ] Create integration tests
- [ ] Setup pytest with fixtures
- [ ] Add coverage reporting

**Deliverables**:
- All examples working
- Test suite with >80% coverage
- Colab notebooks working

---

### Phase 10: Packaging & Release

**Goal**: Publish to PyPI and update GitHub

#### 10.1 PyPI Packaging
- [ ] Finalize `pyproject.toml`
- [ ] Create `setup.py` for legacy support
- [ ] Test installation: `pip install .`
- [ ] Test with `pip install -e .` (editable)
- [ ] Build package: `python -m build`
- [ ] Test upload to TestPyPI
- [ ] Upload to PyPI

#### 10.2 GitHub Release
- [ ] Update `README.md`
- [ ] Update `CHANGELOG.md`
- [ ] Create GitHub release
- [ ] Tag version `v1.0.0`

#### 10.3 CI/CD
- [ ] Setup GitHub Actions for tests
- [ ] Setup GitHub Actions for PyPI publish
- [ ] Add badges to README

**Deliverables**:
- Package on PyPI: `pip install regennexus`
- GitHub release with changelog
- CI/CD pipeline working

---

## Dependencies

### Core Dependencies (required)
```
pyyaml>=6.0              # Config files
websockets>=10.0         # WebSocket transport
aiohttp>=3.8.0           # Async HTTP
pycryptodome>=3.15.0     # Encryption
fastapi>=0.100.0         # REST API
uvicorn>=0.20.0          # ASGI server
click>=8.0.0             # CLI
pydantic>=2.0.0          # Data validation
```

### Optional Dependencies
```
# Raspberry Pi
RPi.GPIO>=0.7.0
picamera>=1.13

# Arduino
pyserial>=3.5

# Jetson
Jetson.GPIO>=2.0.0

# MQTT
aiomqtt>=1.0.0

# ROS 2
rclpy>=1.0.0

# Azure IoT
azure-iot-device>=2.0.0
```

### Development Dependencies
```
pytest>=7.0.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.0.0
mypy>=1.0.0
pdoc>=12.0.0
```

---

## Timeline Estimate

| Phase | Estimated Effort | Priority |
|-------|------------------|----------|
| Phase 1: Bug Fixes & Restructure | Core foundation | CRITICAL |
| Phase 2: Transport Layer | Speed/communication | HIGH |
| Phase 3: Security Module | Protection | HIGH |
| Phase 4: API Server | Integration | HIGH |
| Phase 5: Device Plugins | Hardware support | MEDIUM |
| Phase 6: Configuration | User experience | MEDIUM |
| Phase 7: CLI | User experience | MEDIUM |
| Phase 8: Documentation | Adoption | HIGH |
| Phase 9: Examples & Tests | Quality | HIGH |
| Phase 10: Packaging | Distribution | CRITICAL |

---

## Success Criteria

### Performance
- [ ] Local IPC: < 0.1ms latency
- [ ] UDP/LAN: < 5ms latency
- [ ] WebSocket: < 50ms latency
- [ ] 1000+ messages/second throughput

### Reliability
- [ ] 99.9% message delivery (with queue)
- [ ] Auto-reconnection on disconnect
- [ ] Graceful degradation

### Usability
- [ ] Works out of the box with defaults
- [ ] Single config file for everything
- [ ] Clear error messages
- [ ] Comprehensive documentation

### Quality
- [ ] >80% test coverage
- [ ] No critical bugs
- [ ] Type hints throughout
- [ ] Consistent code style

---

## License

```
MIT License

Copyright (c) 2024 ReGen Designs LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

ATTRIBUTION NOTICE:
This software is RegenNexus UAP (Universal Adapter Protocol), developed by
ReGen Designs LLC. The RegenNexus name and branding must be retained in all
copies and substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Next Steps

1. Review and approve this plan
2. Start Phase 1: Fix critical bugs and restructure
3. Proceed through phases in order
4. Test each phase before moving to next
5. Final testing and PyPI release

---

*Document created: December 2024*
*RegenNexus UAP - Universal Adapter Protocol*
*By ReGen Designs LLC*
