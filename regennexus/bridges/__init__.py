"""
RegenNexus UAP - Bridges Module

Bridges to external systems (ROS2, Azure IoT, MQTT, LLMs).

Copyright (c) 2024-2025 ReGen Designs LLC
"""

# Import bridges (existing implementations)
try:
    from regennexus.bridges.ros_bridge import ROSBridge
except ImportError:
    ROSBridge = None

try:
    from regennexus.bridges.azure_bridge import AzureBridge
except ImportError:
    AzureBridge = None

# LLM Bridge (Ollama, LM Studio, etc.)
try:
    from regennexus.bridges.llm_bridge import (
        LLMBridge,
        LLMConfig,
        LLMProvider,
        LLMResponse,
        UAP_LLM_Agent,
    )
except ImportError:
    LLMBridge = None
    LLMConfig = None
    LLMProvider = None
    LLMResponse = None
    UAP_LLM_Agent = None

# MCP Bridge (Model Context Protocol for hardware)
try:
    from regennexus.bridges.mcp_bridge import (
        MCPServer,
        MCPTool,
        MCPResource,
        MCPPrompt,
        create_hardware_mcp_server,
    )
except ImportError:
    MCPServer = None
    MCPTool = None
    MCPResource = None
    MCPPrompt = None
    create_hardware_mcp_server = None

__all__ = [
    "ROSBridge",
    "AzureBridge",
    "LLMBridge",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "UAP_LLM_Agent",
    "MCPServer",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "create_hardware_mcp_server",
]
