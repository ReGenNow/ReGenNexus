"""
RegenNexus UAP - Custom Exceptions

Custom exception classes for RegenNexus.

Copyright (c) 2024 ReGen Designs LLC
"""


class RegenNexusError(Exception):
    """Base exception for RegenNexus errors."""

    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code or "REGENNEXUS_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(RegenNexusError):
    """Error in configuration."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)


class ConnectionError(RegenNexusError):
    """Connection error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONNECTION_ERROR", details)


class AuthenticationError(RegenNexusError):
    """Authentication error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationError(RegenNexusError):
    """Authorization error (permission denied)."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUTHZ_ERROR", details)


class TransportError(RegenNexusError):
    """Transport layer error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "TRANSPORT_ERROR", details)


class MessageError(RegenNexusError):
    """Message handling error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "MESSAGE_ERROR", details)


class EntityError(RegenNexusError):
    """Entity-related error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "ENTITY_ERROR", details)


class EntityNotFoundError(EntityError):
    """Entity not found error."""

    def __init__(self, entity_id: str):
        super().__init__(
            f"Entity not found: {entity_id}",
            {"entity_id": entity_id},
        )
        self.code = "ENTITY_NOT_FOUND"


class EncryptionError(RegenNexusError):
    """Encryption/decryption error."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "ENCRYPTION_ERROR", details)


class RateLimitError(RegenNexusError):
    """Rate limit exceeded error."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class TimeoutError(RegenNexusError):
    """Operation timeout error."""

    def __init__(self, message: str = "Operation timed out", details: dict = None):
        super().__init__(message, "TIMEOUT_ERROR", details)


class DeviceError(RegenNexusError):
    """Device-related error."""

    def __init__(self, message: str, device_type: str = None, details: dict = None):
        details = details or {}
        if device_type:
            details["device_type"] = device_type
        super().__init__(message, "DEVICE_ERROR", details)


class DeviceNotConnectedError(DeviceError):
    """Device not connected error."""

    def __init__(self, device_id: str, device_type: str = None):
        super().__init__(
            f"Device not connected: {device_id}",
            device_type,
            {"device_id": device_id},
        )
        self.code = "DEVICE_NOT_CONNECTED"
