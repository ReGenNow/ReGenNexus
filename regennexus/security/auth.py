"""
RegenNexus UAP - Authentication Module

Provides authentication mechanisms:
- Token-based authentication (JWT-like)
- API key authentication
- Certificate-based authentication

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import pycryptodome
try:
    from Crypto.PublicKey import ECC
    from Crypto.Hash import SHA384
    from Crypto.Signature import DSS
    from Crypto.Random import get_random_bytes
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class AuthResult(Enum):
    """Authentication result codes."""
    SUCCESS = auto()
    INVALID_TOKEN = auto()
    EXPIRED_TOKEN = auto()
    REVOKED_TOKEN = auto()
    INVALID_KEY = auto()
    INVALID_SIGNATURE = auto()
    MISSING_CREDENTIALS = auto()
    PERMISSION_DENIED = auto()


@dataclass
class AuthenticatedEntity:
    """Information about an authenticated entity."""
    entity_id: str
    auth_method: str
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[float] = None


class Authenticator:
    """Base authenticator interface."""

    def authenticate(self, credentials: Dict[str, Any]) -> Tuple[AuthResult, Optional[AuthenticatedEntity]]:
        """
        Authenticate using provided credentials.

        Args:
            credentials: Authentication credentials

        Returns:
            Tuple of (result, authenticated_entity)
        """
        raise NotImplementedError


class TokenAuth(Authenticator):
    """
    Token-based authentication.

    Generates and validates signed tokens with expiration.
    """

    def __init__(self, secret: Optional[str] = None, expire_hours: int = 24):
        """
        Initialize token authenticator.

        Args:
            secret: Secret key for signing tokens (generated if not provided)
            expire_hours: Default token expiration in hours
        """
        self._secret = (secret or secrets.token_hex(32)).encode("utf-8")
        self._expire_hours = expire_hours
        self._revoked_tokens: Set[str] = set()

    def generate_token(
        self,
        entity_id: str,
        permissions: Optional[List[str]] = None,
        expire_hours: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an authentication token.

        Args:
            entity_id: Entity identifier
            permissions: List of permissions
            expire_hours: Token expiration (uses default if not specified)
            metadata: Additional metadata

        Returns:
            Signed token string
        """
        token_id = str(uuid.uuid4())
        now = time.time()
        expires = now + (expire_hours or self._expire_hours) * 3600

        payload = {
            "jti": token_id,
            "sub": entity_id,
            "iat": int(now),
            "exp": int(expires),
            "perm": permissions or ["read", "write"],
            "meta": metadata or {},
        }

        # Sign the payload
        payload_json = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self._secret,
            payload_json.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # Encode token
        token_data = {
            "payload": payload,
            "signature": signature,
        }
        token = base64.urlsafe_b64encode(
            json.dumps(token_data).encode("utf-8")
        ).decode("utf-8")

        return token

    def authenticate(
        self,
        credentials: Dict[str, Any]
    ) -> Tuple[AuthResult, Optional[AuthenticatedEntity]]:
        """
        Authenticate using a token.

        Args:
            credentials: Dict with "token" key

        Returns:
            Tuple of (result, authenticated_entity)
        """
        token = credentials.get("token")
        if not token:
            return AuthResult.MISSING_CREDENTIALS, None

        return self.validate_token(token)

    def validate_token(
        self,
        token: str
    ) -> Tuple[AuthResult, Optional[AuthenticatedEntity]]:
        """
        Validate a token.

        Args:
            token: Token string

        Returns:
            Tuple of (result, authenticated_entity)
        """
        try:
            # Decode token
            token_data = json.loads(
                base64.urlsafe_b64decode(token.encode("utf-8"))
            )
            payload = token_data["payload"]
            signature = token_data["signature"]

            # Verify signature
            payload_json = json.dumps(payload, sort_keys=True)
            expected_sig = hmac.new(
                self._secret,
                payload_json.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                return AuthResult.INVALID_TOKEN, None

            # Check if revoked
            if payload["jti"] in self._revoked_tokens:
                return AuthResult.REVOKED_TOKEN, None

            # Check expiration
            if time.time() > payload["exp"]:
                return AuthResult.EXPIRED_TOKEN, None

            # Create authenticated entity
            entity = AuthenticatedEntity(
                entity_id=payload["sub"],
                auth_method="token",
                permissions=payload.get("perm", []),
                metadata=payload.get("meta", {}),
                expires_at=payload["exp"],
            )

            return AuthResult.SUCCESS, entity

        except Exception as e:
            logger.debug(f"Token validation failed: {e}")
            return AuthResult.INVALID_TOKEN, None

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.

        Args:
            token: Token to revoke

        Returns:
            True if revocation successful
        """
        try:
            token_data = json.loads(
                base64.urlsafe_b64decode(token.encode("utf-8"))
            )
            token_id = token_data["payload"]["jti"]
            self._revoked_tokens.add(token_id)
            return True
        except Exception:
            return False

    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh a token (generate new token with same claims).

        Args:
            token: Existing token

        Returns:
            New token or None if original is invalid
        """
        result, entity = self.validate_token(token)
        if result != AuthResult.SUCCESS or not entity:
            return None

        return self.generate_token(
            entity_id=entity.entity_id,
            permissions=entity.permissions,
            metadata=entity.metadata,
        )


@dataclass
class APIKey:
    """API key definition."""
    name: str
    key: str
    permissions: List[str] = field(default_factory=lambda: ["read"])
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    enabled: bool = True


class APIKeyAuth(Authenticator):
    """
    API key authentication.

    Simple key-based authentication for applications.
    """

    def __init__(self):
        """Initialize API key authenticator."""
        self._keys: Dict[str, APIKey] = {}

    def create_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
        key: Optional[str] = None
    ) -> APIKey:
        """
        Create a new API key.

        Args:
            name: Key name/identifier
            permissions: List of permissions
            key: Custom key (generated if not provided)

        Returns:
            Created APIKey
        """
        if key is None:
            key = f"rgn_{secrets.token_hex(24)}"

        api_key = APIKey(
            name=name,
            key=key,
            permissions=permissions or ["read"],
        )
        self._keys[key] = api_key
        return api_key

    def add_key(self, api_key: APIKey) -> None:
        """
        Add an existing API key.

        Args:
            api_key: APIKey to add
        """
        self._keys[api_key.key] = api_key

    def remove_key(self, key: str) -> bool:
        """
        Remove an API key.

        Args:
            key: Key string to remove

        Returns:
            True if key was removed
        """
        if key in self._keys:
            del self._keys[key]
            return True
        return False

    def authenticate(
        self,
        credentials: Dict[str, Any]
    ) -> Tuple[AuthResult, Optional[AuthenticatedEntity]]:
        """
        Authenticate using an API key.

        Args:
            credentials: Dict with "api_key" key

        Returns:
            Tuple of (result, authenticated_entity)
        """
        key = credentials.get("api_key")
        if not key:
            return AuthResult.MISSING_CREDENTIALS, None

        return self.validate_key(key)

    def validate_key(
        self,
        key: str
    ) -> Tuple[AuthResult, Optional[AuthenticatedEntity]]:
        """
        Validate an API key.

        Args:
            key: API key string

        Returns:
            Tuple of (result, authenticated_entity)
        """
        api_key = self._keys.get(key)
        if not api_key:
            return AuthResult.INVALID_KEY, None

        if not api_key.enabled:
            return AuthResult.REVOKED_TOKEN, None

        # Update last used
        api_key.last_used = time.time()

        # Create authenticated entity
        entity = AuthenticatedEntity(
            entity_id=api_key.name,
            auth_method="api_key",
            permissions=api_key.permissions,
            metadata={"key_name": api_key.name},
        )

        return AuthResult.SUCCESS, entity

    def list_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys (without exposing actual keys).

        Returns:
            List of key information
        """
        return [
            {
                "name": k.name,
                "key_prefix": k.key[:10] + "...",
                "permissions": k.permissions,
                "created_at": k.created_at,
                "last_used": k.last_used,
                "enabled": k.enabled,
            }
            for k in self._keys.values()
        ]


class AuthenticationManager:
    """
    Unified authentication manager.

    Supports multiple authentication methods and provides
    a single interface for authentication.
    """

    def __init__(
        self,
        token_secret: Optional[str] = None,
        token_expire_hours: int = 24
    ):
        """
        Initialize authentication manager.

        Args:
            token_secret: Secret for token signing
            token_expire_hours: Default token expiration
        """
        self.token_auth = TokenAuth(token_secret, token_expire_hours)
        self.api_key_auth = APIKeyAuth()
        self._ca_cert: Optional[str] = None
        self._ca_key: Optional[str] = None
        self.revoked_certificates: Set[int] = set()

    def authenticate(
        self,
        credentials: Dict[str, Any]
    ) -> Tuple[AuthResult, Optional[AuthenticatedEntity]]:
        """
        Authenticate using any available method.

        Args:
            credentials: Authentication credentials

        Returns:
            Tuple of (result, authenticated_entity)
        """
        # Try token auth
        if "token" in credentials:
            return self.token_auth.authenticate(credentials)

        # Try API key auth
        if "api_key" in credentials:
            return self.api_key_auth.authenticate(credentials)

        return AuthResult.MISSING_CREDENTIALS, None

    def generate_token(
        self,
        entity_id: str,
        permissions: Optional[List[str]] = None,
        expire_hours: Optional[int] = None
    ) -> str:
        """Generate an authentication token."""
        return self.token_auth.generate_token(
            entity_id, permissions, expire_hours
        )

    def create_api_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None
    ) -> APIKey:
        """Create a new API key."""
        return self.api_key_auth.create_key(name, permissions)

    async def setup_certificate_authority(self) -> Tuple[str, str]:
        """
        Set up certificate authority for certificate-based auth.

        Returns:
            Tuple of (ca_cert_pem, ca_key_pem)
        """
        if not HAS_CRYPTO:
            raise ImportError("pycryptodome required for certificate auth")

        # Generate CA key pair
        ca_key = ECC.generate(curve="P-384")

        # Create self-signed certificate
        ca_cert = {
            "version": 1,
            "serial_number": 1,
            "issuer": "RegenNexus CA",
            "subject": "RegenNexus CA",
            "not_before": int(time.time()),
            "not_after": int(time.time() + 365 * 24 * 3600),
            "public_key": ca_key.public_key().export_key(format="PEM"),
            "extensions": {
                "ca": True,
                "key_usage": ["cert_sign", "crl_sign"],
            },
        }

        # Sign certificate
        cert_data = json.dumps(ca_cert, sort_keys=True).encode("utf-8")
        h = SHA384.new(cert_data)
        signer = DSS.new(ca_key, "fips-186-3")
        signature = signer.sign(h)

        ca_cert["signature"] = signature.hex()

        # Convert to PEM
        ca_cert_pem = (
            "-----BEGIN CERTIFICATE-----\n" +
            base64.b64encode(json.dumps(ca_cert).encode()).decode() +
            "\n-----END CERTIFICATE-----"
        )
        ca_key_pem = ca_key.export_key(format="PEM")

        self._ca_cert = ca_cert_pem
        self._ca_key = ca_key_pem

        return ca_cert_pem, ca_key_pem

    async def issue_entity_certificate(
        self,
        entity_id: str,
        public_key: bytes,
        validity_days: int = 30
    ) -> str:
        """
        Issue a certificate for an entity.

        Args:
            entity_id: Entity identifier
            public_key: Entity's public key
            validity_days: Certificate validity period

        Returns:
            Certificate in PEM format
        """
        if not self._ca_key:
            raise ValueError("CA not initialized")

        ca_key = ECC.import_key(self._ca_key)

        entity_cert = {
            "version": 1,
            "serial_number": int(time.time() * 1000),
            "issuer": "RegenNexus CA",
            "subject": f"entity:{entity_id}",
            "not_before": int(time.time()),
            "not_after": int(time.time() + validity_days * 24 * 3600),
            "public_key": base64.b64encode(public_key).decode(),
            "entity_id": entity_id,
        }

        # Sign certificate
        cert_data = json.dumps(entity_cert, sort_keys=True).encode()
        h = SHA384.new(cert_data)
        signer = DSS.new(ca_key, "fips-186-3")
        signature = signer.sign(h)

        entity_cert["signature"] = signature.hex()

        # Convert to PEM
        cert_pem = (
            "-----BEGIN CERTIFICATE-----\n" +
            base64.b64encode(json.dumps(entity_cert).encode()).decode() +
            "\n-----END CERTIFICATE-----"
        )

        return cert_pem

    async def verify_certificate(self, cert_pem: str) -> bool:
        """
        Verify an entity certificate.

        Args:
            cert_pem: Certificate in PEM format

        Returns:
            True if certificate is valid
        """
        if not self._ca_cert:
            return False

        try:
            # Parse certificate
            cert_b64 = cert_pem.split("-----BEGIN CERTIFICATE-----\n")[1]
            cert_b64 = cert_b64.split("\n-----END CERTIFICATE-----")[0]
            cert = json.loads(base64.b64decode(cert_b64))

            # Check revocation
            if cert["serial_number"] in self.revoked_certificates:
                return False

            # Check validity
            now = time.time()
            if now < cert["not_before"] or now > cert["not_after"]:
                return False

            # Verify signature
            signature = bytes.fromhex(cert["signature"])
            cert_copy = {k: v for k, v in cert.items() if k != "signature"}
            cert_data = json.dumps(cert_copy, sort_keys=True).encode()

            # Get CA public key
            ca_b64 = self._ca_cert.split("-----BEGIN CERTIFICATE-----\n")[1]
            ca_b64 = ca_b64.split("\n-----END CERTIFICATE-----")[0]
            ca_cert = json.loads(base64.b64decode(ca_b64))
            ca_public_key = ECC.import_key(ca_cert["public_key"])

            h = SHA384.new(cert_data)
            verifier = DSS.new(ca_public_key, "fips-186-3")
            verifier.verify(h, signature)

            return True

        except Exception as e:
            logger.debug(f"Certificate verification failed: {e}")
            return False

    async def revoke_certificate(self, serial_number: int) -> None:
        """Revoke a certificate by serial number."""
        self.revoked_certificates.add(serial_number)
