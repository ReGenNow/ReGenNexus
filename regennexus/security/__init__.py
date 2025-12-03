"""
RegenNexus UAP - Security Module

Provides encryption, authentication, and rate limiting.

Security features:
- Encryption: AES-128, AES-256-GCM, ECDH-384
- Authentication: Token-based, API keys, Certificates
- Rate Limiting: Request throttling with burst protection

All security features are optional and configurable.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

from regennexus.security.crypto import (
    Encryptor,
    AESEncryptor,
    ECDHKeyExchange,
    CryptoManager,
    EncryptedData,
    generate_key,
    hash_password,
    verify_password,
    HAS_CRYPTO,
)
from regennexus.security.auth import (
    Authenticator,
    TokenAuth,
    APIKeyAuth,
    APIKey,
    AuthResult,
    AuthenticatedEntity,
    AuthenticationManager,
)
from regennexus.security.rate_limit import (
    RateLimiter,
    AdaptiveRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimitState,
)

__all__ = [
    # Crypto
    "Encryptor",
    "AESEncryptor",
    "ECDHKeyExchange",
    "CryptoManager",
    "EncryptedData",
    "generate_key",
    "hash_password",
    "verify_password",
    "HAS_CRYPTO",
    # Auth
    "Authenticator",
    "TokenAuth",
    "APIKeyAuth",
    "APIKey",
    "AuthResult",
    "AuthenticatedEntity",
    "AuthenticationManager",
    # Rate limiting
    "RateLimiter",
    "AdaptiveRateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitState",
]
