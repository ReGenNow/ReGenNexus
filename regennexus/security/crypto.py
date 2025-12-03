"""
RegenNexus UAP - Cryptography Module

Provides encryption functions using pycryptodome.
Supports AES-128, AES-256-GCM, and ECDH-384 key exchange.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import os
import base64
import json
import hashlib
import hmac
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import pycryptodome
try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import scrypt, HKDF
    from Crypto.Hash import SHA256, SHA384, HMAC as CryptoHMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("pycryptodome not installed. Install with: pip install pycryptodome")

# Try to import cryptography for ECDH (optional)
try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF as CryptoHKDF
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


@dataclass
class EncryptedData:
    """Container for encrypted data."""
    ciphertext: bytes
    nonce: bytes
    tag: Optional[bytes] = None

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary with base64 encoding."""
        result = {
            "ciphertext": base64.b64encode(self.ciphertext).decode("utf-8"),
            "nonce": base64.b64encode(self.nonce).decode("utf-8"),
        }
        if self.tag:
            result["tag"] = base64.b64encode(self.tag).decode("utf-8")
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EncryptedData":
        """Deserialize from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            tag=base64.b64decode(data["tag"]) if "tag" in data else None,
        )


def generate_key(length: int = 32) -> bytes:
    """
    Generate a random encryption key.

    Args:
        length: Key length in bytes (16 for AES-128, 32 for AES-256)

    Returns:
        Random key bytes
    """
    if HAS_CRYPTO:
        return get_random_bytes(length)
    return os.urandom(length)


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Hash a password using scrypt.

    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = os.urandom(16)

    if HAS_CRYPTO:
        key = scrypt(password.encode(), salt, 32, N=2**14, r=8, p=1)
    else:
        # Fallback to hashlib
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)

    return key, salt


def verify_password(password: str, hash_bytes: bytes, salt: bytes) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Password to verify
        hash_bytes: Stored hash
        salt: Salt used for hashing

    Returns:
        True if password matches
    """
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hash_bytes)


class Encryptor(ABC):
    """Abstract base class for encryptors."""

    @abstractmethod
    def encrypt(self, plaintext: Union[str, bytes]) -> EncryptedData:
        """Encrypt data."""
        pass

    @abstractmethod
    def decrypt(self, encrypted: EncryptedData) -> bytes:
        """Decrypt data."""
        pass


class AESEncryptor(Encryptor):
    """
    AES encryptor with GCM mode.

    Supports AES-128 and AES-256 with authenticated encryption.
    """

    def __init__(self, key: bytes):
        """
        Initialize AES encryptor.

        Args:
            key: Encryption key (16 bytes for AES-128, 32 bytes for AES-256)
        """
        if not HAS_CRYPTO:
            raise ImportError("pycryptodome required for AES encryption")

        if len(key) not in (16, 32):
            raise ValueError("Key must be 16 or 32 bytes")

        self._key = key

    def encrypt(self, plaintext: Union[str, bytes]) -> EncryptedData:
        """
        Encrypt data using AES-GCM.

        Args:
            plaintext: Data to encrypt

        Returns:
            EncryptedData containing ciphertext, nonce, and tag
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        # Generate random nonce (96 bits recommended for GCM)
        nonce = get_random_bytes(12)

        # Create cipher and encrypt
        cipher = AES.new(self._key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

        return EncryptedData(ciphertext=ciphertext, nonce=nonce, tag=tag)

    def decrypt(self, encrypted: EncryptedData) -> bytes:
        """
        Decrypt data using AES-GCM.

        Args:
            encrypted: EncryptedData to decrypt

        Returns:
            Decrypted plaintext

        Raises:
            ValueError: If authentication fails
        """
        cipher = AES.new(self._key, AES.MODE_GCM, nonce=encrypted.nonce)

        try:
            plaintext = cipher.decrypt_and_verify(
                encrypted.ciphertext,
                encrypted.tag
            )
            return plaintext
        except ValueError as e:
            raise ValueError("Decryption failed: authentication error") from e

    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Encrypt a dictionary.

        Args:
            data: Dictionary to encrypt

        Returns:
            Dictionary with encrypted data
        """
        json_data = json.dumps(data)
        encrypted = self.encrypt(json_data)
        return encrypted.to_dict()

    def decrypt_dict(self, encrypted_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Decrypt a dictionary.

        Args:
            encrypted_data: Encrypted dictionary

        Returns:
            Original dictionary
        """
        encrypted = EncryptedData.from_dict(encrypted_data)
        plaintext = self.decrypt(encrypted)
        return json.loads(plaintext.decode("utf-8"))


class ECDHKeyExchange:
    """
    ECDH key exchange using P-384 curve.

    Allows two parties to establish a shared secret key
    over an insecure channel.
    """

    def __init__(self):
        """Initialize ECDH key exchange."""
        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "cryptography library required for ECDH. "
                "Install with: pip install cryptography"
            )

        self._private_keys: Dict[str, Any] = {}
        self._public_keys: Dict[str, Any] = {}
        self._shared_keys: Dict[Tuple[str, str], bytes] = {}

    def generate_keypair(self, entity_id: str) -> Tuple[bytes, bytes]:
        """
        Generate ECDH keypair for an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        # Generate private key using P-384 curve
        private_key = ec.generate_private_key(
            ec.SECP384R1(),
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Serialize to PEM
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Store keys
        self._private_keys[entity_id] = private_key
        self._public_keys[entity_id] = public_key

        return private_pem, public_pem

    def import_public_key(self, entity_id: str, public_key_pem: bytes) -> bool:
        """
        Import a public key for an entity.

        Args:
            entity_id: Entity identifier
            public_key_pem: Public key in PEM format

        Returns:
            True if import successful
        """
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem,
                backend=default_backend()
            )
            self._public_keys[entity_id] = public_key
            return True
        except Exception as e:
            logger.error(f"Failed to import public key: {e}")
            return False

    def derive_shared_key(
        self,
        local_id: str,
        remote_id: str,
        key_length: int = 32
    ) -> Optional[bytes]:
        """
        Derive a shared key between two entities.

        Args:
            local_id: Local entity ID (must have private key)
            remote_id: Remote entity ID (must have public key)
            key_length: Derived key length (default 32 for AES-256)

        Returns:
            Derived shared key or None if error
        """
        key_pair = (local_id, remote_id)

        # Check if already derived
        if key_pair in self._shared_keys:
            return self._shared_keys[key_pair]

        # Check keys exist
        if local_id not in self._private_keys:
            logger.error(f"No private key for {local_id}")
            return None
        if remote_id not in self._public_keys:
            logger.error(f"No public key for {remote_id}")
            return None

        try:
            # Perform ECDH
            private_key = self._private_keys[local_id]
            public_key = self._public_keys[remote_id]
            shared_secret = private_key.exchange(ec.ECDH(), public_key)

            # Derive key using HKDF
            derived_key = CryptoHKDF(
                algorithm=hashes.SHA384(),
                length=key_length,
                salt=None,
                info=b"RegenNexus-ECDH-Key"
            ).derive(shared_secret)

            # Cache the derived key
            self._shared_keys[key_pair] = derived_key

            return derived_key

        except Exception as e:
            logger.error(f"Failed to derive shared key: {e}")
            return None

    def get_encryptor(self, local_id: str, remote_id: str) -> Optional[AESEncryptor]:
        """
        Get an AES encryptor using derived shared key.

        Args:
            local_id: Local entity ID
            remote_id: Remote entity ID

        Returns:
            AESEncryptor or None if key derivation fails
        """
        shared_key = self.derive_shared_key(local_id, remote_id)
        if shared_key:
            return AESEncryptor(shared_key)
        return None


# Convenience class for backward compatibility
class CryptoManager:
    """
    Crypto manager for message encryption.

    Provides a high-level interface for encrypting/decrypting
    messages between entities.
    """

    def __init__(self):
        """Initialize crypto manager."""
        self._ecdh: Optional[ECDHKeyExchange] = None
        if HAS_CRYPTOGRAPHY:
            self._ecdh = ECDHKeyExchange()

    async def generate_keypair(self, entity_id: str) -> Tuple[bytes, bytes]:
        """Generate keypair for entity."""
        if not self._ecdh:
            raise ImportError("ECDH not available")
        return self._ecdh.generate_keypair(entity_id)

    async def import_public_key(self, entity_id: str, public_key_pem: bytes) -> bool:
        """Import public key for entity."""
        if not self._ecdh:
            return False
        return self._ecdh.import_public_key(entity_id, public_key_pem)

    async def encrypt_message(
        self,
        sender_id: str,
        recipient_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encrypt a message for transmission."""
        if not self._ecdh:
            return message

        encryptor = self._ecdh.get_encryptor(sender_id, recipient_id)
        if not encryptor:
            return message

        try:
            encrypted = encryptor.encrypt_dict(message)
            return {
                "sender": sender_id,
                "recipient": recipient_id,
                "encrypted": True,
                **encrypted,
            }
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return message

    async def decrypt_message(
        self,
        recipient_id: str,
        encrypted_message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Decrypt an encrypted message."""
        if not encrypted_message.get("encrypted"):
            return encrypted_message

        if not self._ecdh:
            return None

        sender_id = encrypted_message.get("sender")
        if not sender_id:
            return None

        encryptor = self._ecdh.get_encryptor(recipient_id, sender_id)
        if not encryptor:
            return None

        try:
            return encryptor.decrypt_dict(encrypted_message)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
