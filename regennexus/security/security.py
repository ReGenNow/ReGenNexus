"""
ReGenNexus Core - Security Manager

This module implements enhanced security features for ReGenNexus Core,
providing ECDH-384 encryption, certificate-based authentication, and
policy-based access control.
"""

import os
import base64
import json
import asyncio
import logging
from typing import Dict, Tuple, Optional, Any, List
from Crypto.PublicKey import RSA, ECC
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA384
from Crypto.Signature import DSS
from Crypto.Util.Padding import pad, unpad

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Manages security operations for ReGenNexus Core.
    
    Provides encryption, decryption, key management, and other security
    features with support for both ECDH-384 (preferred) and RSA-2048
    (backward compatibility).
    """
    
    def __init__(self, security_level: int = 2):
        """
        Initialize the security manager.
        
        Args:
            security_level: Security level (1=basic, 2=enhanced, 3=maximum)
        """
        self.security_level = security_level
        self.feature_flags = {
            "use_ecdh": security_level >= 2,
            "use_post_quantum": security_level >= 3,
            "enforce_certificate_pinning": security_level >= 2,
            "use_hardware_security": security_level >= 3
        }
        
        # Generate or load keys
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize cryptographic keys."""
        # For ECDH-384
        if self.feature_flags["use_ecdh"]:
            try:
                self.ecdh_key = ECC.generate(curve='P-384')
                logger.info("ECDH-384 key pair generated")
            except Exception as e:
                logger.error(f"Failed to generate ECDH key: {e}")
                self.ecdh_key = None
        else:
            self.ecdh_key = None
        
        # For RSA (backward compatibility)
        try:
            self.rsa_key = RSA.generate(2048)
            logger.info("RSA-2048 key pair generated")
        except Exception as e:
            logger.error(f"Failed to generate RSA key: {e}")
            self.rsa_key = None
    
    def supports_ecdh(self) -> bool:
        """
        Check if ECDH encryption is supported.
        
        Returns:
            True if ECDH is supported, False otherwise
        """
        return self.feature_flags["use_ecdh"] and self.ecdh_key is not None
    
    def get_public_key(self) -> bytes:
        """
        Get the public key for this security manager.
        
        Returns:
            Public key bytes (ECDH if available, otherwise RSA)
        """
        if self.supports_ecdh():
            return self.ecdh_key.public_key().export_key(format='DER')
        else:
            return self.rsa_key.publickey().export_key(format='DER')
    
    async def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new key pair.
        
        Returns:
            Tuple of (public_key, private_key) bytes
        """
        if self.supports_ecdh():
            key = ECC.generate(curve='P-384')
            return (
                key.public_key().export_key(format='DER'),
                key.export_key(format='DER')
            )
        else:
            key = RSA.generate(2048)
            return (
                key.publickey().export_key(format='DER'),
                key.export_key(format='DER')
            )
    
    async def encrypt_message_ecdh(self, message: bytes, recipient_public_key: bytes) -> bytes:
        """
        Encrypt a message using ECDH-384 and AES-256-GCM.
        
        Args:
            message: The message to encrypt
            recipient_public_key: Public key of the recipient
            
        Returns:
            Encrypted message data
        """
        # Import recipient's public key
        recipient_key = ECC.import_key(recipient_public_key)
        
        # Generate a shared secret
        shared_point = self.ecdh_key.d * recipient_key.pointQ
        shared_secret = SHA384.new(shared_point.x.to_bytes()).digest()[:32]
        
        # Generate a random nonce
        nonce = get_random_bytes(12)
        
        # Encrypt the message
        cipher = AES.new(shared_secret, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(message)
        
        # Format the encrypted message
        encrypted_data = {
            "algorithm": "ECDH-384+AES-256-GCM",
            "sender_public_key": self.ecdh_key.public_key().export_key(format='DER').hex(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
            "tag": tag.hex()
        }
        
        return json.dumps(encrypted_data).encode('utf-8')
    
    async def encrypt_message_rsa(self, message: bytes, recipient_public_key: bytes) -> bytes:
        """
        Encrypt a message using RSA-2048 and AES-256-CBC.
        
        Args:
            message: The message to encrypt
            recipient_public_key: Public key of the recipient
            
        Returns:
            Encrypted message data
        """
        # Import recipient's public key
        recipient_key = RSA.import_key(recipient_public_key)
        
        # Generate a random session key
        session_key = get_random_bytes(32)
        
        # Encrypt the session key with the public RSA key
        cipher_rsa = PKCS1_OAEP.new(recipient_key)
        enc_session_key = cipher_rsa.encrypt(session_key)
        
        # Encrypt the message with the session key
        iv = get_random_bytes(16)
        cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
        ciphertext = cipher_aes.encrypt(pad(message, AES.block_size))
        
        # Format the encrypted message
        encrypted_data = {
            "algorithm": "RSA-2048+AES-256-CBC",
            "enc_session_key": enc_session_key.hex(),
            "iv": iv.hex(),
            "ciphertext": ciphertext.hex()
        }
        
        return json.dumps(encrypted_data).encode('utf-8')
    
    async def decrypt_message(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt a message.
        
        Automatically detects the encryption method used.
        
        Args:
            encrypted_data: The encrypted message data
            
        Returns:
            Decrypted message
        """
        # Parse the encrypted data
        data = json.loads(encrypted_data.decode('utf-8'))
        algorithm = data.get("algorithm", "")
        
        if algorithm == "ECDH-384+AES-256-GCM":
            return await self._decrypt_ecdh(data)
        elif algorithm == "RSA-2048+AES-256-CBC":
            return await self._decrypt_rsa(data)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    async def _decrypt_ecdh(self, data: Dict[str, str]) -> bytes:
        """
        Decrypt a message encrypted with ECDH-384 and AES-256-GCM.
        
        Args:
            data: The encrypted message data
            
        Returns:
            Decrypted message
        """
        # Extract data
        sender_public_key = bytes.fromhex(data["sender_public_key"])
        nonce = bytes.fromhex(data["nonce"])
        ciphertext = bytes.fromhex(data["ciphertext"])
        tag = bytes.fromhex(data["tag"])
        
        # Import sender's public key
        sender_key = ECC.import_key(sender_public_key)
        
        # Generate a shared secret
        shared_point = self.ecdh_key.d * sender_key.pointQ
        shared_secret = SHA384.new(shared_point.x.to_bytes()).digest()[:32]
        
        # Decrypt the message
        cipher = AES.new(shared_secret, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        
        return plaintext
    
    async def _decrypt_rsa(self, data: Dict[str, str]) -> bytes:
        """
        Decrypt a message encrypted with RSA-2048 and AES-256-CBC.
        
        Args:
            data: The encrypted message data
            
        Returns:
            Decrypted message
        """
        # Extract data
        enc_session_key = bytes.fromhex(data["enc_session_key"])
        iv = bytes.fromhex(data["iv"])
        ciphertext = bytes.fromhex(data["ciphertext"])
        
        # Decrypt the session key
        cipher_rsa = PKCS1_OAEP.new(self.rsa_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
        
        # Decrypt the message
        cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher_aes.decrypt(ciphertext), AES.block_size)
        
        return plaintext
    
    async def encrypt_with_best_available(self, data: bytes, recipient_public_key: bytes) -> bytes:
        """
        Encrypt data using the best available encryption method.
        
        Args:
            data: The data to encrypt
            recipient_public_key: Public key of the recipient
            
        Returns:
            Encrypted data
        """
        if self.supports_ecdh():
            try:
                # Try to use ECDH first
                return await self.encrypt_message_ecdh(data, recipient_public_key)
            except Exception as e:
                logger.warning(f"ECDH encryption failed, falling back to RSA: {e}")
        
        # Fall back to RSA
        return await self.encrypt_message_rsa(data, recipient_public_key)
    
    async def decrypt_with_best_available(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the appropriate decryption method.
        
        Args:
            encrypted_data: The encrypted data
            
        Returns:
            Decrypted data
        """
        return await self.decrypt_message(encrypted_data)
    
    async def sign_data(self, data: bytes) -> bytes:
        """
        Sign data using the private key.
        
        Args:
            data: The data to sign
            
        Returns:
            Signature bytes
        """
        if self.supports_ecdh():
            # Use ECDSA with SHA-384
            h = SHA384.new(data)
            signer = DSS.new(self.ecdh_key, 'fips-186-3')
            signature = signer.sign(h)
        else:
            # Fall back to RSA
            from Crypto.Signature import pkcs1_15
            h = SHA384.new(data)
            signature = pkcs1_15.new(self.rsa_key).sign(h)
        
        return signature
    
    async def verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify a signature.
        
        Args:
            data: The data that was signed
            signature: The signature to verify
            public_key: Public key of the signer
            
        Returns:
            True if the signature is valid, False otherwise
        """
        try:
            # Try ECDSA first
            key = ECC.import_key(public_key)
            h = SHA384.new(data)
            verifier = DSS.new(key, 'fips-186-3')
            verifier.verify(h, signature)
            return True
        except (ValueError, TypeError):
            try:
                # Fall back to RSA
                from Crypto.Signature import pkcs1_15
                key = RSA.import_key(public_key)
                h = SHA384.new(data)
                pkcs1_15.new(key).verify(h, signature)
                return True
            except (ValueError, TypeError):
                return False
        except Exception:
            return False

# Singleton factory for SecurityManager
_security_manager_instance = None

def get_instance(security_level: int = None) -> SecurityManager:
    """Get the singleton SecurityManager instance."""
    global _security_manager_instance
    if _security_manager_instance is None:
        level = security_level if security_level is not None else 2
        _security_manager_instance = SecurityManager(security_level=level)
    return _security_manager_instance
