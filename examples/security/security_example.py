"""
ReGenNexus UAP - Security Example

This example demonstrates the security features of ReGenNexus UAP:
- ECDH-384 key exchange
- AES-256-GCM encryption
- Digital signatures
- Secure message exchange between entities

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import asyncio
import json
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ReGenNexus security module
from regennexus.security.security import SecurityManager


async def demonstrate_key_generation() -> Tuple[SecurityManager, SecurityManager]:
    """Demonstrate key generation for two entities."""
    logger.info("=== Key Generation ===")
    
    # Create security managers for Alice and Bob
    alice = SecurityManager(security_level=2)  # Enhanced security with ECDH
    bob = SecurityManager(security_level=2)
    
    # Get public keys
    alice_pub = alice.get_public_key()
    bob_pub = bob.get_public_key()
    
    logger.info(f"Alice's public key ({len(alice_pub)} bytes): {alice_pub[:30].hex()}...")
    logger.info(f"Bob's public key ({len(bob_pub)} bytes): {bob_pub[:30].hex()}...")
    logger.info(f"ECDH supported: {alice.supports_ecdh()}")
    logger.info(f"Security level: {alice.security_level}")
    
    return alice, bob


async def demonstrate_encryption(alice: SecurityManager, bob: SecurityManager) -> bool:
    """Demonstrate encrypted message exchange."""
    logger.info("\n=== Encrypted Message Exchange ===")
    
    # Get Bob's public key for encryption
    bob_pub = bob.get_public_key()
    
    # Original message
    original_message = b"Hello Bob! This is a secret message from Alice. The password is: hunter2"
    logger.info(f"Original message: {original_message.decode()}")
    
    # Alice encrypts using Bob's public key
    encrypted = await alice.encrypt_with_best_available(original_message, bob_pub)
    enc_data = json.loads(encrypted)
    logger.info(f"Encryption algorithm: {enc_data['algorithm']}")
    logger.info(f"Encrypted data ({len(encrypted)} bytes)")
    
    # Bob decrypts
    decrypted = await bob.decrypt_message(encrypted)
    logger.info(f"Decrypted message: {decrypted.decode()}")
    
    # Verify
    success = original_message == decrypted
    logger.info(f"Decryption successful: {success}")
    
    return success


async def demonstrate_signatures(alice: SecurityManager, bob: SecurityManager) -> bool:
    """Demonstrate digital signature creation and verification."""
    logger.info("\n=== Digital Signatures ===")
    
    # Get Alice's public key
    alice_pub = alice.get_public_key()
    
    # Document to sign
    document = b"I, Alice, hereby authorize the transfer of 1000 tokens to Bob."
    logger.info(f"Document: {document.decode()}")
    
    # Alice signs the document
    signature = await alice.sign_data(document)
    logger.info(f"Signature ({len(signature)} bytes): {signature[:30].hex()}...")
    
    # Bob verifies Alice's signature
    is_valid = await bob.verify_signature(document, signature, alice_pub)
    logger.info(f"Signature valid: {is_valid}")
    
    # Try to verify tampered document
    tampered = b"I, Alice, hereby authorize the transfer of 9999 tokens to Bob."
    is_tampered_valid = await bob.verify_signature(tampered, signature, alice_pub)
    logger.info(f"Tampered document valid: {is_tampered_valid}")
    
    return is_valid and not is_tampered_valid


async def demonstrate_security_levels():
    """Demonstrate different security levels."""
    logger.info("\n=== Security Levels ===")
    
    for level in [1, 2, 3]:
        sm = SecurityManager(security_level=level)
        logger.info(f"Level {level}:")
        logger.info(f"  ECDH enabled: {sm.feature_flags['use_ecdh']}")
        logger.info(f"  Post-quantum ready: {sm.feature_flags['use_post_quantum']}")
        logger.info(f"  Certificate pinning: {sm.feature_flags['enforce_certificate_pinning']}")
        logger.info(f"  Hardware security: {sm.feature_flags['use_hardware_security']}")


async def main():
    """Main function to demonstrate security features."""
    logger.info("Starting ReGenNexus Security Example")
    logger.info("=" * 60)
    
    try:
        # Key generation
        alice, bob = await demonstrate_key_generation()
        
        # Encryption demo
        encryption_success = await demonstrate_encryption(alice, bob)
        
        # Signature demo
        signature_success = await demonstrate_signatures(alice, bob)
        
        # Security levels
        await demonstrate_security_levels()
        
        # Summary
        logger.info("\n=== Summary ===")
        logger.info(f"Encryption test: {'PASSED' if encryption_success else 'FAILED'}")
        logger.info(f"Signature test: {'PASSED' if signature_success else 'FAILED'}")
        
        if encryption_success and signature_success:
            logger.info("\nAll security tests passed!")
        else:
            logger.error("\nSome tests failed!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
