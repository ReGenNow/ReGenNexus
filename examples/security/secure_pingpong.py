#!/usr/bin/env python3
"""
Security Showcase Example

This example demonstrates end-to-end encryption and signing using the
SecurityManager. It performs ECDH-384+AES-256-GCM encryption and RSA-2048+AES-256-CBC
fallback, and shows signature generation/verification.
"""
import asyncio
from regennexus.security.security import SecurityManager

async def secure_pingpong():
    # ECDH-based encryption/demo
    print("=== ECDH-384 + AES-256-GCM Demo ===")
    sm_a = SecurityManager(security_level=2)
    sm_b = SecurityManager(security_level=2)

    msg = b"ping from A"
    print(f"Agent A sends: {msg}")

    encrypted = await sm_a.encrypt_message_ecdh(msg, sm_b.get_public_key())
    print(f"Encrypted data: {encrypted}\n")

    decrypted = await sm_b.decrypt_message(encrypted)
    print(f"Agent B received: {decrypted}\n")

    signature = await sm_a.sign_data(msg)
    print(f"Signature (hex): {signature.hex()}")
    valid = await sm_b.verify_signature(msg, signature, sm_a.get_public_key())
    print(f"Signature valid: {valid}\n")

    # RSA fallback demo
    print("=== RSA-2048 + AES-256-CBC Fallback Demo ===")
    sm_r1 = SecurityManager(security_level=1)
    sm_r2 = SecurityManager(security_level=1)

    msg2 = b"hello RSA fallback"
    print(f"Agent R1 sends: {msg2}")
    encrypted_rsa = await sm_r1.encrypt_message_rsa(msg2, sm_r2.get_public_key())
    print(f"Encrypted RSA data: {encrypted_rsa}\n")

    decrypted_rsa = await sm_r2.decrypt_message(encrypted_rsa)
    print(f"Agent R2 received: {decrypted_rsa}\n")

    signature_rsa = await sm_r1.sign_data(msg2)
    print(f"RSA Signature (hex): {signature_rsa.hex()}")
    valid_rsa = await sm_r2.verify_signature(msg2, signature_rsa, sm_r1.get_public_key())
    print(f"RSA signature valid: {valid_rsa}")

if __name__ == '__main__':
    asyncio.run(secure_pingpong())