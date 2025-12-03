# regennexus/registry/cli.py
from .registry import UAP_Registry
import asyncio

def main():
    registry = UAP_Registry()
    asyncio.run(registry.start())