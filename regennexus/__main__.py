"""
RegenNexus UAP - Main entry point

Allows running RegenNexus as a module:
    python -m regennexus mesh start
    python -m regennexus mesh status
    python -m regennexus doctor

Copyright (c) 2024-2025 ReGen Designs LLC
"""

from regennexus.cli import main

if __name__ == "__main__":
    main()
