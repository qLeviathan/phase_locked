"""
Φ-Mamba CLI Entry Point

Allows running phi-mamba as a module:
    python -m phi_mamba [command] [args]

For help:
    python -m phi_mamba --help
"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
