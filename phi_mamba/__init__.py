"""
Φ-Mamba: Phase-Locked Language Modeling with Golden Ratio Encoding

A revolutionary approach to language modeling that uses:
- Golden ratio (φ) as the fundamental primitive
- Retrocausal encoding for improved coherence
- Topological information storage
- Natural termination through energy decay
"""

from .core import PhiLanguageModel, PhiTokenizer
from .encoding import retrocausal_encode, zeckendorf_decomposition
from .generation import generate_with_phase_lock
from .utils import PHI, PSI, compute_berry_phase

__version__ = "0.1.0"
__author__ = "Marc Castillo"

__all__ = [
    "PhiLanguageModel",
    "PhiTokenizer", 
    "retrocausal_encode",
    "zeckendorf_decomposition",
    "generate_with_phase_lock",
    "PHI",
    "PSI",
    "compute_berry_phase"
]