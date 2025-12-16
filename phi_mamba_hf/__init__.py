"""
Phi-Mamba HuggingFace Integration

Integer-only language models using Zeckendorf-CORDIC arithmetic.
Minimal compute: all operations are shift-add only.
"""

from .configuration_phimamba import PhiMambaConfig
from .modeling_phimamba import (
    PhiMambaModel,
    PhiMambaForCausalLM,
    PhiMambaPreTrainedModel,
)
from .tokenization_phimamba import PhiMambaTokenizer

__version__ = "0.1.0"
__all__ = [
    "PhiMambaConfig",
    "PhiMambaModel",
    "PhiMambaForCausalLM",
    "PhiMambaPreTrainedModel",
    "PhiMambaTokenizer",
]
