"""
Phi-Mamba Configuration

Minimal config for integer-only SSM models.
"""

from typing import Optional


class PhiMambaConfig:
    """
    Configuration for Phi-Mamba integer-only language model.

    All operations use Zeckendorf encoding + CORDIC (shift-add only).
    No floating point in forward pass.

    Model Variants:
    - tiny:  32 shells, 2 layers, ~1MB   (edge devices)
    - small: 64 shells, 4 layers, ~5MB   (mobile)
    - base:  128 shells, 6 layers, ~20MB (desktop)
    """

    model_type = "phi-mamba"

    def __init__(
        self,
        vocab_size: int = 32000,
        max_shells: int = 64,
        num_layers: int = 4,
        scale_bits: int = 16,
        max_position: int = 2048,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_shells = max_shells
        self.num_layers = num_layers
        self.scale_bits = scale_bits
        self.max_position = max_position
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Derived parameters
        self.hidden_size = max_shells  # State dimension = Fibonacci shells
        self.energy_threshold = 1 << (scale_bits - 8)  # Natural termination

    @classmethod
    def tiny(cls) -> "PhiMambaConfig":
        """Tiny model for edge devices (~1MB)"""
        return cls(
            vocab_size=8000,
            max_shells=32,
            num_layers=2,
            scale_bits=16,
            max_position=512,
        )

    @classmethod
    def small(cls) -> "PhiMambaConfig":
        """Small model for mobile (~5MB)"""
        return cls(
            vocab_size=16000,
            max_shells=64,
            num_layers=4,
            scale_bits=16,
            max_position=1024,
        )

    @classmethod
    def base(cls) -> "PhiMambaConfig":
        """Base model for desktop (~20MB)"""
        return cls(
            vocab_size=32000,
            max_shells=128,
            num_layers=6,
            scale_bits=32,
            max_position=2048,
        )

    def to_dict(self) -> dict:
        """Serialize config to dict"""
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "max_shells": self.max_shells,
            "num_layers": self.num_layers,
            "scale_bits": self.scale_bits,
            "max_position": self.max_position,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_size": self.hidden_size,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PhiMambaConfig":
        """Load config from dict"""
        return cls(**{k: v for k, v in config_dict.items() if k != "model_type"})

    def save_pretrained(self, save_directory: str):
        """Save config to directory"""
        import json
        import os
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "PhiMambaConfig":
        """Load config from directory or HF hub"""
        import json
        import os
        config_path = os.path.join(pretrained_path, "config.json")
        with open(config_path, "r") as f:
            return cls.from_dict(json.load(f))
