#!/usr/bin/env python3
"""
Push Phi-Mamba models to HuggingFace Hub

Usage:
    python push_to_hub.py --model tiny --repo your-username/phi-mamba-tiny
    python push_to_hub.py --model small --repo your-username/phi-mamba-small
    python push_to_hub.py --model base --repo your-username/phi-mamba-base
"""

import argparse
import os
import json
import shutil
from pathlib import Path

from configuration_phimamba import PhiMambaConfig
from modeling_phimamba import PhiMambaForCausalLM
from tokenization_phimamba import PhiMambaTokenizer


MODEL_CARD_TEMPLATE = '''---
language:
- en
license: mit
tags:
- phi-mamba
- integer-only
- zeckendorf
- cordic
- ssm
- minimal-compute
library_name: transformers
pipeline_tag: text-generation
---

# Phi-Mamba {variant}

Integer-only language model using Zeckendorf-CORDIC arithmetic.

## Key Features

- **Zero floating-point operations** in forward pass
- **Shift-add only** arithmetic (CORDIC)
- **Fibonacci-based** token encoding (Zeckendorf)
- **Natural termination** via energy decay
- **Minimal compute** - runs on edge devices

## Model Details

| Property | Value |
|----------|-------|
| Parameters | ~{params} |
| Vocab Size | {vocab_size:,} |
| Max Shells | {max_shells} |
| Layers | {num_layers} |
| Scale Bits | {scale_bits} |
| Max Position | {max_position:,} |

## Usage

```python
from phi_mamba_hf import PhiMambaForCausalLM, PhiMambaTokenizer

# Load model
model = PhiMambaForCausalLM.from_pretrained("{repo_id}")
tokenizer = PhiMambaTokenizer.from_pretrained("{repo_id}")

# Generate
inputs = tokenizer("The quick brown", return_tensors="np")
outputs = model.generate(inputs["input_ids"], max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

## Architecture

This model uses the **Phase-Locked** architecture:

1. **Zeckendorf Encoding**: Tokens → Fibonacci index sets (sparse, unique)
2. **Cascade Operations**: Resolve F_i + F_{i+1} = F_{i+2} (bit ops only)
3. **CORDIC Coupling**: Attention via shift-add rotations
4. **Energy Decay**: Natural termination when energy → 0

All operations reduce to:
- Bit shifts (`<<`, `>>`)
- Bitwise ops (`&`, `|`, `^`)
- Integer add/subtract

## Citation

```bibtex
@misc{{phimamba2024,
  title={{Phi-Mamba: Integer-Only Language Models via Zeckendorf-CORDIC Arithmetic}},
  author={{Phase Locked Team}},
  year={{2024}},
  url={{https://github.com/qLeviathan/phase_locked}}
}}
```

## License

MIT License
'''


def create_model_card(config: PhiMambaConfig, variant: str, repo_id: str) -> str:
    """Generate model card content"""
    # Estimate parameters
    params_map = {"tiny": "~1M", "small": "~5M", "base": "~20M"}

    return MODEL_CARD_TEMPLATE.format(
        variant=variant.capitalize(),
        params=params_map.get(variant, "~5M"),
        vocab_size=config.vocab_size,
        max_shells=config.max_shells,
        num_layers=config.num_layers,
        scale_bits=config.scale_bits,
        max_position=config.max_position,
        repo_id=repo_id,
    )


def save_model(variant: str, output_dir: str, repo_id: str = None):
    """Save model variant to directory"""
    print(f"Creating {variant} model...")

    # Get config
    if variant == "tiny":
        config = PhiMambaConfig.tiny()
    elif variant == "small":
        config = PhiMambaConfig.small()
    else:
        config = PhiMambaConfig.base()

    # Create model and tokenizer
    model = PhiMambaForCausalLM(config)
    tokenizer = PhiMambaTokenizer(vocab_size=config.vocab_size)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create model card
    if repo_id:
        card_content = create_model_card(config, variant, repo_id)
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(card_content)

    print(f"Model saved to {output_dir}")
    return output_dir


def push_to_hub(output_dir: str, repo_id: str, token: str = None):
    """Push model to HuggingFace Hub"""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return False

    api = HfApi(token=token)

    # Create repo if needed
    try:
        create_repo(repo_id, private=False, exist_ok=True, token=token)
    except Exception as e:
        print(f"Note: {e}")

    # Upload
    print(f"Uploading to {repo_id}...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print(f"✓ Model uploaded to https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Push Phi-Mamba to HuggingFace Hub")
    parser.add_argument("--model", choices=["tiny", "small", "base", "all"], default="small")
    parser.add_argument("--repo", type=str, help="HuggingFace repo ID (e.g., username/phi-mamba-small)")
    parser.add_argument("--output", type=str, default="./phi_mamba_models")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--token", type=str, help="HuggingFace token")

    args = parser.parse_args()

    variants = ["tiny", "small", "base"] if args.model == "all" else [args.model]

    for variant in variants:
        output_dir = os.path.join(args.output, f"phi-mamba-{variant}")
        repo_id = args.repo or f"phi-mamba-{variant}"

        save_model(variant, output_dir, repo_id)

        if args.push:
            push_to_hub(output_dir, repo_id, args.token)


if __name__ == "__main__":
    main()
