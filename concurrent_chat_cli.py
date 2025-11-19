#!/usr/bin/env python3
"""
Concurrent Model Chat CLI

Run multiple models concurrently:
- Integer-Only Phi-Mamba (Zeckendorf-CORDIC)
- Ollama models (llama3, mistral, etc.)

Usage:
    python concurrent_chat_cli.py --models phi-mamba,llama3
"""

import argparse
import asyncio
import sys
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import time

# Add paths
sys.path.append('phi_mamba_integer')
sys.path.append('phi_mamba')

try:
    from integer_phi_mamba import IntegerPhiMamba
    HAS_INTEGER_MAMBA = True
except ImportError:
    print("Warning: Integer Phi-Mamba not available")
    HAS_INTEGER_MAMBA = False

try:
    import requests
    HAS_OLLAMA = True
except ImportError:
    print("Warning: requests not available. Install with: pip install requests")
    HAS_OLLAMA = False


class ModelInterface:
    """Base interface for models"""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        raise NotImplementedError


class PhiMambaModel(ModelInterface):
    """Integer-only Phi-Mamba model"""

    def __init__(self):
        super().__init__("Phi-Mamba-Integer")
        if not HAS_INTEGER_MAMBA:
            raise ValueError("Integer Phi-Mamba not available")
        self.model = IntegerPhiMamba(vocab_size=50000)

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        return self.model.generate(prompt, max_length=max_tokens)


class OllamaModel(ModelInterface):
    """Ollama model interface"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(f"Ollama-{model_name}")
        if not HAS_OLLAMA:
            raise ValueError("requests library not available")
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get('response', 'No response')
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"


class ConcurrentChatCLI:
    """CLI for running multiple models concurrently"""

    def __init__(self, models: List[ModelInterface]):
        self.models = models
        self.executor = ThreadPoolExecutor(max_workers=len(models))

    def generate_concurrent(self, prompt: str, max_tokens: int = 100) -> Dict[str, str]:
        """Generate from all models concurrently"""
        futures = {}

        # Submit all tasks
        for model in self.models:
            future = self.executor.submit(model.generate, prompt, max_tokens)
            futures[model.name] = future

        # Collect results
        results = {}
        for model_name, future in futures.items():
            try:
                results[model_name] = future.result(timeout=60)
            except Exception as e:
                results[model_name] = f"Error: {str(e)}"

        return results

    def interactive_mode(self):
        """Run interactive chat with all models"""
        print("=" * 80)
        print("  CONCURRENT MODEL CHAT CLI")
        print(f"  Running {len(self.models)} models simultaneously")
        print("  Models: " + ", ".join([m.name for m in self.models]))
        print("=" * 80)
        print()
        print("Commands:")
        print("  Type your prompt and press Enter")
        print("  Type 'quit' or 'exit' to stop")
        print("  Type 'models' to list active models")
        print()

        while True:
            try:
                prompt = input("\nYou: ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if prompt.lower() == 'models':
                    print("\nActive models:")
                    for i, model in enumerate(self.models, 1):
                        print(f"  {i}. {model.name}")
                    continue

                print("\n" + "=" * 80)
                print("Generating responses...")
                start_time = time.time()

                # Generate from all models concurrently
                results = self.generate_concurrent(prompt)

                elapsed = time.time() - start_time

                # Display results
                for model_name, response in results.items():
                    print()
                    print(f"[{model_name}]")
                    print("-" * 80)
                    print(response)
                    print()

                print(f"Total time: {elapsed:.2f}s (concurrent execution)")
                print("=" * 80)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    def benchmark_mode(self, test_prompts: List[str]):
        """Run benchmark on test prompts"""
        print("=" * 80)
        print("  BENCHMARK MODE")
        print(f"  Testing {len(test_prompts)} prompts across {len(self.models)} models")
        print("=" * 80)
        print()

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}/{len(test_prompts)}]")
            print(f"Prompt: {prompt}")
            print()

            start_time = time.time()
            results = self.generate_concurrent(prompt, max_tokens=50)
            elapsed = time.time() - start_time

            for model_name, response in results.items():
                print(f"{model_name}: {response[:100]}...")

            print(f"Time: {elapsed:.2f}s")
            print("-" * 80)


def setup_models(model_names: List[str]) -> List[ModelInterface]:
    """Initialize requested models"""
    models = []

    for name in model_names:
        name_lower = name.lower()

        if name_lower in ['phi-mamba', 'phimamba', 'mamba']:
            if HAS_INTEGER_MAMBA:
                try:
                    models.append(PhiMambaModel())
                    print(f"✓ Loaded {name}")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
            else:
                print(f"✗ {name} not available")

        else:
            # Assume it's an Ollama model
            if HAS_OLLAMA:
                try:
                    models.append(OllamaModel(name))
                    print(f"✓ Loaded Ollama model: {name}")
                except Exception as e:
                    print(f"✗ Failed to load Ollama model {name}: {e}")
            else:
                print(f"✗ Ollama not available")

    return models


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Model Chat CLI - Run multiple models simultaneously"
    )
    parser.add_argument(
        '--models',
        type=str,
        default='phi-mamba,llama3',
        help='Comma-separated list of models (e.g., phi-mamba,llama3,mistral)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark mode instead of interactive'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)'
    )

    args = parser.parse_args()

    # Parse model names
    model_names = [m.strip() for m in args.models.split(',')]

    print("Initializing models...")
    models = setup_models(model_names)

    if not models:
        print("\nError: No models available!")
        print("\nTo use Ollama models:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Run: ollama pull llama3")
        print("  3. Start server: ollama serve")
        sys.exit(1)

    # Create CLI
    cli = ConcurrentChatCLI(models)

    # Run mode
    if args.benchmark:
        test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a haiku about artificial intelligence.",
            "What are the benefits of renewable energy?"
        ]
        cli.benchmark_mode(test_prompts)
    else:
        cli.interactive_mode()


if __name__ == "__main__":
    main()
