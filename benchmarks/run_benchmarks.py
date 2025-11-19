#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Zeckendorf-CORDIC System

Benchmarks:
1. Integer operations vs floating point
2. Zeckendorf encoding speed
3. Token generation speed
4. Memory usage
5. Comparison with Ollama models
"""

import time
import sys
import os
from typing import List, Dict, Tuple
import json

sys.path.append('../phi_mamba_integer')
sys.path.append('../phi_mamba')

try:
    from integer_phi_mamba import IntegerPhiMamba, HAS_RUST
    HAS_PHI_MAMBA = True
except ImportError:
    HAS_PHI_MAMBA = False

try:
    from core import PhiLanguageModel
    HAS_ORIGINAL_MAMBA = True
except ImportError:
    HAS_ORIGINAL_MAMBA = False

try:
    import requests
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class BenchmarkSuite:
    """Comprehensive benchmark suite"""

    def __init__(self):
        self.results = {}
        self.test_prompts = [
            "The cat sat on the",
            "In the beginning was the",
            "Once upon a time in a",
            "The quick brown fox",
            "Machine learning is",
            "Climate change affects",
            "The future of technology",
            "Quantum computing will",
            "Artificial intelligence can",
            "The solar system contains"
        ]

    def benchmark_integer_ops(self) -> Dict:
        """Benchmark integer-only vs floating-point operations"""
        print("\n" + "=" * 60)
        print("BENCHMARK 1: Integer vs Floating Point Operations")
        print("=" * 60)

        results = {}

        # Integer operations
        print("\nTesting integer-only operations...")
        start = time.time()
        for _ in range(1000000):
            # Integer multiply-add
            a = 12345
            b = 67890
            c = (a * b) >> 16  # Scaled integer multiply
        int_time = time.time() - start
        results['integer_ops'] = int_time

        # Floating point operations
        print("Testing floating-point operations...")
        start = time.time()
        for _ in range(1000000):
            # Float multiply-add
            a = 0.12345
            b = 0.67890
            c = a * b
        float_time = time.time() - start
        results['float_ops'] = float_time

        speedup = float_time / int_time
        results['speedup'] = speedup

        print(f"\n✓ Integer operations: {int_time:.4f}s")
        print(f"✓ Float operations: {float_time:.4f}s")
        print(f"✓ Integer is {speedup:.2f}x faster")

        return results

    def benchmark_zeckendorf_encoding(self) -> Dict:
        """Benchmark Zeckendorf encoding speed"""
        print("\n" + "=" * 60)
        print("BENCHMARK 2: Zeckendorf Encoding Speed")
        print("=" * 60)

        if not HAS_PHI_MAMBA:
            print("✗ Phi-Mamba not available")
            return {}

        model = IntegerPhiMamba()
        results = {}

        # Test encoding speed
        test_texts = self.test_prompts * 10  # 100 prompts

        print(f"\nEncoding {len(test_texts)} prompts...")
        start = time.time()
        for text in test_texts:
            states = model.encode(text)
        encoding_time = time.time() - start
        results['encoding_time'] = encoding_time
        results['prompts_per_second'] = len(test_texts) / encoding_time

        print(f"✓ Encoded {len(test_texts)} prompts in {encoding_time:.4f}s")
        print(f"✓ Speed: {results['prompts_per_second']:.2f} prompts/sec")

        if HAS_RUST:
            print("✓ Using Rust-accelerated Zeckendorf encoding")
        else:
            print("✓ Using Python fallback (Rust would be ~10x faster)")

        return results

    def benchmark_generation_speed(self) -> Dict:
        """Benchmark text generation speed"""
        print("\n" + "=" * 60)
        print("BENCHMARK 3: Text Generation Speed")
        print("=" * 60)

        results = {}

        # Phi-Mamba Integer
        if HAS_PHI_MAMBA:
            print("\nTesting Phi-Mamba (Integer-Only)...")
            model = IntegerPhiMamba()

            times = []
            tokens = []
            for prompt in self.test_prompts[:5]:
                start = time.time()
                output = model.generate(prompt, max_length=50)
                elapsed = time.time() - start
                times.append(elapsed)
                tokens.append(len(output.split()))

            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens) / len(tokens)
            tokens_per_sec = avg_tokens / avg_time

            results['phi_mamba'] = {
                'avg_time': avg_time,
                'avg_tokens': avg_tokens,
                'tokens_per_sec': tokens_per_sec
            }

            print(f"✓ Average time: {avg_time:.4f}s")
            print(f"✓ Average tokens: {avg_tokens:.1f}")
            print(f"✓ Speed: {tokens_per_sec:.2f} tokens/sec")

        # Original Phi-Mamba (if available)
        if HAS_ORIGINAL_MAMBA:
            print("\nTesting Original Phi-Mamba (Float)...")
            model = PhiLanguageModel()

            times = []
            for prompt in self.test_prompts[:5]:
                start = time.time()
                output = model.generate(prompt, max_length=50)
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            results['phi_mamba_float'] = {'avg_time': avg_time}

            print(f"✓ Average time: {avg_time:.4f}s")

        return results

    def benchmark_ollama_comparison(self) -> Dict:
        """Benchmark against Ollama models"""
        print("\n" + "=" * 60)
        print("BENCHMARK 4: Comparison with Ollama Models")
        print("=" * 60)

        if not HAS_OLLAMA:
            print("✗ requests library not available")
            return {}

        results = {}
        ollama_models = ['llama3', 'mistral']

        for model_name in ollama_models:
            print(f"\nTesting Ollama: {model_name}...")

            try:
                times = []
                for prompt in self.test_prompts[:3]:  # Fewer prompts for Ollama
                    start = time.time()
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"num_predict": 50}
                        },
                        timeout=60
                    )

                    if response.status_code == 200:
                        elapsed = time.time() - start
                        times.append(elapsed)

                if times:
                    avg_time = sum(times) / len(times)
                    results[model_name] = {'avg_time': avg_time}
                    print(f"✓ Average time: {avg_time:.4f}s")
                else:
                    print(f"✗ No successful responses from {model_name}")

            except Exception as e:
                print(f"✗ Error testing {model_name}: {e}")
                print("   Make sure Ollama is running: ollama serve")

        return results

    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage"""
        print("\n" + "=" * 60)
        print("BENCHMARK 5: Memory Usage")
        print("=" * 60)

        if not HAS_PSUTIL:
            print("✗ psutil not available (install: pip install psutil)")
            return {}

        import psutil
        process = psutil.Process()

        results = {}

        # Initial memory
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB

        # Load Phi-Mamba
        if HAS_PHI_MAMBA:
            model = IntegerPhiMamba()
            after_load_mem = process.memory_info().rss / 1024 / 1024

            # Generate
            for prompt in self.test_prompts:
                model.generate(prompt, max_length=50)

            after_gen_mem = process.memory_info().rss / 1024 / 1024

            results['phi_mamba'] = {
                'initial_mb': initial_mem,
                'after_load_mb': after_load_mem,
                'after_gen_mb': after_gen_mem,
                'model_size_mb': after_load_mem - initial_mem,
                'gen_overhead_mb': after_gen_mem - after_load_mem
            }

            print(f"\n✓ Initial memory: {initial_mem:.2f} MB")
            print(f"✓ After loading model: {after_load_mem:.2f} MB")
            print(f"✓ After generation: {after_gen_mem:.2f} MB")
            print(f"✓ Model size: {results['phi_mamba']['model_size_mb']:.2f} MB")

        return results

    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "🚀" * 30)
        print("ZECKENDORF-CORDIC BENCHMARK SUITE")
        print("Integer-Only Phi-Mamba Transformer")
        print("🚀" * 30)

        self.results['integer_ops'] = self.benchmark_integer_ops()
        self.results['zeckendorf'] = self.benchmark_zeckendorf_encoding()
        self.results['generation'] = self.benchmark_generation_speed()
        self.results['ollama_comparison'] = self.benchmark_ollama_comparison()
        self.results['memory'] = self.benchmark_memory_usage()

        self._print_summary()
        self._save_results()

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if 'integer_ops' in self.results and 'speedup' in self.results['integer_ops']:
            print(f"\n✓ Integer operations are {self.results['integer_ops']['speedup']:.2f}x faster than floats")

        if 'zeckendorf' in self.results and 'prompts_per_second' in self.results['zeckendorf']:
            print(f"✓ Zeckendorf encoding: {self.results['zeckendorf']['prompts_per_second']:.2f} prompts/sec")

        if 'generation' in self.results and 'phi_mamba' in self.results['generation']:
            tokens_per_sec = self.results['generation']['phi_mamba']['tokens_per_sec']
            print(f"✓ Generation speed: {tokens_per_sec:.2f} tokens/sec")

        if 'memory' in self.results and 'phi_mamba' in self.results['memory']:
            model_size = self.results['memory']['phi_mamba']['model_size_mb']
            print(f"✓ Model size: {model_size:.2f} MB")

        print("\n" + "=" * 60)

    def _save_results(self):
        """Save results to JSON"""
        filename = f"benchmark_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")


def main():
    suite = BenchmarkSuite()
    suite.run_all_benchmarks()


if __name__ == "__main__":
    main()
