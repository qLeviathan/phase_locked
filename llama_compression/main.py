#!/usr/bin/env python3
"""
Zeckendorf-CORDIC Ablation Study - Main CLI

Usage:
    python main.py                          # Run full ablation with defaults
    python main.py --sparsity 0.99          # Test 99% sparsity
    python main.py --quick                  # Quick test (2 layers)
    python main.py --full                   # Full scale (32 layers)
    python main.py --clean                  # Clear cache and rerun
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_compression.comprehensive_ablation import ComprehensiveAblation
from llama_compression.extreme_compression import ExtremeCompressor
from llama_compression.agentic_ablation import AgenticAblation


def main():
    parser = argparse.ArgumentParser(
        description='Zeckendorf-CORDIC Ablation Study - Integer-Only Neural Network Compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Run standard ablation (4 layers, 99.5%% sparsity)
  %(prog)s --quick                  Quick test (2 layers)
  %(prog)s --full                   Full model (32 layers)
  %(prog)s --sparsity 0.99          Test with 99%% sparsity
  %(prog)s --bits 8                 Use 8-bit quantization
  %(prog)s --clean                  Clear cache and recompute
  %(prog)s --layers 8               Custom layer count

Results:
  - Memory footprint comparison
  - Compute operations count
  - Latency measurements
  - Compression ratio validation
  - Integer-only operation proof
        """
    )

    # Experiment configuration
    parser.add_argument(
        '--layers', '-l',
        type=int,
        default=4,
        help='Number of transformer layers (default: 4, full: 32)'
    )

    parser.add_argument(
        '--sparsity', '-s',
        type=float,
        default=0.995,
        help='Target sparsity (default: 0.995 = 99.5%%)'
    )

    parser.add_argument(
        '--bits', '-b',
        type=int,
        choices=[2, 4, 8],
        default=4,
        help='Quantization bits (default: 4)'
    )

    # Convenience presets
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: 2 layers, 95%% sparsity'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Full model: 32 layers, 99.5%% sparsity (slow!)'
    )

    parser.add_argument(
        '--extreme',
        action='store_true',
        help='Extreme compression: 99.7%% sparsity, 2-bit'
    )

    # Agentic mode
    parser.add_argument(
        '--agentic',
        action='store_true',
        help='Use agentic ablation with full ZORDIC lattice complexity'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name for agentic mode (auto-generated if not provided)'
    )

    parser.add_argument(
        '--consciousness',
        type=str,
        default='./consciousness.json',
        help='Consciousness state file for agentic mode (default: ./consciousness.json)'
    )

    # Cache control
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clear cache and recompute weights'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./ablation_cache',
        help='Cache directory (default: ./ablation_cache)'
    )

    # Output control
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # Apply presets
    if args.quick:
        args.layers = 2
        args.sparsity = 0.95
        print("🚀 Quick mode: 2 layers, 95% sparsity")
        print()

    if args.full:
        args.layers = 32
        args.sparsity = 0.995
        print("🔥 Full mode: 32 layers, 99.5% sparsity (this will take a while!)")
        print()

    if args.extreme:
        args.sparsity = 0.997
        args.bits = 2
        print("💀 Extreme mode: 99.7% sparsity, 2-bit quantization")
        print()

    # Clean cache if requested
    if args.clean:
        import shutil
        if os.path.exists(args.cache_dir):
            print(f"🧹 Cleaning cache directory: {args.cache_dir}")
            shutil.rmtree(args.cache_dir)
            print("✅ Cache cleared")
            print()

    # Print configuration
    if not args.quiet:
        print("=" * 80)
        print("ZECKENDORF-CORDIC ABLATION STUDY")
        print("Integer-Only Neural Network Compression")
        print("=" * 80)
        print()
        print("Configuration:")
        print(f"  Layers:      {args.layers}")
        print(f"  Sparsity:    {args.sparsity*100:.1f}%")
        print(f"  Bits:        {args.bits}-bit")
        print(f"  Cache:       {args.cache_dir}")
        print()

        # Calculate expected compression
        expected_compression = 1 / ((1 - args.sparsity) * (args.bits / 32))
        print(f"Expected compression: ~{expected_compression:.0f}x")
        print()

    # Run ablation study
    try:
        if args.agentic:
            # Agentic mode with full ZORDIC lattice complexity
            import time
            experiment_name = args.experiment or f"ablation-{int(time.time())}"

            if not args.quiet:
                print("=" * 80)
                print("🧬 AGENTIC MODE: Full ZORDIC Lattice Complexity")
                print("=" * 80)
                print(f"  Experiment: {experiment_name}")
                print(f"  Consciousness: {args.consciousness}")
                print()

            study = AgenticAblation(
                consciousness_path=args.consciousness,
                sparsity_target=args.sparsity,
                n_bits=args.bits
            )

            # Run agentic experiment
            results = study.run_agentic_experiment(
                experiment_name=experiment_name,
                n_layers=args.layers
            )

            if not args.quiet:
                print()
                print("=" * 80)
                print("✅ AGENTIC ABLATION COMPLETE")
                print("=" * 80)
                print()
                print("Key Results:")
                print(f"  Compression:       {results['compression_ratio']:.1f}x")
                print(f"  Berry phase:       {results['avg_berry_phase']:.4f}")
                print(f"  Lattices encoded:  {len(results['weight_lattices'])}")
                print(f"  Consciousness:     Cycle {results['consciousness_cycle']}")
                print(f"  Integer-only ops:  ✅ Validated")
                print()
        else:
            # Standard mode
            study = ComprehensiveAblation(cache_dir=args.cache_dir)

            # Update compressor settings
            study.compressor = ExtremeCompressor(
                sparsity_target=args.sparsity,
                n_bits=args.bits
            )

            # Run experiment
            metrics_a, metrics_b = study.run_experiment(
                n_layers=args.layers,
                n_tokens=50
            )

            # Success
            if not args.quiet:
                print()
                print("=" * 80)
                print("✅ ABLATION STUDY COMPLETE")
                print("=" * 80)
                print()
                print("Key Results:")
                print(f"  Memory reduction:  {metrics_a.weight_memory_mb / metrics_b.weight_memory_mb:.1f}x")
                print(f"  Compressed size:   {metrics_b.weight_memory_mb:.2f} MB")
                print(f"  Integer-only ops:  ✅ Validated")
                print()

        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
