"""
Command-Line Interface for Φ-Mamba

Provides easy access to phi-mamba functionality from the command line.

Usage:
    python -m phi_mamba.cli validate        # Run validation checks
    python -m phi_mamba.cli encode "text"   # Encode text to phi-space
    python -m phi_mamba.cli zeckendorf 100  # Show Zeckendorf decomposition
    python -m phi_mamba.cli fibonacci 20    # Show Fibonacci sequence
    python -m phi_mamba.cli generate "prompt" # Generate text
    python -m phi_mamba.cli info            # Show system information
"""

import argparse
import sys
from typing import List, Optional


def cmd_validate(args) -> int:
    """Run validation checks on all modules."""
    print("=" * 70)
    print("Φ-Mamba Validation Suite")
    print("=" * 70)
    print()

    errors = []

    # Validate constants
    print("[1/3] Validating constants...")
    try:
        from .constants import validate_constants
        validate_constants()
        print("  ✓ Constants validation passed")
    except Exception as e:
        print(f"  ✗ Constants validation failed: {e}")
        errors.append(("constants", str(e)))

    # Validate math core
    print("[2/3] Validating math core...")
    try:
        from .math_core import verify_fibonacci, cassini_identity
        # Test Fibonacci accuracy
        for n in [10, 20, 30, 40]:
            if not verify_fibonacci(n):
                raise AssertionError(f"Fibonacci({n}) mismatch")
        # Test Cassini identity
        for n in range(1, 15):
            cassini_identity(n)
        print("  ✓ Math core validation passed")
    except Exception as e:
        print(f"  ✗ Math core validation failed: {e}")
        errors.append(("math_core", str(e)))

    # Validate phase dynamics
    print("[3/3] Validating phase dynamics...")
    try:
        from .phase_dynamics import validate_phase_dynamics
        validate_phase_dynamics()
        print("  ✓ Phase dynamics validation passed")
    except Exception as e:
        print(f"  ✗ Phase dynamics validation failed: {e}")
        errors.append(("phase_dynamics", str(e)))

    print()
    if errors:
        print(f"✗ Validation completed with {len(errors)} error(s)")
        for module, error in errors:
            print(f"  - {module}: {error}")
        return 1
    else:
        print("✓ All validations passed!")
        return 0


def cmd_encode(args) -> int:
    """Encode text to phi-space representation."""
    from .encoding import retrocausal_encode
    from .constants import PHI

    text = args.text
    tokens = text.split()

    print(f"Input: \"{text}\"")
    print(f"Tokens: {len(tokens)}")
    print()

    states = retrocausal_encode(tokens, vocab_size=args.vocab_size)

    print("Token States:")
    print("-" * 70)
    for state in states:
        future_str = f"{state.future_constraint:.3f}" if state.future_constraint else "None"
        print(f"  [{state.position}] '{state.token}'")
        print(f"      θ={state.theta_total:.4f} rad, E={state.energy:.6f}")
        print(f"      shells={state.active_shells}, future={future_str}")
        print(f"      coherence_weight={state.coherence_weight:.2f}")

    print()
    print("Summary:")
    total_energy = sum(s.energy for s in states)
    avg_shells = sum(len(s.active_shells) for s in states) / len(states) if states else 0
    print(f"  Total energy: {total_energy:.6f}")
    print(f"  Average active shells: {avg_shells:.2f}")

    return 0


def cmd_zeckendorf(args) -> int:
    """Show Zeckendorf decomposition of a number."""
    from .math_core import zeckendorf_decomposition, zeckendorf_to_indices, fibonacci

    n = args.number

    decomp = zeckendorf_decomposition(n)
    indices = zeckendorf_to_indices(n)

    print(f"Zeckendorf Decomposition of {n}")
    print("=" * 50)
    print()

    if decomp:
        fib_strs = [f"F({idx})={val}" for idx, val in zip(indices, decomp)]
        print(f"  {n} = {' + '.join(map(str, decomp))}")
        print(f"      = {' + '.join(fib_strs)}")
        print()
        print(f"  Active shells (indices): {indices}")
        print(f"  Number of terms: {len(decomp)}")

        # Show binary representation
        max_idx = max(indices) if indices else 0
        binary = ['0'] * (max_idx + 1)
        for idx in indices:
            binary[max_idx - idx] = '1'
        print(f"  Z-binary: {''.join(binary)}")
    else:
        print(f"  {n} = 0 (empty decomposition)")

    return 0


def cmd_fibonacci(args) -> int:
    """Show Fibonacci sequence."""
    from .math_core import fibonacci, lucas, fibonacci_sequence

    n = args.count

    print(f"Fibonacci and Lucas Numbers (0 to {n})")
    print("=" * 50)
    print()
    print("  n    F(n)        L(n)")
    print("  " + "-" * 30)

    fibs = fibonacci_sequence(n)

    for i in range(min(n + 1, len(fibs))):
        f_n = fibs[i]
        l_n = lucas(i)
        print(f"  {i:3d}  {f_n:10d}  {l_n:10d}")

    print()
    print("Properties:")
    print(f"  F({n}) = {fibonacci(n)}")
    print(f"  L({n}) = {lucas(n)}")
    print(f"  F({n}) + F({n+2}) = L({n+1}) = {fibonacci(n) + fibonacci(n+2)}")

    return 0


def cmd_generate(args) -> int:
    """Generate text using phi-mamba model."""
    from .core import PhiLanguageModel

    prompt = args.prompt
    max_length = args.max_length
    temperature = args.temperature

    print(f"Prompt: \"{prompt}\"")
    print(f"Max length: {max_length} tokens")
    print(f"Temperature: {temperature}")
    print()

    model = PhiLanguageModel(vocab_size=args.vocab_size)

    print("Generating...")
    result = model.generate(
        prompt,
        max_length=max_length,
        temperature=temperature
    )

    print()
    print("Generated text:")
    print("-" * 50)
    print(result)
    print("-" * 50)

    input_tokens = len(prompt.split())
    output_tokens = len(result.split())
    print(f"\nTokens: {input_tokens} input → {output_tokens} total (+{output_tokens - input_tokens} generated)")

    return 0


def cmd_info(args) -> int:
    """Show system information and constants."""
    from .constants import (
        PHI, PSI, SQRT_5, LN_PHI, PHI_SQUARED, INV_PHI,
        TWO_PI, PENTAGON_ANGLE, GOLDEN_ANGLE,
        DEFAULT_PHASE_TOLERANCE, EPSILON
    )
    from . import __version__, __author__

    print("=" * 70)
    print("Φ-Mamba System Information")
    print("=" * 70)
    print()

    print("Version Information:")
    print(f"  Version: {__version__}")
    print(f"  Author: {__author__}")
    print()

    print("Primary Constants:")
    print(f"  φ (PHI)     = {PHI:.15f}")
    print(f"  ψ (PSI)     = {PSI:.15f}")
    print(f"  √5 (SQRT_5) = {SQRT_5:.15f}")
    print(f"  ln(φ)       = {LN_PHI:.15f}")
    print()

    print("Derived Constants:")
    print(f"  φ²          = {PHI_SQUARED:.15f}")
    print(f"  1/φ         = {INV_PHI:.15f}")
    print()

    print("Angular Constants:")
    print(f"  2π          = {TWO_PI:.15f}")
    print(f"  Pentagon    = {PENTAGON_ANGLE:.15f} rad (108°)")
    print(f"  Golden      = {GOLDEN_ANGLE:.15f} rad (137.5°)")
    print()

    print("Tolerances:")
    print(f"  Phase lock  = {DEFAULT_PHASE_TOLERANCE} rad")
    print(f"  Epsilon     = {EPSILON}")
    print()

    print("Key Identities:")
    print(f"  φ² - φ - 1  = {PHI**2 - PHI - 1:.2e} (should be 0)")
    print(f"  φ × ψ       = {PHI * PSI:.15f} (should be -1)")
    print(f"  φ + ψ       = {PHI + PSI:.15f} (should be 1)")
    print(f"  φ - ψ       = {PHI - PSI:.15f} (should be √5)")

    return 0


def cmd_demo(args) -> int:
    """Run interactive demo of phi-mamba capabilities."""
    from .constants import PHI, PSI
    from .math_core import fibonacci, zeckendorf_decomposition
    from .phase_dynamics import energy_decay_sequence, standing_wave
    from .encoding import retrocausal_encode

    print("=" * 70)
    print("Φ-Mamba Interactive Demo")
    print("=" * 70)
    print()

    # Demo 1: Golden ratio properties
    print("[1] Golden Ratio Properties")
    print("-" * 40)
    print(f"  φ = {PHI:.10f}")
    print(f"  φ² = φ + 1 → {PHI**2:.10f} = {PHI + 1:.10f} ✓")
    print(f"  1/φ = φ - 1 → {1/PHI:.10f} = {PHI - 1:.10f} ✓")
    print()

    # Demo 2: Fibonacci spiral
    print("[2] Fibonacci Sequence (first 15)")
    print("-" * 40)
    fibs = [fibonacci(i) for i in range(15)]
    print(f"  {fibs}")
    print(f"  Ratio F(14)/F(13) = {fibonacci(14)/fibonacci(13):.10f} → φ")
    print()

    # Demo 3: Zeckendorf decomposition
    print("[3] Zeckendorf Decomposition")
    print("-" * 40)
    for n in [17, 42, 100]:
        decomp = zeckendorf_decomposition(n)
        print(f"  {n} = {' + '.join(map(str, decomp))}")
    print()

    # Demo 4: Energy decay
    print("[4] Pentagon Reflection Energy Decay")
    print("-" * 40)
    energies = energy_decay_sequence(1.0, 8)
    for i, e in enumerate(energies):
        bar = "█" * int(e * 50)
        print(f"  Reflection {i}: E={e:.6f} {bar}")
    print()

    # Demo 5: Standing waves
    print("[5] Standing Wave Amplitudes (φᵏ + ψᵏ)")
    print("-" * 40)
    for k in range(7):
        sw = standing_wave(k)
        print(f"  k={k}: {sw:.6f}")
    print()

    # Demo 6: Text encoding
    print("[6] Text Encoding Demo")
    print("-" * 40)
    text = "the cat sat on the mat"
    tokens = text.split()
    states = retrocausal_encode(tokens, vocab_size=10000)
    print(f"  Text: \"{text}\"")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Total energy: {sum(s.energy for s in states):.6f}")
    print(f"  Avg shells: {sum(len(s.active_shells) for s in states) / len(states):.2f}")
    print()

    print("=" * 70)
    print("Demo complete! Use 'python -m phi_mamba.cli --help' for more commands.")
    print("=" * 70)

    return 0


def cmd_benchmark(args) -> int:
    """Run performance benchmarks."""
    import time
    from .math_core import fibonacci, zeckendorf_decomposition, FibonacciCache
    from .encoding import retrocausal_encode

    print("=" * 70)
    print("Φ-Mamba Performance Benchmarks")
    print("=" * 70)
    print()

    iterations = args.iterations

    # Benchmark 1: Fibonacci computation
    print(f"[1] Fibonacci Computation ({iterations} calls)")
    print("-" * 40)
    start = time.perf_counter()
    for i in range(iterations):
        _ = fibonacci(i % 50)
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {ops_per_sec:,.0f} ops/sec")
    print()

    # Benchmark 2: Cached Fibonacci
    print(f"[2] Cached Fibonacci ({iterations} lookups)")
    print("-" * 40)
    cache = FibonacciCache(max_n=100)
    start = time.perf_counter()
    for i in range(iterations):
        _ = cache.get_fibonacci(i % 100)
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {ops_per_sec:,.0f} ops/sec")
    print()

    # Benchmark 3: Zeckendorf decomposition
    print(f"[3] Zeckendorf Decomposition ({iterations} decompositions)")
    print("-" * 40)
    start = time.perf_counter()
    for i in range(1, iterations + 1):
        _ = zeckendorf_decomposition(i)
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {ops_per_sec:,.0f} ops/sec")
    print()

    # Benchmark 4: Text encoding
    tokens_per_batch = 100
    num_batches = iterations // 10
    print(f"[4] Text Encoding ({num_batches} batches × {tokens_per_batch} tokens)")
    print("-" * 40)
    test_tokens = ["word"] * tokens_per_batch
    start = time.perf_counter()
    for _ in range(num_batches):
        _ = retrocausal_encode(test_tokens, vocab_size=50000)
    elapsed = time.perf_counter() - start
    total_tokens = num_batches * tokens_per_batch
    tokens_per_sec = total_tokens / elapsed
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {tokens_per_sec:,.0f} tokens/sec")
    print()

    print("=" * 70)
    print("Benchmarks complete!")
    print("=" * 70)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="phi_mamba",
        description="Φ-Mamba: Phase-Locked Language Modeling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m phi_mamba.cli validate
  python -m phi_mamba.cli encode "the cat sat on the mat"
  python -m phi_mamba.cli zeckendorf 100
  python -m phi_mamba.cli fibonacci 20
  python -m phi_mamba.cli generate "Once upon a time" --max-length 20
  python -m phi_mamba.cli demo
  python -m phi_mamba.cli benchmark
  python -m phi_mamba.cli info
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Run validation checks on all modules"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # encode command
    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode text to phi-space representation"
    )
    encode_parser.add_argument("text", help="Text to encode")
    encode_parser.add_argument(
        "--vocab-size", type=int, default=50000,
        help="Vocabulary size (default: 50000)"
    )
    encode_parser.set_defaults(func=cmd_encode)

    # zeckendorf command
    zeck_parser = subparsers.add_parser(
        "zeckendorf",
        help="Show Zeckendorf decomposition of a number"
    )
    zeck_parser.add_argument("number", type=int, help="Number to decompose")
    zeck_parser.set_defaults(func=cmd_zeckendorf)

    # fibonacci command
    fib_parser = subparsers.add_parser(
        "fibonacci",
        help="Show Fibonacci and Lucas sequences"
    )
    fib_parser.add_argument(
        "count", type=int, nargs="?", default=20,
        help="How many numbers to show (default: 20)"
    )
    fib_parser.set_defaults(func=cmd_fibonacci)

    # generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate text using phi-mamba model"
    )
    gen_parser.add_argument("prompt", help="Starting prompt")
    gen_parser.add_argument(
        "--max-length", type=int, default=20,
        help="Maximum number of tokens (default: 20)"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    gen_parser.add_argument(
        "--vocab-size", type=int, default=50000,
        help="Vocabulary size (default: 50000)"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information and constants"
    )
    info_parser.set_defaults(func=cmd_info)

    # demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run interactive demo of phi-mamba capabilities"
    )
    demo_parser.set_defaults(func=cmd_demo)

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks"
    )
    bench_parser.add_argument(
        "--iterations", type=int, default=10000,
        help="Number of iterations (default: 10000)"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
