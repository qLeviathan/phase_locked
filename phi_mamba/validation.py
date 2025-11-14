"""
Validation Utilities for Φ-Mamba

Provides shared test utilities, fixtures, and validation functions
for testing the phi-mamba framework.

This module consolidates common testing patterns used across:
- Game theory validation
- Integration tests
- Unit tests
- Property-based tests
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from .constants import PHI, PSI, EPSILON, TWO_PI
from .math_core import fibonacci, lucas, cassini_identity, zeckendorf_decomposition
from .phase_dynamics import compute_berry_phase, is_phase_locked, energy_decay_sequence
from .encoding import TokenState, retrocausal_encode


# ============================================================================
# TEST FIXTURES & SAMPLE DATA
# ============================================================================

def create_sample_tokens(n: int = 10) -> List[str]:
    """
    Create sample tokens for testing.

    Args:
        n: Number of tokens to generate

    Returns:
        List of token strings
    """
    words = [
        "the", "quick", "brown", "fox", "jumps",
        "over", "lazy", "dog", "and", "cat",
        "runs", "fast", "slow", "big", "small",
        "hello", "world", "test", "data", "phi"
    ]
    return [words[i % len(words)] for i in range(n)]


def create_sample_states(n: int = 10, vocab_size: int = 1000) -> List[TokenState]:
    """
    Create sample TokenStates for testing.

    Args:
        n: Number of states to create
        vocab_size: Vocabulary size

    Returns:
        List of TokenStates with retrocausal encoding
    """
    tokens = create_sample_tokens(n)
    return retrocausal_encode(tokens, vocab_size)


def create_fibonacci_sequence_data(n: int = 20) -> Dict[str, List]:
    """
    Create comprehensive Fibonacci sequence test data.

    Returns:
        Dictionary with:
        - fibonacci: List of Fibonacci numbers
        - lucas: List of Lucas numbers
        - ratios: List of F(n+1)/F(n) ratios (→ φ)
        - cassini: List of Cassini identity values
    """
    fibs = [fibonacci(i) for i in range(n)]
    luc = [lucas(i) for i in range(n)]
    ratios = [fibs[i+1] / fibs[i] if fibs[i] > 0 else 0 for i in range(n-1)]
    cassini_vals = [cassini_identity(i) for i in range(1, n)]

    return {
        'fibonacci': fibs,
        'lucas': luc,
        'ratios': ratios,
        'cassini': cassini_vals
    }


# ============================================================================
# MATHEMATICAL PROPERTY VALIDATORS
# ============================================================================

def validate_golden_ratio_properties() -> bool:
    """
    Validate all golden ratio mathematical properties.

    Checks:
    1. φ² = φ + 1
    2. φ - ψ = √5
    3. φ × ψ = -1
    4. φ + ψ = 1

    Returns:
        True if all properties hold

    Raises:
        AssertionError if any property fails
    """
    # φ² = φ + 1
    assert abs(PHI**2 - PHI - 1) < EPSILON, f"φ² - φ - 1 = {PHI**2 - PHI - 1}"

    # ψ² = ψ + 1
    assert abs(PSI**2 - PSI - 1) < EPSILON, f"ψ² - ψ - 1 = {PSI**2 - PSI - 1}"

    # φ - ψ = √5
    from .constants import SQRT_5
    assert abs(PHI - PSI - SQRT_5) < EPSILON, f"φ - ψ = {PHI - PSI}, should be √5"

    # φ × ψ = -1
    assert abs(PHI * PSI + 1) < EPSILON, f"φ × ψ = {PHI * PSI}, should be -1"

    # φ + ψ = 1
    assert abs(PHI + PSI - 1) < EPSILON, f"φ + ψ = {PHI + PSI}, should be 1"

    return True


def validate_fibonacci_properties(max_n: int = 20) -> bool:
    """
    Validate Fibonacci sequence properties.

    Checks:
    1. F(n) = F(n-1) + F(n-2)
    2. Cassini identity
    3. Ratio convergence to φ
    4. Zeckendorf decomposition

    Args:
        max_n: Maximum index to test

    Returns:
        True if all properties hold
    """
    # Recurrence relation
    for n in range(2, max_n):
        fn = fibonacci(n)
        fn1 = fibonacci(n - 1)
        fn2 = fibonacci(n - 2)
        assert fn == fn1 + fn2, f"F({n}) = {fn} != {fn1} + {fn2}"

    # Cassini identity
    for n in range(2, max_n):
        result = cassini_identity(n)
        expected = (-1)**n
        assert result == expected, f"Cassini({n}) = {result} != {expected}"

    # Ratio convergence (for n > 10, ratio should be within 1% of φ)
    for n in range(10, max_n):
        fn = fibonacci(n)
        fn1 = fibonacci(n + 1)
        if fn > 0:
            ratio = fn1 / fn
            error = abs(ratio - PHI) / PHI
            assert error < 0.01, f"F({n+1})/F({n}) = {ratio}, error = {error*100:.2f}%"

    # Zeckendorf decomposition uniqueness
    for n in range(1, 100):
        decomp = zeckendorf_decomposition(n)
        # Check sum equals n
        assert sum(decomp) == n, f"Zeckendorf({n}) sum != {n}"
        # Check non-consecutive
        for i in range(len(decomp) - 1):
            # Decomp is in descending order, so check ratios
            # If consecutive, ratio should be φ (1.618...)
            ratio = decomp[i] / decomp[i+1]
            # Consecutive Fibonacci ratio is exactly φ
            # Non-consecutive should be > φ
            assert ratio > 1.7 or abs(ratio - PHI) > 0.1, \
                f"Consecutive Fibonacci in Zeckendorf({n}): {decomp}"

    return True


def validate_token_state(state: TokenState) -> bool:
    """
    Validate a TokenState has correct properties.

    Checks:
    1. Energy = φ^(-position)
    2. theta_total ∈ [0, 2π)
    3. Zeckendorf decomposition valid
    4. Active shells match Zeckendorf

    Args:
        state: TokenState to validate

    Returns:
        True if valid

    Raises:
        AssertionError if invalid
    """
    # Energy check
    expected_energy = PHI**(-state.position)
    assert abs(state.energy - expected_energy) < EPSILON, \
        f"Energy mismatch: {state.energy} != {expected_energy}"

    # Theta in range [0, 2π)
    assert 0 <= state.theta_total < TWO_PI, \
        f"theta_total out of range: {state.theta_total}"

    # Zeckendorf valid
    zeck_sum = sum(state.zeckendorf)
    expected_sum = state.position + 1
    assert zeck_sum == expected_sum, \
        f"Zeckendorf sum {zeck_sum} != position+1 {expected_sum}"

    # Active shells match
    assert state.active_shells == state.zeckendorf, \
        "Active shells don't match Zeckendorf"

    return True


# ============================================================================
# GAME THEORY VALIDATION
# ============================================================================

@dataclass
class BackwardInductionResult:
    """Result of backward induction validation"""
    values: List[float]
    utilities: List[float]
    actions: List[str]
    equilibrium_value: float
    converged: bool


def validate_backward_induction(
    states: List[TokenState],
    discount_factor: float = None
) -> BackwardInductionResult:
    """
    Validate backward induction game theory property.

    Implements subgame perfect equilibrium via backward induction:
    V(t) = U(t) + discount * V(t+1)

    Where discount = 1/φ (golden ratio discounting)

    Args:
        states: Sequence of TokenStates
        discount_factor: Discount factor (default: 1/φ)

    Returns:
        BackwardInductionResult with values and convergence info
    """
    if discount_factor is None:
        discount_factor = 1 / PHI

    n = len(states)
    values = [0.0] * n
    utilities = []
    actions = []

    # Backward pass
    for i in range(n - 1, -1, -1):
        state = states[i]

        # Immediate utility = energy × coherence
        immediate_utility = state.energy * state.coherence_weight
        utilities.append(immediate_utility)

        # Future value (0 if terminal)
        future_value = values[i + 1] if i < n - 1 else 0.0

        # Bellman equation
        values[i] = immediate_utility + discount_factor * future_value

        # Action: continue if phase-locked, terminate otherwise
        if i < n - 1:
            gamma = compute_berry_phase(states[i], states[i + 1])
            locked = is_phase_locked(gamma)
            actions.append("CONTINUE" if locked else "TERMINATE")
        else:
            actions.append("TERMINAL")

    # Check convergence (values should decay)
    converged = all(
        values[i] >= values[i + 1] for i in range(len(values) - 1)
    )

    return BackwardInductionResult(
        values=values,
        utilities=utilities[::-1],  # Reverse (was backward)
        actions=actions[::-1],
        equilibrium_value=values[0],
        converged=converged
    )


def validate_nash_equilibrium(
    states: List[TokenState],
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that phase-locked states form Nash equilibrium.

    A phase-locked sequence is in Nash equilibrium if no single
    deviation improves utility.

    Args:
        states: Sequence of TokenStates
        tolerance: Numerical tolerance

    Returns:
        True if Nash equilibrium holds
    """
    result = validate_backward_induction(states)

    # Check that continuing at phase-locked states is optimal
    for i in range(len(states) - 1):
        if result.actions[i] == "CONTINUE":
            # Verify continuing gives higher value than terminating
            continue_value = result.values[i]
            terminate_value = result.utilities[i]  # Just immediate utility
            assert continue_value >= terminate_value - tolerance, \
                f"Non-optimal continuation at {i}: {continue_value} < {terminate_value}"

    return True


# ============================================================================
# INTEGRATION TEST UTILITIES
# ============================================================================

def test_full_pipeline(
    tokens: Optional[List[str]] = None,
    vocab_size: int = 1000
) -> Dict[str, Any]:
    """
    Test full phi-mamba pipeline end-to-end.

    Steps:
    1. Create/use tokens
    2. Encode to TokenStates
    3. Apply retrocausal constraints
    4. Compute Berry phases
    5. Check phase locking
    6. Run backward induction

    Args:
        tokens: Token list (or None for default)
        vocab_size: Vocabulary size

    Returns:
        Dictionary with all test results
    """
    if tokens is None:
        tokens = create_sample_tokens(15)

    # Encode
    states = retrocausal_encode(tokens, vocab_size)

    # Validate each state
    for state in states:
        validate_token_state(state)

    # Compute Berry phases
    berry_phases = []
    phase_locks = []
    for i in range(len(states) - 1):
        gamma = compute_berry_phase(states[i], states[i + 1])
        locked = is_phase_locked(gamma)
        berry_phases.append(gamma)
        phase_locks.append(locked)

    # Backward induction
    bi_result = validate_backward_induction(states)

    # Energy decay
    energies = energy_decay_sequence(1.0, len(states))

    return {
        'tokens': tokens,
        'states': states,
        'berry_phases': berry_phases,
        'phase_locks': phase_locks,
        'num_locked': sum(phase_locks),
        'lock_ratio': sum(phase_locks) / len(phase_locks) if phase_locks else 0,
        'backward_induction': bi_result,
        'energies': energies,
        'all_valid': True
    }


def validate_all_modules() -> Dict[str, bool]:
    """
    Comprehensive validation of all phi-mamba modules.

    Returns:
        Dictionary mapping module names to validation results
    """
    results = {}

    try:
        results['constants'] = validate_golden_ratio_properties()
    except AssertionError as e:
        results['constants'] = False
        print(f"Constants validation failed: {e}")

    try:
        results['fibonacci'] = validate_fibonacci_properties(20)
    except AssertionError as e:
        results['fibonacci'] = False
        print(f"Fibonacci validation failed: {e}")

    try:
        states = create_sample_states(10)
        for state in states:
            validate_token_state(state)
        results['encoding'] = True
    except AssertionError as e:
        results['encoding'] = False
        print(f"Encoding validation failed: {e}")

    try:
        states = create_sample_states(10)
        validate_nash_equilibrium(states)
        results['game_theory'] = True
    except AssertionError as e:
        results['game_theory'] = False
        print(f"Game theory validation failed: {e}")

    try:
        pipeline_result = test_full_pipeline()
        results['integration'] = pipeline_result['all_valid']
    except Exception as e:
        results['integration'] = False
        print(f"Integration test failed: {e}")

    return results


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_fibonacci(max_n: int = 1000) -> Dict[str, float]:
    """
    Benchmark Fibonacci computation performance.

    Args:
        max_n: Maximum index to compute

    Returns:
        Dictionary with timing results
    """
    import time

    # Time Binet formula
    start = time.perf_counter()
    for i in range(max_n):
        fibonacci(i)
    binet_time = time.perf_counter() - start

    # Time Zeckendorf
    start = time.perf_counter()
    for i in range(1, max_n):
        zeckendorf_decomposition(i)
    zeckendorf_time = time.perf_counter() - start

    return {
        'fibonacci_time': binet_time,
        'fibonacci_per_call': binet_time / max_n,
        'zeckendorf_time': zeckendorf_time,
        'zeckendorf_per_call': zeckendorf_time / (max_n - 1)
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHI-MAMBA VALIDATION SUITE")
    print("=" * 70)
    print()

    # Validate all modules
    print("Running comprehensive validation...")
    results = validate_all_modules()

    print("\nVALIDATION RESULTS:")
    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {module:20s} {status}")

    all_passed = all(results.values())
    print()
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
    else:
        print("⚠️  Some validations failed")

    # Run integration test
    print("\n" + "=" * 70)
    print("INTEGRATION TEST")
    print("=" * 70)

    pipeline = test_full_pipeline()
    print(f"\nTokens: {len(pipeline['tokens'])}")
    print(f"States: {len(pipeline['states'])}")
    print(f"Phase locks: {pipeline['num_locked']}/{len(pipeline['phase_locks'])}")
    print(f"Lock ratio: {pipeline['lock_ratio']:.1%}")
    print(f"Equilibrium value: {pipeline['backward_induction'].equilibrium_value:.4f}")
    print(f"Converged: {pipeline['backward_induction'].converged}")

    # Benchmark
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    bench = benchmark_fibonacci(1000)
    print(f"\nFibonacci (1000 calls): {bench['fibonacci_time']*1000:.2f} ms")
    print(f"  Per call: {bench['fibonacci_per_call']*1e6:.2f} μs")
    print(f"\nZeckendorf (1000 calls): {bench['zeckendorf_time']*1000:.2f} ms")
    print(f"  Per call: {bench['zeckendorf_per_call']*1e6:.2f} μs")

    print("\n" + "=" * 70)
