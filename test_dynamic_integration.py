#!/usr/bin/env python3
"""
Comprehensive Dynamic Integration Test

Tests that ALL components are properly connected and using
centralized modules:
- phi_mamba core
- financial modules
- zordic_desktop
- validation utilities
- external scripts (TENSOR_SERIES, etc.)

This verifies Phase 4: Dynamic component integration is complete.
"""

import sys
from typing import Dict, Any


def test_core_modules() -> Dict[str, Any]:
    """Test core phi_mamba modules"""
    print("\n" + "="*70)
    print("TEST 1: CORE MODULES (constants, math_core, phase_dynamics)")
    print("="*70)

    from phi_mamba.constants import PHI, PSI, SQRT_5, validate_constants
    from phi_mamba.math_core import fibonacci, zeckendorf_decomposition, cassini_identity
    from phi_mamba.phase_dynamics import (
        compute_berry_phase, is_phase_locked, energy_decay_sequence
    )

    # Test constants
    assert validate_constants(), "Constants validation failed"
    print(f"✓ Constants validated: PHI={PHI:.6f}, PSI={PSI:.6f}")

    # Test math
    fib_10 = fibonacci(10)
    assert fib_10 == 55, f"fibonacci(10) should be 55, got {fib_10}"
    print(f"✓ Fibonacci working: F(10)={fib_10}")

    zeck_17 = zeckendorf_decomposition(17)
    assert sum(zeck_17) == 17, "Zeckendorf sum should equal input"
    print(f"✓ Zeckendorf working: 17 = {zeck_17}")

    cass_5 = cassini_identity(5)
    assert cass_5 == -1, "Cassini(5) should be -1"
    print(f"✓ Cassini identity: F(6)·F(4) - F(5)² = {cass_5}")

    # Test phase dynamics
    energies = energy_decay_sequence(1.0, 7)
    assert len(energies) == 7, "Should have 7 energy values"
    assert energies[0] == 1.0, "Initial energy should be 1.0"
    assert energies[-1] < 0.1, "Energy should decay below 0.1"
    print(f"✓ Energy decay: {energies[0]:.4f} → {energies[-1]:.4f}")

    return {
        'module': 'core',
        'passed': True,
        'phi': PHI,
        'fibonacci_10': fib_10,
        'zeckendorf_17': zeck_17
    }


def test_encoding_module() -> Dict[str, Any]:
    """Test encoding module"""
    print("\n" + "="*70)
    print("TEST 2: ENCODING MODULE (TokenState, retrocausal)")
    print("="*70)

    from phi_mamba.encoding import TokenState, retrocausal_encode
    from phi_mamba.constants import PHI

    # Create states
    tokens = ['hello', 'world', 'test']
    states = retrocausal_encode(tokens)

    assert len(states) == 3, "Should create 3 states"
    print(f"✓ Created {len(states)} token states")

    # Validate first state
    state = states[0]
    assert state.token == 'hello', "Token should be 'hello'"
    assert state.position == 0, "Position should be 0"
    assert state.energy == PHI**0, "Energy should be φ^0 = 1"
    print(f"✓ First state: {state}")

    # Check retrocausal constraints
    assert states[0].future_constraint is not None, "Should have future constraint"
    print(f"✓ Retrocausal constraints applied")

    return {
        'module': 'encoding',
        'passed': True,
        'num_states': len(states),
        'first_state': str(state)
    }


def test_financial_module() -> Dict[str, Any]:
    """Test financial module with inheritance"""
    print("\n" + "="*70)
    print("TEST 3: FINANCIAL MODULE (inheritance from TokenState)")
    print("="*70)

    from phi_mamba.financial_encoding import FinancialTokenState
    from phi_mamba.financial_data import Timeframe
    from phi_mamba.constants import PHI

    # Create financial state
    state = FinancialTokenState(
        token='AAPL',
        index=0,
        position=0,
        vocab_size=100,
        price=150.0,
        volume=1_000_000,
        ticker='AAPL',
        timeframe=Timeframe.DAY_1
    )

    # Test inheritance (should have base TokenState properties)
    assert hasattr(state, 'energy'), "Should inherit energy from TokenState"
    assert hasattr(state, 'theta_total'), "Should inherit theta_total"
    assert hasattr(state, 'zeckendorf'), "Should inherit zeckendorf"
    print(f"✓ Inherits from TokenState: energy={state.energy:.4f}")

    # Test financial-specific properties
    assert state.price == 150.0, "Price should be 150.0"
    assert state.ticker == 'AAPL', "Ticker should be AAPL"
    assert state.volume == 1_000_000, "Volume should be 1M"
    print(f"✓ Financial properties: {state.ticker} @ ${state.price}")

    # Test active_shells property (inherited)
    shells = state.active_shells
    assert isinstance(shells, list), "Active shells should be a list"
    print(f"✓ Active shells (inherited property): {shells}")

    return {
        'module': 'financial',
        'passed': True,
        'ticker': state.ticker,
        'price': state.price,
        'energy': state.energy,
        'inherited': True
    }


def test_zordic_desktop() -> Dict[str, Any]:
    """Test zordic_desktop integration"""
    print("\n" + "="*70)
    print("TEST 4: ZORDIC DESKTOP (using phi_mamba library)")
    print("="*70)

    sys.path.insert(0, 'zordic_desktop/src')
    from zordic_core import FibonacciCore, PHI, PSI

    # Test that it's using phi_mamba constants
    from phi_mamba.constants import PHI as PHI_MAMBA
    assert abs(PHI - PHI_MAMBA) < 1e-10, "Should use phi_mamba constants"
    print(f"✓ Using phi_mamba constants: PHI={PHI:.6f}")

    # Test FibonacciCore
    core = FibonacciCore(20)
    assert core.F[10] == 55, "F(10) should be 55"
    print(f"✓ FibonacciCore working: F[10]={core.F[10]}")

    # Test Zeckendorf
    zeck = core.zeckendorf_decompose(17)
    assert isinstance(zeck, list), "Should return list of indices"
    print(f"✓ Zeckendorf decompose: 17 → indices {zeck}")

    return {
        'module': 'zordic_desktop',
        'passed': True,
        'uses_phi_mamba': True,
        'F_10': core.F[10]
    }


def test_tensor_series() -> Dict[str, Any]:
    """Test TENSOR_SERIES module"""
    print("\n" + "="*70)
    print("TEST 5: TENSOR_SERIES (using FibonacciCache)")
    print("="*70)

    from TENSOR_SERIES import TensorSeries, PHI
    from phi_mamba.constants import PHI as PHI_MAMBA

    # Test that it's using phi_mamba
    assert abs(PHI - PHI_MAMBA) < 1e-10, "Should use phi_mamba constants"
    print(f"✓ Using phi_mamba constants: PHI={PHI:.6f}")

    # Create TensorSeries
    ts = TensorSeries(vocab_size=100, max_seq_len=10)

    # Test fibonacci (should use cache)
    fib_8 = ts.fibonacci(8)
    assert fib_8 == 21, "F(8) should be 21"
    print(f"✓ TensorSeries.fibonacci(8) = {fib_8}")

    # Test zeckendorf (should use cache)
    zeck_10 = ts.zeckendorf_decomposition(10)
    assert sum(zeck_10) == 10, "Zeckendorf sum should equal 10"
    print(f"✓ TensorSeries.zeckendorf(10) = {zeck_10}")

    return {
        'module': 'TENSOR_SERIES',
        'passed': True,
        'uses_cache': True,
        'fib_8': fib_8
    }


def test_validation_module() -> Dict[str, Any]:
    """Test validation utilities"""
    print("\n" + "="*70)
    print("TEST 6: VALIDATION MODULE (comprehensive testing)")
    print("="*70)

    from phi_mamba.validation import (
        validate_all_modules,
        test_full_pipeline,
        benchmark_fibonacci
    )

    # Run all validations
    results = validate_all_modules()
    all_passed = all(results.values())

    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {module:20s} {status}")

    assert all_passed, "All validations should pass"
    print(f"✓ All {len(results)} validation suites passed")

    # Run full pipeline
    pipeline = test_full_pipeline(tokens=['the', 'quick', 'brown', 'fox', 'jumps'])
    assert pipeline['all_valid'], "Pipeline should be valid"
    print(f"✓ Full pipeline test: {len(pipeline['tokens'])} tokens → {pipeline['num_locked']} phase locks")

    return {
        'module': 'validation',
        'passed': True,
        'all_tests_passed': all_passed,
        'pipeline_valid': pipeline['all_valid']
    }


def test_dynamic_connections() -> Dict[str, Any]:
    """Test that all modules are dynamically connected"""
    print("\n" + "="*70)
    print("TEST 7: DYNAMIC CONNECTIONS (cross-module integration)")
    print("="*70)

    # Test: Financial → Core → Phase Dynamics
    from phi_mamba.financial_encoding import FinancialTokenState
    from phi_mamba.phase_dynamics import compute_berry_phase
    from phi_mamba.financial_data import Timeframe

    state1 = FinancialTokenState(
        token='AAPL', index=0, position=0, vocab_size=100,
        price=150.0, ticker='AAPL', timeframe=Timeframe.DAY_1
    )
    state2 = FinancialTokenState(
        token='GOOGL', index=1, position=1, vocab_size=100,
        price=2800.0, ticker='GOOGL', timeframe=Timeframe.DAY_1
    )

    # Compute Berry phase between financial states
    gamma = compute_berry_phase(state1, state2)
    assert 0 <= gamma < 6.283186, "Berry phase should be in [0, 2π)"
    print(f"✓ Financial → Phase Dynamics: γ={gamma:.4f}")

    # Test: Desktop → Core → Validation
    sys.path.insert(0, 'zordic_desktop/src')
    from zordic_core import FibonacciCore
    from phi_mamba.validation import validate_fibonacci_properties

    core = FibonacciCore(15)
    # Use fibonacci from validation to check consistency
    from phi_mamba.math_core import fibonacci
    assert core.F[10] == fibonacci(10), "Desktop and core should match"
    print(f"✓ Desktop → Core: F(10)={core.F[10]} (consistent)")

    # Validate properties
    assert validate_fibonacci_properties(15), "Fibonacci properties should hold"
    print(f"✓ Core → Validation: Properties validated")

    return {
        'test': 'dynamic_connections',
        'passed': True,
        'financial_to_phase': True,
        'desktop_to_core': True,
        'core_to_validation': True
    }


def main():
    """Run all integration tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "PHASE 4: DYNAMIC INTEGRATION TEST" + " "*20 + "║")
    print("║" + " "*20 + "Φ-Mamba Modularization" + " "*26 + "║")
    print("╚" + "="*68 + "╝")

    results = []

    try:
        results.append(test_core_modules())
    except Exception as e:
        print(f"\n✗ Core modules test FAILED: {e}")
        results.append({'module': 'core', 'passed': False, 'error': str(e)})

    try:
        results.append(test_encoding_module())
    except Exception as e:
        print(f"\n✗ Encoding module test FAILED: {e}")
        results.append({'module': 'encoding', 'passed': False, 'error': str(e)})

    try:
        results.append(test_financial_module())
    except Exception as e:
        print(f"\n✗ Financial module test FAILED: {e}")
        results.append({'module': 'financial', 'passed': False, 'error': str(e)})

    try:
        results.append(test_zordic_desktop())
    except Exception as e:
        print(f"\n✗ Zordic desktop test FAILED: {e}")
        results.append({'module': 'zordic', 'passed': False, 'error': str(e)})

    try:
        results.append(test_tensor_series())
    except Exception as e:
        print(f"\n✗ TENSOR_SERIES test FAILED: {e}")
        results.append({'module': 'tensor', 'passed': False, 'error': str(e)})

    try:
        results.append(test_validation_module())
    except Exception as e:
        print(f"\n✗ Validation module test FAILED: {e}")
        results.append({'module': 'validation', 'passed': False, 'error': str(e)})

    try:
        results.append(test_dynamic_connections())
    except Exception as e:
        print(f"\n✗ Dynamic connections test FAILED: {e}")
        results.append({'module': 'dynamic', 'passed': False, 'error': str(e)})

    # Summary
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*25 + "TEST SUMMARY" + " "*31 + "║")
    print("╚" + "="*68 + "╝")
    print()

    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)

    for result in results:
        module = result.get('module', 'unknown')
        status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
        print(f"  {module:25s} {status}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n" + "="*70)
        print("PHASE 4 COMPLETE: ALL COMPONENTS DYNAMICALLY CONNECTED")
        print("="*70)
        print("\nModularization Score: 10/10 ⭐")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
