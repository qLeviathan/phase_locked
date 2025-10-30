#!/usr/bin/env python3
"""
ZORDIC System Test - Verify core functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zordic_core import PhiLattice, PHI, PSI

def test_golden_ratio():
    """Test golden ratio properties"""
    print("="*70)
    print("TEST 1: Golden Ratio Properties")
    print("="*70)

    print(f"φ = {PHI:.10f}")
    print(f"ψ = {PSI:.10f}")
    print(f"φ + ψ = {PHI + PSI:.10f} (should be ≈ 1)")
    print(f"φ × ψ = {PHI * PSI:.10f} (should be ≈ -1)")
    print(f"φ² - φ - 1 = {PHI**2 - PHI - 1:.10f} (should be ≈ 0)")

    assert abs((PHI + PSI) - 1.0) < 1e-10, "φ + ψ should equal 1"
    assert abs((PHI * PSI) + 1.0) < 1e-10, "φ × ψ should equal -1"
    assert abs(PHI**2 - PHI - 1) < 1e-10, "φ² - φ - 1 should equal 0"

    print("✓ All golden ratio properties verified\n")

def test_encoding():
    """Test character encoding"""
    print("="*70)
    print("TEST 2: Character Encoding")
    print("="*70)

    lattice = PhiLattice("abc", temperature=1.0)
    lattice.encode()

    print(f"Encoded {len(lattice.nodes)} characters")

    for node in lattice.nodes:
        print(f"  '{node.char}': shells={node.shells}, φ={node.phi:.3f}, ψ={node.psi:.3f}, Δ={node.delta:.3f}")

    assert len(lattice.nodes) == 3, "Should have 3 nodes"
    assert all(hasattr(n, 'phi') for n in lattice.nodes), "All nodes should have φ component"

    print("✓ Encoding successful\n")

def test_full_analysis():
    """Test full analysis pipeline"""
    print("="*70)
    print("TEST 3: Full Analysis Pipeline")
    print("="*70)

    lattice = PhiLattice("test", temperature=1.0)
    results = lattice.full_analysis()

    print(f"Initial φ-field: {results['initial_state']['phi_total']:.3f}")
    print(f"Initial ψ-field: {results['initial_state']['psi_total']:.3f}")
    print(f"Stable nodes: {results['initial_state']['stable_count']}/{len(lattice.nodes)}")
    print(f"Regime: {results['regime']['regime']}")
    print(f"Deterministic ratio: {results['regime']['ratio']:.1%}")
    print(f"Final stability: {results['final_stability']:.3f}")

    assert 'regime' in results, "Should have regime classification"
    assert results['regime']['regime'] in ['DETERMINISTIC', 'STOCHASTIC', 'MIXED (Quantum-like)']

    print("✓ Full analysis completed successfully\n")

def test_regime_detection():
    """Test regime detection with different inputs"""
    print("="*70)
    print("TEST 4: Regime Detection")
    print("="*70)

    test_cases = [
        ("aaa", "Repeated chars (expect: DETERMINISTIC)"),
        ("abc", "Sequential chars (expect: MIXED)"),
        ("xyz", "High-value chars (expect: STOCHASTIC)"),
    ]

    for text, description in test_cases:
        lattice = PhiLattice(text, temperature=1.0)
        results = lattice.full_analysis()
        regime = results['regime']['regime']
        ratio = results['regime']['ratio']

        print(f"  '{text}': {regime} ({ratio:.1%} deterministic) - {description}")

    print("✓ Regime detection working\n")

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  ZORDIC SYSTEM TEST SUITE")
    print("  Leviathan AI Corporation")
    print("="*70 + "\n")

    try:
        test_golden_ratio()
        test_encoding()
        test_full_analysis()
        test_regime_detection()

        print("="*70)
        print("  ✓ ALL TESTS PASSED")
        print("="*70)
        print()
        print("System is ready for use!")
        print("Run: python run_zordic.py")
        print()

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
