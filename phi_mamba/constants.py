"""
Golden Ratio Constants for Φ-Mamba

This module provides a single source of truth for all mathematical constants
used throughout the phi-mamba system.

Mathematical Foundation:
    φ² = φ + 1    (Recursive definition)
    φ² - φ = 1    (Unity emerges from φ)

    φ = (1 + √5) / 2 ≈ 1.618033988749895
    ψ = -1/φ = (1 - √5) / 2 ≈ -0.618033988749895

Properties:
    - φ and ψ are roots of x² - x - 1 = 0
    - φ × ψ = -1
    - φ - ψ = √5
    - φ + ψ = 1
"""

from math import sqrt, log, pi

# ============================================================================
# PRIMARY CONSTANTS
# ============================================================================

# Golden ratio (φ) - The fundamental primitive
PHI = (1 + sqrt(5)) / 2  # ≈ 1.618033988749895

# Golden ratio conjugate (ψ)
PSI = -1 / PHI  # = (1 - sqrt(5)) / 2 ≈ -0.618033988749895

# Alternative PSI definition (mathematically equivalent)
# PSI_ALT = (1 - sqrt(5)) / 2  # Some modules may prefer this form

# Square root of 5
SQRT_5 = sqrt(5)  # ≈ 2.23606797749979

# Natural logarithm of φ
LN_PHI = log(PHI)  # ≈ 0.48121182505960347

# ============================================================================
# DERIVED CONSTANTS
# ============================================================================

# Recursive unity: φ² - φ = 1
# This is the foundation - unity is not primitive but emerges from φ
UNITY = PHI**2 - PHI  # Should equal 1.0 exactly (within floating point precision)

# φ squared
PHI_SQUARED = PHI**2  # = φ + 1 ≈ 2.618033988749895

# Inverse of φ
INV_PHI = 1 / PHI  # = φ - 1 ≈ 0.618033988749895

# ============================================================================
# ANGULAR CONSTANTS
# ============================================================================

# Full circle in radians
TWO_PI = 2 * pi  # ≈ 6.283185307179586

# Pentagon interior angle (related to φ)
PENTAGON_ANGLE = 3 * pi / 5  # 108 degrees

# Golden angle (related to Fibonacci spirals)
GOLDEN_ANGLE = 2 * pi / (PHI**2)  # ≈ 2.399963 radians ≈ 137.5 degrees

# ============================================================================
# NUMERICAL TOLERANCES
# ============================================================================

# Default tolerance for phase locking checks
DEFAULT_PHASE_TOLERANCE = 0.5  # radians

# Tolerance for floating point comparisons
EPSILON = 1e-10

# ============================================================================
# VALIDATION
# ============================================================================

def validate_constants():
    """
    Verify mathematical relationships between constants.

    This ensures numerical precision is maintained and all
    relationships hold within tolerance.

    Returns:
        True if all validations pass

    Raises:
        AssertionError if any validation fails
    """
    # Verify recursive unity: φ² - φ = 1
    unity_check = abs(UNITY - 1.0)
    assert unity_check < EPSILON, f"Recursive unity failed: φ² - φ = {UNITY}, error = {unity_check}"

    # Verify φ satisfies x² - x - 1 = 0
    phi_equation = PHI**2 - PHI - 1
    assert abs(phi_equation) < EPSILON, f"φ equation failed: φ² - φ - 1 = {phi_equation}"

    # Verify ψ satisfies x² - x - 1 = 0
    psi_equation = PSI**2 - PSI - 1
    assert abs(psi_equation) < EPSILON, f"ψ equation failed: ψ² - ψ - 1 = {psi_equation}"

    # Verify φ × ψ = -1
    product = PHI * PSI
    assert abs(product + 1.0) < EPSILON, f"φ × ψ ≠ -1: got {product}"

    # Verify φ - ψ = √5
    diff = PHI - PSI
    assert abs(diff - SQRT_5) < EPSILON, f"φ - ψ ≠ √5: got {diff}"

    # Verify φ + ψ = 1
    sum_check = PHI + PSI
    assert abs(sum_check - 1.0) < EPSILON, f"φ + ψ ≠ 1: got {sum_check}"

    # Verify φ² = φ + 1
    phi_squared_check = abs(PHI_SQUARED - (PHI + 1))
    assert phi_squared_check < EPSILON, f"φ² ≠ φ + 1: error = {phi_squared_check}"

    # Verify 1/φ = φ - 1
    inv_phi_check = abs(INV_PHI - (PHI - 1))
    assert inv_phi_check < EPSILON, f"1/φ ≠ φ - 1: error = {inv_phi_check}"

    return True


# Auto-validate on import (can be disabled if needed)
if __name__ != "__main__":
    validate_constants()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GOLDEN RATIO CONSTANTS - Φ-Mamba")
    print("=" * 70)
    print()
    print("PRIMARY CONSTANTS:")
    print(f"  φ (PHI)     = {PHI:.15f}")
    print(f"  ψ (PSI)     = {PSI:.15f}")
    print(f"  √5 (SQRT_5) = {SQRT_5:.15f}")
    print(f"  ln(φ)       = {LN_PHI:.15f}")
    print()
    print("DERIVED CONSTANTS:")
    print(f"  φ²          = {PHI_SQUARED:.15f}")
    print(f"  1/φ         = {INV_PHI:.15f}")
    print(f"  UNITY       = {UNITY:.15f}")
    print()
    print("VALIDATION:")
    print(f"  φ² - φ - 1  = {PHI**2 - PHI - 1:.2e}")
    print(f"  ψ² - ψ - 1  = {PSI**2 - PSI - 1:.2e}")
    print(f"  φ × ψ       = {PHI * PSI:.15f} (should be -1)")
    print(f"  φ - ψ       = {PHI - PSI:.15f} (should be √5)")
    print(f"  φ + ψ       = {PHI + PSI:.15f} (should be 1)")
    print()

    try:
        validate_constants()
        print("✓ All constant validations passed!")
    except AssertionError as e:
        print(f"✗ Validation failed: {e}")

    print()
    print("=" * 70)
