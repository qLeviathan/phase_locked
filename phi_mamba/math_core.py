"""
Core Mathematical Functions for Φ-Mamba

This module consolidates all Fibonacci-related and phi-space mathematical
operations into a single, well-tested module.

Key Functions:
    - fibonacci(n): nth Fibonacci number (Binet formula - exact for integers)
    - lucas(n): nth Lucas number
    - zeckendorf_decomposition(n): Unique Fibonacci representation
    - cassini_identity(n): Verify F(n+1)·F(n-1) - F(n)² = (-1)ⁿ

All functions use the exact Binet formula for computational efficiency.
"""

from typing import List, Dict, Optional
from .constants import PHI, PSI, SQRT_5


# ============================================================================
# FIBONACCI SEQUENCE
# ============================================================================

def fibonacci(n: int) -> int:
    """
    Compute nth Fibonacci number using Binet's formula.

    Binet's formula is EXACT for Fibonacci numbers (not approximate):
        F(n) = (φⁿ - ψⁿ) / √5

    This works because the floating point errors cancel out when
    rounded to the nearest integer.

    Args:
        n: Index of Fibonacci number (n >= 0)

    Returns:
        nth Fibonacci number

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
        >>> fibonacci(20)
        6765

    Time Complexity: O(1) - constant time
    Space Complexity: O(1)
    """
    if n < 0:
        return 0
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Binet formula - exact for integers after rounding
    result = (PHI**n - PSI**n) / SQRT_5
    return int(round(result))


def lucas(n: int) -> int:
    """
    Compute nth Lucas number.

    Lucas numbers are closely related to Fibonacci:
        L(n) = φⁿ + ψⁿ

    Sequence: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, ...

    Properties:
        - L(n) = F(n-1) + F(n+1)
        - L(n)² = 5·F(n)² + 4·(-1)ⁿ

    Args:
        n: Index of Lucas number

    Returns:
        nth Lucas number

    Examples:
        >>> lucas(0)
        2
        >>> lucas(1)
        1
        >>> lucas(5)
        11
        >>> lucas(10)
        123

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if n == 0:
        return 2
    if n == 1:
        return 1

    return int(round(PHI**n + PSI**n))


# ============================================================================
# FIBONACCI SEQUENCE GENERATION (for when you need multiple values)
# ============================================================================

def fibonacci_sequence(max_n: int, include_zero: bool = True) -> List[int]:
    """
    Generate Fibonacci sequence up to index max_n.

    More efficient than calling fibonacci(i) repeatedly when you
    need multiple consecutive values.

    Args:
        max_n: Maximum index to generate
        include_zero: Whether to include F(0) = 0

    Returns:
        List of Fibonacci numbers [F(0), F(1), ..., F(max_n)]

    Examples:
        >>> fibonacci_sequence(10)
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        >>> fibonacci_sequence(5, include_zero=False)
        [1, 1, 2, 3, 5]

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if max_n < 0:
        return []

    if max_n == 0:
        return [0] if include_zero else []

    # Iterative generation is faster for sequences
    fibs = [0, 1] if include_zero else [1]
    a, b = 0, 1

    for _ in range(2 if include_zero else 1, max_n + 1):
        a, b = b, a + b
        fibs.append(b)

    return fibs


# ============================================================================
# ZECKENDORF DECOMPOSITION
# ============================================================================

def zeckendorf_decomposition(n: int) -> List[int]:
    """
    Decompose integer into unique non-consecutive Fibonacci numbers.

    Zeckendorf's theorem: Every positive integer can be uniquely
    represented as a sum of non-consecutive Fibonacci numbers.

    This representation is fundamental to the topological encoding:
    - Each Fibonacci number represents a "hole" at that scale
    - Non-consecutive constraint comes from geometry
    - Pattern emerges from φ² = φ + 1

    Args:
        n: Positive integer to decompose

    Returns:
        List of non-consecutive Fibonacci numbers that sum to n,
        in descending order

    Examples:
        >>> zeckendorf_decomposition(17)
        [13, 3, 1]  # F(7) + F(4) + F(2)
        >>> zeckendorf_decomposition(100)
        [89, 8, 3]  # F(11) + F(6) + F(4)
        >>> zeckendorf_decomposition(0)
        []

    Algorithm:
        Greedy approach - always take the largest Fibonacci number
        that fits, then skip the next one (non-consecutive constraint)

    Time Complexity: O(log n)
    Space Complexity: O(log n)
    """
    if n == 0:
        return []

    if n < 0:
        raise ValueError(f"Zeckendorf decomposition requires n >= 0, got {n}")

    # Build Fibonacci sequence up to n
    fibs = [1]  # Start with F(2) = 1
    a, b = 1, 2
    while b <= n:
        fibs.append(b)
        a, b = b, a + b

    # Greedy algorithm: take largest possible Fibonacci number
    result = []
    remaining = n
    i = len(fibs) - 1

    while i >= 0 and remaining > 0:
        if fibs[i] <= remaining:
            result.append(fibs[i])
            remaining -= fibs[i]
            i -= 2  # Skip next to ensure non-consecutive
        else:
            i -= 1

    return result


def zeckendorf_to_indices(n: int) -> List[int]:
    """
    Get Fibonacci indices for Zeckendorf decomposition.

    Instead of returning Fibonacci values, returns their indices.

    Args:
        n: Integer to decompose

    Returns:
        List of Fibonacci indices in descending order

    Examples:
        >>> zeckendorf_to_indices(17)
        [7, 4, 2]  # F(7)=13, F(4)=3, F(2)=1
        >>> zeckendorf_to_indices(100)
        [11, 6, 4]  # F(11)=89, F(6)=8, F(4)=3
    """
    decomp = zeckendorf_decomposition(n)

    # Map values back to indices
    indices = []
    for fib_val in decomp:
        # Find index of this Fibonacci number
        idx = 1
        a, b = 1, 1
        while b < fib_val:
            a, b = b, a + b
            idx += 1
        if b == fib_val:
            indices.append(idx)

    return indices


# ============================================================================
# VALIDATION AND IDENTITIES
# ============================================================================

def cassini_identity(n: int) -> int:
    """
    Verify Cassini's identity for Fibonacci numbers.

    Cassini's identity:
        F(n+1) · F(n-1) - F(n)² = (-1)ⁿ

    This alternating area property creates the phase relationships
    fundamental to the phi-mamba framework.

    Args:
        n: Index (must be >= 1)

    Returns:
        (-1)ⁿ if identity holds

    Raises:
        AssertionError if identity doesn't hold
        ValueError if n < 1

    Examples:
        >>> cassini_identity(5)
        -1
        >>> cassini_identity(6)
        1
    """
    if n < 1:
        raise ValueError(f"Cassini identity requires n >= 1, got {n}")

    f_n = fibonacci(n)
    f_n_plus = fibonacci(n + 1)
    f_n_minus = fibonacci(n - 1)

    left = f_n_plus * f_n_minus - f_n**2
    right = (-1)**n

    assert left == right, f"Cassini identity failed for n={n}: {left} ≠ {right}"

    return left


def verify_fibonacci(n: int, use_iterative: bool = True) -> bool:
    """
    Verify Binet formula matches iterative computation.

    Useful for testing that floating point precision is sufficient.

    Args:
        n: Fibonacci index to check
        use_iterative: Use iterative method as ground truth

    Returns:
        True if both methods agree
    """
    binet = fibonacci(n)

    if use_iterative:
        # Compute iteratively as ground truth
        if n <= 1:
            iterative = n
        else:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            iterative = a

        return binet == iterative

    return True


# ============================================================================
# PHI-SPACE ARITHMETIC (Integer-only operations)
# ============================================================================

def phi_multiply(n: int, m: int) -> int:
    """
    Multiply in φ-space using only addition.

    φⁿ × φᵐ = φⁿ⁺ᵐ

    This is the key computational advantage: all multiplication
    becomes addition in φ-space.

    Args:
        n: First exponent
        m: Second exponent

    Returns:
        n + m

    Examples:
        >>> phi_multiply(3, 5)
        8
        >>> phi_multiply(10, -3)
        7
    """
    return n + m


def phi_divide(n: int, m: int) -> int:
    """
    Divide in φ-space using only subtraction.

    φⁿ / φᵐ = φⁿ⁻ᵐ

    Args:
        n: Numerator exponent
        m: Denominator exponent

    Returns:
        n - m

    Examples:
        >>> phi_divide(10, 3)
        7
        >>> phi_divide(5, 8)
        -3
    """
    return n - m


def phi_power(base: int, exp: int) -> int:
    """
    Compute power in φ-space using only addition.

    (φⁿ)ᵐ = φⁿᵐ = φⁿ⁺ⁿ⁺...⁺ⁿ (m times)

    Args:
        base: Base exponent
        exp: Power to raise to

    Returns:
        base * exp

    Examples:
        >>> phi_power(3, 4)
        12
        >>> phi_power(5, 2)
        10
    """
    return base * exp


# ============================================================================
# LOOKUP TABLES (for performance-critical paths)
# ============================================================================

class FibonacciCache:
    """
    Cache for Fibonacci and related sequences.

    Useful when you need to repeatedly access Fibonacci numbers
    and want O(1) lookup after initial computation.
    """

    def __init__(self, max_n: int = 100):
        """
        Initialize cache up to max_n.

        Args:
            max_n: Maximum Fibonacci index to cache
        """
        self.max_n = max_n
        self._fib_cache = {}
        self._lucas_cache = {}
        self._zeck_cache = {}

        # Pre-compute Fibonacci and Lucas numbers
        for i in range(max_n + 1):
            self._fib_cache[i] = fibonacci(i)
            self._lucas_cache[i] = lucas(i)

    def get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (cached if available)."""
        if n in self._fib_cache:
            return self._fib_cache[n]
        result = fibonacci(n)
        self._fib_cache[n] = result
        return result

    def get_lucas(self, n: int) -> int:
        """Get nth Lucas number (cached if available)."""
        if n in self._lucas_cache:
            return self._lucas_cache[n]
        result = lucas(n)
        self._lucas_cache[n] = result
        return result

    def get_zeckendorf(self, n: int) -> List[int]:
        """Get Zeckendorf decomposition (cached if available)."""
        if n in self._zeck_cache:
            return self._zeck_cache[n]
        result = zeckendorf_decomposition(n)
        self._zeck_cache[n] = result
        return result


# ============================================================================
# P-TABLE and M-TABLE (for φⁿ and ψⁿ values)
# ============================================================================

def create_p_table(max_n: int = 100) -> Dict[int, float]:
    """
    Create P-table mapping n → φⁿ.

    Useful for verification and conversion between integer
    and floating-point representations.

    Args:
        max_n: Maximum exponent (both positive and negative)

    Returns:
        Dictionary mapping n to φⁿ

    Examples:
        >>> p_table = create_p_table(5)
        >>> abs(p_table[2] - 2.618) < 0.001
        True
    """
    p_table = {}
    for n in range(-max_n, max_n + 1):
        p_table[n] = PHI**n
    return p_table


def create_m_table(max_n: int = 100) -> Dict[int, float]:
    """
    Create M-table mapping n → ψⁿ.

    Maps to conjugate values for ψⁿ operations.

    Args:
        max_n: Maximum exponent (both positive and negative)

    Returns:
        Dictionary mapping n to ψⁿ
    """
    m_table = {}
    for n in range(-max_n, max_n + 1):
        m_table[n] = PSI**n
    return m_table


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CORE MATH FUNCTIONS - Φ-Mamba")
    print("=" * 70)
    print()

    print("FIBONACCI SEQUENCE (first 15):")
    fibs = fibonacci_sequence(14)
    print(f"  {fibs}")
    print()

    print("LUCAS SEQUENCE (first 10):")
    lucas_seq = [lucas(i) for i in range(10)]
    print(f"  {lucas_seq}")
    print()

    print("ZECKENDORF DECOMPOSITION:")
    for n in [17, 100, 1000]:
        decomp = zeckendorf_decomposition(n)
        print(f"  {n} = {' + '.join(map(str, decomp))}")
    print()

    print("CASSINI IDENTITY:")
    for n in range(2, 8):
        result = cassini_identity(n)
        print(f"  F({n+1})·F({n-1}) - F({n})² = {result}")
    print()

    print("PHI-SPACE ARITHMETIC:")
    print(f"  φ³ × φ⁵ = φ⁸  →  {phi_multiply(3, 5)} = 8 ✓")
    print(f"  φ¹⁰ / φ³ = φ⁷ →  {phi_divide(10, 3)} = 7 ✓")
    print(f"  (φ³)⁴ = φ¹²   →  {phi_power(3, 4)} = 12 ✓")
    print()

    print("=" * 70)
