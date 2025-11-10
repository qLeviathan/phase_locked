//! # φ-Core: Latent n Manifold
//!
//! **Everything is logarithmic in φ-space.**
//!
//! This library implements computation in the natural representation of information growth:
//! - n isn't just an index - it's a compressed manifold of information
//! - All operations are O(1) integer lookups (no floating point)
//! - Memory, time, energy, phase - all encoded in a single integer n
//! - Natural boundaries at Lucas numbers provide automatic stopping conditions
//!
//! ## Core Insight
//!
//! ```text
//! Traditional: multiply(a, b) = a × b        [slow, floating point]
//! φ-space:     add(n_a, n_b) = n_a + n_b    [fast, integer]
//!
//! Because: φⁿ × φᵐ = φⁿ⁺ᵐ
//! And:     F[n+m] ≈ φⁿ⁺ᵐ (exact via convolution)
//! ```
//!
//! ## The n-Encoding Theorem
//!
//! A single integer n encodes:
//! - **Energy**: F_n (Fibonacci number)
//! - **Time**: L_n (Lucas number)
//! - **Address**: Zeckendorf bit pattern
//! - **Errors**: Gaps in Zeckendorf representation (Betti numbers)
//! - **Phase**: (-1)^n (Cassini identity)
//!
//! ## Modules
//!
//! - `latent_n`: The LatentN structure with full decoding
//! - `phi_arithmetic`: Logarithmic operations (addition = multiplication)
//! - `boundary`: Boundary-first puzzle solving (φ forward, ψ backward)
//! - `memory`: Base-φ memory allocation
//! - `maximal`: Detection of natural completion points
//! - `zeckendorf`: Decomposition engine (the program IS the decomposition)
//! - `token_stream`: Generation with Lucas stopping conditions

pub mod latent_n;
pub mod phi_arithmetic;
pub mod boundary;
pub mod memory;
pub mod maximal;
pub mod zeckendorf;
pub mod token_stream;

// ============================================================================
// Precomputed Constants - The Oracle Tables
// ============================================================================

/// Fibonacci numbers F_n for n = 0..93 (max that fits in u64)
/// F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
pub const FIBONACCI: [u64; 93] = generate_fibonacci();

/// Lucas numbers L_n for n = 0..92
/// L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}
pub const LUCAS: [u64; 92] = generate_lucas();

/// Maximal n indices where Lucas-Fibonacci properties align
/// These are natural "rest points" where computation completes
pub const MAXIMAL_N: [usize; 7] = [3, 4, 7, 11, 18, 29, 47];

// ============================================================================
// Constant Generation (compile-time)
// ============================================================================

const fn generate_fibonacci() -> [u64; 93] {
    let mut fib = [0u64; 93];
    fib[0] = 0;
    fib[1] = 1;

    let mut i = 2;
    while i < 93 {
        fib[i] = fib[i - 1].saturating_add(fib[i - 2]);
        i += 1;
    }

    fib
}

const fn generate_lucas() -> [u64; 92] {
    let mut lucas = [0u64; 92];
    lucas[0] = 2;
    lucas[1] = 1;

    let mut i = 2;
    while i < 92 {
        lucas[i] = lucas[i - 1].saturating_add(lucas[i - 2]);
        i += 1;
    }

    lucas
}

// ============================================================================
// Core Utilities
// ============================================================================

/// Golden ratio φ = (1 + √5) / 2 ≈ 1.618...
/// Only use for display/debugging - all computation uses integers
pub const PHI: f64 = 1.618033988749895;

/// Golden conjugate ψ = (1 - √5) / 2 ≈ -0.618...
/// Only use for display/debugging - all computation uses integers
pub const PSI: f64 = -0.618033988749895;

/// Check if n is valid index (< 93 for Fibonacci, < 92 for Lucas)
#[inline]
pub const fn is_valid_n(n: usize) -> bool {
    n < 93
}

/// Check if n is valid Lucas index
#[inline]
pub const fn is_valid_lucas_n(n: usize) -> bool {
    n < 92
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_sequence() {
        assert_eq!(FIBONACCI[0], 0);
        assert_eq!(FIBONACCI[1], 1);
        assert_eq!(FIBONACCI[2], 1);
        assert_eq!(FIBONACCI[3], 2);
        assert_eq!(FIBONACCI[4], 3);
        assert_eq!(FIBONACCI[5], 5);
        assert_eq!(FIBONACCI[6], 8);
        assert_eq!(FIBONACCI[10], 55);
        assert_eq!(FIBONACCI[20], 6765);
    }

    #[test]
    fn test_lucas_sequence() {
        assert_eq!(LUCAS[0], 2);
        assert_eq!(LUCAS[1], 1);
        assert_eq!(LUCAS[2], 3);
        assert_eq!(LUCAS[3], 4);
        assert_eq!(LUCAS[4], 7);
        assert_eq!(LUCAS[5], 11);
        assert_eq!(LUCAS[6], 18);
    }

    #[test]
    fn test_fibonacci_lucas_relation() {
        // L_n = F_{n-1} + F_{n+1}
        for n in 1..20 {
            assert_eq!(
                LUCAS[n],
                FIBONACCI[n - 1] + FIBONACCI[n + 1],
                "Lucas-Fibonacci relation failed at n={}", n
            );
        }
    }

    #[test]
    fn test_cassini_identity() {
        // F_{n-1} * F_{n+1} - F_n^2 = (-1)^n
        for n in 1..30 {
            let left = (FIBONACCI[n - 1] as i128) * (FIBONACCI[n + 1] as i128);
            let right = (FIBONACCI[n] as i128) * (FIBONACCI[n] as i128);
            let diff = left - right;
            let expected = if n % 2 == 0 { -1 } else { 1 };
            assert_eq!(diff, expected, "Cassini identity failed at n={}", n);
        }
    }
}
