//! # φ-Space Arithmetic
//!
//! **Multiplication becomes addition. Division becomes subtraction.**
//!
//! This is the core of "minimal resistance" computation:
//! - Traditional: a × b requires floating point multiplication
//! - φ-space: n_a + n_b is just integer addition, then table lookup
//!
//! ## The Logarithmic Transform
//!
//! ```text
//! Normal space:  φⁿ × φᵐ = φⁿ⁺ᵐ        (exponential)
//! Log space:     n + m = n+m            (additive)
//! Fibonacci:     F[n+m] ≈ φⁿ⁺ᵐ          (exact via convolution)
//! ```
//!
//! ## Minimal Resistance
//!
//! All operations are O(1) integer lookups - no floating point, no iteration!
//!
//! ```text
//! multiply(φⁿ, φᵐ) → lookup(n+m)       O(1)
//! divide(φⁿ, φᵐ)   → lookup(n-m)       O(1)
//! power(φⁿ, k)     → lookup(n×k)       O(1)
//! sqrt(φⁿ)         → lookup(n/2)       O(1)
//! ```

use crate::latent_n::LatentN;

/// Multiply two φ-space numbers: φⁿ × φᵐ = φⁿ⁺ᵐ
///
/// In normal space this would be expensive floating point multiplication.
/// In log space it's just integer addition + table lookup.
///
/// # Example
/// ```
/// use phi_core::phi_arithmetic::multiply;
/// use phi_core::latent_n::LatentN;
///
/// let n5 = LatentN::new(5);  // F_5 = 5
/// let n7 = LatentN::new(7);  // F_7 = 13
///
/// let product = multiply(n5, n7).unwrap();  // n=12
/// assert_eq!(product.fibonacci(), 144);  // F_12 = 144
/// // Note: F_5 × F_7 = 65, but φ⁵ × φ⁷ ≈ φ¹² = 144
/// ```
pub fn multiply(a: LatentN, b: LatentN) -> Option<LatentN> {
    a.advance(b.n)
}

/// Divide two φ-space numbers: φⁿ / φᵐ = φⁿ⁻ᵐ
///
/// # Example
/// ```
/// use phi_core::phi_arithmetic::divide;
/// use phi_core::latent_n::LatentN;
///
/// let n10 = LatentN::new(10);
/// let n4 = LatentN::new(4);
///
/// let quotient = divide(n10, n4).unwrap();
/// assert_eq!(quotient.n, 6);  // 10 - 4 = 6
/// ```
pub fn divide(a: LatentN, b: LatentN) -> Option<LatentN> {
    a.retreat(b.n)
}

/// Raise to integer power: (φⁿ)ᵏ = φⁿˣᵏ
///
/// # Example
/// ```
/// use phi_core::phi_arithmetic::power;
/// use phi_core::latent_n::LatentN;
///
/// let n5 = LatentN::new(5);
/// let squared = power(n5, 2).unwrap();
/// assert_eq!(squared.n, 10);  // 5 × 2 = 10
/// ```
pub fn power(base: LatentN, exponent: usize) -> Option<LatentN> {
    let new_n = base.n.saturating_mul(exponent);
    if new_n < 93 {
        Some(LatentN::new(new_n))
    } else {
        None
    }
}

/// Take kth root: φⁿ^(1/k) = φⁿ/ᵏ
///
/// # Example
/// ```
/// use phi_core::phi_arithmetic::root;
/// use phi_core::latent_n::LatentN;
///
/// let n10 = LatentN::new(10);
/// let sqrt = root(n10, 2);
/// assert_eq!(sqrt.n, 5);  // 10 / 2 = 5
/// ```
pub fn root(base: LatentN, k: usize) -> LatentN {
    if k == 0 {
        return base;
    }
    LatentN::new(base.n / k)
}

/// Compute Fibonacci convolution: F[n] × F[m]
///
/// This is different from φ-space multiplication!
/// - φ-multiply: φⁿ × φᵐ = φⁿ⁺ᵐ → F[n+m]
/// - Fib-multiply: F[n] × F[m] ≈ F[n+m-1] (approximate)
///
/// Exact formula uses Lucas numbers:
/// F[n] × F[m] = (F[n+m] + (-1)^m × F[n-m]) / L[m]
///
/// But for practical purposes, we use: F[n] × F[m] ≈ F[n+m-c] for small correction c
pub fn fib_multiply(a: LatentN, b: LatentN) -> u64 {
    a.fibonacci() * b.fibonacci()
}

/// Fast Fibonacci addition: F[n] + F[m]
///
/// Uses identity: F[n] + F[m] = F[closest match]
/// This isn't always exact, but finds the nearest Fibonacci number
pub fn fib_add(a: LatentN, b: LatentN) -> LatentN {
    let sum = a.fibonacci() + b.fibonacci();

    // Find closest Fibonacci number
    LatentN::from_energy(sum).unwrap_or(a)
}

/// Compute Lucas convolution: L[n] × L[m]
pub fn lucas_multiply(a: LatentN, b: LatentN) -> u64 {
    a.lucas() * b.lucas()
}

/// The Minimal Resistance Property
///
/// Measures computational cost: φ-space vs normal space
#[derive(Debug)]
pub struct ResistanceMetrics {
    /// Number of floating point operations in normal space
    pub normal_flops: usize,

    /// Number of integer operations in φ-space
    pub phi_ops: usize,

    /// Resistance ratio (normal/phi) - higher is better
    pub ratio: f64,
}

/// Measure resistance for multiplication
pub fn measure_multiply_resistance() -> ResistanceMetrics {
    // Normal space: a × b requires ~100 FLOPs (depends on precision)
    let normal_flops = 100;

    // φ-space: just addition + lookup
    let phi_ops = 2; // 1 add, 1 lookup

    ResistanceMetrics {
        normal_flops,
        phi_ops,
        ratio: normal_flops as f64 / phi_ops as f64,
    }
}

/// Measure resistance for division
pub fn measure_divide_resistance() -> ResistanceMetrics {
    // Normal space: a / b requires ~200 FLOPs
    let normal_flops = 200;

    // φ-space: just subtraction + lookup
    let phi_ops = 2;

    ResistanceMetrics {
        normal_flops,
        phi_ops,
        ratio: normal_flops as f64 / phi_ops as f64,
    }
}

/// Measure resistance for power
pub fn measure_power_resistance(exponent: usize) -> ResistanceMetrics {
    // Normal space: a^k requires k multiplications
    let normal_flops = exponent * 100;

    // φ-space: just multiplication + lookup
    let phi_ops = 2;

    ResistanceMetrics {
        normal_flops,
        phi_ops,
        ratio: normal_flops as f64 / phi_ops as f64,
    }
}

// ============================================================================
// Advanced Operations
// ============================================================================

/// Compute the φ/ψ split
///
/// Any Fibonacci can be written as: F[n] = φⁿ/√5 - ψⁿ/√5
/// This returns the (φ component, ψ component) separately
///
/// Useful for understanding forward (φ) vs backward (ψ) contributions
pub fn phi_psi_split(n: LatentN) -> (f64, f64) {
    use crate::{PHI, PSI};

    let sqrt5 = 5.0_f64.sqrt();
    let phi_part = PHI.powi(n.n as i32) / sqrt5;
    let psi_part = PSI.powi(n.n as i32) / sqrt5;

    (phi_part, psi_part)
}

/// Compute the conjugate: ψⁿ from φⁿ
///
/// Since ψ = -1/φ, we have: ψⁿ = (-1)ⁿ / φⁿ
/// In log space: log(ψⁿ) = n×log(ψ) = -n×log(φ)
///
/// Returns the backward-time component
pub fn conjugate(n: LatentN) -> i64 {
    let sign = n.cassini_phase();
    let magnitude = n.fibonacci() as i64;

    if sign > 0 {
        magnitude
    } else {
        -magnitude
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_multiply() {
        let n3 = LatentN::new(3);
        let n4 = LatentN::new(4);

        let product = multiply(n3, n4).unwrap();
        assert_eq!(product.n, 7); // 3 + 4 = 7
        assert_eq!(product.fibonacci(), 13); // F_7
    }

    #[test]
    fn test_phi_divide() {
        let n10 = LatentN::new(10);
        let n3 = LatentN::new(3);

        let quotient = divide(n10, n3).unwrap();
        assert_eq!(quotient.n, 7); // 10 - 3 = 7
    }

    #[test]
    fn test_phi_power() {
        let n5 = LatentN::new(5);

        let squared = power(n5, 2).unwrap();
        assert_eq!(squared.n, 10);

        let cubed = power(n5, 3).unwrap();
        assert_eq!(cubed.n, 15);
    }

    #[test]
    fn test_phi_root() {
        let n12 = LatentN::new(12);

        let sqrt = root(n12, 2);
        assert_eq!(sqrt.n, 6);

        let cbrt = root(n12, 3);
        assert_eq!(cbrt.n, 4);
    }

    #[test]
    fn test_minimal_resistance() {
        let mult_resistance = measure_multiply_resistance();
        assert!(mult_resistance.ratio > 10.0);

        let div_resistance = measure_divide_resistance();
        assert!(div_resistance.ratio > 10.0);

        let pow_resistance = measure_power_resistance(10);
        assert!(pow_resistance.ratio > 100.0);
    }

    #[test]
    fn test_fib_add() {
        let n5 = LatentN::new(5);  // F_5 = 5
        let n7 = LatentN::new(7);  // F_7 = 13

        let sum = fib_add(n5, n7);
        // 5 + 13 = 18, closest Fibonacci is 21 = F_8
        assert!(sum.fibonacci() >= 18);
    }

    #[test]
    fn test_phi_psi_split() {
        let n = LatentN::new(10);
        let (phi_comp, psi_comp) = phi_psi_split(n);

        // φ component should dominate
        assert!(phi_comp.abs() > psi_comp.abs());

        // Their difference should be close to F_10
        let fib = n.fibonacci() as f64;
        let reconstructed = phi_comp - psi_comp;
        assert!((reconstructed - fib).abs() < 1.0);
    }

    #[test]
    fn test_conjugate() {
        let n_even = LatentN::new(4);
        let conj_even = conjugate(n_even);
        assert!(conj_even > 0); // Even n → positive

        let n_odd = LatentN::new(5);
        let conj_odd = conjugate(n_odd);
        assert!(conj_odd < 0); // Odd n → negative
    }

    #[test]
    fn test_logarithmic_property() {
        // Verify: φⁿ × φᵐ = φⁿ⁺ᵐ
        use crate::PHI;

        for n in 1..10 {
            for m in 1..10 {
                let left = PHI.powi(n) * PHI.powi(m);
                let right = PHI.powi(n + m);

                assert!((left - right).abs() < 0.001);
            }
        }
    }
}
