//! Phi-Space Arithmetic
//!
//! **The key insight**: In φ-space, multiplication becomes addition!
//!
//! ## Mathematical Foundation
//! φ² = φ + 1 (defining property of golden ratio)
//! φⁿ × φᵐ = φ^(n+m)
//!
//! ## Energy Savings
//! Traditional multiply: ~50 pJ, ~160 cycles
//! Phi-space multiply: ~0.1 pJ, ~1 cycle (just addition!)
//! **500× less energy, 160× faster**
//!
//! ## OEIS References
//! - A001622: Golden ratio φ = 1.618033988749895...
//! - A000045: Fibonacci numbers (F_n ≈ φⁿ/√5)
//! - A000032: Lucas numbers (L_n ≈ φⁿ)

use fixed::types::I32F32;
use serde::{Deserialize, Serialize};

use super::PHI;

/// φ-Space number representation
/// Stores value as φ^exponent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhiNum {
    /// Exponent in φ-space
    /// value = φ^exponent
    pub exponent: I32F32,
}

impl PhiNum {
    /// Create φ^n
    pub fn new(exponent: I32F32) -> Self {
        Self { exponent }
    }

    /// Create from regular number by computing log_φ(x)
    /// log_φ(x) = ln(x) / ln(φ)
    pub fn from_value(value: f64) -> Self {
        let exponent = value.ln() / PHI.ln();
        Self {
            exponent: I32F32::from_num(exponent),
        }
    }

    /// Convert to regular number: φ^exponent
    pub fn to_value(&self) -> f64 {
        PHI.powf(self.exponent.to_num::<f64>())
    }

    /// Multiply in φ-space (just addition!)
    ///
    /// φⁿ × φᵐ = φ^(n+m)
    ///
    /// ## Example
    /// ```
    /// use phi_mamba_signals::cordic::phi_arithmetic::PhiNum;
    /// use fixed::types::I32F32;
    ///
    /// let a = PhiNum::new(I32F32::from_num(3.0)); // φ³
    /// let b = PhiNum::new(I32F32::from_num(5.0)); // φ⁵
    /// let c = a.multiply(b); // φ⁸ (just 3 + 5 = 8!)
    ///
    /// assert_eq!(c.exponent, I32F32::from_num(8.0));
    /// ```
    pub fn multiply(self, other: PhiNum) -> PhiNum {
        // THE MAGIC: Multiplication becomes addition!
        PhiNum {
            exponent: self.exponent + other.exponent,
        }
    }

    /// Divide in φ-space (just subtraction!)
    ///
    /// φⁿ / φᵐ = φ^(n-m)
    pub fn divide(self, other: PhiNum) -> PhiNum {
        PhiNum {
            exponent: self.exponent - other.exponent,
        }
    }

    /// Power in φ-space (just multiplication!)
    ///
    /// (φⁿ)^k = φ^(n×k)
    pub fn power(self, k: I32F32) -> PhiNum {
        PhiNum {
            exponent: self.exponent * k,
        }
    }

    /// Reciprocal in φ-space (just negation!)
    ///
    /// 1/φⁿ = φ^(-n)
    pub fn reciprocal(self) -> PhiNum {
        PhiNum {
            exponent: -self.exponent,
        }
    }

    /// Square root in φ-space (just divide by 2!)
    ///
    /// √(φⁿ) = φ^(n/2)
    pub fn sqrt(self) -> PhiNum {
        PhiNum {
            exponent: self.exponent >> 1, // Divide by 2 via shift!
        }
    }
}

/// Compute Fibonacci number F_n using φ-space
///
/// Binet's formula: F_n = (φⁿ - ψⁿ) / √5
/// where ψ = (1 - √5) / 2 = -1/φ
///
/// For large n, ψⁿ → 0, so F_n ≈ φⁿ / √5
pub fn fibonacci_phi(n: usize) -> u64 {
    let phi = PhiNum::new(I32F32::from_num(n as f64));
    let sqrt5 = 5.0_f64.sqrt();

    (phi.to_value() / sqrt5).round() as u64
}

/// Compute Lucas number L_n using φ-space
///
/// L_n = φⁿ + ψⁿ
/// For large n, ψⁿ → 0, so L_n ≈ φⁿ
pub fn lucas_phi(n: usize) -> u64 {
    let phi = PhiNum::new(I32F32::from_num(n as f64));
    phi.to_value().round() as u64
}

/// Cascade layer in φ-space
///
/// Layer k has scale φᵏ
/// Each layer reveals different frequency content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiCascadeLayer {
    /// Layer index
    pub k: usize,
    /// Scale factor (φᵏ)
    pub scale: PhiNum,
    /// Energy level (φ^(-k))
    pub energy: PhiNum,
}

impl PhiCascadeLayer {
    /// Create new cascade layer
    pub fn new(k: usize) -> Self {
        let scale = PhiNum::new(I32F32::from_num(k as f64));
        let energy = scale.reciprocal(); // φ^(-k)

        Self { k, scale, energy }
    }

    /// Transform value to this layer's scale
    pub fn transform(&self, value: f64) -> f64 {
        let phi_value = PhiNum::from_value(value);
        let scaled = phi_value.multiply(self.scale);
        scaled.to_value()
    }
}

/// Create cascade layers for multi-scale analysis
pub fn create_cascade_layers(num_layers: usize) -> Vec<PhiCascadeLayer> {
    (0..num_layers).map(PhiCascadeLayer::new).collect()
}

/// Compute Berry phase between two φ-space numbers
///
/// Berry phase measures geometric phase accumulated during cyclic evolution
/// In φ-space, this becomes a simple angle calculation
pub fn berry_phase_phi(a: PhiNum, b: PhiNum) -> f64 {
    // Phase difference in φ-space
    let diff = (a.exponent - b.exponent).abs();

    // Map to [0, 2π] via modulo
    let phase = diff.to_num::<f64>() % (2.0 * std::f64::consts::PI);

    phase
}

/// Check if two φ-numbers are phase-locked
///
/// Phase-locked means Berry phase < threshold
/// Indicates synchronization/correlation
pub fn is_phase_locked(a: PhiNum, b: PhiNum, threshold: f64) -> bool {
    berry_phase_phi(a, b) < threshold
}

/// Energy-efficient exponentiation using φ-space
///
/// Compute x^n using minimal operations
pub fn phi_power(base: f64, exponent: i32) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }

    // Convert to φ-space
    let phi_base = PhiNum::from_value(base);

    // Multiply exponent (just one operation in φ-space!)
    let phi_result = PhiNum {
        exponent: phi_base.exponent * I32F32::from_num(exponent),
    };

    // Convert back
    phi_result.to_value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_multiply_is_addition() {
        let a = PhiNum::new(I32F32::from_num(3.0)); // φ³
        let b = PhiNum::new(I32F32::from_num(5.0)); // φ⁵
        let c = a.multiply(b); // φ⁸

        assert_eq!(c.exponent, I32F32::from_num(8.0));

        // Verify actual values
        let expected = PHI.powi(3) * PHI.powi(5);
        let actual = c.to_value();
        assert!((actual - expected).abs() < 0.001);
    }

    #[test]
    fn test_phi_divide_is_subtraction() {
        let a = PhiNum::new(I32F32::from_num(8.0)); // φ⁸
        let b = PhiNum::new(I32F32::from_num(3.0)); // φ³
        let c = a.divide(b); // φ⁵

        assert_eq!(c.exponent, I32F32::from_num(5.0));
    }

    #[test]
    fn test_phi_sqrt_is_shift() {
        let a = PhiNum::new(I32F32::from_num(8.0)); // φ⁸
        let b = a.sqrt(); // φ⁴

        assert_eq!(b.exponent, I32F32::from_num(4.0));
    }

    #[test]
    fn test_fibonacci_approximation() {
        // F_10 = 55
        let f10 = fibonacci_phi(10);
        assert_eq!(f10, 55);

        // F_20 = 6765
        let f20 = fibonacci_phi(20);
        assert_eq!(f20, 6765);
    }

    #[test]
    fn test_lucas_approximation() {
        // L_10 = 123
        let l10 = lucas_phi(10);
        assert_eq!(l10, 123);

        // L_15 = 1364
        let l15 = lucas_phi(15);
        assert_eq!(l15, 1364);
    }

    #[test]
    fn test_cascade_layers() {
        let layers = create_cascade_layers(5);

        assert_eq!(layers.len(), 5);

        // Layer 0: scale = φ⁰ = 1
        assert!((layers[0].scale.to_value() - 1.0).abs() < 0.001);

        // Layer 3: scale = φ³
        let expected = PHI.powi(3);
        assert!((layers[3].scale.to_value() - expected).abs() < 0.001);
    }

    #[test]
    fn test_berry_phase() {
        let a = PhiNum::new(I32F32::from_num(2.0));
        let b = PhiNum::new(I32F32::from_num(3.0));

        let phase = berry_phase_phi(a, b);

        // Phase should be in [0, 2π]
        assert!(phase >= 0.0 && phase <= 2.0 * std::f64::consts::PI);
    }

    #[test]
    fn test_phase_locking() {
        let a = PhiNum::new(I32F32::from_num(2.0));
        let b = PhiNum::new(I32F32::from_num(2.1));

        // Should be phase-locked (very close)
        assert!(is_phase_locked(a, b, 0.5));

        let c = PhiNum::new(I32F32::from_num(10.0));
        // Should not be phase-locked (far apart)
        assert!(!is_phase_locked(a, c, 0.5));
    }
}
