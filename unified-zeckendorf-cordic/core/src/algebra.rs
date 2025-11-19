//! # Algebraic Structures for Zeckendorf-CORDIC System
//!
//! Core algebraic types representing the mathematical foundation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// ZeckendorfField: Integer-only field representation
///
/// Every value v ∈ Z can be represented as:
/// ```text
/// v = (p/q) where p,q ∈ ℤ, gcd(p,q) = 1
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZeckendorfField {
    /// Numerator (always an integer)
    pub numerator: i64,
    /// Denominator (always positive, usually power of 2 for efficient division)
    pub denominator: i64,
}

impl ZeckendorfField {
    /// Create from integer
    pub fn from_integer(n: i64) -> Self {
        Self {
            numerator: n,
            denominator: 1,
        }
    }

    /// Create from rational (p/q)
    pub fn from_rational(p: i64, q: i64) -> Self {
        assert!(q > 0, "Denominator must be positive");
        let gcd = Self::gcd(p.abs(), q);
        Self {
            numerator: p / gcd,
            denominator: q / gcd,
        }
    }

    /// Check if this is a valid integer representation
    pub fn is_valid(&self) -> bool {
        self.denominator > 0 && Self::gcd(self.numerator.abs(), self.denominator) <= self.denominator
    }

    /// GCD via Euclidean algorithm (integer-only)
    pub fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Convert to f64 for display/testing only (NOT used in computations)
    pub fn to_f64_display(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

impl fmt::Display for ZeckendorfField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// LucasRing: Lucas numbers form a ring under our operations
///
/// For φ-based calculations:
/// ```text
/// φⁿ ≈ (Fₙ₊₁, Fₙ) in Q²
/// Lucas identity: Lₙ = Fₙ₊₁ + Fₙ₋₁
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LucasRing {
    /// The Lucas number value
    pub value: u64,
    /// Index in the Lucas sequence
    pub index: u32,
}

impl LucasRing {
    /// Create from index
    pub fn from_index(index: u32) -> Self {
        let value = crate::sequences::lucas(index as usize);
        Self { value, index }
    }

    /// Create from value (finds closest Lucas number)
    pub fn from_value(target: u64) -> Self {
        let mut index = 0;
        let mut prev = 2u64;
        let mut curr = 1u64;

        if target <= 2 {
            return Self { value: 2, index: 0 };
        }

        while curr < target {
            index += 1;
            let next = prev + curr;
            prev = curr;
            curr = next;
        }

        // Return closest
        if (curr - target) < (target - prev) {
            Self { value: curr, index: index + 1 }
        } else {
            Self { value: prev, index }
        }
    }

    /// Get corresponding Fibonacci numbers for φ representation
    /// φⁿ ≈ (F_{n+1}, F_n)
    pub fn to_phi_pair(&self) -> (u64, u64) {
        let f_n = crate::sequences::fibonacci(self.index as usize);
        let f_n_plus_1 = crate::sequences::fibonacci(self.index as usize + 1);
        (f_n_plus_1, f_n)
    }
}

/// BitLattice: Zeckendorf bit representation
///
/// Invariant: No adjacent 1s (Zeckendorf property)
/// κ(11) = 100 (cascade rule)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitLattice {
    /// Bit vector (true = 1, false = 0)
    pub bits: Vec<bool>,
}

impl BitLattice {
    /// Create empty lattice
    pub fn new() -> Self {
        Self { bits: Vec::new() }
    }

    /// Create from integer using Zeckendorf decomposition
    pub fn from_integer(n: u64) -> Self {
        let zeck = crate::sequences::zeckendorf_decomposition(n);
        let mut bits = Vec::new();

        if zeck.is_empty() || (zeck.len() == 1 && zeck[0] == 0) {
            return Self { bits: vec![false] };
        }

        // Find max Fibonacci index
        let max_fib = *zeck.last().unwrap();
        let mut max_idx = 0;
        for i in 0..64 {
            if crate::sequences::fibonacci(i) >= max_fib {
                max_idx = i;
                break;
            }
        }

        // Build bit vector
        for i in 2..=max_idx {
            let fib = crate::sequences::fibonacci(i);
            bits.push(zeck.contains(&fib));
        }

        bits.reverse();

        Self { bits }
    }

    /// Convert back to integer
    pub fn to_integer(&self) -> u64 {
        let mut sum = 0u64;
        let len = self.bits.len();

        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                let fib_idx = len - i + 1;
                sum += crate::sequences::fibonacci(fib_idx);
            }
        }

        sum
    }

    /// Check if valid Zeckendorf form (no adjacent 1s)
    pub fn is_valid(&self) -> bool {
        for i in 0..self.bits.len().saturating_sub(1) {
            if self.bits[i] && self.bits[i + 1] {
                return false; // Adjacent 1s
            }
        }
        true
    }

    /// Apply cascade operator κ to resolve adjacent 1s
    pub fn cascade(&mut self) {
        loop {
            let mut changed = false;

            for i in 0..self.bits.len().saturating_sub(1) {
                if self.bits[i] && self.bits[i + 1] {
                    // Found adjacent 1s: 11 → 100
                    self.bits[i] = false;
                    self.bits[i + 1] = false;

                    // Need to add 1 at position i+2
                    if i + 2 < self.bits.len() {
                        self.bits[i + 2] = !self.bits[i + 2];
                    } else {
                        self.bits.push(true);
                    }

                    changed = true;
                    break;
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// Count "holes" (zeros in the bit pattern)
    pub fn holes(&self) -> usize {
        self.bits.iter().filter(|&&b| !b).count()
    }
}

impl Default for BitLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BitLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &bit in &self.bits {
            write!(f, "{}", if bit { '1' } else { '0' })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeckendorf_field() {
        let x = ZeckendorfField::from_integer(5);
        assert_eq!(x.numerator, 5);
        assert_eq!(x.denominator, 1);
        assert!(x.is_valid());

        let y = ZeckendorfField::from_rational(6, 3);
        assert_eq!(y.numerator, 2);
        assert_eq!(y.denominator, 1);
    }

    #[test]
    fn test_lucas_ring() {
        let l = LucasRing::from_index(5);
        assert_eq!(l.value, 11); // L_5 = 11

        let phi_pair = l.to_phi_pair();
        assert_eq!(phi_pair, (8, 5)); // (F_6, F_5)
    }

    #[test]
    fn test_bit_lattice() {
        let lattice = BitLattice::from_integer(17);
        // 17 = 13 + 3 + 1 = F_7 + F_4 + F_2
        // Binary: 10100
        assert_eq!(lattice.to_string(), "10100");
        assert!(lattice.is_valid());

        assert_eq!(lattice.to_integer(), 17);
    }

    #[test]
    fn test_bit_cascade() {
        let mut lattice = BitLattice {
            bits: vec![true, true], // Invalid: 11
        };
        assert!(!lattice.is_valid());

        lattice.cascade();
        assert!(lattice.is_valid());
        // 11 → 100
        assert_eq!(lattice.to_string(), "100");
    }

    #[test]
    fn test_cascade_complex() {
        let mut lattice = BitLattice {
            bits: vec![true, true, true], // 111
        };
        lattice.cascade();
        assert!(lattice.is_valid());
        // 111 → 1001
        assert_eq!(lattice.to_string(), "1001");
    }
}
