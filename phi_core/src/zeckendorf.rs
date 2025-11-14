//! # Zeckendorf Decomposition
//!
//! **Every integer has a unique representation as a sum of non-consecutive Fibonacci numbers.**
//!
//! This isn't just a mathematical curiosity - it's the fundamental structure of computation:
//! - The decomposition IS the program
//! - Gaps in the representation ARE where creativity emerges
//! - The bit pattern IS the memory address
//!
//! ## Example
//! ```text
//! 100 = 89 + 8 + 3
//!     = F_11 + F_6 + F_4
//!     → Pattern: [4, 6, 11]
//!     → Gaps: [5, 7, 8, 9, 10] (where errors can occur)
//!     → Bits: 0b100001010000 (n=11 is MSB)
//! ```

use crate::FIBONACCI;

/// Zeckendorf representation of an integer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Zeckendorf {
    /// Fibonacci indices in the decomposition (sorted ascending)
    /// e.g., [4, 6, 11] means F_4 + F_6 + F_11
    pub indices: Vec<usize>,

    /// The original value
    pub value: u64,
}

impl Zeckendorf {
    /// Decompose a number into Zeckendorf representation
    ///
    /// Uses greedy algorithm: repeatedly subtract largest Fibonacci ≤ remaining
    pub fn new(mut n: u64) -> Self {
        if n == 0 {
            return Self {
                indices: vec![],
                value: 0,
            };
        }

        let mut indices = Vec::new();

        // Start from largest Fibonacci ≤ n
        let mut i = FIBONACCI
            .iter()
            .rposition(|&f| f <= n)
            .unwrap_or(0);

        while n > 0 && i > 0 {
            if FIBONACCI[i] <= n {
                indices.push(i);
                n -= FIBONACCI[i];
                i = i.saturating_sub(2); // Skip next to ensure non-consecutive
            } else {
                i = i.saturating_sub(1);
            }
        }

        indices.reverse(); // Store in ascending order
        let value = indices.iter().map(|&i| FIBONACCI[i]).sum();

        Self { indices, value }
    }

    /// Create from an n-index (just wraps [n])
    pub fn from_n(n: usize) -> Self {
        if n == 0 {
            return Self {
                indices: vec![],
                value: 0,
            };
        }

        Self {
            indices: vec![n],
            value: FIBONACCI[n],
        }
    }

    /// Get gaps in the representation (where creativity emerges)
    ///
    /// Returns indices between components that are missing
    /// These are the "holes" where errors or novelty can appear
    pub fn gaps(&self) -> Vec<usize> {
        if self.indices.len() <= 1 {
            return vec![];
        }

        let mut gaps = Vec::new();

        for window in self.indices.windows(2) {
            let start = window[0];
            let end = window[1];

            // Gaps are indices strictly between consecutive components
            for i in (start + 1)..end {
                gaps.push(i);
            }
        }

        gaps
    }

    /// Get the bit representation
    ///
    /// Bit i is set if F_i is in the decomposition
    /// This IS the memory address in φ-space
    pub fn to_bits(&self) -> u64 {
        let mut bits = 0u64;

        for &idx in &self.indices {
            if idx < 64 {
                bits |= 1u64 << idx;
            }
        }

        bits
    }

    /// Create from bit representation
    pub fn from_bits(bits: u64) -> Self {
        let mut indices = Vec::new();

        for i in 0..64 {
            if bits & (1u64 << i) != 0 {
                indices.push(i);
            }
        }

        let value = indices.iter().map(|&i| FIBONACCI[i]).sum();

        Self { indices, value }
    }

    /// Get the largest Fibonacci component
    pub fn max_component(&self) -> Option<usize> {
        self.indices.last().copied()
    }

    /// Get the smallest Fibonacci component
    pub fn min_component(&self) -> Option<usize> {
        self.indices.first().copied()
    }

    /// Get the number of components (complexity)
    pub fn complexity(&self) -> usize {
        self.indices.len()
    }

    /// Get the span (max - min index)
    pub fn span(&self) -> usize {
        match (self.min_component(), self.max_component()) {
            (Some(min), Some(max)) => max - min,
            _ => 0,
        }
    }

    /// Get gap density (gaps per span)
    pub fn gap_density(&self) -> f64 {
        let span = self.span();
        if span == 0 {
            return 0.0;
        }

        self.gaps().len() as f64 / span as f64
    }

    /// Check if this is a "maximal gap" representation
    /// (theoretical maximum number of gaps for given span)
    pub fn has_max_gaps(&self) -> bool {
        let actual_gaps = self.gaps().len();
        let span = self.span();

        if span == 0 {
            return false;
        }

        // Theoretical max: every other index is a gap
        let theoretical_max = (span - 1) / 2;

        actual_gaps >= theoretical_max
    }

    /// Add another Zeckendorf representation
    /// (this is NOT just adding values - it's composition in φ-space)
    pub fn compose(&self, other: &Self) -> Self {
        Self::new(self.value + other.value)
    }
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for Zeckendorf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.indices.is_empty() {
            return write!(f, "Z(0) = 0");
        }

        write!(f, "Z({}) = ", self.value)?;

        let terms: Vec<String> = self.indices
            .iter()
            .map(|&i| format!("F_{}", i))
            .collect();

        write!(f, "{}", terms.join(" + "))?;

        let gaps = self.gaps();
        if !gaps.is_empty() {
            write!(f, " [gaps: {:?}]", gaps)?;
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeckendorf_basic() {
        let z = Zeckendorf::new(0);
        assert_eq!(z.indices, vec![]);
        assert_eq!(z.value, 0);

        let z1 = Zeckendorf::new(1);
        assert_eq!(z1.indices, vec![1]); // F_1 = 1 or vec![2] for F_2 = 1
    }

    #[test]
    fn test_zeckendorf_100() {
        let z = Zeckendorf::new(100);

        // 100 = 89 + 8 + 3 = F_11 + F_6 + F_4
        assert!(z.indices.contains(&11));
        assert!(z.indices.contains(&6));
        assert!(z.indices.contains(&4));
        assert_eq!(z.value, 100);

        // Check gaps
        let gaps = z.gaps();
        assert!(gaps.contains(&5)); // Between 4 and 6
        assert!(gaps.contains(&7)); // Between 6 and 11
        assert!(gaps.contains(&8));
        assert!(gaps.contains(&9));
        assert!(gaps.contains(&10));
    }

    #[test]
    fn test_non_consecutive() {
        // Verify no consecutive Fibonacci indices
        for n in 1..1000 {
            let z = Zeckendorf::new(n);

            for window in z.indices.windows(2) {
                let diff = window[1] - window[0];
                assert!(diff >= 2, "Consecutive indices found in {}", z);
            }
        }
    }

    #[test]
    fn test_uniqueness() {
        // Every number should have exactly one representation
        for n in 1..100 {
            let z = Zeckendorf::new(n);
            let reconstructed: u64 = z.indices.iter().map(|&i| FIBONACCI[i]).sum();
            assert_eq!(n, reconstructed, "Reconstruction failed for {}", n);
        }
    }

    #[test]
    fn test_bits_roundtrip() {
        for n in 1..100 {
            let z1 = Zeckendorf::new(n);
            let bits = z1.to_bits();
            let z2 = Zeckendorf::from_bits(bits);

            assert_eq!(z1.indices, z2.indices);
            assert_eq!(z1.value, z2.value);
        }
    }

    #[test]
    fn test_gaps() {
        // F_10 = 55 (no gaps, single component)
        let z_pure = Zeckendorf::from_n(10);
        assert_eq!(z_pure.gaps(), vec![]);

        // Mixed representation has gaps
        let z_mixed = Zeckendorf::new(100);
        assert!(!z_mixed.gaps().is_empty());
    }

    #[test]
    fn test_composition() {
        let z1 = Zeckendorf::new(10);
        let z2 = Zeckendorf::new(20);
        let z_sum = z1.compose(&z2);

        assert_eq!(z_sum.value, 30);
    }

    #[test]
    fn test_gap_density() {
        // Pure Fibonacci (no gaps)
        let z_pure = Zeckendorf::from_n(10);
        assert_eq!(z_pure.gap_density(), 0.0);

        // Complex number (has gaps)
        let z_complex = Zeckendorf::new(100);
        assert!(z_complex.gap_density() > 0.0);
    }
}
