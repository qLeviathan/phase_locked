//! # Latent n - The Universal Encoding
//!
//! **n isn't just an index - it's a compressed manifold of information.**
//!
//! A single integer n encodes everything:
//! - Position in sequence → Energy (F_n)
//! - Time coordinate → Lucas number (L_n)
//! - Memory address → Zeckendorf bit pattern
//! - Error sites → Gaps in Zeckendorf representation
//! - Direction → Cassini phase (-1)^n

use crate::{FIBONACCI, LUCAS, is_valid_n, is_valid_lucas_n};
use crate::zeckendorf::Zeckendorf;

/// The complete universe encoded in a single integer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentN {
    /// The raw index (0..93)
    pub n: usize,
}

/// The decoded universe from a LatentN
#[derive(Debug, Clone, PartialEq)]
pub struct Universe {
    /// Energy level = F_n
    pub energy: u64,

    /// Time coordinate = L_n
    pub time: u64,

    /// Memory address = Zeckendorf bit pattern
    pub address: u64,

    /// Error sites = gaps in Zeckendorf representation
    pub error_sites: Vec<usize>,

    /// Direction = (-1)^n (forward=1, backward=-1)
    pub direction: i8,

    /// The original n
    pub n: usize,
}

impl LatentN {
    /// Create a new LatentN from index
    ///
    /// # Panics
    /// Panics if n >= 93 (beyond u64 Fibonacci range)
    pub const fn new(n: usize) -> Self {
        assert!(is_valid_n(n), "n must be < 93");
        Self { n }
    }

    /// Create from index without validation (use carefully)
    #[inline]
    pub const fn new_unchecked(n: usize) -> Self {
        Self { n }
    }

    /// Get the Fibonacci value F_n (energy)
    #[inline]
    pub const fn fibonacci(&self) -> u64 {
        FIBONACCI[self.n]
    }

    /// Get the Lucas value L_n (time)
    #[inline]
    pub fn lucas(&self) -> u64 {
        if is_valid_lucas_n(self.n) {
            LUCAS[self.n]
        } else {
            0 // Beyond Lucas table range
        }
    }

    /// Get Zeckendorf representation (memory address)
    pub fn zeckendorf(&self) -> Zeckendorf {
        Zeckendorf::from_n(self.n)
    }

    /// Get Cassini phase: (-1)^n
    #[inline]
    pub const fn cassini_phase(&self) -> i8 {
        if self.n % 2 == 0 { 1 } else { -1 }
    }

    /// Decode the complete universe from this n
    pub fn decode(&self) -> Universe {
        let zeck = self.zeckendorf();
        let error_sites = zeck.gaps();

        Universe {
            energy: self.fibonacci(),
            time: self.lucas(),
            address: zeck.to_bits(),
            error_sites,
            direction: self.cassini_phase(),
            n: self.n,
        }
    }

    /// Encode from energy level (find n where F_n ≈ energy)
    pub fn from_energy(energy: u64) -> Option<Self> {
        FIBONACCI
            .iter()
            .position(|&f| f >= energy)
            .map(|n| Self::new(n))
    }

    /// Encode from time coordinate (find n where L_n ≈ time)
    pub fn from_time(time: u64) -> Option<Self> {
        LUCAS
            .iter()
            .position(|&l| l >= time)
            .map(|n| Self::new(n))
    }

    /// Get the next n in sequence (φ direction)
    #[inline]
    pub const fn next(&self) -> Option<Self> {
        if self.n + 1 < 93 {
            Some(Self::new(self.n + 1))
        } else {
            None
        }
    }

    /// Get the previous n in sequence (ψ direction)
    #[inline]
    pub const fn prev(&self) -> Option<Self> {
        if self.n > 0 {
            Some(Self::new(self.n - 1))
        } else {
            None
        }
    }

    /// Advance by k steps (φ^k direction)
    pub const fn advance(&self, k: usize) -> Option<Self> {
        let new_n = self.n.saturating_add(k);
        if new_n < 93 {
            Some(Self::new(new_n))
        } else {
            None
        }
    }

    /// Retreat by k steps (ψ^k direction)
    pub const fn retreat(&self, k: usize) -> Option<Self> {
        if self.n >= k {
            Some(Self::new(self.n - k))
        } else {
            None
        }
    }
}

impl Universe {
    /// Check if this universe is in a stable state
    /// (energy and time are co-aligned)
    pub fn is_stable(&self) -> bool {
        // Stable when F_n and L_n have similar magnitude
        let ratio = self.time as f64 / self.energy.max(1) as f64;
        (1.5..2.5).contains(&ratio)
    }

    /// Check if this is an error-prone state
    /// (has gaps in Zeckendorf representation)
    pub fn has_errors(&self) -> bool {
        !self.error_sites.is_empty()
    }

    /// Check if moving forward in time
    pub fn is_forward(&self) -> bool {
        self.direction > 0
    }

    /// Get the error density (gaps per bit)
    pub fn error_density(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.error_sites.len() as f64 / self.n as f64
    }
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for LatentN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} [F={}, L={}, φ={}]",
            self.n,
            self.fibonacci(),
            self.lucas(),
            if self.cassini_phase() > 0 { "+" } else { "-" }
        )
    }
}

impl std::fmt::Display for Universe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Universe(n={}, E={}, T={}, addr=0x{:x}, errors={}, dir={})",
            self.n,
            self.energy,
            self.time,
            self.address,
            self.error_sites.len(),
            if self.direction > 0 { "→" } else { "←" }
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_n_creation() {
        let n5 = LatentN::new(5);
        assert_eq!(n5.n, 5);
        assert_eq!(n5.fibonacci(), 5);
        assert_eq!(n5.lucas(), 11);
    }

    #[test]
    fn test_cassini_phase() {
        assert_eq!(LatentN::new(0).cassini_phase(), 1);
        assert_eq!(LatentN::new(1).cassini_phase(), -1);
        assert_eq!(LatentN::new(2).cassini_phase(), 1);
        assert_eq!(LatentN::new(3).cassini_phase(), -1);
    }

    #[test]
    fn test_navigation() {
        let n = LatentN::new(5);
        assert_eq!(n.next().unwrap().n, 6);
        assert_eq!(n.prev().unwrap().n, 4);
        assert_eq!(n.advance(3).unwrap().n, 8);
        assert_eq!(n.retreat(2).unwrap().n, 3);
    }

    #[test]
    fn test_from_energy() {
        let n = LatentN::from_energy(100).unwrap();
        assert!(n.fibonacci() >= 100);
        assert_eq!(n.n, 12); // F_12 = 144
    }

    #[test]
    fn test_universe_decode() {
        let n = LatentN::new(10);
        let universe = n.decode();

        assert_eq!(universe.n, 10);
        assert_eq!(universe.energy, 55); // F_10
        assert_eq!(universe.time, 123); // L_10
        assert_eq!(universe.direction, 1); // even n → forward
    }

    #[test]
    fn test_boundary_conditions() {
        // n=0 should work
        let n0 = LatentN::new(0);
        assert_eq!(n0.fibonacci(), 0);
        assert!(n0.prev().is_none());

        // n=92 should work (last valid index)
        let n92 = LatentN::new(92);
        assert!(n92.fibonacci() > 0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_n() {
        LatentN::new(93); // Should panic
    }
}
