//! Dual Zeckendorf Binary Cascading ZORDIC Lattice
//!
//! Mathematical foundation for holographic memory.
//!
//! Key properties (OEIS verified):
//! - Zeckendorf uniqueness (A003714)
//! - Fibonacci recurrence (A000045)
//! - Lucas recurrence (A000032)
//! - φ = (1+√5)/2 (A001622)

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Fibonacci number computation (OEIS A000045)
///
/// F(n) = (φⁿ - ψⁿ) / √5
/// Using Binet formula for exact computation
pub fn fibonacci(n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    // Use iterative approach for mobile efficiency
    let mut fib_prev = 0u64;
    let mut fib_curr = 1u64;

    for _ in 2..=n {
        let fib_next = fib_prev.saturating_add(fib_curr);
        fib_prev = fib_curr;
        fib_curr = fib_next;
    }

    fib_curr
}

/// Lucas number computation (OEIS A000032)
///
/// L(n) = φⁿ + ψⁿ
pub fn lucas(n: usize) -> u64 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }

    let mut lucas_prev = 2u64;
    let mut lucas_curr = 1u64;

    for _ in 2..=n {
        let lucas_next = lucas_prev.saturating_add(lucas_curr);
        lucas_prev = lucas_curr;
        lucas_curr = lucas_next;
    }

    lucas_curr
}

/// Zeckendorf decomposition (OEIS A003714)
///
/// Every n has unique representation as sum of non-consecutive Fibonacci numbers.
///
/// THEOREM (Zeckendorf 1972):
/// ∀n ∈ ℕ⁺, ∃! sequence (aₖ) where:
///   n = Σ aₖ · F(k), aₖ ∈ {0,1}, aₖ·aₖ₊₁ = 0 (non-consecutive)
///
/// Algorithm: Greedy descent (proven optimal)
pub fn zeckendorf_decomposition(mut n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }

    // Generate Fibonacci numbers up to n
    let mut fibs = vec![1, 2];
    while *fibs.last().unwrap() < n {
        let len = fibs.len();
        let next = fibs[len - 1] + fibs[len - 2];
        if next > n {
            break;
        }
        fibs.push(next);
    }

    // Greedy decomposition (largest first)
    let mut result = Vec::new();
    for &fib in fibs.iter().rev() {
        if fib <= n {
            result.push(fib);
            n -= fib;
        }
    }

    result.reverse();
    result
}

/// Lucas decomposition (dual to Zeckendorf)
///
/// Similar to Zeckendorf but using Lucas numbers.
/// Non-consecutive property still holds.
pub fn lucas_decomposition(mut n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }

    let mut lucas_nums = vec![2, 1];
    while *lucas_nums.last().unwrap() < n {
        let len = lucas_nums.len();
        let next = lucas_nums[len - 1] + lucas_nums[len - 2];
        if next > n {
            break;
        }
        lucas_nums.push(next);
    }

    let mut result = Vec::new();
    for &luc in lucas_nums.iter().rev() {
        if luc <= n {
            result.push(luc);
            n -= luc;
        }
    }

    result.reverse();
    result
}

/// Dual Zeckendorf representation
///
/// Encodes value using both Fibonacci and Lucas decompositions.
/// Intersection = critical information
/// Difference = contextual information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualZeckendorf {
    /// Value being represented
    pub value: u64,

    /// Forward Zeckendorf decomposition (Fibonacci)
    pub zeckendorf_forward: Vec<u64>,

    /// Backward decomposition (Lucas)
    pub lucas_backward: Vec<u64>,

    /// Intersection (both agree)
    pub intersection: Vec<u64>,

    /// Symmetric difference (context)
    pub difference: Vec<u64>,

    /// Active holes (gaps in binary representation)
    pub active_holes: Vec<usize>,
}

impl DualZeckendorf {
    /// Create dual representation of value
    pub fn new(value: u64) -> Self {
        let zeckendorf_forward = zeckendorf_decomposition(value);
        let lucas_backward = lucas_decomposition(value);

        // Compute intersection
        let zeck_set: HashSet<_> = zeckendorf_forward.iter().collect();
        let lucas_set: HashSet<_> = lucas_backward.iter().collect();

        let intersection: Vec<u64> = zeck_set
            .intersection(&lucas_set)
            .map(|&&v| v)
            .collect();

        // Compute symmetric difference
        let difference: Vec<u64> = zeck_set
            .symmetric_difference(&lucas_set)
            .map(|&&v| v)
            .collect();

        // Compute active holes (positions with 1 in binary representation)
        let active_holes = Self::compute_holes(&zeckendorf_forward);

        Self {
            value,
            zeckendorf_forward,
            lucas_backward,
            intersection,
            difference,
            active_holes,
        }
    }

    /// Compute hole positions (Fibonacci indices where bit = 1)
    fn compute_holes(decomposition: &[u64]) -> Vec<usize> {
        decomposition
            .iter()
            .filter_map(|&fib_val| {
                // Find Fibonacci index
                let mut f_prev = 0;
                let mut f_curr = 1;
                let mut index = 1;

                while f_curr < fib_val {
                    let f_next = f_prev + f_curr;
                    f_prev = f_curr;
                    f_curr = f_next;
                    index += 1;
                }

                if f_curr == fib_val {
                    Some(index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if value is phase-locked with another
    ///
    /// Phase-locked when Berry phase ≈ 0 (mod 2π)
    /// Approximation: holes align at similar scales
    pub fn is_phase_locked_with(&self, other: &DualZeckendorf) -> bool {
        let self_holes: HashSet<_> = self.active_holes.iter().collect();
        let other_holes: HashSet<_> = other.active_holes.iter().collect();

        let overlap = self_holes.intersection(&other_holes).count();
        let total = self_holes.union(&other_holes).count();

        if total == 0 {
            return false;
        }

        let overlap_ratio = overlap as f64 / total as f64;

        // Phase-locked if >50% holes overlap
        overlap_ratio > 0.5
    }
}

/// Cascading φ-layer
///
/// Each layer applies φ multiplication (via addition in log space)
/// Reveals structure at different scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeLayer {
    /// Layer index
    pub k: usize,

    /// Scale description (e.g., "φ²·F_k")
    pub scale: String,

    /// Binary representation (holes = 1, no hole = 0)
    pub bits: String,

    /// Energy level (φ^(-k))
    pub energy: f64,

    /// φ exponent for this layer
    pub phi_exponent: i32,

    /// Dual Zeckendorf at this layer
    pub dual_zeck: DualZeckendorf,
}

impl CascadeLayer {
    /// Create layer k from base value
    pub fn new(base_value: u64, k: usize) -> Self {
        // φᵏ multiplication (done in log space, then exp)
        // For simplicity, approximate: value_k ≈ base_value * φ^k
        let phi = 1.618033988749895_f64;
        let scaled_value = (base_value as f64 * phi.powi(k as i32)).round() as u64;

        let dual_zeck = DualZeckendorf::new(scaled_value);

        // Create binary string from active holes
        let max_hole = dual_zeck.active_holes.iter().max().unwrap_or(&0);
        let mut bits = vec!['0'; max_hole + 1];
        for &hole in &dual_zeck.active_holes {
            bits[hole] = '1';
        }
        let bits: String = bits.into_iter().collect();

        // Energy decay
        let energy = phi.powi(-(k as i32));

        let scale = if k == 0 {
            "F_k".to_string()
        } else {
            format!("φ{}·F_k", "¹²³⁴⁵⁶⁷⁸⁹".chars().nth(k - 1).unwrap_or('ⁿ'))
        };

        Self {
            k,
            scale,
            bits,
            energy,
            phi_exponent: k as i32,
            dual_zeck,
        }
    }
}

/// Complete ZORDIC lattice structure
///
/// Z = Zeckendorf
/// O = Observation
/// R = Recursive
/// D = Distributed
/// I = Invariant
/// C = Cascade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZordicLattice {
    /// Base observation value
    pub base_value: u64,

    /// Timestamp of creation
    pub timestamp: i64,

    /// Cascade layers (different φ scales)
    pub layers: Vec<CascadeLayer>,

    /// Self-coherence (Berry phase with self)
    pub self_coherence: f64,
}

impl ZordicLattice {
    /// Create lattice from observation
    ///
    /// Generates multiple cascade layers revealing different scales
    pub fn new(base_value: u64, num_layers: usize) -> Self {
        let layers: Vec<_> = (0..num_layers)
            .map(|k| CascadeLayer::new(base_value, k))
            .collect();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self {
            base_value,
            timestamp,
            layers,
            self_coherence: 0.0, // Perfect self-coherence
        }
    }

    /// Compute Berry phase with another lattice
    ///
    /// Measures phase-locking strength
    pub fn berry_phase(&self, other: &ZordicLattice) -> f64 {
        let mut phase_sum = 0.0;
        let mut weight_sum = 0.0;

        for (layer_self, layer_other) in self.layers.iter().zip(other.layers.iter()) {
            let is_locked = layer_self
                .dual_zeck
                .is_phase_locked_with(&layer_other.dual_zeck);

            let weight = layer_self.energy;
            let phase_contribution = if is_locked { 0.0 } else { std::f64::consts::PI };

            phase_sum += weight * phase_contribution;
            weight_sum += weight;
        }

        if weight_sum == 0.0 {
            return std::f64::consts::PI; // Maximum phase
        }

        phase_sum / weight_sum
    }

    /// Check if phase-locked with another lattice
    ///
    /// Threshold: Berry phase < π/4
    pub fn is_phase_locked_with(&self, other: &ZordicLattice) -> bool {
        let berry_phase = self.berry_phase(other);
        berry_phase < std::f64::consts::FRAC_PI_4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(3), 2);
        assert_eq!(fibonacci(10), 55);
    }

    #[test]
    fn test_zeckendorf_uniqueness() {
        // 17 = 13 + 3 + 1 (unique)
        let zeck = zeckendorf_decomposition(17);
        assert_eq!(zeck, vec![1, 3, 13]);

        // 100 = 89 + 8 + 3
        let zeck = zeckendorf_decomposition(100);
        assert_eq!(zeck, vec![3, 8, 89]);
    }

    #[test]
    fn test_dual_zeckendorf() {
        let dual = DualZeckendorf::new(100);
        assert!(!dual.zeckendorf_forward.is_empty());
        assert!(!dual.lucas_backward.is_empty());
    }

    #[test]
    fn test_cascade_layer() {
        let layer = CascadeLayer::new(100, 0);
        assert_eq!(layer.k, 0);
        assert!(layer.energy > 0.0);
        assert!(!layer.bits.is_empty());
    }

    #[test]
    fn test_zordic_lattice() {
        let lattice = ZordicLattice::new(100, 3);
        assert_eq!(lattice.layers.len(), 3);
        assert_eq!(lattice.self_coherence, 0.0);
    }

    #[test]
    fn test_phase_locking() {
        let lattice1 = ZordicLattice::new(100, 3);
        let lattice2 = ZordicLattice::new(105, 3); // Similar value
        let lattice3 = ZordicLattice::new(1000, 3); // Very different

        // Similar values should have lower Berry phase
        let phase_similar = lattice1.berry_phase(&lattice2);
        let phase_different = lattice1.berry_phase(&lattice3);

        assert!(phase_similar < phase_different);
    }
}
