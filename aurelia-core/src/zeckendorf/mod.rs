/**
 * Zeckendorf-Fibonacci Lattice Mathematics
 *
 * Implements the mathematical core of Aurelia's market perception:
 * - Zeckendorf decomposition (OEIS A003714)
 * - Bidirectional φ/ψ lattice coordinates
 * - Fibonacci price levels
 * - Lucas time projection
 * - Berry phase calculation
 */

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// Golden ratio conjugate ψ = (1 - √5) / 2
pub const PSI: f64 = -0.618033988749895;

/// Fibonacci sequence cache (precomputed for efficiency)
pub const FIBONACCI: [u64; 93] = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
    6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040,
    1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986,
    102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903,
    2971215073, 4807526976, 7778742049, 12586269025, 20365011074, 32951280099,
    53316291173, 86267571272, 139583862445, 225851433717, 365435296162, 591286729879,
    956722026041, 1548008755920, 2504730781961, 4052739537881, 6557470319842,
    10610209857723, 17167680177565, 27777890035288, 44945570212853, 72723460248141,
    117669030460994, 190392490709135, 308061521170129, 498454011879264, 806515533049393,
    1304969544928657, 2111485077978050, 3416454622906707, 5527939700884757,
    8944394323791464, 14472334024676221, 23416728348467685, 37889062373143906,
    61305790721611591, 99194853094755497, 160500643816367088, 259695496911122585,
    420196140727489673, 679891637638612258, 1100087778366101931, 1779979416004714189,
    2880067194370816120, 4660046610375530309, 7540113804746346429,
];

/// Lucas sequence cache (used for time projection)
pub const LUCAS: [u64; 92] = [
    2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778,
    9349, 15127, 24476, 39603, 64079, 103682, 167761, 271443, 439204, 710647, 1149851,
    1860498, 3010349, 4870847, 7881196, 12752043, 20633239, 33385282, 54018521, 87403803,
    141422324, 228826127, 370248451, 599074578, 969323029, 1568397607, 2537720636,
    4106118243, 6643838879, 10749957122, 17393796001, 28143753123, 45537549124,
    73681302247, 119218851371, 192900153618, 312119004989, 505019158607, 817138163596,
    1322157322203, 2139295485799, 3461452808002, 5600748293801, 9062201101803,
    14662949395604, 23725150497407, 38388099893011, 62113250390418, 100501350283429,
    162614600673847, 263115950957276, 425730551631123, 688846502588399, 1114577054219522,
    1803423556807921, 2918000611027443, 4721424167835364, 7639424778862807,
    12360848946698171, 20000273725560978, 32361122672259149, 52361396397820127,
    84722519070079276, 137083915467899403, 221806434537978679, 358890350005878082,
    580696784543856761, 939587134549734843, 1520283919093591604, 2459871053643326447,
    3980154972736918051, 6440026026380244498, 10420180999117162549,
];

/// Standard Fibonacci retracement levels
pub const FIB_LEVELS: [f64; 9] = [
    0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618,
];

/// Zeckendorf encoder - converts integers to unique Fibonacci representations
#[derive(Debug, Clone)]
pub struct ZeckendorfEncoder;

impl ZeckendorfEncoder {
    /// Decompose integer into non-consecutive Fibonacci indices (OEIS A003714)
    ///
    /// Example: 100 = F(12) + F(9) + F(6) + F(2) = 89 + 8 + 3 + 1
    pub fn decompose(n: u64) -> Vec<usize> {
        if n == 0 {
            return vec![0];
        }

        let mut indices = Vec::new();
        let mut remaining = n;

        // Find largest Fibonacci number ≤ n
        let mut i = FIBONACCI.len() - 1;
        while i > 0 && FIBONACCI[i] > remaining {
            i -= 1;
        }

        // Greedy algorithm: take largest Fib, skip next (non-consecutive)
        while remaining > 0 && i >= 2 {
            if FIBONACCI[i] <= remaining {
                indices.push(i);
                remaining -= FIBONACCI[i];
                i = i.saturating_sub(2); // Skip next to ensure non-consecutive
            } else {
                i -= 1;
            }
        }

        indices.reverse(); // Return in ascending order
        indices
    }

    /// Compute φ^n (golden ratio power)
    pub fn phi_power(n: usize) -> f64 {
        PHI.powi(n as i32)
    }

    /// Compute ψ^n (conjugate power)
    pub fn psi_power(n: usize) -> f64 {
        PSI.powi(n as i32)
    }
}

/// Point in the bidirectional φ/ψ lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticePoint {
    pub x: f64, // φ-component (growth/expansion)
    pub y: f64, // ψ-component (decay/contraction)
    pub magnitude: f64,
    pub phase: f64, // Angle in lattice space
}

impl LatticePoint {
    /// Compute lattice coordinates from Zeckendorf decomposition
    ///
    /// Uses time modulation via Lucas sequence for temporal dynamics
    pub fn from_value(value: i64, time_index: usize) -> Self {
        let z_indices = ZeckendorfEncoder::decompose(value.unsigned_abs());

        // Sum φ^i and ψ^i components
        let phi_component: f64 = z_indices
            .iter()
            .map(|&i| ZeckendorfEncoder::phi_power(i))
            .sum();

        let psi_component: f64 = z_indices
            .iter()
            .map(|&i| ZeckendorfEncoder::psi_power(i))
            .sum();

        // Time modulation using Lucas numbers (creates temporal waves)
        let lucas_8 = LUCAS[8] as f64; // L(8) = 47
        let time_phase = 2.0 * PI * ((time_index % lucas_8 as usize) as f64 / lucas_8);

        // Rotate in lattice space
        let x = phi_component * time_phase.cos() - psi_component * time_phase.sin();
        let y = phi_component * time_phase.sin() + psi_component * time_phase.cos();

        // Apply sign if value was negative
        let (x, y) = if value < 0 { (-x, -y) } else { (x, y) };

        let magnitude = (x * x + y * y).sqrt();
        let phase = y.atan2(x);

        Self {
            x,
            y,
            magnitude,
            phase,
        }
    }

    /// Calculate distance to another lattice point
    pub fn distance_to(&self, other: &LatticePoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate Berry phase (geometric phase for correlation detection)
    ///
    /// Returns angle in [-π, π]. Threshold for correlation: |θ| < π/4
    pub fn berry_phase(&self, other: &LatticePoint) -> f64 {
        // Compute Berry phase as angle between vectors
        let dot = self.x * other.x + self.y * other.y;
        let cross = self.x * other.y - self.y * other.x;
        cross.atan2(dot)
    }

    /// Check if two points are correlated (Berry phase test)
    pub fn is_correlated(&self, other: &LatticePoint) -> bool {
        let berry = self.berry_phase(other);
        berry.abs() < PI / 4.0 // Threshold: π/4 radians
    }
}

/// Fibonacci price level calculator
#[derive(Debug, Clone)]
pub struct FibonacciLevels {
    pub swing_high: f64,
    pub swing_low: f64,
    pub range: f64,
    pub levels: Vec<(f64, f64)>, // (ratio, price)
}

impl FibonacciLevels {
    /// Calculate Fibonacci retracement levels
    pub fn retracement(swing_high: f64, swing_low: f64) -> Self {
        let range = swing_high - swing_low;

        let levels = FIB_LEVELS
            .iter()
            .map(|&ratio| {
                let price = swing_high - (range * ratio);
                (ratio, price)
            })
            .collect();

        Self {
            swing_high,
            swing_low,
            range,
            levels,
        }
    }

    /// Calculate Fibonacci extension levels
    pub fn extension(swing_high: f64, swing_low: f64) -> Self {
        let range = swing_high - swing_low;

        let levels = FIB_LEVELS
            .iter()
            .map(|&ratio| {
                let price = swing_high + (range * ratio);
                (ratio, price)
            })
            .collect();

        Self {
            swing_high,
            swing_low,
            range,
            levels,
        }
    }

    /// Find nearest Fibonacci level to a given price
    pub fn nearest_level(&self, price: f64) -> Option<(f64, f64)> {
        self.levels
            .iter()
            .min_by(|a, b| {
                let dist_a = (a.1 - price).abs();
                let dist_b = (b.1 - price).abs();
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .copied()
    }

    /// Check if price is near a Fibonacci level (within tolerance)
    pub fn is_at_level(&self, price: f64, tolerance_pct: f64) -> Option<f64> {
        for (ratio, level_price) in &self.levels {
            let tolerance = self.range * tolerance_pct / 100.0;
            if (price - level_price).abs() < tolerance {
                return Some(*ratio);
            }
        }
        None
    }
}

/// Lucas time projector - predicts future turning points
#[derive(Debug, Clone)]
pub struct LucasTimeProjector;

impl LucasTimeProjector {
    /// Get Lucas number at index
    pub fn lucas(n: usize) -> u64 {
        if n < LUCAS.len() {
            LUCAS[n]
        } else {
            0
        }
    }

    /// Project future time windows using Lucas sequence
    ///
    /// Returns bar counts from current bar: [4, 7, 11, 18, 29, 47, ...]
    pub fn project_windows(current_bar: usize, lookforward: usize) -> Vec<usize> {
        (3..lookforward + 3)
            .map(|i| current_bar + Self::lucas(i) as usize)
            .collect()
    }

    /// Check if current bar is at a Lucas time window
    pub fn is_at_lucas_window(bar: usize, tolerance: usize) -> Option<usize> {
        for (i, &lucas_val) in LUCAS.iter().enumerate() {
            if (bar as i64 - lucas_val as i64).abs() <= tolerance as i64 {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeckendorf_decomposition() {
        // 100 = 89 + 8 + 3 (F12 + F6 + F4)
        let indices = ZeckendorfEncoder::decompose(100);
        let sum: u64 = indices.iter().map(|&i| FIBONACCI[i]).sum();
        assert_eq!(sum, 100);

        // Check non-consecutive property
        for i in 0..indices.len() - 1 {
            assert!(indices[i + 1] - indices[i] >= 2);
        }
    }

    #[test]
    fn test_lattice_point() {
        let point1 = LatticePoint::from_value(100, 0);
        let point2 = LatticePoint::from_value(100, 0);

        // Same values should produce same point
        assert!((point1.x - point2.x).abs() < 0.001);
        assert!((point1.y - point2.y).abs() < 0.001);
    }

    #[test]
    fn test_fibonacci_levels() {
        let levels = FibonacciLevels::retracement(460.0, 440.0);

        assert_eq!(levels.range, 20.0);

        // 0.618 level should be at 460 - (20 * 0.618) = 447.64
        let level_618 = levels.levels.iter().find(|(r, _)| (*r - 0.618).abs() < 0.001);
        assert!(level_618.is_some());
        assert!((level_618.unwrap().1 - 447.64).abs() < 0.01);
    }

    #[test]
    fn test_lucas_projection() {
        let windows = LucasTimeProjector::project_windows(0, 5);

        // Should return [4, 7, 11, 18, 29]
        assert_eq!(windows[0], 4);
        assert_eq!(windows[1], 7);
        assert_eq!(windows[2], 11);
        assert_eq!(windows[3], 18);
        assert_eq!(windows[4], 29);
    }

    #[test]
    fn test_berry_phase() {
        let point1 = LatticePoint::from_value(100, 0);
        let point2 = LatticePoint::from_value(105, 0); // Similar value

        let berry = point1.berry_phase(&point2);

        // Similar values should have small Berry phase
        assert!(berry.abs() < PI / 4.0);
        assert!(point1.is_correlated(&point2));
    }

    #[test]
    fn test_phi_psi_property() {
        // φ × ψ = -1
        let product = PHI * PSI;
        assert!((product + 1.0).abs() < 0.0001);
    }
}
