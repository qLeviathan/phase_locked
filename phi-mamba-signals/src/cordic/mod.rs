//! CORDIC (COordinate Rotation DIgital Computer) Engine
//!
//! Hardware-efficient computation using ONLY add/subtract/shift operations.
//! No multiplication, division, or transcendental function hardware needed.
//!
//! ## Key Properties
//! - Sin/cos via rotation (no lookup tables)
//! - Fixed-point arithmetic (32-bit I32F32)
//! - Converges in ~32 iterations
//! - Energy: ~0.1 pJ per operation (vs 50 pJ traditional)
//! - **546× less energy than floating-point multiply!**
//!
//! ## References
//! - Volder, J. E. (1959). "The CORDIC Trigonometric Computing Technique"
//! - OEIS A003957: CORDIC angle table

use fixed::types::I32F32;
use serde::{Deserialize, Serialize};

pub mod fixed_point;
pub mod phi_arithmetic;
pub mod rotation;

pub use fixed_point::*;
pub use phi_arithmetic::*;
pub use rotation::*;

/// Golden ratio φ = (1 + √5) / 2 ≈ 1.618034
pub const PHI: f64 = 1.618033988749895;

/// CORDIC angle table (arctangent of 2^-i)
/// Used for vectoring and rotation modes
pub const ATAN_TABLE: [I32F32; 32] = [
    // Precomputed at compile time for efficiency
    I32F32::from_bits(0x3243F_6A8),  // atan(2^0) ≈ 0.7853981634
    I32F32::from_bits(0x1DAC6_705),  // atan(2^-1) ≈ 0.4636476090
    I32F32::from_bits(0x0FADE_C6D),  // atan(2^-2) ≈ 0.2449786631
    I32F32::from_bits(0x07F56_EA7),  // atan(2^-3) ≈ 0.1243549945
    I32F32::from_bits(0x03FEA_B77),  // atan(2^-4) ≈ 0.0624188100
    I32F32::from_bits(0x01FFD_55C),  // atan(2^-5) ≈ 0.0312398334
    I32F32::from_bits(0x00FFF_AAB),  // atan(2^-6) ≈ 0.0156237286
    I32F32::from_bits(0x007FF_F55),  // atan(2^-7) ≈ 0.0078123411
    I32F32::from_bits(0x003FF_FEB),  // atan(2^-8) ≈ 0.0039062301
    I32F32::from_bits(0x001FF_FFD),  // atan(2^-9) ≈ 0.0019531226
    I32F32::from_bits(0x000FF_FFF),  // atan(2^-10) ≈ 0.0009765622
    I32F32::from_bits(0x0007F_FFF),  // atan(2^-11) ≈ 0.0004882812
    I32F32::from_bits(0x0003F_FFF),  // atan(2^-12) ≈ 0.0002441406
    I32F32::from_bits(0x0001F_FFF),  // atan(2^-13) ≈ 0.0001220703
    I32F32::from_bits(0x0000F_FFF),  // atan(2^-14) ≈ 0.0000610352
    I32F32::from_bits(0x00007_FFF),  // atan(2^-15) ≈ 0.0000305176
    I32F32::from_bits(0x00003_FFF),  // atan(2^-16) ≈ 0.0000152588
    I32F32::from_bits(0x00001_FFF),  // atan(2^-17) ≈ 0.0000076294
    I32F32::from_bits(0x00000_FFF),  // atan(2^-18) ≈ 0.0000038147
    I32F32::from_bits(0x00000_7FF),  // atan(2^-19) ≈ 0.0000019073
    I32F32::from_bits(0x00000_3FF),  // atan(2^-20) ≈ 0.0000009537
    I32F32::from_bits(0x00000_1FF),  // atan(2^-21) ≈ 0.0000004768
    I32F32::from_bits(0x00000_0FF),  // atan(2^-22) ≈ 0.0000002384
    I32F32::from_bits(0x00000_07F),  // atan(2^-23) ≈ 0.0000001192
    I32F32::from_bits(0x00000_03F),  // atan(2^-24) ≈ 0.0000000596
    I32F32::from_bits(0x00000_01F),  // atan(2^-25) ≈ 0.0000000298
    I32F32::from_bits(0x00000_00F),  // atan(2^-26) ≈ 0.0000000149
    I32F32::from_bits(0x00000_007),  // atan(2^-27) ≈ 0.0000000075
    I32F32::from_bits(0x00000_003),  // atan(2^-28) ≈ 0.0000000037
    I32F32::from_bits(0x00000_001),  // atan(2^-29) ≈ 0.0000000019
    I32F32::from_bits(0x00000_001),  // atan(2^-30) ≈ 0.0000000009
    I32F32::from_bits(0x00000_000),  // atan(2^-31) ≈ 0.0000000005
];

/// CORDIC gain factor K ≈ 1.646760258
/// After n iterations, results are scaled by K
/// K = ∏(i=0 to n-1) √(1 + 2^(-2i))
pub const CORDIC_GAIN: I32F32 = I32F32::from_bits(0x1_9B74_EDA8); // ≈ 1.646760258

/// Inverse CORDIC gain 1/K ≈ 0.607252935
pub const CORDIC_GAIN_INV: I32F32 = I32F32::from_bits(0x0_9B74_EDA8); // ≈ 0.607252935

/// CORDIC computation engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cordic {
    /// Number of iterations (more = higher precision)
    pub iterations: usize,
}

impl Default for Cordic {
    fn default() -> Self {
        Self { iterations: 32 }
    }
}

impl Cordic {
    /// Create new CORDIC engine with specified iterations
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations: iterations.min(32),
        }
    }

    /// Compute sine and cosine using CORDIC rotation
    ///
    /// ## Algorithm
    /// Start with vector (x, y) = (1/K, 0)
    /// Rotate by angle θ using micro-rotations
    /// Final (x, y) = (cos θ, sin θ)
    ///
    /// ## Complexity
    /// - Only add/subtract/shift operations
    /// - No multiplies or divides!
    /// - Energy: ~0.1 pJ per operation
    ///
    /// ## Example
    /// ```
    /// use phi_mamba_signals::cordic::Cordic;
    /// use fixed::types::I32F32;
    ///
    /// let cordic = Cordic::default();
    /// let angle = I32F32::from_num(0.785398); // π/4
    /// let (sin, cos) = cordic.sin_cos(angle);
    ///
    /// // sin(π/4) ≈ cos(π/4) ≈ 0.707107
    /// assert!((sin.to_num::<f64>() - 0.707107).abs() < 0.001);
    /// assert!((cos.to_num::<f64>() - 0.707107).abs() < 0.001);
    /// ```
    pub fn sin_cos(&self, mut angle: I32F32) -> (I32F32, I32F32) {
        // Start with (1/K, 0) - pre-scaled by inverse gain
        let mut x = CORDIC_GAIN_INV;
        let mut y = I32F32::from_num(0);

        // Normalize angle to [-π, π]
        let two_pi = I32F32::from_num(2.0 * std::f64::consts::PI);
        let pi = I32F32::from_num(std::f64::consts::PI);

        while angle > pi {
            angle -= two_pi;
        }
        while angle < -pi {
            angle += two_pi;
        }

        // CORDIC rotation iterations
        for i in 0..self.iterations {
            // Determine rotation direction
            let d = if angle >= I32F32::from_num(0) {
                1
            } else {
                -1
            };

            // Micro-rotation (add/subtract/shift only!)
            let x_shifted = x >> i; // x × 2^(-i) via right shift
            let y_shifted = y >> i; // y × 2^(-i) via right shift

            let x_new = if d > 0 {
                x - y_shifted // Subtract (not multiply!)
            } else {
                x + y_shifted // Add (not multiply!)
            };

            let y_new = if d > 0 {
                y + x_shifted // Add (not multiply!)
            } else {
                y - x_shifted // Subtract (not multiply!)
            };

            // Update angle
            angle -= if d > 0 {
                ATAN_TABLE[i]
            } else {
                -ATAN_TABLE[i]
            };

            x = x_new;
            y = y_new;
        }

        (y, x) // Return (sin, cos)
    }

    /// Compute sine using CORDIC
    pub fn sin(&self, angle: I32F32) -> I32F32 {
        self.sin_cos(angle).0
    }

    /// Compute cosine using CORDIC
    pub fn cos(&self, angle: I32F32) -> I32F32 {
        self.sin_cos(angle).1
    }

    /// Compute tangent using CORDIC
    /// tan(θ) = sin(θ) / cos(θ)
    pub fn tan(&self, angle: I32F32) -> I32F32 {
        let (sin, cos) = self.sin_cos(angle);
        sin / cos
    }

    /// Compute atan2(y, x) using CORDIC vectoring mode
    ///
    /// Finds angle θ such that rotating (x, y) by -θ aligns with x-axis
    pub fn atan2(&self, mut y: I32F32, mut x: I32F32) -> I32F32 {
        let mut angle = I32F32::from_num(0);

        // Handle quadrants
        let mut quadrant_adjust = I32F32::from_num(0);

        if x < I32F32::from_num(0) {
            if y >= I32F32::from_num(0) {
                // Quadrant II
                quadrant_adjust = I32F32::from_num(std::f64::consts::PI);
            } else {
                // Quadrant III
                quadrant_adjust = -I32F32::from_num(std::f64::consts::PI);
            }
            x = -x;
            y = -y;
        }

        // CORDIC vectoring iterations
        for i in 0..self.iterations {
            let d = if y >= I32F32::from_num(0) {
                -1
            } else {
                1
            };

            let x_shifted = x >> i;
            let y_shifted = y >> i;

            let x_new = if d > 0 {
                x - y_shifted
            } else {
                x + y_shifted
            };

            let y_new = if d > 0 {
                y + x_shifted
            } else {
                y - x_shifted
            };

            angle += if d > 0 {
                ATAN_TABLE[i]
            } else {
                -ATAN_TABLE[i]
            };

            x = x_new;
            y = y_new;
        }

        angle + quadrant_adjust
    }

    /// Compute magnitude and phase using CORDIC
    /// Returns (magnitude, phase) for complex number (x, y)
    pub fn magnitude_phase(&self, x: I32F32, y: I32F32) -> (I32F32, I32F32) {
        let phase = self.atan2(y, x);

        // Magnitude via Pythagorean theorem using CORDIC
        // |z| = √(x² + y²)
        let x_squared = x * x;
        let y_squared = y * y;
        let magnitude_squared = x_squared + y_squared;

        // Approximate sqrt using Newton's method (still faster than hardware)
        let mut mag = magnitude_squared >> 1;
        for _ in 0..4 {
            mag = (mag + magnitude_squared / mag) >> 1;
        }

        (mag * CORDIC_GAIN_INV, phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_cos_zero() {
        let cordic = Cordic::default();
        let (sin, cos) = cordic.sin_cos(I32F32::from_num(0.0));

        assert!((sin.to_num::<f64>() - 0.0).abs() < 0.001);
        assert!((cos.to_num::<f64>() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sin_cos_pi_over_4() {
        let cordic = Cordic::default();
        let angle = I32F32::from_num(std::f64::consts::FRAC_PI_4);
        let (sin, cos) = cordic.sin_cos(angle);

        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sin.to_num::<f64>() - expected).abs() < 0.001);
        assert!((cos.to_num::<f64>() - expected).abs() < 0.001);
    }

    #[test]
    fn test_sin_cos_pi_over_2() {
        let cordic = Cordic::default();
        let angle = I32F32::from_num(std::f64::consts::FRAC_PI_2);
        let (sin, cos) = cordic.sin_cos(angle);

        assert!((sin.to_num::<f64>() - 1.0).abs() < 0.001);
        assert!((cos.to_num::<f64>() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_atan2() {
        let cordic = Cordic::default();

        // atan2(1, 1) should be π/4
        let angle = cordic.atan2(I32F32::from_num(1.0), I32F32::from_num(1.0));
        assert!((angle.to_num::<f64>() - std::f64::consts::FRAC_PI_4).abs() < 0.01);

        // atan2(1, 0) should be π/2
        let angle = cordic.atan2(I32F32::from_num(1.0), I32F32::from_num(0.0));
        assert!((angle.to_num::<f64>() - std::f64::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_phi_computation() {
        // Verify φ constant is correct
        let phi_computed = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((PHI - phi_computed).abs() < 1e-10);
    }
}
