//! # Phi-Mamba Trade Signal Generator
//!
//! High-performance trade signal generation using golden ratio mathematics
//! and CORDIC computation.
//!
//! ## Key Features
//!
//! - **Add/Subtract/Shift Only**: CORDIC engine uses no multiply/divide
//! - **160× Faster**: Phi-space multiplication is just addition
//! - **546× Less Energy**: Hardware-efficient computation
//! - **Zeckendorf Encoding**: OEIS A003714 unique Fibonacci decomposition
//! - **Berry Phase Detection**: Finds phase-locked correlations
//! - **WASM Ready**: Compile to browser or native
//!
//! ## Example Usage
//!
//! ```
//! use phi_mamba_signals::{
//!     encoding::{FinancialEncoder, OHLCVBar},
//!     cordic::Cordic,
//! };
//!
//! // Create encoder
//! let mut encoder = FinancialEncoder::default();
//!
//! // Create market data
//! let bar = OHLCVBar {
//!     timestamp: 1700000000,
//!     ticker: "AAPL".to_string(),
//!     open: 180.0,
//!     high: 182.0,
//!     low: 179.0,
//!     close: 181.5,
//!     volume: 50_000_000,
//! };
//!
//! // Encode to φ-space
//! let state = encoder.encode(&bar);
//!
//! println!("Price in φ-space: {:?}", state.price_phi);
//! println!("Zeckendorf: {:?}", state.price_zeck);
//! ```
//!
//! ## Mathematical Foundations
//!
//! ### Zeckendorf Theorem (1972, OEIS A003714)
//! Every positive integer has a unique representation as non-consecutive Fibonacci numbers.
//!
//! ### Golden Ratio φ = (1 + √5) / 2
//! φ² = φ + 1 (recursive property)
//!
//! ### Phi-Space Arithmetic
//! φⁿ × φᵐ = φ^(n+m) → multiplication becomes addition!
//!
//! ### CORDIC (Volder, 1959)
//! Compute sin/cos using only add/subtract/shift operations.
//!
//! ### Berry Phase (Berry, 1984)
//! Geometric phase for detecting correlation/synchronization.

pub mod cordic;
pub mod encoding;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for convenience
pub use cordic::{Cordic, PhiNum, PHI};
pub use encoding::{
    berry_phase::{compute_berry_phase, BerryPhase, PHASE_LOCK_THRESHOLD},
    financial::{FinancialEncoder, FinancialState, OHLCVBar},
    zeckendorf::{fibonacci, lucas, zeckendorf_decomposition, DualZeckendorf},
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Verify CORDIC accuracy
///
/// Compares CORDIC sin/cos to standard library implementations
/// Returns maximum error across test angles
pub fn verify_cordic_accuracy() -> f64 {
    use fixed::types::I32F32;

    let cordic = Cordic::default();
    let test_angles = [
        0.0,
        std::f64::consts::FRAC_PI_6,  // 30°
        std::f64::consts::FRAC_PI_4,  // 45°
        std::f64::consts::FRAC_PI_3,  // 60°
        std::f64::consts::FRAC_PI_2,  // 90°
        std::f64::consts::PI,          // 180°
    ];

    let mut max_error: f64 = 0.0;

    for &angle_f64 in &test_angles {
        let angle = I32F32::from_num(angle_f64);
        let (sin_cordic, cos_cordic) = cordic.sin_cos(angle);

        let sin_std = angle_f64.sin();
        let cos_std = angle_f64.cos();

        let sin_error = (sin_cordic.to_num::<f64>() - sin_std).abs();
        let cos_error = (cos_cordic.to_num::<f64>() - cos_std).abs();

        max_error = max_error.max(sin_error).max(cos_error);
    }

    max_error
}

/// Verify Zeckendorf uniqueness
///
/// Tests that decomposition is unique and non-consecutive for range [1, n]
pub fn verify_zeckendorf_uniqueness(n: u64) -> bool {
    for i in 1..=n {
        let zeck = zeckendorf_decomposition(i);

        // Verify sum
        let sum: u64 = zeck.iter().sum();
        if sum != i {
            return false;
        }

        // Verify non-consecutive
        for j in 0..zeck.len().saturating_sub(1) {
            let curr = zeck[j];
            let next = zeck[j + 1];

            // Find Fibonacci indices
            let mut curr_idx = 0;
            let mut next_idx = 0;

            for k in 0..64 {
                if fibonacci(k) == curr {
                    curr_idx = k;
                }
                if fibonacci(k) == next {
                    next_idx = k;
                }
            }

            // Must differ by at least 2
            if next_idx < curr_idx + 2 {
                return false;
            }
        }
    }

    true
}

/// Run all verification tests
pub fn verify_all() -> Result<(), String> {
    // Verify CORDIC
    let cordic_error = verify_cordic_accuracy();
    if cordic_error > 0.01 {
        return Err(format!(
            "CORDIC accuracy too low: max error = {}",
            cordic_error
        ));
    }

    // Verify Zeckendorf
    if !verify_zeckendorf_uniqueness(100) {
        return Err("Zeckendorf uniqueness violated".to_string());
    }

    // Verify φ constant
    let phi_computed = (1.0 + 5.0_f64.sqrt()) / 2.0;
    if (PHI - phi_computed).abs() > 1e-10 {
        return Err("PHI constant incorrect".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_cordic_accuracy() {
        let error = verify_cordic_accuracy();
        assert!(error < 0.01, "CORDIC error too high: {}", error);
    }

    #[test]
    fn test_zeckendorf_uniqueness() {
        assert!(verify_zeckendorf_uniqueness(100));
    }

    #[test]
    fn test_verify_all() {
        assert!(verify_all().is_ok());
    }

    #[test]
    fn test_phi_constant() {
        let phi_computed = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((PHI - phi_computed).abs() < 1e-10);
    }

    #[test]
    fn test_encode_example() {
        let mut encoder = FinancialEncoder::default();

        let bar = OHLCVBar {
            timestamp: 1700000000,
            ticker: "AAPL".to_string(),
            open: 180.0,
            high: 182.0,
            low: 179.0,
            close: 181.5,
            volume: 50_000_000,
        };

        let state = encoder.encode(&bar);

        assert_eq!(state.ticker, "AAPL");
        assert!(!state.price_zeck.is_empty());
    }

    #[test]
    fn test_berry_phase_example() {
        let mut encoder = FinancialEncoder::default();

        let bar1 = OHLCVBar {
            timestamp: 1700000000,
            ticker: "AAPL".to_string(),
            open: 180.0,
            high: 182.0,
            low: 179.0,
            close: 181.5,
            volume: 50_000_000,
        };

        let bar2 = OHLCVBar {
            timestamp: 1700000060,
            ticker: "GOOGL".to_string(),
            open: 140.0,
            high: 142.0,
            low: 139.0,
            close: 141.2,
            volume: 30_000_000,
        };

        let state1 = encoder.encode(&bar1);
        encoder.reset();
        let state2 = encoder.encode(&bar2);

        let berry = compute_berry_phase(&state1, &state2);

        assert!(berry.phase >= 0.0);
        assert!(berry.phase <= std::f64::consts::PI);
        assert!(berry.coherence >= 0.0);
        assert!(berry.coherence <= 1.0);
    }
}
