//! Financial Data Encoding to Phi-Space
//!
//! Converts OHLCV (Open, High, Low, Close, Volume) bars into
//! phi-space representations using Zeckendorf decomposition

use fixed::types::I32F32;
use serde::{Deserialize, Serialize};

use super::zeckendorf::{zeckendorf_decomposition, DualZeckendorf};
use crate::cordic::{Cordic, PhiNum, PHI};

/// OHLCV bar (market data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    pub timestamp: i64,
    pub ticker: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

/// Encoded financial state in φ-space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialState {
    pub timestamp: i64,
    pub ticker: String,

    // Price encoding
    pub price_angle: I32F32,     // Price change → angle
    pub price_zeck: Vec<u64>,    // Zeckendorf decomposition of price
    pub price_phi: PhiNum,        // Price in φ-space

    // Volume encoding
    pub volume_energy: I32F32,   // Volume → energy level
    pub volume_zeck: Vec<u64>,   // Zeckendorf decomposition of volume

    // Volatility encoding
    pub volatility_angle: I32F32, // High-Low range → angle

    // Dual Zeckendorf lattice
    pub dual_zeck: DualZeckendorf,
}

/// Financial encoder
pub struct FinancialEncoder {
    cordic: Cordic,
    angular_sensitivity: f64,
    position: usize,
}

impl Default for FinancialEncoder {
    fn default() -> Self {
        Self {
            cordic: Cordic::default(),
            angular_sensitivity: 0.1, // 1% price change = 0.1 radians
            position: 0,
        }
    }
}

impl FinancialEncoder {
    pub fn new(angular_sensitivity: f64) -> Self {
        Self {
            cordic: Cordic::default(),
            angular_sensitivity,
            position: 0,
        }
    }

    /// Encode OHLCV bar to φ-space
    pub fn encode(&mut self, bar: &OHLCVBar) -> FinancialState {
        // Price change percentage
        let price_change_pct = if bar.open != 0.0 {
            ((bar.close - bar.open) / bar.open) * 100.0
        } else {
            0.0
        };

        // Price change → angle
        let price_angle_rad = (price_change_pct * self.angular_sensitivity * PHI)
            % (2.0 * std::f64::consts::PI);
        let price_angle = I32F32::from_num(price_angle_rad);

        // Encode price to Zeckendorf
        let price_int = (bar.close.abs() as u64).max(1);
        let price_zeck = zeckendorf_decomposition(price_int);

        // Price in φ-space
        let price_phi = PhiNum::from_value(bar.close.abs());

        // Volume → energy level
        // Normalize by position (exponential decay)
        let volume_normalized = (bar.volume as f64).ln() / 20.0; // Log scale
        let energy = volume_normalized * (PHI.powi(-(self.position as i32)));
        let volume_energy = I32F32::from_num(energy);

        // Encode volume to Zeckendorf
        let volume_int = ((bar.volume as f64).ln() as u64).max(1);
        let volume_zeck = zeckendorf_decomposition(volume_int);

        // Volatility (high-low range) → angle
        let volatility_pct = if bar.close != 0.0 {
            ((bar.high - bar.low) / bar.close) * 100.0
        } else {
            0.0
        };
        let volatility_angle_rad = (volatility_pct * self.angular_sensitivity)
            % (2.0 * std::f64::consts::PI);
        let volatility_angle = I32F32::from_num(volatility_angle_rad);

        // Dual Zeckendorf lattice
        let dual_zeck = DualZeckendorf::new(price_int);

        self.position += 1;

        FinancialState {
            timestamp: bar.timestamp,
            ticker: bar.ticker.clone(),
            price_angle,
            price_zeck,
            price_phi,
            volume_energy,
            volume_zeck,
            volatility_angle,
            dual_zeck,
        }
    }

    /// Encode multiple bars (batch processing)
    pub fn encode_batch(&mut self, bars: &[OHLCVBar]) -> Vec<FinancialState> {
        bars.iter().map(|bar| self.encode(bar)).collect()
    }

    /// Reset position counter
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Compute field coherence between two financial states
pub fn field_coherence(a: &FinancialState, b: &FinancialState) -> f64 {
    let cordic = Cordic::default();

    // Angle difference
    let angle_diff = (a.price_angle - b.price_angle).abs();

    // Energy difference
    let energy_diff = (a.volume_energy - b.volume_energy).abs();

    // Phi-space difference
    let phi_diff = (a.price_phi.exponent - b.price_phi.exponent).abs();

    // Combine into coherence score [0, 1]
    let total_diff = angle_diff.to_num::<f64>()
        + energy_diff.to_num::<f64>()
        + phi_diff.to_num::<f64>();

    // Coherence = 1 / (1 + difference)
    1.0 / (1.0 + total_diff)
}

/// Compute correlation matrix for multiple tickers
pub fn correlation_matrix(states: &[FinancialState]) -> Vec<Vec<f64>> {
    let n = states.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 1.0;
            } else {
                matrix[i][j] = field_coherence(&states[i], &states[j]);
            }
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bar(ticker: &str, close: f64) -> OHLCVBar {
        OHLCVBar {
            timestamp: 1700000000,
            ticker: ticker.to_string(),
            open: close * 0.99,
            high: close * 1.01,
            low: close * 0.98,
            close,
            volume: 1_000_000,
        }
    }

    #[test]
    fn test_encode_bar() {
        let mut encoder = FinancialEncoder::default();
        let bar = create_test_bar("AAPL", 182.50);

        let state = encoder.encode(&bar);

        assert_eq!(state.ticker, "AAPL");
        assert!(!state.price_zeck.is_empty());
        assert!(!state.volume_zeck.is_empty());
    }

    #[test]
    fn test_encode_batch() {
        let mut encoder = FinancialEncoder::default();
        let bars = vec![
            create_test_bar("AAPL", 182.50),
            create_test_bar("GOOGL", 141.20),
            create_test_bar("MSFT", 378.91),
        ];

        let states = encoder.encode_batch(&bars);

        assert_eq!(states.len(), 3);
        assert_eq!(states[0].ticker, "AAPL");
        assert_eq!(states[1].ticker, "GOOGL");
        assert_eq!(states[2].ticker, "MSFT");
    }

    #[test]
    fn test_field_coherence() {
        let mut encoder = FinancialEncoder::default();

        let bar1 = create_test_bar("AAPL", 182.50);
        let bar2 = create_test_bar("AAPL", 182.75); // Very similar

        let state1 = encoder.encode(&bar1);
        encoder.reset();
        let state2 = encoder.encode(&bar2);

        let coherence = field_coherence(&state1, &state2);

        // Should have high coherence (similar prices)
        assert!(coherence > 0.5);
    }

    #[test]
    fn test_correlation_matrix() {
        let mut encoder = FinancialEncoder::default();
        let bars = vec![
            create_test_bar("AAPL", 182.50),
            create_test_bar("GOOGL", 141.20),
        ];

        let states = encoder.encode_batch(&bars);
        let matrix = correlation_matrix(&states);

        // Diagonal should be 1.0 (perfect correlation with self)
        assert!((matrix[0][0] - 1.0).abs() < 0.001);
        assert!((matrix[1][1] - 1.0).abs() < 0.001);

        // Off-diagonal should be < 1.0
        assert!(matrix[0][1] < 1.0);
    }

    #[test]
    fn test_price_phi_encoding() {
        let mut encoder = FinancialEncoder::default();
        let bar = create_test_bar("AAPL", 182.50);

        let state = encoder.encode(&bar);

        // Price in φ-space should be positive
        assert!(state.price_phi.to_value() > 0.0);

        // Should be close to original price
        let price_recovered = state.price_phi.to_value();
        assert!((price_recovered - 182.50).abs() < 1.0);
    }

    #[test]
    fn test_zeckendorf_decomposition_sum() {
        let mut encoder = FinancialEncoder::default();
        let bar = create_test_bar("AAPL", 182.50);

        let state = encoder.encode(&bar);

        // Sum of Zeckendorf terms should equal price integer
        let sum: u64 = state.price_zeck.iter().sum();
        assert_eq!(sum, 182);
    }
}
