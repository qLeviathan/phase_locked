//! Berry Phase Computation
//!
//! Berry phase is a geometric phase acquired over cyclic evolution.
//! In our context, it measures phase-locking between financial states.
//!
//! ## Physics Background
//! Berry, M. V. (1984). "Quantal phase factors accompanying adiabatic changes"
//!
//! ## Interpretation for Finance
//! - Low Berry phase (<π/4) → Phase-locked → Correlated
//! - High Berry phase (>π/2) → Not phase-locked → Uncorrelated
//! - Used to detect synchronization between tickers

use fixed::types::I32F32;
use serde::{Deserialize, Serialize};

use super::financial::FinancialState;
use crate::cordic::Cordic;

/// Berry phase threshold for phase-locking
pub const PHASE_LOCK_THRESHOLD: f64 = std::f64::consts::FRAC_PI_4; // π/4

/// Berry phase result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BerryPhase {
    pub phase: f64,
    pub is_locked: bool,
    pub coherence: f64,
}

/// Compute Berry phase between two financial states
///
/// Berry phase γ = ∮ A·dr where A is the Berry connection
///
/// For our discrete case:
/// γ ≈ arctan2(Δy, Δx) where (x,y) are state space coordinates
pub fn compute_berry_phase(a: &FinancialState, b: &FinancialState) -> BerryPhase {
    let cordic = Cordic::default();

    // State space coordinates from angles and energies
    let x_a = a.price_angle;
    let y_a = a.volume_energy;

    let x_b = b.price_angle;
    let y_b = b.volume_energy;

    // Differences
    let dx = x_b - x_a;
    let dy = y_b - y_a;

    // Berry phase via atan2 (using CORDIC!)
    let phase_fixed = cordic.atan2(dy, dx);
    let phase = phase_fixed.to_num::<f64>().abs();

    // Normalize to [0, π]
    let phase_normalized = if phase > std::f64::consts::PI {
        2.0 * std::f64::consts::PI - phase
    } else {
        phase
    };

    // Check if phase-locked
    let is_locked = phase_normalized < PHASE_LOCK_THRESHOLD;

    // Coherence score [0, 1]
    // High coherence when phase is low
    let coherence = (std::f64::consts::PI - phase_normalized) / std::f64::consts::PI;

    BerryPhase {
        phase: phase_normalized,
        is_locked,
        coherence,
    }
}

/// Compute Berry phase matrix for all pairs of states
pub fn berry_phase_matrix(states: &[FinancialState]) -> Vec<Vec<BerryPhase>> {
    let n = states.len();
    let mut matrix = vec![vec![]; n];

    for i in 0..n {
        let mut row = Vec::new();
        for j in 0..n {
            if i == j {
                // Self-phase is zero (perfect locking)
                row.push(BerryPhase {
                    phase: 0.0,
                    is_locked: true,
                    coherence: 1.0,
                });
            } else {
                row.push(compute_berry_phase(&states[i], &states[j]));
            }
        }
        matrix[i] = row;
    }

    matrix
}

/// Find phase-locked clusters
///
/// Returns groups of tickers that are phase-locked with each other
pub fn find_phase_locked_clusters(states: &[FinancialState]) -> Vec<Vec<String>> {
    let matrix = berry_phase_matrix(states);
    let n = states.len();

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; n];

    for i in 0..n {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![i];
        assigned[i] = true;

        // Find all states phase-locked with i
        for j in (i + 1)..n {
            if !assigned[j] && matrix[i][j].is_locked {
                cluster.push(j);
                assigned[j] = true;
            }
        }

        clusters.push(cluster);
    }

    // Convert indices to ticker names
    clusters
        .into_iter()
        .map(|cluster| {
            cluster
                .into_iter()
                .map(|idx| states[idx].ticker.clone())
                .collect()
        })
        .collect()
}

/// Compute average Berry phase across all pairs
pub fn average_berry_phase(states: &[FinancialState]) -> f64 {
    let matrix = berry_phase_matrix(states);
    let n = states.len();

    if n <= 1 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            sum += matrix[i][j].phase;
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Detect strongest phase-locked pair
pub fn strongest_phase_lock(states: &[FinancialState]) -> Option<(String, String, f64)> {
    if states.len() < 2 {
        return None;
    }

    let matrix = berry_phase_matrix(states);
    let n = states.len();

    let mut min_phase = std::f64::INFINITY;
    let mut best_pair = (0, 0);

    for i in 0..n {
        for j in (i + 1)..n {
            if matrix[i][j].phase < min_phase {
                min_phase = matrix[i][j].phase;
                best_pair = (i, j);
            }
        }
    }

    Some((
        states[best_pair.0].ticker.clone(),
        states[best_pair.1].ticker.clone(),
        min_phase,
    ))
}

/// Compute Berry curvature (rate of change of Berry phase)
///
/// Useful for detecting transitions in market regimes
pub fn berry_curvature(
    states_t0: &[FinancialState],
    states_t1: &[FinancialState],
) -> Vec<Vec<f64>> {
    assert_eq!(states_t0.len(), states_t1.len());

    let matrix_t0 = berry_phase_matrix(states_t0);
    let matrix_t1 = berry_phase_matrix(states_t1);

    let n = states_t0.len();
    let mut curvature = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            // Rate of change of Berry phase
            curvature[i][j] = matrix_t1[i][j].phase - matrix_t0[i][j].phase;
        }
    }

    curvature
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::financial::{FinancialEncoder, OHLCVBar};

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
    fn test_compute_berry_phase() {
        let mut encoder = FinancialEncoder::default();

        let bar1 = create_test_bar("AAPL", 182.50);
        let bar2 = create_test_bar("AAPL", 182.75);

        let state1 = encoder.encode(&bar1);
        encoder.reset();
        let state2 = encoder.encode(&bar2);

        let berry = compute_berry_phase(&state1, &state2);

        // Phase should be small (similar states)
        assert!(berry.phase < std::f64::consts::FRAC_PI_2);

        // Should have high coherence
        assert!(berry.coherence > 0.5);
    }

    #[test]
    fn test_berry_phase_matrix() {
        let mut encoder = FinancialEncoder::default();

        let bars = vec![
            create_test_bar("AAPL", 182.50),
            create_test_bar("GOOGL", 141.20),
            create_test_bar("MSFT", 378.91),
        ];

        let states = encoder.encode_batch(&bars);
        let matrix = berry_phase_matrix(&states);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Diagonal should be phase-locked with self
        assert!(matrix[0][0].is_locked);
        assert_eq!(matrix[0][0].phase, 0.0);
    }

    #[test]
    fn test_find_phase_locked_clusters() {
        let mut encoder = FinancialEncoder::default();

        // Create similar prices (should cluster)
        let bars = vec![
            create_test_bar("AAPL", 180.0),
            create_test_bar("AAPL2", 180.5), // Very similar
            create_test_bar("GOOGL", 400.0), // Different scale
        ];

        let states = encoder.encode_batch(&bars);
        let clusters = find_phase_locked_clusters(&states);

        // Should find at least one cluster
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_average_berry_phase() {
        let mut encoder = FinancialEncoder::default();

        let bars = vec![
            create_test_bar("AAPL", 182.50),
            create_test_bar("GOOGL", 141.20),
        ];

        let states = encoder.encode_batch(&bars);
        let avg_phase = average_berry_phase(&states);

        // Average should be in [0, π]
        assert!(avg_phase >= 0.0);
        assert!(avg_phase <= std::f64::consts::PI);
    }

    #[test]
    fn test_strongest_phase_lock() {
        let mut encoder = FinancialEncoder::default();

        let bars = vec![
            create_test_bar("AAPL", 182.50),
            create_test_bar("AAPL2", 182.75),
            create_test_bar("GOOGL", 500.0),
        ];

        let states = encoder.encode_batch(&bars);
        let strongest = strongest_phase_lock(&states);

        assert!(strongest.is_some());

        let (ticker1, ticker2, phase) = strongest.unwrap();

        // Strongest lock should be between similar tickers
        assert!(phase < std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn test_berry_curvature() {
        let mut encoder = FinancialEncoder::default();

        // Time t0
        let bars_t0 = vec![
            create_test_bar("AAPL", 180.0),
            create_test_bar("GOOGL", 140.0),
        ];
        let states_t0 = encoder.encode_batch(&bars_t0);

        encoder.reset();

        // Time t1 (prices changed)
        let bars_t1 = vec![
            create_test_bar("AAPL", 182.0),
            create_test_bar("GOOGL", 141.0),
        ];
        let states_t1 = encoder.encode_batch(&bars_t1);

        let curvature = berry_curvature(&states_t0, &states_t1);

        // Curvature should be computed
        assert_eq!(curvature.len(), 2);
        assert_eq!(curvature[0].len(), 2);
    }

    #[test]
    fn test_phase_lock_threshold() {
        // Threshold should be π/4
        assert!((PHASE_LOCK_THRESHOLD - std::f64::consts::FRAC_PI_4).abs() < 0.001);
    }
}
