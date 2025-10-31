//! CORDIC Encoding Loop
//!
//! Continuously encodes OHLCV bars to φ-space
//! Target: <15μs per bar

use crate::metrics::{LatencyMeasurement, LatencyTimer, LatencyTracker};
use crate::shared_memory::{SerializedState, SharedBuffer};
use crossbeam::queue::ArrayQueue;
use phi_mamba_signals::{compute_berry_phase, FinancialEncoder, OHLCVBar};
use std::sync::Arc;
use tokio::time::{interval, Duration};

/// Encoding statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct EncoderStats {
    pub bars_processed: u64,
    pub avg_latency_us: f64,
    pub p99_latency_us: u64,
    pub throughput_hz: f64,
}

/// CORDIC encoder (runs in dedicated thread)
pub struct Encoder {
    /// Phi-Mamba encoder
    encoder: FinancialEncoder,

    /// Input queue (lock-free)
    input: Arc<ArrayQueue<OHLCVBar>>,

    /// Output buffer
    output: Arc<SharedBuffer>,

    /// Latency tracker
    metrics: LatencyTracker,

    /// Statistics
    bars_processed: u64,
}

impl Encoder {
    pub fn new(
        input: Arc<ArrayQueue<OHLCVBar>>,
        output: Arc<SharedBuffer>,
        metrics: LatencyTracker,
    ) -> Self {
        Self {
            encoder: FinancialEncoder::default(),
            input,
            output,
            metrics,
            bars_processed: 0,
        }
    }

    /// Run encoding loop (async)
    pub async fn run(mut self) {
        tracing::info!("CORDIC encoder started");

        let mut tick_interval = interval(Duration::from_micros(100)); // 10kHz poll rate

        loop {
            tick_interval.tick().await;

            // Process all available bars
            while let Some(bar) = self.input.pop() {
                let mut measurement = LatencyMeasurement::new();
                measurement.timestamp_us = chrono::Utc::now().timestamp_micros() as u64;

                // CORDIC encoding
                let timer = LatencyTimer::new("cordic");
                let state = self.encoder.encode(&bar);
                measurement.cordic_us = timer.elapsed_us();

                // Convert to serializable format
                let serialized = SerializedState::from(&state);

                // Push to shared buffer
                let timer = LatencyTimer::new("ipc");
                if !self.output.push_state(serialized) {
                    tracing::warn!("Shared buffer full, dropping state");
                }
                measurement.ipc_us = timer.elapsed_us();

                // Record metrics
                self.metrics.record(measurement);
                self.bars_processed += 1;

                // Log every 1000 bars
                if self.bars_processed % 1000 == 0 {
                    let stats = self.metrics.stats(Some(1000));
                    tracing::info!(
                        "Processed {} bars | Latency: p50={}μs p99={}μs violations={:.2}%",
                        self.bars_processed,
                        stats.p50_us,
                        stats.p99_us,
                        stats.violation_rate * 100.0
                    );
                }
            }
        }
    }

    /// Get encoder statistics
    pub fn stats(&self) -> EncoderStats {
        let latency_stats = self.metrics.stats(Some(1000));

        EncoderStats {
            bars_processed: self.bars_processed,
            avg_latency_us: latency_stats.mean_us,
            p99_latency_us: latency_stats.p99_us,
            throughput_hz: if latency_stats.mean_us > 0.0 {
                1_000_000.0 / latency_stats.mean_us
            } else {
                0.0
            },
        }
    }
}

/// Berry phase computer (runs periodically)
pub struct BerryComputer {
    /// Output buffer
    output: Arc<SharedBuffer>,

    /// Latency tracker
    metrics: LatencyTracker,

    /// Cache of recent states (for incremental computation)
    state_cache: Vec<phi_mamba_signals::FinancialState>,
}

impl BerryComputer {
    pub fn new(output: Arc<SharedBuffer>, metrics: LatencyTracker) -> Self {
        Self {
            output,
            metrics,
            state_cache: Vec::with_capacity(100),
        }
    }

    /// Run Berry phase computation loop
    pub async fn run(mut self) {
        tracing::info!("Berry phase computer started");

        let mut tick_interval = interval(Duration::from_millis(10)); // 100Hz

        loop {
            tick_interval.tick().await;

            // Get current states from buffer
            let states = self.output.drain_states();

            if states.is_empty() {
                continue;
            }

            // TODO: Convert SerializedState back to FinancialState for Berry phase computation
            // For now, just update the matrix with dummy data

            let n = states.len().min(10);
            let mut matrix = vec![vec![0.0; n]; n];

            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        matrix[i][j] = 0.0; // Self-correlation
                    } else {
                        // Placeholder: would compute actual Berry phase here
                        matrix[i][j] = 0.5;
                    }
                }
            }

            self.output.update_berry_matrix(matrix);
        }
    }
}

/// Tauri command to get encoder stats
#[tauri::command]
pub async fn get_encoder_stats(
    metrics: tauri::State<'_, Arc<LatencyTracker>>,
) -> Result<serde_json::Value, String> {
    let stats = metrics.stats(Some(1000));

    Ok(serde_json::json!({
        "bars_processed": stats.count,
        "avg_latency_us": stats.mean_us,
        "p50_latency_us": stats.p50_us,
        "p95_latency_us": stats.p95_us,
        "p99_latency_us": stats.p99_us,
        "max_latency_us": stats.max_us,
        "budget_violations": stats.budget_violations,
        "violation_rate": stats.violation_rate,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let input = Arc::new(ArrayQueue::new(1024));
        let output = Arc::new(SharedBuffer::new());
        let metrics = LatencyTracker::new();

        let encoder = Encoder::new(input, output, metrics);

        assert_eq!(encoder.bars_processed, 0);
    }

    #[tokio::test]
    async fn test_encoder_loop() {
        let input = Arc::new(ArrayQueue::new(1024));
        let output = Arc::new(SharedBuffer::new());
        let metrics = LatencyTracker::new();

        // Push test bar
        let bar = OHLCVBar {
            timestamp: 1700000000,
            ticker: "AAPL".to_string(),
            open: 180.0,
            high: 182.0,
            low: 179.0,
            close: 181.5,
            volume: 50_000_000,
        };

        input.push(bar).ok();

        let encoder = Encoder::new(input.clone(), output.clone(), metrics.clone());

        // Run for short time
        tokio::select! {
            _ = encoder.run() => {},
            _ = tokio::time::sleep(Duration::from_millis(100)) => {},
        }

        // Check output buffer received state
        let states = output.drain_states();
        assert!(!states.is_empty());
    }
}
