//! Latency Monitoring
//!
//! Every operation is measured. Budget: <1ms end-to-end

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Latency budget (microseconds)
pub const LATENCY_BUDGET_US: u64 = 1000; // 1ms

/// Component latency targets (microseconds)
pub const TARGET_WEBSOCKET_US: u64 = 100;   // 0.1ms
pub const TARGET_CORDIC_US: u64 = 15;       // 15μs
pub const TARGET_BERRY_US: u64 = 2;         // 2μs
pub const TARGET_IPC_US: u64 = 50;          // 0.05ms
pub const TARGET_TOTAL_US: u64 = 800;       // 0.8ms (20% margin)

/// Latency measurement for single tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    /// When tick was received
    pub timestamp_us: u64,

    /// Component latencies (microseconds)
    pub websocket_us: u64,
    pub cordic_us: u64,
    pub berry_us: u64,
    pub ipc_us: u64,
    pub total_us: u64,

    /// Budget violations
    pub exceeds_budget: bool,
    pub slowest_component: String,
}

impl LatencyMeasurement {
    pub fn new() -> Self {
        Self {
            timestamp_us: 0,
            websocket_us: 0,
            cordic_us: 0,
            berry_us: 0,
            ipc_us: 0,
            total_us: 0,
            exceeds_budget: false,
            slowest_component: String::new(),
        }
    }

    /// Check if any component exceeds its target
    pub fn check_budgets(&mut self) {
        self.total_us = self.websocket_us + self.cordic_us + self.berry_us + self.ipc_us;
        self.exceeds_budget = self.total_us > LATENCY_BUDGET_US;

        // Find slowest component
        let mut max = 0u64;
        let mut name = String::new();

        if self.websocket_us > max {
            max = self.websocket_us;
            name = "websocket".to_string();
        }
        if self.cordic_us > max {
            max = self.cordic_us;
            name = "cordic".to_string();
        }
        if self.berry_us > max {
            max = self.berry_us;
            name = "berry".to_string();
        }
        if self.ipc_us > max {
            max = self.ipc_us;
            name = "ipc".to_string();
        }

        self.slowest_component = name;
    }
}

/// Aggregated latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: u64,
    pub mean_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub budget_violations: u64,
    pub violation_rate: f64,
}

/// Latency tracker (thread-safe)
#[derive(Clone)]
pub struct LatencyTracker {
    measurements: Arc<parking_lot::RwLock<Vec<LatencyMeasurement>>>,
    total_count: Arc<AtomicU64>,
    violation_count: Arc<AtomicU64>,
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self {
            measurements: Arc::new(parking_lot::RwLock::new(Vec::with_capacity(10000))),
            total_count: Arc::new(AtomicU64::new(0)),
            violation_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record measurement
    pub fn record(&self, mut measurement: LatencyMeasurement) {
        measurement.check_budgets();

        self.total_count.fetch_add(1, Ordering::Relaxed);

        if measurement.exceeds_budget {
            self.violation_count.fetch_add(1, Ordering::Relaxed);

            tracing::warn!(
                "LATENCY BUDGET EXCEEDED: {}μs > {}μs (slowest: {})",
                measurement.total_us,
                LATENCY_BUDGET_US,
                measurement.slowest_component
            );
        }

        let mut lock = self.measurements.write();

        // Keep last 10k measurements
        if lock.len() >= 10000 {
            lock.remove(0);
        }

        lock.push(measurement);
    }

    /// Get statistics for last N measurements
    pub fn stats(&self, last_n: Option<usize>) -> LatencyStats {
        let lock = self.measurements.read();

        let measurements: Vec<_> = if let Some(n) = last_n {
            lock.iter().rev().take(n).cloned().collect()
        } else {
            lock.clone()
        };

        if measurements.is_empty() {
            return LatencyStats {
                count: 0,
                mean_us: 0.0,
                min_us: 0,
                max_us: 0,
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
                budget_violations: 0,
                violation_rate: 0.0,
            };
        }

        let mut totals: Vec<u64> = measurements.iter().map(|m| m.total_us).collect();
        totals.sort_unstable();

        let count = totals.len() as u64;
        let sum: u64 = totals.iter().sum();
        let mean = sum as f64 / count as f64;

        let min = totals[0];
        let max = totals[totals.len() - 1];

        let p50_idx = (count as f64 * 0.50) as usize;
        let p95_idx = (count as f64 * 0.95) as usize;
        let p99_idx = (count as f64 * 0.99) as usize;

        let p50 = totals[p50_idx.min(totals.len() - 1)];
        let p95 = totals[p95_idx.min(totals.len() - 1)];
        let p99 = totals[p99_idx.min(totals.len() - 1)];

        let violations = measurements
            .iter()
            .filter(|m| m.exceeds_budget)
            .count() as u64;

        LatencyStats {
            count,
            mean_us: mean,
            min_us: min,
            max_us: max,
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
            budget_violations: violations,
            violation_rate: violations as f64 / count as f64,
        }
    }

    /// Clear all measurements
    pub fn clear(&self) {
        self.measurements.write().clear();
        self.total_count.store(0, Ordering::Relaxed);
        self.violation_count.store(0, Ordering::Relaxed);
    }
}

/// Timer for measuring component latency
pub struct LatencyTimer {
    start: Instant,
    component: String,
}

impl LatencyTimer {
    pub fn new(component: &str) -> Self {
        Self {
            start: Instant::now(),
            component: component.to_string(),
        }
    }

    /// Get elapsed microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Check if exceeded budget
    pub fn exceeds_budget(&self, budget_us: u64) -> bool {
        self.elapsed_us() > budget_us
    }
}

impl Drop for LatencyTimer {
    fn drop(&mut self) {
        let elapsed = self.elapsed_us();
        tracing::trace!("{}: {}μs", self.component, elapsed);
    }
}

/// Helper macros
#[macro_export]
macro_rules! measure {
    ($name:expr, $body:expr) => {{
        let _timer = $crate::metrics::LatencyTimer::new($name);
        $body
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_measurement() {
        let mut m = LatencyMeasurement::new();
        m.websocket_us = 100;
        m.cordic_us = 15;
        m.berry_us = 2;
        m.ipc_us = 50;
        m.check_budgets();

        assert_eq!(m.total_us, 167);
        assert!(!m.exceeds_budget);
        assert_eq!(m.slowest_component, "websocket");
    }

    #[test]
    fn test_budget_violation() {
        let mut m = LatencyMeasurement::new();
        m.websocket_us = 2000; // Exceeds budget!
        m.check_budgets();

        assert!(m.exceeds_budget);
    }

    #[test]
    fn test_latency_tracker() {
        let tracker = LatencyTracker::new();

        for i in 0..100 {
            let mut m = LatencyMeasurement::new();
            m.total_us = 500 + i * 10;
            m.check_budgets();
            tracker.record(m);
        }

        let stats = tracker.stats(None);

        assert_eq!(stats.count, 100);
        assert!(stats.mean_us > 0.0);
        assert!(stats.min_us <= stats.max_us);
        assert!(stats.p50_us <= stats.p95_us);
        assert!(stats.p95_us <= stats.p99_us);
    }

    #[test]
    fn test_latency_timer() {
        let timer = LatencyTimer::new("test");
        std::thread::sleep(Duration::from_micros(100));
        let elapsed = timer.elapsed_us();

        assert!(elapsed >= 100);
        assert!(elapsed < 1000); // Should be ~100μs, not ms
    }
}
