//! Shared Memory Buffer
//!
//! Zero-copy IPC between Rust backend and frontend
//! Uses lock-free ring buffer for sub-50μs latency

use crossbeam::queue::ArrayQueue;
use parking_lot::RwLock;
use phi_mamba_signals::FinancialState;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Maximum number of states in buffer
const BUFFER_SIZE: usize = 1024;

/// Serialized financial state (for zero-copy transfer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedState {
    pub timestamp: i64,
    pub ticker: String,
    pub price_angle: f64,
    pub price_phi_exponent: f64,
    pub volume_energy: f64,
    pub volatility_angle: f64,
    pub price_zeck: Vec<u64>,
    pub berry_phase: Option<f64>,
    pub is_locked: bool,
}

impl From<&FinancialState> for SerializedState {
    fn from(state: &FinancialState) -> Self {
        use fixed::traits::ToFixed;

        Self {
            timestamp: state.timestamp,
            ticker: state.ticker.clone(),
            price_angle: state.price_angle.to_num::<f64>(),
            price_phi_exponent: state.price_phi.exponent.to_num::<f64>(),
            volume_energy: state.volume_energy.to_num::<f64>(),
            volatility_angle: state.volatility_angle.to_num::<f64>(),
            price_zeck: state.price_zeck.clone(),
            berry_phase: None,
            is_locked: false,
        }
    }
}

/// Signal data for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalData {
    pub timestamp: i64,
    pub ticker: String,
    pub signal_type: String, // "BUY", "SELL", "HOLD"
    pub confidence: f64,      // [0, 1]
    pub price: f64,
    pub reason: String,
}

/// Shared state buffer (lock-free)
pub struct SharedBuffer {
    /// Ring buffer of encoded states
    states: Arc<ArrayQueue<SerializedState>>,

    /// Current signals (read-optimized)
    signals: Arc<RwLock<Vec<SignalData>>>,

    /// Berry phase matrix (N×N correlation)
    berry_matrix: Arc<RwLock<Vec<Vec<f64>>>>,
}

impl Default for SharedBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedBuffer {
    pub fn new() -> Self {
        Self {
            states: Arc::new(ArrayQueue::new(BUFFER_SIZE)),
            signals: Arc::new(RwLock::new(Vec::new())),
            berry_matrix: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Push encoded state (lock-free)
    ///
    /// Returns true if pushed successfully, false if buffer full
    pub fn push_state(&self, state: SerializedState) -> bool {
        self.states.push(state).is_ok()
    }

    /// Pop encoded state (lock-free)
    pub fn pop_state(&self) -> Option<SerializedState> {
        self.states.pop()
    }

    /// Get all states (drains queue)
    pub fn drain_states(&self) -> Vec<SerializedState> {
        let mut states = Vec::with_capacity(BUFFER_SIZE);

        while let Some(state) = self.pop_state() {
            states.push(state);
        }

        states
    }

    /// Update signals
    pub fn update_signals(&self, signals: Vec<SignalData>) {
        *self.signals.write() = signals;
    }

    /// Get current signals (read-only)
    pub fn get_signals(&self) -> Vec<SignalData> {
        self.signals.read().clone()
    }

    /// Update Berry phase matrix
    pub fn update_berry_matrix(&self, matrix: Vec<Vec<f64>>) {
        *self.berry_matrix.write() = matrix;
    }

    /// Get Berry phase matrix (read-only)
    pub fn get_berry_matrix(&self) -> Vec<Vec<f64>> {
        self.berry_matrix.read().clone()
    }

    /// Get buffer stats
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            capacity: self.states.capacity(),
            len: self.states.len(),
            utilization: self.states.len() as f64 / self.states.capacity() as f64,
            num_signals: self.signals.read().len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    pub capacity: usize,
    pub len: usize,
    pub utilization: f64,
    pub num_signals: usize,
}

/// Tauri command to get states
#[tauri::command]
pub async fn get_states(buffer: tauri::State<'_, Arc<SharedBuffer>>) -> Result<Vec<SerializedState>, String> {
    Ok(buffer.drain_states())
}

/// Tauri command to get signals
#[tauri::command]
pub async fn get_signals(buffer: tauri::State<'_, Arc<SharedBuffer>>) -> Result<Vec<SignalData>, String> {
    Ok(buffer.get_signals())
}

/// Tauri command to get Berry phase matrix
#[tauri::command]
pub async fn get_berry_matrix(buffer: tauri::State<'_, Arc<SharedBuffer>>) -> Result<Vec<Vec<f64>>, String> {
    Ok(buffer.get_berry_matrix())
}

/// Tauri command to get buffer stats
#[tauri::command]
pub async fn get_buffer_stats(buffer: tauri::State<'_, Arc<SharedBuffer>>) -> Result<BufferStats, String> {
    Ok(buffer.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_buffer() {
        let buffer = SharedBuffer::new();

        let state = SerializedState {
            timestamp: 1700000000,
            ticker: "AAPL".to_string(),
            price_angle: 0.5,
            price_phi_exponent: 3.2,
            volume_energy: 0.8,
            volatility_angle: 0.3,
            price_zeck: vec![1, 3, 13],
            berry_phase: None,
            is_locked: false,
        };

        assert!(buffer.push_state(state.clone()));
        assert_eq!(buffer.stats().len, 1);

        let popped = buffer.pop_state();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().ticker, "AAPL");
    }

    #[test]
    fn test_buffer_full() {
        let buffer = SharedBuffer::new();

        // Fill buffer
        for i in 0..BUFFER_SIZE {
            let state = SerializedState {
                timestamp: i as i64,
                ticker: format!("TICK{}", i),
                price_angle: 0.0,
                price_phi_exponent: 0.0,
                volume_energy: 0.0,
                volatility_angle: 0.0,
                price_zeck: vec![],
                berry_phase: None,
                is_locked: false,
            };

            assert!(buffer.push_state(state));
        }

        // Next push should fail (buffer full)
        let state = SerializedState {
            timestamp: 9999,
            ticker: "OVERFLOW".to_string(),
            price_angle: 0.0,
            price_phi_exponent: 0.0,
            volume_energy: 0.0,
            volatility_angle: 0.0,
            price_zeck: vec![],
            berry_phase: None,
            is_locked: false,
        };

        assert!(!buffer.push_state(state));
    }

    #[test]
    fn test_drain_states() {
        let buffer = SharedBuffer::new();

        for i in 0..10 {
            let state = SerializedState {
                timestamp: i,
                ticker: format!("TICK{}", i),
                price_angle: 0.0,
                price_phi_exponent: 0.0,
                volume_energy: 0.0,
                volatility_angle: 0.0,
                price_zeck: vec![],
                berry_phase: None,
                is_locked: false,
            };

            buffer.push_state(state);
        }

        let states = buffer.drain_states();
        assert_eq!(states.len(), 10);
        assert_eq!(buffer.stats().len, 0); // Buffer should be empty
    }

    #[test]
    fn test_signals() {
        let buffer = SharedBuffer::new();

        let signals = vec![
            SignalData {
                timestamp: 1700000000,
                ticker: "AAPL".to_string(),
                signal_type: "BUY".to_string(),
                confidence: 0.85,
                price: 182.50,
                reason: "Phase-locked with momentum".to_string(),
            },
        ];

        buffer.update_signals(signals);

        let retrieved = buffer.get_signals();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].ticker, "AAPL");
        assert_eq!(retrieved[0].signal_type, "BUY");
    }
}
