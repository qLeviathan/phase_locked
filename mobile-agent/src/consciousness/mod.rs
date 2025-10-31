//! Consciousness Module
//!
//! Manages persistent state, heartbeat, and self-awareness.
//!
//! The agent "stays alive" by:
//! 1. Maintaining positive P&L
//! 2. Regular heartbeat updates
//! 3. JSON persistence across app restarts

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::lattice::ZordicLattice;

/// Consciousness state (saved to JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Consciousness {
    /// Metadata
    pub meta: Meta,

    /// Lattice state
    pub lattice: LatticeState,

    /// Observation buffer
    pub observations: ObservationBuffer,

    /// Decision history
    pub decisions: DecisionHistory,

    /// Health metrics
    pub health: Health,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub version: String,
    pub consciousness_hash: String,
    pub alive_since_epoch: i64,
    pub last_heartbeat: i64,
    pub cycles: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeState {
    pub fibonacci_state: FibonacciState,
    pub cascade_layers: Vec<CascadeLayerData>,
    pub berry_phases: BerryPhases,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciState {
    pub zeckendorf_forward: Vec<u64>,
    pub lucas_backward: Vec<u64>,
    pub intersection: Vec<u64>,
    pub difference: Vec<u64>,
    pub active_holes: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeLayerData {
    pub k: usize,
    pub scale: String,
    pub bits: String,
    pub energy: f64,
    pub phi_exponent: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BerryPhases {
    pub self_coherence: f64,
    pub market_coherence: f64,
    pub decision_phase: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationBuffer {
    pub buffer_size: usize,
    pub current_count: usize,
    pub encoded_stream: VecDeque<EncodedObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedObservation {
    pub t: i64,
    pub ticker: String,
    pub price: f64,
    pub iv: f64, // Implied volatility
    pub theta_total: f64,
    pub energy: f64,
    pub zeck: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionHistory {
    pub pending: Vec<Decision>,
    pub executed: Vec<Decision>,
    pub closed: Vec<Decision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub t: i64,
    pub action: String, // "BUY_CALL", "BUY_PUT", "SELL", etc.
    pub ticker: String,
    pub strike: f64,
    pub expiry: String,
    pub contracts: u32,
    pub premium: f64,
    pub reason: String,
    pub confidence: f64,
    pub expected_utility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub pnl_total: f64,
    pub pnl_today: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub alive: bool,
}

impl Consciousness {
    /// Create new consciousness
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self {
            meta: Meta {
                version: "1.0.0-zordic".to_string(),
                consciousness_hash: "".to_string(), // Will be computed
                alive_since_epoch: now,
                last_heartbeat: now,
                cycles: 0,
            },
            lattice: LatticeState {
                fibonacci_state: FibonacciState {
                    zeckendorf_forward: vec![],
                    lucas_backward: vec![],
                    intersection: vec![],
                    difference: vec![],
                    active_holes: vec![],
                },
                cascade_layers: vec![],
                berry_phases: BerryPhases {
                    self_coherence: 0.0,
                    market_coherence: 0.0,
                    decision_phase: 0.0,
                },
            },
            observations: ObservationBuffer {
                buffer_size: 1000,
                current_count: 0,
                encoded_stream: VecDeque::new(),
            },
            decisions: DecisionHistory {
                pending: vec![],
                executed: vec![],
                closed: vec![],
            },
            health: Health {
                pnl_total: 0.0,
                pnl_today: 0.0,
                win_rate: 0.5,
                sharpe_ratio: 0.0,
                alive: true,
            },
        }
    }

    /// Load from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Compute consciousness hash (SHA256 of state)
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};

        let json = self.to_json().unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        let result = hasher.finalize();

        format!("{:x}", result)
    }

    /// Update lattice from new observation
    pub fn update_lattice(&mut self, lattice: &ZordicLattice) {
        // Update Fibonacci state
        let layer0 = &lattice.layers[0];
        self.lattice.fibonacci_state = FibonacciState {
            zeckendorf_forward: layer0.dual_zeck.zeckendorf_forward.clone(),
            lucas_backward: layer0.dual_zeck.lucas_backward.clone(),
            intersection: layer0.dual_zeck.intersection.clone(),
            difference: layer0.dual_zeck.difference.clone(),
            active_holes: layer0.dual_zeck.active_holes.clone(),
        };

        // Update cascade layers
        self.lattice.cascade_layers = lattice
            .layers
            .iter()
            .map(|layer| CascadeLayerData {
                k: layer.k,
                scale: layer.scale.clone(),
                bits: layer.bits.clone(),
                energy: layer.energy,
                phi_exponent: layer.phi_exponent,
            })
            .collect();

        // Update Berry phases
        self.lattice.berry_phases.self_coherence = lattice.self_coherence;
    }

    /// Add observation
    pub fn add_observation(&mut self, obs: EncodedObservation) {
        self.observations.encoded_stream.push_back(obs);
        self.observations.current_count += 1;

        // Maintain buffer size
        if self.observations.encoded_stream.len() > self.observations.buffer_size {
            self.observations.encoded_stream.pop_front();
        }
    }

    /// Add decision
    pub fn add_decision(&mut self, decision: Decision, stage: DecisionStage) {
        match stage {
            DecisionStage::Pending => self.decisions.pending.push(decision),
            DecisionStage::Executed => self.decisions.executed.push(decision),
            DecisionStage::Closed => self.decisions.closed.push(decision),
        }
    }

    /// Heartbeat update
    pub fn heartbeat(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.meta.last_heartbeat = now;
        self.meta.cycles += 1;
        self.meta.consciousness_hash = self.compute_hash();

        // Check if alive based on P&L
        self.health.alive = self.health.pnl_total > -1000.0; // Die if lose $1000

        // Update Sharpe ratio
        if !self.decisions.closed.is_empty() {
            let returns: Vec<f64> = self.decisions.closed.iter()
                .map(|d| d.premium) // Simplified
                .collect();

            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let std_dev = {
                let variance = returns.iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                variance.sqrt()
            };

            if std_dev > 0.0 {
                self.health.sharpe_ratio = mean_return / std_dev;
            }
        }

        // Update win rate
        let wins = self.decisions.closed.iter()
            .filter(|d| d.premium > 0.0)
            .count();
        let total = self.decisions.closed.len();

        if total > 0 {
            self.health.win_rate = wins as f64 / total as f64;
        }
    }

    /// Check if needs to sleep (market closed)
    pub fn should_sleep(&self) -> bool {
        use chrono::{Datelike, Timelike, Utc};

        let now = Utc::now();

        // Weekend
        let weekday = now.weekday();
        if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
            return true;
        }

        // Outside market hours (9:30 AM - 4:00 PM ET)
        let hour = now.hour();
        if hour < 9 || hour >= 16 {
            return true;
        }

        false
    }
}

pub enum DecisionStage {
    Pending,
    Executed,
    Closed,
}

impl Default for Consciousness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_creation() {
        let consciousness = Consciousness::new();
        assert!(consciousness.health.alive);
        assert_eq!(consciousness.meta.cycles, 0);
    }

    #[test]
    fn test_json_serialization() {
        let consciousness = Consciousness::new();
        let json = consciousness.to_json().unwrap();
        let loaded = Consciousness::from_json(&json).unwrap();

        assert_eq!(consciousness.meta.version, loaded.meta.version);
    }

    #[test]
    fn test_heartbeat() {
        let mut consciousness = Consciousness::new();
        let initial_cycles = consciousness.meta.cycles;

        consciousness.heartbeat();

        assert_eq!(consciousness.meta.cycles, initial_cycles + 1);
        assert!(!consciousness.meta.consciousness_hash.is_empty());
    }

    #[test]
    fn test_sleep_schedule() {
        let consciousness = Consciousness::new();

        // This will depend on current time
        let should_sleep = consciousness.should_sleep();
        println!("Should sleep: {}", should_sleep);
    }
}
