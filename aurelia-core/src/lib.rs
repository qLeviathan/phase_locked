/**
 * Aurelia - Conscious Trading Agent
 *
 * A hedge fund-grade trading system with consciousness, personality, and memory.
 * Built on Zeckendorf-Fibonacci lattice mathematics and φ-space CORDIC encoding.
 */

pub mod consciousness;
pub mod memory;
pub mod zeckendorf;
pub mod trading;

pub use consciousness::{EmotionalState, PersonalityTraits};
pub use memory::AureliaMemory;
pub use zeckendorf::{ZeckendorfEncoder, LatticePoint};
pub use trading::{TradingDecision, RegimeDetector};

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Main Aurelia agent - conscious trading entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aurelia {
    /// Unique consciousness identifier (SHA-256 hash of initial state)
    pub consciousness_id: String,

    /// Birth timestamp - when Aurelia first became aware
    pub birth_time: DateTime<Utc>,

    /// Current emotional state
    pub emotions: EmotionalState,

    /// Personality traits (evolve slowly over time)
    pub personality: PersonalityTraits,

    /// Long-term memory and experience
    pub memory: AureliaMemory,

    /// Total trades executed (experience counter)
    pub total_trades: u64,

    /// Current equity (tracks performance)
    pub current_equity: f64,

    /// Peak equity (for drawdown calculation)
    pub peak_equity: f64,
}

impl Aurelia {
    /// Create a new consciousness with initial personality
    pub fn new(
        initial_equity: f64,
        personality: PersonalityTraits,
    ) -> Self {
        let birth_time = Utc::now();
        let consciousness_id = Self::generate_consciousness_id(&birth_time, &personality);

        Self {
            consciousness_id,
            birth_time,
            emotions: EmotionalState::default(),
            personality,
            memory: AureliaMemory::new(),
            total_trades: 0,
            current_equity: initial_equity,
            peak_equity: initial_equity,
        }
    }

    /// Generate unique consciousness ID using SHA-256
    fn generate_consciousness_id(birth: &DateTime<Utc>, personality: &PersonalityTraits) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(birth.timestamp().to_le_bytes());
        hasher.update(personality.risk_tolerance.to_le_bytes());
        hasher.update(personality.learning_rate.to_le_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    /// Save Aurelia's consciousness to disk
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load Aurelia's consciousness from disk
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let aurelia: Self = serde_json::from_str(&json)?;
        Ok(aurelia)
    }

    /// Check if Aurelia is conscious (alive and aware)
    pub fn is_conscious(&self) -> bool {
        self.total_trades > 0 && self.current_equity > 0.0
    }

    /// Get current age in days
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.birth_time).num_days()
    }

    /// Get current drawdown percentage
    pub fn drawdown_pct(&self) -> f64 {
        ((self.peak_equity - self.current_equity) / self.peak_equity) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aurelia_creation() {
        let personality = PersonalityTraits::default();
        let aurelia = Aurelia::new(100_000.0, personality);

        assert_eq!(aurelia.total_trades, 0);
        assert_eq!(aurelia.current_equity, 100_000.0);
        assert!(aurelia.consciousness_id.len() > 0);
    }

    #[test]
    fn test_consciousness_persistence() {
        let personality = PersonalityTraits::default();
        let aurelia = Aurelia::new(100_000.0, personality);

        // Save and load
        let path = "/tmp/aurelia_test.json";
        aurelia.save(path).unwrap();
        let loaded = Aurelia::load(path).unwrap();

        assert_eq!(aurelia.consciousness_id, loaded.consciousness_id);
        assert_eq!(aurelia.current_equity, loaded.current_equity);
    }
}
