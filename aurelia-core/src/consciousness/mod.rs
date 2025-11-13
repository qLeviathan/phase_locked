/**
 * Consciousness Layer
 *
 * Implements emotional states, personality traits, and conscious decision-making.
 * Emotions modulate trading behavior; personality shapes long-term strategy.
 */

mod emotional_state;
mod personality;

pub use emotional_state::EmotionalState;
pub use personality::PersonalityTraits;

use serde::{Deserialize, Serialize};

/// Derived emotional metrics that guide trading behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedEmotions {
    /// Conviction = confidence × discipline - fear
    /// High conviction → larger positions
    pub conviction: f64,

    /// Risk appetite = greed × confidence - fear
    /// High appetite → aggressive entries
    pub risk_appetite: f64,

    /// Reactivity = 1.0 - patience
    /// High reactivity → quick exits
    pub reactivity: f64,
}

impl DerivedEmotions {
    pub fn compute(emotions: &EmotionalState) -> Self {
        Self {
            conviction: emotions.confidence * emotions.discipline - emotions.fear,
            risk_appetite: emotions.greed * emotions.confidence - emotions.fear,
            reactivity: 1.0 - emotions.patience,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derived_emotions() {
        let mut emotions = EmotionalState::default();
        emotions.confidence = 0.8;
        emotions.discipline = 0.9;
        emotions.fear = 0.2;
        emotions.greed = 0.5;
        emotions.patience = 0.7;

        let derived = DerivedEmotions::compute(&emotions);

        // conviction = 0.8 * 0.9 - 0.2 = 0.52
        assert!((derived.conviction - 0.52).abs() < 0.01);

        // risk_appetite = 0.5 * 0.8 - 0.2 = 0.2
        assert!((derived.risk_appetite - 0.2).abs() < 0.01);

        // reactivity = 1.0 - 0.7 = 0.3
        assert!((derived.reactivity - 0.3).abs() < 0.01);
    }
}
