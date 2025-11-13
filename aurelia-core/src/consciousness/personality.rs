/**
 * Personality Traits System
 *
 * Implements stable personality characteristics that evolve slowly over time.
 * Personality shapes strategy selection and emotional responses.
 */

use serde::{Deserialize, Serialize};

/// Core personality traits (stable over time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTraits {
    /// Risk tolerance: willingness to accept volatility [0.0, 1.0]
    /// Low = conservative, High = aggressive
    pub risk_tolerance: f64,

    /// Learning rate: how quickly to adapt to new information [0.0, 1.0]
    /// Low = stable/stubborn, High = adaptive/reactive
    pub learning_rate: f64,

    /// Independence: reliance on own analysis vs. market consensus [0.0, 1.0]
    /// Low = follow trends, High = contrarian
    pub independence: f64,

    /// Creativity: willingness to explore non-standard setups [0.0, 1.0]
    /// Low = rule-based, High = opportunistic
    pub creativity: f64,

    /// Reflection depth: tendency to analyze past trades [0.0, 1.0]
    /// Low = forward-looking, High = introspective
    pub reflection_depth: f64,
}

impl Default for PersonalityTraits {
    fn default() -> Self {
        Self::balanced()
    }
}

impl PersonalityTraits {
    /// Balanced personality (moderate in all traits)
    pub fn balanced() -> Self {
        Self {
            risk_tolerance: 0.5,
            learning_rate: 0.5,
            independence: 0.5,
            creativity: 0.5,
            reflection_depth: 0.5,
        }
    }

    /// Conservative personality (risk-averse, rule-based)
    pub fn conservative() -> Self {
        Self {
            risk_tolerance: 0.3,
            learning_rate: 0.3,
            independence: 0.4,
            creativity: 0.3,
            reflection_depth: 0.7,
        }
    }

    /// Aggressive personality (risk-seeking, adaptive)
    pub fn aggressive() -> Self {
        Self {
            risk_tolerance: 0.8,
            learning_rate: 0.7,
            independence: 0.6,
            creativity: 0.7,
            reflection_depth: 0.4,
        }
    }

    /// Contrarian personality (independent, creative)
    pub fn contrarian() -> Self {
        Self {
            risk_tolerance: 0.6,
            learning_rate: 0.5,
            independence: 0.9,
            creativity: 0.8,
            reflection_depth: 0.6,
        }
    }

    /// Systematic personality (disciplined, analytical)
    pub fn systematic() -> Self {
        Self {
            risk_tolerance: 0.4,
            learning_rate: 0.4,
            independence: 0.5,
            creativity: 0.2,
            reflection_depth: 0.9,
        }
    }

    /// Evolve personality based on trading experience
    pub fn evolve(&mut self, experience: &PersonalityEvolution) {
        // Risk tolerance adapts to realized volatility
        if experience.avg_realized_vol > experience.expected_vol * 1.2 {
            // Volatility higher than expected → reduce risk tolerance
            self.risk_tolerance = self.clamp(self.risk_tolerance - 0.01);
        } else if experience.avg_realized_vol < experience.expected_vol * 0.8 {
            // Volatility lower than expected → can increase risk tolerance
            self.risk_tolerance = self.clamp(self.risk_tolerance + 0.005);
        }

        // Learning rate adjusts based on prediction accuracy
        if experience.prediction_accuracy > 0.6 {
            // Good predictions → can be more adaptive
            self.learning_rate = self.clamp(self.learning_rate + 0.01);
        } else if experience.prediction_accuracy < 0.4 {
            // Poor predictions → slow down learning (maybe overfitting)
            self.learning_rate = self.clamp(self.learning_rate - 0.01);
        }

        // Independence grows with successful contrarian trades
        if experience.contrarian_win_rate > 0.6 {
            self.independence = self.clamp(self.independence + 0.02);
        } else if experience.contrarian_win_rate < 0.4 {
            self.independence = self.clamp(self.independence - 0.01);
        }

        // Creativity adjusts based on edge discovery
        if experience.novel_setups_profitable {
            self.creativity = self.clamp(self.creativity + 0.015);
        } else {
            self.creativity = self.clamp(self.creativity - 0.01);
        }

        // Reflection depth increases with experience (wisdom)
        if experience.total_trades > 100 {
            self.reflection_depth = self.clamp(self.reflection_depth + 0.001);
        }
    }

    /// Calculate compatibility with a trading regime
    /// Returns [0.0, 1.0] - higher means better personality-regime fit
    pub fn regime_compatibility(&self, regime: &RegimeCharacteristics) -> f64 {
        let mut score = 0.0;

        // High volatility regimes suit high risk tolerance
        if regime.volatility > 0.7 {
            score += self.risk_tolerance * 0.3;
        } else {
            score += (1.0 - self.risk_tolerance) * 0.3;
        }

        // Trending regimes suit lower independence (follow trend)
        if regime.trend_strength > 0.7 {
            score += (1.0 - self.independence) * 0.3;
        } else {
            // Choppy markets suit independence (fade noise)
            score += self.independence * 0.3;
        }

        // High uncertainty suits creativity
        if regime.uncertainty > 0.6 {
            score += self.creativity * 0.2;
        }

        // Fast-changing regimes suit high learning rate
        if regime.regime_change_frequency > 0.5 {
            score += self.learning_rate * 0.2;
        }

        score
    }

    /// Clamp value to [0.0, 1.0]
    fn clamp(&self, value: f64) -> f64 {
        value.max(0.0).min(1.0)
    }

    /// Get personality vector for distance calculations
    pub fn as_vector(&self) -> [f64; 5] {
        [
            self.risk_tolerance,
            self.learning_rate,
            self.independence,
            self.creativity,
            self.reflection_depth,
        ]
    }

    /// Calculate Euclidean distance to another personality
    pub fn distance_to(&self, other: &PersonalityTraits) -> f64 {
        let v1 = self.as_vector();
        let v2 = other.as_vector();

        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Data for personality evolution
#[derive(Debug, Clone)]
pub struct PersonalityEvolution {
    pub avg_realized_vol: f64,
    pub expected_vol: f64,
    pub prediction_accuracy: f64,
    pub contrarian_win_rate: f64,
    pub novel_setups_profitable: bool,
    pub total_trades: u64,
}

/// Characteristics of a market regime
#[derive(Debug, Clone)]
pub struct RegimeCharacteristics {
    pub volatility: f64,          // [0, 1]
    pub trend_strength: f64,      // [0, 1]
    pub uncertainty: f64,         // [0, 1]
    pub regime_change_frequency: f64, // [0, 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_personality_presets() {
        let conservative = PersonalityTraits::conservative();
        let aggressive = PersonalityTraits::aggressive();

        assert!(conservative.risk_tolerance < aggressive.risk_tolerance);
        assert!(conservative.creativity < aggressive.creativity);
    }

    #[test]
    fn test_personality_evolution() {
        let mut personality = PersonalityTraits::balanced();
        let initial_risk = personality.risk_tolerance;

        // High volatility environment
        let experience = PersonalityEvolution {
            avg_realized_vol: 0.5,
            expected_vol: 0.3,
            prediction_accuracy: 0.5,
            contrarian_win_rate: 0.5,
            novel_setups_profitable: false,
            total_trades: 50,
        };

        personality.evolve(&experience);

        // Risk tolerance should decrease
        assert!(personality.risk_tolerance < initial_risk);
    }

    #[test]
    fn test_regime_compatibility() {
        let conservative = PersonalityTraits::conservative();
        let aggressive = PersonalityTraits::aggressive();

        // High volatility regime
        let volatile_regime = RegimeCharacteristics {
            volatility: 0.9,
            trend_strength: 0.3,
            uncertainty: 0.7,
            regime_change_frequency: 0.6,
        };

        let conservative_score = conservative.regime_compatibility(&volatile_regime);
        let aggressive_score = aggressive.regime_compatibility(&volatile_regime);

        // Aggressive personality should score higher in volatile regime
        assert!(aggressive_score > conservative_score);
    }

    #[test]
    fn test_personality_distance() {
        let p1 = PersonalityTraits::conservative();
        let p2 = PersonalityTraits::aggressive();
        let p3 = PersonalityTraits::conservative();

        let dist_12 = p1.distance_to(&p2);
        let dist_13 = p1.distance_to(&p3);

        // Same personalities should have 0 distance
        assert!(dist_13 < 0.01);

        // Different personalities should have non-zero distance
        assert!(dist_12 > 0.1);
    }
}
