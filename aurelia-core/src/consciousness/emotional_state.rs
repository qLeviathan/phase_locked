/**
 * Emotional State System
 *
 * Implements dynamic emotional states that respond to trading outcomes.
 * Emotions are bounded [0.0, 1.0] and evolve based on experience.
 */

use serde::{Deserialize, Serialize};

/// Core emotional state (all values in [0.0, 1.0])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Confidence: Belief in current strategy
    /// Increases with wins, decreases with losses
    pub confidence: f64,

    /// Fear: Risk aversion level
    /// Spikes on large losses, decays slowly
    pub fear: f64,

    /// Greed: Profit-seeking intensity
    /// Increases with wins, tempered by discipline
    pub greed: f64,

    /// Patience: Willingness to wait for optimal setup
    /// Increases with experience, decreases with losses
    pub patience: f64,

    /// Discipline: Adherence to trading rules
    /// Core personality trait, slowly evolving
    pub discipline: f64,
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            confidence: 0.5,  // Neutral starting point
            fear: 0.3,        // Slight baseline caution
            greed: 0.4,       // Moderate profit seeking
            patience: 0.6,    // Above-average patience
            discipline: 0.7,  // Strong rule adherence
        }
    }
}

impl EmotionalState {
    /// Create a balanced emotional state
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create a cautious emotional state (conservative trading)
    pub fn cautious() -> Self {
        Self {
            confidence: 0.4,
            fear: 0.6,
            greed: 0.2,
            patience: 0.8,
            discipline: 0.9,
        }
    }

    /// Create an aggressive emotional state (risk-seeking)
    pub fn aggressive() -> Self {
        Self {
            confidence: 0.7,
            fear: 0.2,
            greed: 0.7,
            patience: 0.4,
            discipline: 0.6,
        }
    }

    /// Update emotions based on trade outcome
    pub fn update_from_trade(
        &mut self,
        profit_pct: f64,
        was_disciplined: bool,
        setup_quality: f64,  // [0.0, 1.0]
    ) {
        // Profit/loss impact
        if profit_pct > 0.0 {
            self.on_winning_trade(profit_pct, setup_quality);
        } else {
            self.on_losing_trade(profit_pct.abs(), setup_quality);
        }

        // Discipline feedback
        if was_disciplined {
            self.discipline = self.clamp(self.discipline + 0.01);
            self.patience = self.clamp(self.patience + 0.01);
        } else {
            self.discipline = self.clamp(self.discipline - 0.02);
            self.greed = self.clamp(self.greed + 0.03); // Undisciplined = greedy
        }

        // Natural emotional decay (return to baseline)
        self.decay_emotions(0.99);
    }

    /// Process winning trade
    fn on_winning_trade(&mut self, profit_pct: f64, setup_quality: f64) {
        let impact = (profit_pct * 10.0).min(0.15); // Cap impact at 15%

        // Wins boost confidence
        self.confidence = self.clamp(self.confidence + impact * setup_quality);

        // Reduce fear
        self.fear = self.clamp(self.fear - impact * 0.5);

        // Increase greed (tempered by discipline)
        let greed_increase = impact * (1.0 - self.discipline * 0.5);
        self.greed = self.clamp(self.greed + greed_increase);

        // Good setups reward patience
        if setup_quality > 0.7 {
            self.patience = self.clamp(self.patience + 0.02);
        }
    }

    /// Process losing trade
    fn on_losing_trade(&mut self, loss_pct: f64, setup_quality: f64) {
        let impact = (loss_pct * 10.0).min(0.2); // Cap impact at 20%

        // Losses erode confidence
        self.confidence = self.clamp(self.confidence - impact);

        // Spike fear (more than confidence loss)
        self.fear = self.clamp(self.fear + impact * 1.5);

        // Reduce greed
        self.greed = self.clamp(self.greed - impact * 0.5);

        // Poor setups punish patience (impatience creeps in)
        if setup_quality < 0.3 {
            self.patience = self.clamp(self.patience - 0.03);
        }
    }

    /// Update emotions from drawdown (portfolio-level)
    pub fn update_from_drawdown(&mut self, drawdown_pct: f64) {
        if drawdown_pct > 5.0 {
            // Significant drawdown triggers fear
            let fear_spike = (drawdown_pct / 100.0).min(0.3);
            self.fear = self.clamp(self.fear + fear_spike);

            // Erode confidence
            self.confidence = self.clamp(self.confidence - fear_spike * 0.5);

            // Reduce greed
            self.greed = self.clamp(self.greed - fear_spike * 0.3);
        }
    }

    /// Update emotions from win streak
    pub fn update_from_streak(&mut self, consecutive_wins: u32) {
        if consecutive_wins >= 3 {
            // Winning streak boosts confidence
            let boost = (consecutive_wins as f64 * 0.02).min(0.1);
            self.confidence = self.clamp(self.confidence + boost);

            // But also increases greed (overconfidence risk)
            self.greed = self.clamp(self.greed + boost * 0.5);

            // Reduce fear
            self.fear = self.clamp(self.fear - boost * 0.3);
        }
    }

    /// Natural decay towards balanced state (emotional homeostasis)
    fn decay_emotions(&mut self, decay_rate: f64) {
        let target = Self::balanced();

        self.confidence = self.confidence * decay_rate + target.confidence * (1.0 - decay_rate);
        self.fear = self.fear * decay_rate + target.fear * (1.0 - decay_rate);
        self.greed = self.greed * decay_rate + target.greed * (1.0 - decay_rate);
        self.patience = self.patience * decay_rate + target.patience * (1.0 - decay_rate);
        // Discipline doesn't decay - it's more stable
    }

    /// Clamp value to [0.0, 1.0]
    fn clamp(&self, value: f64) -> f64 {
        value.max(0.0).min(1.0)
    }

    /// Get overall emotional stability (inverse of variance)
    pub fn stability(&self) -> f64 {
        let mean = (self.confidence + self.fear + self.greed + self.patience + self.discipline) / 5.0;
        let variance = [
            (self.confidence - mean).powi(2),
            (self.fear - mean).powi(2),
            (self.greed - mean).powi(2),
            (self.patience - mean).powi(2),
            (self.discipline - mean).powi(2),
        ].iter().sum::<f64>() / 5.0;

        1.0 - variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winning_trade_updates() {
        let mut emotions = EmotionalState::default();
        let initial_confidence = emotions.confidence;
        let initial_fear = emotions.fear;

        // 5% profit, good setup, disciplined
        emotions.update_from_trade(0.05, true, 0.8);

        assert!(emotions.confidence > initial_confidence);
        assert!(emotions.fear < initial_fear);
        assert!(emotions.discipline > 0.7); // Should increase
    }

    #[test]
    fn test_losing_trade_updates() {
        let mut emotions = EmotionalState::default();
        let initial_confidence = emotions.confidence;
        let initial_fear = emotions.fear;

        // -3% loss, poor setup, undisciplined
        emotions.update_from_trade(-0.03, false, 0.2);

        assert!(emotions.confidence < initial_confidence);
        assert!(emotions.fear > initial_fear);
        assert!(emotions.discipline < 0.7); // Should decrease
    }

    #[test]
    fn test_drawdown_impact() {
        let mut emotions = EmotionalState::default();
        let initial_fear = emotions.fear;

        // 10% drawdown
        emotions.update_from_drawdown(10.0);

        assert!(emotions.fear > initial_fear);
        assert!(emotions.confidence < 0.5);
    }

    #[test]
    fn test_clamping() {
        let mut emotions = EmotionalState::default();

        // Simulate extreme winning streak
        for _ in 0..20 {
            emotions.update_from_trade(0.10, true, 1.0);
        }

        // All values should be clamped [0, 1]
        assert!(emotions.confidence <= 1.0 && emotions.confidence >= 0.0);
        assert!(emotions.fear <= 1.0 && emotions.fear >= 0.0);
        assert!(emotions.greed <= 1.0 && emotions.greed >= 0.0);
        assert!(emotions.patience <= 1.0 && emotions.patience >= 0.0);
        assert!(emotions.discipline <= 1.0 && emotions.discipline >= 0.0);
    }
}
