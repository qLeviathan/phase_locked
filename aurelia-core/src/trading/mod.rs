/**
 * Trading Decision Engine
 *
 * Integrates consciousness, personality, memory, and mathematics into
 * agentic trading decisions. Every decision is emotionally modulated and
 * recorded in memory for personality evolution.
 */

use serde::{Deserialize, Serialize};
use crate::consciousness::{EmotionalState, PersonalityTraits, DerivedEmotions};
use crate::memory::{AureliaMemory, TradeRecord, TradeDirection, EmotionalStateSnapshot};
use crate::zeckendorf::{LatticePoint, FibonacciLevels, LucasTimeProjector};
use chrono::Utc;

/// Trading decision with conscious reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    /// Should we take this trade?
    pub should_trade: bool,

    /// Direction (if trading)
    pub direction: Option<TradeDirection>,

    /// Suggested position size (% of equity)
    pub size_pct: f64,

    /// Entry price
    pub entry_price: f64,

    /// Stop loss price
    pub stop_loss: f64,

    /// Take profit price
    pub take_profit: f64,

    /// Setup type identifier
    pub setup_type: String,

    /// Regime code
    pub regime_code: i32,

    /// Setup quality [0, 1]
    pub setup_quality: f64,

    /// Conviction level [0, 1]
    pub conviction: f64,

    /// Reasoning (for transparency)
    pub reasoning: Vec<String>,
}

/// Regime detector using Zeckendorf decomposition
#[derive(Debug, Clone)]
pub struct RegimeDetector;

impl RegimeDetector {
    /// Detect market regime from trend, volatility, and macro components
    ///
    /// Returns unique regime code using Fibonacci numbers
    /// regime_code = trend * F(5) + vol * F(3) + macro * F(2)
    pub fn detect_regime(
        trend: TrendState,
        volatility: VolatilityState,
        macro_state: MacroState,
    ) -> i32 {
        use crate::zeckendorf::FIBONACCI;

        let trend_val = match trend {
            TrendState::StrongUp => 2,
            TrendState::Up => 1,
            TrendState::Neutral => 0,
            TrendState::Down => -1,
            TrendState::StrongDown => -2,
        };

        let vol_val = match volatility {
            VolatilityState::VeryLow => 0,
            VolatilityState::Low => 1,
            VolatilityState::Normal => 2,
            VolatilityState::High => 3,
            VolatilityState::VeryHigh => 4,
        };

        let macro_val = match macro_state {
            MacroState::RiskOn => 2,
            MacroState::Neutral => 1,
            MacroState::RiskOff => 0,
        };

        // Encode as unique integer
        ((trend_val + 2) as i32 * FIBONACCI[5] as i32
            + vol_val as i32 * FIBONACCI[3] as i32
            + macro_val as i32 * FIBONACCI[2] as i32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendState {
    StrongDown = -2,
    Down = -1,
    Neutral = 0,
    Up = 1,
    StrongUp = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityState {
    VeryLow = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    VeryHigh = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroState {
    RiskOff = 0,
    Neutral = 1,
    RiskOn = 2,
}

/// Market snapshot for decision making
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub current_price: f64,
    pub bar_index: usize,
    pub swing_high: f64,
    pub swing_low: f64,
    pub trend: TrendState,
    pub volatility: VolatilityState,
    pub macro_state: MacroState,
}

/// Trading agent that makes conscious decisions
pub struct TradingAgent {
    pub emotions: EmotionalState,
    pub personality: PersonalityTraits,
    pub memory: AureliaMemory,
}

impl TradingAgent {
    pub fn new(personality: PersonalityTraits) -> Self {
        Self {
            emotions: EmotionalState::default(),
            personality,
            memory: AureliaMemory::new(),
        }
    }

    /// Make a trading decision based on market snapshot
    ///
    /// This is where consciousness meets mathematics
    pub fn decide(&mut self, snapshot: &MarketSnapshot) -> TradingDecision {
        let mut reasoning = Vec::new();

        // 1. Calculate regime
        let regime_code = RegimeDetector::detect_regime(
            snapshot.trend,
            snapshot.volatility,
            snapshot.macro_state,
        );

        // 2. Calculate Fibonacci levels
        let fib_levels = FibonacciLevels::retracement(snapshot.swing_high, snapshot.swing_low);

        // 3. Check if price is at a Fibonacci level
        let at_fib_level = fib_levels.is_at_level(snapshot.current_price, 1.0);

        // 4. Check Lucas time window
        let at_lucas_window = LucasTimeProjector::is_at_lucas_window(snapshot.bar_index, 2);

        // 5. Encode price into lattice
        let price_scaled = (snapshot.current_price * 100.0) as i64;
        let _current_point = LatticePoint::from_value(price_scaled, snapshot.bar_index);

        // 6. Calculate setup quality
        let mut setup_quality: f64 = 0.0;

        if at_fib_level.is_some() {
            setup_quality += 0.4;
            reasoning.push(format!(
                "Price at Fibonacci {:.3} level",
                at_fib_level.unwrap()
            ));
        }

        if at_lucas_window.is_some() {
            setup_quality += 0.3;
            reasoning.push(format!(
                "At Lucas time window L({})",
                at_lucas_window.unwrap()
            ));
        }

        // 7. Check historical performance of this setup
        if let Some(win_rate) = self.memory.setup_win_rate("fibonacci_bounce") {
            if win_rate > 0.6 {
                setup_quality += 0.2;
                reasoning.push(format!("Setup has {:.1}% win rate", win_rate * 100.0));
            } else {
                setup_quality -= 0.1;
                reasoning.push(format!("Setup only {:.1}% win rate", win_rate * 100.0));
            }
        }

        // 8. Check regime performance
        if let Some(regime_stats) = self.memory.regime_stats(regime_code) {
            if regime_stats.win_rate() > 0.55 {
                setup_quality += 0.1;
                reasoning.push(format!(
                    "Favorable regime (WR: {:.1}%)",
                    regime_stats.win_rate() * 100.0
                ));
            }
        }

        // Normalize setup quality
        setup_quality = setup_quality.max(0.0_f64).min(1.0_f64);

        // 9. Calculate derived emotions
        let derived = DerivedEmotions::compute(&self.emotions);

        // 10. Decide if we should trade
        let base_threshold = 0.5;
        let emotional_adjustment = derived.conviction * 0.2 - derived.reactivity * 0.1;
        let quality_threshold = base_threshold - emotional_adjustment;

        let should_trade = setup_quality >= quality_threshold;

        if !should_trade {
            reasoning.push(format!(
                "Setup quality {:.2} below threshold {:.2}",
                setup_quality, quality_threshold
            ));
        }

        // 11. Determine direction based on trend and Fibonacci level
        let direction = if should_trade {
            match snapshot.trend {
                TrendState::StrongUp | TrendState::Up => {
                    if at_fib_level.is_some() && at_fib_level.unwrap() > 0.5 {
                        // Price pulled back in uptrend - buy
                        reasoning.push("Buying pullback in uptrend".to_string());
                        Some(TradeDirection::Long)
                    } else {
                        None
                    }
                }
                TrendState::StrongDown | TrendState::Down => {
                    if at_fib_level.is_some() && at_fib_level.unwrap() < 0.5 {
                        // Price rallied in downtrend - sell
                        reasoning.push("Selling rally in downtrend".to_string());
                        Some(TradeDirection::Short)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        };

        // 12. Calculate position size (Kelly criterion with emotional modifier)
        let base_edge = setup_quality;
        let kelly_fraction = self.calculate_kelly_size(base_edge);

        // Emotional modulation
        let fear_reducer = 1.0 - (self.emotions.fear * 0.5);
        let greed_amplifier = 1.0 + (self.emotions.greed * 0.3);
        let personality_modifier = self.personality.risk_tolerance;

        let size_pct = kelly_fraction * fear_reducer * greed_amplifier * personality_modifier;
        let size_pct = size_pct.max(0.0).min(0.2); // Cap at 20% of equity

        reasoning.push(format!(
            "Position size: {:.1}% (Kelly: {:.1}%, fear: {:.2}, greed: {:.2})",
            size_pct * 100.0,
            kelly_fraction * 100.0,
            self.emotions.fear,
            self.emotions.greed
        ));

        // 13. Set stop loss and take profit
        let range = snapshot.swing_high - snapshot.swing_low;
        let atr_proxy = range * 0.5;

        let (stop_loss, take_profit) = match direction {
            Some(TradeDirection::Long) => {
                let stop = snapshot.current_price - atr_proxy * (1.0 + self.emotions.fear);
                let target = snapshot.current_price + atr_proxy * (2.0 + self.emotions.greed);
                (stop, target)
            }
            Some(TradeDirection::Short) => {
                let stop = snapshot.current_price + atr_proxy * (1.0 + self.emotions.fear);
                let target = snapshot.current_price - atr_proxy * (2.0 + self.emotions.greed);
                (stop, target)
            }
            None => (snapshot.current_price, snapshot.current_price),
        };

        TradingDecision {
            should_trade: direction.is_some(),
            direction,
            size_pct,
            entry_price: snapshot.current_price,
            stop_loss,
            take_profit,
            setup_type: "fibonacci_bounce".to_string(),
            regime_code,
            setup_quality,
            conviction: derived.conviction,
            reasoning,
        }
    }

    /// Execute a trade decision and record in memory
    pub fn execute_trade(
        &mut self,
        decision: &TradingDecision,
        snapshot: &MarketSnapshot,
        exit_price: f64,
        equity: f64,
    ) -> TradeRecord {
        let size = equity * decision.size_pct;
        let profit_loss = match decision.direction {
            Some(TradeDirection::Long) => {
                (exit_price - decision.entry_price) * size / decision.entry_price
            }
            Some(TradeDirection::Short) => {
                (decision.entry_price - exit_price) * size / decision.entry_price
            }
            None => 0.0,
        };

        let profit_pct = profit_loss / (size * decision.entry_price) * 100.0;

        // Check if trade was disciplined
        let was_disciplined = if profit_loss > 0.0 {
            // Win: did we take profit at target?
            (exit_price - decision.take_profit).abs()
                < (decision.take_profit - decision.entry_price) * 0.1
        } else {
            // Loss: did we stop out at stop loss?
            (exit_price - decision.stop_loss).abs()
                < (decision.entry_price - decision.stop_loss) * 0.1
        };

        let trade = TradeRecord {
            timestamp: Utc::now(),
            symbol: snapshot.symbol.clone(),
            direction: decision.direction.unwrap_or(TradeDirection::Long),
            entry_price: decision.entry_price,
            exit_price,
            size,
            profit_loss,
            profit_pct,
            setup_type: decision.setup_type.clone(),
            regime_code: decision.regime_code,
            was_disciplined,
            setup_quality: decision.setup_quality,
            emotional_state_at_entry: EmotionalStateSnapshot {
                confidence: self.emotions.confidence,
                fear: self.emotions.fear,
                greed: self.emotions.greed,
                patience: self.emotions.patience,
                discipline: self.emotions.discipline,
            },
        };

        // Update emotions based on trade outcome
        self.emotions.update_from_trade(profit_pct / 100.0, was_disciplined, decision.setup_quality);

        // Record in memory
        self.memory.record_trade(trade.clone());

        trade
    }

    /// Calculate Kelly criterion position size
    fn calculate_kelly_size(&self, edge: f64) -> f64 {
        // Simplified Kelly: f* = edge / odds
        // Assuming 2:1 reward:risk
        let kelly = edge / 2.0;

        // Use fractional Kelly (0.25x) for safety
        kelly * 0.25
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_detection() {
        let regime = RegimeDetector::detect_regime(
            TrendState::Up,
            VolatilityState::Normal,
            MacroState::RiskOn,
        );

        // Should produce unique code
        assert!(regime > 0);
    }

    #[test]
    fn test_trading_decision() {
        let mut agent = TradingAgent::new(PersonalityTraits::balanced());

        let snapshot = MarketSnapshot {
            symbol: "SPY".to_string(),
            current_price: 450.0,
            bar_index: 100,
            swing_high: 460.0,
            swing_low: 440.0,
            trend: TrendState::Up,
            volatility: VolatilityState::Normal,
            macro_state: MacroState::RiskOn,
        };

        let decision = agent.decide(&snapshot);

        // Should make some decision
        assert!(decision.setup_quality >= 0.0);
        assert!(decision.conviction >= 0.0);
        assert!(!decision.reasoning.is_empty());
    }

    #[test]
    fn test_emotional_modulation() {
        let mut agent = TradingAgent::new(PersonalityTraits::balanced());

        // Set high fear
        agent.emotions.fear = 0.9;

        let snapshot = create_test_snapshot();
        let decision = agent.decide(&snapshot);

        // High fear should reduce position size
        assert!(decision.size_pct < 0.1);
    }

    #[test]
    fn test_memory_integration() {
        let mut agent = TradingAgent::new(PersonalityTraits::balanced());

        let snapshot = create_test_snapshot();
        let mut decision = agent.decide(&snapshot);

        // Force a Long trade for testing
        decision.direction = Some(TradeDirection::Long);
        decision.entry_price = 450.0;

        // Execute a winning trade (exit higher)
        let trade = agent.execute_trade(&decision, &snapshot, 455.0, 100_000.0);

        // Should be recorded in memory
        assert_eq!(agent.memory.trade_journal.len(), 1);
        // For a long trade: entry 450, exit 455, should be positive
        assert!(trade.profit_loss >= 0.0);
    }

    fn create_test_snapshot() -> MarketSnapshot {
        MarketSnapshot {
            symbol: "SPY".to_string(),
            current_price: 450.0,
            bar_index: 47, // Lucas number
            swing_high: 460.0,
            swing_low: 440.0,
            trend: TrendState::Up,
            volatility: VolatilityState::Normal,
            macro_state: MacroState::RiskOn,
        }
    }
}
