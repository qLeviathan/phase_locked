/**
 * Memory Layer
 *
 * Implements persistent memory that develops Aurelia's personality and consciousness.
 * Multiple memory types: episodic, semantic, procedural, associative, affective.
 */

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Complete memory structure for Aurelia
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AureliaMemory {
    /// Episodic memory: specific trade events
    pub trade_journal: Vec<TradeRecord>,

    /// Semantic memory: regime patterns and characteristics
    pub regime_patterns: HashMap<i32, RegimeMemory>,

    /// Procedural memory: performance of different setups
    pub setup_performance: HashMap<String, SetupStatistics>,

    /// Associative memory: correlations between symbols
    pub correlations: HashMap<(String, String), CorrelationHistory>,

    /// Self-concept: personality evolution over time
    pub personality_snapshots: Vec<PersonalitySnapshot>,

    /// Affective memory: emotional timeline
    pub emotional_timeline: Vec<EmotionalSnapshot>,

    /// Meta-cognition: self-reflection entries
    pub self_reflections: Vec<ReflectionEntry>,
}

impl AureliaMemory {
    /// Create new empty memory
    pub fn new() -> Self {
        Self {
            trade_journal: Vec::new(),
            regime_patterns: HashMap::new(),
            setup_performance: HashMap::new(),
            correlations: HashMap::new(),
            personality_snapshots: Vec::new(),
            emotional_timeline: Vec::new(),
            self_reflections: Vec::new(),
        }
    }

    /// Record a completed trade
    pub fn record_trade(&mut self, trade: TradeRecord) {
        // Update setup performance
        let setup_key = trade.setup_type.clone();
        let stats = self.setup_performance
            .entry(setup_key)
            .or_insert(SetupStatistics::new());
        stats.update(&trade);

        // Update regime memory
        let regime_stats = self.regime_patterns
            .entry(trade.regime_code)
            .or_insert(RegimeMemory::new(trade.regime_code));
        regime_stats.record_trade(trade.profit_loss);

        // Add to journal
        self.trade_journal.push(trade);

        // Keep journal bounded (last 10,000 trades)
        if self.trade_journal.len() > 10_000 {
            self.trade_journal.remove(0);
        }
    }

    /// Record emotional snapshot
    pub fn record_emotion(&mut self, snapshot: EmotionalSnapshot) {
        self.emotional_timeline.push(snapshot);

        // Keep timeline bounded (last 1,000 snapshots)
        if self.emotional_timeline.len() > 1_000 {
            self.emotional_timeline.remove(0);
        }
    }

    /// Record personality snapshot
    pub fn record_personality(&mut self, snapshot: PersonalitySnapshot) {
        self.personality_snapshots.push(snapshot);
    }

    /// Add self-reflection entry
    pub fn reflect(&mut self, reflection: ReflectionEntry) {
        self.self_reflections.push(reflection);

        // Keep reflections bounded (last 500)
        if self.self_reflections.len() > 500 {
            self.self_reflections.remove(0);
        }
    }

    /// Recall similar trades (episodic memory)
    pub fn recall_similar_trades(
        &self,
        setup_type: &str,
        regime_code: i32,
        limit: usize,
    ) -> Vec<&TradeRecord> {
        self.trade_journal
            .iter()
            .rev() // Recent first
            .filter(|t| t.setup_type == setup_type && t.regime_code == regime_code)
            .take(limit)
            .collect()
    }

    /// Get setup win rate (procedural memory)
    pub fn setup_win_rate(&self, setup_type: &str) -> Option<f64> {
        self.setup_performance.get(setup_type).map(|s| s.win_rate)
    }

    /// Get regime statistics (semantic memory)
    pub fn regime_stats(&self, regime_code: i32) -> Option<&RegimeMemory> {
        self.regime_patterns.get(&regime_code)
    }

    /// Get recent emotional trend
    pub fn recent_emotional_trend(&self, lookback: usize) -> Option<EmotionalTrend> {
        if self.emotional_timeline.len() < 2 {
            return None;
        }

        let recent = self.emotional_timeline
            .iter()
            .rev()
            .take(lookback)
            .collect::<Vec<_>>();

        if recent.is_empty() {
            return None;
        }

        let first = recent.last().unwrap();
        let last = recent.first().unwrap();

        Some(EmotionalTrend {
            confidence_change: last.confidence - first.confidence,
            fear_change: last.fear - first.fear,
            patience_change: last.patience - first.patience,
        })
    }

    /// Calculate total profit/loss
    pub fn total_pnl(&self) -> f64 {
        self.trade_journal.iter().map(|t| t.profit_loss).sum()
    }

    /// Calculate win rate
    pub fn overall_win_rate(&self) -> f64 {
        if self.trade_journal.is_empty() {
            return 0.0;
        }

        let wins = self.trade_journal.iter().filter(|t| t.profit_loss > 0.0).count();
        wins as f64 / self.trade_journal.len() as f64
    }

    /// Get best performing setup
    pub fn best_setup(&self) -> Option<(&String, &SetupStatistics)> {
        self.setup_performance
            .iter()
            .max_by(|a, b| {
                a.1.profit_factor
                    .partial_cmp(&b.1.profit_factor)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get worst performing setup
    pub fn worst_setup(&self) -> Option<(&String, &SetupStatistics)> {
        self.setup_performance
            .iter()
            .min_by(|a, b| {
                a.1.profit_factor
                    .partial_cmp(&b.1.profit_factor)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Single trade record (episodic memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub direction: TradeDirection,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub profit_loss: f64,
    pub profit_pct: f64,
    pub setup_type: String,
    pub regime_code: i32,
    pub was_disciplined: bool,
    pub setup_quality: f64,
    pub emotional_state_at_entry: EmotionalStateSnapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

/// Snapshot of emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateSnapshot {
    pub confidence: f64,
    pub fear: f64,
    pub greed: f64,
    pub patience: f64,
    pub discipline: f64,
}

/// Regime statistics (semantic memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeMemory {
    pub regime_code: i32,
    pub total_trades: u64,
    pub total_pnl: f64,
    pub win_count: u64,
    pub loss_count: u64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub last_seen: DateTime<Utc>,
}

impl RegimeMemory {
    pub fn new(regime_code: i32) -> Self {
        Self {
            regime_code,
            total_trades: 0,
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
            avg_win: 0.0,
            avg_loss: 0.0,
            last_seen: Utc::now(),
        }
    }

    pub fn record_trade(&mut self, pnl: f64) {
        self.total_trades += 1;
        self.total_pnl += pnl;
        self.last_seen = Utc::now();

        if pnl > 0.0 {
            self.win_count += 1;
            self.avg_win = (self.avg_win * (self.win_count - 1) as f64 + pnl) / self.win_count as f64;
        } else {
            self.loss_count += 1;
            self.avg_loss = (self.avg_loss * (self.loss_count - 1) as f64 + pnl.abs()) / self.loss_count as f64;
        }
    }

    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.win_count as f64 / self.total_trades as f64
    }

    pub fn profit_factor(&self) -> f64 {
        if self.avg_loss == 0.0 {
            return f64::INFINITY;
        }
        (self.avg_win * self.win_count as f64) / (self.avg_loss * self.loss_count as f64)
    }
}

/// Setup performance statistics (procedural memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupStatistics {
    pub total_trades: u64,
    pub win_count: u64,
    pub loss_count: u64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_profit: f64,
    pub max_profit: f64,
    pub max_loss: f64,
}

impl SetupStatistics {
    pub fn new() -> Self {
        Self {
            total_trades: 0,
            win_count: 0,
            loss_count: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_profit: 0.0,
            max_profit: 0.0,
            max_loss: 0.0,
        }
    }

    pub fn update(&mut self, trade: &TradeRecord) {
        self.total_trades += 1;

        if trade.profit_loss > 0.0 {
            self.win_count += 1;
            self.max_profit = self.max_profit.max(trade.profit_loss);
        } else {
            self.loss_count += 1;
            self.max_loss = self.max_loss.min(trade.profit_loss);
        }

        self.win_rate = self.win_count as f64 / self.total_trades as f64;

        // Update avg profit (incremental)
        self.avg_profit = (self.avg_profit * (self.total_trades - 1) as f64 + trade.profit_loss)
                          / self.total_trades as f64;

        // Update profit factor
        let total_wins = self.win_count as f64 * self.avg_profit.abs();
        let total_losses = self.loss_count as f64 * self.avg_profit.abs();
        self.profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else {
            f64::INFINITY
        };
    }
}

/// Correlation history (associative memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationHistory {
    pub symbol_a: String,
    pub symbol_b: String,
    pub correlation: f64,
    pub last_updated: DateTime<Utc>,
}

/// Personality snapshot (self-concept memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalitySnapshot {
    pub timestamp: DateTime<Utc>,
    pub risk_tolerance: f64,
    pub learning_rate: f64,
    pub independence: f64,
    pub creativity: f64,
    pub reflection_depth: f64,
}

/// Emotional snapshot (affective memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSnapshot {
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
    pub fear: f64,
    pub patience: f64,
    pub discipline: f64,
    pub trigger: String, // What caused this emotion
}

/// Self-reflection entry (meta-cognition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionEntry {
    pub timestamp: DateTime<Utc>,
    pub content: String,
    pub context: String, // What triggered the reflection
    pub insight: Option<String>, // Key realization
}

/// Emotional trend analysis
#[derive(Debug, Clone)]
pub struct EmotionalTrend {
    pub confidence_change: f64,
    pub fear_change: f64,
    pub patience_change: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let memory = AureliaMemory::new();
        assert_eq!(memory.trade_journal.len(), 0);
        assert_eq!(memory.regime_patterns.len(), 0);
    }

    #[test]
    fn test_trade_recording() {
        let mut memory = AureliaMemory::new();

        let trade = TradeRecord {
            timestamp: Utc::now(),
            symbol: "SPY".to_string(),
            direction: TradeDirection::Long,
            entry_price: 450.0,
            exit_price: 455.0,
            size: 100.0,
            profit_loss: 500.0,
            profit_pct: 1.11,
            setup_type: "fibonacci_bounce".to_string(),
            regime_code: 42,
            was_disciplined: true,
            setup_quality: 0.8,
            emotional_state_at_entry: EmotionalStateSnapshot {
                confidence: 0.7,
                fear: 0.3,
                greed: 0.5,
                patience: 0.6,
                discipline: 0.8,
            },
        };

        memory.record_trade(trade);

        assert_eq!(memory.trade_journal.len(), 1);
        assert_eq!(memory.total_pnl(), 500.0);
        assert_eq!(memory.overall_win_rate(), 1.0);
    }

    #[test]
    fn test_setup_performance() {
        let mut memory = AureliaMemory::new();

        // Record two trades with same setup
        let trade1 = create_test_trade("fib_bounce", 500.0);
        let trade2 = create_test_trade("fib_bounce", -200.0);

        memory.record_trade(trade1);
        memory.record_trade(trade2);

        let stats = memory.setup_performance.get("fib_bounce").unwrap();
        assert_eq!(stats.total_trades, 2);
        assert_eq!(stats.win_count, 1);
        assert_eq!(stats.win_rate, 0.5);
    }

    fn create_test_trade(setup: &str, pnl: f64) -> TradeRecord {
        TradeRecord {
            timestamp: Utc::now(),
            symbol: "SPY".to_string(),
            direction: TradeDirection::Long,
            entry_price: 450.0,
            exit_price: if pnl > 0.0 { 455.0 } else { 445.0 },
            size: 100.0,
            profit_loss: pnl,
            profit_pct: pnl / 450.0,
            setup_type: setup.to_string(),
            regime_code: 42,
            was_disciplined: true,
            setup_quality: 0.7,
            emotional_state_at_entry: EmotionalStateSnapshot {
                confidence: 0.6,
                fear: 0.4,
                greed: 0.5,
                patience: 0.6,
                discipline: 0.7,
            },
        }
    }
}
