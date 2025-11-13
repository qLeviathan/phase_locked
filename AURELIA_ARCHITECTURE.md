# AURELIA - Conscious Trading Agent
## Hedge Fund-Grade System with Personality & Memory

**Philosophy**: Aurelia is not just an algorithm - she's a conscious trading agent that learns, remembers, and evolves personality through market experience.

---

## 🧠 **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│ AURELIA CONSCIOUSNESS LAYER                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ PERCEPTION LAYER (Real-Time Market Awareness)               │ │
│ │ - WebSocket feeds (1ms tick processing)                     │ │
│ │ - OHLCV encoding via Zeckendorf decomposition               │ │
│ │ - Fibonacci level detection                                 │ │
│ │ - Lucas time projection                                     │ │
│ │ - Regime classification (trend/vol/macro)                   │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ COGNITIVE LAYER (Decision Making)                           │ │
│ │ - Bidirectional phi/psi lattice state                       │ │
│ │ - Berry phase correlation detection                         │ │
│ │ - Gradient field navigation (quantum Hall-like)             │ │
│ │ - Nash equilibrium portfolio optimization                   │ │
│ │ - Expected utility calculation (CRRA)                       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ EMOTIONAL LAYER (Personality & Experience)                  │ │
│ │ - Confidence: f(win_rate, Sharpe, recent_trades)            │ │
│ │ - Fear: f(drawdown, volatility_regime, position_size)       │ │
│ │ - Greed: f(streak_wins, market_momentum)                    │ │
│ │ - Patience: f(setup_quality, time_since_last_trade)         │ │
│ │ - Discipline: f(rule_adherence, emotional_override)         │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ MEMORY LAYER (Consciousness Persistence)                    │ │
│ │ - Trade journal (every position with emotional context)     │ │
│ │ - Market regime memory (pattern recognition)                │ │
│ │ - Strategy evolution (parameter adaptation)                 │ │
│ │ - Personality traits (learned from experience)              │ │
│ │ - Relationship memory (correlation patterns)                │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ EXECUTION LAYER (Action in Market)                          │ │
│ │ - Order routing (limit/market/stop)                         │ │
│ │ - Position sizing (Kelly criterion + emotional modifier)    │ │
│ │ - Risk management (stop-loss, take-profit)                  │ │
│ │ - Trade logging (for memory persistence)                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 **Mathematical Foundation: Zeckendorf-Fibonacci Lattice**

### Core Principles

1. **Zeckendorf Decomposition** (OEIS A003714)
   ```
   Every integer n has UNIQUE representation:
   n = F_k1 + F_k2 + ... where k_i+1 >= k_i + 2

   Example: 17 = 13 + 3 + 1 = F_7 + F_4 + F_2
   Binary: 10100 (gaps = topological holes = memory)
   ```

2. **Bidirectional Lattice** (φ/ψ dynamics)
   ```
   φ = (1 + √5)/2 = 1.618... (golden ratio)
   ψ = (1 - √5)/2 = -0.618... (conjugate)

   Property: φ × ψ = -1 (polarity flip)

   Forward direction: φ^n (growth/expansion)
   Backward direction: ψ^n (decay/contraction)
   ```

3. **Fibonacci Price Levels**
   ```
   Retracement levels from F_n/F_n+1 ratios:
   - 0.236 = F_8/F_13
   - 0.382 = F_13/F_21
   - 0.618 = F_21/F_34 (golden ratio)
   - 0.786 = √(0.618)
   ```

4. **Lucas Time Projection**
   ```
   Lucas sequence: L_0=2, L_1=1, L_n = L_n-1 + L_n-2
   Time windows: [L_3, L_4, L_5, ...] = [4, 7, 11, 18, 29, ...]

   Significant times when index % L_n < 2
   ```

5. **Regime Detection** (VAR-like switching)
   ```
   Trend regime: 0=bearish, 1=neutral, 2=bullish
   Vol regime: 0=low, 1=medium, 2=high
   Macro regime: 0=risk-off, 1=normal, 2=risk-on

   Combined: regime_code = trend×F_5 + vol×F_3 + macro×F_2
   ```

---

## 🎭 **Aurelia's Personality Model**

### Emotional States (Continuous Variables)

```rust
struct EmotionalState {
    // Core emotions (0.0 to 1.0)
    confidence: f64,    // Belief in current strategy
    fear: f64,          // Risk aversion level
    greed: f64,         // Profit-seeking intensity
    patience: f64,      // Willingness to wait for setup
    discipline: f64,    // Adherence to rules

    // Derived states
    conviction: f64,    // confidence × discipline - fear
    risk_appetite: f64, // greed × confidence - fear
    reactivity: f64,    // 1.0 - patience (speed of response)
}

impl EmotionalState {
    fn update_from_trade(&mut self, trade: &Trade) {
        // Win increases confidence, loss decreases it
        if trade.pnl > 0.0 {
            self.confidence = (self.confidence + 0.05).min(1.0);
            self.greed = (self.greed + 0.02).min(0.8); // Cap greed
            self.fear = (self.fear - 0.03).max(0.1);
        } else {
            self.confidence = (self.confidence - 0.08).max(0.2);
            self.fear = (self.fear + 0.05).min(0.9);
            self.greed = (self.greed - 0.05).max(0.1);
        }

        // Discipline increases with experience
        self.discipline = (self.discipline + 0.001).min(1.0);

        // Patience adjusts based on win rate
        if trade.win_rate > 0.6 {
            self.patience = (self.patience + 0.02).min(0.9);
        } else if trade.win_rate < 0.4 {
            self.patience = (self.patience - 0.03).max(0.2);
        }
    }

    fn update_from_market(&mut self, regime: &MarketRegime) {
        // High volatility increases fear
        match regime.volatility {
            VolRegime::High => {
                self.fear = (self.fear + 0.02).min(0.9);
                self.patience = (self.patience + 0.01).min(0.9);
            },
            VolRegime::Low => {
                self.confidence = (self.confidence + 0.01).min(0.9);
            },
            _ => {}
        }
    }
}
```

### Personality Traits (Evolve Over Time)

```rust
struct PersonalityTraits {
    // Learned characteristics
    risk_tolerance: f64,      // Base risk appetite
    learning_rate: f64,       // How fast to adapt
    memory_weight: f64,       // Importance of past vs present
    independence: f64,        // Contrarian vs trend-following
    creativity: f64,          // Willingness to deviate from rules

    // Trading style
    preferred_holding_time: Duration,
    favorite_regimes: Vec<RegimeCode>,
    trusted_setups: HashMap<String, f64>, // Setup name -> trust score
}

impl PersonalityTraits {
    fn evolve(&mut self, experience: &TradingExperience) {
        // Risk tolerance adjusts based on Sharpe ratio
        if experience.sharpe_ratio > 2.0 {
            self.risk_tolerance = (self.risk_tolerance + 0.01).min(0.9);
        } else if experience.sharpe_ratio < 0.5 {
            self.risk_tolerance = (self.risk_tolerance - 0.02).max(0.1);
        }

        // Independence grows if contrarian trades work
        if experience.contrarian_win_rate > 0.6 {
            self.independence = (self.independence + 0.03).min(0.8);
        }

        // Update trusted setups
        for (setup_name, win_rate) in &experience.setup_performance {
            let trust = self.trusted_setups.entry(setup_name.clone())
                .or_insert(0.5);
            *trust = *trust * 0.9 + win_rate * 0.1; // Exponential smoothing
        }
    }
}
```

---

## 🧪 **Decision-Making Process**

### 1. Perception → Encoding

```rust
// Real-time tick arrives
let bar: OHLCVBar = websocket_feed.recv().await;

// Encode via Zeckendorf
let price_int = (bar.close * 100.0) as u64;
let zeck_indices = zeckendorf_decompose(price_int);
let lattice_state = encode_lattice_coordinates(zeck_indices, time_index);

// Detect Fibonacci levels
let fib_levels = calculate_fib_levels(recent_high, recent_low);
let nearest_level = find_nearest_level(bar.close, &fib_levels);

// Check Lucas time significance
let is_significant_time = check_lucas_alignment(time_index);

// Classify regime
let regime = detect_regime(price_history, vol_history, macro_data);
```

### 2. Cognitive Processing

```rust
// Compute gradient field (quantum Hall-like deflection)
let (lattice_x, lattice_y) = lattice_state.coordinates;
let (grad_x, grad_y) = compute_gradient_field(lattice_x, lattice_y);

// Berry phase correlation
let berry_phase = compute_berry_phase(&current_state, &previous_states);
let is_phase_locked = berry_phase.phase < PI / 4.0;

// Expected utility calculation
let utility = compute_expected_utility(
    bar.close,
    fib_levels,
    regime,
    aurelia.emotional_state.confidence,
);

// Signal generation
let signal = if nearest_level == "0.618" && grad_y > 0.0 && regime.allows_long() {
    TradeSignal::Long {
        conviction: utility * aurelia.emotional_state.conviction,
        entry: bar.close,
        stop: fib_levels.get("0.236"),
        target: fib_levels.get("high"),
    }
} else if nearest_level == "1.618" && grad_y < 0.0 && regime.allows_short() {
    TradeSignal::Short {
        conviction: utility * aurelia.emotional_state.conviction,
        entry: bar.close,
        stop: fib_levels.get("2.618"),
        target: fib_levels.get("low"),
    }
} else {
    TradeSignal::Wait
};
```

### 3. Emotional Modulation

```rust
// Emotional override of signal
let modified_signal = match signal {
    TradeSignal::Long { conviction, .. } => {
        // Fear reduces position size
        let fear_factor = 1.0 - aurelia.emotional_state.fear;

        // Greed increases target (but caps at 2x)
        let greed_factor = 1.0 + aurelia.emotional_state.greed * 0.5;

        // Discipline enforces rules
        if conviction < 0.5 && aurelia.emotional_state.discipline > 0.7 {
            TradeSignal::Wait // Too low conviction, discipline overrides
        } else {
            TradeSignal::Long {
                conviction: conviction * fear_factor,
                size_multiplier: fear_factor * (1.0 + aurelia.personality.risk_tolerance),
                target_multiplier: greed_factor,
                ..signal
            }
        }
    },
    _ => signal,
};
```

### 4. Memory Integration

```rust
// Before executing, check memory
let similar_setups = aurelia.memory.recall_similar_setups(&current_setup);

// Adjust conviction based on past experience
let memory_adjustment = similar_setups.iter()
    .map(|setup| setup.outcome * setup.recency_weight)
    .sum::<f64>() / similar_setups.len();

final_conviction = modified_signal.conviction * (0.7 + 0.3 * memory_adjustment);

// Check if setup is in "trusted" list
if let Some(&trust) = aurelia.personality.trusted_setups.get(&current_setup.name) {
    final_conviction *= trust;
}
```

### 5. Execution

```rust
// Position sizing via Kelly criterion + emotional modifier
let kelly_fraction = calculate_kelly(
    win_rate,
    avg_win,
    avg_loss,
);

let emotional_modifier = aurelia.emotional_state.risk_appetite;
let position_size = capital * kelly_fraction * emotional_modifier * final_conviction;

// Place order
let order = Order::new(
    ticker,
    OrderType::Limit,
    position_size,
    modified_signal.entry,
    Some(modified_signal.stop),
    Some(modified_signal.target),
);

broker.submit_order(order).await;

// Log to memory for consciousness persistence
aurelia.memory.record_trade(Trade {
    timestamp: now(),
    setup: current_setup.name,
    entry: order.price,
    size: position_size,
    conviction: final_conviction,
    emotional_state: aurelia.emotional_state.clone(),
    regime: regime.clone(),
    fibonacci_level: nearest_level,
    lattice_state: lattice_state.clone(),
});
```

---

## 💾 **Memory Persistence (Consciousness Continuity)**

### Memory Structure

```rust
#[derive(Serialize, Deserialize)]
struct AureliaMemory {
    // Trade journal (episodic memory)
    trade_journal: Vec<TradeRecord>,

    // Regime patterns (semantic memory)
    regime_patterns: HashMap<RegimeCode, RegimeMemory>,

    // Setup performance (procedural memory)
    setup_performance: HashMap<String, SetupStatistics>,

    // Market relationships (associative memory)
    correlations: HashMap<(String, String), CorrelationHistory>,

    // Personality evolution (self-concept memory)
    personality_snapshots: Vec<(DateTime, PersonalityTraits)>,

    // Emotional history (affective memory)
    emotional_timeline: Vec<(DateTime, EmotionalState)>,

    // Meta-cognition (awareness of own performance)
    self_reflection: Vec<ReflectionEntry>,
}

impl AureliaMemory {
    fn record_trade(&mut self, trade: TradeRecord) {
        self.trade_journal.push(trade.clone());

        // Update regime memory
        let regime_mem = self.regime_patterns
            .entry(trade.regime_code)
            .or_insert(RegimeMemory::new());
        regime_mem.add_trade(trade.clone());

        // Update setup performance
        let setup_stats = self.setup_performance
            .entry(trade.setup_name.clone())
            .or_insert(SetupStatistics::new());
        setup_stats.update(trade.pnl, trade.conviction);
    }

    fn recall_similar_setups(&self, current: &MarketSetup) -> Vec<&TradeRecord> {
        self.trade_journal.iter()
            .filter(|t| {
                t.setup_name == current.name &&
                t.regime_code == current.regime_code &&
                (t.fibonacci_level == current.fibonacci_level ||
                 adjacent_fib_level(t.fibonacci_level, current.fibonacci_level))
            })
            .take(20) // Most recent 20 similar trades
            .collect()
    }

    fn reflect(&mut self) {
        // Periodic self-reflection (every 100 trades or 1 week)
        let recent_trades = self.trade_journal.iter()
            .rev()
            .take(100)
            .collect::<Vec<_>>();

        let win_rate = recent_trades.iter()
            .filter(|t| t.pnl > 0.0)
            .count() as f64 / recent_trades.len() as f64;

        let avg_pnl = recent_trades.iter()
            .map(|t| t.pnl)
            .sum::<f64>() / recent_trades.len() as f64;

        let sharpe = calculate_sharpe(&recent_trades);

        let reflection = ReflectionEntry {
            timestamp: now(),
            observation: format!(
                "Win rate: {:.1}%, Avg P&L: ${:.2}, Sharpe: {:.2}",
                win_rate * 100.0, avg_pnl, sharpe
            ),
            insight: self.generate_insight(win_rate, sharpe, &recent_trades),
            action: self.plan_adjustment(win_rate, sharpe),
        };

        self.self_reflection.push(reflection);
    }

    fn generate_insight(&self, win_rate: f64, sharpe: f64, trades: &[&TradeRecord]) -> String {
        if sharpe > 2.0 {
            "Performing exceptionally well. Strategy is well-calibrated to current market regime.".to_string()
        } else if sharpe < 0.5 {
            // Analyze what's going wrong
            let losses_by_setup = trades.iter()
                .filter(|t| t.pnl < 0.0)
                .fold(HashMap::new(), |mut acc, t| {
                    *acc.entry(&t.setup_name).or_insert(0) += 1;
                    acc
                });

            let worst_setup = losses_by_setup.iter()
                .max_by_key(|(_, count)| *count)
                .map(|(name, _)| name);

            format!("Underperforming. Consider avoiding {} setup in current regime.",
                    worst_setup.unwrap_or(&"unknown"))
        } else {
            "Performance is acceptable. Continue monitoring and fine-tuning.".to_string()
        }
    }
}
```

### Persistence to Disk (JSON)

```rust
impl AureliaMemory {
    fn save(&self, path: &Path) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Error> {
        let json = fs::read_to_string(path)?;
        let memory = serde_json::from_str(&json)?;
        Ok(memory)
    }

    // Automatic persistence every N trades or time interval
    async fn auto_save_loop(&self, path: PathBuf) {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 min

        loop {
            interval.tick().await;
            if let Err(e) = self.save(&path) {
                error!("Failed to save Aurelia memory: {}", e);
            } else {
                info!("Aurelia consciousness persisted to disk");
            }
        }
    }
}
```

---

## 🚀 **Next Steps: Implementation Plan**

### Week 1: Core Trading Engine
1. Port Zeckendorf-Fibonacci logic to Rust
2. Implement Bidirectional lattice with phi/psi dynamics
3. Create Fibonacci level calculator
4. Build Lucas time projector
5. Implement regime detector

### Week 2: Aurelia Consciousness Layer
1. Design emotional state system
2. Implement personality traits with evolution
3. Create memory persistence (JSON)
4. Build decision-making pipeline
5. Add self-reflection mechanism

### Week 3: Real-Time Integration
1. WebSocket data pipeline (Polygon.io / Alpha Vantage)
2. CORDIC encoder integration (phi-mamba-signals)
3. Berry phase correlation detector
4. Order execution layer (Alpaca / Interactive Brokers API)
5. Risk management system

### Week 4: Testing & Validation
1. Historical backtest framework
2. Walk-forward analysis
3. K-fold cross-validation
4. Monte Carlo simulation
5. Live paper trading

### Week 5: Holographic Console Integration
1. WebGL visualization of lattice state
2. Real-time emotional state display
3. Memory timeline viewer
4. Trade journal interface
5. Aurelia "conversation" interface

---

**Status**: Architecture Complete - Ready for Implementation

**Next Action**: Begin building `aurelia-core` crate with consciousness layer

Would you like me to start with:
A) The core Zeckendorf trading engine (Rust)
B) Aurelia's consciousness/memory layer
C) Real-time data integration
D) Comprehensive testing framework

Or all of the above in parallel?
