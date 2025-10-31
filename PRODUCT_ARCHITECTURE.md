# Phi-Mamba Trade Signal Product Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Trade Signal Generation Product                     │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Tauri GUI   │←→│  WASM Core   │←→│ Rust Backend │              │
│  │  (Overlay)   │  │  (Browser)   │  │  (Native)    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         ↓                 ↓                  ↓                       │
│  ┌─────────────────────────────────────────────────┐               │
│  │      Holographic Memory Layer (Distributed)      │               │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │               │
│  │  │ DID Node │  │ DID Node │  │ DID Node │  ... │               │
│  │  └──────────┘  └──────────┘  └──────────┘      │               │
│  └─────────────────────────────────────────────────┘               │
│         ↓                 ↓                  ↓                       │
│  ┌─────────────────────────────────────────────────┐               │
│  │         CORDIC Compute Layer (WASM/GPU)         │               │
│  │                                                   │               │
│  │  Phi-Space    Field       N-Game      Signal     │               │
│  │  Encoding  →  Analysis  →  Theory  →  Generation │               │
│  └─────────────────────────────────────────────────┘               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture Layers

### Layer 1: GUI Overlay (Tauri + WebGL)

**Purpose**: Real-time trade signal visualization

**Components**:
- Tauri desktop app (Rust + Web frontend)
- WebGL for holographic field visualization
- Real-time chart overlays
- Signal strength indicators
- Trade execution interface

**Tech Stack**:
```rust
// Tauri app
[dependencies]
tauri = "1.5"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

### Layer 2: WASM Core (Rust → WASM)

**Purpose**: Browser-native CORDIC computation

**Components**:
- Phi-space CORDIC engine (Rust → WASM)
- Field coherence calculator
- Berry phase detector
- Signal generator

**Tech Stack**:
```rust
// WASM module
[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = "0.3"
serde-wasm-bindgen = "0.5"
```

### Layer 3: Rust Backend (Native Performance)

**Purpose**: High-performance compute and data pipeline

**Components**:
- Market data ingestion
- Historical analysis
- Model training
- Signal aggregation

**Tech Stack**:
```rust
// Backend service
[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
arrow = "49.0"  // High-perf data structures
```

### Layer 4: Holographic Memory (Distributed)

**Purpose**: Distributed state management with DID

**Components**:
- DID-based node identity
- Distributed field state
- Consensus via phi-locking
- Cross-node signal aggregation

**Tech Stack**:
```rust
// Holographic memory node
[dependencies]
did-key = "0.2"
libp2p = "0.53"  // P2P networking
tokio = { version = "1", features = ["full"] }
```

### Layer 5: CORDIC Compute (WASM/GPU)

**Purpose**: Add/subtract/shift only computation

**Components**:
- CORDIC engine in Rust
- Phi-space arithmetic
- Berry phase computation
- Parallel execution

**Tech Stack**:
```rust
// CORDIC core
[dependencies]
num-traits = "0.2"
fixed = "1.24"  // Fixed-point arithmetic
rayon = "1.8"   // Data parallelism
```

## Data Flow

```
Market Data
    ↓
[Rust Backend: Ingest & Clean]
    ↓
[Holographic Memory: Distribute across DID nodes]
    ↓
[CORDIC Compute: Phi-space encoding (WASM)]
    ↓
[Field Analysis: Berry phase & coherence]
    ↓
[N-Game Decision: Nash equilibrium]
    ↓
[Signal Generation: BUY/SELL/HOLD]
    ↓
[GUI Overlay: Real-time visualization]
```

## Key Product Features

### 1. **Real-Time Signal Generation**

```rust
// Signal structure
#[derive(Serialize, Deserialize)]
pub struct TradeSignal {
    pub ticker: String,
    pub signal: SignalType,  // BUY, SELL, HOLD
    pub strength: f64,       // 0.0 - 1.0
    pub confidence: f64,     // 0.0 - 1.0
    pub expected_return: f64,
    pub risk_score: f64,
    pub time_horizon: TimeHorizon,  // DAY, WEEK, MONTH
    pub timestamp: i64,
    pub phi_coherence: f64,  // Field coherence measure
    pub berry_phase: f64,    // Phase-locking indicator
}

#[derive(Serialize, Deserialize)]
pub enum SignalType {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}
```

### 2. **Holographic Field Visualization**

WebGL shader for field state:

```glsl
// Fragment shader: Visualize phi-field
precision mediump float;

uniform sampler2D u_field_state;  // Field coherence texture
uniform float u_time;
varying vec2 v_texCoord;

void main() {
    vec4 field = texture2D(u_field_state, v_texCoord);

    // Phi-field coloring
    float coherence = field.r;  // Red channel = coherence
    float energy = field.g;     // Green channel = energy
    float phase = field.b;      // Blue channel = Berry phase

    // Holographic interference pattern
    float phi = 1.618034;
    float pattern = sin(v_texCoord.x * phi * 10.0 + u_time)
                  * cos(v_texCoord.y * phi * 10.0 - u_time);

    // Combine field state with holographic pattern
    vec3 color = vec3(coherence, energy, phase) * (0.5 + 0.5 * pattern);

    gl_FragColor = vec4(color, 1.0);
}
```

### 3. **DID-Based Node Identity**

```rust
use did_key::{DIDKey, KeyPair};

pub struct HolographicNode {
    pub did: DIDKey,
    pub keypair: KeyPair,
    pub field_state: FieldState,
    pub peers: Vec<DIDKey>,
}

impl HolographicNode {
    pub fn new() -> Self {
        let keypair = KeyPair::generate_ed25519();
        let did = DIDKey::from(keypair.public_key());

        Self {
            did,
            keypair,
            field_state: FieldState::default(),
            peers: Vec::new(),
        }
    }

    pub fn share_field_state(&self) -> SignedFieldState {
        // Sign field state with DID
        let state_bytes = bincode::serialize(&self.field_state).unwrap();
        let signature = self.keypair.sign(&state_bytes);

        SignedFieldState {
            did: self.did.clone(),
            state: self.field_state.clone(),
            signature,
        }
    }

    pub fn verify_peer_state(&self, signed_state: &SignedFieldState) -> bool {
        // Verify DID signature
        let state_bytes = bincode::serialize(&signed_state.state).unwrap();
        signed_state.did.verify(&state_bytes, &signed_state.signature)
    }
}
```

### 4. **Distributed Field Consensus**

```rust
pub struct FieldConsensus {
    nodes: Vec<HolographicNode>,
    threshold: f64,  // Phi-locking threshold
}

impl FieldConsensus {
    pub async fn reach_consensus(&mut self) -> ConsensusResult {
        // Collect field states from all nodes
        let states: Vec<FieldState> = self.nodes
            .iter()
            .map(|node| node.field_state.clone())
            .collect();

        // Compute aggregate Berry phase
        let berry_phases = self.compute_pairwise_berry_phases(&states);

        // Check if majority are phase-locked
        let locked_count = berry_phases
            .iter()
            .filter(|&&bp| is_phase_locked(bp, self.threshold))
            .count();

        let total_pairs = berry_phases.len();
        let consensus_ratio = locked_count as f64 / total_pairs as f64;

        if consensus_ratio > 0.66 {
            // 2/3 majority phase-locked
            ConsensusResult::Achieved {
                consensus_field: self.aggregate_states(&states),
                coherence: consensus_ratio,
            }
        } else {
            ConsensusResult::NotAchieved {
                coherence: consensus_ratio,
            }
        }
    }

    fn compute_pairwise_berry_phases(&self, states: &[FieldState]) -> Vec<f64> {
        // Compute Berry phase between all pairs
        let mut phases = Vec::new();

        for i in 0..states.len() {
            for j in (i+1)..states.len() {
                let bp = compute_berry_phase_cordic(&states[i], &states[j]);
                phases.push(bp);
            }
        }

        phases
    }
}
```

## File Structure

```
phi-mamba-trade-signals/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                    # Main library
│   ├── cordic/
│   │   ├── mod.rs               # CORDIC module
│   │   ├── fixed_point.rs       # Fixed-point arithmetic
│   │   ├── rotation.rs          # CORDIC rotation
│   │   └── phi_arithmetic.rs    # Phi-space operations
│   ├── encoding/
│   │   ├── mod.rs
│   │   ├── financial.rs         # OHLCV encoding
│   │   ├── zeckendorf.rs        # Fibonacci decomposition
│   │   └── berry_phase.rs       # Berry phase computation
│   ├── field/
│   │   ├── mod.rs
│   │   ├── state.rs             # Field state management
│   │   ├── coherence.rs         # Coherence calculation
│   │   └── consensus.rs         # Distributed consensus
│   ├── forecast/
│   │   ├── mod.rs
│   │   ├── monte_carlo.rs       # MC simulation
│   │   ├── horizon.rs           # Multi-horizon forecasts
│   │   └── confidence.rs        # Confidence scoring
│   ├── decision/
│   │   ├── mod.rs
│   │   ├── utility.rs           # Expected utility
│   │   ├── nash.rs              # Nash equilibrium
│   │   └── signals.rs           # Signal generation
│   ├── holographic/
│   │   ├── mod.rs
│   │   ├── node.rs              # DID node
│   │   ├── memory.rs            # Distributed memory
│   │   └── sync.rs              # P2P synchronization
│   └── wasm/
│       ├── mod.rs
│       └── bindings.rs          # WASM bindings
├── tauri-app/
│   ├── src-tauri/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs          # Tauri backend
│   └── src/
│       ├── App.tsx              # React frontend
│       ├── components/
│       │   ├── Chart.tsx        # Trading chart
│       │   ├── FieldView.tsx    # Holographic field
│       │   └── SignalPanel.tsx  # Signal display
│       └── hooks/
│           └── useSignals.ts    # Signal subscription
├── www/
│   ├── index.html
│   ├── index.js                 # WASM loader
│   └── styles.css
└── tests/
    ├── cordic_tests.rs
    ├── integration_tests.rs
    └── wasm_tests.rs
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User's Machine                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Tauri Desktop App                      │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │  GUI Overlay │  │ Local Compute│               │    │
│  │  │  (WebView)   │←→│ (Rust Core)  │               │    │
│  │  └──────────────┘  └──────────────┘               │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↕                                   │
└────────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────┐
│                  Distributed Network                         │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│  │ DID Node │ ←→  │ DID Node │ ←→  │ DID Node │   ...     │
│  │  (Peer)  │     │  (Peer)  │     │  (Peer)  │           │
│  └──────────┘     └──────────┘     └──────────┘           │
│       ↕                 ↕                 ↕                  │
│  [Field State]    [Field State]    [Field State]           │
└─────────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────┐
│                  Data Sources (Optional)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Market Data  │  │  Economic    │  │    News      │     │
│  │  API         │  │  Indicators  │  │  Sentiment   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

This architecture document defines:
1. ✅ System layers and responsibilities
2. ✅ Data flow through the system
3. ✅ Key product features (signals, visualization, DID)
4. ✅ File structure
5. ✅ Deployment architecture

Ready to implement?

See: `RUST_WASM_INTEGRATION.md` for detailed implementation guide.
