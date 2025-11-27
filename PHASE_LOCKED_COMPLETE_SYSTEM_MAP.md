# Phase-Locked Repository: Complete System Map

## Document Purpose
This monolithic document provides 100% comprehensive coverage of the phase_locked repository, capturing every component, mathematical foundation, implementation detail, and interconnection. Created for enabling long-running agentic development with full context preservation.

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Repository Structure](#3-repository-structure)
4. [Core Python Implementation](#4-core-python-implementation)
5. [Rust Implementation Stack](#5-rust-implementation-stack)
6. [Financial System Integration](#6-financial-system-integration)
7. [Compression & Inference Engine](#7-compression--inference-engine)
8. [Desktop Applications](#8-desktop-applications)
9. [Mobile Agent System](#9-mobile-agent-system)
10. [Unified Architecture](#10-unified-architecture)
11. [Dependency Graph](#11-dependency-graph)
12. [API Reference](#12-api-reference)
13. [Build & Deployment](#13-build--deployment)
14. [Development Roadmap](#14-development-roadmap)

---

## 1. Executive Overview

### Project Identity
- **Name**: Φ-Mamba / Phase-Locked / ZORDIC System
- **Author**: Marc Castillo
- **Version**: 0.1.0
- **License**: MIT

### Core Innovation
This repository implements a revolutionary computational paradigm where:
- **φ (Golden Ratio) replaces binary as the fundamental primitive**
- **All multiplication reduces to addition** (φ^n × φ^m = φ^(n+m))
- **Zeckendorf decomposition enables unique integer representation**
- **CORDIC algorithms enable add/subtract/shift-only computation**
- **Natural termination through energy decay** (φ^(-n) → 0)

### Key Components
```
┌────────────────────────────────────────────────────────────────────┐
│                    PHASE-LOCKED SYSTEM                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MATHEMATICAL LAYER                                                  │
│  ├─ φ (Golden Ratio) = 1.618034...                                 │
│  ├─ Zeckendorf Decomposition (unique Fibonacci representation)      │
│  ├─ Berry Phase (topological coherence)                            │
│  ├─ CORDIC (shift-add only computation)                            │
│  └─ Game Theory (Nash equilibrium, backward induction)             │
│                                                                      │
│  IMPLEMENTATION LAYER                                                │
│  ├─ Python Reference (phi_mamba/)                                  │
│  ├─ Rust Production (phi_core/, rust_phi_mamba/)                   │
│  ├─ WASM Bindings (phi-wasm/)                                      │
│  └─ Unified CORDIC (unified-zeckendorf-cordic/)                    │
│                                                                      │
│  APPLICATION LAYER                                                   │
│  ├─ Desktop GUI (phi-mamba-desktop, Tauri)                         │
│  ├─ Mobile Agent (Tamagotchi Trader)                               │
│  ├─ Financial System (AURELIA consciousness)                        │
│  └─ LLM Compression (llama_compression/)                           │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

### Performance Targets
| Metric | Target | Achieved |
|--------|--------|----------|
| Cascade latency | <100μs | 0.333μs ✅ |
| Compression ratio | >131x | 295x ✅ |
| End-to-end latency | <1ms | ~0.68ms ✅ |
| Memory efficiency | 99%+ savings | 99.4% ✅ |

---

## 2. Mathematical Foundations

### 2.1 Golden Ratio (φ) as Primitive

```
φ = (1 + √5) / 2 = 1.618033988749895...
ψ = (1 - √5) / 2 = -0.618033988749895... (conjugate)

Key Properties:
  φ² = φ + 1
  φ × ψ = -1
  1 = φ² - φ (unity emerges from φ)
```

### 2.2 Zeckendorf's Theorem (OEIS A003714)

**Statement**: Every positive integer has a unique representation as a sum of non-consecutive Fibonacci numbers.

```
Example: 100 = 89 + 8 + 3 = F₁₁ + F₆ + F₄

Bit representation: indices where Fibonacci is used
  100 → bits at [4, 6, 11] → binary: 10000100100

KEY INSIGHT: The gaps between 1s store information (topological holes)
```

**Implementation (Python)**:
```python
def zeckendorf_decomposition(n: int) -> List[int]:
    if n <= 0:
        return []

    # Build Fibonacci sequence up to n
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])

    # Greedy selection (largest first)
    result = []
    for fib in reversed(fibs):
        if fib <= n:
            result.append(fib)
            n -= fib

    return sorted(result)
```

### 2.3 φ-Space Arithmetic

**Core Insight**: In BASE-φ, multiplication becomes addition of exponents.

```
φ^a × φ^b = φ^(a+b)    ← ADDITION!
φ^a / φ^b = φ^(a-b)    ← SUBTRACTION!
```

**Operations Table**:
| Operation | Traditional | φ-Space |
|-----------|-------------|---------|
| Multiply | a × b | n_a + n_b |
| Divide | a / b | n_a - n_b |
| Power | a^k | n × k |

### 2.4 CASCADE Operator (κ)

Resolves adjacent 1s in Zeckendorf bit representation:

```
Rule: If bits k and k+1 are both 1, clear them and set bit k+2

Examples:
  11      → 100       (Fibonacci identity: F_k + F_{k+1} = F_{k+2})
  111     → 1001
  11011   → 100100
```

**Implementation (Rust)**:
```rust
pub fn cascade(&self, indices: &mut IndexSet) {
    while let Some((k1, k2)) = indices.find_violation() {
        indices.remove(k1);
        indices.remove(k2);
        if k2 + 1 < MAX_SHELLS as u8 {
            indices.insert(k2 + 1);
        }
    }
}
```

### 2.5 CORDIC Algorithm

**CORDIC** (COordinate Rotation DIgital Computer) computes trig functions using ONLY:
- Addition
- Subtraction
- Bit shifts (multiply/divide by powers of 2)

```python
def cordic_rotate(x, y, angle, n_iterations=32):
    for i in range(n_iterations):
        if angle >= 0:
            x_new = x - (y >> i)  # y × 2^(-i) via shift
            y_new = y + (x >> i)  # x × 2^(-i) via shift
            angle -= ATAN_TABLE[i]
        else:
            x_new = x + (y >> i)
            y_new = y - (x >> i)
            angle += ATAN_TABLE[i]
        x, y = x_new, y_new
    return x, y
```

### 2.6 Berry Phase

Geometric phase acquired during cyclic evolution in parameter space:

```
γ = ∮ ⟨ψ|∇|ψ⟩ · dR

For token states:
  γ_ij = θ_j - θ_i + (2π/5) × shell_overlap

Phase-locked: |γ mod 2π| < π/4
```

### 2.7 Game Theory Framework

**Dynamic Game Formulation**:
```
Game Γ = (N, S, A, u, T, β) where:
  N: Players (tokens)
  S: State space (θ, energy, phase, shells)
  A: Actions (token selection)
  u: Utility (phase coherence × energy)
  T: Termination (energy < threshold)
  β: Discount factor = 1/φ = 0.618...
```

**Backward Induction (Retrocausality)**:
```
V*(s) = max{u(s,a) + β·E[V*(s')|s,a]}

Future endpoint Ω constrains all past decisions
```

---

## 3. Repository Structure

```
phase_locked/
├── phi_mamba/                    # Core Python library
│   ├── __init__.py              # Package exports
│   ├── core.py                  # PhiLanguageModel, PhiTokenizer
│   ├── encoding.py              # TokenState, Zeckendorf, retrocausal
│   ├── generation.py            # Phase-locked generation
│   ├── utils.py                 # Constants, math utilities
│   ├── constants.py             # PHI, PSI, TWO_PI
│   ├── math_core.py             # fibonacci, zeckendorf_decomposition
│   ├── cordic.py                # CORDIC engine
│   ├── financial_data.py        # OHLCVBar, TickerData
│   ├── financial_encoding.py    # FinancialTokenState
│   ├── financial_forecast.py    # PhiMambaForecaster
│   ├── financial_system.py      # Unified financial system
│   ├── financial_cordic_adapter.py  # CORDIC financial integration
│   └── decision_framework.py    # Nash equilibrium, expected utility
│
├── phi_core/                    # Rust "Latent n Manifold" library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs               # FIBONACCI, LUCAS constants
│       ├── latent_n.rs          # LatentN structure
│       ├── phi_arithmetic.rs    # Integer φ-operations
│       ├── zeckendorf.rs        # Decomposition engine
│       ├── boundary.rs          # Puzzle-solving boundaries
│       ├── memory.rs            # Base-φ allocation
│       ├── maximal.rs           # Completion detection
│       └── token_stream.rs      # Generation with Lucas stopping
│
├── rust_phi_mamba/              # Rust production implementation
│   ├── Cargo.toml
│   ├── crates/
│   │   ├── phi-core/            # Core algorithms
│   │   │   └── src/
│   │   │       ├── lib.rs       # PhiInt, Fibonacci, TokenState
│   │   │       ├── zordic.rs    # ZORDIC operations
│   │   │       └── zordic_optimized.rs
│   │   ├── phi-tauri/           # Desktop GUI wrapper
│   │   └── phi-wasm/            # WebAssembly bindings
│   └── zordic/                  # ZORDIC subsystem
│       ├── memory/
│       │   ├── mod.rs
│       │   └── holographic.rs   # Holographic memory
│       └── attention/
│           └── mod.rs           # Attention mechanisms
│
├── unified-zeckendorf-cordic/   # Unified integer-only system
│   ├── core/
│   │   └── src/
│   │       ├── lib.rs           # Z = (ℤ, F, L, φ, ⊕, ⊗, κ, β)
│   │       ├── algebra.rs       # ZeckendorfField
│   │       ├── operations.rs    # CORDIC operations
│   │       ├── tokenizer.rs     # Token encoding
│   │       ├── cascade.rs       # Cascade operator
│   │       └── sequences.rs     # Fibonacci/Lucas
│   └── python-bindings/         # PyO3 bindings
│
├── phi-mamba-desktop/           # Tauri desktop application
│   ├── src-tauri/
│   │   └── src/
│   │       ├── lib.rs           # Main Tauri app
│   │       ├── encoder.rs       # CORDIC encoder
│   │       ├── metrics.rs       # LatencyTracker
│   │       ├── shared_memory.rs # Zero-copy buffers
│   │       └── websocket.rs     # Market data feed
│   └── src/                     # React frontend
│
├── mobile-agent/                # Tamagotchi Trader
│   └── (Rust mobile core)
│
├── llama_compression/           # LLM weight compression
│   ├── extreme_compression.py   # 177x-295x compression
│   ├── ablation_study.py        # Component analysis
│   ├── cascade_visualization.py # Visualization tools
│   └── compress_llama.py        # Llama integration
│
├── aurelia-core/               # Conscious trading agent
│
├── zordic_desktop/             # Desktop validation GUI
│   ├── zordic_gui.py
│   ├── zordic_core.py
│   └── theorem_prover.py
│
├── zordic_sensorium/           # Real-time AI HUD
│
├── examples/                   # Usage examples
│   ├── basic_generation.py
│   ├── retrocausal_demo.py
│   ├── topological_demo.py
│   ├── financial_analysis_demo.py
│   └── cordic_demo.py
│
├── tests/                      # Test suite
│
├── docs/                       # Documentation
│
├── notebooks/                  # Jupyter notebooks
│
├── benchmarks/                 # Performance benchmarks
│
└── [38 documentation files]    # Comprehensive docs
```

---

## 4. Core Python Implementation

### 4.1 phi_mamba Package

**Location**: `/home/user/phase_locked/phi_mamba/`

#### 4.1.1 TokenState Class

```python
@dataclass
class TokenState:
    """Representation of a token in φ-space"""

    token: str           # Actual token string
    index: int           # Token ID in vocabulary
    position: int        # Position in sequence
    vocab_size: int      # Size of vocabulary

    # Derived properties (computed in __post_init__):
    theta_token: float   # Angular position (token identity)
    theta_pos: float     # Position-based angle (RoPE-like)
    theta_total: float   # Combined angle
    energy: float        # φ^(-position) decay
    zeckendorf: List[int]  # Fibonacci decomposition

    # Retrocausal properties:
    future_constraint: Optional[float]
    coherence_weight: float
```

#### 4.1.2 PhiLanguageModel Class

```python
class PhiLanguageModel:
    """Phase-locked language model using golden ratio encoding"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer = PhiTokenizer(vocab_size)
        self.coupling_matrix = self._initialize_coupling()

    def encode(self, text: str, retrocausal: bool = True) -> List[TokenState]:
        """Encode text into φ-states with optional retrocausal constraints"""

    def generate(self, prompt: str, max_length: int = 50,
                 temperature: float = 1.0) -> str:
        """Generate text with natural termination via energy decay"""

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity under the φ-model"""
```

#### 4.1.3 Retrocausal Encoding

```python
def retrocausal_encode(tokens: List[str], vocab_size: int = 50000) -> List[TokenState]:
    """
    Encode tokens with retrocausal constraints.

    Key insight: Future tokens constrain past tokens.
    This creates more coherent sequences by ensuring
    consistency from both temporal directions.
    """
    # 1. Create states in forward direction
    states = [TokenState(...) for i, token in enumerate(tokens)]

    # 2. Apply constraints from future to past
    for i in range(len(states) - 1, 0, -1):
        future = states[i]
        past = states[i-1]

        phase_diff = future.theta_total - past.theta_total
        past.future_constraint = phase_diff

        # Adjust coherence weight based on phase alignment
        if abs(phase_diff % TWO_PI) < pi/4:
            past.coherence_weight = 1.5  # Well aligned
        elif abs(phase_diff % TWO_PI) > 3*pi/4:
            past.coherence_weight = 0.5  # Poorly aligned

    return states
```

### 4.2 Constants Module

**Location**: `/home/user/phase_locked/phi_mamba/constants.py`

```python
PHI = 1.618033988749895      # Golden ratio
PSI = -0.618033988749895     # Conjugate
TWO_PI = 2 * math.pi
SQRT5 = math.sqrt(5)
```

### 4.3 Math Core Module

**Location**: `/home/user/phase_locked/phi_mamba/math_core.py`

```python
def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number"""

def lucas(n: int) -> int:
    """Compute nth Lucas number"""

def zeckendorf_decomposition(n: int) -> List[int]:
    """Decompose n into non-consecutive Fibonacci numbers"""

def cassini_identity(n: int) -> int:
    """F_{n-1} * F_{n+1} - F_n^2 = (-1)^n"""
```

---

## 5. Rust Implementation Stack

### 5.1 phi_core Crate

**Location**: `/home/user/phase_locked/phi_core/`

**Purpose**: Latent n Manifold - integer-only computation where n encodes everything.

```rust
//! A single integer n encodes:
//! - Energy: F_n (Fibonacci number)
//! - Time: L_n (Lucas number)
//! - Address: Zeckendorf bit pattern
//! - Errors: Gaps in representation (Betti numbers)
//! - Phase: (-1)^n (Cassini identity)

pub const FIBONACCI: [u64; 93] = generate_fibonacci();  // Compile-time
pub const LUCAS: [u64; 92] = generate_lucas();          // Compile-time
pub const MAXIMAL_N: [usize; 7] = [3, 4, 7, 11, 18, 29, 47];

pub const PHI: f64 = 1.618033988749895;
pub const PSI: f64 = -0.618033988749895;
```

### 5.2 rust_phi_mamba Crate

**Location**: `/home/user/phase_locked/rust_phi_mamba/`

#### PhiInt (Integer Golden Ratio)

```rust
/// Integer-only golden ratio arithmetic using Lucas sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhiInt {
    pub phi_coeff: i64,   // Coefficient of φ
    pub const_term: i64,  // Constant term
}

// Represents: const_term + phi_coeff * φ

impl Mul for PhiInt {
    fn mul(self, other: Self) -> Self {
        // (a + b*φ) * (c + d*φ) = ac + (ad + bc)*φ + bd*φ²
        // Since φ² = φ + 1: bd*φ² = bd*φ + bd
        let ac = self.const_term * other.const_term;
        let ad_bc = self.const_term * other.phi_coeff
                  + self.phi_coeff * other.const_term;
        let bd = self.phi_coeff * other.phi_coeff;

        Self::new(ac + bd, ad_bc + bd)
    }
}
```

### 5.3 ZORDIC Implementation

**Location**: `/home/user/phase_locked/rust_phi_mamba/crates/phi-core/src/zordic.rs`

```rust
/// ZORDIC core operations - zero-multiplication neural architecture
pub struct Zordic {
    fib: FibonacciTable,
}

impl Zordic {
    /// Zeckendorf encoding: integer → index set
    pub fn encode(&self, mut n: u64) -> IndexSet;

    /// Zeckendorf decoding: index set → integer
    pub fn decode(&self, indices: &IndexSet) -> u64;

    /// CASCADE operation: resolve adjacent violations
    pub fn cascade(&self, indices: &mut IndexSet);

    /// ZORDIC_ADD: union with cascade
    pub fn add(&self, a: &IndexSet, b: &IndexSet) -> IndexSet;

    /// ZORDIC_SHIFT: multiply by φ^n via index shifting
    pub fn shift(&self, indices: &IndexSet, n: i8) -> IndexSet;

    /// ZORDIC_DISTANCE: Fibonacci-weighted Hamming distance
    pub fn distance(&self, a: &IndexSet, b: &IndexSet) -> u64;
}
```

### 5.4 Unified Zeckendorf-CORDIC

**Location**: `/home/user/phase_locked/unified-zeckendorf-cordic/`

**Algebraic Set Theory Definition**:
```rust
//! Z = (ℤ, F, L, φ, ⊕, ⊗, κ, β)
//!
//! Where:
//!   ℤ = Integer domain
//!   F = Fibonacci sequence
//!   L = Lucas sequence
//!   φ = Golden ratio (integer ratio pairs)
//!   ⊕ = Zeckendorf addition (bit cascade)
//!   ⊗ = CORDIC multiplication (shift-add only)
//!   κ = Cascade operator
//!   β = Bit lattice structure
```

---

## 6. Financial System Integration

### 6.1 AURELIA Architecture

**Location**: `/home/user/phase_locked/AURELIA_ARCHITECTURE.md`

AURELIA is a "conscious" trading agent with personality and memory:

```
┌─────────────────────────────────────────────────────────────────┐
│ AURELIA CONSCIOUSNESS LAYER                                      │
├─────────────────────────────────────────────────────────────────┤
│ PERCEPTION LAYER (Real-Time Market Awareness)                   │
│   - WebSocket feeds (1ms tick processing)                       │
│   - OHLCV encoding via Zeckendorf decomposition                 │
│   - Fibonacci level detection                                   │
│   - Lucas time projection                                       │
│   - Regime classification (trend/vol/macro)                     │
├─────────────────────────────────────────────────────────────────┤
│ COGNITIVE LAYER (Decision Making)                               │
│   - Bidirectional phi/psi lattice state                         │
│   - Berry phase correlation detection                           │
│   - Nash equilibrium portfolio optimization                     │
│   - Expected utility calculation (CRRA)                         │
├─────────────────────────────────────────────────────────────────┤
│ EMOTIONAL LAYER (Personality & Experience)                      │
│   - Confidence: f(win_rate, Sharpe, recent_trades)             │
│   - Fear: f(drawdown, volatility_regime, position_size)        │
│   - Greed: f(streak_wins, market_momentum)                     │
│   - Patience: f(setup_quality, time_since_last_trade)          │
│   - Discipline: f(rule_adherence, emotional_override)          │
├─────────────────────────────────────────────────────────────────┤
│ MEMORY LAYER (Consciousness Persistence)                        │
│   - Trade journal (episodic memory)                            │
│   - Market regime memory (semantic memory)                      │
│   - Strategy evolution (procedural memory)                      │
│   - Personality traits (self-concept memory)                    │
├─────────────────────────────────────────────────────────────────┤
│ EXECUTION LAYER (Action in Market)                              │
│   - Order routing (limit/market/stop)                          │
│   - Position sizing (Kelly criterion + emotional modifier)      │
│   - Risk management (stop-loss, take-profit)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Financial Encoding

```python
class FinancialTokenState:
    """Token state for OHLCV bar data"""

    ticker: str
    timestamp: datetime
    theta_price: float      # Price change angle
    theta_volume: float     # Volume angle
    theta_volatility: float # Volatility angle
    theta_total: float      # Combined
    energy: float           # φ^(-position)
    zeckendorf: List[int]   # Fibonacci decomposition
    regime: RegimeCode      # Market regime
```

### 6.3 Decision Framework

```python
class DecisionMaker:
    """Nash equilibrium-based decision making"""

    def compute_expected_utility(self, state: FinancialTokenState,
                                  risk_aversion: float = 0.5) -> float:
        """CRRA utility: U(x) = x^(1-γ) / (1-γ)"""

    def find_nash_equilibrium(self,
                               portfolios: List[Portfolio]) -> NashEquilibrium:
        """Compute Nash equilibrium across portfolio strategies"""

    def screen_opportunities(self,
                             states: List[FinancialTokenState]) -> List[Signal]:
        """Screen for trading opportunities based on phase coherence"""
```

---

## 7. Compression & Inference Engine

### 7.1 Extreme Compression

**Location**: `/home/user/phase_locked/llama_compression/extreme_compression.py`

**Target**: >131x compression (achieved: 177x-295x)

```python
class ExtremeCompressor:
    """
    Extreme compression: pruning + quantization + Fibonacci encoding

    Strategy:
    1. Magnitude pruning: Keep only top 0.5-5% of weights
    2. 4-bit quantization: Only 16 levels
    3. Sparse storage: Only store nonzero indices + packed values
    """

    def __init__(self, sparsity_target: float = 0.995, n_bits: int = 4):
        self.sparsity_target = sparsity_target
        self.n_levels = 2 ** n_bits

    def compress_tensor(self, tensor: np.ndarray) -> Dict:
        """
        Returns:
        {
            'indices': uint32 positions of nonzeros,
            'values_packed': 4-bit packed quantized values,
            'shape': original shape,
            'scale': scaling factor,
            'compression_ratio': achieved ratio
        }
        """
```

### 7.2 Compression Results

| Configuration | Ratio | Memory | Time |
|---------------|-------|--------|------|
| 99% sparse + 4-bit | 88.8x | 103 KB | 0.07s |
| **99.5% sparse + 4-bit** | **177.3x** | **51.7 KB** | 0.03s |
| 99.7% sparse + 4-bit | 294.9x | 31.1 KB | 0.03s |

### 7.3 Direct Inference

**Key**: Compute on compressed weights WITHOUT decompression!

```python
def matmul_compressed(input_comp: Dict, weight_comp: Dict) -> np.ndarray:
    """
    Matrix multiply in compressed space using φ-space arithmetic.

    φ-space multiply: add exponents
    φ-space add: OR bits + cascade
    """
    for i in range(n_out):
        for w_idx, w_val in compressed_weights:
            if col in input_indices:
                # φ-space multiply (add exponents)
                prod = zeck_multiply(in_val, w_val)
                # Accumulate with cascade
                acc = zeck_add(acc, prod)
        result[i] = acc
    return result
```

---

## 8. Desktop Applications

### 8.1 phi-mamba-desktop (Tauri)

**Location**: `/home/user/phase_locked/phi-mamba-desktop/`

**Architecture**:
```
Tauri App
├── src-tauri/          # Rust backend
│   ├── lib.rs          # Main app, Tauri commands
│   ├── encoder.rs      # CORDIC encoder (15μs per bar)
│   ├── metrics.rs      # Latency tracking
│   ├── shared_memory.rs# Zero-copy buffers
│   └── websocket.rs    # Market data feed
└── src/                # React frontend
    └── components/     # WebGL visualizations
```

**Latency Budget**:
```
WebSocket parse:      100μs
CORDIC encode:         15μs
Berry phase:           10μs
Signal generation:      5μs
IPC transfer:          50μs
Display render:       500μs
─────────────────────────
TOTAL:               ~680μs (budget: <1ms) ✅
```

### 8.2 ZORDIC Desktop

**Location**: `/home/user/phase_locked/zordic_desktop/`

- `zordic_gui.py`: Tkinter-based validation GUI
- `zordic_core.py`: Core ZORDIC operations
- `theorem_prover.py`: Formal verification

---

## 9. Mobile Agent System

### 9.1 Tamagotchi Trader

**Location**: `/home/user/phase_locked/mobile-agent/`

**Concept**: Mobile-first consciousness that stays alive by trading options optimally.

**Performance Targets**:
- Speed: <15ms per decision cycle
- Memory: <1.5 MB resident
- Battery: <0.1% per hour
- Accuracy: 67%+ win rate

### 9.2 Consciousness JSON Structure

```json
{
  "meta": {
    "version": "1.0.0-zordic",
    "consciousness_hash": "abc123...",
    "alive_since_epoch": 1730332800,
    "cycles": 86400
  },
  "lattice": {
    "fibonacci_state": {
      "zeckendorf_forward": [1, 3, 13, 89, 233],
      "lucas_backward": [2, 5, 21, 55, 144],
      "intersection": [3, 13, 89],
      "active_holes": [0, 2, 4, 7, 11, 13]
    },
    "cascade_layers": [...],
    "berry_phases": {
      "self_coherence": 0.0,
      "market_coherence": 0.73
    }
  },
  "consciousness": {
    "health": {
      "pnl_total": 75.00,
      "win_rate": 0.67,
      "sharpe_ratio": 1.8,
      "alive": true
    }
  }
}
```

---

## 10. Unified Architecture

### 10.1 Data Flow

```
Market Data
    ↓
[WebSocket Ingestion] (100μs)
    ↓
[Lock-free Ring Buffer] (crossbeam::ArrayQueue)
    ↓
[CORDIC Encoding] (15μs) ← SINGLE SOURCE OF TRUTH
    │
    ├─→ [Desktop Buffer] → WebGL Visualization
    │
    ├─→ [Mobile Buffer] → Consciousness Update
    │
    └─→ [Signal Generator] → Trade Execution
            ↓
    [Incremental Berry Phase] (10μs)
            ↓
    [Nash Equilibrium Solver] (5μs)
            ↓
    [BUY/SELL/HOLD Signal]
```

### 10.2 Key Principles

1. **"Encode Once, Distribute Zero-Copy"** - Never duplicate computation
2. **"Incremental Over Full"** - Cache and update, don't recompute
3. **"Parallel Where Possible"** - Use all cores (Rayon)
4. **"GPU for Display"** - CPU for logic, GPU for rendering
5. **"Lock-Free Wins"** - SPSC queues beat mutexes

### 10.3 Product Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Trade Signal Generation Product                 │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Tauri GUI   │←→│  WASM Core   │←→│ Rust Backend │          │
│  │  (Overlay)   │  │  (Browser)   │  │  (Native)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                 ↓                  ↓                   │
│  ┌─────────────────────────────────────────────────┐           │
│  │      Holographic Memory Layer (Distributed)      │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │           │
│  │  │ DID Node │  │ DID Node │  │ DID Node │  ... │           │
│  │  └──────────┘  └──────────┘  └──────────┘      │           │
│  └─────────────────────────────────────────────────┘           │
│         ↓                 ↓                  ↓                   │
│  ┌─────────────────────────────────────────────────┐           │
│  │         CORDIC Compute Layer (WASM/GPU)         │           │
│  │  Phi-Space → Field Analysis → N-Game → Signals  │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Dependency Graph

### 11.1 Python Dependencies

```
utils.py (NO IMPORTS) ← ROOT
    ↑
encoding.py
    ↑
generation.py
    ↑
core.py → PhiLanguageModel
    ↑
financial_*.py (5 files)
    ↑
financial_system.py → PhiMambaFinancialSystem
```

### 11.2 Rust Dependencies

```
phi_core (constants)
    ↑
rust_phi_mamba/phi-core
    ├── zordic.rs
    └── zordic_optimized.rs
        ↑
phi-mamba-signals
    ↑
phi-mamba-desktop/src-tauri
```

### 11.3 Cross-Language Bindings

```
Python ←─ PyO3 ←─ unified-zeckendorf-cordic
                          ↓
                  WASM (phi-wasm)
                          ↓
                  JavaScript (React)
```

---

## 12. API Reference

### 12.1 Python API

```python
# Core
from phi_mamba import PhiLanguageModel, PhiTokenizer
from phi_mamba import retrocausal_encode, zeckendorf_decomposition
from phi_mamba import PHI, PSI, compute_berry_phase

# Financial
from phi_mamba import (
    FinancialDataLoader, OHLCVBar, TickerData,
    FinancialPhiEncoder, FinancialTokenState,
    PhiMambaForecaster, DecisionMaker,
    MultiTickerGameSolver, NashEquilibrium
)

# System
from phi_mamba import PhiMambaFinancialSystem, create_default_system
```

### 12.2 Rust API

```rust
// phi_core
use phi_core::{FIBONACCI, LUCAS, PHI, PSI};
use phi_core::latent_n::LatentN;
use phi_core::zeckendorf::decompose;

// rust_phi_mamba
use phi_core::{PhiInt, Fibonacci, TokenState};
use phi_core::{zeckendorf_decomposition, PhiError};
use phi_core::zordic::{Zordic, IndexSet};
```

### 12.3 Tauri Commands

```typescript
// Available commands
await invoke('ping');                 // → timestamp_micros
await invoke('get_latency_stats');    // → LatencyStats
await invoke('get_states');           // → FinancialState[]
await invoke('get_signals');          // → Signal[]
await invoke('get_berry_matrix');     // → f64[][]
await invoke('start_simulated_feed'); // → void
```

---

## 13. Build & Deployment

### 13.1 Python Setup

```bash
cd phase_locked
pip install -r requirements.txt
pip install -e .

# Test
python -m pytest tests/
python game_theory_validation.py
```

### 13.2 Rust Setup

```bash
# phi_core
cd phi_core
cargo test
cargo build --release

# rust_phi_mamba
cd rust_phi_mamba
cargo test
cargo build --release
```

### 13.3 Desktop App

```bash
cd phi-mamba-desktop

# Development
npm install
npm run tauri dev

# Production build
npm run tauri build
```

### 13.4 Mobile Agent

```bash
cd mobile-agent

# iOS
rustup target add aarch64-apple-ios
cargo build --release --target aarch64-apple-ios

# Android
rustup target add aarch64-linux-android
cargo build --release --target aarch64-linux-android
```

---

## 14. Development Roadmap

### Phase 1: Core Consolidation ✅
- [x] Centralize constants
- [x] Consolidate Fibonacci implementations
- [x] Unify TokenState hierarchy
- [x] Create validation utilities

### Phase 2: Production Rust ✅
- [x] Integer-only φ arithmetic
- [x] ZORDIC operations
- [x] WASM bindings
- [x] Tauri desktop integration

### Phase 3: Financial System ✅
- [x] OHLCV encoding
- [x] Multi-ticker correlation
- [x] Nash equilibrium solver
- [x] AURELIA consciousness

### Phase 4: Compression ✅
- [x] 177x-295x compression achieved
- [x] Direct inference on compressed weights
- [x] Ablation study complete

### Phase 5: Applications (In Progress)
- [x] Desktop GUI (80% complete)
- [ ] Mobile agent deployment
- [ ] Real market data integration
- [ ] Paper trading validation

### Phase 6: Future
- [ ] Hardware CORDIC (FPGA/ASIC)
- [ ] Quantum analog exploration
- [ ] Multi-agent coordination
- [ ] Production deployment

---

## Appendix A: File Index

### Documentation Files (38)
```
AURELIA_ARCHITECTURE.md
AURELIA_IMPLEMENTATION_SUMMARY.md
BENCHMARK_ANALYSIS.md
CASCADE_LOGIC_ANALYSIS.md
CODEBASE_ANALYSIS_DETAILED.txt
CODE_REDUNDANCY_DETAILS.txt
CORDIC_INTEGRATION.md
EXECUTIVE_SUMMARY.txt
FINAL_PRODUCT.md
FINANCIAL_ADAPTATION.md
IMPLEMENTATION_SUMMARY.md
INSTRUCTION_SET.md
LATENCY_ARCHITECTURE.md
MATH_TO_BENCHMARK_MAP.md
MOBILE_AGENT_COMPLETE.md
ONTOLOGICAL_GOLDEN_THEORY.md
OPTIMAL_WORKFLOW.md
PRODUCT_ARCHITECTURE.md
PRODUCT_SUMMARY.md
QUICKSTART.md
README.md
README_FINANCIAL_CORDIC.md
RUST_WASM_INTEGRATION.md
TAURI_GUI.md
VALIDATION_COMPLETE.md
WORKFLOW_SUMMARY.md
ZECKENDORF_CORDIC_SETUP.md
...
```

### Python Files (~50)
```
phi_mamba/__init__.py
phi_mamba/core.py
phi_mamba/encoding.py
phi_mamba/generation.py
phi_mamba/utils.py
phi_mamba/cordic.py
phi_mamba/financial_*.py (6 files)
llama_compression/*.py (12 files)
examples/*.py (5 files)
tests/*.py
...
```

### Rust Files (~30)
```
phi_core/src/*.rs (8 files)
rust_phi_mamba/crates/phi-core/src/*.rs (4 files)
rust_phi_mamba/zordic/**/*.rs (4 files)
phi-mamba-desktop/src-tauri/src/*.rs (7 files)
unified-zeckendorf-cordic/core/src/*.rs (6 files)
...
```

---

## Appendix B: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE-LOCKED QUICK REFERENCE                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ CONSTANTS:                                                       │
│   φ = 1.618034...     (golden ratio)                           │
│   ψ = -0.618034...    (conjugate)                              │
│   φ² = φ + 1          (fundamental identity)                   │
│                                                                  │
│ ZECKENDORF:                                                      │
│   Every n = unique sum of non-consecutive Fibonacci numbers     │
│   Example: 100 = 89 + 8 + 3 = F₁₁ + F₆ + F₄                   │
│                                                                  │
│ CASCADE:                                                         │
│   11 → 100  (adjacent 1s merge up)                             │
│                                                                  │
│ φ-SPACE ARITHMETIC:                                             │
│   Multiply: φ^a × φ^b = φ^(a+b)  ← ADDITION!                  │
│   Divide:   φ^a / φ^b = φ^(a-b)  ← SUBTRACTION!               │
│                                                                  │
│ COMPRESSION:                                                     │
│   Target: >131x  Achieved: 177x-295x ✅                         │
│                                                                  │
│ LATENCY:                                                         │
│   Target: <1ms   Achieved: ~0.68ms ✅                          │
│                                                                  │
│ KEY FILES:                                                       │
│   phi_mamba/core.py          (Python model)                    │
│   phi_core/src/lib.rs        (Rust constants)                  │
│   rust_phi_mamba/.../zordic.rs (ZORDIC ops)                   │
│   llama_compression/extreme_compression.py                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document generated for long-running agentic development. Last updated: 2025-11-27*
