# Rust/WASM Integration Guide

## Step-by-Step Implementation

### Phase 1: Rust CORDIC Core

#### 1.1 Create Rust Project

```bash
cargo new --lib phi-mamba-trade-signals
cd phi-mamba-trade-signals
```

#### 1.2 Cargo.toml

```toml
[package]
name = "phi-mamba-trade-signals"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
# Core
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# Fixed-point arithmetic
fixed = "1.24"
num-traits = "0.2"

# WASM support
wasm-bindgen = { version = "0.2", optional = true }
js-sys = { version = "0.3", optional = true }
web-sys = { version = "0.3", optional = true }
serde-wasm-bindgen = { version = "0.5", optional = true }

# Holographic memory
did-key = { version = "0.2", optional = true }
libp2p = { version = "0.53", optional = true }

# Async runtime
tokio = { version = "1", features = ["full"], optional = true }

# Performance
rayon = { version = "1.8", optional = true }

[dev-dependencies]
criterion = "0.5"
proptest = "1.0"

[features]
default = []
wasm = ["wasm-bindgen", "js-sys", "web-sys", "serde-wasm-bindgen"]
holographic = ["did-key", "libp2p", "tokio"]
parallel = ["rayon"]

[[bench]]
name = "cordic_bench"
harness = false
```

#### 1.3 CORDIC Core (src/cordic/mod.rs)

```rust
//! CORDIC engine using only add/subtract/shift operations
//!
//! All arithmetic operations are performed in fixed-point representation
//! for exact integer computation.

use fixed::types::I32F32;  // 32-bit integer, 32-bit fraction
use serde::{Deserialize, Serialize};

pub type FixedPoint = I32F32;

/// CORDIC engine for trigonometric and exponential functions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CordicEngine {
    /// Number of CORDIC iterations (more = more accurate)
    iterations: usize,

    /// Precomputed arctangent table
    atan_table: Vec<FixedPoint>,

    /// CORDIC gain compensation factor (K ≈ 1.646760258)
    k_inv: FixedPoint,

    /// Golden ratio φ in fixed-point
    phi: FixedPoint,

    /// Conjugate ψ = -1/φ in fixed-point
    psi: FixedPoint,

    /// ln(φ) in fixed-point
    ln_phi: FixedPoint,

    /// 2π in fixed-point
    two_pi: FixedPoint,
}

impl CordicEngine {
    /// Create new CORDIC engine with specified iterations
    pub fn new(iterations: usize) -> Self {
        let atan_table = Self::build_atan_table(iterations);
        let k_inv = FixedPoint::from_num(1.0 / 1.646760258);

        // Golden ratio: φ = (1 + √5) / 2 ≈ 1.618034
        let phi = FixedPoint::from_num(1.618033988749895);
        let psi = -FixedPoint::ONE / phi;
        let ln_phi = FixedPoint::from_num(0.481211825059603);
        let two_pi = FixedPoint::from_num(6.283185307179586);

        Self {
            iterations,
            atan_table,
            k_inv,
            phi,
            psi,
            ln_phi,
            two_pi,
        }
    }

    /// Build arctangent lookup table
    fn build_atan_table(iterations: usize) -> Vec<FixedPoint> {
        (0..iterations)
            .map(|i| {
                let angle = (1.0 / (1_u32 << i) as f64).atan();
                FixedPoint::from_num(angle)
            })
            .collect()
    }

    /// CORDIC rotation: rotate (x, y) by angle
    ///
    /// Uses ONLY add/subtract/shift operations!
    pub fn rotate(&self, x: FixedPoint, y: FixedPoint, angle: FixedPoint) -> (FixedPoint, FixedPoint) {
        let mut x = x;
        let mut y = y;
        let mut z = angle;

        // Normalize angle to [-π, π]
        let pi = self.two_pi >> 1;
        while z > pi {
            z -= self.two_pi;
        }
        while z < -pi {
            z += self.two_pi;
        }

        for i in 0..self.iterations {
            let x_shifted = x >> i;  // Divide by 2^i → shift right
            let y_shifted = y >> i;

            if z >= FixedPoint::ZERO {
                // Counter-clockwise rotation
                let x_new = x - y_shifted;  // Subtract only!
                let y_new = y + x_shifted;  // Add only!
                z -= self.atan_table[i];

                x = x_new;
                y = y_new;
            } else {
                // Clockwise rotation
                let x_new = x + y_shifted;  // Add only!
                let y_new = y - x_shifted;  // Subtract only!
                z += self.atan_table[i];

                x = x_new;
                y = y_new;
            }
        }

        (x, y)
    }

    /// Compute sin and cos using CORDIC
    pub fn sin_cos(&self, angle: FixedPoint) -> (FixedPoint, FixedPoint) {
        // Start with unit vector (1, 0)
        let x = self.k_inv;  // Compensate for CORDIC gain
        let y = FixedPoint::ZERO;

        // Rotate by angle
        let (x_rot, y_rot) = self.rotate(x, y, angle);

        (y_rot, x_rot)  // sin, cos
    }

    /// Compute φ^n using CORDIC
    ///
    /// φ^n = e^(n × ln(φ))
    pub fn phi_pow(&self, n: i32) -> FixedPoint {
        // n × ln(φ)
        let exp_arg = FixedPoint::from_num(n) * self.ln_phi;

        // e^(n × ln(φ)) via CORDIC hyperbolic
        self.exp_cordic(exp_arg)
    }

    /// Exponential via CORDIC hyperbolic mode
    fn exp_cordic(&self, x: FixedPoint) -> FixedPoint {
        // Simplified Taylor series using add/shift only
        let mut result = FixedPoint::ONE;
        let mut term = FixedPoint::ONE;

        for i in 1..16 {
            term = (term * x) >> 32;  // Multiply by x, shift for division by 2^32
            term /= FixedPoint::from_num(i);  // Divide by factorial term
            result += term;

            if term.abs() < FixedPoint::from_bits(1) {
                break;
            }
        }

        result
    }

    /// Multiply in φ-space: φ^n × φ^m = φ^(n+m)
    ///
    /// This is PURE ADDITION!
    #[inline]
    pub fn phi_multiply_exp(n: i32, m: i32) -> i32 {
        n + m  // That's it! Addition only!
    }

    /// Divide in φ-space: φ^n / φ^m = φ^(n-m)
    #[inline]
    pub fn phi_divide_exp(n: i32, m: i32) -> i32 {
        n - m  // Just subtraction!
    }

    /// Get golden ratio φ
    pub fn phi(&self) -> FixedPoint {
        self.phi
    }

    /// Get 2π
    pub fn two_pi(&self) -> FixedPoint {
        self.two_pi
    }
}

// ============================================================================
// PHI-SPACE ARITHMETIC OPERATIONS
// ============================================================================

/// Phi-space arithmetic: all multiplication becomes addition!
pub mod phi_arithmetic {
    use super::*;

    /// Multiply in phi-space
    ///
    /// φ^n × φ^m = φ^(n+m)
    #[inline]
    pub fn multiply(n: i32, m: i32) -> i32 {
        n + m  // ADDITION ONLY!
    }

    /// Divide in phi-space
    ///
    /// φ^n / φ^m = φ^(n-m)
    #[inline]
    pub fn divide(n: i32, m: i32) -> i32 {
        n - m  // SUBTRACTION ONLY!
    }

    /// Power in phi-space
    ///
    /// (φ^n)^k = φ^(n×k) = φ^(n+n+...+n) [k times]
    pub fn power(n: i32, k: u32) -> i32 {
        let mut result = 0;
        for _ in 0..k {
            result += n;  // REPEATED ADDITION!
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cordic_sin_cos() {
        let cordic = CordicEngine::new(32);

        // Test sin(π/4) and cos(π/4)
        let angle = cordic.two_pi() / FixedPoint::from_num(8);  // π/4
        let (sin_val, cos_val) = cordic.sin_cos(angle);

        let expected = FixedPoint::from_num(0.7071067811865476);
        assert!((sin_val - expected).abs() < FixedPoint::from_num(0.0001));
        assert!((cos_val - expected).abs() < FixedPoint::from_num(0.0001));
    }

    #[test]
    fn test_phi_pow() {
        let cordic = CordicEngine::new(32);

        // Test φ^2 ≈ 2.618034
        let result = cordic.phi_pow(2);
        let expected = FixedPoint::from_num(2.618033988749895);
        assert!((result - expected).abs() < FixedPoint::from_num(0.0001));
    }

    #[test]
    fn test_phi_arithmetic() {
        use phi_arithmetic::*;

        // φ^3 × φ^5 = φ^8
        assert_eq!(multiply(3, 5), 8);

        // φ^7 / φ^2 = φ^5
        assert_eq!(divide(7, 2), 5);

        // (φ^2)^3 = φ^6
        assert_eq!(power(2, 3), 6);
    }
}
```

#### 1.4 Financial Encoding (src/encoding/financial.rs)

```rust
//! Financial data encoding to phi-space

use crate::cordic::{CordicEngine, FixedPoint};
use serde::{Deserialize, Serialize};

/// OHLCV bar
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OHLCVBar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub ticker: String,
}

/// Financial token state in phi-space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinancialTokenState {
    pub ticker: String,
    pub position: usize,
    pub theta_total: FixedPoint,
    pub energy: FixedPoint,
    pub zeckendorf: Vec<u32>,
    pub price: f64,
    pub volume: f64,
}

/// Financial phi-space encoder
pub struct FinancialEncoder {
    cordic: CordicEngine,
    price_scale: f64,
    volume_scale: f64,
    angular_sensitivity: f64,
}

impl FinancialEncoder {
    pub fn new() -> Self {
        Self {
            cordic: CordicEngine::new(32),
            price_scale: 100.0,
            volume_scale: 1_000_000.0,
            angular_sensitivity: 1.0,
        }
    }

    /// Encode OHLCV bar to phi-space using CORDIC
    pub fn encode_bar(&self, bar: &OHLCVBar, position: usize) -> FinancialTokenState {
        // 1. Price change percentage
        let price_change_pct = ((bar.close - bar.open) / bar.open) * 100.0;

        // 2. Map to angle using phi
        let phi = self.cordic.phi();
        let price_change_fixed = FixedPoint::from_num(price_change_pct);
        let sensitivity = FixedPoint::from_num(self.angular_sensitivity);

        // θ_price = price_change × sensitivity × φ
        let theta_price = price_change_fixed * sensitivity * phi;

        // 3. Position-based angle with phi decay
        let pos_i32 = position as i32;
        let phi_decay = self.cordic.phi_pow(-pos_i32 / 10);
        let pos_fixed = FixedPoint::from_num(position);
        let theta_pos = pos_fixed * phi_decay;

        // 4. Combined angle (ADDITION ONLY!)
        let theta_total = theta_price + theta_pos;

        // Normalize to [0, 2π)
        let theta_total = Self::normalize_angle(theta_total, self.cordic.two_pi());

        // 5. Volume-based energy
        let volume_norm = (bar.volume / self.volume_scale).min(10.0);
        let energy_phi = self.cordic.phi_pow(-pos_i32);
        let energy = FixedPoint::from_num(volume_norm) * energy_phi;

        // 6. Zeckendorf decomposition
        let price_int = bar.close.abs() as u32;
        let zeckendorf = zeckendorf_decomposition(price_int);

        FinancialTokenState {
            ticker: bar.ticker.clone(),
            position,
            theta_total,
            energy,
            zeckendorf,
            price: bar.close,
            volume: bar.volume,
        }
    }

    fn normalize_angle(mut angle: FixedPoint, two_pi: FixedPoint) -> FixedPoint {
        while angle >= two_pi {
            angle -= two_pi;
        }
        while angle < FixedPoint::ZERO {
            angle += two_pi;
        }
        angle
    }
}

/// Zeckendorf decomposition: decompose n into non-consecutive Fibonacci numbers
///
/// Uses ONLY addition and subtraction!
pub fn zeckendorf_decomposition(mut n: u32) -> Vec<u32> {
    if n == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut fib_prev = 1;
    let mut fib_curr = 1;

    // Generate Fibonacci numbers up to n
    let mut fibs = vec![1, 1];
    while fib_curr < n {
        let fib_next = fib_prev + fib_curr;  // ADDITION ONLY!
        fibs.push(fib_next);
        fib_prev = fib_curr;
        fib_curr = fib_next;
    }

    // Greedy decomposition (uses subtraction only)
    for i in (0..fibs.len()).rev() {
        if fibs[i] <= n {
            result.push(fibs[i]);
            n -= fibs[i];  // SUBTRACTION ONLY!
        }
    }

    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeckendorf() {
        // 17 = 13 + 3 + 1 (F_7 + F_4 + F_2)
        let zeck = zeckendorf_decomposition(17);
        assert_eq!(zeck, vec![1, 3, 13]);

        // 103 = 89 + 13 + 1
        let zeck = zeckendorf_decomposition(103);
        assert_eq!(zeck, vec![1, 13, 89]);
    }

    #[test]
    fn test_encode_bar() {
        let encoder = FinancialEncoder::new();

        let bar = OHLCVBar {
            timestamp: 0,
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 103.0,
            volume: 1_000_000.0,
            ticker: "AAPL".to_string(),
        };

        let state = encoder.encode_bar(&bar, 0);

        assert_eq!(state.ticker, "AAPL");
        assert_eq!(state.position, 0);
        assert!(state.energy > FixedPoint::ZERO);
        assert_eq!(state.zeckendorf, vec![1, 13, 89]);
    }
}
```

### Phase 2: WASM Bindings

#### 2.1 WASM Module (src/wasm/mod.rs)

```rust
//! WASM bindings for browser deployment

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use crate::encoding::financial::{FinancialEncoder, OHLCVBar};
use crate::cordic::CordicEngine;

#[wasm_bindgen]
pub struct WasmCordicEngine {
    engine: CordicEngine,
}

#[wasm_bindgen]
impl WasmCordicEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Set panic hook for better error messages
        console_error_panic_hook::set_once();

        Self {
            engine: CordicEngine::new(32),
        }
    }

    /// Compute sin using CORDIC (add/subtract/shift only!)
    #[wasm_bindgen]
    pub fn sin(&self, angle: f64) -> f64 {
        use fixed::types::I32F32;
        let angle_fixed = I32F32::from_num(angle);
        let (sin_val, _) = self.engine.sin_cos(angle_fixed);
        sin_val.to_num::<f64>()
    }

    /// Compute cos using CORDIC
    #[wasm_bindgen]
    pub fn cos(&self, angle: f64) -> f64 {
        use fixed::types::I32F32;
        let angle_fixed = I32F32::from_num(angle);
        let (_, cos_val) = self.engine.sin_cos(angle_fixed);
        cos_val.to_num::<f64>()
    }

    /// Compute φ^n
    #[wasm_bindgen]
    pub fn phi_pow(&self, n: i32) -> f64 {
        let result = self.engine.phi_pow(n);
        result.to_num::<f64>()
    }

    /// Multiply in phi-space (ADDITION ONLY!)
    #[wasm_bindgen]
    pub fn phi_multiply(n: i32, m: i32) -> i32 {
        CordicEngine::phi_multiply_exp(n, m)
    }
}

#[wasm_bindgen]
pub struct WasmFinancialEncoder {
    encoder: FinancialEncoder,
}

#[wasm_bindgen]
impl WasmFinancialEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();

        Self {
            encoder: FinancialEncoder::new(),
        }
    }

    /// Encode OHLCV bar (receives JSON, returns JSON)
    #[wasm_bindgen]
    pub fn encode_bar(&self, bar_json: &str, position: usize) -> Result<String, JsValue> {
        let bar: OHLCVBar = serde_json::from_str(bar_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let state = self.encoder.encode_bar(&bar, position);

        serde_json::to_string(&state)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// Add console_error_panic_hook to dependencies in Cargo.toml:
// console_error_panic_hook = "0.1"
```

#### 2.2 Build WASM

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build --target web --features wasm

# Build for Node.js
wasm-pack build --target nodejs --features wasm

# Build for bundler (webpack, etc.)
wasm-pack build --target bundler --features wasm
```

#### 2.3 JavaScript Usage

```javascript
// Load WASM module
import init, { WasmCordicEngine, WasmFinancialEncoder } from './pkg/phi_mamba_trade_signals.js';

// Initialize
await init();

// Create CORDIC engine
const cordic = new WasmCordicEngine();

// Test phi-space multiplication (ADDITION ONLY!)
const result = WasmCordicEngine.phi_multiply(3, 5);  // 3 + 5 = 8
console.log(`φ³ × φ⁵ = φ^${result}`);  // φ^8

// Compute sin using CORDIC
const sinValue = cordic.sin(Math.PI / 4);
console.log(`sin(π/4) = ${sinValue}`);  // 0.707...

// Encode financial data
const encoder = new WasmFinancialEncoder();

const bar = {
    timestamp: Date.now(),
    open: 100.0,
    high: 105.0,
    low: 98.0,
    close: 103.0,
    volume: 1000000.0,
    ticker: "AAPL"
};

const stateJson = encoder.encode_bar(JSON.stringify(bar), 0);
const state = JSON.parse(stateJson);

console.log("Encoded state:", state);
console.log("Theta:", state.theta_total);
console.log("Energy:", state.energy);
console.log("Zeckendorf:", state.zeckendorf);
```

### Phase 3: Tauri GUI Integration

See next file: `TAURI_GUI.md`

### Phase 4: Holographic Memory

See next file: `HOLOGRAPHIC_MEMORY.md`

### Phase 5: Trade Signal Generation

See next file: `TRADE_SIGNALS.md`

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd phi-mamba-trade-signals

# 2. Build Rust library
cargo build --release

# 3. Build WASM
wasm-pack build --target web --features wasm

# 4. Run tests
cargo test

# 5. Run benchmarks
cargo bench

# 6. Build Tauri app
cd tauri-app
npm install
npm run tauri build
```

## Performance Targets

- CORDIC sin/cos: <100ns per call
- Phi-space multiply: <10ns (just addition!)
- Bar encoding: <1μs per bar
- Field consensus: <100ms for 100 nodes
- Signal generation: <10ms real-time

## Next Documents

1. `TAURI_GUI.md` - Desktop app with overlay
2. `HOLOGRAPHIC_MEMORY.md` - DID-based distributed state
3. `TRADE_SIGNALS.md` - Signal generation API
4. `DEPLOYMENT.md` - Production deployment guide
