//! WASM bindings for Phi-Mamba
//! 
//! Exposes the core Phi-Mamba functionality to JavaScript/TypeScript

use phi_core::{zeckendorf_decomposition, Fibonacci, PhiInt, TokenState, PHI, PSI};
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// WASM wrapper for TokenState
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTokenState {
    token: String,
    theta: f64,
    energy: f64,
    shells: Vec<usize>,
    phase: f64,
    future_constraint: Option<f64>,
}

#[wasm_bindgen]
impl WasmTokenState {
    #[wasm_bindgen(constructor)]
    pub fn new(token: String) -> Self {
        Self {
            token,
            theta: 0.0,
            energy: 1.0,
            shells: vec![],
            phase: 0.0,
            future_constraint: None,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn token(&self) -> String {
        self.token.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn energy(&self) -> f64 {
        self.energy
    }
    
    #[wasm_bindgen(getter)]
    pub fn phase(&self) -> f64 {
        self.phase
    }
    
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// WASM wrapper for Fibonacci operations
#[wasm_bindgen]
pub struct WasmFibonacci {
    fib: Fibonacci,
}

#[wasm_bindgen]
impl WasmFibonacci {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            fib: Fibonacci::new(),
        }
    }
    
    pub fn get(&mut self, n: usize) -> Result<String, JsValue> {
        if n > 93 {
            return Err(JsValue::from_str("Fibonacci index too large for u64"));
        }
        Ok(self.fib.get(n).to_string())
    }
}

/// WASM wrapper for Zeckendorf decomposition
#[wasm_bindgen]
pub fn wasm_zeckendorf_decomposition(n: u32) -> Result<Vec<u32>, JsValue> {
    zeckendorf_decomposition(n as u64)
        .map(|v| v.into_iter().map(|x| x as u32).collect())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// WASM wrapper for PhiInt arithmetic
#[wasm_bindgen]
pub struct WasmPhiInt {
    inner: PhiInt,
}

#[wasm_bindgen]
impl WasmPhiInt {
    #[wasm_bindgen(constructor)]
    pub fn new(const_term: i32, phi_coeff: i32) -> Self {
        Self {
            inner: PhiInt::new(const_term as i64, phi_coeff as i64),
        }
    }
    
    pub fn from_int(n: i32) -> Self {
        Self {
            inner: PhiInt::from_int(n as i64),
        }
    }
    
    pub fn phi() -> Self {
        Self {
            inner: PhiInt::phi(),
        }
    }
    
    pub fn add(&self, other: &WasmPhiInt) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }
    
    pub fn multiply(&self, other: &WasmPhiInt) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }
    
    pub fn to_string(&self) -> String {
        if self.inner.phi_coeff == 0 {
            self.inner.const_term.to_string()
        } else if self.inner.const_term == 0 {
            format!("{}φ", self.inner.phi_coeff)
        } else {
            format!("{} + {}φ", self.inner.const_term, self.inner.phi_coeff)
        }
    }
}

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("Phi-Mamba WASM initialized");
    console_log!("Golden ratio (φ) = {}", PHI);
    console_log!("Conjugate (ψ) = {}", PSI);
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    fn test_wasm_fibonacci() {
        let mut fib = WasmFibonacci::new();
        assert_eq!(fib.get(10).unwrap(), "55");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_zeckendorf() {
        let result = wasm_zeckendorf_decomposition(50).unwrap();
        assert_eq!(result, vec![8, 5, 2]);
    }
}