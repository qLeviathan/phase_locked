//! Core implementation of Phi-Mamba in Rust
//! 
//! This crate provides the fundamental mathematical primitives and game-theoretic
//! foundations for language modeling using golden ratio arithmetic.

use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};
use thiserror::Error;

/// The golden ratio constant φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// The conjugate golden ratio ψ = 1/φ = φ - 1
pub const PSI: f64 = 0.6180339887498949;

/// Error types for Phi operations
#[derive(Error, Debug)]
pub enum PhiError {
    #[error("Invalid Fibonacci index: {0}")]
    InvalidFibonacciIndex(usize),
    
    #[error("Zeckendorf decomposition failed for: {0}")]
    ZeckendorfFailed(u64),
    
    #[error("Energy calculation overflow")]
    EnergyOverflow,
}

/// Result type for Phi operations
pub type PhiResult<T> = Result<T, PhiError>;

/// Represents a token state in the game-theoretic framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenState {
    /// Token identifier
    pub token: String,
    
    /// Angular position in phase space
    pub theta: f64,
    
    /// Energy level (φ^(-position))
    pub energy: f64,
    
    /// Zeckendorf decomposition (Fibonacci representation)
    pub shells: Vec<usize>,
    
    /// Berry phase
    pub phase: f64,
    
    /// Retrocausal constraint from future
    pub future_constraint: Option<f64>,
}

/// Fibonacci number generator using integer-only arithmetic
pub struct Fibonacci {
    cache: Vec<u64>,
}

impl Fibonacci {
    pub fn new() -> Self {
        Self {
            cache: vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        }
    }
    
    /// Get the nth Fibonacci number (0-indexed)
    pub fn get(&mut self, n: usize) -> u64 {
        while self.cache.len() <= n {
            let len = self.cache.len();
            let next = self.cache[len - 1] + self.cache[len - 2];
            self.cache.push(next);
        }
        self.cache[n]
    }
}

/// Zeckendorf decomposition: represent n as sum of non-consecutive Fibonacci numbers
pub fn zeckendorf_decomposition(n: u64) -> PhiResult<Vec<usize>> {
    if n == 0 {
        return Ok(vec![]);
    }
    
    let mut fib = Fibonacci::new();
    let mut decomposition = Vec::new();
    let mut remaining = n;
    
    // Find largest Fibonacci number <= n
    let mut k = 0;
    while fib.get(k + 1) <= n {
        k += 1;
    }
    
    // Greedy algorithm for Zeckendorf decomposition
    while remaining > 0 && k > 0 {
        let fib_k = fib.get(k);
        if fib_k <= remaining {
            decomposition.push(k);
            remaining -= fib_k;
            k = k.saturating_sub(2); // Skip next to ensure non-consecutive
        } else {
            k -= 1;
        }
    }
    
    Ok(decomposition)
}

/// Integer-only golden ratio arithmetic using Lucas sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhiInt {
    /// Coefficient of φ in the representation a + b*φ
    pub phi_coeff: i64,
    /// Constant term in the representation a + b*φ
    pub const_term: i64,
}

impl PhiInt {
    pub fn new(const_term: i64, phi_coeff: i64) -> Self {
        Self { phi_coeff, const_term }
    }
    
    /// Create from a single integer (n = n + 0*φ)
    pub fn from_int(n: i64) -> Self {
        Self::new(n, 0)
    }
    
    /// The golden ratio itself (0 + 1*φ)
    pub fn phi() -> Self {
        Self::new(0, 1)
    }
}

impl Add for PhiInt {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self::new(
            self.const_term + other.const_term,
            self.phi_coeff + other.phi_coeff,
        )
    }
}

impl Sub for PhiInt {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        Self::new(
            self.const_term - other.const_term,
            self.phi_coeff - other.phi_coeff,
        )
    }
}

impl Mul for PhiInt {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        // (a + b*φ) * (c + d*φ) = ac + (ad + bc)*φ + bd*φ²
        // Since φ² = φ + 1, we have bd*φ² = bd*φ + bd
        let ac = self.const_term * other.const_term;
        let ad_bc = self.const_term * other.phi_coeff + self.phi_coeff * other.const_term;
        let bd = self.phi_coeff * other.phi_coeff;
        
        Self::new(ac + bd, ad_bc + bd)
    }
}

pub mod zordic;
pub mod zordic_optimized;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fibonacci() {
        let mut fib = Fibonacci::new();
        assert_eq!(fib.get(0), 0);
        assert_eq!(fib.get(1), 1);
        assert_eq!(fib.get(10), 55);
        assert_eq!(fib.get(20), 6765);
    }
    
    #[test]
    fn test_zeckendorf() {
        assert_eq!(zeckendorf_decomposition(100).unwrap(), vec![10, 7, 4, 2]);
        assert_eq!(zeckendorf_decomposition(50).unwrap(), vec![8, 5, 2]);
        assert_eq!(zeckendorf_decomposition(1).unwrap(), vec![1]);
    }
    
    #[test]
    fn test_phi_arithmetic() {
        let phi = PhiInt::phi();
        let phi_squared = phi * phi;
        
        // φ² = φ + 1
        assert_eq!(phi_squared, PhiInt::new(1, 1));
        
        // φ³ = φ² + φ = 2φ + 1
        let phi_cubed = phi_squared * phi;
        assert_eq!(phi_cubed, PhiInt::new(1, 2));
    }
}