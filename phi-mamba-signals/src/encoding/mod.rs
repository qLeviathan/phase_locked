//! Financial Data Encoding to Phi-Space
//!
//! Converts market data (OHLCV bars) into φ-space representations
//! using Zeckendorf decomposition and Berry phase analysis

pub mod berry_phase;
pub mod financial;
pub mod zeckendorf;

pub use berry_phase::*;
pub use financial::*;
pub use zeckendorf::*;
