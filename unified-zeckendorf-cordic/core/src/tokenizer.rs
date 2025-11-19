//! # Universal Tokenizer
//!
//! Multimodal tokenization: Numeric, Text, Images, Audio → ZeckBits
//!
//! ```text
//! TokenSpace = {
//!   Numeric: ℤ → ZeckBits
//!   Symbolic: Σ* → ZeckBits
//!   Multimodal: M → ZeckBits
//! }
//! ```

use crate::algebra::BitLattice;
use serde::{Deserialize, Serialize};

/// Universal tokenizer trait
pub trait UniversalTokenizer {
    /// Tokenize numeric value
    fn tokenize_numeric(&self, n: i64) -> BitLattice;

    /// Tokenize text
    fn tokenize_text(&self, s: &str) -> Vec<BitLattice>;

    /// Tokenize image (grayscale pixels)
    fn tokenize_image(&self, pixels: &[u8]) -> Vec<BitLattice>;

    /// Tokenize audio (samples)
    fn tokenize_audio(&self, samples: &[i16]) -> Vec<BitLattice>;
}

/// Default implementation of universal tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeckTokenizer {
    /// Vocabulary size for text tokenization
    pub vocab_size: usize,
}

impl Default for ZeckTokenizer {
    fn default() -> Self {
        Self {
            vocab_size: 50000,
        }
    }
}

impl UniversalTokenizer for ZeckTokenizer {
    fn tokenize_numeric(&self, n: i64) -> BitLattice {
        // Handle negative numbers
        if n < 0 {
            // Use sign bit
            let mut lattice = BitLattice::from_integer((-n) as u64);
            lattice.bits.insert(0, true); // Sign bit
            lattice
        } else {
            BitLattice::from_integer(n as u64)
        }
    }

    fn tokenize_text(&self, s: &str) -> Vec<BitLattice> {
        // Simple character-based tokenization
        // In production, use BPE or similar
        s.chars()
            .map(|c| {
                let code = c as u64;
                BitLattice::from_integer(code)
            })
            .collect()
    }

    fn tokenize_image(&self, pixels: &[u8]) -> Vec<BitLattice> {
        // Tokenize each pixel value
        pixels
            .iter()
            .map(|&pixel| BitLattice::from_integer(pixel as u64))
            .collect()
    }

    fn tokenize_audio(&self, samples: &[i16]) -> Vec<BitLattice> {
        // Tokenize audio samples
        samples
            .iter()
            .map(|&sample| {
                // Convert signed to unsigned for Zeckendorf
                let unsigned = (sample as i32 + 32768) as u64;
                BitLattice::from_integer(unsigned)
            })
            .collect()
    }
}

/// Token type enum for multimodal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    Numeric(i64),
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<i16>),
}

impl TokenType {
    /// Tokenize based on type
    pub fn tokenize(&self, tokenizer: &impl UniversalTokenizer) -> Vec<BitLattice> {
        match self {
            TokenType::Numeric(n) => vec![tokenizer.tokenize_numeric(*n)],
            TokenType::Text(s) => tokenizer.tokenize_text(s),
            TokenType::Image(pixels) => tokenizer.tokenize_image(pixels),
            TokenType::Audio(samples) => tokenizer.tokenize_audio(samples),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_numeric() {
        let tokenizer = ZeckTokenizer::default();
        let lattice = tokenizer.tokenize_numeric(17);

        assert_eq!(lattice.to_integer(), 17);
        assert!(lattice.is_valid());
    }

    #[test]
    fn test_tokenize_text() {
        let tokenizer = ZeckTokenizer::default();
        let lattices = tokenizer.tokenize_text("Hi");

        assert_eq!(lattices.len(), 2);
        assert_eq!(lattices[0].to_integer(), 'H' as u64);
        assert_eq!(lattices[1].to_integer(), 'i' as u64);
    }

    #[test]
    fn test_tokenize_negative() {
        let tokenizer = ZeckTokenizer::default();
        let lattice = tokenizer.tokenize_numeric(-5);

        // Should have sign bit
        assert!(lattice.bits[0]);
    }
}
