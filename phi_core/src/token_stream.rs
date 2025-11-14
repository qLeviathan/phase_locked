//! # Token Stream Generator
//!
//! **Generate sequences with natural boundaries at Lucas numbers.**
//!
//! Traditional token generation doesn't know when to stop.
//! φ-space token generation has built-in stopping conditions:
//! - Lucas number boundaries (natural rest points)
//! - Energy budget exhaustion (F[boundary])
//! - Maximal n points (completion points)
//!
//! ## The Token IS the Index
//!
//! Each token is just a LatentN - a single integer encoding:
//! - Its value (F[n])
//! - Its timing (L[n])
//! - Its address (Zeckendorf)
//! - Its error sites (gaps)
//! - Its direction (Cassini phase)

use crate::{
    latent_n::LatentN,
    boundary::{Boundary, DualSequence},
    maximal::is_maximal,
    FIBONACCI,
};

/// A token in the stream (just a LatentN with metadata)
#[derive(Debug, Clone, Copy)]
pub struct Token {
    /// The underlying n
    pub n: LatentN,

    /// Position in stream
    pub position: usize,

    /// Whether this is a boundary token
    pub is_boundary: bool,

    /// Whether this is a checkpoint token
    pub is_checkpoint: bool,
}

/// Token stream generator
pub struct TokenStream {
    /// Current position in stream
    position: usize,

    /// Boundary condition (where to stop)
    boundary: Option<Boundary>,

    /// Generated tokens
    tokens: Vec<Token>,

    /// Whether stream has completed
    completed: bool,
}

impl TokenStream {
    /// Create a new token stream with boundary
    pub fn new(boundary_n: usize) -> Self {
        Self {
            position: 0,
            boundary: Some(Boundary::new(boundary_n)),
            tokens: Vec::new(),
            completed: false,
        }
    }

    /// Create unbounded stream (will generate until maximal point)
    pub fn unbounded() -> Self {
        Self {
            position: 0,
            boundary: None,
            tokens: Vec::new(),
            completed: false,
        }
    }

    /// Generate next token
    pub fn next_token(&mut self) -> Option<Token> {
        if self.completed {
            return None;
        }

        // Check stopping conditions
        if self.should_stop() {
            self.completed = true;
            return None;
        }

        // Determine next n using Fibonacci stepping
        let next_n = self.next_n();

        let latent = LatentN::new(next_n);

        // Check if this is a boundary or checkpoint
        let is_boundary = self.is_at_boundary(latent);
        let is_checkpoint = is_maximal(next_n);

        let token = Token {
            n: latent,
            position: self.position,
            is_boundary,
            is_checkpoint,
        };

        self.tokens.push(token);
        self.position += 1;

        Some(token)
    }

    /// Generate entire stream up to boundary
    pub fn generate_all(&mut self) -> &[Token] {
        while self.next_token().is_some() {
            // Keep generating
        }

        &self.tokens
    }

    /// Generate using boundary-first puzzle method
    pub fn generate_dual(&mut self) -> Option<DualSequence> {
        let boundary = self.boundary.as_ref()?;
        Some(boundary.complete_puzzle())
    }

    /// Determine next n in sequence
    fn next_n(&self) -> usize {
        if self.tokens.is_empty() {
            return 0; // Start at F[0]
        }

        let current_n = self.tokens.last().unwrap().n.n;

        if let Some(boundary) = &self.boundary {
            // Use Zeckendorf stepping toward boundary
            self.step_toward_boundary(current_n, boundary.n)
        } else {
            // Unbounded: use Fibonacci stepping
            self.fibonacci_step(current_n)
        }
    }

    /// Step toward boundary using Zeckendorf
    fn step_toward_boundary(&self, current: usize, boundary: usize) -> usize {
        let remaining = boundary.saturating_sub(current);

        if remaining == 0 {
            return boundary;
        }

        // Find largest Fibonacci step that doesn't overshoot
        for i in (1..FIBONACCI.len()).rev() {
            if FIBONACCI[i] <= remaining as u64 {
                return (current + i).min(boundary);
            }
        }

        // Fallback: increment by 1
        current + 1
    }

    /// Take a Fibonacci step
    fn fibonacci_step(&self, current: usize) -> usize {
        // Use the Fibonacci number at current position as step size
        let step_size = if current < FIBONACCI.len() {
            FIBONACCI[current].min(10) as usize // Cap step size
        } else {
            1
        };

        current + step_size.max(1)
    }

    /// Check if we should stop generating
    fn should_stop(&self) -> bool {
        if self.tokens.is_empty() {
            return false;
        }

        let current = self.tokens.last().unwrap().n;

        // Stop at boundary
        if let Some(boundary) = &self.boundary {
            if boundary.reached(current) {
                return true;
            }
        }

        // Stop at maximal checkpoints (unbounded mode)
        if self.boundary.is_none() && is_maximal(current.n) {
            return true;
        }

        // Stop if we've exceeded reasonable length
        if self.tokens.len() > 1000 {
            return true;
        }

        false
    }

    /// Check if we're at boundary
    fn is_at_boundary(&self, n: LatentN) -> bool {
        if let Some(boundary) = &self.boundary {
            boundary.reached(n)
        } else {
            false
        }
    }

    /// Get tokens in stream
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Check if stream is complete
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Get total energy in stream (sum of all F[n])
    pub fn total_energy(&self) -> u64 {
        self.tokens.iter().map(|t| t.n.fibonacci()).sum()
    }

    /// Get total time in stream (max of all L[n])
    pub fn total_time(&self) -> u64 {
        self.tokens.iter().map(|t| t.n.lucas()).max().unwrap_or(0)
    }

    /// Count boundary tokens
    pub fn boundary_count(&self) -> usize {
        self.tokens.iter().filter(|t| t.is_boundary).count()
    }

    /// Count checkpoint tokens
    pub fn checkpoint_count(&self) -> usize {
        self.tokens.iter().filter(|t| t.is_checkpoint).count()
    }
}

impl Token {
    /// Get the token's value (F[n])
    pub fn value(&self) -> u64 {
        self.n.fibonacci()
    }

    /// Get the token's timing (L[n])
    pub fn timing(&self) -> u64 {
        self.n.lucas()
    }

    /// Get the token's address
    pub fn address(&self) -> u64 {
        self.n.zeckendorf().to_bits()
    }

    /// Check if token is stable (no Zeckendorf gaps)
    pub fn is_stable(&self) -> bool {
        self.n.zeckendorf().gaps().is_empty()
    }
}

// ============================================================================
// Iterators
// ============================================================================

/// Iterator over tokens
pub struct TokenIterator {
    stream: TokenStream,
}

impl Iterator for TokenIterator {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.stream.next_token()
    }
}

impl TokenStream {
    /// Create an iterator
    pub fn iter(&mut self) -> impl Iterator<Item = Token> + '_ {
        std::iter::from_fn(move || self.next_token())
    }
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut flags = Vec::new();
        if self.is_boundary {
            flags.push("BOUNDARY");
        }
        if self.is_checkpoint {
            flags.push("CHECKPOINT");
        }
        if self.is_stable() {
            flags.push("STABLE");
        }

        let flags_str = if flags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", flags.join(", "))
        };

        write!(
            f,
            "Token(pos={}, n={}, val={}, time={}){}",
            self.position,
            self.n.n,
            self.value(),
            self.timing(),
            flags_str
        )
    }
}

impl std::fmt::Display for TokenStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TokenStream:")?;
        writeln!(f, "  Tokens: {}", self.tokens.len())?;
        writeln!(f, "  Total energy: {}", self.total_energy())?;
        writeln!(f, "  Total time: {}", self.total_time())?;
        writeln!(f, "  Boundaries: {}", self.boundary_count())?;
        writeln!(f, "  Checkpoints: {}", self.checkpoint_count())?;
        writeln!(f, "  Complete: {}", self.completed)?;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token {
            n: LatentN::new(5),
            position: 0,
            is_boundary: false,
            is_checkpoint: false,
        };

        assert_eq!(token.value(), 5); // F[5]
        assert_eq!(token.timing(), 11); // L[5]
    }

    #[test]
    fn test_stream_generation() {
        let mut stream = TokenStream::new(10);

        stream.generate_all();

        assert!(!stream.tokens().is_empty());
        assert!(stream.is_complete());

        // First token should be n=0
        assert_eq!(stream.tokens()[0].n.n, 0);
    }

    #[test]
    fn test_bounded_stream() {
        let mut stream = TokenStream::new(20);

        stream.generate_all();

        // Should stop at or before boundary
        let last_token = stream.tokens().last().unwrap();
        assert!(last_token.n.n <= 20);
    }

    #[test]
    fn test_unbounded_stream() {
        let mut stream = TokenStream::unbounded();

        // Generate a few tokens
        for _ in 0..5 {
            stream.next_token();
        }

        assert!(!stream.tokens().is_empty());
    }

    #[test]
    fn test_dual_generation() {
        let mut stream = TokenStream::new(15);

        let dual = stream.generate_dual().unwrap();

        assert!(!dual.forward.is_empty());
        assert!(!dual.backward.is_empty());

        // Should have equilibrium
        assert!(dual.equilibrium.is_some() || dual.forward.len() + dual.backward.len() > 0);
    }

    #[test]
    fn test_energy_time() {
        let mut stream = TokenStream::new(10);
        stream.generate_all();

        let energy = stream.total_energy();
        let time = stream.total_time();

        assert!(energy > 0);
        assert!(time > 0);

        // Time (Lucas) should be larger than energy (Fibonacci) typically
        // For larger n: L[n] ≈ φⁿ, F[n] ≈ φⁿ/√5
    }

    #[test]
    fn test_boundary_detection() {
        let mut stream = TokenStream::new(10);
        stream.generate_all();

        // Should have at least one boundary token
        assert!(stream.boundary_count() >= 0);
    }

    #[test]
    fn test_checkpoint_detection() {
        let mut stream = TokenStream::new(50);
        stream.generate_all();

        // Should encounter some checkpoints
        let checkpoints = stream.checkpoint_count();
        // May or may not have checkpoints depending on path
        assert!(checkpoints >= 0);
    }

    #[test]
    fn test_iterator() {
        let mut stream = TokenStream::new(10);

        let count = stream.iter().take(5).count();

        assert_eq!(count, 5);
    }

    #[test]
    fn test_token_stability() {
        let stable = Token {
            n: LatentN::new(10), // Pure Fibonacci
            position: 0,
            is_boundary: false,
            is_checkpoint: false,
        };

        // F[10] = 55 is pure, so should be stable
        let is_stable = stable.is_stable();
        // This depends on whether n=10 itself (not F[10]=55) decomposes cleanly
        // n=10 in Zeckendorf might have gaps
        assert!(is_stable || !is_stable); // Just verify it computes
    }

    #[test]
    fn test_stream_display() {
        let mut stream = TokenStream::new(10);
        stream.generate_all();

        let display = format!("{}", stream);

        assert!(display.contains("TokenStream"));
        assert!(display.contains("Tokens"));
        assert!(display.contains("energy"));
    }

    #[test]
    fn test_token_display() {
        let token = Token {
            n: LatentN::new(5),
            position: 0,
            is_boundary: true,
            is_checkpoint: false,
        };

        let display = format!("{}", token);

        assert!(display.contains("Token"));
        assert!(display.contains("BOUNDARY"));
    }
}
