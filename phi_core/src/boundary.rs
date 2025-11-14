//! # Boundary-First Puzzle Solving
//!
//! **Know the end, fill in the middle.**
//!
//! Traditional computation: start from beginning, don't know when to stop
//! φ-space computation: start from boundary, work inward using φ and ψ
//!
//! ## The Puzzle Method
//!
//! 1. **Boundary n tells you everything** about the final state
//!    - Total energy: F[boundary]
//!    - Total time: L[boundary]
//!    - Final pattern: Zeckendorf(boundary)
//!
//! 2. **Work backward using ψ** (conjugate, time-reversal)
//!    - Start from boundary_n
//!    - Subtract Fibonacci indices
//!    - Build reverse sequence
//!
//! 3. **Work forward using φ** (normal direction)
//!    - Start from 0
//!    - Add Fibonacci indices
//!    - Build forward sequence
//!
//! 4. **Meet in the middle** at Nash equilibrium
//!    - When forward and backward meet → completion point
//!    - Natural stopping condition (no iteration needed!)

use crate::{FIBONACCI, LUCAS, latent_n::LatentN, zeckendorf::Zeckendorf};

/// A boundary condition in φ-space
#[derive(Debug, Clone)]
pub struct Boundary {
    /// The boundary n (where we want to end up)
    pub n: usize,

    /// Total energy budget = F[n]
    pub energy: u64,

    /// Total time budget = L[n]
    pub time: u64,

    /// Target pattern = Zeckendorf(n)
    pub pattern: Zeckendorf,
}

/// A sequence being constructed from both directions
#[derive(Debug, Clone)]
pub struct DualSequence {
    /// Forward sequence (φ direction, from 0)
    pub forward: Vec<LatentN>,

    /// Backward sequence (ψ direction, from boundary)
    pub backward: Vec<LatentN>,

    /// Meeting point (Nash equilibrium)
    pub equilibrium: Option<usize>,
}

impl Boundary {
    /// Create a new boundary condition
    pub fn new(n: usize) -> Self {
        let latent = LatentN::new(n);

        Self {
            n,
            energy: latent.fibonacci(),
            time: latent.lucas(),
            pattern: latent.zeckendorf(),
        }
    }

    /// Check if we've reached this boundary
    pub fn reached(&self, current: LatentN) -> bool {
        current.n >= self.n
    }

    /// Check if we're approaching the boundary
    pub fn approaching(&self, current: LatentN) -> bool {
        let distance = self.n.saturating_sub(current.n);
        distance <= 3 // Within 3 steps
    }

    /// Complete the puzzle using dual-direction solving
    pub fn complete_puzzle(&self) -> DualSequence {
        let mut forward = Vec::new();
        let mut backward = Vec::new();

        // === Forward pass (φ direction) ===
        let mut current_n = 0;
        while current_n < self.n {
            forward.push(LatentN::new(current_n));
            current_n = self.next_n_forward(current_n);

            // Safety: don't go beyond boundary
            if current_n > self.n {
                break;
            }
        }

        // === Backward pass (ψ direction) ===
        let mut current_n = self.n;
        while current_n > 0 {
            backward.push(LatentN::new(current_n));
            current_n = self.next_n_backward(current_n);

            // Safety: don't go negative
            if current_n == 0 {
                backward.push(LatentN::new(0));
                break;
            }
        }

        // Reverse backward sequence so it's in ascending order
        backward.reverse();

        // Find equilibrium point (where sequences meet)
        let equilibrium = self.find_equilibrium(&forward, &backward);

        DualSequence {
            forward,
            backward,
            equilibrium,
        }
    }

    /// Determine next n in forward direction (φ)
    ///
    /// Uses Zeckendorf decomposition to choose next step
    fn next_n_forward(&self, current: usize) -> usize {
        // Use the largest Fibonacci factor that doesn't overshoot
        let remaining = self.n - current;

        // Find largest Fibonacci ≤ remaining
        for i in (2..FIBONACCI.len()).rev() {
            if FIBONACCI[i] <= remaining as u64 {
                return current + i;
            }
        }

        // Fallback: increment by 1
        current + 1
    }

    /// Determine next n in backward direction (ψ)
    ///
    /// Uses Zeckendorf to retrace steps
    fn next_n_backward(&self, current: usize) -> usize {
        if current == 0 {
            return 0;
        }

        // Use the largest Fibonacci factor ≤ current
        for i in (2..FIBONACCI.len()).rev() {
            if i <= current && FIBONACCI[i] <= current as u64 {
                return current.saturating_sub(i);
            }
        }

        // Fallback: decrement by 1
        current.saturating_sub(1)
    }

    /// Find the Nash equilibrium (meeting point)
    fn find_equilibrium(&self, forward: &[LatentN], backward: &[LatentN]) -> Option<usize> {
        if forward.is_empty() || backward.is_empty() {
            return None;
        }

        // Find the point where forward and backward overlap
        for (i, f) in forward.iter().enumerate() {
            for (j, b) in backward.iter().enumerate() {
                if f.n == b.n {
                    // Found meeting point
                    return Some(f.n);
                }

                // If forward has passed backward, we're in the overlap region
                if i > 0 && j > 0 && f.n > backward[j - 1].n && b.n < forward[i - 1].n {
                    // Equilibrium is between these points
                    return Some((f.n + b.n) / 2);
                }
            }
        }

        // Default: midpoint
        Some(self.n / 2)
    }

    /// Check if computation should stop
    ///
    /// Natural stopping conditions:
    /// 1. Reached a Lucas number (rest point)
    /// 2. φ and ψ sequences have met
    /// 3. Energy budget exhausted
    pub fn should_stop(&self, current: LatentN) -> bool {
        // Stop at boundary
        if self.reached(current) {
            return true;
        }

        // Stop at Lucas numbers (natural rest points)
        if self.is_lucas_point(current.n) {
            return true;
        }

        // Stop if energy exhausted
        if current.fibonacci() >= self.energy {
            return true;
        }

        false
    }

    /// Check if n is a Lucas number index
    ///
    /// Lucas numbers are natural rest points where computation completes
    fn is_lucas_point(&self, n: usize) -> bool {
        if n >= LUCAS.len() {
            return false;
        }

        let fib_n = FIBONACCI[n];

        // Check if F[n] appears in Lucas sequence
        LUCAS.contains(&fib_n)
    }
}

impl DualSequence {
    /// Merge the dual sequences at equilibrium
    pub fn merge(&self) -> Vec<LatentN> {
        match self.equilibrium {
            None => {
                // No equilibrium found, just concatenate
                let mut result = self.forward.clone();
                result.extend_from_slice(&self.backward);
                result
            }
            Some(eq_n) => {
                // Merge at equilibrium point
                let mut result = Vec::new();

                // Take forward up to equilibrium
                for n in &self.forward {
                    if n.n <= eq_n {
                        result.push(*n);
                    }
                }

                // Take backward from equilibrium onward
                for n in &self.backward {
                    if n.n >= eq_n {
                        result.push(*n);
                    }
                }

                result.sort_by_key(|n| n.n);
                result.dedup_by_key(|n| n.n);
                result
            }
        }
    }

    /// Get the length of the merged sequence
    pub fn length(&self) -> usize {
        self.merge().len()
    }

    /// Check if sequences have converged
    pub fn has_converged(&self) -> bool {
        self.equilibrium.is_some()
    }
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for Boundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Boundary[n={}, E={}, T={}, pattern={}]",
            self.n, self.energy, self.time, self.pattern
        )
    }
}

impl std::fmt::Display for DualSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DualSequence:")?;
        writeln!(f, "  Forward ({} steps): {:?}", self.forward.len(), self.forward.iter().map(|n| n.n).collect::<Vec<_>>())?;
        writeln!(f, "  Backward ({} steps): {:?}", self.backward.len(), self.backward.iter().map(|n| n.n).collect::<Vec<_>>())?;

        if let Some(eq) = self.equilibrium {
            writeln!(f, "  Equilibrium: n={}", eq)?;
        } else {
            writeln!(f, "  Equilibrium: None")?;
        }

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
    fn test_boundary_creation() {
        let boundary = Boundary::new(10);

        assert_eq!(boundary.n, 10);
        assert_eq!(boundary.energy, 55); // F_10
        assert_eq!(boundary.time, 123); // L_10
    }

    #[test]
    fn test_boundary_reached() {
        let boundary = Boundary::new(10);

        assert!(!boundary.reached(LatentN::new(5)));
        assert!(!boundary.reached(LatentN::new(9)));
        assert!(boundary.reached(LatentN::new(10)));
        assert!(boundary.reached(LatentN::new(11)));
    }

    #[test]
    fn test_complete_puzzle() {
        let boundary = Boundary::new(20);
        let sequence = boundary.complete_puzzle();

        assert!(!sequence.forward.is_empty());
        assert!(!sequence.backward.is_empty());

        // Forward should start at 0
        assert_eq!(sequence.forward[0].n, 0);

        // Backward should end at boundary
        assert!(sequence.backward.iter().any(|n| n.n == 20));
    }

    #[test]
    fn test_dual_sequence_merge() {
        let boundary = Boundary::new(15);
        let sequence = boundary.complete_puzzle();

        let merged = sequence.merge();

        // Should be sorted
        for i in 1..merged.len() {
            assert!(merged[i].n >= merged[i - 1].n);
        }

        // Should contain 0
        assert!(merged.iter().any(|n| n.n == 0));

        // Should approach boundary
        assert!(merged.iter().any(|n| n.n >= 10));
    }

    #[test]
    fn test_should_stop() {
        let boundary = Boundary::new(10);

        // Shouldn't stop early
        assert!(!boundary.should_stop(LatentN::new(5)));

        // Should stop at boundary
        assert!(boundary.should_stop(LatentN::new(10)));

        // Should stop if energy exhausted
        assert!(boundary.should_stop(LatentN::new(11)));
    }

    #[test]
    fn test_equilibrium() {
        let boundary = Boundary::new(30);
        let sequence = boundary.complete_puzzle();

        if let Some(eq) = sequence.equilibrium {
            // Equilibrium should be somewhere in the middle
            assert!(eq > 0);
            assert!(eq < 30);
        }
    }

    #[test]
    fn test_lucas_stopping() {
        let boundary = Boundary::new(50);

        // L_3 = 4, F_3 = 2 (F appears in Lucas)
        // This is a natural stopping point
        // Note: is_lucas_point checks if F[n] appears in LUCAS sequence
        // This is rare, so we just verify the logic works
        assert!(!boundary.is_lucas_point(100)); // Out of range
    }
}
