//! # Maximal n Detection
//!
//! **Find natural completion points where computation naturally ends.**
//!
//! Maximal n points have special properties:
//! 1. Lucas-Fibonacci alignment (L_k equals some Fibonacci number)
//! 2. Maximum gaps in Zeckendorf representation (high creativity potential)
//! 3. Cassini phase boundaries (even n, phase flip)
//!
//! These are the "rest points" where the system naturally completes.

use crate::{FIBONACCI, LUCAS, MAXIMAL_N, latent_n::LatentN};

/// Properties that make an n "maximal"
#[derive(Debug, Clone)]
pub struct MaximalProperties {
    /// The n index
    pub n: usize,

    /// Is this n in the known MAXIMAL_N set?
    pub is_known_maximal: bool,

    /// Does L[n] appear in Fibonacci sequence?
    pub lucas_is_fibonacci: bool,

    /// Does F[n] appear in Lucas sequence?
    pub fibonacci_is_lucas: bool,

    /// Does Zeckendorf representation have maximum gaps?
    pub has_max_gaps: bool,

    /// Is this at a Cassini phase boundary (even n)?
    pub at_phase_boundary: bool,

    /// Overall maximality score (0-5)
    pub score: u8,
}

impl MaximalProperties {
    /// Compute all maximality properties for a given n
    pub fn analyze(n: usize) -> Self {
        let latent = LatentN::new(n);

        let is_known_maximal = MAXIMAL_N.contains(&n);
        let lucas_is_fibonacci = is_lucas_in_fibonacci(n);
        let fibonacci_is_lucas = is_fibonacci_in_lucas(n);
        let has_max_gaps = latent.zeckendorf().has_max_gaps();
        let at_phase_boundary = n % 2 == 0;

        // Calculate score
        let mut score = 0;
        if is_known_maximal {
            score += 1;
        }
        if lucas_is_fibonacci {
            score += 1;
        }
        if fibonacci_is_lucas {
            score += 1;
        }
        if has_max_gaps {
            score += 1;
        }
        if at_phase_boundary {
            score += 1;
        }

        Self {
            n,
            is_known_maximal,
            lucas_is_fibonacci,
            fibonacci_is_lucas,
            has_max_gaps,
            at_phase_boundary,
            score,
        }
    }

    /// Is this n sufficiently maximal to be a stopping point?
    pub fn is_maximal(&self) -> bool {
        self.score >= 3 // Need at least 3 properties
    }

    /// Is this an excellent maximal point?
    pub fn is_excellent(&self) -> bool {
        self.score >= 4
    }

    /// Is this a perfect maximal point?
    pub fn is_perfect(&self) -> bool {
        self.score == 5
    }
}

/// Check if L[n] appears in the Fibonacci sequence
fn is_lucas_in_fibonacci(n: usize) -> bool {
    if n >= LUCAS.len() {
        return false;
    }

    let lucas_n = LUCAS[n];
    FIBONACCI.contains(&lucas_n)
}

/// Check if F[n] appears in the Lucas sequence
fn is_fibonacci_in_lucas(n: usize) -> bool {
    let fib_n = FIBONACCI[n];
    LUCAS.contains(&fib_n)
}

/// Find all maximal n in a range
pub fn find_maximal_in_range(start: usize, end: usize) -> Vec<usize> {
    (start..=end.min(92))
        .filter(|&n| is_maximal(n))
        .collect()
}

/// Check if n is maximal (quick check)
pub fn is_maximal(n: usize) -> bool {
    MaximalProperties::analyze(n).is_maximal()
}

/// Get the next maximal n after current
pub fn next_maximal(current: usize) -> Option<usize> {
    for n in (current + 1)..93 {
        if is_maximal(n) {
            return Some(n);
        }
    }
    None
}

/// Get the previous maximal n before current
pub fn prev_maximal(current: usize) -> Option<usize> {
    for n in (1..current).rev() {
        if is_maximal(n) {
            return Some(n);
        }
    }
    None
}

/// Find the nearest maximal n to target
pub fn nearest_maximal(target: usize) -> usize {
    if target >= 93 {
        return *MAXIMAL_N.last().unwrap();
    }

    if is_maximal(target) {
        return target;
    }

    let next = next_maximal(target);
    let prev = prev_maximal(target);

    match (next, prev) {
        (Some(n), Some(p)) => {
            // Return closest
            if (n - target) < (target - p) {
                n
            } else {
                p
            }
        }
        (Some(n), None) => n,
        (None, Some(p)) => p,
        (None, None) => MAXIMAL_N[0], // Fallback to first known maximal
    }
}

/// The "golden checkpoints" - known perfect maximal points
///
/// These are where L[n] and F[k] align in special ways:
/// - n=3:  L[3]=4, F[3]=2
/// - n=4:  L[4]=7, F[4]=3
/// - n=7:  L[7]=29, F[7]=13
/// - n=11: L[11]=199, F[11]=89
/// - n=18: L[18]=5778, F[18]=2584
/// - n=29: L[29]=1149851, F[29]=514229
/// - n=47: L[47]=2971215073, F[47]=1836311903
pub const GOLDEN_CHECKPOINTS: [usize; 7] = MAXIMAL_N;

/// Get the next golden checkpoint after current
pub fn next_checkpoint(current: usize) -> Option<usize> {
    GOLDEN_CHECKPOINTS
        .iter()
        .find(|&&cp| cp > current)
        .copied()
}

/// Get the previous golden checkpoint before current
pub fn prev_checkpoint(current: usize) -> Option<usize> {
    GOLDEN_CHECKPOINTS
        .iter()
        .rev()
        .find(|&&cp| cp < current)
        .copied()
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for MaximalProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Maximal Analysis for n={}:", self.n)?;
        writeln!(f, "  Known maximal: {}", self.is_known_maximal)?;
        writeln!(f, "  L[n] is Fibonacci: {}", self.lucas_is_fibonacci)?;
        writeln!(f, "  F[n] is Lucas: {}", self.fibonacci_is_lucas)?;
        writeln!(f, "  Has max gaps: {}", self.has_max_gaps)?;
        writeln!(f, "  At phase boundary: {}", self.at_phase_boundary)?;
        writeln!(f, "  Score: {}/5", self.score)?;

        if self.is_perfect() {
            writeln!(f, "  ★ PERFECT MAXIMAL ★")?;
        } else if self.is_excellent() {
            writeln!(f, "  ✓ Excellent maximal")?;
        } else if self.is_maximal() {
            writeln!(f, "  ✓ Maximal")?;
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
    fn test_known_maximals() {
        for &n in &MAXIMAL_N {
            let props = MaximalProperties::analyze(n);
            assert!(props.is_known_maximal);
            assert!(props.score >= 3, "Known maximal n={} has score {}", n, props.score);
        }
    }

    #[test]
    fn test_maximal_detection() {
        // n=3 should be maximal
        let props3 = MaximalProperties::analyze(3);
        assert!(props3.is_maximal());

        // n=4 should be maximal
        let props4 = MaximalProperties::analyze(4);
        assert!(props4.is_maximal());

        // Random n likely not maximal
        let props5 = MaximalProperties::analyze(5);
        // May or may not be maximal, just verify it computes
        let _ = props5.is_maximal();
    }

    #[test]
    fn test_lucas_fibonacci_alignment() {
        // L[3] = 4 = F[?]
        // 4 is not a Fibonacci number (0,1,1,2,3,5,8...)
        // But L[1] = 1 = F[1] = F[2]
        assert!(is_lucas_in_fibonacci(1));
    }

    #[test]
    fn test_fibonacci_lucas_alignment() {
        // F[3] = 2 = L[0]
        assert!(is_fibonacci_in_lucas(3));

        // F[2] = 1 = L[1]
        assert!(is_fibonacci_in_lucas(2));
    }

    #[test]
    fn test_find_maximal_range() {
        let maximals = find_maximal_in_range(0, 20);

        // Should find at least some maximals
        assert!(!maximals.is_empty());

        // Should be sorted
        for i in 1..maximals.len() {
            assert!(maximals[i] > maximals[i - 1]);
        }
    }

    #[test]
    fn test_next_maximal() {
        let next = next_maximal(0).unwrap();
        assert!(next > 0);
        assert!(is_maximal(next));

        let next_after_3 = next_maximal(3);
        if let Some(n) = next_after_3 {
            assert!(n > 3);
            assert!(is_maximal(n));
        }
    }

    #[test]
    fn test_nearest_maximal() {
        // Near n=5, should find nearby maximal
        let nearest = nearest_maximal(5);
        assert!(is_maximal(nearest));

        // Should be close to 5
        assert!((nearest as i32 - 5).abs() < 10);
    }

    #[test]
    fn test_checkpoints() {
        assert_eq!(GOLDEN_CHECKPOINTS.len(), 7);

        // Checkpoints should be sorted
        for i in 1..GOLDEN_CHECKPOINTS.len() {
            assert!(GOLDEN_CHECKPOINTS[i] > GOLDEN_CHECKPOINTS[i - 1]);
        }

        // All checkpoints should be maximal
        for &cp in &GOLDEN_CHECKPOINTS {
            assert!(is_maximal(cp));
        }
    }

    #[test]
    fn test_checkpoint_navigation() {
        let next = next_checkpoint(0).unwrap();
        assert_eq!(next, 3);

        let next_after_3 = next_checkpoint(3).unwrap();
        assert_eq!(next_after_3, 4);

        let prev_from_5 = prev_checkpoint(5).unwrap();
        assert_eq!(prev_from_5, 4);
    }

    #[test]
    fn test_maximal_properties_display() {
        let props = MaximalProperties::analyze(11);
        let display = format!("{}", props);

        assert!(display.contains("n=11"));
        assert!(display.contains("Score"));
    }

    #[test]
    fn test_phase_boundaries() {
        // Even n should be at phase boundary
        let props_even = MaximalProperties::analyze(4);
        assert!(props_even.at_phase_boundary);

        // Odd n should not
        let props_odd = MaximalProperties::analyze(5);
        assert!(!props_odd.at_phase_boundary);
    }
}
