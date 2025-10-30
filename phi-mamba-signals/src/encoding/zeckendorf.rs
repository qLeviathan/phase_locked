//! Zeckendorf Decomposition (OEIS A003714)
//!
//! Every positive integer has a UNIQUE representation as a sum of
//! non-consecutive Fibonacci numbers.
//!
//! ## Theorem (Zeckendorf, 1972)
//! ∀n ∈ ℕ⁺, ∃! decomposition n = Σ F_kᵢ where kᵢ₊₁ ≥ kᵢ + 2
//!
//! ## Example
//! 17 = 13 + 3 + 1 = F_7 + F_4 + F_2
//! Binary: 10100 (gaps = topological holes = memory)
//!
//! ## OEIS References
//! - A003714: Zeckendorf representation
//! - A000045: Fibonacci numbers
//! - A000032: Lucas numbers

use serde::{Deserialize, Serialize};

/// Fibonacci number (OEIS A000045)
///
/// F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
pub fn fibonacci(n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut fib_prev = 0u64;
    let mut fib_curr = 1u64;

    for _ in 2..=n {
        let fib_next = fib_prev.saturating_add(fib_curr);
        fib_prev = fib_curr;
        fib_curr = fib_next;
    }

    fib_curr
}

/// Lucas number (OEIS A000032)
///
/// L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}
pub fn lucas(n: usize) -> u64 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }

    let mut lucas_prev = 2u64;
    let mut lucas_curr = 1u64;

    for _ in 2..=n {
        let lucas_next = lucas_prev.saturating_add(lucas_curr);
        lucas_prev = lucas_curr;
        lucas_curr = lucas_next;
    }

    lucas_curr
}

/// Generate Fibonacci sequence up to max_value
pub fn fibonacci_sequence(max_value: u64) -> Vec<u64> {
    if max_value == 0 {
        return vec![0];
    }
    if max_value == 1 {
        return vec![0, 1];
    }

    let mut fibs = vec![1, 2];

    loop {
        let len = fibs.len();
        let next = fibs[len - 1].saturating_add(fibs[len - 2]);

        if next > max_value {
            break;
        }

        fibs.push(next);
    }

    fibs
}

/// Zeckendorf decomposition (OEIS A003714)
///
/// Greedy algorithm: Start from largest Fibonacci ≤ n,
/// subtract it, repeat. Proven to give unique non-consecutive representation.
///
/// ## Example
/// ```
/// use phi_mamba_signals::encoding::zeckendorf_decomposition;
///
/// let zeck = zeckendorf_decomposition(17);
/// assert_eq!(zeck, vec![1, 3, 13]); // 17 = 13 + 3 + 1
/// ```
pub fn zeckendorf_decomposition(mut n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![0];
    }

    // Generate Fibonacci numbers up to n
    let fibs = fibonacci_sequence(n);

    let mut result = Vec::new();

    // Greedy algorithm (proven optimal)
    for &fib in fibs.iter().rev() {
        if fib <= n {
            result.push(fib);
            n -= fib; // Subtraction only!
        }
    }

    result.reverse();
    result
}

/// Dual Zeckendorf representation
///
/// Forward: Fibonacci decomposition
/// Backward: Lucas decomposition
/// Intersection: Where both agree (critical information)
/// Difference: Where they disagree (context)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualZeckendorf {
    pub value: u64,
    pub fibonacci_terms: Vec<u64>,
    pub lucas_terms: Vec<u64>,
    pub intersection: Vec<u64>,
    pub difference: Vec<u64>,
}

impl DualZeckendorf {
    /// Create dual representation
    pub fn new(value: u64) -> Self {
        let fibonacci_terms = zeckendorf_decomposition(value);

        // Lucas decomposition (similar algorithm)
        let lucas_terms = lucas_decomposition(value);

        // Find intersection and difference
        let intersection: Vec<u64> = fibonacci_terms
            .iter()
            .filter(|&f| lucas_terms.contains(f))
            .copied()
            .collect();

        let difference: Vec<u64> = fibonacci_terms
            .iter()
            .chain(lucas_terms.iter())
            .filter(|&f| !intersection.contains(f))
            .copied()
            .collect();

        Self {
            value,
            fibonacci_terms,
            lucas_terms,
            intersection,
            difference,
        }
    }

    /// Convert to binary string (1 = Fibonacci present, 0 = gap)
    pub fn to_binary_string(&self) -> String {
        if self.value == 0 {
            return "0".to_string();
        }

        // Find maximum Fibonacci index
        let max_fib = *self.fibonacci_terms.last().unwrap_or(&1);
        let mut max_index = 0;

        for i in 0..64 {
            if fibonacci(i) >= max_fib {
                max_index = i;
                break;
            }
        }

        // Build binary string
        let mut binary = String::new();

        for i in (2..=max_index).rev() {
            let fib = fibonacci(i);
            if self.fibonacci_terms.contains(&fib) {
                binary.push('1');
            } else {
                binary.push('0');
            }
        }

        if binary.is_empty() {
            "1".to_string()
        } else {
            binary
        }
    }

    /// Get "holes" (positions with 0s in binary representation)
    pub fn holes(&self) -> Vec<usize> {
        let binary = self.to_binary_string();

        binary
            .chars()
            .enumerate()
            .filter_map(|(i, c)| if c == '0' { Some(i) } else { None })
            .collect()
    }
}

/// Lucas decomposition (similar to Zeckendorf but with Lucas numbers)
fn lucas_decomposition(mut n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![0];
    }

    // Generate Lucas numbers up to n
    let mut lucas_nums = vec![2, 1];

    loop {
        let len = lucas_nums.len();
        let next = lucas_nums[len - 1] + lucas_nums[len - 2];

        if next > n {
            break;
        }

        lucas_nums.push(next);
    }

    let mut result = Vec::new();

    // Greedy algorithm
    for &luc in lucas_nums.iter().rev() {
        if luc <= n {
            result.push(luc);
            n -= luc;
        }
    }

    result.reverse();
    result
}

/// Encode integer to Zeckendorf binary (for bit lattice operations)
pub fn to_zeckendorf_binary(n: u64) -> Vec<bool> {
    let dual = DualZeckendorf::new(n);
    let binary_str = dual.to_binary_string();

    binary_str.chars().map(|c| c == '1').collect()
}

/// Decode from Zeckendorf binary
pub fn from_zeckendorf_binary(bits: &[bool]) -> u64 {
    let mut sum = 0u64;

    for (i, &bit) in bits.iter().rev().enumerate() {
        if bit {
            sum += fibonacci(i + 2); // Start from F_2 = 1
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(3), 2);
        assert_eq!(fibonacci(4), 3);
        assert_eq!(fibonacci(5), 5);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    #[test]
    fn test_lucas() {
        assert_eq!(lucas(0), 2);
        assert_eq!(lucas(1), 1);
        assert_eq!(lucas(2), 3);
        assert_eq!(lucas(3), 4);
        assert_eq!(lucas(4), 7);
        assert_eq!(lucas(5), 11);
        assert_eq!(lucas(10), 123);
    }

    #[test]
    fn test_zeckendorf_decomposition() {
        // 17 = 13 + 3 + 1
        let zeck = zeckendorf_decomposition(17);
        assert_eq!(zeck, vec![1, 3, 13]);

        // 100 = 89 + 8 + 3
        let zeck = zeckendorf_decomposition(100);
        assert_eq!(zeck, vec![3, 8, 89]);

        // Verify sum
        let sum: u64 = zeck.iter().sum();
        assert_eq!(sum, 100);
    }

    #[test]
    fn test_zeckendorf_uniqueness() {
        // Test that decomposition is unique and non-consecutive
        for n in 1..=100 {
            let zeck = zeckendorf_decomposition(n);

            // Verify sum equals n
            let sum: u64 = zeck.iter().sum();
            assert_eq!(sum, n);

            // Verify non-consecutive (no adjacent Fibonacci numbers)
            for i in 0..zeck.len().saturating_sub(1) {
                let curr = zeck[i];
                let next = zeck[i + 1];

                // Find indices
                let mut curr_idx = 0;
                let mut next_idx = 0;

                for j in 0..50 {
                    if fibonacci(j) == curr {
                        curr_idx = j;
                    }
                    if fibonacci(j) == next {
                        next_idx = j;
                    }
                }

                // Indices should differ by at least 2
                assert!(next_idx >= curr_idx + 2);
            }
        }
    }

    #[test]
    fn test_dual_zeckendorf() {
        let dual = DualZeckendorf::new(17);

        assert_eq!(dual.value, 17);
        assert_eq!(dual.fibonacci_terms, vec![1, 3, 13]);

        // Binary representation
        let binary = dual.to_binary_string();
        assert!(binary.contains('1'));
        assert!(binary.contains('0'));
    }

    #[test]
    fn test_to_binary_string() {
        let dual = DualZeckendorf::new(17);
        let binary = dual.to_binary_string();

        // 17 = F_7 + F_4 + F_2 = 13 + 3 + 1
        // Binary from F_7 down: 10100
        assert_eq!(binary, "10100");
    }

    #[test]
    fn test_holes() {
        let dual = DualZeckendorf::new(17);
        let holes = dual.holes();

        // Binary 10100 has holes at positions 1 and 3
        assert_eq!(holes, vec![1, 3]);
    }

    #[test]
    fn test_zeckendorf_binary_roundtrip() {
        for n in 1..=50 {
            let bits = to_zeckendorf_binary(n);
            let decoded = from_zeckendorf_binary(&bits);
            assert_eq!(decoded, n);
        }
    }
}
