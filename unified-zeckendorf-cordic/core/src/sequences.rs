//! # Integer Sequences: Fibonacci and Lucas
//!
//! Core sequences for the Zeckendorf-CORDIC system.

/// Fibonacci number F_n (OEIS A000045)
///
/// F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
pub fn fibonacci(n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let mut f_prev = 0u64;
    let mut f_curr = 1u64;

    for _ in 2..=n {
        let f_next = f_prev.saturating_add(f_curr);
        f_prev = f_curr;
        f_curr = f_next;
    }

    f_curr
}

/// Lucas number L_n (OEIS A000032)
///
/// L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}
pub fn lucas(n: usize) -> u64 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }

    let mut l_prev = 2u64;
    let mut l_curr = 1u64;

    for _ in 2..=n {
        let l_next = l_prev.saturating_add(l_curr);
        l_prev = l_curr;
        l_curr = l_next;
    }

    l_curr
}

/// Zeckendorf decomposition (OEIS A003714)
///
/// Unique representation as sum of non-consecutive Fibonacci numbers
pub fn zeckendorf_decomposition(mut n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![0];
    }

    // Generate Fibonacci numbers up to n
    let mut fibs = vec![1, 2];
    loop {
        let len = fibs.len();
        let next = fibs[len - 1] + fibs[len - 2];
        if next > n {
            break;
        }
        fibs.push(next);
    }

    let mut result = Vec::new();

    // Greedy algorithm (proven optimal)
    for &fib in fibs.iter().rev() {
        if fib <= n {
            result.push(fib);
            n -= fib;
        }
    }

    result.reverse();
    result
}

/// Phi approximation using Fibonacci ratios
///
/// φ ≈ F_{n+1} / F_n as n → ∞
pub fn phi_approximation(n: usize) -> (u64, u64) {
    let f_n = fibonacci(n);
    let f_n_plus_1 = fibonacci(n + 1);
    (f_n_plus_1, f_n)
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
        assert_eq!(fibonacci(5), 5);
        assert_eq!(fibonacci(10), 55);
    }

    #[test]
    fn test_lucas() {
        assert_eq!(lucas(0), 2);
        assert_eq!(lucas(1), 1);
        assert_eq!(lucas(2), 3);
        assert_eq!(lucas(5), 11);
    }

    #[test]
    fn test_zeckendorf() {
        let zeck = zeckendorf_decomposition(17);
        assert_eq!(zeck, vec![1, 3, 13]);
        assert_eq!(zeck.iter().sum::<u64>(), 17);
    }
}
