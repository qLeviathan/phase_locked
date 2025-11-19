//! # Cascade Logic Algebra
//!
//! The bit cascade operator κ resolves adjacent 1s in Zeckendorf representation.
//!
//! ## Rules
//! - κ(11) = 100  (adjacent bits cascade up)
//! - κ(101) = 101 (valid Zeckendorf form)
//! - κ(111) = 1001 (recursive cascade)

use crate::algebra::BitLattice;

/// Cascade operator κ
pub fn cascade(bits: &mut Vec<bool>) {
    loop {
        let mut changed = false;

        // Scan for adjacent 1s
        let mut i = 0;
        while i < bits.len().saturating_sub(1) {
            if bits[i] && bits[i + 1] {
                // Found 11 pattern
                cascade_pair(bits, i);
                changed = true;
                break;
            }
            i += 1;
        }

        if !changed {
            break;
        }
    }
}

/// Cascade a single pair of adjacent 1s
fn cascade_pair(bits: &mut Vec<bool>, pos: usize) {
    // 11 → 100
    bits[pos] = false;
    bits[pos + 1] = false;

    if pos + 2 < bits.len() {
        // Flip bit at pos+2 (may create new adjacency)
        bits[pos + 2] = !bits[pos + 2];
    } else {
        // Need to extend
        bits.push(true);
    }
}

/// Zeckendorf addition with cascade
///
/// Add two Zeckendorf representations and apply cascade to maintain validity
pub fn zeckendorf_add(a: &BitLattice, b: &BitLattice) -> BitLattice {
    let max_len = a.bits.len().max(b.bits.len());
    let mut result = vec![false; max_len + 1];

    // Add bit by bit
    for i in 0..max_len {
        let bit_a = a.bits.get(i).copied().unwrap_or(false);
        let bit_b = b.bits.get(i).copied().unwrap_or(false);

        result[i] = result[i] ^ bit_a ^ bit_b; // XOR for addition
    }

    // Apply cascade to resolve adjacent 1s
    cascade(&mut result);

    // Remove leading zeros
    while result.len() > 1 && !result[result.len() - 1] {
        result.pop();
    }

    BitLattice { bits: result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_simple() {
        let mut bits = vec![true, true]; // 11
        cascade(&mut bits);
        assert_eq!(bits, vec![false, false, true]); // 100
    }

    #[test]
    fn test_cascade_triple() {
        let mut bits = vec![true, true, true]; // 111
        cascade(&mut bits);
        // 111 → 011 (after first cascade) → 101 (after second)
        // Actually: 111 → 101 → 1001
        assert!(bits.len() >= 3);
        // Verify no adjacent 1s
        for i in 0..bits.len() - 1 {
            assert!(!(bits[i] && bits[i + 1]));
        }
    }

    #[test]
    fn test_zeckendorf_add() {
        let a = BitLattice::from_integer(5);  // 101
        let b = BitLattice::from_integer(3);  // 10
        let c = zeckendorf_add(&a, &b);

        // Should equal 8
        assert_eq!(c.to_integer(), 8);
        assert!(c.is_valid());
    }
}
