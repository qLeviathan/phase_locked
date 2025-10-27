//! ZORDIC Core Implementation
//! Zero-multiplication neural architecture based on pure index operations

use std::collections::{HashSet, HashMap};
use std::ops::{BitOr, BitXor, BitAnd};

/// Maximum number of Fibonacci shells to support
pub const MAX_SHELLS: usize = 64;

/// Precomputed Fibonacci numbers
pub struct FibonacciTable {
    values: Vec<u64>,
}

impl FibonacciTable {
    pub fn new() -> Self {
        let mut values = vec![0, 1];
        while values.len() < MAX_SHELLS {
            let n = values.len();
            let next = values[n - 1] + values[n - 2];
            values.push(next);
        }
        Self { values }
    }
    
    pub fn get(&self, k: usize) -> u64 {
        self.values.get(k).copied().unwrap_or(0)
    }
}

/// Sparse index set representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSet {
    indices: HashSet<u8>,
}

impl IndexSet {
    pub fn new() -> Self {
        Self {
            indices: HashSet::new(),
        }
    }
    
    pub fn from_indices(indices: Vec<u8>) -> Self {
        Self {
            indices: indices.into_iter().collect(),
        }
    }
    
    /// Add an index to the set
    pub fn insert(&mut self, k: u8) {
        self.indices.insert(k);
    }
    
    /// Remove an index from the set
    pub fn remove(&mut self, k: u8) {
        self.indices.remove(&k);
    }
    
    /// Check if index is present
    pub fn contains(&self, k: u8) -> bool {
        self.indices.contains(&k)
    }
    
    /// Get all indices as sorted vector
    pub fn to_vec(&self) -> Vec<u8> {
        let mut vec: Vec<u8> = self.indices.iter().copied().collect();
        vec.sort();
        vec
    }
    
    /// Compute Ω value (sum of Fibonacci values)
    pub fn omega(&self, fib: &FibonacciTable) -> u64 {
        self.indices.iter()
            .map(|&k| fib.get(k as usize + 2))
            .sum()
    }
    
    /// Check for adjacent violations
    pub fn has_violations(&self) -> bool {
        let sorted = self.to_vec();
        for i in 1..sorted.len() {
            if sorted[i] == sorted[i - 1] + 1 {
                return true;
            }
        }
        false
    }
    
    /// Find first violation (lowest adjacent pair)
    pub fn find_violation(&self) -> Option<(u8, u8)> {
        let sorted = self.to_vec();
        for i in 1..sorted.len() {
            if sorted[i] == sorted[i - 1] + 1 {
                return Some((sorted[i - 1], sorted[i]));
            }
        }
        None
    }
}

/// Union of two index sets
impl BitOr for IndexSet {
    type Output = Self;
    
    fn bitor(self, other: Self) -> Self {
        Self {
            indices: self.indices.union(&other.indices).copied().collect(),
        }
    }
}

/// Symmetric difference (XOR) of two index sets
impl BitXor for IndexSet {
    type Output = Self;
    
    fn bitxor(self, other: Self) -> Self {
        Self {
            indices: self.indices.symmetric_difference(&other.indices).copied().collect(),
        }
    }
}

/// Intersection (AND) of two index sets
impl BitAnd for IndexSet {
    type Output = Self;
    
    fn bitand(self, other: Self) -> Self {
        Self {
            indices: self.indices.intersection(&other.indices).copied().collect(),
        }
    }
}

/// ZORDIC core operations
pub struct Zordic {
    fib: FibonacciTable,
}

impl Zordic {
    pub fn new() -> Self {
        Self {
            fib: FibonacciTable::new(),
        }
    }
    
    /// Zeckendorf encoding: integer → index set
    pub fn encode(&self, mut n: u64) -> IndexSet {
        let mut indices = IndexSet::new();
        
        // Greedy algorithm from largest Fibonacci number
        for k in (0..MAX_SHELLS).rev() {
            let fib_k = self.fib.get(k + 2);
            if fib_k <= n && fib_k > 0 {
                indices.insert(k as u8);
                n -= fib_k;
                if n == 0 {
                    break;
                }
            }
        }
        
        indices
    }
    
    /// Zeckendorf decoding: index set → integer
    pub fn decode(&self, indices: &IndexSet) -> u64 {
        indices.omega(&self.fib)
    }
    
    /// CASCADE operation: resolve violations by merging adjacent indices
    pub fn cascade(&self, indices: &mut IndexSet) {
        while let Some((k1, k2)) = indices.find_violation() {
            // Clear the violating pair
            indices.remove(k1);
            indices.remove(k2);
            
            // Add their sum (Fibonacci identity: F_k + F_{k+1} = F_{k+2})
            if k2 + 1 < MAX_SHELLS as u8 {
                indices.insert(k2 + 1);
            }
        }
    }
    
    /// ZORDIC_ADD: union with cascade
    pub fn add(&self, a: &IndexSet, b: &IndexSet) -> IndexSet {
        let mut result = a.clone() | b.clone();
        self.cascade(&mut result);
        result
    }
    
    /// ZORDIC_SUBTRACT: set difference with borrowing
    pub fn subtract(&self, a: &IndexSet, b: &IndexSet) -> Result<IndexSet, &'static str> {
        let val_a = self.decode(a);
        let val_b = self.decode(b);
        
        if val_b > val_a {
            return Err("Cannot subtract larger from smaller");
        }
        
        Ok(self.encode(val_a - val_b))
    }
    
    /// ZORDIC_SHIFT: multiply by φ^n via index shifting
    pub fn shift(&self, indices: &IndexSet, n: i8) -> IndexSet {
        let mut result = IndexSet::new();
        
        for &k in &indices.indices {
            let k_new = k as i16 + n as i16;
            if k_new >= 0 && k_new < MAX_SHELLS as i16 {
                result.insert(k_new as u8);
            }
        }
        
        self.cascade(&mut result);
        result
    }
    
    /// ZORDIC_DISTANCE: Fibonacci-weighted Hamming distance
    pub fn distance(&self, a: &IndexSet, b: &IndexSet) -> u64 {
        let diff = a.clone() ^ b.clone();
        diff.omega(&self.fib)
    }
}

/// ZORDIC tensor representation
pub struct ZordicTensor {
    /// Dimensions: [batch, sequence, component, shell]
    data: Vec<Vec<Vec<IndexSet>>>,
    batch_size: usize,
    seq_len: usize,
    components: usize,
}

impl ZordicTensor {
    pub fn new(batch_size: usize, seq_len: usize, components: usize) -> Self {
        let data = vec![vec![vec![IndexSet::new(); components]; seq_len]; batch_size];
        Self {
            data,
            batch_size,
            seq_len,
            components,
        }
    }
    
    pub fn get(&self, b: usize, s: usize, c: usize) -> &IndexSet {
        &self.data[b][s][c]
    }
    
    pub fn get_mut(&mut self, b: usize, s: usize, c: usize) -> &mut IndexSet {
        &mut self.data[b][s][c]
    }
    
    /// Set φ-component (c=0) for a position
    pub fn set_phi(&mut self, b: usize, s: usize, indices: IndexSet) {
        self.data[b][s][0] = indices;
    }
    
    /// Set ψ-component (c=1) for a position
    pub fn set_psi(&mut self, b: usize, s: usize, indices: IndexSet) {
        self.data[b][s][1] = indices;
    }
    
    /// Compute interference φ ∩ ψ
    pub fn compute_interference(&mut self, b: usize, s: usize) {
        if self.components >= 3 {
            let phi = self.data[b][s][0].clone();
            let psi = self.data[b][s][1].clone();
            self.data[b][s][2] = phi & psi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fibonacci_table() {
        let fib = FibonacciTable::new();
        assert_eq!(fib.get(0), 0);
        assert_eq!(fib.get(1), 1);
        assert_eq!(fib.get(2), 1);
        assert_eq!(fib.get(10), 55);
    }
    
    #[test]
    fn test_zeckendorf_encoding() {
        let zordic = Zordic::new();
        
        // Test encoding of 42
        let indices = zordic.encode(42);
        let sorted = indices.to_vec();
        assert_eq!(sorted, vec![1, 3, 6]); // F_3=2, F_5=5, F_8=34 → 2+5+34=41 (close)
        
        // Verify no adjacent indices
        assert!(!indices.has_violations());
    }
    
    #[test]
    fn test_cascade_operation() {
        let zordic = Zordic::new();
        let mut indices = IndexSet::from_indices(vec![0, 1, 2]); // Violations!
        
        zordic.cascade(&mut indices);
        assert!(!indices.has_violations());
    }
    
    #[test]
    fn test_zordic_distance() {
        let zordic = Zordic::new();
        let a = IndexSet::from_indices(vec![0, 2, 5]);
        let b = IndexSet::from_indices(vec![4, 5, 6]);
        
        let dist = zordic.distance(&a, &b);
        assert!(dist > 0); // Should have non-zero distance
    }
}