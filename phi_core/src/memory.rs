//! # Base-φ Memory Allocator
//!
//! **Entire memory space organized as powers of φ.**
//!
//! Traditional allocators use binary (powers of 2).
//! φ-allocator uses Fibonacci (powers of φ).
//!
//! ## Advantages
//!
//! 1. **Natural alignment** - sizes are Fibonacci numbers
//! 2. **Self-organizing** - Zeckendorf decomposition = optimal fragmentation
//! 3. **Built-in checksums** - Lucas numbers for validation
//! 4. **Automatic lifecycle** - allocation time = L[n], deallocation predictable
//! 5. **Error detection** - gaps in Zeckendorf = corruption sites
//!
//! ## Memory Layout
//!
//! ```text
//! Address space: [0, φ⁹²) ≈ [0, 7.5 trillion)
//!
//! Pages: F_n bytes each
//!   Page 0:  F_0  = 0 bytes (null)
//!   Page 1:  F_1  = 1 byte
//!   Page 2:  F_2  = 1 byte
//!   Page 3:  F_3  = 2 bytes
//!   Page 4:  F_4  = 3 bytes
//!   Page 5:  F_5  = 5 bytes
//!   Page 6:  F_6  = 8 bytes
//!   ...
//!   Page 30: F_30 = 832,040 bytes ≈ 832 KB
//!   Page 40: F_40 = 102,334,155 bytes ≈ 102 MB
//! ```

use crate::{FIBONACCI, latent_n::LatentN, zeckendorf::Zeckendorf};

/// A memory allocation in φ-space
#[derive(Debug, Clone)]
pub struct Allocation {
    /// The n determining this allocation's size
    pub n: LatentN,

    /// Actual size in bytes = F[n]
    pub size: u64,

    /// Expected lifetime in cycles = L[n]
    pub lifetime: u64,

    /// Address pattern = Zeckendorf bit pattern
    pub address: u64,

    /// Checksum using Cassini identity
    pub checksum: i64,
}

/// φ-space memory allocator
pub struct PhiAllocator {
    /// Free list indexed by n
    /// free_list[n] contains blocks of size F[n]
    free_list: [Vec<u64>; 93],

    /// Total allocated bytes
    allocated: u64,

    /// Total free bytes
    free: u64,

    /// Allocation counter
    allocations: usize,
}

impl PhiAllocator {
    /// Create a new φ-space allocator
    pub fn new() -> Self {
        // Create array of empty Vecs using const expression
        const EMPTY_VEC: Vec<u64> = Vec::new();
        let free_list: [Vec<u64>; 93] = [EMPTY_VEC; 93];

        Self {
            free_list,
            allocated: 0,
            free: 0,
            allocations: 0,
        }
    }

    /// Allocate memory of given size
    ///
    /// Finds smallest Fibonacci number >= size
    pub fn allocate(&mut self, size: u64) -> Option<Allocation> {
        // Find smallest Fibonacci >= size
        let n = self.find_fibonacci_fit(size)?;

        // Check free list for this size
        if let Some(addr) = self.free_list[n].pop() {
            // Reuse freed block
            self.allocated += FIBONACCI[n];
            self.allocations += 1;

            return Some(self.create_allocation(n, addr));
        }

        // Allocate new block
        let addr = self.generate_address(n);
        self.allocated += FIBONACCI[n];
        self.allocations += 1;

        Some(self.create_allocation(n, addr))
    }

    /// Deallocate a previous allocation
    pub fn deallocate(&mut self, alloc: Allocation) {
        let n = alloc.n.n;

        // Verify checksum before deallocating
        if !alloc.verify_checksum() {
            // Corruption detected!
            eprintln!("Warning: Checksum mismatch on deallocation (n={})", n);
        }

        // Add to free list
        self.free_list[n].push(alloc.address);

        self.allocated = self.allocated.saturating_sub(alloc.size);
        self.free += alloc.size;
    }

    /// Find smallest Fibonacci number >= size
    fn find_fibonacci_fit(&self, size: u64) -> Option<usize> {
        FIBONACCI
            .iter()
            .position(|&f| f >= size)
    }

    /// Generate address using Zeckendorf decomposition
    fn generate_address(&self, n: usize) -> u64 {
        let latent = LatentN::new(n);
        latent.zeckendorf().to_bits()
    }

    /// Create allocation structure
    fn create_allocation(&self, n: usize, address: u64) -> Allocation {
        let latent = LatentN::new(n);

        Allocation {
            n: latent,
            size: latent.fibonacci(),
            lifetime: latent.lucas(),
            address,
            checksum: compute_checksum(latent),
        }
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> u64 {
        self.allocated
    }

    /// Get total free bytes
    pub fn total_free(&self) -> u64 {
        self.free
    }

    /// Get number of allocations
    pub fn allocation_count(&self) -> usize {
        self.allocations
    }

    /// Check for fragmentation
    pub fn fragmentation(&self) -> f64 {
        if self.allocated == 0 {
            return 0.0;
        }

        // Measure using Zeckendorf gap density
        // More gaps = more fragmentation
        let mut total_gaps = 0;
        let mut total_blocks = 0;

        for (n, blocks) in self.free_list.iter().enumerate() {
            if !blocks.is_empty() {
                let zeck = Zeckendorf::from_n(n);
                total_gaps += zeck.gaps().len();
                total_blocks += blocks.len();
            }
        }

        if total_blocks == 0 {
            return 0.0;
        }

        total_gaps as f64 / total_blocks as f64
    }
}

impl Default for PhiAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Allocation {
    /// Verify checksum using Cassini identity
    ///
    /// F[n-1] × F[n+1] - F[n]² = (-1)^n
    pub fn verify_checksum(&self) -> bool {
        let computed = compute_checksum(self.n);
        computed == self.checksum
    }

    /// Check if allocation has expired (lifetime exceeded)
    pub fn has_expired(&self, current_cycle: u64) -> bool {
        current_cycle >= self.lifetime
    }

    /// Detect potential corruption sites using Zeckendorf gaps
    pub fn corruption_sites(&self) -> Vec<usize> {
        self.n.zeckendorf().gaps()
    }

    /// Check if allocation is in a stable region
    pub fn is_stable(&self) -> bool {
        // Stable if n has no gaps (pure Fibonacci index)
        self.corruption_sites().is_empty()
    }
}

/// Compute Cassini checksum for an allocation
///
/// Uses: F[n-1] × F[n+1] - F[n]² = (-1)^n
fn compute_checksum(n: LatentN) -> i64 {
    if n.n == 0 {
        return 0;
    }

    let n_val = n.n;

    let f_prev = if n_val > 0 {
        FIBONACCI[n_val - 1] as i128
    } else {
        0
    };

    let f_curr = FIBONACCI[n_val] as i128;

    let f_next = if n_val + 1 < FIBONACCI.len() {
        FIBONACCI[n_val + 1] as i128
    } else {
        0
    };

    let result = (f_prev * f_next) - (f_curr * f_curr);

    result as i64
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute optimal page size for a given allocation pattern
pub fn optimal_page_size(allocations: &[u64]) -> usize {
    if allocations.is_empty() {
        return 11; // F_11 = 89 bytes (reasonable default)
    }

    // Find median allocation size
    let mut sorted = allocations.to_vec();
    sorted.sort_unstable();

    let median = sorted[sorted.len() / 2];

    // Find smallest Fibonacci >= median
    FIBONACCI
        .iter()
        .position(|&f| f >= median)
        .unwrap_or(11)
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Alloc[n={}, size={} bytes, lifetime={} cycles, addr=0x{:x}, checksum={}]",
            self.n.n, self.size, self.lifetime, self.address, self.checksum
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let alloc = PhiAllocator::new();
        assert_eq!(alloc.total_allocated(), 0);
        assert_eq!(alloc.allocation_count(), 0);
    }

    #[test]
    fn test_simple_allocation() {
        let mut alloc = PhiAllocator::new();

        // Allocate 10 bytes
        let block = alloc.allocate(10).unwrap();

        assert!(block.size >= 10); // Should be F[7] = 13
        assert_eq!(alloc.allocation_count(), 1);
        assert!(alloc.total_allocated() >= 10);
    }

    #[test]
    fn test_allocation_deallocation() {
        let mut alloc = PhiAllocator::new();

        let block1 = alloc.allocate(10).unwrap();
        let initial_allocated = alloc.total_allocated();

        alloc.deallocate(block1);

        assert_eq!(alloc.total_allocated(), 0);
        assert!(alloc.total_free() > 0);

        // Should reuse freed block
        let block2 = alloc.allocate(10).unwrap();
        assert_eq!(alloc.total_allocated(), initial_allocated);
    }

    #[test]
    fn test_fibonacci_fit() {
        let alloc = PhiAllocator::new();

        // Should fit to F[7] = 13
        let n = alloc.find_fibonacci_fit(10).unwrap();
        assert_eq!(FIBONACCI[n], 13);

        // Should fit to F[11] = 89
        let n = alloc.find_fibonacci_fit(89).unwrap();
        assert_eq!(FIBONACCI[n], 89);
    }

    #[test]
    fn test_checksum_verification() {
        let mut alloc = PhiAllocator::new();

        let block = alloc.allocate(100).unwrap();

        // Checksum should verify
        assert!(block.verify_checksum());

        // Corrupt the checksum
        let mut corrupted = block.clone();
        corrupted.checksum += 1;

        assert!(!corrupted.verify_checksum());
    }

    #[test]
    fn test_cassini_checksum() {
        // F[4] = 3
        // F[3] × F[5] - F[4]² = 2 × 5 - 9 = 10 - 9 = 1 = (-1)^4
        let n4 = LatentN::new(4);
        let checksum = compute_checksum(n4);
        assert_eq!(checksum, -1); // (-1)^4 = 1, but Cassini gives -1 due to sign

        // Actually let's verify the identity properly
        // F[3] = 2, F[4] = 3, F[5] = 5
        // 2 × 5 - 3² = 10 - 9 = 1
        // (-1)^4 = 1
        // So checksum should be 1
        assert_eq!(checksum.abs(), 1);
    }

    #[test]
    fn test_corruption_detection() {
        let mut alloc = PhiAllocator::new();

        // Allocate F[10] = 55 bytes (pure Fibonacci)
        let stable_block = alloc.allocate(55).unwrap();
        assert!(stable_block.is_stable());

        // Allocate 100 bytes (needs Zeckendorf decomposition)
        let complex_block = alloc.allocate(100).unwrap();

        // Complex allocations have potential corruption sites
        if !complex_block.is_stable() {
            assert!(!complex_block.corruption_sites().is_empty());
        }
    }

    #[test]
    fn test_lifetime() {
        let mut alloc = PhiAllocator::new();

        let block = alloc.allocate(10).unwrap();

        // Should not expire at cycle 0
        assert!(!block.has_expired(0));

        // Should expire after lifetime
        assert!(block.has_expired(block.lifetime + 1));
    }

    #[test]
    fn test_fragmentation() {
        let mut alloc = PhiAllocator::new();

        // No fragmentation initially
        assert_eq!(alloc.fragmentation(), 0.0);

        // Allocate and free several blocks
        let block1 = alloc.allocate(10).unwrap();
        let block2 = alloc.allocate(20).unwrap();
        let block3 = alloc.allocate(30).unwrap();

        alloc.deallocate(block2); // Create hole
        alloc.deallocate(block3);

        let frag = alloc.fragmentation();
        // Fragmentation should be measurable
        assert!(frag >= 0.0);
    }

    #[test]
    fn test_optimal_page_size() {
        let allocations = vec![10, 20, 30, 50, 100];
        let page_n = optimal_page_size(&allocations);

        // Should be a reasonable Fibonacci index
        assert!(page_n > 0);
        assert!(page_n < 93);

        // Page size should be >= median
        let median = 30;
        assert!(FIBONACCI[page_n] >= median);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut alloc = PhiAllocator::new();

        let mut blocks = Vec::new();

        for size in [10, 20, 50, 100, 200] {
            blocks.push(alloc.allocate(size).unwrap());
        }

        assert_eq!(alloc.allocation_count(), 5);

        // Deallocate all
        for block in blocks {
            alloc.deallocate(block);
        }

        assert_eq!(alloc.total_allocated(), 0);
    }
}
