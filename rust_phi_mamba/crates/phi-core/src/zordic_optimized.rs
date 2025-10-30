//! Optimized ZORDIC operations using bit manipulation
//! Based on advanced C implementation insights

use std::time::Instant;

/// Optimized CASCADE using direct bit manipulation
/// Claims 1M+ operations per second
#[inline]
pub fn cascade_bits(mut bits: u64) -> u64 {
    loop {
        // Find adjacent 1s: bits & (bits << 1)
        let adjacent = bits & (bits << 1);
        
        // No violations? We're done
        if adjacent == 0 {
            break;
        }
        
        // Find position of lowest violation
        let pos = adjacent.trailing_zeros() as u32;
        
        // Clear bits at positions k and k+1
        bits &= !(3u64 << pos);
        
        // Set bit at position k+2
        if pos + 2 < 64 {
            bits |= 1u64 << (pos + 2);
        }
    }
    
    bits
}

/// Convert index set to bit representation
#[inline]
pub fn indices_to_bits(indices: &[u8]) -> u64 {
    let mut bits = 0u64;
    for &idx in indices {
        if idx < 64 {
            bits |= 1u64 << idx;
        }
    }
    bits
}

/// Convert bit representation to index set
#[inline]
pub fn bits_to_indices(mut bits: u64) -> Vec<u8> {
    let mut indices = Vec::with_capacity(bits.count_ones() as usize);
    while bits != 0 {
        let idx = bits.trailing_zeros() as u8;
        indices.push(idx);
        bits &= bits - 1; // Clear lowest bit
    }
    indices
}

/// Optimized Zeckendorf encoding using bit operations
pub fn encode_bits(mut n: u64) -> u64 {
    // Precomputed Fibonacci numbers up to F_92
    const FIB: [u64; 94] = [
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
        2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418,
        317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465,
        14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296,
        433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976,
        7778742049, 12586269025, 20365011074, 32951280099, 53316291173,
        86267571272, 139583862445, 225851433717, 365435296162, 591286729879,
        956722026041, 1548008755920, 2504730781961, 4052739537881, 6557470319842,
        10610209857723, 17167680177565, 27777890035288, 44945570212853,
        72723460248141, 117669030460994, 190392490709135, 308061521170129,
        498454011879264, 806515533049393, 1304969544928657, 2111485077978050,
        3416454622906707, 5527939700884757, 8944394323791464, 14472334024676221,
        23416728348467685, 37889062373143906, 61305790721611591, 99194853094755497,
        160500643816367088, 259695496911122585, 420196140727489673,
        679891637638612258, 1100087778366101931, 1779979416004714189,
        2880067194370816120, 4660046610375530309, 7540113804746346429,
        12200160415121876738
    ];
    
    if n == 0 {
        return 0;
    }
    
    let mut bits = 0u64;
    
    // Greedy algorithm from largest Fibonacci
    for k in (0..64).rev() {
        if k + 2 < 93 && FIB[k + 2] <= n && FIB[k + 2] > 0 {
            bits |= 1u64 << k;
            n -= FIB[k + 2];
            if n == 0 {
                break;
            }
        }
    }
    
    bits
}

/// Hamming distance between two bit patterns
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Fibonacci-weighted Hamming distance
pub fn weighted_distance(a: u64, b: u64) -> u64 {
    const FIB: [u64; 66] = [
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
        987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
        121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578,
        5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155,
        165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903,
        2971215073, 4807526976, 7778742049, 12586269025, 20365011074,
        32951280099, 53316291173, 86267571272, 139583862445, 225851433717,
        365435296162, 591286729879, 956722026041, 1548008755920, 2504730781961,
        4052739537881, 6557470319842, 10610209857723, 17167680177565
    ];
    
    let mut diff = a ^ b;
    let mut distance = 0u64;
    
    while diff != 0 {
        let pos = diff.trailing_zeros() as usize;
        if pos + 2 < FIB.len() {
            distance += FIB[pos + 2];
        }
        diff &= diff - 1; // Clear lowest bit
    }
    
    distance
}

/// Benchmark CASCADE operations
pub fn benchmark_cascade(iterations: usize) {
    println!("Benchmarking CASCADE operations...");
    
    // Test patterns with violations
    let test_patterns = vec![
        0b111u64,      // Three adjacent 1s
        0b11011u64,    // Two pairs of adjacent
        0b1111111u64,  // Seven adjacent 1s
        0b101010101u64, // No violations
    ];
    
    let start = Instant::now();
    let mut result = 0u64;
    
    for _ in 0..iterations {
        for &pattern in &test_patterns {
            result ^= cascade_bits(pattern);
        }
    }
    
    let elapsed = start.elapsed();
    let ops_per_sec = (iterations * test_patterns.len()) as f64 / elapsed.as_secs_f64();
    
    println!("Iterations: {}", iterations);
    println!("Time: {:?}", elapsed);
    println!("Operations per second: {:.0}", ops_per_sec);
    println!("Final result: {:064b}", result); // Prevent optimization
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cascade_bits() {
        // Test basic cascade
        assert_eq!(cascade_bits(0b11), 0b100); // 11 → 100
        assert_eq!(cascade_bits(0b111), 0b1000); // 111 → 1000
        assert_eq!(cascade_bits(0b1011), 0b10000); // 1011 → 10000
        
        // Test no violations
        assert_eq!(cascade_bits(0b101), 0b101); // No change
        assert_eq!(cascade_bits(0b10001), 0b10001); // No change
    }
    
    #[test]
    fn test_bit_conversions() {
        let indices = vec![0, 2, 5, 8];
        let bits = indices_to_bits(&indices);
        assert_eq!(bits, 0b100100101);
        
        let recovered = bits_to_indices(bits);
        assert_eq!(recovered, indices);
    }
    
    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0b1010, 0b1100), 2);
        assert_eq!(hamming_distance(0b1111, 0b0000), 4);
        assert_eq!(hamming_distance(0b1010, 0b1010), 0);
    }
    
    #[test]
    fn benchmark_performance() {
        benchmark_cascade(100_000);
    }
}