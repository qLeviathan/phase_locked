//! # Minimal Resistance Demo
//!
//! Demonstrates how φ-space computation provides minimal resistance
//! compared to traditional floating point operations.

use phi_core::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                                                              ║");
    println!("║          φ-CORE: MINIMAL RESISTANCE COMPUTATION              ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_n_encoding();
    demo_phi_arithmetic();
    demo_boundary_solving();
    demo_memory_allocation();
    demo_token_generation();
    demo_resistance_metrics();
}

fn demo_n_encoding() {
    println!("═══ 1. N-ENCODING THEOREM ═══\n");
    println!("A single integer n encodes an entire universe:\n");

    let n = latent_n::LatentN::new(10);
    let universe = n.decode();

    println!("  n = 10");
    println!("  ├─ Energy:    F[10] = {}", universe.energy);
    println!("  ├─ Time:      L[10] = {}", universe.time);
    println!("  ├─ Address:   0x{:x}", universe.address);
    println!("  ├─ Errors:    {} gaps", universe.error_sites.len());
    println!("  └─ Direction: {} (forward)\n", if universe.direction > 0 { "→" } else { "←" });
}

fn demo_phi_arithmetic() {
    println!("═══ 2. LOGARITHMIC ARITHMETIC ═══\n");
    println!("Multiplication becomes addition:\n");

    let n5 = latent_n::LatentN::new(5);
    let n7 = latent_n::LatentN::new(7);

    println!("  φ⁵ × φ⁷ = φ⁵⁺⁷ = φ¹²");
    println!("  ├─ Traditional: F[5] × F[7] = {} × {} = {}",
             n5.fibonacci(), n7.fibonacci(), n5.fibonacci() * n7.fibonacci());
    println!("  └─ φ-space:     n5 + n7 = {} + {} = {}", n5.n, n7.n, n5.n + n7.n);

    let product = phi_arithmetic::multiply(n5, n7).unwrap();
    println!("  Result: F[{}] = {}\n", product.n, product.fibonacci());

    let resistance = phi_arithmetic::measure_multiply_resistance();
    println!("  Resistance ratio: {:.1}x faster\n", resistance.ratio);
}

fn demo_boundary_solving() {
    println!("═══ 3. BOUNDARY-FIRST PUZZLE SOLVING ═══\n");
    println!("Know the end, fill in the middle:\n");

    let boundary = boundary::Boundary::new(20);
    println!("  Boundary: n = 20");
    println!("  ├─ Energy budget: F[20] = {}", boundary.energy);
    println!("  ├─ Time budget:   L[20] = {}", boundary.time);
    println!("  └─ Pattern:       {} components\n", boundary.pattern.complexity());

    let sequence = boundary.complete_puzzle();
    println!("  Dual-direction solving:");
    println!("  ├─ Forward (φ):  {} steps", sequence.forward.len());
    println!("  ├─ Backward (ψ): {} steps", sequence.backward.len());

    if let Some(eq) = sequence.equilibrium {
        println!("  └─ Meet at:      n = {} (Nash equilibrium)\n", eq);
    } else {
        println!("  └─ Equilibrium:  merged\n");
    }
}

fn demo_memory_allocation() {
    println!("═══ 4. BASE-φ MEMORY ALLOCATION ═══\n");
    println!("Memory organized as powers of φ:\n");

    let mut allocator = memory::PhiAllocator::new();

    println!("  Allocate 100 bytes:");
    let block = allocator.allocate(100).unwrap();

    println!("  ├─ Size:     {} bytes (F[{}])", block.size, block.n.n);
    println!("  ├─ Lifetime: {} cycles (L[{}])", block.lifetime, block.n.n);
    println!("  ├─ Address:  0x{:x} (Zeckendorf)", block.address);
    println!("  ├─ Checksum: {} (Cassini)", block.checksum);
    println!("  └─ Stable:   {}\n", if block.is_stable() { "yes" } else { "no (has gaps)" });

    if !block.is_stable() {
        println!("  Corruption sites (gaps in Zeckendorf): {:?}\n", block.corruption_sites());
    }

    allocator.deallocate(block);
    println!("  Deallocated. Memory returned to free list.\n");
}

fn demo_token_generation() {
    println!("═══ 5. TOKEN STREAM WITH NATURAL BOUNDARIES ═══\n");
    println!("Generation stops automatically:\n");

    let mut stream = token_stream::TokenStream::new(30);
    stream.generate_all();

    println!("  Stream to boundary n=30:");
    println!("  ├─ Tokens generated: {}", stream.tokens().len());
    println!("  ├─ Total energy:     {}", stream.total_energy());
    println!("  ├─ Total time:       {}", stream.total_time());
    println!("  ├─ Boundaries hit:   {}", stream.boundary_count());
    println!("  └─ Checkpoints:      {}\n", stream.checkpoint_count());

    println!("  First 5 tokens:");
    for (i, token) in stream.tokens().iter().take(5).enumerate() {
        let flags = if token.is_checkpoint {
            " [CHECKPOINT]"
        } else if token.is_boundary {
            " [BOUNDARY]"
        } else {
            ""
        };
        println!("    {}. n={}, val={}{}", i, token.n.n, token.value(), flags);
    }
    println!();
}

fn demo_resistance_metrics() {
    println!("═══ 6. RESISTANCE METRICS ═══\n");
    println!("Computational efficiency comparison:\n");

    let mult_r = phi_arithmetic::measure_multiply_resistance();
    let div_r = phi_arithmetic::measure_divide_resistance();
    let pow_r = phi_arithmetic::measure_power_resistance(10);

    println!("  Operation      | Normal FLOPs | φ-ops | Ratio");
    println!("  ─────────────────────────────────────────────");
    println!("  Multiply       | {:>12} | {:>5} | {:>5.0}x", mult_r.normal_flops, mult_r.phi_ops, mult_r.ratio);
    println!("  Divide         | {:>12} | {:>5} | {:>5.0}x", div_r.normal_flops, div_r.phi_ops, div_r.ratio);
    println!("  Power(x, 10)   | {:>12} | {:>5} | {:>5.0}x", pow_r.normal_flops, pow_r.phi_ops, pow_r.ratio);
    println!();

    println!("  All φ-space operations are O(1) integer table lookups.");
    println!("  No floating point. No iteration. No complexity.\n");
}
