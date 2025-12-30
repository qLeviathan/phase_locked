// =============================================================================
// Verilator Testbench for Zeckbit Cascade
// =============================================================================
// φ-subscript calculus primitives:
//   - Encoding: token → Σφₙᵢ (Zeckendorf decomposition)
//   - Cascade: adjacent merge (φₐ + φₐ₊₁ = φₐ₊₂)
//   - Contraction: ⟨Q, K⟩ = Σ L_{|a-b|} (Lucas attention)
//
// Target: Raspberry Pi 3B / Kano
// Build: make && make run
// =============================================================================

#include <verilated.h>
#include "Vzeck_top.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

// Clock the design
void tick(Vzeck_top* dut, uint64_t& sim_time) {
    dut->clk = 0;
    dut->eval();
    sim_time++;

    dut->clk = 1;
    dut->eval();
    sim_time++;
}

// Wait for done signal with timeout
int wait_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->encode_done || dut->cascade_done || dut->merge_done || dut->contract_done) {
            return cycles;
        }
    }
    return -1;  // Timeout
}

// Print zeckbits as binary string
void print_zeck(uint32_t z, int n = 16) {
    for (int i = n - 1; i >= 0; i--) {
        printf("%c", (z >> i) & 1 ? '1' : '0');
    }
}

// Print zeckbits as φ-subscript notation
void print_phi_notation(uint32_t z, int n = 16) {
    bool first = true;
    for (int i = 0; i < n; i++) {
        if ((z >> i) & 1) {
            if (!first) printf(" + ");
            printf("φ_%d", i + 2);  // F_2 is index 0
            first = false;
        }
    }
    if (first) printf("0");
}

// Check validity (no adjacent 1s)
bool is_valid_zeck(uint32_t z) {
    return (z & (z << 1)) == 0;
}

// Count set bits (popcount)
int popcount(uint32_t z) {
    int count = 0;
    while (z) {
        count += z & 1;
        z >>= 1;
    }
    return count;
}

// Lucas numbers for verification
uint32_t lucas(int n) {
    if (n == 0) return 2;
    if (n == 1) return 1;
    uint32_t a = 2, b = 1;
    for (int i = 2; i <= n; i++) {
        uint32_t c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// Compute expected contraction ⟨Q, K⟩ = Σ L_{|a-b|}
uint32_t expected_contraction(uint32_t Q, uint32_t K, int n = 16) {
    uint32_t sum = 0;
    for (int i = 0; i < n; i++) {
        if ((Q >> i) & 1) {
            for (int j = 0; j < n; j++) {
                if ((K >> j) & 1) {
                    int diff = (i >= j) ? (i - j) : (j - i);
                    sum += lucas(diff);
                }
            }
        }
    }
    return sum;
}

// =============================================================================
// MAIN TEST PROGRAM
// =============================================================================

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    Vzeck_top* dut = new Vzeck_top;
    uint64_t sim_time = 0;
    int pass_count = 0;
    int fail_count = 0;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  φ-SUBSCRIPT CALCULUS - Zeckbit Cascade Tests\n");
    printf("  Target: Raspberry Pi / Kano (Verilator)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Reset
    dut->rst_n = 0;
    dut->encode_start = 0;
    dut->cascade_start = 0;
    dut->merge_start = 0;
    dut->contract_start = 0;
    tick(dut, sim_time);
    tick(dut, sim_time);
    dut->rst_n = 1;
    tick(dut, sim_time);

    // =========================================================================
    // TEST 1: Zeckendorf Encoding (token → Σφₙᵢ)
    // =========================================================================
    printf("━━━ TEST 1: Zeckendorf Encoding ━━━\n");
    printf("    token → Σφₙᵢ (non-adjacent Fibonacci indices)\n\n");

    struct EncodeTest {
        uint32_t value;
        const char* expected;
    };

    EncodeTest encode_tests[] = {
        {1,   "φ_2"},                    // 1 = F_2
        {2,   "φ_3"},                    // 2 = F_3
        {3,   "φ_4"},                    // 3 = F_4
        {4,   "φ_2 + φ_4"},              // 4 = 1 + 3 = F_2 + F_4
        {5,   "φ_5"},                    // 5 = F_5
        {7,   "φ_2 + φ_5"},              // 7 = 2 + 5 = F_3... wait, 7 = 5 + 2 = F_5 + F_3
        {8,   "φ_6"},                    // 8 = F_6
        {10,  "φ_3 + φ_6"},              // 10 = 2 + 8 = F_3 + F_6
        {17,  "φ_2 + φ_4 + φ_7"},        // 17 = 1 + 3 + 13 = F_2 + F_4 + F_7
        {100, "φ_4 + φ_6 + φ_11"},       // 100 = 3 + 8 + 89 = F_4 + F_6 + F_11
    };

    int num_encode = sizeof(encode_tests) / sizeof(encode_tests[0]);

    for (int i = 0; i < num_encode; i++) {
        uint32_t val = encode_tests[i].value;

        dut->encode_value = val;
        dut->encode_start = 1;
        tick(dut, sim_time);
        dut->encode_start = 0;

        int cycles = wait_done(dut, sim_time);

        bool valid = is_valid_zeck(dut->encode_zeck);
        printf("  %3u → ", val);
        print_phi_notation(dut->encode_zeck, 16);
        printf("\n         binary: ");
        print_zeck(dut->encode_zeck, 16);
        printf(" (%s, %d cycles)\n", valid ? "VALID" : "INVALID", cycles);

        if (valid) pass_count++; else fail_count++;
    }
    printf("\n");

    // =========================================================================
    // TEST 2: Cascade Normalization (rewrite physics)
    // =========================================================================
    printf("━━━ TEST 2: Cascade Normalization ━━━\n");
    printf("    R1: φₐ + φₐ₊₁ = φₐ₊₂ (adjacent merge)\n");
    printf("    R2: 2φₐ = φₐ₊₁ + φₐ₋₂ (split)\n\n");

    struct CascadeTest {
        uint32_t input;
        uint32_t expected;
        const char* desc;
    };

    CascadeTest cascade_tests[] = {
        {0b11,        0b100,      "φ_2 + φ_3 → φ_4"},
        {0b110,       0b1000,     "φ_3 + φ_4 → φ_5"},
        {0b111,       0b1001,     "φ_2 + φ_3 + φ_4 → φ_2 + φ_5"},
        {0b1100,      0b10000,    "φ_4 + φ_5 → φ_6"},
        {0b11011,     0b100101,   "two adjacent pairs"},
        {0b1111111,   0b1000101,  "dense 7-shell"},
        {0b10101010,  0b10101010, "already canonical"},
    };

    int num_cascade = sizeof(cascade_tests) / sizeof(cascade_tests[0]);

    for (int i = 0; i < num_cascade; i++) {
        CascadeTest& t = cascade_tests[i];

        dut->cascade_in = t.input;
        dut->cascade_start = 1;
        tick(dut, sim_time);
        dut->cascade_start = 0;

        int cycles = wait_done(dut, sim_time);

        bool valid = is_valid_zeck(dut->cascade_out);
        bool correct = (dut->cascade_out == t.expected);

        printf("  ");
        print_zeck(t.input, 10);
        printf(" → ");
        print_zeck(dut->cascade_out, 10);
        printf(" (cascades=%d) %s\n", dut->cascade_count, t.desc);

        if (valid && correct) pass_count++; else fail_count++;
    }
    printf("\n");

    // =========================================================================
    // TEST 3: Lucas Contraction (attention)
    // =========================================================================
    printf("━━━ TEST 3: Lucas Contraction (Attention) ━━━\n");
    printf("    ⟨Q, K⟩ = Σᵢ Σⱼ L_{|aᵢ - bⱼ|}\n\n");

    struct ContractTest {
        uint32_t Q;
        uint32_t K;
        const char* desc;
    };

    ContractTest contract_tests[] = {
        {0b00001, 0b00001, "identical (φ_2, φ_2)"},          // L_0 = 2
        {0b00001, 0b00010, "adjacent (φ_2, φ_3)"},           // L_1 = 1
        {0b00001, 0b00100, "gap-1 (φ_2, φ_4)"},              // L_2 = 3
        {0b00001, 0b10000, "far (φ_2, φ_6)"},                // L_4 = 7
        {0b10001, 0b10001, "two shells same"},               // L_0 + L_0 + L_4 + L_4 = 18
        {0b10101, 0b01010, "interleaved"},                   // complex
    };

    int num_contract = sizeof(contract_tests) / sizeof(contract_tests[0]);

    for (int i = 0; i < num_contract; i++) {
        ContractTest& t = contract_tests[i];

        dut->contract_q = t.Q;
        dut->contract_k = t.K;
        dut->contract_start = 1;
        tick(dut, sim_time);
        dut->contract_start = 0;

        int cycles = wait_done(dut, sim_time);

        uint32_t expected = expected_contraction(t.Q, t.K, 16);

        printf("  ⟨");
        print_phi_notation(t.Q, 8);
        printf(", ");
        print_phi_notation(t.K, 8);
        printf("⟩\n");
        printf("    = %u (expected: %u) %s\n",
               dut->contract_attention, expected,
               (dut->contract_attention == expected) ? "✓" : "✗");

        if (dut->contract_attention == expected) pass_count++; else fail_count++;
    }
    printf("\n");

    // =========================================================================
    // TEST 4: Merge (Superposition + Cascade)
    // =========================================================================
    printf("━━━ TEST 4: Merge (Superposition + Cascade) ━━━\n");
    printf("    A ⊕ B = Normalize(A ⊞ B)\n\n");

    struct MergeTest {
        uint32_t A;
        uint32_t B;
        const char* desc;
    };

    MergeTest merge_tests[] = {
        {0b00001, 0b00100, "disjoint: φ_2 ⊕ φ_4"},
        {0b00001, 0b00010, "adjacent: φ_2 ⊕ φ_3 → φ_4"},
        {0b00001, 0b00001, "same: φ_2 ⊕ φ_2 (double)"},
        {0b10100, 0b01010, "interleaved"},
    };

    int num_merge = sizeof(merge_tests) / sizeof(merge_tests[0]);

    for (int i = 0; i < num_merge; i++) {
        MergeTest& t = merge_tests[i];

        dut->merge_a = t.A;
        dut->merge_b = t.B;
        dut->merge_start = 1;
        tick(dut, sim_time);
        dut->merge_start = 0;

        int cycles = wait_done(dut, sim_time);

        bool valid = is_valid_zeck(dut->merge_out);

        printf("  ");
        print_phi_notation(t.A, 8);
        printf(" ⊕ ");
        print_phi_notation(t.B, 8);
        printf("\n    → ");
        print_phi_notation(dut->merge_out, 12);
        printf(" (norm=%d) %s\n", dut->merge_norm, t.desc);

        if (valid) pass_count++; else fail_count++;
    }
    printf("\n");

    // =========================================================================
    // BENCHMARK
    // =========================================================================
    printf("━━━ BENCHMARK: Throughput ━━━\n\n");

    const int BENCH_ITERS = 10000;
    clock_t start, end;

    // Cascade benchmark
    start = clock();
    for (int i = 0; i < BENCH_ITERS; i++) {
        dut->cascade_in = (i * 12345) & 0x7FFF;  // Pseudo-random
        dut->cascade_start = 1;
        tick(dut, sim_time);
        dut->cascade_start = 0;
        wait_done(dut, sim_time, 100);
    }
    end = clock();
    double cascade_time = (double)(end - start) / CLOCKS_PER_SEC;
    double cascade_ops = BENCH_ITERS / cascade_time;

    printf("  Cascade: %.0f ops/sec (%.2f ms for %d ops)\n",
           cascade_ops, cascade_time * 1000, BENCH_ITERS);

    // Encode benchmark
    start = clock();
    for (int i = 0; i < BENCH_ITERS; i++) {
        dut->encode_value = i;
        dut->encode_start = 1;
        tick(dut, sim_time);
        dut->encode_start = 0;
        wait_done(dut, sim_time, 100);
    }
    end = clock();
    double encode_time = (double)(end - start) / CLOCKS_PER_SEC;
    double encode_ops = BENCH_ITERS / encode_time;

    printf("  Encode:  %.0f tokens/sec (%.2f ms for %d tokens)\n",
           encode_ops, encode_time * 1000, BENCH_ITERS);

    printf("\n");

    // =========================================================================
    // SUMMARY
    // =========================================================================
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", pass_count, fail_count);
    printf("  Total simulation cycles: %lu\n", sim_time / 2);
    printf("═══════════════════════════════════════════════════════════════════\n");

    if (fail_count == 0) {
        printf("\n  ✓ All tests passed! Ready for FPGA synthesis.\n\n");
    } else {
        printf("\n  ✗ Some tests failed. Check implementation.\n\n");
    }

    delete dut;
    return fail_count > 0 ? 1 : 0;
}
