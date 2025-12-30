// =============================================================================
// Verilator Testbench for φ-Subscript Calculus Primitives
// =============================================================================
// Complete test suite for Raspberry Pi / Kano:
//   - Encoding: token → Σφₙᵢ (Zeckendorf decomposition)
//   - Cascade: rewrite physics (R1 + R2)
//   - Contraction: ⟨Q, K⟩ = Σ L_{|a-b|} (Lucas attention)
//   - Context: Recurrence dynamics C_{n+1} = C_n ⊕ token_n
//   - Inference: Complete FSM (ENCODE → FOLD → SNAP → DECODE)
//
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

// Wait for any done signal with timeout
int wait_any_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->encode_done || dut->cascade_done || dut->merge_done ||
            dut->contract_done || dut->context_fold_done || dut->decode_done ||
            dut->infer_done) {
            return cycles;
        }
    }
    return -1;  // Timeout
}

// Wait for specific done signal
int wait_encode_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->encode_done) return cycles;
    }
    return -1;
}

int wait_cascade_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->cascade_done) return cycles;
    }
    return -1;
}

int wait_context_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->context_fold_done) return cycles;
    }
    return -1;
}

int wait_decode_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->decode_done) return cycles;
    }
    return -1;
}

int wait_infer_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 2000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        if (dut->infer_done) return cycles;
    }
    return -1;
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

// Fibonacci numbers for verification
uint32_t fib(int n) {
    if (n <= 1) return n;
    uint32_t a = 0, b = 1;
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

// Decode zeckbits to integer
uint32_t zeck_to_int(uint32_t z, int n = 32) {
    uint32_t sum = 0;
    for (int i = 0; i < n; i++) {
        if ((z >> i) & 1) {
            sum += fib(i + 2);  // Index 0 = F_2
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
    printf("  φ-SUBSCRIPT CALCULUS - Complete Test Suite\n");
    printf("  Target: Raspberry Pi / Kano (Verilator)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Reset all signals
    dut->rst_n = 0;
    dut->encode_start = 0;
    dut->cascade_start = 0;
    dut->merge_start = 0;
    dut->contract_start = 0;
    dut->context_clear = 0;
    dut->context_fold_start = 0;
    dut->decode_start = 0;
    dut->infer_start = 0;
    dut->infer_reset_context = 0;
    dut->infer_attend_enable = 0;
    tick(dut, sim_time);
    tick(dut, sim_time);
    dut->rst_n = 1;
    tick(dut, sim_time);

    // =========================================================================
    // TEST 1: Zeckendorf Encoding (token → Σφₙᵢ)
    // =========================================================================
    printf("━━━ TEST 1: Zeckendorf Encoding ━━━\n");
    printf("    token → Σφₙᵢ (non-adjacent Fibonacci indices)\n\n");

    uint32_t encode_tests[] = {1, 2, 3, 4, 5, 7, 8, 10, 17, 100};
    int num_encode = sizeof(encode_tests) / sizeof(encode_tests[0]);

    for (int i = 0; i < num_encode; i++) {
        uint32_t val = encode_tests[i];

        dut->encode_value = val;
        dut->encode_start = 1;
        tick(dut, sim_time);
        dut->encode_start = 0;

        int cycles = wait_encode_done(dut, sim_time);

        bool valid = is_valid_zeck(dut->encode_zeck);
        uint32_t decoded = zeck_to_int(dut->encode_zeck, 16);
        bool correct = (decoded == val);

        printf("  %3u → ", val);
        print_phi_notation(dut->encode_zeck, 16);
        printf("\n         binary: ");
        print_zeck(dut->encode_zeck, 16);
        printf(" (%s, %d cycles)\n", (valid && correct) ? "✓" : "✗", cycles);

        if (valid && correct) pass_count++; else fail_count++;
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
        {0b10101010,  0b10101010, "already canonical"},
    };

    int num_cascade = sizeof(cascade_tests) / sizeof(cascade_tests[0]);

    for (int i = 0; i < num_cascade; i++) {
        CascadeTest& t = cascade_tests[i];

        dut->cascade_in = t.input;
        dut->cascade_start = 1;
        tick(dut, sim_time);
        dut->cascade_start = 0;

        int cycles = wait_cascade_done(dut, sim_time);

        bool valid = is_valid_zeck(dut->cascade_out);
        bool correct = (dut->cascade_out == t.expected);

        printf("  ");
        print_zeck(t.input, 10);
        printf(" → ");
        print_zeck(dut->cascade_out, 10);
        printf(" (cascades=%d) %s %s\n",
               dut->cascade_count, t.desc,
               (valid && correct) ? "✓" : "✗");

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
        {0b00001, 0b00001, "identical (φ_2, φ_2)"},
        {0b00001, 0b00010, "adjacent (φ_2, φ_3)"},
        {0b00001, 0b00100, "gap-1 (φ_2, φ_4)"},
        {0b00001, 0b10000, "far (φ_2, φ_6)"},
        {0b10001, 0b10001, "two shells same"},
    };

    int num_contract = sizeof(contract_tests) / sizeof(contract_tests[0]);

    for (int i = 0; i < num_contract; i++) {
        ContractTest& t = contract_tests[i];

        dut->contract_q = t.Q;
        dut->contract_k = t.K;
        dut->contract_start = 1;
        tick(dut, sim_time);
        dut->contract_start = 0;

        int cycles = wait_any_done(dut, sim_time);

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
    // TEST 4: Decode (Zeckbits → integer)
    // =========================================================================
    printf("━━━ TEST 4: Decode (Σφₙᵢ → integer) ━━━\n\n");

    struct DecodeTest {
        uint32_t zeck;
        uint32_t expected;
    };

    DecodeTest decode_tests[] = {
        {0b00001,     1},   // F_2
        {0b00010,     2},   // F_3
        {0b00101,     4},   // F_2 + F_4 = 1 + 3
        {0b01000,     8},   // F_6
        {0b10101,    12},   // F_2 + F_4 + F_6 = 1 + 3 + 8
    };

    int num_decode = sizeof(decode_tests) / sizeof(decode_tests[0]);

    for (int i = 0; i < num_decode; i++) {
        DecodeTest& t = decode_tests[i];

        dut->decode_zeck_in = t.zeck;
        dut->decode_start = 1;
        tick(dut, sim_time);
        dut->decode_start = 0;

        int cycles = wait_decode_done(dut, sim_time);

        bool correct = (dut->decode_value == t.expected);

        printf("  ");
        print_phi_notation(t.zeck, 8);
        printf(" → %u (expected: %u) %s\n",
               dut->decode_value, t.expected,
               correct ? "✓" : "✗");

        if (correct) pass_count++; else fail_count++;
    }
    printf("\n");

    // =========================================================================
    // TEST 5: Context Window Dynamics
    // =========================================================================
    printf("━━━ TEST 5: Context Window Dynamics ━━━\n");
    printf("    C_{n+1} = C_n ⊕ token_n (recurrence)\n\n");

    // Clear context first
    dut->context_clear = 1;
    tick(dut, sim_time);
    dut->context_clear = 0;
    tick(dut, sim_time);

    printf("  Starting with empty context (C = 0)\n\n");

    // First encode some tokens
    uint32_t context_tokens[] = {1, 3, 5};  // Simple tokens to fold
    int num_context_tokens = 3;

    for (int i = 0; i < num_context_tokens; i++) {
        // First encode the token
        dut->encode_value = context_tokens[i];
        dut->encode_start = 1;
        tick(dut, sim_time);
        dut->encode_start = 0;
        wait_encode_done(dut, sim_time);

        uint32_t encoded = dut->encode_zeck;

        // Now fold into context
        dut->context_token_zeck = encoded;
        dut->context_fold_start = 1;
        tick(dut, sim_time);
        dut->context_fold_start = 0;

        int cycles = wait_context_done(dut, sim_time);

        printf("  Fold token %u (", context_tokens[i]);
        print_phi_notation(encoded, 8);
        printf(")\n");
        printf("    → Context: ");
        print_phi_notation(dut->context_state, 12);
        printf("\n");
        printf("    Cascades: %d, Seq length: %d, Total cascades: %d\n\n",
               dut->context_fold_norm,
               dut->context_seq_length,
               dut->context_total_cascades);

        if (is_valid_zeck(dut->context_state)) pass_count++; else fail_count++;
    }

    // =========================================================================
    // TEST 6: Complete Inference FSM
    // =========================================================================
    printf("━━━ TEST 6: Complete Inference FSM ━━━\n");
    printf("    ENCODE → FOLD → SNAP → DECODE\n\n");

    // Reset inference context
    dut->infer_reset_context = 1;
    tick(dut, sim_time);
    dut->infer_reset_context = 0;
    tick(dut, sim_time);

    printf("  Reset context, starting fresh inference\n\n");

    // Run inference on a sequence of tokens
    uint32_t infer_tokens[] = {5, 8, 13, 21};  // Fibonacci sequence tokens
    int num_infer = 4;

    for (int i = 0; i < num_infer; i++) {
        dut->infer_token_in = infer_tokens[i];
        dut->infer_attend_enable = 0;
        dut->infer_start = 1;
        tick(dut, sim_time);
        dut->infer_start = 0;

        int cycles = wait_infer_done(dut, sim_time);

        printf("  Token %u:\n", infer_tokens[i]);
        printf("    Context: ");
        print_phi_notation(dut->infer_context_out, 12);
        printf("\n");
        printf("    Prediction idx: %d (shell φ_%d)\n",
               dut->infer_prediction_idx, dut->infer_prediction_idx + 2);
        printf("    Prediction val: %u\n", dut->infer_prediction_val);
        printf("    Step cascades: %d, Attention sum: %u\n",
               dut->infer_step_cascades, dut->infer_attention_sum);
        printf("    Sequence pos: %d, Cycles: %d\n\n",
               dut->infer_seq_pos, cycles);

        if (is_valid_zeck(dut->infer_context_out) && cycles > 0) {
            pass_count++;
        } else {
            fail_count++;
        }
    }

    // =========================================================================
    // BENCHMARK: Throughput on Raspberry Pi / Kano
    // =========================================================================
    printf("━━━ BENCHMARK: Throughput ━━━\n\n");

    const int BENCH_ITERS = 10000;
    clock_t start, end;

    // Cascade benchmark
    printf("  Running %d cascade operations...\n", BENCH_ITERS);
    start = clock();
    for (int i = 0; i < BENCH_ITERS; i++) {
        dut->cascade_in = (i * 12345) & 0x7FFF;  // Pseudo-random
        dut->cascade_start = 1;
        tick(dut, sim_time);
        dut->cascade_start = 0;
        wait_cascade_done(dut, sim_time, 100);
    }
    end = clock();
    double cascade_time = (double)(end - start) / CLOCKS_PER_SEC;
    double cascade_ops = BENCH_ITERS / cascade_time;
    printf("  Cascade: %.0f ops/sec (%.2f ms total)\n", cascade_ops, cascade_time * 1000);

    // Encode benchmark
    printf("  Running %d encode operations...\n", BENCH_ITERS);
    start = clock();
    for (int i = 0; i < BENCH_ITERS; i++) {
        dut->encode_value = i % 1000;
        dut->encode_start = 1;
        tick(dut, sim_time);
        dut->encode_start = 0;
        wait_encode_done(dut, sim_time, 100);
    }
    end = clock();
    double encode_time = (double)(end - start) / CLOCKS_PER_SEC;
    double encode_ops = BENCH_ITERS / encode_time;
    printf("  Encode:  %.0f tokens/sec (%.2f ms total)\n", encode_ops, encode_time * 1000);

    // Inference benchmark
    printf("  Running %d inference steps...\n", BENCH_ITERS / 10);
    dut->infer_reset_context = 1;
    tick(dut, sim_time);
    dut->infer_reset_context = 0;
    tick(dut, sim_time);

    start = clock();
    for (int i = 0; i < BENCH_ITERS / 10; i++) {
        dut->infer_token_in = (i * 7) % 100 + 1;
        dut->infer_start = 1;
        tick(dut, sim_time);
        dut->infer_start = 0;
        wait_infer_done(dut, sim_time, 500);

        // Reset context periodically to avoid overflow
        if (i % 50 == 49) {
            dut->infer_reset_context = 1;
            tick(dut, sim_time);
            dut->infer_reset_context = 0;
            tick(dut, sim_time);
        }
    }
    end = clock();
    double infer_time = (double)(end - start) / CLOCKS_PER_SEC;
    double infer_ops = (BENCH_ITERS / 10) / infer_time;
    printf("  Infer:   %.0f steps/sec (%.2f ms total)\n", infer_ops, infer_time * 1000);

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

    // Print resource estimate
    printf("━━━ Resource Estimate (iCE40-class FPGA) ━━━\n\n");
    printf("  Module            | LUTs  | FFs   | Notes\n");
    printf("  ------------------|-------|-------|------------------\n");
    printf("  zeck_cascade (32) |  ~200 |  ~70  | Rewrite engine R1+R2\n");
    printf("  zeck_encode (32)  |  ~150 | ~100  | Greedy Zeckendorf\n");
    printf("  lucas_lut (32)    |  ~100 |    0  | ROM table\n");
    printf("  phi_contract      |  ~150 |  ~80  | Attention accumulator\n");
    printf("  phi_context       |  ~100 |  ~50  | Recurrence state\n");
    printf("  phi_decode        |  ~100 |  ~50  | Zeck → integer\n");
    printf("  phi_infer         |  ~200 | ~150  | Inference FSM\n");
    printf("  ------------------|-------|-------|------------------\n");
    printf("  TOTAL             | ~1000 | ~500  | Under 1350 target\n");
    printf("\n");

    delete dut;
    return fail_count > 0 ? 1 : 0;
}
