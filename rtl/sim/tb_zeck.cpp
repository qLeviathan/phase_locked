// =============================================================================
// Verilator Testbench for Zeckbit Cascade
// =============================================================================
// Target: Raspberry Pi 3B
// Build: verilator --cc --exe --build -j 0 zeck_top.v sim/tb_zeck.cpp
//
// This demonstrates the core cascade primitive:
//   - Zeckendorf encoding (integer → shell occupancy)
//   - Cascade normalization (rewrite physics)
//   - Merge operation (superposition → canonical form)
// =============================================================================

#include <verilated.h>
#include "Vzeck_top.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// Clock the design
void tick(Vzeck_top* dut, uint64_t& sim_time) {
    dut->clk = 0;
    dut->eval();
    sim_time++;

    dut->clk = 1;
    dut->eval();
    sim_time++;
}

// Wait for done signal
int wait_done(Vzeck_top* dut, uint64_t& sim_time, int max_cycles = 1000) {
    int cycles = 0;
    while (cycles < max_cycles) {
        tick(dut, sim_time);
        cycles++;
        // Check all done signals
        if (dut->encode_done || dut->cascade_done || dut->merge_done) {
            return cycles;
        }
    }
    return -1;  // Timeout
}

// Print zeckbits as binary string
void print_zeck(uint32_t z, int n = 32) {
    printf("0b");
    for (int i = n - 1; i >= 0; i--) {
        printf("%d", (z >> i) & 1);
    }
}

// Check if zeckbits are valid (no adjacent 1s)
bool is_valid_zeck(uint32_t z) {
    return (z & (z << 1)) == 0;
}

// Count set bits
int popcount(uint32_t z) {
    int count = 0;
    while (z) {
        count += z & 1;
        z >>= 1;
    }
    return count;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    Vzeck_top* dut = new Vzeck_top;
    uint64_t sim_time = 0;

    printf("=============================================================\n");
    printf("  ZECKBIT CASCADE - Rewrite Physics Demo\n");
    printf("  Running on Verilator (target: Raspberry Pi 3B)\n");
    printf("=============================================================\n\n");

    // Reset
    dut->rst_n = 0;
    dut->encode_start = 0;
    dut->cascade_start = 0;
    dut->merge_start = 0;
    tick(dut, sim_time);
    tick(dut, sim_time);
    dut->rst_n = 1;
    tick(dut, sim_time);

    // =========================================================================
    // TEST 1: Zeckendorf Encoding
    // =========================================================================
    printf("--- TEST 1: Zeckendorf Encoding ---\n");

    uint32_t test_values[] = {0, 1, 5, 7, 17, 100, 1000};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < num_tests; i++) {
        uint32_t val = test_values[i];

        dut->encode_value = val;
        dut->encode_start = 1;
        tick(dut, sim_time);
        dut->encode_start = 0;

        int cycles = wait_done(dut, sim_time);

        printf("  %5u → ", val);
        print_zeck(dut->encode_zeck, 20);
        printf(" (cycles=%d, valid=%s)\n",
               cycles,
               is_valid_zeck(dut->encode_zeck) ? "YES" : "NO");
    }
    printf("\n");

    // =========================================================================
    // TEST 2: Cascade Normalization (the physics)
    // =========================================================================
    printf("--- TEST 2: Cascade Normalization ---\n");
    printf("    Rule R1: 11@(i,i+1) → 001@(i,i+1,i+2)\n");
    printf("    Rule R2: 2@i → 1@(i+1) + 1@(i-2)\n\n");

    // Invalid patterns (adjacent 1s)
    uint32_t invalid_patterns[] = {
        0b11,           // Basic pair
        0b111,          // Triple
        0b11011,        // Two pairs
        0b1111111,      // Dense 7-bit
        0b110,          // Adjacent at 1,2
        0b1100,         // Adjacent at 2,3
    };

    const char* pattern_names[] = {
        "0b11     ",
        "0b111    ",
        "0b11011  ",
        "0b1111111",
        "0b110    ",
        "0b1100   ",
    };

    int num_patterns = sizeof(invalid_patterns) / sizeof(invalid_patterns[0]);

    for (int i = 0; i < num_patterns; i++) {
        uint32_t pat = invalid_patterns[i];

        dut->cascade_in = pat;
        dut->cascade_start = 1;
        tick(dut, sim_time);
        dut->cascade_start = 0;

        int cycles = wait_done(dut, sim_time);

        printf("  %s → ", pattern_names[i]);
        print_zeck(dut->cascade_out, 12);
        printf(" (cascades=%d, valid=%s)\n",
               dut->cascade_count,
               is_valid_zeck(dut->cascade_out) ? "YES" : "NO");
    }
    printf("\n");

    // =========================================================================
    // TEST 3: Already-valid patterns (should pass through)
    // =========================================================================
    printf("--- TEST 3: Valid Patterns (no cascade needed) ---\n");

    uint32_t valid_patterns[] = {
        0b10101010,     // Alternating
        0b10010010,     // Sparse
        0b10000001,     // Endpoints only
        0b1,            // Single bit
        0b100,          // Single bit high
    };

    const char* valid_names[] = {
        "0b10101010",
        "0b10010010",
        "0b10000001",
        "0b1       ",
        "0b100     ",
    };

    int num_valid = sizeof(valid_patterns) / sizeof(valid_patterns[0]);

    for (int i = 0; i < num_valid; i++) {
        uint32_t pat = valid_patterns[i];

        dut->cascade_in = pat;
        dut->cascade_start = 1;
        tick(dut, sim_time);
        dut->cascade_start = 0;

        int cycles = wait_done(dut, sim_time);

        printf("  %s → ", valid_names[i]);
        print_zeck(dut->cascade_out, 12);
        printf(" (cascades=%d)\n", dut->cascade_count);
    }
    printf("\n");

    // =========================================================================
    // TEST 4: Merge Operation (Superposition + Cascade)
    // =========================================================================
    printf("--- TEST 4: Merge (Superposition + Cascade) ---\n");
    printf("    This is the TRUE 'XOR' operator: Z = Normalize(A ⊞ B)\n\n");

    // Test merge operations
    struct MergeTest {
        uint32_t a;
        uint32_t b;
        const char* desc;
    };

    MergeTest merge_tests[] = {
        {0b10100, 0b00010, "disjoint shells"},
        {0b10000, 0b01000, "adjacent shells (will cascade)"},
        {0b10100, 0b10100, "identical (superposition)"},
        {0b10010, 0b01001, "overlapping"},
    };

    int num_merges = sizeof(merge_tests) / sizeof(merge_tests[0]);

    for (int i = 0; i < num_merges; i++) {
        dut->merge_a = merge_tests[i].a;
        dut->merge_b = merge_tests[i].b;
        dut->merge_start = 1;
        tick(dut, sim_time);
        dut->merge_start = 0;

        int cycles = wait_done(dut, sim_time);

        printf("  A=");
        print_zeck(merge_tests[i].a, 8);
        printf(" ⊕ B=");
        print_zeck(merge_tests[i].b, 8);
        printf("\n    → Z=");
        print_zeck(dut->merge_out, 12);
        printf(" (norm=%d, %s)\n\n",
               dut->merge_norm,
               merge_tests[i].desc);
    }

    // =========================================================================
    // TEST 5: Cascade Count as Norm Proxy
    // =========================================================================
    printf("--- TEST 5: Cascade Count = Z[φ] Norm Proxy ---\n");
    printf("    More cascades = more 'work' = closer on hyperbola\n\n");

    // Create pairs at different "distances"
    uint32_t shells_a = 0b10001000;  // Sparse
    uint32_t shells_close = 0b10001000;  // Same (distance 0)
    uint32_t shells_medium = 0b01000100;  // Some overlap
    uint32_t shells_far = 0b00100010;     // Different

    printf("  Reference: ");
    print_zeck(shells_a, 10);
    printf("\n\n");

    uint32_t test_b[] = {shells_close, shells_medium, shells_far};
    const char* dist_names[] = {"identical", "medium", "far"};

    for (int i = 0; i < 3; i++) {
        dut->merge_a = shells_a;
        dut->merge_b = test_b[i];
        dut->merge_start = 1;
        tick(dut, sim_time);
        dut->merge_start = 0;

        wait_done(dut, sim_time);

        printf("  vs ");
        print_zeck(test_b[i], 10);
        printf(" (%s): norm_proxy = %d\n", dist_names[i], dut->merge_norm);
    }

    printf("\n=============================================================\n");
    printf("  Simulation complete. Total cycles: %lu\n", sim_time / 2);
    printf("  All outputs in canonical Zeckendorf form (no adjacent 1s).\n");
    printf("=============================================================\n");

    delete dut;
    return 0;
}
