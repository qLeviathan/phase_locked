# Cascade Logic Verilog Port Specification

## Complete φ-Mechanics Implementation Guide for FPGA/Verilator

**Generated from repository scan of phase_locked/**
**Target: ~1350 LUTs, Zero floating-point, Deterministic timing**

---

## 1. Core Algorithm: Cascade Normalization

### 1.1 Algorithm Specification

**Source Files:**
- `validate_zordic.py:17-34` - Reference Python implementation
- `unified-zeckendorf-cordic/core/src/cascade.rs:13-48` - Rust implementation
- `llama_compression/compress_llama.py:59-93` - Optimized bit-based version
- `phi_mamba/unified_tensor.py:243-297` - CUDA kernel reference

**Pseudocode (Hardware-Ready):**
```
CASCADE_NORMALIZE(bits[63:0]):
    max_iterations = 64  // Worst case for 64-bit input

    REPEAT max_iterations:
        // Step 1: Detect adjacent 1s
        adjacent = bits AND (bits << 1)

        // Step 2: Check termination
        IF adjacent == 0:
            RETURN bits  // Valid Zeckendorf form

        // Step 3: Find lowest violation position
        // Use two's complement trick: pos = ctz(adjacent)
        lowest_bit = adjacent AND (-adjacent)
        pos = count_trailing_zeros(lowest_bit)

        // Step 4: Apply cascade rule: F_k + F_{k+1} = F_{k+2}
        mask = ~(3 << pos)       // Clear bits at pos and pos+1
        bits = bits AND mask
        bits = bits OR (1 << (pos + 2))  // Set bit at pos+2

    RETURN bits
```

### 1.2 Bit-Width Analysis

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Maximum Fibonacci Index** | 63 (F_63 = 6,557,470,319,842) | For 64-bit operation |
| **Working Register Width** | 64 bits | Matches uint64 in implementations |
| **Recommended for FPGA** | 32 bits | F_32 = 2,178,309 (sufficient for most uses) |
| **Minimum Practical** | 20 bits | F_20 = 6,765 |

### 1.3 Iteration Count Analysis

**Worst-Case Iterations:**
- For N-bit input: Maximum N/2 cascades
- 64-bit: Max 32 iterations
- 32-bit: Max 16 iterations
- 20-bit: Max 10 iterations

**Typical Case:** 2-5 iterations for random inputs

### 1.4 Verilog Module Specification

```verilog
module cascade_normalize #(
    parameter WIDTH = 64,
    parameter MAX_ITER = 32
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [WIDTH-1:0] bits_in,
    output reg  [WIDTH-1:0] bits_out,
    output reg  done,
    output reg  [5:0] iteration_count  // Debug/profiling
);
```

---

## 2. Lookup Tables (LUTs)

### 2.1 Fibonacci LUT

**Source Files:**
- `phi_mamba/math_core.py:109-146` - fibonacci_sequence()
- `unified-zeckendorf-cordic/core/src/sequences.rs:7-26` - Rust fibonacci()
- `validate_zordic.py:13-15` - Simple precomputation

**Table Contents (20 entries for minimal implementation):**
```
Index | F_n        | Binary Width
------|------------|-------------
0     | 0          | 1
1     | 1          | 1
2     | 1          | 1
3     | 2          | 2
4     | 3          | 2
5     | 5          | 3
6     | 8          | 4
7     | 13         | 4
8     | 21         | 5
9     | 34         | 6
10    | 55         | 6
11    | 89         | 7
12    | 144        | 8
13    | 233        | 8
14    | 377        | 9
15    | 610        | 10
16    | 987        | 10
17    | 1597       | 11
18    | 2584       | 12
19    | 4181       | 13
20    | 6765       | 13
```

**Verilog Implementation:**
```verilog
// Read-only ROM - synthesizes to LUT or BRAM
module fibonacci_lut #(
    parameter DEPTH = 64,      // Number of entries
    parameter WIDTH = 64       // Bits per entry
)(
    input  wire [5:0] index,
    output wire [WIDTH-1:0] fib_value
);

// For small tables (< 32 entries), use combinational logic
reg [WIDTH-1:0] FIB_TABLE [0:DEPTH-1];
initial begin
    FIB_TABLE[0]  = 64'd0;
    FIB_TABLE[1]  = 64'd1;
    FIB_TABLE[2]  = 64'd1;
    FIB_TABLE[3]  = 64'd2;
    FIB_TABLE[4]  = 64'd3;
    FIB_TABLE[5]  = 64'd5;
    FIB_TABLE[6]  = 64'd8;
    FIB_TABLE[7]  = 64'd13;
    FIB_TABLE[8]  = 64'd21;
    FIB_TABLE[9]  = 64'd34;
    FIB_TABLE[10] = 64'd55;
    FIB_TABLE[11] = 64'd89;
    FIB_TABLE[12] = 64'd144;
    FIB_TABLE[13] = 64'd233;
    FIB_TABLE[14] = 64'd377;
    FIB_TABLE[15] = 64'd610;
    FIB_TABLE[16] = 64'd987;
    FIB_TABLE[17] = 64'd1597;
    FIB_TABLE[18] = 64'd2584;
    FIB_TABLE[19] = 64'd4181;
    FIB_TABLE[20] = 64'd6765;
    // Continue to F_63 if needed...
end

assign fib_value = FIB_TABLE[index];
endmodule
```

### 2.2 Lucas LUT

**Source Files:**
- `phi_mamba/math_core.py:65-103` - lucas()
- `unified-zeckendorf-cordic/core/src/sequences.rs:28-49` - Rust lucas()

**Table Contents (20 entries):**
```
Index | L_n  | Notes
------|------|-------
0     | 2    | Special case
1     | 1    |
2     | 3    | L_n = F_{n-1} + F_{n+1}
3     | 4    |
4     | 7    |
5     | 11   |
6     | 18   |
7     | 29   |
8     | 47   |
9     | 76   |
10    | 123  |
...
```

**Key Property for Validation:** `L_n² - 5·F_n² = 4·(-1)^n` (Discrete Lorentz invariant)

### 2.3 CORDIC Arctangent LUT

**Source File:** `phi_mamba/cordic.py:55-81`

**Table Contents (16 entries for 32-bit precision):**
```verilog
// Fixed-point atan(2^-i) values
// Scale factor: 2^32 represents 2π
localparam [31:0] ATAN_TABLE [0:15] = '{
    32'h20000000,  // atan(1)   = π/4 ≈ 0.7854
    32'h12E4051E,  // atan(1/2) ≈ 0.4636
    32'h09FB385B,  // atan(1/4) ≈ 0.2450
    32'h051111D4,  // atan(1/8) ≈ 0.1244
    32'h028B0D43,  // atan(1/16)
    32'h0145D7E1,
    32'h00A2F61E,
    32'h00517C55,
    32'h0028BE53,
    32'h00145F2F,
    32'h000A2F98,
    32'h000517CC,
    32'h00028BE6,
    32'h000145F3,
    32'h0000A2FA,
    32'h0000517D
};
```

### 2.4 LUT Access Patterns

| LUT Type | Access Pattern | Ports Needed | Notes |
|----------|----------------|--------------|-------|
| Fibonacci | Random (decomposition) | 1 read | Index from bit position |
| Lucas | Sequential (validation) | 1 read | Optional validation |
| CORDIC atan | Sequential (iteration) | 1 read | One per CORDIC step |

---

## 3. XOR Attention Mechanism

### 3.1 Algorithm Specification

**Source Files:**
- `phi_mamba/unified_tensor.py:60-63` - hamming_distance()
- `phi_mamba/unified_tensor.py:300-342` - phase_attention_kernel
- `bit_cascade/phinet.py:187-198` - interference()

**Core Operations:**

```
XOR_ATTENTION(query_bits, key_bits_array):
    // Step 1: Compute Hamming distance to all keys
    FOR EACH key IN key_bits_array:
        xor_result = query XOR key
        distance = popcount(xor_result)
        score = 64 - distance  // Resonance score

    // Step 2: Find best match (highest score)
    best_idx = argmax(scores)

    // Step 3: OR-combine for attention
    attended = query OR key_bits_array[best_idx]

    RETURN attended
```

**Verilog Implementation:**
```verilog
module xor_attention #(
    parameter WIDTH = 64,
    parameter NUM_KEYS = 16
)(
    input  wire [WIDTH-1:0] query,
    input  wire [WIDTH-1:0] keys [0:NUM_KEYS-1],
    output wire [WIDTH-1:0] attended,
    output wire [3:0] best_idx
);

wire [6:0] distances [0:NUM_KEYS-1];  // 0-64 range
wire [6:0] scores [0:NUM_KEYS-1];

genvar i;
generate
    for (i = 0; i < NUM_KEYS; i = i + 1) begin : dist_calc
        wire [WIDTH-1:0] xor_result;
        assign xor_result = query ^ keys[i];

        // Popcount - synthesizes to LUT tree
        popcount #(.WIDTH(WIDTH)) pc (
            .bits(xor_result),
            .count(distances[i])
        );

        assign scores[i] = WIDTH - distances[i];
    end
endgenerate

// Find maximum score (parallel comparator tree)
// ... argmax logic ...

assign attended = query | keys[best_idx];
endmodule
```

### 3.2 Shell Distance Calculation

**Formula:** `shell_distance = popcount(shells1 XOR shells2)`

Where shells are bit vectors representing active Fibonacci indices.

---

## 4. The Foundational Constraint: φ·ψ = -1

### 4.1 Mathematical Properties (from `phi_mamba/constants.py`)

```
φ = (1 + √5) / 2 ≈ 1.618033988749895
ψ = (1 - √5) / 2 ≈ -0.618033988749895 = -1/φ

Key Relations:
- φ × ψ = -1
- φ + ψ = 1
- φ - ψ = √5
- φ² = φ + 1
- ψ² = ψ + 1
```

### 4.2 Integer Approximations for Hardware

**Source:** `TENSOR_SERIES.py:26-27`

```verilog
// Fibonacci ratio approximation of 1/φ
// F_14/F_15 = 377/610 ≈ 0.6180327...
localparam [15:0] INV_PHI_NUM = 16'd377;   // Numerator
localparam [15:0] INV_PHI_DEN = 16'd610;   // Denominator

// Higher precision: F_20/F_21
localparam [15:0] INV_PHI_NUM_HP = 16'd6765;
localparam [15:0] INV_PHI_DEN_HP = 16'd10946;
```

### 4.3 ψ-View as Bit Reversal

**Source:** `phi_mamba/unified_tensor.py:28-47`

The ψ (conjugate) view is implemented as bit-reversal:

```verilog
// Bit reversal for ψ-view computation
module bit_reverse_64(
    input  wire [63:0] x,
    output wire [63:0] y
);
    // Swap adjacent bits
    wire [63:0] s1 = ((x & 64'h5555555555555555) << 1) |
                     ((x & 64'hAAAAAAAAAAAAAAAA) >> 1);
    // Swap pairs
    wire [63:0] s2 = ((s1 & 64'h3333333333333333) << 2) |
                     ((s1 & 64'hCCCCCCCCCCCCCCCC) >> 2);
    // Swap nibbles
    wire [63:0] s3 = ((s2 & 64'h0F0F0F0F0F0F0F0F) << 4) |
                     ((s2 & 64'hF0F0F0F0F0F0F0F0) >> 4);
    // Swap bytes
    wire [63:0] s4 = ((s3 & 64'h00FF00FF00FF00FF) << 8) |
                     ((s3 & 64'hFF00FF00FF00FF00) >> 8);
    // Swap 16-bit words
    wire [63:0] s5 = ((s4 & 64'h0000FFFF0000FFFF) << 16) |
                     ((s4 & 64'hFFFF0000FFFF0000) >> 16);
    // Swap 32-bit words
    assign y = (s5 << 32) | (s5 >> 32);
endmodule
```

### 4.4 Interference Pattern

```verilog
// Standing wave interference = bits that agree in φ and ψ views
wire [63:0] phi_bits;
wire [63:0] psi_bits;  // bit-reversed phi_bits
wire [63:0] interference = phi_bits & psi_bits;
```

---

## 5. Zeckendorf Decomposition

### 5.1 Algorithm (Greedy)

**Source:** `phi_mamba/math_core.py:153-213`

```
ZECKENDORF_DECOMPOSE(n):
    IF n == 0: RETURN empty

    // Find largest Fibonacci <= n
    fibs = [1, 2, 3, 5, 8, 13, ...]  // Precomputed

    result_bits = 0
    remaining = n
    i = len(fibs) - 1

    WHILE i >= 0 AND remaining > 0:
        IF fibs[i] <= remaining:
            result_bits |= (1 << i)  // Set bit i
            remaining -= fibs[i]
            i -= 2  // Skip next (non-consecutive constraint)
        ELSE:
            i -= 1

    RETURN result_bits
```

### 5.2 Verilog Implementation

```verilog
module zeckendorf_encode #(
    parameter WIDTH = 64,
    parameter MAX_FIB_IDX = 63
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [WIDTH-1:0] value_in,
    output reg  [WIDTH-1:0] bits_out,
    output reg  done
);

// Fibonacci LUT
wire [WIDTH-1:0] fib_val;
fibonacci_lut #(.DEPTH(MAX_FIB_IDX+1), .WIDTH(WIDTH)) fib_lut (
    .index(current_idx),
    .fib_value(fib_val)
);

reg [WIDTH-1:0] remaining;
reg [6:0] current_idx;
reg [WIDTH-1:0] result_bits;

// State machine for greedy decomposition
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 0;
        bits_out <= 0;
    end else if (start) begin
        remaining <= value_in;
        current_idx <= MAX_FIB_IDX;
        result_bits <= 0;
        done <= 0;
    end else if (!done && current_idx > 0) begin
        if (fib_val <= remaining && fib_val > 0) begin
            result_bits <= result_bits | (1 << current_idx);
            remaining <= remaining - fib_val;
            current_idx <= (current_idx >= 2) ? current_idx - 2 : 0;
        end else begin
            current_idx <= current_idx - 1;
        end

        if (remaining == 0 || current_idx == 0) begin
            bits_out <= result_bits;
            done <= 1;
        end
    end
end
endmodule
```

---

## 6. Recommended Verilog Module Hierarchy

```
zordic_top
├── fibonacci_lut          (ROM, ~20-64 entries)
├── lucas_lut              (ROM, optional, ~20-64 entries)
├── cordic_engine          (CORDIC for trig operations)
│   └── atan_lut           (ROM, 16-32 entries)
├── zeckendorf_encoder     (Value → Zeckendorf bits)
├── cascade_normalize      (Resolve adjacent 1s)
├── xor_attention          (Hamming distance + OR combine)
│   └── popcount           (Parallel bit count)
├── bit_reverse            (φ ↔ ψ view conversion)
└── phase_memory           (Optional: phase-indexed storage)
```

### 6.1 Estimated Resource Usage (iCE40 or similar)

| Module | LUTs | FFs | Notes |
|--------|------|-----|-------|
| fibonacci_lut (20 entries) | ~80 | 0 | Combinational ROM |
| cascade_normalize | ~200 | ~70 | FSM + bit ops |
| popcount_64 | ~120 | 0 | Parallel tree |
| xor_attention (16 keys) | ~300 | ~100 | Parallel compare |
| zeckendorf_encode | ~150 | ~100 | FSM + subtract |
| bit_reverse_64 | ~100 | 0 | Pure wiring |
| **Total** | **~950** | **~270** | Under 1350 target |

---

## 7. Test Vectors

### 7.1 Cascade Test Cases

**Source:** `validate_zordic.py:63-68`, `unified-zeckendorf-cordic/core/src/cascade.rs:77-99`

```
Input (binary)  | Output (binary) | Description
----------------|-----------------|-------------
0b11            | 0b100           | Basic pair
0b111           | 0b1001          | Triple cascade
0b11011         | 0b100101        | Two pairs
0b1111111       | 0b1000101       | Dense 7-bit
0b10101010      | 0b10101010      | Already valid
0b110           | 0b1000          | Adjacent at pos 1,2
0b1100          | 0b10000         | Adjacent at pos 2,3
```

### 7.2 Zeckendorf Decomposition Test Cases

**Source:** `phi_mamba/math_core.py:173-178`

```
Value | Zeckendorf Sum        | Bit Pattern
------|----------------------|-------------
0     | (none)               | 0b0
1     | F_2 = 1              | 0b100
2     | F_3 = 2              | 0b1000
3     | F_2 + F_4 = 1 + 2    | 0b1100 → cascade → 0b10000
5     | F_5 = 5              | 0b100000
7     | F_5 + F_2 = 5 + 2    | 0b100100
17    | F_7 + F_4 + F_2      | 0b10010100
100   | F_11 + F_6 + F_4     | 0b10001010100
```

### 7.3 XOR Attention Test Cases

```
Query (bin)     | Key (bin)       | Hamming Dist | Score
----------------|-----------------|--------------|------
0b10101010      | 0b10101010      | 0            | 64
0b10101010      | 0b01010101      | 8            | 56
0b11111111      | 0b00000000      | 8            | 56
0b10000001      | 0b10000000      | 1            | 63
```

---

## 8. Critical Path Analysis

### 8.1 Cascade Single Iteration
```
Adjacent detect (AND, shift)  →  1 cycle
Find lowest bit (CTZ)         →  1-2 cycles (implementation dependent)
Apply cascade (AND, OR)       →  1 cycle
Total per iteration           →  ~3-4 cycles
```

### 8.2 Full Pipeline Options

**Option A: Fully Pipelined (High Throughput)**
- 1 cascade iteration per cycle
- Max 32 cycles for 64-bit
- Throughput: 1 value / 32 cycles

**Option B: FSM-Based (Low Area)**
- Variable cycles based on input
- Typical: 2-5 cycles for random data
- Area: ~40% of pipelined

**Option C: Hybrid (Unroll 4 iterations)**
- 4 cascade steps per cycle
- Max 8 cycles for 64-bit
- Good balance

---

## 9. Answers to Specific Questions

### Q1: Maximum Fibonacci index used?
**Answer:** 63 for full 64-bit, but 20-32 is practical for most uses.
- F_20 = 6,765 (13 bits)
- F_32 = 2,178,309 (22 bits)
- F_63 = 6,557,470,319,842 (43 bits)

### Q2: Cascade iterations worst-case?
**Answer:** N/2 for N-bit input.
- 32-bit: max 16 iterations
- 64-bit: max 32 iterations
- Typical: 2-5 iterations

### Q3: LUT access arbitration needed?
**Answer:** No. All LUTs are single-read, and access is sequential or indexed by bit position. Simple combinational ROM is sufficient.

### Q4: Core clock-to-clock operation?
**Answer:** Two options:
1. **One cascade step per cycle** (simple, predictable)
2. **Full normalization per cycle** (more LUTs, uses unrolled cascade)

### Q5: Any floating-point?
**Answer:** Zero. All implementations use:
- Fixed-point (scaled integers)
- Fibonacci ratios (e.g., 377/610 for 1/φ)
- Pure bit operations

### Q6: Test vectors available?
**Answer:** Yes. See Section 7 above, derived from:
- `validate_zordic.py`
- `unified-zeckendorf-cordic/core/src/cascade.rs` tests
- `phi_mamba/math_core.py` examples

---

## 10. Gaps and Clarifications Needed

### 10.1 Ambiguities Found

1. **Shell distance normalization**: Some implementations use raw Hamming distance, others normalize by max(|shells1|, |shells2|). Need to confirm which is canonical.

2. **Phase encoding**: `unified_tensor.py` uses 16-bit phase (0-65535 = 0-2π), but CORDIC uses 32-bit. Confirm target precision.

3. **Attention aggregation**: Some files use OR-combine, others use weighted accumulation. Which is authoritative?

4. **Energy decay rate**: Varies between implementations (right-shift 1 vs Fibonacci ratio). Confirm for hardware.

### 10.2 Missing Information

1. **Target clock frequency**: Not specified. Recommend designing for 100 MHz with retiming flexibility.

2. **Memory interface**: If LUTs exceed on-chip capacity, need SRAM/BRAM interface spec.

3. **Host communication**: SPI? UART? Need interface for Verilator/FPGA.

4. **Reset/initialization**: Warm start from saved state? Cold start only?

---

## 11. Quick Start: Minimal Implementation

For Verilator testing on Raspberry Pi 3B:

```verilog
// Minimal 20-bit, 10-iteration cascade
module cascade_mini(
    input  wire clk,
    input  wire [19:0] bits_in,
    output reg  [19:0] bits_out,
    input  wire start,
    output reg  done
);
    reg [19:0] bits;
    reg [3:0] iter;

    always @(posedge clk) begin
        if (start) begin
            bits <= bits_in;
            iter <= 0;
            done <= 0;
        end else if (!done) begin
            wire [19:0] adj = bits & (bits << 1);
            if (adj == 0) begin
                bits_out <= bits;
                done <= 1;
            end else begin
                // Find lowest set bit in adj
                reg [4:0] pos;
                pos = 0;
                while (pos < 19 && !adj[pos]) pos = pos + 1;
                // Cascade
                bits <= (bits & ~(20'h3 << pos)) | (20'h1 << (pos + 2));
                iter <= iter + 1;
                if (iter >= 10) begin
                    bits_out <= bits;
                    done <= 1;
                end
            end
        end
    end
endmodule
```

---

## 12. References

### Repository Files (Priority Order)
1. `validate_zordic.py` - Performance validation with test vectors
2. `phi_mamba/unified_tensor.py` - CUDA kernels (translate to Verilog)
3. `unified-zeckendorf-cordic/core/src/cascade.rs` - Rust reference
4. `phi_mamba/math_core.py` - Mathematical foundations
5. `phi_mamba/cordic.py` - CORDIC implementation
6. `docs/math_foundations.md` - Theory documentation

### OEIS Sequences
- A000045: Fibonacci numbers
- A000032: Lucas numbers
- A003714: Zeckendorf representations

---

**Document Version:** 1.0
**Generated:** 2024 (via repository scan)
**Target:** Verilator simulation on Raspberry Pi 3B, FPGA synthesis (Lattice iCE40)
