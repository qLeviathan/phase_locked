# φ-Subscript Calculus RTL

Pure rewrite physics for FPGA/Verilator. No Boolean XOR. No floating point.

## Target Hardware
- **Immediate:** Verilator on Raspberry Pi 3B / Kano
- **Next:** iCE40 or Lattice ECP5 FPGA

## The φ-Subscript Notation

```
φₙ ≡ φⁿ           (n ∈ ℤ)
φₐφᵦ = φₐ₊ᵦ       (product = index sum)
φ̄ₙ = (-1)ⁿφ₋ₙ    (conjugate)
```

**Contraction (inner product):**
```
⟨φₐ, φᵦ⟩ = φₐφ̄ᵦ + φ̄ₐφᵦ = L_{a-b}    (Lucas number!)
```

**Norm:**
```
‖φₙ‖² = φₙφ̄ₙ = (-1)ⁿ
```

## The Seven Primitives

| Primitive | Transformer Equiv | Implementation |
|-----------|-------------------|----------------|
| **ENCODE** | Embedding | token → Σφₙᵢ (Zeckendorf) |
| **CASCADE** | FFN | R1: φₐ+φₐ₊₁=φₐ₊₂, R2: 2φₐ=φₐ₊₁+φₐ₋₂ |
| **MERGE** | Residual | A ⊕ B = Normalize(A ⊞ B) |
| **CONTRACT** | Attention | ⟨Q,K⟩ = Σ L_{|a-b|} |
| **CONTEXT** | RNN state | C_{n+1} = C_n ⊕ token_n |
| **DECODE** | Unembedding | Σφₙᵢ → integer |
| **INFER** | Forward pass | Complete FSM |

## Key Insight

**Lucas numbers ARE the attention weights.**

The contraction `⟨Q, K⟩ = Σᵢ Σⱼ L_{|aᵢ - bⱼ|}` falls out of the algebra.
No learned parameters. Geometry determines attention.

**Cascade count IS the Z[φ] norm.**

The number of rewrite steps during merge IS computing distance on the hyperbola.
More cascades = more work = shells farther apart = less similar.

## Quick Start on Raspberry Pi / Kano

```bash
# Clone and enter
cd phase_locked/rtl

# Install Verilator (one-time setup)
sudo apt update && sudo apt install -y verilator build-essential

# Build
make

# Run tests
make run

# See benchmark
make benchmark
```

## Context Window Dynamics

The context state evolves through the recurrence:

```
C_{n+1} = C_n ⊕ token_n
```

Where ⊕ is superposition + cascade (the MERGE operator).

The cascade_count during each fold IS the attention weight for that token.
Tokens that cause more cascades are more "resonant" with context.

**Usage:**
```verilog
// Clear context
context_clear = 1;
@(posedge clk);
context_clear = 0;

// Fold each token (pre-encoded as Zeckbits)
for each token:
    context_token_zeck = encoded_token;
    context_fold_start = 1;
    @(posedge clk);
    context_fold_start = 0;
    wait(context_fold_done);
    // context_state now contains accumulated context
    // context_fold_norm is the attention weight for this token
```

## Inference FSM

The complete inference pipeline:

```
IDLE → ENCODE → FOLD → SNAP → DECODE → OUTPUT
```

| State | Description |
|-------|-------------|
| IDLE | Wait for start signal |
| ENCODE | Convert token to Zeckbits |
| FOLD | Merge token into context (C = C ⊕ token) |
| SNAP | Find highest occupied shell |
| DECODE | Convert context to prediction integer |
| OUTPUT | Present results |

**Usage:**
```verilog
// Reset context
infer_reset_context = 1;
@(posedge clk);
infer_reset_context = 0;

// Process each token
for each token:
    infer_token_in = token_value;
    infer_start = 1;
    @(posedge clk);
    infer_start = 0;
    wait(infer_done);
    // infer_context_out = current context state
    // infer_prediction_idx = highest shell index
    // infer_prediction_val = decoded prediction
    // infer_step_cascades = attention weight for this step
    // infer_attention_sum = cumulative attention
```

## Expected Output

```
═══════════════════════════════════════════════════════════════════
  φ-SUBSCRIPT CALCULUS - Complete Test Suite
  Target: Raspberry Pi / Kano (Verilator)
═══════════════════════════════════════════════════════════════════

━━━ TEST 1: Zeckendorf Encoding ━━━
    token → Σφₙᵢ (non-adjacent Fibonacci indices)

    1 → φ_2
    5 → φ_5
   17 → φ_2 + φ_4 + φ_7

━━━ TEST 2: Cascade Normalization ━━━
    R1: φₐ + φₐ₊₁ = φₐ₊₂ (adjacent merge)

  0000000011 → 0000000100 (cascades=1) φ_2 + φ_3 → φ_4
  0000000111 → 0000001001 (cascades=2) φ_2 + φ_3 + φ_4 → φ_2 + φ_5

━━━ TEST 3: Lucas Contraction (Attention) ━━━
    ⟨Q, K⟩ = Σᵢ Σⱼ L_{|aᵢ - bⱼ|}

  ⟨φ_2, φ_2⟩ = 2  (L_0 = 2)
  ⟨φ_2, φ_3⟩ = 1  (L_1 = 1)
  ⟨φ_2, φ_4⟩ = 3  (L_2 = 3)

━━━ TEST 5: Context Window Dynamics ━━━
    C_{n+1} = C_n ⊕ token_n (recurrence)

  Starting with empty context (C = 0)

  Fold token 1 (φ_2)
    → Context: φ_2
    Cascades: 0, Seq length: 1, Total cascades: 0

━━━ TEST 6: Complete Inference FSM ━━━
    ENCODE → FOLD → SNAP → DECODE

  Token 5:
    Context: φ_5
    Prediction idx: 3 (shell φ_5)
    Prediction val: 5
    Step cascades: 0, Attention sum: 0
```

## Rewrite Rules (The Physics)

**R1 (Adjacency Merge):** Two adjacent shells collapse upward
```
φₐ + φₐ₊₁ = φₐ₊₂
```
This is F_k + F_{k+1} = F_{k+2}

**R2 (Split Overflow):** Double-occupancy splits
```
2φₐ = φₐ₊₁ + φₐ₋₂
```
This is 2·F_k = F_{k+1} + F_{k-2}

## Files

| File | Description |
|------|-------------|
| `zeck_cascade.v` | Rewrite engine (R1 + R2) |
| `zeck_encode.v` | Greedy Zeckendorf decomposition |
| `zeck_merge.v` | Superposition + cascade |
| `lucas_lut.v` | Lucas number ROM |
| `phi_contract.v` | Attention via contraction |
| `phi_context.v` | Context window dynamics |
| `phi_decode.v` | Zeckendorf to integer |
| `phi_infer.v` | Complete inference FSM |
| `zeck_top.v` | Top-level wrapper |
| `sim/tb_zeck.cpp` | Verilator testbench |
| `setup_pi.sh` | Install script for Pi/Kano |
| `benchmark.sh` | Throughput measurement |

## Resource Estimate

```
Module            | LUTs  | FFs   | Notes
------------------|-------|-------|------------------
zeck_cascade (32) |  ~200 |  ~70  | Rewrite engine R1+R2
zeck_encode (32)  |  ~150 | ~100  | Greedy Zeckendorf
lucas_lut (32)    |  ~100 |    0  | ROM table
phi_contract      |  ~150 |  ~80  | Attention accumulator
phi_context       |  ~100 |  ~50  | Recurrence state
phi_decode        |  ~100 |  ~50  | Zeck → integer
phi_infer         |  ~200 | ~150  | Inference FSM
zeck_merge        |   ~50 |  ~10  | Superposition
------------------|-------|-------|------------------
TOTAL             | ~1050 | ~510  | Under 1350 target
```

## Inference Path

```
1. ENCODE tokens → shell occupancy states
2. Fold context via repeated MERGE (⊕)
   → cascade_count IS the attention weight
3. CASCADE to canonical form
4. SNAP to highest occupied shell
5. DECODE back to integer prediction
```

## What This Replaces

| Transformer | φ-mechanics | Why |
|-------------|-------------|-----|
| Embedding table | Zeckendorf encode | Address IS content |
| Q·K^T matmul | Lucas contraction | Integer, no divide |
| Softmax | Native shell ordering | No normalization needed |
| FFN layers | Cascade rewrite | Physics, not learned |
| Position encoding | Native | Index IS position |
| RNN state | Context folding | C_{n+1} = C_n ⊕ token_n |

## Lucas Table

```
L_0=2   L_1=1   L_2=3   L_3=4   L_4=7   L_5=11  L_6=18  L_7=29
L_8=47  L_9=76  L_10=123 ...
```

Higher shell distance = smaller Lucas number = less attention.
This IS the attention decay curve, emerging from φ·ψ = -1.

## No Floating Point

All operations are:
- Integer addition (for superposition)
- Bit manipulation (for rule detection)
- State machine (for rewrite engine)
- Integer LUT lookup (for Lucas)

The only approximation needed is if you want to convert back to decimal,
but the cascade itself is exact integer arithmetic.

## Benchmark Targets

| Platform | Expected Throughput |
|----------|---------------------|
| Verilator on Pi 3B | ~100K ops/sec |
| iCE40 at 12MHz | ~10M ops/sec |
| ECP5 at 50MHz | ~40M ops/sec |
