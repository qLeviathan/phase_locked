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

## The Four Primitives

| Primitive | Transformer Equiv | Implementation |
|-----------|-------------------|----------------|
| **ENCODE** | Embedding | token → Σφₙᵢ (Zeckendorf) |
| **CASCADE** | FFN | R1: φₐ+φₐ₊₁=φₐ₊₂, R2: 2φₐ=φₐ₊₁+φₐ₋₂ |
| **MERGE** | Residual | A ⊕ B = Normalize(A ⊞ B) |
| **CONTRACT** | Attention | ⟨Q,K⟩ = Σ L_{|a-b|} |

## Key Insight

**Lucas numbers ARE the attention weights.**

The contraction `⟨Q, K⟩ = Σᵢ Σⱼ L_{|aᵢ - bⱼ|}` falls out of the algebra.
No learned parameters. Geometry determines attention.

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

## Expected Output

```
═══════════════════════════════════════════════════════════════════
  φ-SUBSCRIPT CALCULUS - Zeckbit Cascade Tests
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
zeck_merge        |   ~50 |  ~10  | Superposition
------------------|-------|-------|------------------
TOTAL             |  ~650 | ~260  | Under 1350 target
```

## Inference Path

```
1. ENCODE tokens → shell occupancy states
2. Fold context via repeated MERGE (⊕)
3. CASCADE to canonical form
4. CONTRACT query against context keys
5. Lucas sums = attention weights
6. Nearest occupied cell = prediction
```

## What This Replaces

| Transformer | φ-mechanics | Why |
|-------------|-------------|-----|
| Embedding table | Zeckendorf encode | Address IS content |
| Q·K^T matmul | Lucas contraction | Integer, no divide |
| Softmax | Native shell ordering | No normalization needed |
| FFN layers | Cascade rewrite | Physics, not learned |
| Position encoding | Native | Index IS position |

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
