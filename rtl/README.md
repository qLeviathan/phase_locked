# Zeckbit Cascade RTL

Pure rewrite physics in synthesizable Verilog. Target: Verilator on Raspberry Pi 3B.

## Core Insight

**This is NOT binary arithmetic.** The "bit" is Fibonacci-shell occupancy with hard constraints and rewrite physics. Binary intuition will mislead you.

## The Primitives

### 1. Shell Occupancy (Zeckbit)
```
Z[i] ∈ {0,1} = occupancy of shell i (Fibonacci index)
```

### 2. Validity Constraint
```
No adjacency: Z[i]=1 ⇒ Z[i+1]=0 and Z[i-1]=0
```

### 3. Rewrite Rules

**R1 (Adjacency Merge):** Two adjacent shells collapse upward
```
Pattern: ... 1 1 ... at (i, i+1)
Rewrite: ... 0 0 1 ... at (i, i+1, i+2)
```
This is: F_k + F_{k+1} = F_{k+2}

**R2 (Split Overflow):** Double-occupancy splits
```
Pattern: 2 at i
Rewrite: 1 at (i+1) + 1 at (i-2)
```
This is: 2·F_k = F_{k+1} + F_{k-2}

### 4. The "XOR" Operator (It's NOT Boolean XOR)

```
⊕ ≡ Superposition + Cascade
Z = Normalize(A ⊞ B)
```

Where `⊞` is raw shell addition, and `Normalize` applies R1/R2 until stable.

**Properties:**
- Commutative: A ⊕ B = B ⊕ A
- Deterministic
- Canonical output
- **NOT bitwise** (shells are coupled by Fibonacci identities)

## The Key Output: Cascade Count

The number of rewrite steps (cascade_count) **IS** the Z[φ] norm:
- More cascades = more "work" to reach canonical form
- This measures distance on the discrete hyperbola
- Higher cascade count = closer shells = higher "attention weight"

```
N(a + bφ) = (a + bφ)(a + bψ) = a² + ab - b²
```

The shell metric `L² - 5F² = 4(-1)ⁿ` is the same thing viewed algebraically.

## Modules

| Module | Function |
|--------|----------|
| `zeck_cascade.v` | Rewrite engine (R1 + R2) |
| `zeck_encode.v` | Integer → Zeckbits (greedy decomposition) |
| `zeck_merge.v` | Superposition + Cascade (true "XOR") |
| `zeck_top.v` | Top-level wrapper |

## Build & Run

```bash
# On Raspberry Pi 3B (install Verilator first)
sudo apt install verilator

# Build
make

# Run demo
make run

# With waveform
make wave
```

## Test Vectors

### Cascade (R1)
| Input | Output | Cascades |
|-------|--------|----------|
| `0b11` | `0b100` | 1 |
| `0b111` | `0b1001` | 2 |
| `0b11011` | `0b100101` | 2 |
| `0b1111111` | `0b1000101` | 4 |

### Valid (No Cascade)
| Input | Output | Cascades |
|-------|--------|----------|
| `0b10101010` | `0b10101010` | 0 |
| `0b10010010` | `0b10010010` | 0 |

## Why This Works for Language Models

1. **Context = Shell Occupancy**: Each token activates certain shells
2. **Merge = Interference**: Combining tokens creates violations
3. **Cascade = Resolution**: Physics determines the stable state
4. **Norm = Attention**: Cascade count measures semantic distance

The cascade count during addition IS computing the Z[φ] norm operationally.

## No Floating Point

All operations are:
- Integer addition (for superposition)
- Bit manipulation (for rule detection)
- State machine (for rewrite engine)

The Fibonacci ratios (e.g., 377/610 ≈ 1/φ) are used only if you need φ approximation, but the core cascade needs no such thing.

## Resource Estimate (iCE40-class FPGA)

| Module | LUTs | FFs |
|--------|------|-----|
| zeck_cascade (32-bit) | ~200 | ~70 |
| zeck_encode (32-bit) | ~150 | ~100 |
| zeck_merge (32-bit) | ~50 | ~10 |
| **Total** | **~400** | **~180** |

Well under the 1350 LUT target.
