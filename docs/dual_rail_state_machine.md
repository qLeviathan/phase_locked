# φ-ψ Dual-Rail Sequence State Machine
## LUT-Based Architecture with Lucas State Tracking

**Core Principle:** We never compute φⁿ or ψⁿ. We track indices and Fibonacci/Lucas coefficient pairs.

---

## §1 The Dual-Coefficient Representation

Every power of φ decomposes into integer coefficients:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BINET DECOMPOSITION (INTEGER FORM)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   φⁿ = Fₙ · φ + Fₙ₋₁        (Fibonacci coefficients)                        │
│   ψⁿ = Fₙ · ψ + Fₙ₋₁        (Same coefficients, conjugate eigenvalue)       │
│                                                                              │
│   STANDING WAVE (sum of paths):                                             │
│   φⁿ + ψⁿ = Lₙ              (Lucas number - integer!)                       │
│                                                                              │
│   TRAVELING WAVE (difference of paths):                                      │
│   φⁿ - ψⁿ = Fₙ · √5         (Fibonacci × √5)                                │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   STATE REPRESENTATION:                                                      │
│                                                                              │
│       (n, Fₙ, Lₙ)  ←  Three integers encode complete φⁿ information        │
│                                                                              │
│       Fₙ: "Position" - where we are in the sequence                        │
│       Lₙ: "Velocity" - standing wave amplitude at this point               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §1.1 The F-L Coefficient Table

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FIBONACCI-LUCAS STATE TABLE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────┬──────┬──────┬──────────────────────────────────────────────────┐ │
│   │  n   │  Fₙ  │  Lₙ  │  Interpretation                                  │ │
│   ├──────┼──────┼──────┼──────────────────────────────────────────────────┤ │
│   │  0   │   0  │   2  │  Origin: no position, max constructive interf.  │ │
│   │  1   │   1  │   1  │  Unit step: position=1, velocity=1              │ │
│   │  2   │   1  │   3  │  φ² = φ+1                                        │ │
│   │  3   │   2  │   4  │                                                  │ │
│   │  4   │   3  │   7  │                                                  │ │
│   │  5   │   5  │  11  │                                                  │ │
│   │  6   │   8  │  18  │                                                  │ │
│   │  7   │  13  │  29  │                                                  │ │
│   │  8   │  21  │  47  │                                                  │ │
│   │  9   │  34  │  76  │                                                  │ │
│   │ 10   │  55  │ 123  │                                                  │ │
│   └──────┴──────┴──────┴──────────────────────────────────────────────────┘ │
│                                                                              │
│   RECURRENCES (both use same rule):                                         │
│       Fₙ = Fₙ₋₁ + Fₙ₋₂       with F₀=0, F₁=1                               │
│       Lₙ = Lₙ₋₁ + Lₙ₋₂       with L₀=2, L₁=1                               │
│                                                                              │
│   CROSS-RELATIONS:                                                           │
│       Lₙ = Fₙ₊₁ + Fₙ₋₁       (Lucas bridges adjacent Fibonacci)            │
│       Lₙ = 2Fₙ₊₁ - Fₙ        (alternative form)                            │
│       Fₙ · Lₙ = F₂ₙ          (product gives doubled index)                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §2 Dual-Rail State Machine

### §2.1 State Definition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE STRUCTURE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   S = (A_φ, A_ψ, F_acc, L_acc, τ_history)                                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   A_φ : uint64      Zeckendorf bits for φ-rail (position channel)  │   │
│   │   A_ψ : uint64      Zeckendorf bits for ψ-rail (velocity channel)  │   │
│   │                                                                     │   │
│   │   F_acc : int       Accumulated Fibonacci coefficient              │   │
│   │   L_acc : int       Accumulated Lucas coefficient                  │   │
│   │                                                                     │   │
│   │   τ_history : []    Cascade counts (cross-rail work quanta)        │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   The (F_acc, L_acc) pair tracks aggregate state across the sequence.       │
│   The (A_φ, A_ψ) pair tracks shell-level topology.                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §2.2 Rail Operators (Verlet Structure)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAIL OPERATORS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ╔═══════════════════════════════════════════════════════════════════════╗ │
│   ║  RAIL    │ EIGENVALUE │ SEQUENCE │ OPERATOR σ       │ CHANNEL        ║ │
│   ╠═══════════════════════════════════════════════════════════════════════╣ │
│   ║  φ-rail  │  φ ≈ 1.618 │    Fₙ    │ (a,b)→(a+b, a)   │ Position       ║ │
│   ║  ψ-rail  │  ψ ≈ -0.618│    Lₙ    │ (a,b)→(a+b, a)   │ Velocity       ║ │
│   ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                              │
│   FORWARD STEP σ:                                                           │
│   ───────────────                                                           │
│       (Fₙ₋₁, Fₙ₋₂) → (Fₙ₋₁ + Fₙ₋₂, Fₙ₋₁) = (Fₙ, Fₙ₋₁)                    │
│       (Lₙ₋₁, Lₙ₋₂) → (Lₙ₋₁ + Lₙ₋₂, Lₙ₋₁) = (Lₙ, Lₙ₋₁)                    │
│                                                                              │
│   BACKWARD STEP σ⁻¹:                                                        │
│   ─────────────────                                                         │
│       (Fₙ, Fₙ₋₁) → (Fₙ₋₁, Fₙ - Fₙ₋₁) = (Fₙ₋₁, Fₙ₋₂)                      │
│       (Lₙ, Lₙ₋₁) → (Lₙ₋₁, Lₙ - Lₙ₋₁) = (Lₙ₋₁, Lₙ₋₂)                      │
│                                                                              │
│   BOTH rails use the SAME recurrence - they differ only in initial          │
│   conditions: F starts (0,1), L starts (2,1).                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §2.3 Lucas as Velocity Channel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LUCAS STATE TRACKING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   WHY LUCAS FOR VELOCITY:                                                   │
│   ═══════════════════════                                                   │
│                                                                              │
│   Lₙ = φⁿ + ψⁿ   (standing wave: forward + backward interference)          │
│                                                                              │
│   At any state, Lucas tells us the TOTAL AMPLITUDE of the standing wave.   │
│   This is the "velocity" - how much energy is oscillating at this point.   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   L₀ = 2    Maximum constructive interference at origin            │   │
│   │   L₁ = 1    Unit velocity                                          │   │
│   │   Lₙ → φⁿ   As n→∞, ψⁿ→0, standing wave ≈ traveling wave          │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   STATE ANALYSIS via (F, L) pair:                                           │
│   ────────────────────────────────                                           │
│                                                                              │
│   ┌──────────────────┬─────────────────────────────────────────────────┐    │
│   │   Condition      │   Interpretation                                │    │
│   ├──────────────────┼─────────────────────────────────────────────────┤    │
│   │   L > 2F         │   Early sequence, strong constructive interf.   │    │
│   │   L ≈ φ·F        │   Asymptotic regime, stable propagation         │    │
│   │   L = F+1 + F-1  │   Lucas bridges: state is "between" shells      │    │
│   │   F·L = F₂ₙ      │   Doubling: combined state jumps 2 shells       │    │
│   └──────────────────┴─────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §3 Complete State Machine

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║              φ-ψ DUAL-RAIL STATE MACHINE WITH LUCAS TRACKING                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                         LUT BANK                                        │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   FIB[k]       : k → Fₖ                  (position at shell k)         │ ║
║  │   LUC[k]       : k → Lₖ                  (velocity at shell k)         │ ║
║  │   ZECK[n]      : n → bits                (value → topology)            │ ║
║  │   UNZECK[bits] : bits → n                (topology → value)            │ ║
║  │   POP[16b]     : bits → popcount         (shell count)                 │ ║
║  │   VOCAB[tok]   : token → rank            (corpus frequency)            │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                           INITIALIZATION                                ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   S₀ = {                                                                ║ ║
║  ║       A_φ   : ZECK[VOCAB[t₀]]    // φ-rail topology                    ║ ║
║  ║       A_ψ   : 0                   // ψ-rail topology                    ║ ║
║  ║       F_acc : FIB[max_shell(A_φ)] // Fibonacci state                   ║ ║
║  ║       L_acc : LUC[max_shell(A_φ)] // Lucas state                       ║ ║
║  ║       τ     : []                  // cascade history                    ║ ║
║  ║       pos   : 0                                                         ║ ║
║  ║   }                                                                     ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                         MAIN PROCESSING LOOP                            │ ║
║  │                       FOR each token t_i:                               │ ║
║  └──────────────────────────────────┬──────────────────────────────────────┘ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 1: TOKENIZE + RAIL SELECT                       ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   bits_i = ZECK[VOCAB[t_i]]                                             ║ ║
║  ║                                                                         ║ ║
║  ║   VERLET ALTERNATION:                                                   ║ ║
║  ║       pos % 2 == 0  →  target = A_φ   (position update)                ║ ║
║  ║       pos % 2 == 1  →  target = A_ψ   (velocity update)                ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 2: ACCUMULATE                                   ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   temp = target | bits_i          // Zeckendorf addition (may violate) ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 3: CASCADE NORMALIZATION                        ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   τ_local = 0                                                          ║ ║
║  ║                                                                         ║ ║
║  ║   WHILE has_adjacent_11(temp):                                          ║ ║
║  ║   ┌───────────────────────────────────────────────────────────────────┐║ ║
║  ║   │  FOR k = MAX_SHELL down to 1:      // Priority: high shells first │║ ║
║  ║   │      IF temp[k] AND temp[k-1]:                                    │║ ║
║  ║   │                                                                   │║ ║
║  ║   │          // CASCADE: Fₖ + Fₖ₋₁ = Fₖ₊₁                            │║ ║
║  ║   │          temp[k]   ← 0                                            │║ ║
║  ║   │          temp[k-1] ← 0                                            │║ ║
║  ║   │          temp[k+1] ← 1                                            │║ ║
║  ║   │                                                                   │║ ║
║  ║   │          τ_local++                                                │║ ║
║  ║   │          BREAK   // rescan from top                               │║ ║
║  ║   │                                                                   │║ ║
║  ║   └───────────────────────────────────────────────────────────────────┘║ ║
║  ║                                                                         ║ ║
║  ║   target ← temp                                                        ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 4: UPDATE LUCAS STATE                           ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   Compute aggregate (F, L) from both rails:                             ║ ║
║  ║                                                                         ║ ║
║  ║   ┌───────────────────────────────────────────────────────────────────┐║ ║
║  ║   │                                                                   │║ ║
║  ║   │   // Extract active shells from both rails                        │║ ║
║  ║   │   shells_φ = active_indices(A_φ)   // e.g., {7, 4, 2}            │║ ║
║  ║   │   shells_ψ = active_indices(A_ψ)   // e.g., {5, 3}               │║ ║
║  ║   │                                                                   │║ ║
║  ║   │   // Position: sum of Fibonacci values                            │║ ║
║  ║   │   F_acc = Σₖ∈shells_φ FIB[k] + Σₖ∈shells_ψ FIB[k]                │║ ║
║  ║   │                                                                   │║ ║
║  ║   │   // Velocity: sum of Lucas values                                │║ ║
║  ║   │   L_acc = Σₖ∈shells_φ LUC[k] + Σₖ∈shells_ψ LUC[k]                │║ ║
║  ║   │                                                                   │║ ║
║  ║   └───────────────────────────────────────────────────────────────────┘║ ║
║  ║                                                                         ║ ║
║  ║   STATE ANALYSIS from (F_acc, L_acc):                                   ║ ║
║  ║                                                                         ║ ║
║  ║   ┌───────────────────────────────────────────────────────────────────┐║ ║
║  ║   │                                                                   │║ ║
║  ║   │   ratio = L_acc / F_acc                                           │║ ║
║  ║   │                                                                   │║ ║
║  ║   │   ratio > φ   →  constructive interference dominant               │║ ║
║  ║   │   ratio ≈ φ   →  asymptotic stable state                          │║ ║
║  ║   │   ratio < φ   →  destructive interference (rare)                  │║ ║
║  ║   │                                                                   │║ ║
║  ║   │   momentum = L_acc - φ·F_acc   (deviation from equilibrium)      │║ ║
║  ║   │                                                                   │║ ║
║  ║   └───────────────────────────────────────────────────────────────────┘║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 5: EMIT STATE                                   ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   state[pos] = {                                                        ║ ║
║  ║       A_φ     : A_φ,                                                    ║ ║
║  ║       A_ψ     : A_ψ,                                                    ║ ║
║  ║       F       : F_acc,           // Position                            ║ ║
║  ║       L       : L_acc,           // Velocity                            ║ ║
║  ║       τ       : τ_local,         // Work performed this step            ║ ║
║  ║       ratio   : L_acc / F_acc    // Phase state indicator               ║ ║
║  ║   }                                                                     ║ ║
║  ║                                                                         ║ ║
║  ║   τ_history.append(τ_local)                                             ║ ║
║  ║   pos++                                                                 ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §4 Lucas-Based State Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STATE ANALYSIS FUNCTIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Given state S = (A_φ, A_ψ, F, L, τ):                                      │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   1. PHASE RATIO                                                             │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       ρ = L / F                                                             │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────────┐   │
│       │   ρ → φ ≈ 1.618   as sequence progresses (asymptotic)          │   │
│       │   ρ > φ           early in sequence, more constructive interf. │   │
│       │   ρ = 2/0 = ∞     at origin (L₀=2, F₀=0)                       │   │
│       └─────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   2. STANDING WAVE AMPLITUDE                                                 │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       Standing wave at shell k:  Lₖ = φᵏ + ψᵏ                              │
│                                                                              │
│       Total standing wave:  L_total = Σₖ∈active Lₖ                         │
│                                                                              │
│       This measures total constructive interference across all shells.      │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   3. SHELL ENERGY DISTRIBUTION                                               │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       For each active shell k:                                               │
│                                                                              │
│           position_contrib[k] = FIB[k] / F_acc                              │
│           velocity_contrib[k] = LUC[k] / L_acc                              │
│                                                                              │
│       High-k shells dominate (exponential growth of F, L)                   │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   4. MOMENTUM (Deviation from Equilibrium)                                   │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       At equilibrium: L = φ·F  (standing wave = φ × position)              │
│                                                                              │
│       Momentum:  μ = L - φ·F                                                │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────────┐   │
│       │   μ > 0   →  excess standing wave (accelerating)               │   │
│       │   μ = 0   →  equilibrium (coasting)                             │   │
│       │   μ < 0   →  deficit (decelerating) - rare in forward pass     │   │
│       └─────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   5. CASCADE WORK (τ Analysis)                                               │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       τ_local: cascades this step                                           │
│       τ_total = Σ τ_history: cumulative work                                │
│       τ_rate = τ_total / pos: average work per token                        │
│                                                                              │
│       High τ_local indicates:                                                │
│         • Token created many shell collisions                               │
│         • Information-dense addition                                         │
│         • Non-local state reorganization                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §5 Cross-Rail Operations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-RAIL LUCAS OPERATIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   The φ and ψ rails interact via Lucas identities:                          │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   PRODUCT IDENTITY (Shell Doubling)                                          │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       Fₙ · Lₙ = F₂ₙ                                                         │
│                                                                              │
│       Multiplying position by velocity DOUBLES the shell index.             │
│       This is a "boost" operation - jump forward 2n shells.                 │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   SUM IDENTITY (Adjacent Bridging)                                           │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       Lₙ = Fₙ₊₁ + Fₙ₋₁                                                      │
│                                                                              │
│       Lucas at n equals sum of Fibonacci at n±1.                            │
│       Velocity "sees" adjacent position shells.                              │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   DIFFERENCE IDENTITY                                                        │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       Lₙ - Fₙ = 2·Fₙ₋₁                                                      │
│                                                                              │
│       Difference gives twice the previous position.                          │
│       Shadow memory via subtraction.                                         │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   NEGATIVE INDEX EXTENSION                                                   │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       F₋ₙ = (-1)ⁿ⁺¹ · Fₙ                                                    │
│       L₋ₙ = (-1)ⁿ · Lₙ                                                      │
│                                                                              │
│       Negative indices = signed versions of positive indices.                │
│       ψ-rail naturally handles backward traversal.                           │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   CONVOLUTION (Sequence Combination)                                         │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       Fₘ₊ₙ = Fₘ·Fₙ₊₁ + Fₘ₋₁·Fₙ                                              │
│       Lₘ₊ₙ = (Lₘ·Lₙ + 5·Fₘ·Fₙ) / 2                                          │
│                                                                              │
│       Combining two states at indices m and n.                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §6 Attention via Shell Distance

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      ATTENTION COMPUTATION                                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  For positions i, j with states Sᵢ, Sⱼ:                                      ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   SHELL DISTANCE:                                                       │ ║
║  │   ───────────────                                                       │ ║
║  │                                                                         │ ║
║  │       d_φ = POPCOUNT[ A_φⁱ ⊕ A_φʲ ]    // Position disagreement        │ ║
║  │       d_ψ = POPCOUNT[ A_ψⁱ ⊕ A_ψʲ ]    // Velocity disagreement        │ ║
║  │       d   = d_φ + d_ψ                   // Total shell distance         │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   LUCAS-WEIGHTED ATTENTION:                                             │ ║
║  │   ─────────────────────────                                             │ ║
║  │                                                                         │ ║
║  │       // Shell distance → Lucas decay                                   │ ║
║  │       weight = LUC[MAX_SHELL - d]   (higher d = lower weight)          │ ║
║  │                                                                         │ ║
║  │       // Or equivalently via Fibonacci:                                 │ ║
║  │       weight = FIB[MAX_SHELL - d + 1] + FIB[MAX_SHELL - d - 1]         │ ║
║  │                                                                         │ ║
║  │   INTERPRETATION:                                                       │ ║
║  │       d = 0:  Perfect shell match → maximum attention (L_max)          │ ║
║  │       d = 1:  One shell differs → attention = L_{max-1}                │ ║
║  │       d = k:  k shells differ → attention decays as L_{max-k}          │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   PHASE-AWARE ATTENTION:                                                │ ║
║  │   ──────────────────────                                                │ ║
║  │                                                                         │ ║
║  │       // Include Lucas ratio in attention                               │ ║
║  │       phase_match = |ρᵢ - ρⱼ|   where ρ = L/F                          │ ║
║  │                                                                         │ ║
║  │       α_ij = weight × (1 / (1 + phase_match)) × CAUSAL[i,j]            │ ║
║  │                                                                         │ ║
║  │   States with similar L/F ratios attend more strongly.                  │ ║
║  │   This captures "phase alignment" between positions.                    │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §7 Generation (Next Token Selection)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        TOKEN GENERATION                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Current state: S = (A_φ, A_ψ, F, L, τ_history)                              ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   FOR each candidate token t ∈ vocabulary:                              │ ║
║  │                                                                         │ ║
║  │       bits_t = ZECK[VOCAB[t]]                                           │ ║
║  │                                                                         │ ║
║  │       // Simulate adding this token                                     │ ║
║  │       S' = APPLY_CASCADE(S, bits_t)                                     │ ║
║  │                                                                         │ ║
║  │       // Extract new state properties                                   │ ║
║  │       τ_t     = cascade_count(S → S')                                  │ ║
║  │       F'      = new Fibonacci accumulator                               │ ║
║  │       L'      = new Lucas accumulator                                   │ ║
║  │       ρ'      = L' / F'                                                 │ ║
║  │                                                                         │ ║
║  │       // Attention to context                                           │ ║
║  │       attn_t  = Σⱼ α(S', state[j])                                      │ ║
║  │                                                                         │ ║
║  │       // SCORE FUNCTION                                                 │ ║
║  │       score_t = f(τ_t, attn_t, ρ', freq[t])                            │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   SCORING COMPONENTS:                                                   │ ║
║  │   ───────────────────                                                   │ ║
║  │                                                                         │ ║
║  │   1. CASCADE COST (τ_t)                                                 │ ║
║  │      Lower τ = smoother transition = higher score                       │ ║
║  │      High τ = significant reorganization = lower score (or bonus?)     │ ║
║  │                                                                         │ ║
║  │   2. ATTENTION FIT (attn_t)                                             │ ║
║  │      Higher attention to context = better fit = higher score            │ ║
║  │                                                                         │ ║
║  │   3. PHASE CONTINUITY (ρ')                                              │ ║
║  │      |ρ' - ρ| small = smooth phase evolution = higher score            │ ║
║  │      Large jumps in L/F ratio = discontinuity = lower score            │ ║
║  │                                                                         │ ║
║  │   4. CORPUS FREQUENCY (freq[t])                                         │ ║
║  │      Prior probability from training corpus                             │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  SELECTION:                                                                   ║
║  ──────────                                                                   ║
║      next_token = argmax_t { score_t }                                       ║
║                                                                               ║
║      // Or temperature sampling:                                              ║
║      P(t) = exp(score_t / T) / Σ exp(score / T)                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §8 State Reconstruction (Reverse Pass)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEQUENCE RECONSTRUCTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Given final state S_final = (A_φ, A_ψ, F, L) and τ_history:               │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   INVERSE CASCADE                                                            │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   Forward cascade:   Fₖ + Fₖ₋₁ → Fₖ₊₁   (two shells → one shell)          │
│   Inverse cascade:   Fₖ₊₁ → Fₖ + Fₖ₋₁   (one shell → two shells)          │
│                                                                              │
│   τ_history tells us exactly how many inverse cascades to apply             │
│   at each step, working backwards.                                           │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   LUCAS-GUIDED RECONSTRUCTION                                                │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   At each reverse step:                                                      │
│                                                                              │
│   1. Current (F, L) known                                                   │
│   2. Apply τ[pos] inverse cascades to get pre-cascade state                 │
│   3. Use Verlet alternation to know which rail was modified                 │
│   4. Extract token bits from rail difference                                 │
│   5. Look up token from UNZECK[bits]                                        │
│                                                                              │
│   The Lucas channel provides redundancy:                                     │
│       Expected: Lₙ = Fₙ₊₁ + Fₙ₋₁                                            │
│       If mismatch: reconstruction error detected                             │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   SHADOW MEMORY (Δ Operators)                                                │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       Δ_F = Fₙ - Fₙ₋₂ = Fₙ₋₁      (position shadow)                        │
│       Δ_L = Lₙ - Lₙ₋₂             (velocity shadow)                         │
│                                                                              │
│   Shadow memory encodes previous state via subtraction only.                 │
│   No additional storage needed - it's implicit in (F, L).                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §9 Complete Data Flow

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          DATA FLOW SUMMARY                                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║                           INPUT TOKENS                                        ║
║                               │                                               ║
║                               ▼                                               ║
║                     ┌─────────────────┐                                      ║
║                     │  VOCAB → rank   │                                      ║
║                     └────────┬────────┘                                      ║
║                              │                                               ║
║                              ▼                                               ║
║                     ┌─────────────────┐                                      ║
║                     │  ZECK → bits    │                                      ║
║                     └────────┬────────┘                                      ║
║                              │                                               ║
║              ┌───────────────┴───────────────┐                               ║
║              ▼                               ▼                               ║
║      ┌───────────────┐               ┌───────────────┐                       ║
║      │    φ-RAIL     │               │    ψ-RAIL     │                       ║
║      │  (Position)   │               │  (Velocity)   │                       ║
║      │               │               │               │                       ║
║      │   A_φ bits    │               │   A_ψ bits    │                       ║
║      │   F_contrib   │               │   L_contrib   │                       ║
║      └───────┬───────┘               └───────┬───────┘                       ║
║              │                               │                               ║
║              └───────────────┬───────────────┘                               ║
║                              │                                               ║
║                              ▼                                               ║
║                     ┌─────────────────┐                                      ║
║                     │    CASCADE      │                                      ║
║                     │  NORMALIZATION  │                                      ║
║                     │    (τ count)    │                                      ║
║                     └────────┬────────┘                                      ║
║                              │                                               ║
║                              ▼                                               ║
║                     ┌─────────────────┐                                      ║
║                     │  LUCAS STATE    │                                      ║
║                     │   COMPUTATION   │                                      ║
║                     │                 │                                      ║
║                     │  F_acc = Σ Fₖ   │                                      ║
║                     │  L_acc = Σ Lₖ   │                                      ║
║                     │  ρ = L/F        │                                      ║
║                     └────────┬────────┘                                      ║
║                              │                                               ║
║              ┌───────────────┴───────────────┐                               ║
║              ▼                               ▼                               ║
║      ┌───────────────┐               ┌───────────────┐                       ║
║      │   ATTENTION   │               │  GENERATION   │                       ║
║      │               │               │               │                       ║
║      │  XOR+POPCOUNT │               │  Score each   │                       ║
║      │  Lucas weight │               │  candidate    │                       ║
║      │  Phase match  │               │  via (τ,α,ρ)  │                       ║
║      └───────────────┘               └───────┬───────┘                       ║
║                                              │                               ║
║                                              ▼                               ║
║                                      OUTPUT TOKEN                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §10 Key Identities Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FIBONACCI-LUCAS IDENTITIES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────┬────────────────────────────────────────────┐   │
│  │       Identity          │              Use Case                      │   │
│  ├─────────────────────────┼────────────────────────────────────────────┤   │
│  │  Lₙ = Fₙ₊₁ + Fₙ₋₁       │  Lucas from adjacent Fibonacci            │   │
│  │  Lₙ = 2Fₙ₊₁ - Fₙ        │  Alternative Lucas computation            │   │
│  │  Fₙ · Lₙ = F₂ₙ          │  Shell doubling (boost operation)         │   │
│  │  Lₙ² = 5Fₙ² + 4(-1)ⁿ    │  Structural identity (no runtime check)  │   │
│  │  Fₘ₊ₙ = Fₘ·Fₙ₊₁+Fₘ₋₁·Fₙ │  Combining indices (convolution)         │   │
│  │  Lₘ₊ₙ = ½(Lₘ·Lₙ+5Fₘ·Fₙ) │  Lucas convolution                        │   │
│  │  L₂ₙ = Lₙ² - 2(-1)ⁿ     │  Lucas doubling                           │   │
│  │  F₂ₙ = Fₙ · Lₙ          │  Fibonacci doubling                       │   │
│  │  Lₙ - Fₙ = 2Fₙ₋₁        │  Shadow memory access                     │   │
│  │  Lₙ + Fₙ = 2Fₙ₊₁        │  Forward prediction                       │   │
│  │  F₋ₙ = (-1)ⁿ⁺¹ Fₙ       │  Negative index (ψ-rail)                  │   │
│  │  L₋ₙ = (-1)ⁿ Lₙ         │  Negative Lucas                           │   │
│  └─────────────────────────┴────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §11 Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE SUMMARY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STATE = (A_φ, A_ψ, F, L, τ)                                               │
│                                                                              │
│       A_φ : Zeckendorf bits on position rail                                │
│       A_ψ : Zeckendorf bits on velocity rail                                │
│       F   : Accumulated Fibonacci (position)                                │
│       L   : Accumulated Lucas (velocity / standing wave)                    │
│       τ   : Cascade history (work performed)                                │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   CORE OPERATIONS:                                                           │
│                                                                              │
│       1. TOKEN → Zeckendorf bits via LUT                                    │
│       2. Bits accumulate on alternating φ/ψ rails (Verlet)                  │
│       3. Priority cascade normalizes illegal states                          │
│       4. Lucas state (L) tracks standing wave amplitude                     │
│       5. Attention via XOR + POPCOUNT on shell bits                         │
│       6. Generation scores candidates by (τ, attention, L/F ratio)          │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   LUCAS PROVIDES:                                                            │
│                                                                              │
│       • Velocity channel complementing Fibonacci position                   │
│       • Standing wave amplitude: Lₙ = φⁿ + ψⁿ                               │
│       • Phase state indicator: ρ = L/F → φ asymptotically                  │
│       • Shadow memory: Lₙ - Fₙ = 2Fₙ₋₁                                      │
│       • Shell doubling: Fₙ · Lₙ = F₂ₙ                                       │
│       • Reconstruction redundancy via Lₙ = Fₙ₊₁ + Fₙ₋₁                      │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   ALL INTEGER ARITHMETIC.  NO FLOATING POINT.  DETERMINISTIC.               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
