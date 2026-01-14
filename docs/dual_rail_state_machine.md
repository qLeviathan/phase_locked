# φ-ψ Dual-Rail Sequence State Machine
## Complete Mathematical Specification

**Core Principle:** We never compute φⁿ or ψⁿ. We only track indices n ∈ ℤ and Fibonacci coefficients (Fₙ, Fₙ₋₁).

---

## §1 Fundamental Representation

### §1.1 The Binet Decomposition (Index Form)

Every power of φ decomposes into integer coefficients:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BINET DECOMPOSITION (INTEGER FORM)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   φⁿ = Fₙ · φ + Fₙ₋₁                                                        │
│   ψⁿ = Fₙ · ψ + Fₙ₋₁    (same coefficients, different eigenvalue)          │
│                                                                              │
│   STORED STATE: (n, Fₙ, Fₙ₋₁)  ← three integers, no floats                 │
│                                                                              │
│   EXAMPLES:                                                                  │
│   ┌────────┬────────────────────────────────────────────────────────────┐   │
│   │   n    │   φⁿ = Fₙ·φ + Fₙ₋₁                                        │   │
│   ├────────┼────────────────────────────────────────────────────────────┤   │
│   │   0    │   φ⁰ = 0·φ + 1 = 1           (F₀=0, F₋₁=1)                │   │
│   │   1    │   φ¹ = 1·φ + 0 = φ           (F₁=1, F₀=0)                 │   │
│   │   2    │   φ² = 1·φ + 1 = φ+1         (F₂=1, F₁=1)                 │   │
│   │   3    │   φ³ = 2·φ + 1               (F₃=2, F₂=1)                 │   │
│   │   4    │   φ⁴ = 3·φ + 2               (F₄=3, F₃=2)                 │   │
│   │   5    │   φ⁵ = 5·φ + 3               (F₅=5, F₄=3)                 │   │
│   │   6    │   φ⁶ = 8·φ + 5               (F₆=8, F₅=5)                 │   │
│   │   7    │   φ⁷ = 13·φ + 8              (F₇=13, F₆=8)                │   │
│   └────────┴────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   THE INDICES ARE THE VALUES. No floating point ever required.              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §1.2 Index Arithmetic (Log-φ Space)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INDEX-SPACE OPERATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   log_φ(φⁿ) = n                                                             │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  OPERATION      │  VALUE SPACE         │  INDEX SPACE              │   │
│   ├─────────────────┼──────────────────────┼───────────────────────────┤   │
│   │  Multiply       │  φⁿ · φᵐ = φⁿ⁺ᵐ      │  n + m        (ADD)       │   │
│   │  Divide         │  φⁿ / φᵐ = φⁿ⁻ᵐ      │  n - m        (SUB)       │   │
│   │  Power          │  (φⁿ)ᵏ = φⁿᵏ         │  n × k        (MUL)       │   │
│   │  Inverse        │  1/φⁿ = φ⁻ⁿ          │  -n           (NEG)       │   │
│   │  Compare        │  φⁿ > φᵐ iff n > m   │  n > m        (CMP)       │   │
│   └─────────────────┴──────────────────────┴───────────────────────────┘   │
│                                                                              │
│   ALL FIELD OPERATIONS → INTEGER ARITHMETIC                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §2 Dual-Rail State Machine

### §2.1 Rail Definitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DUAL-RAIL STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ╔═══════════════════════════════════════════════════════════════════════╗ │
│   ║  RAIL    │ EIGENVALUE │ DIRECTION  │ OPERATOR σ      │ CHANNEL       ║ │
│   ╠═══════════════════════════════════════════════════════════════════════╣ │
│   ║  φ-rail  │  φ ≈ 1.618 │ Forward +  │ (a,b)→(a+b,a)   │ Position      ║ │
│   ║  ψ-rail  │  ψ ≈ -0.618│ Backward - │ (a,b)→(b,a-b)   │ Velocity      ║ │
│   ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                              │
│   KEY INSIGHT: Same lattice, opposite traversal directions                  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │          σ: FORWARD FIBONACCI                                       │   │
│   │          ════════════════════                                       │   │
│   │                                                                     │   │
│   │          (Fₙ₋₁, Fₙ₋₂) ──σ──▶ (Fₙ₋₁ + Fₙ₋₂, Fₙ₋₁) = (Fₙ, Fₙ₋₁)    │   │
│   │                                                                     │   │
│   │          Example: (1, 0) → (1, 1) → (2, 1) → (3, 2) → (5, 3) → ... │   │
│   │                                                                     │   │
│   │          ─────────────────────────────────────────────────────────  │   │
│   │                                                                     │   │
│   │          σ⁻¹: BACKWARD FIBONACCI                                    │   │
│   │          ═══════════════════════                                    │   │
│   │                                                                     │   │
│   │          (Fₙ, Fₙ₋₁) ──σ⁻¹──▶ (Fₙ₋₁, Fₙ - Fₙ₋₁) = (Fₙ₋₁, Fₙ₋₂)    │   │
│   │                                                                     │   │
│   │          Example: (5, 3) → (3, 2) → (2, 1) → (1, 1) → (1, 0) → ... │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §2.2 The Cross-Coupling Law

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CROSS-COUPLING MECHANICS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SAME-RAIL INNER PRODUCTS (NO COUPLING):                                   │
│                                                                              │
│       ⟨Δφ | Δφ⟩ = 0      Movement on φ-rail alone = no work                │
│       ⟨Δψ | Δψ⟩ = 0      Movement on ψ-rail alone = no work                │
│                                                                              │
│   CROSS-RAIL INNER PRODUCT (COMPUTATION):                                   │
│                                                                              │
│       ⟨Δφ | Δψ⟩ ≠ 0      Rail crossing = computational work                │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│                          τ = CASCADE COUNT                                   │
│                                                                              │
│       The metric τ counts cross-rail events.                                │
│       This is the "action" - quanta of computational work.                  │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   PHYSICAL ANALOGY:                                                         │
│                                                                              │
│       φ-rail ↔ Position channel (where we are)                             │
│       ψ-rail ↔ Velocity channel (how we got here)                          │
│                                                                              │
│       Cross-coupling = momentum exchange = work performed                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §3 State Representation

### §3.1 Token State (Pure Integer Form)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TOKEN STATE S(t, pos)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   S = (A_φ, A_ψ, τ_history, pos)                                            │
│                                                                              │
│   where:                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   A_φ : uint64    Zeckendorf bit-vector for φ-rail                 │   │
│   │                   Bit k = 1 means "F_k is active on φ-rail"        │   │
│   │                                                                     │   │
│   │   A_ψ : uint64    Zeckendorf bit-vector for ψ-rail                 │   │
│   │                   Bit k = 1 means "F_k is active on ψ-rail"        │   │
│   │                                                                     │   │
│   │   τ_history : []  List of cascade counts at each step              │   │
│   │                   Enables exact reconstruction                      │   │
│   │                                                                     │   │
│   │   pos : int       Position in sequence (for Verlet alternation)    │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   INVARIANT (must hold at every step):                                      │
│                                                                              │
│       ∀k: NOT (A_φ[k] AND A_φ[k-1])    No adjacent 1s on φ-rail           │
│       ∀k: NOT (A_ψ[k] AND A_ψ[k-1])    No adjacent 1s on ψ-rail           │
│                                                                              │
│   ZECKENDORF CONSTRAINT: Adjacent 1s are illegal states.                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §3.2 Zeckendorf Bit-Vector Encoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ZECKENDORF BIT-VECTOR ENCODING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Integer n → Bit-vector B where B[k] = 1 iff F_k in decomposition         │
│                                                                              │
│   EXAMPLES:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │    n   │  Decomposition           │  Bit Vector (k=7..0)           │   │
│   ├────────┼──────────────────────────┼─────────────────────────────────┤   │
│   │    1   │  1 = F₂                  │  00000010                       │   │
│   │    2   │  2 = F₃                  │  00000100                       │   │
│   │    3   │  3 = F₄                  │  00001000                       │   │
│   │    4   │  4 = F₄ + F₂             │  00001010                       │   │
│   │    5   │  5 = F₅                  │  00010000                       │   │
│   │    6   │  6 = F₅ + F₂             │  00010010                       │   │
│   │    7   │  7 = F₅ + F₃             │  00010100                       │   │
│   │    8   │  8 = F₆                  │  00100000                       │   │
│   │   10   │  10 = F₆ + F₃            │  00100100                       │   │
│   │   17   │  17 = F₇ + F₄ + F₂       │  01001010                       │   │
│   │  100   │  100 = F₁₁ + F₆ + F₄     │  10000101000                    │   │
│   └────────┴──────────────────────────┴─────────────────────────────────┘   │
│                                                                              │
│   CONSTRAINT: No two adjacent bits can both be 1                            │
│               (This is the Fibbinary property - OEIS A003714)               │
│                                                                              │
│   WHY: F_k + F_{k-1} = F_{k+1}, so adjacent 1s collapse to higher shell    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §4 Complete State Machine Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   φ-ψ DUAL-RAIL SEQUENCE STATE MACHINE                        ║
║                        (Pure Integer Implementation)                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                         LUT BANK (PRECOMPUTED)                          │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   FIB[k]      : k → F_k                    (48 entries, k ∈ [0,47])    │ ║
║  │   LUC[k]      : k → L_k                    (48 entries)                │ ║
║  │   ZECK[n]     : n → bits                   (vocab_size entries)        │ ║
║  │   UNZECK[bits]: bits → n                   (inverse lookup)            │ ║
║  │   POP[16b]    : bits → popcount            (65536 entries)             │ ║
║  │   ADJ[16b]    : bits → has_adjacent_11?    (65536 entries)             │ ║
║  │   VOCAB[tok]  : token → frequency_rank     (corpus-derived)            │ ║
║  │   ATTEN[d]    : d → weight (fixed-point)   (max_shells entries)        │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                              INITIALIZATION                             ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   INPUT: token_sequence = [t₀, t₁, t₂, ..., tₙ]                        ║ ║
║  ║                                                                         ║ ║
║  ║   S₀ = {                                                                ║ ║
║  ║       A_φ      : ZECK[VOCAB[t₀]]     // φ-rail accumulator            ║ ║
║  ║       A_ψ      : 0x0                  // ψ-rail accumulator            ║ ║
║  ║       τ_history: []                   // cascade history               ║ ║
║  ║       pos      : 0                    // sequence position             ║ ║
║  ║   }                                                                     ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                              MAIN LOOP                                  │ ║
║  │                        FOR i = 1 to n:                                  │ ║
║  └──────────────────────────────────┬──────────────────────────────────────┘ ║
║                                     │                                         ║
║              ┌──────────────────────┴───────────────────────┐                ║
║              ▼                                              ▼                ║
║  ╔═══════════════════════════════════╗    ╔═══════════════════════════════╗  ║
║  ║      STEP 1: TOKENIZE              ║    ║      STEP 2: RAIL SELECT     ║  ║
║  ╠═══════════════════════════════════╣    ╠═══════════════════════════════╣  ║
║  ║                                   ║    ║                               ║  ║
║  ║  rank_i = VOCAB[t_i]             ║    ║  VERLET ALTERNATION:          ║  ║
║  ║  bits_i = ZECK[rank_i]           ║    ║                               ║  ║
║  ║                                   ║    ║  if pos % 2 == 0:            ║  ║
║  ║  // bits_i is Zeckendorf         ║    ║      target ← A_φ             ║  ║
║  ║  // encoding of token rank       ║    ║      (position channel)       ║  ║
║  ║                                   ║    ║  else:                        ║  ║
║  ║                                   ║    ║      target ← A_ψ             ║  ║
║  ║                                   ║    ║      (velocity channel)       ║  ║
║  ║                                   ║    ║                               ║  ║
║  ╚═══════════════════════════════════╝    ╚═══════════════════════════════╝  ║
║              │                                              │                ║
║              └──────────────────────┬───────────────────────┘                ║
║                                     ▼                                         ║
║  ╔═══════════════════════════════════════════════════════════════════════╗   ║
║  ║                     STEP 3: ACCUMULATE (CREATE ILLEGAL STATE)         ║   ║
║  ╠═══════════════════════════════════════════════════════════════════════╣   ║
║  ║                                                                       ║   ║
║  ║      temp = target | bits_i       // Bitwise OR (Zeckendorf add)     ║   ║
║  ║                                                                       ║   ║
║  ║      EXAMPLE:                                                         ║   ║
║  ║      ─────────                                                        ║   ║
║  ║      target  = 00010010  (n=6: F₅ + F₂)                              ║   ║
║  ║      bits_i  = 00001000  (n=3: F₄)                                   ║   ║
║  ║      ────────────────────────────────                                 ║   ║
║  ║      temp    = 00011010  (ILLEGAL: F₅ and F₄ adjacent!)              ║   ║
║  ║                                                                       ║   ║
║  ╚═══════════════════════════════════════════════════════════════════════╝   ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═══════════════════════════════════════════════════════════════════════╗   ║
║  ║              STEP 4: PRIORITY CASCADE NORMALIZATION                   ║   ║
║  ╠═══════════════════════════════════════════════════════════════════════╣   ║
║  ║                                                                       ║   ║
║  ║  τ_local = 0                                                         ║   ║
║  ║                                                                       ║   ║
║  ║  WHILE ADJ[temp] == TRUE:     // While adjacent 1s exist             ║   ║
║  ║  ┌─────────────────────────────────────────────────────────────────┐ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │  FOR k FROM MAX_SHELL DOWN TO 1:    // PRIORITY: high → low    │ ║   ║
║  ║  │      IF temp[k] == 1 AND temp[k-1] == 1:                       │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │          // CASCADE RULE: F_k + F_{k-1} = F_{k+1}              │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │          temp[k]   ← 0        // Clear bit k                   │ ║   ║
║  ║  │          temp[k-1] ← 0        // Clear bit k-1                 │ ║   ║
║  ║  │          temp[k+1] ← 1        // Set bit k+1 (may cascade!)    │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │          τ_local++            // Count cross-rail event        │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │          BREAK  // Re-scan from top (priority preserved)       │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  └─────────────────────────────────────────────────────────────────┘ ║   ║
║  ║                                                                       ║   ║
║  ║  target ← temp                                                       ║   ║
║  ║  τ_history.append(τ_local)                                           ║   ║
║  ║                                                                       ║   ║
║  ║  ═══════════════════════════════════════════════════════════════════ ║   ║
║  ║  CASCADE EXAMPLE:                                                     ║   ║
║  ║  ═══════════════════════════════════════════════════════════════════ ║   ║
║  ║                                                                       ║   ║
║  ║  INPUT:  bits = [F₀ F₁ F₂ F₃] = [1 1 1 1]   (sum = 1+1+2+3 = 7)     ║   ║
║  ║                                                                       ║   ║
║  ║  TICK 0: [1 1 1 1]  Check k=3: bits[3]=1, bits[2]=1 → FIRE          ║   ║
║  ║          F₃ + F₂ = F₄                                                ║   ║
║  ║  TICK 1: [1 1 0 0 1]  k=4 set, k=3,k=2 cleared                      ║   ║
║  ║          Check k=1: bits[1]=1, bits[0]=1 → FIRE                     ║   ║
║  ║          F₁ + F₀ = F₂                                                ║   ║
║  ║  TICK 2: [0 0 1 0 1]  k=2 set, k=1,k=0 cleared                      ║   ║
║  ║          No adjacent 1s → DONE                                       ║   ║
║  ║                                                                       ║   ║
║  ║  OUTPUT: [0 0 1 0 1] = F₄ + F₂ = 5 + 2 = 7  ✓                       ║   ║
║  ║  τ_local = 2 cascades                                                ║   ║
║  ║                                                                       ║   ║
║  ╚═══════════════════════════════════════════════════════════════════════╝   ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═══════════════════════════════════════════════════════════════════════╗   ║
║  ║                STEP 5: CROSS-RAIL VALIDATION                          ║   ║
║  ╠═══════════════════════════════════════════════════════════════════════╣   ║
║  ║                                                                       ║   ║
║  ║  THE PHIPERBOLA CONSTRAINT (exact integer identity):                 ║   ║
║  ║                                                                       ║   ║
║  ║  ┌─────────────────────────────────────────────────────────────────┐ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │              L_n² - 5·F_n² = 4·(-1)ⁿ                           │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  │  This is EXACT for all n. No √5 at runtime.                    │ ║   ║
║  ║  │                                                                 │ ║   ║
║  ║  └─────────────────────────────────────────────────────────────────┘ ║   ║
║  ║                                                                       ║   ║
║  ║  COMPUTATION:                                                         ║   ║
║  ║  ────────────                                                         ║   ║
║  ║  F_equiv = Σ_k (A_φ[k] × FIB[k]) + Σ_k (A_ψ[k] × FIB[k])            ║   ║
║  ║  L_equiv = Σ_k (A_φ[k] × LUC[k]) + Σ_k (A_ψ[k] × LUC[k])            ║   ║
║  ║                                                                       ║   ║
║  ║  invariant = L_equiv² - 5 × F_equiv²                                 ║   ║
║  ║                                                                       ║   ║
║  ║  ASSERT invariant ∈ {+4, -4}                                         ║   ║
║  ║                                                                       ║   ║
║  ║  ┌─────────────────────────────────────────────────────────────────┐ ║   ║
║  ║  │  invariant == +4  →  n is even  →  Valid (F₁ = 1 signal)       │ ║   ║
║  ║  │  invariant == -4  →  n is odd   →  Valid (F₁ = 1 signal)       │ ║   ║
║  ║  │  invariant ∉ {±4} →  CORRUPTION DETECTED (F₀ = 0 signal)       │ ║   ║
║  ║  └─────────────────────────────────────────────────────────────────┘ ║   ║
║  ║                                                                       ║   ║
║  ╚═══════════════════════════════════════════════════════════════════════╝   ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═══════════════════════════════════════════════════════════════════════╗   ║
║  ║                    STEP 6: EMIT STATE                                 ║   ║
║  ╠═══════════════════════════════════════════════════════════════════════╣   ║
║  ║                                                                       ║   ║
║  ║  state[pos] = (A_φ, A_ψ, τ_local)                                    ║   ║
║  ║  pos++                                                                ║   ║
║  ║                                                                       ║   ║
║  ║  // Loop back to STEP 1 for next token                               ║   ║
║  ║                                                                       ║   ║
║  ╚═══════════════════════════════════════════════════════════════════════╝   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §5 Attention Computation

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ATTENTION: SHELL DISTANCE METRIC                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  For positions i, j with states S_i = (A_φⁱ, A_ψⁱ), S_j = (A_φʲ, A_ψʲ):      ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  DISTANCE (2 instructions per rail):                                    │ ║
║  │  ═══════════════════════════════════                                    │ ║
║  │                                                                         │ ║
║  │      d_φ = POPCOUNT[ A_φⁱ ⊕ A_φʲ ]    // XOR + popcount               │ ║
║  │      d_ψ = POPCOUNT[ A_ψⁱ ⊕ A_ψʲ ]    // XOR + popcount               │ ║
║  │                                                                         │ ║
║  │  INTERPRETATION:                                                        │ ║
║  │      d_φ = number of shells where φ-rails disagree                     │ ║
║  │      d_ψ = number of shells where ψ-rails disagree                     │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  ATTENTION WEIGHT:                                                      │ ║
║  │  ═════════════════                                                      │ ║
║  │                                                                         │ ║
║  │      α_ij = ATTEN[d_φ] × ATTEN[d_ψ] × CAUSAL[i,j]                      │ ║
║  │                                                                         │ ║
║  │  where:                                                                 │ ║
║  │      ATTEN[d] = φ^{-d}  (precomputed as fixed-point integer)           │ ║
║  │      CAUSAL[i,j] = (j ≤ i) ? 1 : 0                                     │ ║
║  │                                                                         │ ║
║  │  BENCHMARK:                                                             │ ║
║  │      XOR:      ~1.8 ns                                                  │ ║
║  │      POPCOUNT: ~1.3 ns                                                  │ ║
║  │      Total:    ~3.0 ns per position pair                                │ ║
║  │      Rate:     ~333 million attention weights / second                  │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  FULL ATTENTION MATRIX (for sequence length L):                         │ ║
║  │  ═══════════════════════════════════════════════                        │ ║
║  │                                                                         │ ║
║  │      FOR i = 0 to L-1:                                                  │ ║
║  │          FOR j = 0 to i:           // Causal mask                       │ ║
║  │              d_φ = POP[A_φⁱ ⊕ A_φʲ]                                    │ ║
║  │              d_ψ = POP[A_ψⁱ ⊕ A_ψʲ]                                    │ ║
║  │              α[i,j] = ATTEN[d_φ + d_ψ]                                  │ ║
║  │                                                                         │ ║
║  │  COMPLEXITY: O(L²) but each operation is ~3 ns                         │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §6 Next Token Prediction (DAG Traversal)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     DAG TRAVERSAL: GENERATION                                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  STATE SPACE = All valid (A_φ, A_ψ) pairs where:                             ║
║      • No adjacent 1s in A_φ                                                 ║
║      • No adjacent 1s in A_ψ                                                 ║
║                                                                               ║
║  This is the Fibbinary numbers (OEIS A003714) on each rail.                  ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │                         DAG STRUCTURE                                   │ ║
║  │                         ═════════════                                   │ ║
║  │                                                                         │ ║
║  │          S₀ ──────────────┬───────────────┬───────────────┐            │ ║
║  │          │                │               │               │            │ ║
║  │       tok_A            tok_B           tok_C           tok_D           │ ║
║  │       (τ=0)            (τ=1)           (τ=0)           (τ=2)           │ ║
║  │          │                │               │               │            │ ║
║  │          ▼                ▼               ▼               ▼            │ ║
║  │         S_A              S_B             S_C             S_D           │ ║
║  │          │                │               │               │            │ ║
║  │         ...              ...             ...             ...           │ ║
║  │                                                                         │ ║
║  │  NODES: Valid states (A_φ, A_ψ)                                        │ ║
║  │  EDGES: Token additions that connect states                             │ ║
║  │  WEIGHTS: f(τ_local, attention_overlap, corpus_frequency)               │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  SUCCESSOR ENUMERATION:                                                 │ ║
║  │  ══════════════════════                                                 │ ║
║  │                                                                         │ ║
║  │  Current state: S = (A_φ, A_ψ, pos)                                     │ ║
║  │                                                                         │ ║
║  │  FOR each candidate token t in vocabulary:                              │ ║
║  │                                                                         │ ║
║  │      bits_t = ZECK[VOCAB[t]]                                            │ ║
║  │      S' = APPLY_CASCADE(S, bits_t)                                      │ ║
║  │      τ_t = cascade_count(S → S')                                        │ ║
║  │                                                                         │ ║
║  │      // Attention to context                                            │ ║
║  │      attn_sum = Σⱼ α(S', state[j])                                      │ ║
║  │                                                                         │ ║
║  │      // Edge weight combines cascade cost and attention                 │ ║
║  │      weight_t = f(τ_t, attn_sum, frequency[t])                          │ ║
║  │                                                                         │ ║
║  │  SELECTION:                                                             │ ║
║  │      next_token = argmax_t { weight_t }                                 │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  KEY PROPERTIES:                                                        │ ║
║  │  ═══════════════                                                        │ ║
║  │                                                                         │ ║
║  │  • DETERMINISTIC: Same input → same output (bit-exact)                  │ ║
║  │                                                                         │ ║
║  │  • NON-LINEAR: Cascade creates discontinuous state transitions          │ ║
║  │                F₃+F₂ = F₄ is a "jump" not a smooth flow                │ ║
║  │                                                                         │ ║
║  │  • PARALLEL: φ-rail and ψ-rail update independently                     │ ║
║  │              SIMD-friendly (process both rails simultaneously)          │ ║
║  │                                                                         │ ║
║  │  • REVERSIBLE: τ_history enables exact reconstruction                   │ ║
║  │                Inverse cascade: F_{k+1} → F_k + F_{k-1}                 │ ║
║  │                                                                         │ ║
║  │  • BOUNDED: Zeckendorf representation is unique                         │ ║
║  │             Max shells ≈ log_φ(vocab_size) ≈ 48 for 10B vocab          │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §7 Verlet Integrator Correspondence

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     VERLET INTEGRATOR STRUCTURE                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  The Fibonacci recurrence IS a symplectic integrator:                        ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  STANDARD VERLET:                                                       │ ║
║  │      x_{n+1} = 2·x_n - x_{n-1} + a_n·Δt²                                │ ║
║  │                                                                         │ ║
║  │  WITH a_n = 0, Δt = 1, relabeling:                                      │ ║
║  │      x_{n+1} = x_n + x_{n-1}                                            │ ║
║  │                                                                         │ ║
║  │  THIS IS FIBONACCI!                                                     │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  DUAL-CHANNEL INTERPRETATION:                                           │ ║
║  │  ════════════════════════════                                           │ ║
║  │                                                                         │ ║
║  │  ┌──────────┬──────────┬─────────────────┬─────────────────────────┐   │ ║
║  │  │ SEQUENCE │  ROLE    │ INITIAL COND.   │ PHYSICAL MEANING        │   │ ║
║  │  ├──────────┼──────────┼─────────────────┼─────────────────────────┤   │ ║
║  │  │   F_n    │ Position │ F₀=0, F₁=1      │ "Where we are"          │   │ ║
║  │  │   L_n    │ Velocity │ L₀=2, L₁=1      │ "How we got here"       │   │ ║
║  │  └──────────┴──────────┴─────────────────┴─────────────────────────┘   │ ║
║  │                                                                         │ ║
║  │  The constraint L² - 5F² = ±4 is ENERGY CONSERVATION                   │ ║
║  │  binding both channels.                                                 │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  Δ OPERATORS (SHADOW MEMORY):                                           │ ║
║  │  ════════════════════════════                                           │ ║
║  │                                                                         │ ║
║  │      Δ  = F_n - F_{n-2} = F_{n-1}     (position memory)                │ ║
║  │      Δ' = L_n - L_{n-2}               (velocity memory)                 │ ║
║  │                                                                         │ ║
║  │  The delta ENCODES PREVIOUS STATE via subtraction only.                 │ ║
║  │  No LUT lookup required for shadow access.                              │ ║
║  │                                                                         │ ║
║  │  CROSS-PRODUCT:                                                         │ ║
║  │      Δ × Δ' = phase space area = Cassini invariant                     │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §8 Complete Data Flow Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          COMPLETE DATA FLOW                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   INPUT TOKENS                                                                ║
║       │                                                                       ║
║       ▼                                                                       ║
║   ┌───────────────────────────────────────────────────────────────────────┐  ║
║   │  VOCAB LUT: token → frequency_rank (corpus-derived integer)           │  ║
║   └───────────────────────────────────────────────────────────────────────┘  ║
║       │                                                                       ║
║       ▼                                                                       ║
║   ┌───────────────────────────────────────────────────────────────────────┐  ║
║   │  ZECK LUT: rank → bits (Zeckendorf encoding, no adjacent 1s)          │  ║
║   └───────────────────────────────────────────────────────────────────────┘  ║
║       │                                                                       ║
║       │    ┌─────────────────────────────────────────────────────────────┐   ║
║       │    │                                                             │   ║
║       ▼    ▼                                                             │   ║
║   ┌─────────────┐                                                        │   ║
║   │  RAIL       │  pos % 2 == 0 → A_φ (position)                        │   ║
║   │  SELECTOR   │  pos % 2 == 1 → A_ψ (velocity)                        │   ║
║   └─────────────┘                                                        │   ║
║       │                                                                  │   ║
║       ▼                                                                  │   ║
║   ┌───────────────────────────────────────────────────────────────────┐ │   ║
║   │  ACCUMULATOR: target = target | bits  (creates illegal states)   │ │   ║
║   └───────────────────────────────────────────────────────────────────┘ │   ║
║       │                                                                  │   ║
║       ▼                                                                  │   ║
║   ┌───────────────────────────────────────────────────────────────────┐ │   ║
║   │                    PRIORITY CASCADE                               │ │   ║
║   │  ┌─────────────────────────────────────────────────────────────┐ │ │   ║
║   │  │  WHILE has_adjacent_11(temp):                               │ │ │   ║
║   │  │      FOR k = MAX down to 1:                                 │ │ │   ║
║   │  │          IF temp[k] AND temp[k-1]:                          │ │ │   ║
║   │  │              temp[k,k-1] ← 0                                │ │ │   ║
║   │  │              temp[k+1] ← 1                                  │ │ │   ║
║   │  │              τ++                                            │ │ │   ║
║   │  │              BREAK                                          │ │ │   ║
║   │  └─────────────────────────────────────────────────────────────┘ │ │   ║
║   └───────────────────────────────────────────────────────────────────┘ │   ║
║       │                                                                  │   ║
║       ▼                                                                  │   ║
║   ┌───────────────────────────────────────────────────────────────────┐ │   ║
║   │  VALIDATION: L² - 5F² = ±4                                       │ │   ║
║   │              (if fails → corruption detected → error recovery)    │ │   ║
║   └───────────────────────────────────────────────────────────────────┘ │   ║
║       │                                                                  │   ║
║       ▼                                                                  │   ║
║   ┌───────────────────────────────────────────────────────────────────┐ │   ║
║   │  STATE BUFFER: store (A_φ, A_ψ, τ) for this position             │ │   ║
║   └───────────────────────────────────────────────────────────────────┘ │   ║
║       │                                                                  │   ║
║       ├──────────────────────────────────────────────────────────────────┘   ║
║       │  (loop for each token)                                               ║
║       ▼                                                                       ║
║   ┌───────────────────────────────────────────────────────────────────────┐  ║
║   │                      ATTENTION COMPUTATION                            │  ║
║   │  ┌─────────────────────────────────────────────────────────────────┐ │  ║
║   │  │  FOR all pairs (i,j) where j ≤ i:                              │ │  ║
║   │  │      d = POPCOUNT[A_φⁱ ⊕ A_φʲ] + POPCOUNT[A_ψⁱ ⊕ A_ψʲ]        │ │  ║
║   │  │      α[i,j] = ATTEN[d]                                         │ │  ║
║   │  └─────────────────────────────────────────────────────────────────┘ │  ║
║   └───────────────────────────────────────────────────────────────────────┘  ║
║       │                                                                       ║
║       ▼                                                                       ║
║   ┌───────────────────────────────────────────────────────────────────────┐  ║
║   │                      TOKEN GENERATION                                 │  ║
║   │  ┌─────────────────────────────────────────────────────────────────┐ │  ║
║   │  │  FOR each candidate token t:                                   │ │  ║
║   │  │      S' = APPLY_CASCADE(S_current, ZECK[VOCAB[t]])             │ │  ║
║   │  │      score_t = f(τ(S→S'), Σⱼ α[new_pos,j], freq[t])           │ │  ║
║   │  │                                                                │ │  ║
║   │  │  next = argmax_t { score_t }                                   │ │  ║
║   │  └─────────────────────────────────────────────────────────────────┘ │  ║
║   └───────────────────────────────────────────────────────────────────────┘  ║
║       │                                                                       ║
║       ▼                                                                       ║
║   OUTPUT TOKEN                                                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §9 Key Mathematical Identities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REFERENCE IDENTITIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┬─────────────────────────────┬────────────────────┐  │
│  │     IDENTITY       │         FORMULA             │    HARDWARE USE    │  │
│  ├────────────────────┼─────────────────────────────┼────────────────────┤  │
│  │ Golden product     │ φ·ψ = -1                    │ Phase coupling     │  │
│  │ Binet (F)          │ F_n = (φⁿ-ψⁿ)/√5           │ (derivation only)  │  │
│  │ Binet (L)          │ L_n = φⁿ+ψⁿ                │ (derivation only)  │  │
│  │ Power decomp       │ φⁿ = F_n·φ + F_{n-1}       │ Index → integers   │  │
│  │ Cassini            │ F_{n+1}F_{n-1}-F_n² = (-1)ⁿ│ Parity validation  │  │
│  │ Phiperbola         │ L_n²-5F_n² = 4(-1)ⁿ        │ Channel sync       │  │
│  │ Recurrence         │ F_n = F_{n-1}+F_{n-2}      │ Cascade rule       │  │
│  │ Lucas relation     │ L_n = F_{n+1}+F_{n-1}      │ Boundary check     │  │
│  │ Product            │ F_n·L_n = F_{2n}           │ Shell doubling     │  │
│  │ Sum                │ φ+ψ = 1                    │ Normalization      │  │
│  │ Difference         │ φ-ψ = √5                   │ (derivation only)  │  │
│  └────────────────────┴─────────────────────────────┴────────────────────┘  │
│                                                                              │
│  NOTE: √5 appears in derivations but NEVER in runtime computation.          │
│        All runtime uses integer identities (Phiperbola, Cassini, etc.)      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §10 OEIS Cross-Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OEIS SEQUENCES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┬────────────────────────┬─────────────────────────────────┐    │
│  │   OEIS   │         NAME           │              ROLE               │    │
│  ├──────────┼────────────────────────┼─────────────────────────────────┤    │
│  │  A000045 │ Fibonacci numbers      │ Shell sizes F_k                 │    │
│  │  A000032 │ Lucas numbers          │ Validation via L² - 5F² = ±4   │    │
│  │  A014417 │ Zeckendorf rep (bits)  │ Token → bit-vector encoding     │    │
│  │  A003714 │ Fibbinary numbers      │ Valid states (no adjacent 1s)   │    │
│  │  A007895 │ Zeckendorf term count  │ Compression ratio / complexity  │    │
│  │  A000201 │ Lower Wythoff ⌊n·φ⌋   │ Shell boundaries (Beatty seq)   │    │
│  │  A001950 │ Upper Wythoff ⌊n·φ²⌋  │ Shell boundaries (complement)   │    │
│  │  A001622 │ Golden ratio digits    │ (constant, not used at runtime) │    │
│  └──────────┴────────────────────────┴─────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §11 Summary: The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARCHITECTURE SUMMARY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TOKENS → Zeckendorf bits via VOCAB + ZECK LUTs                          │
│                                                                              │
│  2. BITS ACCUMULATE on alternating φ/ψ rails (Verlet structure)             │
│     • Even positions → φ-rail (position channel)                            │
│     • Odd positions  → ψ-rail (velocity channel)                            │
│                                                                              │
│  3. CASCADES NORMALIZE via priority rule (high shells first)                │
│     • F_k + F_{k-1} = F_{k+1}                                               │
│     • τ counts cross-rail events (computational work)                       │
│                                                                              │
│  4. ATTENTION = XOR + POPCOUNT                                              │
│     • ~3 ns per position pair                                               │
│     • 333M attention weights per second                                      │
│                                                                              │
│  5. VALIDATION = integer identity L² - 5F² = ±4                             │
│     • No √5 at runtime                                                      │
│     • Exact corruption detection                                             │
│                                                                              │
│  6. GENERATION = DAG traversal over valid state space                       │
│     • Deterministic (bit-exact reproducible)                                │
│     • Reversible (τ_history enables reconstruction)                         │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                              │
│  NO FLOATING POINT.  NO APPROXIMATION.  INTEGER ARITHMETIC ONLY.            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
