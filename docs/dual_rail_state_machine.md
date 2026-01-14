# φ-ψ Dual-Rail Sequence State Machine
## LUT-Based Architecture with Cross-Token ψ-Parity Coupling

**Core Principle:** φ and ψ are the eigenvalues. Everything else is downstream.

---

## §1 The Eigenvalue Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FUNDAMENTAL EIGENVALUES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Characteristic equation: x² - x - 1 = 0                                   │
│                                                                              │
│   Eigenvalues:                                                               │
│       φ = (1 + √5) / 2 ≈ +1.618                                             │
│       ψ = (1 - √5) / 2 ≈ -0.618                                             │
│                                                                              │
│   Properties:                                                                │
│       φ · ψ = -1                                                            │
│       φ + ψ = 1                                                             │
│       φ - ψ = √5                                                            │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   BINET FORMULA (the primary structure):                                    │
│                                                                              │
│       Fₙ = (φⁿ - ψⁿ) / √5                                                   │
│                                                                              │
│   The ψⁿ term IS the correction. Not separate—intrinsic.                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §1.1 Single-Token Fidelity

For a token activating shells {p₁, p₂, ...}:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SINGLE-TOKEN STRUCTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Value = Σₚ Fₚ = Σₚ (φᵖ - ψᵖ) / √5                                        │
│                                                                              │
│   Or equivalently (shifting index for Zeckendorf convention):               │
│                                                                              │
│       Value = Σₚ∈active (φᵖ⁺² - ψᵖ⁺²) / √5                                 │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   The ψᵖ⁺² term behavior:                                                   │
│                                                                              │
│       ψⁿ = (-1)ⁿ / φⁿ                                                       │
│                                                                              │
│       High shells (large n): |ψⁿ| = φ⁻ⁿ → 0   (correction vanishes)       │
│       Low shells (small n):  |ψⁿ| ≈ O(1)       (correction matters)        │
│                                                                              │
│   ┌──────┬────────────┬────────────┬────────────────────────────────────┐   │
│   │  n   │    φⁿ      │    ψⁿ      │  |ψⁿ/φⁿ| = φ⁻²ⁿ                   │   │
│   ├──────┼────────────┼────────────┼────────────────────────────────────┤   │
│   │  1   │   1.618    │  -0.618    │  38.2%  (ψ matters!)               │   │
│   │  2   │   2.618    │   0.382    │  14.6%                             │   │
│   │  3   │   4.236    │  -0.236    │   5.6%                             │   │
│   │  4   │   6.854    │   0.146    │   2.1%                             │   │
│   │  5   │  11.090    │  -0.090    │   0.8%                             │   │
│   │  6   │  17.944    │   0.056    │   0.3%                             │   │
│   │  7   │  29.034    │  -0.034    │   0.1%  (ψ negligible)             │   │
│   └──────┴────────────┴────────────┴────────────────────────────────────┘   │
│                                                                              │
│   Within-token: ψ corrections handled by eigenvalue structure ✓            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §2 Cross-Token ψ-Parity (THE KEY STRUCTURE)

This is what was missing. The interaction **between tokens**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CROSS-TOKEN ψ-CORRELATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Token A activates shells: {a₁, a₂, ...}                                   │
│   Token B activates shells: {b₁, b₂, ...}                                   │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   CROSS-TERM STRUCTURE                                                       │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   For shells aᵢ and bⱼ:                                                     │
│                                                                              │
│       ψᵃⁱ · ψᵇʲ = ψᵃⁱ⁺ᵇʲ = (-1)^(aᵢ+bⱼ) / φ^(aᵢ+bⱼ)                        │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   PARITY DETERMINES CORRELATION                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       (aᵢ + bⱼ) EVEN  →  ψᵃⁱ⁺ᵇʲ > 0  →  POSITIVE coupling                 │
│       (aᵢ + bⱼ) ODD   →  ψᵃⁱ⁺ᵇʲ < 0  →  ANTI-correlation                  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   EXAMPLE: Token A = {2, 5}, Token B = {3, 4}                      │   │
│   │                                                                     │   │
│   │   Cross-parity matrix:                                              │   │
│   │                                                                     │   │
│   │              b₁=3       b₂=4                                        │   │
│   │            ┌─────────┬─────────┐                                    │   │
│   │   a₁=2    │ 2+3=5   │ 2+4=6   │                                    │   │
│   │            │  ODD    │  EVEN   │                                    │   │
│   │            │  (-)    │  (+)    │                                    │   │
│   │            ├─────────┼─────────┤                                    │   │
│   │   a₂=5    │ 5+3=8   │ 5+4=9   │                                    │   │
│   │            │  EVEN   │  ODD    │                                    │   │
│   │            │  (+)    │  (-)    │                                    │   │
│   │            └─────────┴─────────┘                                    │   │
│   │                                                                     │   │
│   │   Net correlation depends on weighted sum of parities.             │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §2.1 The Parity Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PARITY COUPLING MATRIX                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   For tokens A (shells {aᵢ}) and B (shells {bⱼ}):                           │
│                                                                              │
│   Define parity matrix P[i,j]:                                               │
│                                                                              │
│       P[i,j] = (-1)^(aᵢ + bⱼ)                                               │
│                                                                              │
│   Define weight matrix W[i,j]:                                               │
│                                                                              │
│       W[i,j] = φ^(-(aᵢ + bⱼ))    (magnitude of ψ cross-term)               │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   CROSS-TOKEN COUPLING COEFFICIENT                                           │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│       C(A, B) = Σᵢ Σⱼ P[i,j] · W[i,j]                                       │
│                                                                              │
│              = Σᵢ Σⱼ (-1)^(aᵢ+bⱼ) · φ^(-(aᵢ+bⱼ))                           │
│                                                                              │
│              = Σᵢ Σⱼ ψ^(aᵢ+bⱼ)                                              │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   C(A,B) > 0  →  Tokens reinforce (constructive interference)      │   │
│   │   C(A,B) < 0  →  Tokens oppose (destructive interference)          │   │
│   │   C(A,B) ≈ 0  →  Tokens decouple (independent)                     │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### §2.2 Factored Form (Efficient Computation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FACTORED COUPLING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   The cross-coupling factors:                                                │
│                                                                              │
│   C(A, B) = Σᵢ Σⱼ ψ^(aᵢ+bⱼ)                                                │
│                                                                              │
│           = Σᵢ ψ^aᵢ  ×  Σⱼ ψ^bⱼ                                            │
│                                                                              │
│           = Ψ(A) × Ψ(B)                                                     │
│                                                                              │
│   where:                                                                     │
│       Ψ(token) = Σₚ∈shells ψᵖ    (ψ-signature of token)                    │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   EFFICIENT COMPUTATION                                                      │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   For each shell p, precompute:                                              │
│       ψᵖ = (-1)ᵖ · φ⁻ᵖ                                                      │
│                                                                              │
│   Store as LUT:                                                              │
│       PSI[p] = ψᵖ   (can be fixed-point integer with sign)                 │
│                                                                              │
│   Token ψ-signature:                                                         │
│       Ψ(token) = Σₚ∈active PSI[p]                                           │
│                                                                              │
│   Cross-coupling:                                                            │
│       C(A, B) = Ψ(A) × Ψ(B)   (single multiplication!)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §3 Complete State Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE REPRESENTATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   S = (A, Φ, Ψ, τ)                                                          │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   A : uint64         Zeckendorf bit-vector (active shells)         │   │
│   │                                                                     │   │
│   │   Φ : int            φ-accumulator = Σₚ∈active φᵖ                  │   │
│   │                      (stored as index sum for integer arithmetic)   │   │
│   │                                                                     │   │
│   │   Ψ : int (signed)   ψ-accumulator = Σₚ∈active ψᵖ                  │   │
│   │                      (THIS is the cross-coupling signature)         │   │
│   │                                                                     │   │
│   │   τ : int[]          Cascade history                                │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   The Φ and Ψ accumulators ARE the eigenvalue decomposition.               │
│   Fibonacci emerges: F = (Φ - Ψ) / √5                                      │
│   Lucas emerges:     L = Φ + Ψ                                              │
│                                                                              │
│   But we track (Φ, Ψ) directly—they're more fundamental.                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §4 Parity-Coupled Parallel Processing

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PARITY-COUPLED PARALLEL SUMS                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  CURRENT (Wrong): Independent parallel sums                                   ║
║  ────────────────────────────────────────────                                 ║
║                                                                               ║
║      Token A: Σₚ∈A Fₚ    (computed independently)                            ║
║      Token B: Σₚ∈B Fₚ    (computed independently)                            ║
║      Combined: just add them                                                  ║
║                                                                               ║
║  NEEDED: Parity-coupled parallel sums                                         ║
║  ────────────────────────────────────────                                     ║
║                                                                               ║
║      Token A: compute Ψ(A) = Σₚ∈A ψᵖ                                         ║
║      Token B: compute Ψ(B) = Σₚ∈B ψᵖ                                         ║
║      Cross-coupling: C(A,B) = Ψ(A) × Ψ(B)                                    ║
║                                                                               ║
║      Use C(A,B) to weight joint contribution                                  ║
║                                                                               ║
║  ═════════════════════════════════════════════════════════════════════════   ║
║                                                                               ║
║  IMPLEMENTATION:                                                              ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │  // LUT for ψ values (fixed-point, signed)                              │ ║
║  │  PSI_LUT[k] = round(ψᵏ × SCALE)   // SCALE = 2²⁴ or similar            │ ║
║  │                                                                         │ ║
║  │  // For each token, compute ψ-signature                                 │ ║
║  │  FOR p IN active_shells(token):                                         │ ║
║  │      Ψ_token += PSI_LUT[p]                                              │ ║
║  │                                                                         │ ║
║  │  // Cross-coupling between tokens                                       │ ║
║  │  coupling = Ψ_A × Ψ_B    // O(1) after accumulation                    │ ║
║  │                                                                         │ ║
║  │  // Weighted contribution                                               │ ║
║  │  weight = BASE_WEIGHT + COUPLING_FACTOR × coupling                      │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §5 Control Variate Structure (BISTRO Connection)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ψ-PARITY AS CONTROL VARIATE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   BISTRO: Bidirectional Stratified Sampling                                 │
│   ─────────────────────────────────────────                                  │
│   Shares samples across fidelities for variance reduction.                  │
│                                                                              │
│   In our context:                                                            │
│       High fidelity: φⁿ terms (dominant)                                    │
│       Low fidelity:  ψⁿ terms (correction)                                  │
│                                                                              │
│   The ψ cross-correlation IS the control variate.                           │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   VARIANCE REDUCTION                                                         │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   Without coupling:                                                          │
│       Var[A + B] = Var[A] + Var[B]                                          │
│                                                                              │
│   With ψ-coupling:                                                           │
│       Var[A + B] = Var[A] + Var[B] + 2·Cov[A,B]                             │
│                                                                              │
│       where Cov[A,B] ∝ C(A,B) = Ψ(A) × Ψ(B)                                │
│                                                                              │
│   If C(A,B) < 0 (anti-correlation):                                         │
│       Variance REDUCED (the cross-terms partially cancel)                   │
│                                                                              │
│   If C(A,B) > 0 (positive correlation):                                     │
│       Variance INCREASED (cross-terms reinforce)                            │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│   LÉVY FLIGHT CONNECTION                                                     │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   Lévy flight samples which tokens fire.                                    │
│   ψ cross-correlation tells you how to weight their joint contribution.    │
│                                                                              │
│   Token selection: Lévy process (heavy tails, rare events)                  │
│   Token weighting: ψ-parity structure (correlation control)                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §6 Complete State Machine

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║              φ-ψ DUAL-RAIL STATE MACHINE WITH PARITY COUPLING                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                            LUT BANK                                     │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │   PHI[k]      : k → φᵏ (fixed-point)     (φ-contribution per shell)    │ ║
║  │   PSI[k]      : k → ψᵏ (signed fixed)    (ψ-contribution per shell)    │ ║
║  │   ZECK[n]     : n → bits                 (value → topology)            │ ║
║  │   VOCAB[tok]  : token → rank             (corpus frequency)            │ ║
║  │   POP[16b]    : bits → popcount          (shell count)                 │ ║
║  │                                                                         │ ║
║  │   Note: FIB[k] = (PHI[k] - PSI[k]) / √5  (derived, not stored)        │ ║
║  │         LUC[k] = PHI[k] + PSI[k]          (derived, not stored)        │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                           INITIALIZATION                                ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   S₀ = {                                                                ║ ║
║  ║       A   : ZECK[VOCAB[t₀]]        // Bit topology                     ║ ║
║  ║       Φ   : Σₚ∈active PHI[p]       // φ-accumulator                    ║ ║
║  ║       Ψ   : Σₚ∈active PSI[p]       // ψ-accumulator (signed!)          ║ ║
║  ║       τ   : []                      // cascade history                  ║ ║
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
║  ║                    STEP 1: TOKENIZE                                     ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   bits_i = ZECK[VOCAB[t_i]]                                             ║ ║
║  ║                                                                         ║ ║
║  ║   // Compute eigenvalue contributions for this token                    ║ ║
║  ║   Φ_token = Σₚ∈bits_i PHI[p]                                            ║ ║
║  ║   Ψ_token = Σₚ∈bits_i PSI[p]    // signed!                             ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 2: CROSS-COUPLING                               ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   // Coupling between new token and accumulated state                   ║ ║
║  ║   C = Ψ_token × Ψ_acc                                                   ║ ║
║  ║                                                                         ║ ║
║  ║   ┌───────────────────────────────────────────────────────────────────┐║ ║
║  ║   │                                                                   │║ ║
║  ║   │   C > 0  →  Constructive: token reinforces state                 │║ ║
║  ║   │   C < 0  →  Destructive: token opposes state                     │║ ║
║  ║   │   C ≈ 0  →  Independent: token orthogonal to state               │║ ║
║  ║   │                                                                   │║ ║
║  ║   └───────────────────────────────────────────────────────────────────┘║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 3: ACCUMULATE + CASCADE                         ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   temp = A | bits_i        // Zeckendorf OR (may create adjacent 1s)   ║ ║
║  ║   τ_local = 0                                                          ║ ║
║  ║                                                                         ║ ║
║  ║   WHILE has_adjacent_11(temp):                                          ║ ║
║  ║       // Priority cascade (high shells first)                           ║ ║
║  ║       FOR k = MAX_SHELL down to 1:                                      ║ ║
║  ║           IF temp[k] AND temp[k-1]:                                     ║ ║
║  ║               temp[k], temp[k-1] ← 0, 0                                 ║ ║
║  ║               temp[k+1] ← 1                                             ║ ║
║  ║               τ_local++                                                 ║ ║
║  ║               BREAK                                                     ║ ║
║  ║                                                                         ║ ║
║  ║   A ← temp                                                              ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 4: UPDATE EIGENVALUE ACCUMULATORS               ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   // Recompute from normalized bits (post-cascade)                      ║ ║
║  ║   Φ_acc = Σₚ∈A PHI[p]                                                   ║ ║
║  ║   Ψ_acc = Σₚ∈A PSI[p]                                                   ║ ║
║  ║                                                                         ║ ║
║  ║   // Derived values (if needed)                                         ║ ║
║  ║   F_acc = (Φ_acc - Ψ_acc) / √5     // Fibonacci                        ║ ║
║  ║   L_acc = Φ_acc + Ψ_acc             // Lucas                           ║ ║
║  ║                                                                         ║ ║
║  ╚═════════════════════════════════════════════════════════════════════════╝ ║
║                                     │                                         ║
║                                     ▼                                         ║
║  ╔═════════════════════════════════════════════════════════════════════════╗ ║
║  ║                    STEP 5: EMIT STATE                                   ║ ║
║  ╠═════════════════════════════════════════════════════════════════════════╣ ║
║  ║                                                                         ║ ║
║  ║   state[pos] = {                                                        ║ ║
║  ║       A     : A,          // Bit topology                               ║ ║
║  ║       Φ     : Φ_acc,      // φ-eigenvalue accumulator                  ║ ║
║  ║       Ψ     : Ψ_acc,      // ψ-eigenvalue accumulator (signed)         ║ ║
║  ║       C     : C,          // Cross-coupling with previous state         ║ ║
║  ║       τ     : τ_local     // Cascade work                               ║ ║
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

## §7 Attention with Parity Coupling

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PARITY-AWARE ATTENTION                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  For positions i, j with states Sᵢ = (Aᵢ, Φᵢ, Ψᵢ), Sⱼ = (Aⱼ, Φⱼ, Ψⱼ):       ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   SHELL DISTANCE (topology):                                            │ ║
║  │   ──────────────────────────                                            │ ║
║  │       d = POPCOUNT[ Aᵢ ⊕ Aⱼ ]                                          │ ║
║  │                                                                         │ ║
║  │   CROSS-COUPLING (eigenvalue):                                          │ ║
║  │   ────────────────────────────                                          │ ║
║  │       C_ij = Ψᵢ × Ψⱼ                                                   │ ║
║  │                                                                         │ ║
║  │   COMBINED ATTENTION:                                                   │ ║
║  │   ───────────────────                                                   │ ║
║  │       α_ij = f(d) × g(C_ij) × CAUSAL[i,j]                              │ ║
║  │                                                                         │ ║
║  │       where:                                                            │ ║
║  │           f(d) = φ⁻ᵈ or LUT[MAX-d]    (distance decay)                 │ ║
║  │           g(C) = 1 + λ·sign(C)·|C|^α  (coupling modulation)            │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   INTERPRETATION:                                                       │ ║
║  │                                                                         │ ║
║  │   • Shell distance d: How different are the topologies?                │ ║
║  │   • Cross-coupling C: Do the ψ-parities align or oppose?               │ ║
║  │                                                                         │ ║
║  │   Low d + high C  →  Strong positive attention (similar, reinforcing)  │ ║
║  │   Low d + low C   →  Moderate attention (similar, independent)         │ ║
║  │   Low d + neg C   →  Reduced attention (similar, opposing)             │ ║
║  │   High d + any C  →  Weak attention (distant)                           │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §8 Token Generation with Coupling

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PARITY-COUPLED GENERATION                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Current state: S = (A, Φ, Ψ, τ_history)                                     ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   FOR each candidate token t ∈ vocabulary:                              │ ║
║  │                                                                         │ ║
║  │       bits_t = ZECK[VOCAB[t]]                                           │ ║
║  │       Ψ_t = Σₚ∈bits_t PSI[p]                                            │ ║
║  │                                                                         │ ║
║  │       // Cross-coupling with current state                              │ ║
║  │       C_t = Ψ_t × Ψ_acc                                                 │ ║
║  │                                                                         │ ║
║  │       // Simulate cascade                                               │ ║
║  │       (A', τ_t) = APPLY_CASCADE(A, bits_t)                              │ ║
║  │                                                                         │ ║
║  │       // Attention to context (includes parity)                         │ ║
║  │       attn_t = Σⱼ α(S'_t, state[j])                                     │ ║
║  │                                                                         │ ║
║  │       // SCORE incorporates coupling                                    │ ║
║  │       score_t = f(τ_t, attn_t, C_t, freq[t])                           │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                         │ ║
║  │   SCORING WITH COUPLING:                                                │ ║
║  │   ──────────────────────                                                │ ║
║  │                                                                         │ ║
║  │   score_t = base_score × (1 + β·C_t)                                    │ ║
║  │                                                                         │ ║
║  │   where:                                                                │ ║
║  │       base_score = attention × frequency × cascade_penalty              │ ║
║  │       β = coupling strength parameter                                   │ ║
║  │       C_t = Ψ_t × Ψ_acc (can be positive or negative)                  │ ║
║  │                                                                         │ ║
║  │   Tokens with positive coupling → boosted                               │ ║
║  │   Tokens with negative coupling → suppressed                            │ ║
║  │   Tokens with zero coupling → neutral                                   │ ║
║  │                                                                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §9 Data Flow Summary

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          DATA FLOW                                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║                           INPUT TOKEN                                         ║
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
║              ┌───────────────┼───────────────┐                               ║
║              ▼               ▼               ▼                               ║
║      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                        ║
║      │  Σ PHI[p]   │ │  Σ PSI[p]   │ │   CASCADE   │                        ║
║      │ (φ-accum)   │ │ (ψ-accum)   │ │   (τ count) │                        ║
║      └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                        ║
║             │               │               │                                ║
║             └───────────────┼───────────────┘                                ║
║                             │                                                ║
║                             ▼                                                ║
║              ┌──────────────────────────────┐                                ║
║              │      CROSS-COUPLING          │                                ║
║              │   C = Ψ_token × Ψ_state      │                                ║
║              └──────────────┬───────────────┘                                ║
║                             │                                                ║
║              ┌──────────────┴──────────────┐                                 ║
║              ▼                             ▼                                 ║
║      ┌─────────────┐               ┌─────────────┐                          ║
║      │  ATTENTION  │               │ GENERATION  │                          ║
║      │             │               │             │                          ║
║      │  d + C_ij   │               │  score(τ,   │                          ║
║      │  weighting  │               │   attn, C)  │                          ║
║      └─────────────┘               └──────┬──────┘                          ║
║                                           │                                  ║
║                                           ▼                                  ║
║                                    OUTPUT TOKEN                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## §10 Key Insight Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE PARITY STRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   φ and ψ are eigenvalues. Everything else derives from them.               │
│                                                                              │
│   WITHIN TOKEN:                                                              │
│       Fₙ = (φⁿ - ψⁿ)/√5                                                     │
│       The ψⁿ term is the correction (vanishes for high n)                   │
│       This is handled automatically by Binet structure ✓                    │
│                                                                              │
│   ACROSS TOKENS:                                                             │
│       ψᵃ × ψᵇ = ψᵃ⁺ᵇ = (-1)^(a+b) / φ^(a+b)                                │
│                                                                              │
│       Parity of (a+b) determines correlation:                               │
│           EVEN → positive coupling (reinforce)                              │
│           ODD  → anti-correlation (oppose)                                  │
│                                                                              │
│       This was missing. Now it's the cross-coupling term C.                 │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   STATE = (A, Φ, Ψ, τ)                                                      │
│                                                                              │
│       A : bit topology (Zeckendorf)                                         │
│       Φ : φ-accumulator = Σ φᵖ                                              │
│       Ψ : ψ-accumulator = Σ ψᵖ (signed!)  ← THE KEY                        │
│       τ : cascade history                                                    │
│                                                                              │
│   Cross-coupling C = Ψ_new × Ψ_old                                          │
│                                                                              │
│   Fibonacci: F = (Φ - Ψ)/√5  (derived)                                      │
│   Lucas:     L = Φ + Ψ        (derived)                                      │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│   The ψ-parity matrix is the control variate for variance reduction.        │
│   Lévy flight samples tokens. ψ-coupling weights their joint contribution.  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
