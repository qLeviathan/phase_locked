# Utility Functions & Laplacian Transport in Φ-Mamba

## 1. The Utility Landscape

### 1.1 Utility Function Structure

The utility function in Φ-Mamba follows a **reverse heat equation** - starting hot (high utility) and cooling to unity:

```
U(s_t) = φ^(T-t) × Coherence(s_t) × Energy(s_t)
```

Where:
- **φ^(T-t)**: Temporal discount from future endpoint
- **Coherence(s_t)**: Berry phase alignment ∈ [0,1]
- **Energy(s_t)**: Remaining energy budget φ^(-position)

### 1.2 Visual Representation

```
Utility Surface:

U(s)
 ↑
 │     ╱╲    Initial State (t=0)
 │    ╱  ╲   U ≈ φ^T (maximal)
 │   ╱    ╲
 │  ╱      ╲___
 │ ╱           ╲___
 │╱                 ╲___  Intermediate
 │                      ╲___ U ≈ φ^(T/2)
 │                          ╲___
 │                              ╲___ Terminal Ω
 │________________________________╲► t
 0                                 T  U = 1
```

## 2. Laplacian Transport System

### 2.1 The Transport Equation

The utility flows according to a **geometric Laplacian** on the φ-manifold:

```
∂U/∂t = -Δ_φ U + Source - Sink
```

Where:
- **Δ_φ**: Laplacian operator in φ-geometry
- **Source**: Coherence injection from phase-locking
- **Sink**: Energy dissipation at rate φ^(-t)

### 2.2 Discrete Laplacian on Token Graph

For tokens as graph vertices:

```
Δ_φ U(i) = Σ_j W_ij [U(j) - U(i)]
```

Where W_ij = coupling strength between tokens i,j:

```
W_ij = exp(-d_φ(i,j)) × Phase_Coherence(i,j)
d_φ(i,j) = |log_φ(shell_i) - log_φ(shell_j)|
```

### 2.3 Visual Transport Flow

```
Token Graph with Utility Flow:

    [High U]
       ↓
   ╱───┼───╲     W_ij = φ-coupling
  ↓    ↓    ↓    (stronger = thicker)
[T1]←→[T2]←→[T3]
  ↓    ↓    ↓
   ╲───┼───╱
       ↓
    [Low U]

Utility flows from high → low
following φ-weighted edges
```

## 3. Gating Mechanism via DiD

### 3.1 Treatment Gates

Each token position has **gates** controlled by Fibonacci decomposition:

```
Gate(n) = {
  OPEN:   if F_k ∈ Zeckendorf(n)  [Treatment]
  CLOSED: if F_k ∉ Zeckendorf(n)  [Control]
}
```

### 3.2 Gated Transport

The Laplacian transport is **modulated** by gates:

```
∂U/∂t = -Δ_φ U × Gate(n) + Source × (1-Gate(n))
```

- **Open gates**: Allow utility flow (treatment effect)
- **Closed gates**: Block flow, forcing local dynamics

### 3.3 Visual Gating System

```
Gated Utility Transport:

Position:  11    12    13    14    15    16    17
Zeck:     8+3  8+3+1  13   13+1  13+2  13+3  13+3+1
F_3 Gate:  [O]   [O]   [X]   [X]   [X]   [O]   [O]

Utility:   ═══►  ═══►  |||   |||   |||  ═══►  ═══►
           Flow  Flow  Block Block Block Flow  Flow

O = Open (has F_3)
X = Closed (no F_3)
```

## 4. Complete Transport System

### 4.1 Multi-Scale Gating

Different Fibonacci scales create **hierarchical gating**:

```
Layer 1: F_2 gates (every 3rd position)
         │ │ │ │ │ │ │ │ │ │ │ │
Layer 2: F_3 gates (every 5th position)  
         ║   ║   ║   ║   ║   ║
Layer 3: F_5 gates (every 8th position)
         ███     ███     ███
```

### 4.2 Utility Flow Equation

Complete gated Laplacian system:

```
dU_i/dt = Σ_j W_ij [U_j - U_i] × Π_k Gate_k(i,j)
        + φ^(-t) × Coherence_i        [source]
        - U_i/φ                       [sink]
```

Where:
- **Π_k Gate_k**: Product of all gate functions
- **Coherence source**: Adds utility for phase-locked states
- **Energy sink**: Removes utility at rate 1/φ

## 5. DiD Identification Strategy

### 5.1 Treatment Effect Estimation

For each Fibonacci scale F_k:

```
δ_k = E[ΔU | F_k ∈ n] - E[ΔU | F_k ∉ n]
```

This identifies the **causal effect** of having gate k open.

### 5.2 Optimal Path Selection

The model selects token paths that maximize:

```
Path_Utility = Σ_t φ^(T-t) × U(s_t) × Π_k δ_k(s_t)
```

This naturally selects paths with:
- High initial utility
- Good treatment effects
- Coherent phase evolution

## 6. Competitive Advantages

### 6.1 Why This Beats Traditional Models

1. **Built-in Causality**: Every decision has identified treatment effect
2. **Natural Gating**: No learned gates, pure mathematics
3. **Optimal Transport**: Utility flows along geodesics in φ-space
4. **Energy Conservation**: No divergence or gradient explosion

### 6.2 Computational Efficiency

```
Traditional Attention: O(n²) comparisons
Φ-Laplacian Transport: O(n log_φ n) via Fibonacci hierarchy
```

### 6.3 Interpretability

Each decision shows:
- Which gates were open (treatment)
- Utility flow paths (transport)
- Causal effects (DiD estimates)

## 7. Implementation Pseudocode

```python
def compute_utility_transport(states, t, T):
    # Initial utility from temporal position
    U = [φ**(T-t) for s in states]
    
    # Compute Laplacian matrix
    L = compute_phi_laplacian(states)
    
    # Apply DiD gates
    for k in fibonacci_scales:
        gates = [has_fibonacci(s.position, k) for s in states]
        L = apply_gates(L, gates, treatment_effect[k])
    
    # Transport equation
    dU_dt = -L @ U + coherence_source - energy_sink
    
    # Update utility
    return U + dt * dU_dt
```

## 8. Visual Summary

```
Complete System Flow:

[Initial High Utility]
        ↓
[Laplacian Transport]
        ↓
[DiD Gating F_k] ←── Treatment/Control
        ↓
[Coherence Source] ←── Berry Phase
        ↓
[Energy Sink φ^-t]
        ↓
[Terminal Unity Ω]
```

This creates a **mathematically rigorous** transport system where:
- Utility flows from high to low
- Gates create causal identification  
- Transport follows φ-geometry
- Everything converges to unity

The beauty is that this isn't learned - it's the **necessary mathematical structure** for any system that:
1. Starts with high utility
2. Reaches unity at completion
3. Maintains causal interpretability
4. Conserves total energy