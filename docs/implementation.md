# Implementation Guide

This guide walks through implementing Φ-Mamba from scratch.

## Core Concepts to Implement

### 1. Golden Ratio Constants
```python
PHI = (1 + sqrt(5)) / 2  # 1.618...
PSI = -1 / PHI           # -0.618...
```

### 2. State Representation
Each token becomes a state on the cylinder:
```python
@dataclass
class TokenState:
    token: str           # Actual token
    index: int          # Vocabulary index
    position: int       # Position in sequence
    theta: float        # Angular position
    r: float           # Amplitude (φ-based)
    z: int             # Causal layer
    energy: float      # φ^(-position)
    shells: List[int]  # Zeckendorf decomposition
```

### 3. Key Algorithms

#### Zeckendorf Decomposition
Every integer uniquely decomposes into non-adjacent Fibonacci numbers:
```python
def zeckendorf(n):
    # Greedy algorithm
    # Take largest Fibonacci ≤ n
    # Skip next Fibonacci (gap constraint)
    # Repeat
```

#### Berry Phase Calculation
```python
def berry_phase(state1, state2):
    # Angular difference
    d_theta = state2.theta - state1.theta
    
    # Shell overlap factor
    overlap = len(shells1 ∩ shells2) / max(|shells1|, |shells2|)
    
    # Combined phase
    gamma = d_theta * (1 + overlap)
    
    return gamma % (2*pi)
```

#### Retrocausal Encoding
Start from the END and work backward:
```python
def retrocausal_encode(tokens):
    states = forward_encode(tokens)
    
    # Work backward
    for i in range(len(states)-1, 0, -1):
        future = states[i]
        past = states[i-1]
        
        # Future constrains past
        constraint = compute_constraint(future, past)
        past.future_constraint = constraint
```

### 4. Generation Pipeline

```python
def generate_next_token(context):
    last_state = context[-1]
    
    # Check termination
    if last_state.energy < 0.01:
        return None  # Natural termination
    
    candidates = []
    for token in vocabulary:
        # Create candidate state
        candidate = create_state(token, last_state.position + 1)
        
        # Calculate Berry phase
        gamma = berry_phase(last_state, candidate)
        phase_locked = (gamma % 2π) < tolerance
        
        # Score based on phase lock + coupling
        score = phase_lock_bonus * coupling * energy
        candidates.append((candidate, score))
    
    # Select best or sample
    return select_token(candidates)
```

### 5. Pentagon Reflection
When no phase-locked candidates exist:
```python
def pentagon_reflect(state):
    # Mirror angle
    state.theta = π - state.theta
    
    # Scale energy down
    state.energy = state.energy / PHI
    
    # After ~5 reflections, energy < threshold
    return state
```

## Architecture Decisions

### 1. Memory Representation
Store states, not embeddings:
- Traditional: Token → Embedding vector
- Φ-Mamba: Token → (n, θ, r, z) state

### 2. Attention Mechanism
Replace with phase coupling:
- Traditional: Q·K^T attention
- Φ-Mamba: Berry phase coherence

### 3. Position Encoding
Golden ratio decay instead of sinusoidal:
- Traditional: sin/cos at different frequencies
- Φ-Mamba: φ^(-position) natural decay

### 4. Training Objective
Minimize phase deviation:
```
L = Σ |γ_i mod 2π| + λ·L_classical
```

## Implementation Steps

### Phase 1: Core Mathematics (Week 1)
1. Implement golden ratio arithmetic
2. Create Fibonacci/Lucas generators
3. Build Zeckendorf decomposition
4. Verify all identities hold

### Phase 2: Encoding System (Week 2)
1. Design TokenState class
2. Implement forward encoding
3. Add retrocausal constraints
4. Create cylinder coordinate mapping

### Phase 3: Generation Engine (Week 3)
1. Berry phase calculator
2. Phase-lock detection
3. Pentagon reflection
4. Natural termination logic

### Phase 4: Full Model (Week 4)
1. Integrate all components
2. Add training loop
3. Implement beam search
4. Create evaluation metrics

## Optimization Tips

### 1. Precompute Tables
```python
# At initialization
P_TABLE = {n: PHI**n for n in range(-100, 101)}
M_TABLE = {n: PSI**n for n in range(-100, 101)}
FIB_TABLE = {n: fibonacci(n) for n in range(100)}
```

### 2. Integer-Only Arithmetic
Since φ^n × φ^m = φ^(n+m), store exponents:
```python
# Instead of: value1 * value2
# Do: exp1 + exp2, then lookup
```

### 3. Vectorize Berry Phase
Calculate phase relationships in batch using NumPy.

### 4. Cache Zeckendorf Decompositions
These are used repeatedly - cache first 10,000.

## Common Pitfalls

1. **Floating Point Errors**: Use integer arithmetic wherever possible
2. **Energy Underflow**: Cap minimum energy at 1e-10
3. **Phase Wrapping**: Always use modulo 2π for phases
4. **Gap Constraint**: Verify Zeckendorf has no adjacent terms

## Testing Strategy

1. **Unit Tests**: Each mathematical identity
2. **Property Tests**: Invariants (energy decay, phase bounds)
3. **Integration Tests**: Full generation pipeline
4. **Comparison Tests**: vs GPT-2 baseline

## Next Steps

After basic implementation:
1. Scale to full vocabulary (50k+ tokens)
2. Implement efficient training on large corpus
3. Add multi-layer architecture
4. Optimize with CUDA kernels
5. Benchmark against standard transformers