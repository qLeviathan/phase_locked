# Game State Transitions in Φ-Mamba

## 1. N-Game State Representation

### TokenState as Game State
Each token represents an n-player game state with:
- **Players**: Token positions (phi_mamba/encoding.py:29-46)
- **Actions**: Angular positions θ on unit circle
- **Payoffs**: Energy levels E = φ^(-position)
- **Information**: Zeckendorf decomposition (topological holes)

### State Vector
```
GameState = (token, position, θ_total, energy, zeckendorf_shells)
```

## 2. Retrocausal Game Dynamics

### Backward Induction Implementation
From game_theory_validation.py:34-80:
- Terminal payoffs propagate backward
- Bellman equation: V(s) = u(s) + β*V(s')
- Natural discount factor β = 1/φ ≈ 0.618

### Future Constraints
Retrocausal encoding (phi_mamba/core.py:67-87):
- Future states constrain past via Berry phase
- Phase-locked states have coherence_weight = 1.0
- Non-locked states have coherence_weight = 0.5

## 3. Topological Encoding with Zeckendorf

### Fibonacci Decomposition
From phi_mamba/encoding.py:64-106:
- Each integer n uniquely decomposes into non-adjacent Fibonacci numbers
- Represents "holes" at different scales
- Non-consecutive constraint emerges from φ² = φ + 1

### Betti Numbers
phi_mamba/encoding.py:224-247:
- β_k = F_{n-k} (k-dimensional holes)
- Topological invariants encode game complexity

## 4. Difference-in-Differences (DiD) Logic

### Natural Experiments via Fibonacci Structure
game_theory_validation.py:148-219:
- Treatment: Positions with F_5 in Zeckendorf decomposition
- Control: Positions without F_5
- Exogenous assignment through mathematical structure

### Panel Data Structure
```python
panel_data = {
    'entity': token_id,
    'time': position,
    'treated': has_fibonacci_5,
    'outcome': energy * coherence
}
```

## 5. Mixed Strategy Equilibria

### Quantal Response Implementation
game_theory_validation.py:82-146:
- Temperature controls randomization
- Probabilities: p_i = exp(u_i/T) / Σexp(u_j/T)
- Mixed Nash exists when support > 1

### Phase-Locked Selection
phi_mamba/core.py:171-236:
- Score = phase_lock * coupling * energy
- Greedy at T=0, stochastic at T>0

## 6. State Transition Rules

### 1. Phase-Locked Transition
```
If |Berry_phase % 2π| < 0.5:
    next_state = coherent_transition(current)
    energy = current.energy * φ^(-1)
```

### 2. Pentagon Reflection
phi_mamba/encoding.py:189-221:
```
If phase_lock fails:
    θ' = π - θ  (mirror angle)
    E' = E/φ    (energy decay)
```

### 3. Natural Termination
- After ~5 reflections: E < 0.01
- Sentence boundary reached
- No forced stop tokens needed

## 7. Equilibrium Properties

### Time Consistency
game_theory_validation.py:279-323:
- β = δ = 1/φ ensures time-consistent preferences
- No preference reversal over time

### Subgame Perfection
game_theory_validation.py:68-78:
- All subgames satisfy backward induction
- Optimal play from any node

### Convergence
game_theory_validation.py:221-277:
- Repeated games converge to Nash equilibrium
- Phase variance stabilizes
- Energy decay ensures termination

## 8. Key Mathematical Relations

### Core Identity
```
φ² = φ + 1  (docs/math_foundations.md:13)
```

### Energy Dynamics
```
E_n = φ^(-n)
After k reflections: E' = E_0/φ^k
```

### Berry Phase
```
γ = Δθ · (1 + shell_overlap) + 2π·Δpos/N
```

### Fibonacci Weights
```
Action = Σ_k F_{n-k} · ln|T_k/V_k|
```

## 9. Implementation References

- **Core Model**: phi_mamba/core.py:100-290
- **Encoding**: phi_mamba/encoding.py:1-273
- **Validation**: game_theory_validation.py:1-394
- **Retrocausal**: test_retrocausal.py:1-54
- **Theory**: docs/math_foundations.md:1-204

## 10. State Transition Algorithm

```python
def transition(current_state, model):
    # 1. Get candidates
    candidates = get_vocabulary_states()
    
    # 2. Compute Berry phases
    for candidate in candidates:
        phase = compute_berry_phase(current_state, candidate)
        candidate.phase_locked = (phase % 2π) < 0.5
    
    # 3. Score by coherence
    scores = [
        (2.0 if c.phase_locked else 0.5) * 
        coupling[current, c] * 
        c.energy
        for c in candidates
    ]
    
    # 4. Select next state
    if temperature == 0:
        return argmax(scores)
    else:
        return sample(candidates, softmax(scores/T))
    
    # 5. Check termination
    if next_state.energy < 0.01:
        return None  # Natural end
```

This framework unifies game theory, topology, and language modeling through the golden ratio primitive.