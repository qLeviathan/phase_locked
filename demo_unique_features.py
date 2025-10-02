#!/usr/bin/env python3
"""
Demo of Φ-Mamba's unique features
"""

import numpy as np
from phi_mamba import PhiLanguageModel, PHI
from phi_mamba.generation import beam_search, retrocausal_reranking
from phi_mamba.encoding import zeckendorf_decomposition

print("=== Φ-MAMBA UNIQUE FEATURES DEMO ===\n")

# Initialize model
model = PhiLanguageModel()

# Feature 1: Natural termination without [EOS]
print("1. NATURAL TERMINATION (No [EOS] token needed)")
print("-" * 50)
prompt = "The cat"
print(f"Generating from: '{prompt}'")
print("\nWatch energy decay:")

states = model.encode(prompt)
generated_tokens = []

for i in range(20):
    next_state = model._generate_next(states + generated_tokens, temperature=0.8)
    
    if next_state is None:
        print(f"\n✓ Natural termination at position {len(states) + i}")
        print("  (Energy depleted - no forced cutoff!)")
        break
        
    generated_tokens.append(next_state)
    print(f"  Step {i+1}: '{next_state.token}' (E = {next_state.energy:.4f})")
    
    if next_state.token in ['.', '!', '?']:
        print(f"\n✓ Punctuation boundary at position {len(states) + i}")
        break

full_text = prompt + " " + " ".join([s.token for s in generated_tokens])
print(f"\nGenerated: '{full_text}'")

# Feature 2: Topological information storage
print("\n\n2. TOPOLOGICAL INFORMATION (Bits emerge from geometry)")
print("-" * 50)
print("Traditional: Store bits directly")
print("Φ-Mamba: Bits emerge from Zeckendorf topology\n")

for n in [15, 23, 42]:
    zeck = zeckendorf_decomposition(n)
    
    # Show which "holes" are active
    print(f"Number {n}:")
    print(f"  Zeckendorf: {' + '.join(map(str, zeck))}")
    print(f"  Active holes at scales: {zeck}")
    
    # Create "bit" representation
    max_fib = 50
    bits = []
    fib_n = 1
    fib_val = 1
    while fib_val <= max_fib:
        if fib_val in zeck:
            bits.append('1')
        else:
            bits.append('0')
        # Next Fibonacci
        fib_n += 1
        if fib_n <= 2:
            fib_val = fib_n
        else:
            fib_val = sum(zeck_decomposition(fib_n-1)) + sum(zeck_decomposition(fib_n-2)) if zeck_decomposition(fib_n-1) and zeck_decomposition(fib_n-2) else fib_n
    
    print(f"  Emergent bit pattern: {''.join(bits[:8])}...")
    print(f"  No adjacent 1s - gap constraint from φ² = φ + 1\n")

# Feature 3: Computation with addition only
print("\n3. ADDITION-ONLY COMPUTATION")
print("-" * 50)
print("Traditional: Multiply embeddings, compute exp(), etc.")
print("Φ-Mamba: Everything reduces to integer addition\n")

# Show some operations
operations = [
    ("Multiply", "φ³ × φ⁵", 3, 5, "3 + 5 = 8"),
    ("Divide", "φ⁷ / φ²", 7, -2, "7 + (-2) = 5"),
    ("Power", "(φ²)³", 2, 2, "2 + 2 + 2 = 6")
]

for op_name, expr, *args in operations:
    if op_name == "Multiply":
        n1, n2, addition = args
        result_exp = n1 + n2
        print(f"{op_name}: {expr} = φ⁸")
        print(f"  Traditional: {PHI**n1:.3f} × {PHI**n2:.3f} = {PHI**result_exp:.3f}")
        print(f"  Φ-Mamba: {addition} → φ⁸\n")
    elif op_name == "Divide":
        n1, n2, addition = args
        result_exp = n1 + n2
        print(f"{op_name}: {expr} = φ⁵")
        print(f"  Traditional: {PHI**n1:.3f} / {PHI**(-n2):.3f} = {PHI**result_exp:.3f}")
        print(f"  Φ-Mamba: {addition} → φ⁵\n")
    elif op_name == "Power":
        base, exp_val, addition = args
        result_exp = base * 3
        print(f"{op_name}: {expr} = φ⁶")
        print(f"  Traditional: {PHI**base:.3f}³ = {PHI**result_exp:.3f}")
        print(f"  Φ-Mamba: {addition} → φ⁶\n")

# Feature 4: Retrocausal coherence
print("\n4. RETROCAUSAL ENCODING (Future constrains past)")
print("-" * 50)
print("Traditional: Only use past context")
print("Φ-Mamba: Future tokens constrain past tokens\n")

# Use beam search to generate multiple sequences
prompt = "The"
beams = beam_search(model, model.encode(prompt), beam_width=3, max_length=10)

print(f"Generated {len(beams)} sequences from '{prompt}':")
for i, sequence in enumerate(beams):
    text = " ".join([s.token for s in sequence])
    print(f"  {i+1}. {text}")

# Rerank using retrocausal coherence
print("\nReranking with retrocausal coherence:")
ranked = retrocausal_reranking(model, beams)

for i, (sequence, score) in enumerate(ranked[:3]):
    text = " ".join([s.token for s in sequence])
    print(f"  {i+1}. {text} (coherence: {score:.3f})")

# Feature 5: Unity is not primitive
print("\n\n5. RECURSIVE UNITY (1 = φ² - φ)")
print("-" * 50)
print("Traditional: 0 and 1 are axioms")
print("Φ-Mamba: φ is the axiom, 1 emerges\n")

print(f"φ = {PHI:.10f} (the only primitive)")
print(f"1 = φ² - φ = {PHI**2:.10f} - {PHI:.10f} = {PHI**2 - PHI:.10f}")
print(f"0 = ln(φ⁰) = ln(1) = {np.log(1)}")
print("\nThis means every '1' in the system encodes φ-structure!")

# Summary
print("\n\n" + "="*50)
print("SUMMARY: What makes Φ-Mamba different")
print("="*50)
print("1. Natural termination through energy decay")
print("2. Information stored as topology, not bits")
print("3. All operations reduce to integer addition")
print("4. Future constrains past (retrocausal)")
print("5. Unity emerges from φ (recursive foundation)")
print("\nThis is a fundamentally different approach to computation!")