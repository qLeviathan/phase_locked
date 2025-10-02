#!/usr/bin/env python3
"""
Quick test of retrocausal encoding functionality
"""

from phi_mamba import PhiLanguageModel

# Initialize model
model = PhiLanguageModel()

# Test sentence
sentence = "The golden ratio emerges naturally from recursive unity"

# Forward encoding
print("=== FORWARD ENCODING ===")
forward_states = model.encode(sentence, retrocausal=False)
print(f"Sentence: {sentence}")
print(f"Number of tokens: {len(forward_states)}")
print("\nFirst 3 tokens (forward only):")
for state in forward_states[:3]:
    print(f"  {state.token}: energy={state.energy:.4f}, "
          f"coherence={state.coherence_weight}")

# Retrocausal encoding
print("\n=== RETROCAUSAL ENCODING ===")
retro_states = model.encode(sentence, retrocausal=True)
print("\nFirst 3 tokens (with future constraints):")
for state in retro_states[:3]:
    print(f"  {state.token}: energy={state.energy:.4f}, "
          f"coherence={state.coherence_weight:.2f}")

# Show how energy decays
print("\n=== ENERGY DECAY ===")
print("Position | Token     | Energy    | Natural End?")
print("-" * 50)
for i, state in enumerate(retro_states):
    end_marker = "✓ END" if state.energy < 0.01 else ""
    print(f"{i:^8} | {state.token:^9} | {state.energy:^9.6f} | {end_marker}")

# Calculate average coherence improvement
forward_coherence = sum(s.coherence_weight for s in forward_states) / len(forward_states)
retro_coherence = sum(s.coherence_weight for s in retro_states) / len(retro_states)

print(f"\nAverage coherence:")
print(f"  Forward:     {forward_coherence:.3f}")
print(f"  Retrocausal: {retro_coherence:.3f}")
print(f"  Improvement: {(retro_coherence/forward_coherence - 1)*100:.1f}%")

# Generate a continuation
print("\n=== GENERATION TEST ===")
prompt = "The golden ratio"
generated = model.generate(prompt, max_length=20, temperature=0.8)
print(f"Prompt: '{prompt}'")
print(f"Generated: '{generated}'")