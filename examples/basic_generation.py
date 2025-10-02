"""
Basic generation example for Φ-Mamba
Demonstrates natural termination and phase-locked generation
"""

import sys
sys.path.append('..')

from phi_mamba import PhiLanguageModel


def main():
    print("=== Φ-Mamba Basic Generation Demo ===\n")
    
    # Initialize model
    model = PhiLanguageModel(vocab_size=50000)
    print("Model initialized with φ-based encoding\n")
    
    # Example 1: Simple generation
    print("Example 1: Simple Generation")
    print("-" * 40)
    
    prompt1 = "The cat"
    print(f"Prompt: '{prompt1}'")
    
    generated1 = model.generate(prompt1, max_length=20, temperature=0.8)
    print(f"Generated: '{generated1}'")
    print()
    
    # Example 2: Longer prompt
    print("Example 2: Longer Prompt")
    print("-" * 40)
    
    prompt2 = "The cat sat on the"
    print(f"Prompt: '{prompt2}'")
    
    generated2 = model.generate(prompt2, max_length=20, temperature=0.8)
    print(f"Generated: '{generated2}'")
    print()
    
    # Example 3: Demonstrating natural termination
    print("Example 3: Natural Termination (watch energy decay)")
    print("-" * 40)
    
    prompt3 = "The dog"
    print(f"Prompt: '{prompt3}'")
    
    # Generate with verbose mode to see energy decay
    states = model.encode(prompt3, retrocausal=True)
    generated_states = []
    
    print("\nGeneration process:")
    for i in range(20):
        # Manually call generation to see energy
        next_state = model._generate_next(states + generated_states, temperature=0.8)
        
        if next_state is None:
            print(f"Step {i+1}: Natural termination (energy exhausted)")
            break
            
        print(f"Step {i+1}: '{next_state.token}' (energy={next_state.energy:.4f})")
        generated_states.append(next_state)
        
        if next_state.token in ['.', '!', '?']:
            print(f"Step {i+1}: Punctuation boundary reached")
            break
    
    final_text = prompt3 + " " + " ".join([s.token for s in generated_states])
    print(f"\nFinal: '{final_text}'")
    print()
    
    # Example 4: Temperature comparison
    print("Example 4: Temperature Effects")
    print("-" * 40)
    
    prompt4 = "The"
    print(f"Prompt: '{prompt4}'")
    
    print("\nTemperature = 0 (greedy):")
    gen_greedy = model.generate(prompt4, max_length=15, temperature=0)
    print(f"  '{gen_greedy}'")
    
    print("\nTemperature = 0.5 (focused):")
    gen_focused = model.generate(prompt4, max_length=15, temperature=0.5)
    print(f"  '{gen_focused}'")
    
    print("\nTemperature = 1.0 (balanced):")
    gen_balanced = model.generate(prompt4, max_length=15, temperature=1.0)
    print(f"  '{gen_balanced}'")
    
    print("\nTemperature = 2.0 (creative):")
    gen_creative = model.generate(prompt4, max_length=15, temperature=2.0)
    print(f"  '{gen_creative}'")
    
    # Example 5: Perplexity calculation
    print("\nExample 5: Perplexity Calculation")
    print("-" * 40)
    
    test_sentence = "The cat sat on the mat"
    perplexity = model.compute_perplexity(test_sentence)
    print(f"Test sentence: '{test_sentence}'")
    print(f"Perplexity: {perplexity:.2f}")
    print()
    
    # Key insights
    print("Key Observations:")
    print("-" * 40)
    print("1. Generation stops naturally when energy depletes")
    print("2. No explicit [EOS] token needed")
    print("3. Phase-locked transitions create coherent sequences")
    print("4. Temperature controls exploration vs exploitation")
    print("5. Retrocausal encoding improves coherence")


if __name__ == "__main__":
    main()