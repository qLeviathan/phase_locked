#!/usr/bin/env python3
"""
Letter Self-Organization Analysis using Phi-Mamba Framework

Demonstrates how letters self-organize for faster recall to make words,
using golden ratio principles and Fibonacci patterns.
"""

import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json
from phi_mamba.encoding import zeckendorf_decomposition
from phi_mamba.utils import PHI, fibonacci
import time

class LetterSelfOrganization:
    """Analyze letter patterns using golden ratio principles"""
    
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        self.words = []
        self.letter_freq = Counter()
        self.bigram_freq = Counter()
        self.letter_to_fib = {}
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_stats": {},
            "letter_analysis": {},
            "bigram_analysis": {},
            "fibonacci_patterns": {},
            "self_organization": {}
        }
    
    def load_corpus(self):
        """Load and preprocess word corpus"""
        print("Loading corpus...")
        with open(self.corpus_file, 'r') as f:
            self.words = [word.strip().lower() for word in f.readlines() if word.strip()]
        
        self.results["corpus_stats"] = {
            "total_words": len(self.words),
            "unique_words": len(set(self.words)),
            "avg_word_length": np.mean([len(w) for w in self.words])
        }
        print(f"Loaded {len(self.words)} words")
    
    def analyze_letter_frequencies(self):
        """Analyze individual letter frequencies"""
        print("\nAnalyzing letter frequencies...")
        
        # Count all letters
        for word in self.words:
            for letter in word:
                if letter.isalpha():
                    self.letter_freq[letter] += 1
        
        # Sort by frequency
        sorted_letters = sorted(self.letter_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Assign Fibonacci indices based on frequency rank
        for rank, (letter, freq) in enumerate(sorted_letters):
            fib_index = rank + 1  # Start from F_1
            self.letter_to_fib[letter] = {
                "rank": rank,
                "frequency": freq,
                "fibonacci_index": fib_index,
                "fibonacci_value": fibonacci(fib_index),
                "zeckendorf": zeckendorf_decomposition(ord(letter))
            }
        
        self.results["letter_analysis"] = {
            "frequency_distribution": dict(sorted_letters),
            "letter_to_fibonacci": self.letter_to_fib,
            "total_letters": sum(self.letter_freq.values())
        }
    
    def analyze_bigrams(self):
        """Analyze letter pair (bigram) frequencies"""
        print("Analyzing bigrams...")
        
        for word in self.words:
            for i in range(len(word) - 1):
                if word[i].isalpha() and word[i+1].isalpha():
                    bigram = word[i] + word[i+1]
                    self.bigram_freq[bigram] += 1
        
        # Top bigrams
        top_bigrams = sorted(self.bigram_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        
        self.results["bigram_analysis"] = {
            "top_50_bigrams": dict(top_bigrams),
            "total_bigrams": sum(self.bigram_freq.values()),
            "unique_bigrams": len(self.bigram_freq)
        }
    
    def analyze_fibonacci_patterns(self):
        """Find Fibonacci patterns in word structures"""
        print("Analyzing Fibonacci patterns...")
        
        fibonacci_word_lengths = defaultdict(list)
        fibonacci_position_patterns = []
        
        # Check if word lengths follow Fibonacci numbers
        fib_numbers = [fibonacci(i) for i in range(1, 20)]
        
        for word in self.words[:1000]:  # Analyze first 1000 words for efficiency
            word_len = len(word)
            
            # Check if word length is a Fibonacci number
            if word_len in fib_numbers:
                fib_index = fib_numbers.index(word_len) + 1
                fibonacci_word_lengths[word_len].append(word)
            
            # Analyze letter positions using golden ratio
            if len(word) > 3:
                positions = []
                for i, letter in enumerate(word):
                    if letter in self.letter_to_fib:
                        # Calculate phase based on position
                        phase = (i / len(word)) * 2 * np.pi
                        energy = PHI ** (-i)
                        positions.append({
                            "letter": letter,
                            "position": i,
                            "phase": phase,
                            "energy": energy,
                            "fibonacci_rank": self.letter_to_fib[letter]["fibonacci_index"]
                        })
                
                if positions:
                    fibonacci_position_patterns.append({
                        "word": word,
                        "pattern": positions
                    })
        
        self.results["fibonacci_patterns"] = {
            "fibonacci_length_words": {k: v[:10] for k, v in fibonacci_word_lengths.items()},
            "pattern_count": len(fibonacci_position_patterns),
            "sample_patterns": fibonacci_position_patterns[:5]
        }
    
    def analyze_self_organization(self):
        """Analyze how letters self-organize for efficient recall"""
        print("Analyzing self-organization patterns...")
        
        # Energy minimization in common words
        energy_profiles = []
        
        # Get most common words
        word_freq = Counter(self.words)
        common_words = [word for word, _ in word_freq.most_common(100)]
        
        for word in common_words:
            if len(word) > 2:
                total_energy = 0
                letter_energies = []
                
                for i, letter in enumerate(word):
                    if letter in self.letter_to_fib:
                        # Energy based on position and frequency rank
                        position_energy = PHI ** (-i)
                        freq_rank = self.letter_to_fib[letter]["rank"]
                        freq_energy = PHI ** (-freq_rank/26)  # Normalized by alphabet size
                        
                        combined_energy = position_energy * freq_energy
                        total_energy += combined_energy
                        
                        letter_energies.append({
                            "letter": letter,
                            "position": i,
                            "energy": combined_energy
                        })
                
                energy_profiles.append({
                    "word": word,
                    "total_energy": total_energy,
                    "avg_energy": total_energy / len(word),
                    "profile": letter_energies
                })
        
        # Sort by average energy (lower = more efficient)
        energy_profiles.sort(key=lambda x: x["avg_energy"])
        
        # Analyze phase coherence in word formation
        phase_coherent_words = []
        for word in common_words[:50]:
            if len(word) > 3:
                phases = []
                for i, letter in enumerate(word):
                    phase = (i / len(word)) * 2 * np.pi
                    phases.append(phase)
                
                # Calculate phase coherence (how evenly distributed)
                phase_diffs = np.diff(phases)
                coherence = 1 / (1 + np.std(phase_diffs))
                
                phase_coherent_words.append({
                    "word": word,
                    "coherence": coherence,
                    "phase_variance": np.var(phases)
                })
        
        phase_coherent_words.sort(key=lambda x: x["coherence"], reverse=True)
        
        self.results["self_organization"] = {
            "energy_efficient_words": energy_profiles[:10],
            "phase_coherent_words": phase_coherent_words[:10],
            "insights": {
                "avg_energy_common": np.mean([w["avg_energy"] for w in energy_profiles]),
                "avg_coherence": np.mean([w["coherence"] for w in phase_coherent_words])
            }
        }
    
    def visualize_results(self):
        """Create visualizations of the analysis"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Letter frequency distribution with Fibonacci overlay
        ax1 = axes[0, 0]
        letters = [item[0] for item in sorted(self.letter_freq.items(), key=lambda x: x[1], reverse=True)]
        frequencies = [self.letter_freq[l] for l in letters]
        
        ax1.bar(letters, frequencies)
        ax1.set_title('Letter Frequency Distribution')
        ax1.set_xlabel('Letters')
        ax1.set_ylabel('Frequency')
        
        # Overlay Fibonacci curve
        fib_curve = [fibonacci(i+1) * max(frequencies) / fibonacci(26) for i in range(26)]
        ax1.plot(range(26), fib_curve, 'r--', label='Fibonacci scaling')
        ax1.legend()
        
        # 2. Energy landscape of common words
        ax2 = axes[0, 1]
        energy_data = self.results["self_organization"]["energy_efficient_words"][:10]
        words = [d["word"] for d in energy_data]
        energies = [d["avg_energy"] for d in energy_data]
        
        ax2.bar(range(len(words)), energies)
        ax2.set_xticks(range(len(words)))
        ax2.set_xticklabels(words, rotation=45)
        ax2.set_title('Average Energy per Letter (Lower = More Efficient)')
        ax2.set_ylabel('Average Energy')
        
        # 3. Bigram network (top connections)
        ax3 = axes[1, 0]
        top_bigrams = list(self.results["bigram_analysis"]["top_50_bigrams"].items())[:20]
        bigram_names = [bg[0] for bg in top_bigrams]
        bigram_counts = [bg[1] for bg in top_bigrams]
        
        ax3.barh(bigram_names, bigram_counts)
        ax3.set_title('Top 20 Bigrams')
        ax3.set_xlabel('Frequency')
        
        # 4. Phase coherence distribution
        ax4 = axes[1, 1]
        coherence_data = self.results["self_organization"]["phase_coherent_words"]
        coherence_values = [d["coherence"] for d in coherence_data]
        
        ax4.hist(coherence_values, bins=20, edgecolor='black')
        ax4.set_title('Phase Coherence Distribution')
        ax4.set_xlabel('Coherence Score')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('rust_phi_mamba/notes/letter_self_organization.png', dpi=150)
        plt.close()
        
        print("Visualization saved to: rust_phi_mamba/notes/letter_self_organization.png")
    
    def save_results(self):
        """Save analysis results"""
        output_file = 'rust_phi_mamba/notes/letter_self_organization_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("=== Letter Self-Organization Analysis ===\n")
        
        self.load_corpus()
        self.analyze_letter_frequencies()
        self.analyze_bigrams()
        self.analyze_fibonacci_patterns()
        self.analyze_self_organization()
        self.visualize_results()
        self.save_results()
        
        # Print key insights
        print("\n=== Key Insights ===")
        print(f"1. Most frequent letter: '{list(self.letter_freq.most_common(1))[0][0]}' "
              f"({list(self.letter_freq.most_common(1))[0][1]} occurrences)")
        print(f"2. Most efficient word: '{self.results['self_organization']['energy_efficient_words'][0]['word']}' "
              f"(avg energy: {self.results['self_organization']['energy_efficient_words'][0]['avg_energy']:.4f})")
        print(f"3. Most phase-coherent word: '{self.results['self_organization']['phase_coherent_words'][0]['word']}' "
              f"(coherence: {self.results['self_organization']['phase_coherent_words'][0]['coherence']:.4f})")
        print(f"4. Found {len(self.results['fibonacci_patterns']['fibonacci_length_words'])} "
              f"different Fibonacci-length word groups")

if __name__ == "__main__":
    analyzer = LetterSelfOrganization('rust_phi_mamba/notes/words_10k.txt')
    analyzer.run_analysis()