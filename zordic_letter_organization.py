#!/usr/bin/env python3
"""
ZORDIC Letter Self-Organization
Demonstrates how letters organize using pure index operations
"""

import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json
import time

class ZordicCore:
    """Pure Python implementation of ZORDIC operations"""
    
    def __init__(self, max_shells=32):
        self.max_shells = max_shells
        self.fib = self._precompute_fibonacci()
    
    def _precompute_fibonacci(self):
        """Precompute Fibonacci numbers"""
        fib = [0, 1]
        while len(fib) < self.max_shells + 2:
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def encode(self, n):
        """Zeckendorf encoding: integer → index set"""
        if n == 0:
            return set()
        
        indices = set()
        # Start from largest Fibonacci number
        for k in range(self.max_shells - 1, -1, -1):
            fib_k = self.fib[k + 2]
            if fib_k <= n and fib_k > 0:
                indices.add(k)
                n -= fib_k
                if n == 0:
                    break
        
        return indices
    
    def decode(self, indices):
        """Zeckendorf decoding: index set → integer (Ω value)"""
        return sum(self.fib[k + 2] for k in indices)
    
    def cascade(self, indices):
        """CASCADE operation: resolve adjacent violations"""
        indices = set(indices)  # Make a copy
        
        while True:
            # Find violations (adjacent indices)
            violations = []
            sorted_indices = sorted(indices)
            for i in range(len(sorted_indices) - 1):
                if sorted_indices[i + 1] == sorted_indices[i] + 1:
                    violations.append((sorted_indices[i], sorted_indices[i + 1]))
            
            if not violations:
                break
            
            # Process first violation
            k1, k2 = violations[0]
            indices.remove(k1)
            indices.remove(k2)
            if k2 + 1 < self.max_shells:
                indices.add(k2 + 1)
        
        return indices
    
    def distance(self, indices_a, indices_b):
        """Fibonacci-weighted Hamming distance"""
        diff = indices_a.symmetric_difference(indices_b)
        return sum(self.fib[k + 2] for k in diff)
    
    def shift(self, indices, n):
        """Shift indices by n (multiply by φ^n)"""
        shifted = set()
        for k in indices:
            k_new = k + n
            if 0 <= k_new < self.max_shells:
                shifted.add(k_new)
        return self.cascade(shifted)


class ZordicLetterOrganization:
    """Letter self-organization using ZORDIC principles"""
    
    def __init__(self, corpus_file, max_shells=32):
        self.corpus_file = corpus_file
        self.zordic = ZordicCore(max_shells)
        self.words = []
        self.letter_indices = {}  # letter → index set
        self.bigram_indices = {}  # bigram → index set
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "zordic_analysis": {},
        }
    
    def load_corpus(self):
        """Load word corpus"""
        print("Loading corpus...")
        with open(self.corpus_file, 'r') as f:
            self.words = [word.strip().lower() for word in f.readlines() 
                         if word.strip() and len(word.strip()) < 20][:5000]
        print(f"Loaded {len(self.words)} words")
    
    def analyze_letter_encoding(self):
        """Encode letters using ZORDIC principles"""
        print("\nAnalyzing letter encodings...")
        
        # Count letter frequencies
        letter_freq = Counter()
        for word in self.words:
            for letter in word:
                if letter.isalpha():
                    letter_freq[letter] += 1
        
        # Assign indices based on frequency rank
        # Most frequent letters get smaller Ω values (more efficient)
        sorted_letters = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (letter, freq) in enumerate(sorted_letters):
            # Use Fibonacci numbers for encoding
            # More frequent letters get smaller numbers
            base_value = self.zordic.fib[rank + 2] if rank + 2 < len(self.zordic.fib) else rank * 5
            indices = self.zordic.encode(base_value)
            self.letter_indices[letter] = indices
            
            self.results["zordic_analysis"][letter] = {
                "frequency": freq,
                "rank": rank,
                "indices": list(indices),
                "omega": self.zordic.decode(indices),
                "energy": 1.0 / (rank + 1)  # Higher rank = lower energy
            }
    
    def analyze_word_patterns(self):
        """Analyze how letters combine in words using ZORDIC"""
        print("Analyzing word patterns...")
        
        word_patterns = []
        
        for word in self.words[:1000]:  # Analyze subset
            if len(word) < 2:
                continue
            
            # Encode word as sequence of index sets
            word_encoding = []
            total_indices = set()
            
            for letter in word:
                if letter in self.letter_indices:
                    indices = self.letter_indices[letter]
                    word_encoding.append(indices)
                    total_indices = total_indices.union(indices)
            
            # Check for violations in combined indices
            violations_before = self._count_violations(total_indices)
            cascaded = self.zordic.cascade(total_indices)
            violations_after = self._count_violations(cascaded)
            
            # Measure compression ratio
            omega_before = self.zordic.decode(total_indices)
            omega_after = self.zordic.decode(cascaded)
            compression = omega_after / omega_before if omega_before > 0 else 1.0
            
            word_patterns.append({
                "word": word,
                "length": len(word),
                "violations_before": violations_before,
                "violations_after": violations_after,
                "compression_ratio": compression,
                "indices_before": len(total_indices),
                "indices_after": len(cascaded)
            })
        
        # Sort by compression efficiency
        word_patterns.sort(key=lambda x: x["compression_ratio"])
        
        self.results["word_patterns"] = {
            "most_compressible": word_patterns[:10],
            "least_compressible": word_patterns[-10:],
            "avg_compression": np.mean([w["compression_ratio"] for w in word_patterns])
        }
    
    def analyze_bigram_cascades(self):
        """Analyze how letter pairs cascade"""
        print("Analyzing bigram cascades...")
        
        bigram_cascades = {}
        
        for word in self.words:
            for i in range(len(word) - 1):
                if word[i].isalpha() and word[i+1].isalpha():
                    bigram = word[i] + word[i+1]
                    
                    if bigram not in bigram_cascades:
                        # Combine indices of both letters
                        indices1 = self.letter_indices.get(word[i], set())
                        indices2 = self.letter_indices.get(word[i+1], set())
                        
                        # ZORDIC addition with cascade
                        combined = indices1.union(indices2)
                        cascaded = self.zordic.cascade(combined)
                        
                        bigram_cascades[bigram] = {
                            "indices1": list(indices1),
                            "indices2": list(indices2),
                            "combined": list(combined),
                            "cascaded": list(cascaded),
                            "distance": self.zordic.distance(indices1, indices2),
                            "compression": len(cascaded) / len(combined) if combined else 1.0
                        }
        
        # Find most interactive bigrams (most compression)
        sorted_bigrams = sorted(bigram_cascades.items(), 
                               key=lambda x: x[1]["compression"])
        
        self.results["bigram_cascades"] = {
            "most_interactive": dict(sorted_bigrams[:10]),
            "least_interactive": dict(sorted_bigrams[-10:])
        }
    
    def demonstrate_attention(self):
        """Demonstrate ZORDIC attention on common words"""
        print("Demonstrating ZORDIC attention...")
        
        # Select common 3-letter words
        three_letter_words = [w for w in self.words if len(w) == 3][:20]
        
        attention_results = []
        
        for word in three_letter_words:
            # Encode each position
            positions = []
            for letter in word:
                if letter in self.letter_indices:
                    positions.append(self.letter_indices[letter])
            
            if len(positions) == 3:
                # Compute pairwise distances (attention scores)
                distances = []
                for i in range(3):
                    row = []
                    for j in range(3):
                        dist = self.zordic.distance(positions[i], positions[j])
                        row.append(-dist)  # Negative distance = similarity
                    distances.append(row)
                
                # Find attention pattern (winner-take-all)
                attention_pattern = []
                for row in distances:
                    winner = row.index(max(row))
                    attention_pattern.append(winner)
                
                attention_results.append({
                    "word": word,
                    "pattern": attention_pattern,
                    "self_attention": sum(1 for i, w in enumerate(attention_pattern) if i == w)
                })
        
        self.results["attention_demo"] = attention_results
    
    def _count_violations(self, indices):
        """Count adjacent index violations"""
        sorted_indices = sorted(indices)
        violations = 0
        for i in range(len(sorted_indices) - 1):
            if sorted_indices[i + 1] == sorted_indices[i] + 1:
                violations += 1
        return violations
    
    def visualize_results(self):
        """Create visualizations"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Letter encoding efficiency
        ax1 = axes[0, 0]
        letters = sorted(self.letter_indices.keys(), 
                        key=lambda x: len(self.letter_indices[x]))
        indices_counts = [len(self.letter_indices[l]) for l in letters]
        
        ax1.bar(range(26), indices_counts[:26])
        ax1.set_xticks(range(26))
        ax1.set_xticklabels(letters[:26])
        ax1.set_title('Letter Encoding Efficiency (Index Count)')
        ax1.set_ylabel('Number of Indices')
        
        # 2. Word compression distribution
        ax2 = axes[0, 1]
        compressions = [w["compression_ratio"] for w in self.results["word_patterns"]["most_compressible"]]
        words = [w["word"] for w in self.results["word_patterns"]["most_compressible"]]
        
        ax2.barh(range(len(words)), compressions)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words)
        ax2.set_title('Most Compressible Words')
        ax2.set_xlabel('Compression Ratio')
        
        # 3. Bigram interaction heatmap
        ax3 = axes[1, 0]
        # Create a small heatmap of bigram distances
        common_letters = 'etaoin'
        distance_matrix = []
        for l1 in common_letters:
            row = []
            for l2 in common_letters:
                if l1 in self.letter_indices and l2 in self.letter_indices:
                    dist = self.zordic.distance(self.letter_indices[l1], 
                                               self.letter_indices[l2])
                    row.append(dist)
                else:
                    row.append(0)
            distance_matrix.append(row)
        
        im = ax3.imshow(distance_matrix, cmap='viridis')
        ax3.set_xticks(range(len(common_letters)))
        ax3.set_yticks(range(len(common_letters)))
        ax3.set_xticklabels(list(common_letters))
        ax3.set_yticklabels(list(common_letters))
        ax3.set_title('Letter Pair Distances (ZORDIC)')
        plt.colorbar(im, ax=ax3)
        
        # 4. Attention patterns
        ax4 = axes[1, 1]
        if "attention_demo" in self.results:
            self_attention_counts = [r["self_attention"] for r in self.results["attention_demo"]]
            ax4.hist(self_attention_counts, bins=[0, 1, 2, 3, 4], edgecolor='black')
            ax4.set_title('Self-Attention Distribution in 3-Letter Words')
            ax4.set_xlabel('Number of Self-Attention Positions')
            ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('rust_phi_mamba/zordic/docs/zordic_letter_organization.png', dpi=150)
        plt.close()
        
        print("Visualization saved!")
    
    def save_results(self):
        """Save analysis results"""
        output_file = 'rust_phi_mamba/zordic/docs/zordic_letter_analysis.json'
        with open(output_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            json.dump(self.results, f, indent=2, default=list)
        print(f"Results saved to: {output_file}")
    
    def run_analysis(self):
        """Run complete ZORDIC analysis"""
        print("=== ZORDIC Letter Self-Organization Analysis ===\n")
        
        self.load_corpus()
        self.analyze_letter_encoding()
        self.analyze_word_patterns()
        self.analyze_bigram_cascades()
        self.demonstrate_attention()
        self.visualize_results()
        self.save_results()
        
        # Print insights
        print("\n=== ZORDIC Insights ===")
        print(f"1. Average word compression: {self.results['word_patterns']['avg_compression']:.3f}")
        print(f"2. Most compressible word: '{self.results['word_patterns']['most_compressible'][0]['word']}'")
        print(f"3. Most interactive bigram: '{list(self.results['bigram_cascades']['most_interactive'].keys())[0]}'")
        print("\nZORDIC demonstrates natural compression through cascade operations!")

if __name__ == "__main__":
    analyzer = ZordicLetterOrganization('rust_phi_mamba/notes/words_10k.txt')
    analyzer.run_analysis()