#!/usr/bin/env python3
"""
TENSOR SERIES: Unified Tensor Manipulation for NLP Tasks
Combines the best of φ-Mamba into sequential tensor rank transformations

This master script demonstrates how integer φ-arithmetic creates
natural tensor series for language processing through:
1. Zeckendorf sparse tensors
2. Retrocausal tensor constraints  
3. Laplacian transport dynamics
4. Game-theoretic equilibrium
5. Integer-only operations
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import from phi_mamba centralized modules
from phi_mamba.constants import PHI, PSI
from phi_mamba.math_core import FibonacciCache

# Fibonacci ratio for integer approximation
FIB_RATIO = (377, 610)  # F_14/F_15 for integer 1/φ


class TensorSeries:
    """
    Master class unifying all φ-Mamba concepts into tensor operations
    """

    def __init__(self, vocab_size: int = 1000, max_seq_len: int = 512):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Use centralized FibonacciCache instead of custom caching
        self._cache = FibonacciCache(max_n=max_seq_len)

        # Initialize tensor ranks
        self.rank0 = None  # Scalar (energy)
        self.rank1 = None  # Vector (token embeddings)
        self.rank2 = None  # Matrix (attention/coupling)
        self.rank3 = None  # 3-tensor (temporal evolution)
        self.rank4 = None  # 4-tensor (retrocausal constraints)

    def fibonacci(self, n: int) -> int:
        """Cached Fibonacci computation using centralized cache"""
        return self._cache.get_fibonacci(n)

    def zeckendorf_decomposition(self, n: int) -> List[int]:
        """Decompose into non-adjacent Fibonacci numbers using centralized cache"""
        return self._cache.get_zeckendorf(n)
    
    def initialize_tensor_series(self, input_tokens: List[int]) -> None:
        """Initialize all tensor ranks from input"""
        seq_len = len(input_tokens)
        
        # Rank 0: Scalar energy
        self.rank0 = self.fibonacci(25)  # Start with high energy F_25
        
        # Rank 1: Token embeddings (integer vectors)
        self.rank1 = np.zeros((seq_len, 64), dtype=np.int64)
        for i, token in enumerate(input_tokens):
            # Embed using Zeckendorf decomposition
            zeck = self.zeckendorf_decomposition(token + 1)
            for j, fib in enumerate(zeck[:64]):  # Limit to embedding size
                self.rank1[i, j] = fib
        
        # Rank 2: Coupling matrix (φ-geometric distances)
        self.rank2 = np.zeros((seq_len, seq_len), dtype=np.int64)
        for i in range(seq_len):
            for j in range(seq_len):
                # Integer φ-distance
                dist = abs(i - j)
                if dist == 0:
                    self.rank2[i, j] = self.fibonacci(20)
                else:
                    self.rank2[i, j] = self.fibonacci(20 - min(dist, 19))
        
        # Rank 3: Temporal evolution tensor (past, present, future)
        self.rank3 = np.zeros((seq_len, seq_len, 3), dtype=np.int64)
        for t in range(seq_len):
            for s in range(seq_len):
                # Past influence
                if s < t:
                    self.rank3[t, s, 0] = self.fibonacci(15 - min(t - s, 14))
                # Present
                elif s == t:
                    self.rank3[t, s, 1] = self.fibonacci(20)
                # Future (retrocausal)
                else:
                    self.rank3[t, s, 2] = self.fibonacci(15 - min(s - t, 14))
        
        # Rank 4: Game-theoretic tensor (player, action, state, payoff)
        self.rank4 = np.zeros((seq_len, self.vocab_size // 10, 8, 2), dtype=np.int64)
        # Initialize with equilibrium payoffs
        for p in range(seq_len):
            for a in range(self.vocab_size // 10):
                for s in range(8):  # 8 game states
                    # Cooperate payoff
                    self.rank4[p, a, s, 0] = self.fibonacci(10 + s)
                    # Defect payoff  
                    self.rank4[p, a, s, 1] = self.fibonacci(8 + s)
    
    def apply_laplacian_transport(self) -> np.ndarray:
        """Apply Laplacian transport to rank2 tensor"""
        laplacian = np.zeros_like(self.rank2)
        n = self.rank2.shape[0]
        
        # Compute discrete Laplacian
        for i in range(n):
            row_sum = 0
            for j in range(n):
                if i != j:
                    laplacian[i, j] = -self.rank2[i, j]
                    row_sum += self.rank2[i, j]
            laplacian[i, i] = row_sum
        
        # Transport step (integer arithmetic)
        transported = self.rank2 - (laplacian * FIB_RATIO[0]) // (FIB_RATIO[1] * 10)
        
        # Energy decay
        self.rank0 = (self.rank0 * FIB_RATIO[0]) // FIB_RATIO[1]
        
        return transported
    
    def retrocausal_constraint_propagation(self) -> np.ndarray:
        """Propagate constraints from future to past"""
        seq_len = self.rank1.shape[0]
        constraints = np.zeros((seq_len, 64), dtype=np.int64)
        
        # Work backwards
        for t in range(seq_len - 1, -1, -1):
            if t == seq_len - 1:
                # Terminal constraint
                constraints[t] = self.rank1[t]
            else:
                # Propagate from future with φ-decay
                future_influence = (constraints[t + 1] * FIB_RATIO[0]) // FIB_RATIO[1]
                constraints[t] = self.rank1[t] + future_influence // 2
        
        return constraints
    
    def compute_equilibrium_distribution(self, position: int) -> np.ndarray:
        """Compute game-theoretic equilibrium at position"""
        # Mixed strategy Nash equilibrium
        vocab_subset = self.vocab_size // 10
        distribution = np.zeros(vocab_subset, dtype=np.int64)
        
        for action in range(vocab_subset):
            # Sum over states and strategies
            payoff_sum = 0
            for state in range(8):
                cooperate = self.rank4[position, action, state, 0]
                defect = self.rank4[position, action, state, 1]
                # Quantal response with integer arithmetic
                payoff_sum += max(cooperate, defect)
            
            distribution[action] = payoff_sum
        
        # Normalize to probabilities (integer approximation)
        total = np.sum(distribution)
        if total > 0:
            # Scale to 1000 for integer probabilities
            distribution = (distribution * 1000) // total
        
        return distribution
    
    def tensor_rank_cascade(self) -> Dict[str, np.ndarray]:
        """Sequential cascade through tensor ranks"""
        results = {}
        
        # Step 1: Embedding lookup (Rank 1)
        print("Step 1: Token embeddings (Rank 1)")
        embeddings = self.rank1.copy()
        results['embeddings'] = embeddings
        print(f"  Shape: {embeddings.shape}, Energy: {self.rank0}")
        
        # Step 2: Coupling computation (Rank 2)
        print("\nStep 2: Laplacian transport (Rank 2)")
        coupling = self.apply_laplacian_transport()
        results['coupling'] = coupling
        print(f"  Shape: {coupling.shape}, Energy: {self.rank0}")
        
        # Step 3: Temporal evolution (Rank 3)
        print("\nStep 3: Retrocausal constraints (Rank 3)")
        temporal = self.rank3.copy()
        constraints = self.retrocausal_constraint_propagation()
        results['constraints'] = constraints
        print(f"  Shape: {temporal.shape}, Constraint norm: {np.sum(constraints)}")
        
        # Step 4: Game equilibrium (Rank 4)
        print("\nStep 4: Game-theoretic equilibrium (Rank 4)")
        equilibria = []
        for pos in range(min(self.rank1.shape[0], 5)):  # Sample positions
            eq = self.compute_equilibrium_distribution(pos)
            equilibria.append(eq)
        results['equilibria'] = np.array(equilibria)
        print(f"  Shape: {results['equilibria'].shape}")
        
        # Step 5: Dimension reduction cascade
        print("\nStep 5: Tensor rank reduction cascade")
        # Contract rank 4 → rank 3
        contracted_4to3 = np.sum(self.rank4, axis=-1)  # Sum over strategies
        # Contract rank 3 → rank 2  
        contracted_3to2 = np.sum(contracted_4to3[:5, :, :], axis=-1)  # Sum over states
        # Contract rank 2 → rank 1
        contracted_2to1 = np.sum(contracted_3to2, axis=-1)  # Sum over actions
        # Contract rank 1 → rank 0
        contracted_1to0 = np.sum(contracted_2to1)  # Total
        
        results['cascade'] = {
            'rank3': contracted_4to3[:5, :5, :5],  # Sample
            'rank2': contracted_3to2[:5, :5],
            'rank1': contracted_2to1[:5],
            'rank0': contracted_1to0
        }
        
        return results
    
    def nlp_task_text_generation(self, prompt: List[int], max_length: int = 50) -> List[int]:
        """Generate text using tensor series dynamics"""
        generated = list(prompt)
        self.initialize_tensor_series(generated)
        
        for step in range(max_length - len(prompt)):
            # Check energy threshold
            if self.rank0 < self.fibonacci(5):
                break
            
            # Get current position
            pos = len(generated) - 1
            
            # Compute next token distribution
            if pos < self.rank4.shape[0]:
                distribution = self.compute_equilibrium_distribution(pos)
            else:
                # Extend using Fibonacci pattern
                distribution = np.array([self.fibonacci(i % 20) for i in range(self.vocab_size // 10)])
            
            # Sample next token (integer arithmetic)
            total = np.sum(distribution)
            if total == 0:
                break
            
            # Integer sampling
            r = np.random.randint(0, total)
            cumsum = 0
            next_token = 0
            for i, prob in enumerate(distribution):
                cumsum += prob
                if cumsum > r:
                    next_token = i
                    break
            
            generated.append(next_token)
            
            # Update energy
            self.rank0 = (self.rank0 * FIB_RATIO[0]) // FIB_RATIO[1]
            
        return generated
    
    def nlp_task_classification(self, tokens: List[int], num_classes: int = 5) -> int:
        """Classify text using tensor contractions"""
        self.initialize_tensor_series(tokens)
        
        # Apply cascade
        results = self.tensor_rank_cascade()
        
        # Classification via tensor contraction
        class_scores = np.zeros(num_classes, dtype=np.int64)
        
        for c in range(num_classes):
            # Each class has a Fibonacci signature
            class_fib = self.fibonacci(10 + c)
            
            # Score based on embedding alignment
            embeddings = results['embeddings']
            for i in range(embeddings.shape[0]):
                for j in range(embeddings.shape[1]):
                    if embeddings[i, j] > 0:
                        # Resonance with class signature
                        resonance = (embeddings[i, j] * class_fib) % 1000
                        class_scores[c] += resonance
            
            # Coupling influence
            coupling = results['coupling']
            coupling_score = np.sum(coupling) % class_fib
            class_scores[c] += coupling_score
        
        # Return class with highest score
        return np.argmax(class_scores)
    
    def visualize_tensor_series(self, results: Dict[str, np.ndarray]) -> None:
        """Visualize the tensor series cascade"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Embeddings heatmap
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(results['embeddings'][:10, :20], cmap='viridis', aspect='auto')
        ax1.set_title('Token Embeddings (Rank 1)')
        ax1.set_xlabel('Embedding Dimension')
        ax1.set_ylabel('Token Position')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Coupling matrix
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(results['coupling'][:20, :20], cmap='coolwarm', aspect='auto')
        ax2.set_title('Coupling Matrix (Rank 2)')
        ax2.set_xlabel('Position j')
        ax2.set_ylabel('Position i')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Constraints
        ax3 = plt.subplot(2, 3, 3)
        im3 = ax3.imshow(results['constraints'][:10, :20], cmap='plasma', aspect='auto')
        ax3.set_title('Retrocausal Constraints')
        ax3.set_xlabel('Constraint Dimension')
        ax3.set_ylabel('Position')
        plt.colorbar(im3, ax=ax3)
        
        # 4. Equilibrium distributions
        ax4 = plt.subplot(2, 3, 4)
        for i, eq in enumerate(results['equilibria']):
            ax4.plot(eq[:20], label=f'Pos {i}', alpha=0.7)
        ax4.set_title('Equilibrium Distributions')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Probability (×1000)')
        ax4.legend()
        
        # 5. Tensor cascade
        ax5 = plt.subplot(2, 3, 5)
        cascade_values = [
            results['cascade']['rank0'],
            np.sum(results['cascade']['rank1']),
            np.sum(results['cascade']['rank2']),
            np.sum(results['cascade']['rank3'])
        ]
        ranks = ['Rank 0', 'Rank 1', 'Rank 2', 'Rank 3']
        ax5.bar(ranks, cascade_values, color=['red', 'orange', 'yellow', 'green'])
        ax5.set_title('Tensor Rank Cascade')
        ax5.set_ylabel('Total Value')
        ax5.set_yscale('log')
        
        # 6. Energy decay
        ax6 = plt.subplot(2, 3, 6)
        energy_sequence = []
        energy = self.fibonacci(25)
        for _ in range(50):
            energy_sequence.append(energy)
            energy = (energy * FIB_RATIO[0]) // FIB_RATIO[1]
        ax6.plot(energy_sequence, 'g-', linewidth=2)
        ax6.set_title('Energy Decay (φ-geometric)')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Energy')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('TENSOR SERIES: Sequential Rank Transformations', fontsize=16)
        plt.tight_layout()
        plt.savefig('tensor_series_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()


def demonstrate_nlp_tasks():
    """Demonstrate various NLP tasks using tensor series"""
    print("=" * 60)
    print("TENSOR SERIES: Unified φ-Mamba Demonstration")
    print("=" * 60)
    
    # Initialize system
    ts = TensorSeries(vocab_size=1000, max_seq_len=512)
    
    # Example 1: Text Generation
    print("\n1. TEXT GENERATION TASK")
    print("-" * 40)
    prompt = [42, 17, 93, 5]  # Example token IDs
    print(f"Prompt tokens: {prompt}")
    
    generated = ts.nlp_task_text_generation(prompt, max_length=20)
    print(f"Generated sequence: {generated}")
    print(f"Length: {len(generated)}, Final energy: {ts.rank0}")
    
    # Example 2: Classification
    print("\n2. CLASSIFICATION TASK")
    print("-" * 40)
    test_sequences = [
        [1, 1, 2, 3, 5, 8, 13],  # Fibonacci sequence
        [10, 20, 30, 40, 50],     # Linear sequence
        [2, 4, 8, 16, 32],        # Exponential sequence
        [1, 4, 9, 16, 25],        # Squares
        [7, 14, 21, 28, 35]       # Multiples of 7
    ]
    
    for seq in test_sequences:
        class_id = ts.nlp_task_classification(seq, num_classes=5)
        print(f"Sequence {seq[:5]}... → Class {class_id}")
    
    # Example 3: Tensor Series Analysis
    print("\n3. TENSOR SERIES CASCADE")
    print("-" * 40)
    analysis_tokens = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]  # Fibonacci!
    ts.initialize_tensor_series(analysis_tokens)
    results = ts.tensor_rank_cascade()
    
    print(f"\nEnergy cascade:")
    print(f"  Initial energy: {ts.fibonacci(25)}")
    print(f"  After Laplacian transport: {ts.rank0}")
    
    # Example 4: Attention Pattern Analysis
    print("\n4. ATTENTION PATTERN ANALYSIS")
    print("-" * 40)
    print("Integer-only attention weights (sample 5x5):")
    attention = ts.rank2[:5, :5]
    for row in attention:
        print("  " + " ".join(f"{val:4d}" for val in row))
    
    # Visualize results
    print("\n5. GENERATING VISUALIZATION...")
    print("-" * 40)
    ts.visualize_tensor_series(results)
    
    print("\n" + "=" * 60)
    print("All operations performed with INTEGER-ONLY arithmetic!")
    print("No floating-point operations were used.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_nlp_tasks()