"""
Unit tests for Φ-Mamba core functionality
"""

import pytest
import numpy as np
from math import sqrt, pi

import sys
sys.path.append('..')

from phi_mamba import PhiLanguageModel, PhiTokenizer
from phi_mamba.encoding import TokenState, zeckendorf_decomposition
from phi_mamba.utils import PHI, PSI, fibonacci, compute_berry_phase


class TestPhiConstants:
    """Test golden ratio constants and relationships"""
    
    def test_phi_value(self):
        """Test φ = (1 + √5)/2"""
        expected = (1 + sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-10
        
    def test_psi_value(self):
        """Test ψ = -1/φ"""
        expected = -1 / PHI
        assert abs(PSI - expected) < 1e-10
        
    def test_phi_squared(self):
        """Test φ² = φ + 1"""
        assert abs(PHI**2 - (PHI + 1)) < 1e-10
        
    def test_phi_psi_product(self):
        """Test φ·ψ = -1"""
        assert abs(PHI * PSI - (-1)) < 1e-10
        
    def test_phi_psi_sum(self):
        """Test φ + ψ = 1"""
        assert abs(PHI + PSI - 1) < 1e-10
        
    def test_recursive_unity(self):
        """Test 1 = φ² - φ"""
        unity = PHI**2 - PHI
        assert abs(unity - 1.0) < 1e-10


class TestFibonacci:
    """Test Fibonacci and Lucas sequences"""
    
    def test_fibonacci_values(self):
        """Test first 10 Fibonacci numbers"""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for n, exp in enumerate(expected):
            assert fibonacci(n) == exp
            
    def test_binet_formula(self):
        """Test Binet formula F_n = (φⁿ - ψⁿ)/√5"""
        for n in range(20):
            binet = int(round((PHI**n - PSI**n) / sqrt(5)))
            assert fibonacci(n) == binet
            
    def test_cassini_identity(self):
        """Test F_{n+1}·F_{n-1} - F_n² = (-1)ⁿ"""
        for n in range(1, 10):
            f_n = fibonacci(n)
            f_n_plus = fibonacci(n + 1)
            f_n_minus = fibonacci(n - 1)
            
            left = f_n_plus * f_n_minus - f_n**2
            right = (-1)**n
            
            assert left == right


class TestZeckendorf:
    """Test Zeckendorf decomposition"""
    
    def test_basic_decompositions(self):
        """Test known Zeckendorf decompositions"""
        test_cases = [
            (1, [1]),
            (2, [2]),
            (3, [3]),
            (4, [3, 1]),
            (5, [5]),
            (6, [5, 1]),
            (7, [5, 2]),
            (8, [8]),
            (9, [8, 1]),
            (10, [8, 2]),
            (17, [13, 3, 1])
        ]
        
        for n, expected in test_cases:
            result = zeckendorf_decomposition(n)
            assert result == expected
            
    def test_no_consecutive_fibonacci(self):
        """Test that no consecutive Fibonacci numbers appear"""
        for n in range(1, 100):
            zeck = zeckendorf_decomposition(n)
            
            # Check no consecutive Fibonacci numbers
            fib_indices = []
            for z in zeck:
                # Find which Fibonacci number this is
                idx = 0
                f = fibonacci(idx)
                while f < z:
                    idx += 1
                    f = fibonacci(idx)
                fib_indices.append(idx)
            
            # Check no consecutive indices
            fib_indices.sort()
            for i in range(len(fib_indices) - 1):
                assert fib_indices[i+1] - fib_indices[i] >= 2
                
    def test_sum_equals_original(self):
        """Test that sum of decomposition equals original number"""
        for n in range(1, 100):
            zeck = zeckendorf_decomposition(n)
            assert sum(zeck) == n


class TestTokenState:
    """Test TokenState representation"""
    
    def test_token_state_creation(self):
        """Test creating a token state"""
        state = TokenState(
            token="cat",
            index=5,
            position=2,
            vocab_size=1000
        )
        
        assert state.token == "cat"
        assert state.index == 5
        assert state.position == 2
        assert 0 <= state.theta_token < 2*pi
        assert state.energy == PHI**(-2)
        
    def test_energy_decay(self):
        """Test energy decays with position"""
        energies = []
        for pos in range(10):
            state = TokenState("test", 0, pos, 1000)
            energies.append(state.energy)
            
        # Check monotonic decay
        for i in range(len(energies) - 1):
            assert energies[i] > energies[i+1]
            
        # Check decay rate is φ
        for i in range(len(energies) - 1):
            ratio = energies[i] / energies[i+1]
            assert abs(ratio - PHI) < 1e-10


class TestPhiTokenizer:
    """Test tokenizer functionality"""
    
    def test_basic_encoding(self):
        """Test basic text encoding"""
        tokenizer = PhiTokenizer()
        
        text = "the cat sat"
        states = tokenizer.encode(text)
        
        assert len(states) == 3
        assert states[0].token == "the"
        assert states[1].token == "cat"
        assert states[2].token == "sat"
        
    def test_retrocausal_encoding(self):
        """Test retrocausal encoding adds constraints"""
        tokenizer = PhiTokenizer()
        
        text = "the cat sat"
        
        # Forward encoding
        forward = tokenizer.encode(text, retrocausal=False)
        
        # Retrocausal encoding
        retro = tokenizer.encode(text, retrocausal=True)
        
        # Check that retrocausal adds future constraints
        assert retro[0].future_constraint is not None
        assert retro[1].future_constraint is not None
        assert retro[2].future_constraint is None  # Last token has no future
        
    def test_decode(self):
        """Test decoding states back to text"""
        tokenizer = PhiTokenizer()
        
        original = "the cat sat"
        states = tokenizer.encode(original)
        decoded = tokenizer.decode(states)
        
        assert decoded == original


class TestBerryPhase:
    """Test Berry phase calculations"""
    
    def test_phase_between_states(self):
        """Test Berry phase between two states"""
        state1 = TokenState("cat", 1, 0, 100)
        state2 = TokenState("sat", 2, 1, 100)
        
        gamma = compute_berry_phase(state1, state2)
        
        # Should be between 0 and 2π
        assert 0 <= gamma < 2*pi
        
    def test_phase_lock_detection(self):
        """Test phase lock detection"""
        # Create states that should be phase-locked
        state1 = TokenState("the", 0, 0, 100)
        state2 = TokenState("cat", 1, 1, 100)
        
        # Artificially set angles to be phase-locked
        state1.theta_total = 0
        state2.theta_total = 0
        
        gamma = compute_berry_phase(state1, state2)
        
        # Small phase difference should be locked
        assert gamma < 0.5  # Within tolerance


class TestPhiLanguageModel:
    """Test full language model"""
    
    def test_model_creation(self):
        """Test model initialization"""
        model = PhiLanguageModel(vocab_size=1000)
        
        assert model.vocab_size == 1000
        assert model.coupling_matrix.shape == (1000, 1000)
        
    def test_basic_generation(self):
        """Test basic text generation"""
        model = PhiLanguageModel()
        
        prompt = "the cat"
        generated = model.generate(prompt, max_length=10, temperature=0.8)
        
        # Should start with prompt
        assert generated.startswith(prompt)
        
        # Should be longer than prompt
        assert len(generated.split()) > len(prompt.split())
        
    def test_natural_termination(self):
        """Test that generation terminates naturally"""
        model = PhiLanguageModel()
        
        # Start with a longer prompt to deplete energy faster
        prompt = "the cat sat on the mat and"
        generated = model.generate(prompt, max_length=50, temperature=0.8)
        
        # Should not reach max_length due to energy depletion
        tokens = generated.split()
        assert len(tokens) < len(prompt.split()) + 50
        
    def test_temperature_effects(self):
        """Test temperature parameter effects"""
        model = PhiLanguageModel()
        prompt = "the"
        
        # Greedy (temperature=0) should be deterministic
        gen1 = model.generate(prompt, max_length=5, temperature=0)
        gen2 = model.generate(prompt, max_length=5, temperature=0)
        
        assert gen1 == gen2
        
        # High temperature should produce different results
        results = []
        for _ in range(5):
            gen = model.generate(prompt, max_length=5, temperature=2.0)
            results.append(gen)
            
        # At least some should be different
        assert len(set(results)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])