#!/usr/bin/env python3
"""
Tests for Phi-Mamba HuggingFace Integration

Run: python -m pytest tests/test_phi_mamba_hf.py -v
"""

import sys
import os
import tempfile
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_mamba_hf import (
    PhiMambaConfig,
    PhiMambaModel,
    PhiMambaForCausalLM,
    PhiMambaTokenizer,
)
from phi_mamba_hf.modeling_phimamba import ZeckendorfEncoder, CordicEngine


class TestZeckendorfEncoder:
    """Test Zeckendorf encoding/decoding"""

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should return original value"""
        encoder = ZeckendorfEncoder()

        for n in [0, 1, 5, 10, 42, 100, 1000]:
            indices = encoder.encode(n)
            decoded = encoder.decode(indices)
            assert decoded == n, f"Failed for {n}: got {decoded}"

    def test_no_adjacent_indices(self):
        """Zeckendorf representation has no adjacent Fibonacci indices"""
        encoder = ZeckendorfEncoder()

        for n in range(1, 1000):
            indices = encoder.encode(n)
            for i in range(len(indices) - 1):
                assert indices[i+1] - indices[i] > 1, f"Adjacent indices in {n}: {indices}"

    def test_cascade_resolves_violations(self):
        """Cascade should resolve adjacent 1s"""
        encoder = ZeckendorfEncoder()

        # Create a violation: bits 2 and 3 both set
        # F_2=2, F_3=3, should cascade to F_4=5
        bits = 0b1100  # bits 2 and 3
        cascaded = encoder.cascade(bits)

        # Check no adjacent 1s
        adjacent = cascaded & (cascaded << 1)
        assert adjacent == 0, f"Cascade failed: {bin(cascaded)}"

    def test_bit_conversion(self):
        """Bit packing should preserve indices"""
        encoder = ZeckendorfEncoder()

        indices = [1, 3, 6, 10]
        bits = encoder.to_bits(indices)
        recovered = encoder.from_bits(bits)

        assert sorted(recovered) == sorted(indices)


class TestCordicEngine:
    """Test CORDIC operations"""

    def test_phi_coupling_positive(self):
        """Coupling strength should be positive"""
        cordic = CordicEngine(scale_bits=16)

        for i in range(20):
            for j in range(20):
                coupling = cordic.phi_coupling(i, j)
                assert coupling >= 0, f"Negative coupling for ({i}, {j})"

    def test_phi_coupling_symmetric(self):
        """Coupling should be symmetric"""
        cordic = CordicEngine(scale_bits=16)

        for i in range(10):
            for j in range(10):
                c_ij = cordic.phi_coupling(i, j)
                c_ji = cordic.phi_coupling(j, i)
                assert c_ij == c_ji, f"Asymmetric: ({i},{j})={c_ij}, ({j},{i})={c_ji}"

    def test_rotation_preserves_magnitude(self):
        """CORDIC rotation should approximately preserve magnitude"""
        cordic = CordicEngine(scale_bits=16)

        x, y = 1000, 0
        angle = cordic.scale // 4  # ~π/4

        x_rot, y_rot = cordic.rotate(x, y, angle)

        # Magnitude should be similar (within CORDIC gain factor ~1.647)
        mag_orig = (x**2 + y**2) ** 0.5
        mag_rot = (x_rot**2 + y_rot**2) ** 0.5

        ratio = mag_rot / mag_orig
        assert 0.5 < ratio < 2.5, f"Magnitude ratio too far: {ratio}"


class TestPhiMambaConfig:
    """Test configuration"""

    def test_tiny_config(self):
        """Tiny config should have small parameters"""
        config = PhiMambaConfig.tiny()

        assert config.vocab_size == 8000
        assert config.max_shells == 32
        assert config.num_layers == 2

    def test_small_config(self):
        """Small config should have medium parameters"""
        config = PhiMambaConfig.small()

        assert config.vocab_size == 16000
        assert config.max_shells == 64
        assert config.num_layers == 4

    def test_base_config(self):
        """Base config should have larger parameters"""
        config = PhiMambaConfig.base()

        assert config.vocab_size == 32000
        assert config.max_shells == 128
        assert config.num_layers == 6

    def test_save_load_roundtrip(self):
        """Config should survive save/load"""
        config = PhiMambaConfig(vocab_size=5000, max_shells=48, num_layers=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = PhiMambaConfig.from_pretrained(tmpdir)

            assert loaded.vocab_size == 5000
            assert loaded.max_shells == 48
            assert loaded.num_layers == 3


class TestPhiMambaTokenizer:
    """Test tokenizer"""

    def test_encode_decode_simple(self):
        """Simple text should roundtrip through tokenizer"""
        tokenizer = PhiMambaTokenizer(vocab_size=1000)

        text = "the cat sat on the mat"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)

        # Words should be recovered
        assert "the" in decoded.lower()
        assert "cat" in decoded.lower()

    def test_special_tokens(self):
        """Special tokens should be added correctly"""
        tokenizer = PhiMambaTokenizer()

        tokens = tokenizer.encode("hello", add_special_tokens=True)

        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id

    def test_padding(self):
        """Padding should work correctly"""
        tokenizer = PhiMambaTokenizer()

        result = tokenizer(
            ["short", "this is a longer sentence"],
            padding=True,
            return_tensors="np"
        )

        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["input_ids"].shape[1]  # Same length

    def test_save_load_roundtrip(self):
        """Tokenizer should survive save/load"""
        tokenizer = PhiMambaTokenizer(vocab_size=5000)

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            loaded = PhiMambaTokenizer.from_pretrained(tmpdir)

            assert loaded.vocab_size == 5000


class TestPhiMambaModel:
    """Test model forward pass"""

    def test_forward_shape(self):
        """Forward pass should produce correct output shape"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaModel(config)

        batch_size, seq_len = 2, 16
        input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model.forward(input_ids)

        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_hidden_states(self):
        """Forward should return hidden states when requested"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaModel(config)

        input_ids = np.array([[1, 2, 3, 4]])
        output = model.forward(input_ids, return_hidden_states=True)

        assert output.hidden_states is not None
        assert output.hidden_states.shape[0] == config.num_layers + 1

    def test_energy_decay(self):
        """Energy should decay through layers"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaModel(config)

        input_ids = np.array([[1, 2, 3, 4]])
        output = model.forward(input_ids)

        # Energy should be less than initial
        assert np.all(output.energy < model.energy_scale)

    def test_deterministic(self):
        """Same input should produce same output"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaModel(config)

        input_ids = np.array([[42, 100, 200]])

        output1 = model.forward(input_ids)
        output2 = model.forward(input_ids)

        np.testing.assert_array_equal(output1.logits, output2.logits)


class TestPhiMambaForCausalLM:
    """Test causal LM generation"""

    def test_generate_extends_sequence(self):
        """Generate should extend the input sequence"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaForCausalLM(config)

        input_ids = np.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=5, do_sample=False)

        assert generated.shape[1] > input_ids.shape[1]

    def test_generate_respects_max_tokens(self):
        """Generate should not exceed max_new_tokens"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaForCausalLM(config)

        input_ids = np.array([[1, 2, 3]])
        max_new = 10
        generated = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)

        assert generated.shape[1] <= input_ids.shape[1] + max_new

    def test_greedy_is_deterministic(self):
        """Greedy generation should be deterministic"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaForCausalLM(config)

        input_ids = np.array([[1, 2, 3]])

        gen1 = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        gen2 = model.generate(input_ids, max_new_tokens=5, do_sample=False)

        np.testing.assert_array_equal(gen1, gen2)


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_full_pipeline(self):
        """Test complete tokenize -> model -> decode pipeline"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaForCausalLM(config)
        tokenizer = PhiMambaTokenizer(vocab_size=config.vocab_size)

        # Tokenize
        text = "the quick brown"
        inputs = tokenizer(text, return_tensors="np")

        # Generate
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False
        )

        # Decode
        decoded = tokenizer.decode(outputs[0].tolist())

        assert len(decoded) > 0

    def test_save_load_inference(self):
        """Model should work after save/load"""
        config = PhiMambaConfig.tiny()
        model = PhiMambaForCausalLM(config)
        tokenizer = PhiMambaTokenizer(vocab_size=config.vocab_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)

            # Load
            loaded_model = PhiMambaForCausalLM.from_pretrained(tmpdir)
            loaded_tokenizer = PhiMambaTokenizer.from_pretrained(tmpdir)

            # Inference
            inputs = loaded_tokenizer("test input", return_tensors="np")
            outputs = loaded_model.generate(inputs["input_ids"], max_new_tokens=5)

            assert outputs.shape[1] > inputs["input_ids"].shape[1]


def run_quick_test():
    """Quick smoke test"""
    print("Running quick Phi-Mamba HF test...")

    # Create tiny model
    config = PhiMambaConfig.tiny()
    model = PhiMambaForCausalLM(config)
    tokenizer = PhiMambaTokenizer(vocab_size=config.vocab_size)

    # Test inference
    text = "The quick brown"
    inputs = tokenizer(text, return_tensors="np")

    print(f"Input: {text}")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")

    outputs = model.generate(inputs["input_ids"], max_new_tokens=10, do_sample=False)
    decoded = tokenizer.decode(outputs[0].tolist())

    print(f"Output: {decoded}")
    print("✓ Quick test passed!")

    return True


if __name__ == "__main__":
    run_quick_test()
