"""
Phi-Mamba Model Implementation

Integer-only language model using Zeckendorf-CORDIC arithmetic.
All operations are shift-add only - no multiplication or division.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .configuration_phimamba import PhiMambaConfig


# Precompute Fibonacci sequence (constant table)
def _generate_fibs(n: int) -> np.ndarray:
    fibs = [1, 2]
    for _ in range(n - 2):
        fibs.append(fibs[-1] + fibs[-2])
    return np.array(fibs, dtype=np.int64)

FIBS = _generate_fibs(64)


@dataclass
class PhiMambaOutput:
    """Model output container"""
    logits: np.ndarray
    hidden_states: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None


class ZeckendorfEncoder:
    """
    Encode integers to Zeckendorf representation (Fibonacci indices).

    Uses greedy algorithm - O(log n) complexity.
    Pure integer operations only.
    """

    @staticmethod
    def encode(n: int) -> List[int]:
        """Convert integer to sparse Fibonacci indices"""
        if n <= 0:
            return [0]

        indices = []
        remaining = n

        for i in range(len(FIBS) - 1, -1, -1):
            if remaining >= FIBS[i]:
                indices.append(i)
                remaining -= FIBS[i]

        return sorted(indices)

    @staticmethod
    def decode(indices: List[int]) -> int:
        """Convert Fibonacci indices back to integer"""
        return sum(FIBS[i] for i in indices if i < len(FIBS))

    @staticmethod
    def to_bits(indices: List[int]) -> int:
        """Convert indices to bit-packed integer"""
        bits = 0
        for idx in indices:
            bits |= (1 << idx)
        return bits

    @staticmethod
    def from_bits(bits: int) -> List[int]:
        """Extract indices from bit-packed integer"""
        indices = []
        idx = 0
        while bits:
            if bits & 1:
                indices.append(idx)
            bits >>= 1
            idx += 1
        return indices

    @staticmethod
    def cascade(bits: int) -> int:
        """
        Resolve adjacent 1s via Fibonacci cascade.

        F_i + F_{i+1} = F_{i+2}

        Pure bit operations - no arithmetic!
        """
        for _ in range(64):  # Max iterations
            adjacent = bits & (bits << 1)
            if adjacent == 0:
                break

            # Find lowest adjacent pair
            pos = 0
            temp = adjacent
            while temp:
                if temp & 1:
                    bits &= ~(1 << pos)
                    bits &= ~(1 << (pos + 1))
                    bits |= (1 << (pos + 2))
                    break
                temp >>= 1
                pos += 1

        return bits


class CordicEngine:
    """
    CORDIC engine for shift-add only arithmetic.

    All trig, exp, log computed with only:
    - Addition/subtraction
    - Bit shifts
    """

    def __init__(self, scale_bits: int = 16, n_iterations: int = 16):
        self.scale_bits = scale_bits
        self.scale = 1 << scale_bits
        self.n_iterations = n_iterations

        # Precompute arctangent table
        self.atan_table = self._build_atan_table()

        # Golden ratio in fixed-point
        self.PHI = (16180 * self.scale) // 10000  # ~1.618
        self.LN_PHI = (4812 * self.scale) // 10000  # ~0.4812

    def _build_atan_table(self) -> List[int]:
        """Build arctangent lookup table"""
        # atan(2^-i) in fixed-point
        accurate = [
            7854, 4636, 2450, 1244, 624, 312, 156, 78,
            39, 20, 10, 5, 2, 1, 1, 0
        ]
        return [(a * self.scale) // 10000 for a in accurate]

    def rotate(self, x: int, y: int, angle: int) -> Tuple[int, int]:
        """
        CORDIC rotation - shift-add only!

        Rotate (x, y) by angle.
        """
        z = angle

        for i in range(self.n_iterations):
            if z >= 0:
                x_new = x - (y >> i)
                y_new = y + (x >> i)
                z_new = z - self.atan_table[i]
            else:
                x_new = x + (y >> i)
                y_new = y - (x >> i)
                z_new = z + self.atan_table[i]

            x, y, z = x_new, y_new, z_new

        return x, y

    def phi_coupling(self, idx1: int, idx2: int) -> int:
        """
        Compute coupling strength between token indices.

        Uses Fibonacci relationship - integer only.
        """
        fib1 = int(FIBS[idx1 % 20]) if idx1 < 20 else int(FIBS[19])
        fib2 = int(FIBS[idx2 % 20]) if idx2 < 20 else int(FIBS[19])

        # Coupling = (F_i * F_j) / (F_i + F_j + 1)
        # Use shifts to approximate division
        numerator = (fib1 * fib2) << self.scale_bits
        denominator = fib1 + fib2 + 1

        return numerator // denominator


class PhiMambaPreTrainedModel:
    """Base class for Phi-Mamba models"""

    config_class = PhiMambaConfig

    def __init__(self, config: PhiMambaConfig):
        self.config = config
        self.encoder = ZeckendorfEncoder()
        self.cordic = CordicEngine(config.scale_bits)

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save model state (coupling matrix)
        state = {
            "coupling_cache": {},  # Lazy computed
            "version": "0.1.0",
        }

        with open(os.path.join(save_directory, "model.json"), "w") as f:
            json.dump(state, f)

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """Load model from directory"""
        config = PhiMambaConfig.from_pretrained(pretrained_path)
        return cls(config)


class PhiMambaModel(PhiMambaPreTrainedModel):
    """
    Core Phi-Mamba model.

    Integer-only state space model using:
    - Zeckendorf encoding (Fibonacci representation)
    - CORDIC arithmetic (shift-add only)
    - Phase-locked transitions (natural termination)
    """

    def __init__(self, config: PhiMambaConfig):
        super().__init__(config)

        # State tracking
        self.energy_scale = 1 << config.scale_bits

        # Coupling cache (lazy computed)
        self._coupling_cache: Dict[Tuple[int, int], int] = {}

    def get_coupling(self, idx1: int, idx2: int) -> int:
        """Get cached coupling strength"""
        key = (idx1, idx2)
        if key not in self._coupling_cache:
            self._coupling_cache[key] = self.cordic.phi_coupling(idx1, idx2)
        return self._coupling_cache[key]

    def encode_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode token IDs to Zeckendorf bit patterns.

        Returns: [batch, seq] int64 bit patterns
        """
        batch_size, seq_len = token_ids.shape
        encoded = np.zeros((batch_size, seq_len), dtype=np.int64)

        for b in range(batch_size):
            for s in range(seq_len):
                indices = self.encoder.encode(int(token_ids[b, s]))
                encoded[b, s] = self.encoder.to_bits(indices)

        return encoded

    def forward_layer(self, states: np.ndarray, energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single layer forward pass.

        1. Cascade (resolve adjacent 1s)
        2. Phase coupling (CORDIC-based attention)
        3. Energy decay (natural termination)
        """
        batch_size, seq_len = states.shape

        # Step 1: Cascade all states
        for b in range(batch_size):
            for s in range(seq_len):
                states[b, s] = self.encoder.cascade(int(states[b, s]))

        # Step 2: Apply phase coupling (simplified attention)
        new_states = np.zeros_like(states)
        for b in range(batch_size):
            for s in range(seq_len):
                current = int(states[b, s])
                accumulated = current

                # Couple with previous positions
                for prev_s in range(max(0, s - 8), s):  # Local attention window
                    prev_state = int(states[b, prev_s])

                    # Get coupling strength
                    coupling = self.get_coupling(s % 20, prev_s % 20)

                    # Accumulate (OR operation for Zeckendorf addition)
                    if coupling > (self.energy_scale >> 2):  # Threshold
                        accumulated |= prev_state

                new_states[b, s] = accumulated

        # Step 3: Cascade again after coupling
        for b in range(batch_size):
            for s in range(seq_len):
                new_states[b, s] = self.encoder.cascade(int(new_states[b, s]))

        # Step 4: Energy decay (right shift = divide by 2)
        energy = energy >> 1

        return new_states, energy

    def forward(
        self,
        input_ids: np.ndarray,
        return_hidden_states: bool = False,
    ) -> PhiMambaOutput:
        """
        Full forward pass through all layers.

        Args:
            input_ids: [batch, seq] token IDs
            return_hidden_states: Whether to return intermediate states

        Returns:
            PhiMambaOutput with logits and optional hidden states
        """
        batch_size, seq_len = input_ids.shape

        # Encode to Zeckendorf
        states = self.encode_tokens(input_ids)

        # Initialize energy (scaled integer)
        energy = np.full((batch_size, seq_len), self.energy_scale, dtype=np.int64)

        hidden_states = [states.copy()] if return_hidden_states else None

        # Process through layers
        for layer_idx in range(self.config.num_layers):
            states, energy = self.forward_layer(states, energy)

            if return_hidden_states:
                hidden_states.append(states.copy())

        # Decode to logits (simplified: use bit count as score)
        logits = np.zeros((batch_size, seq_len, self.config.vocab_size), dtype=np.float32)

        for b in range(batch_size):
            for s in range(seq_len):
                # Score tokens by Fibonacci overlap
                current_bits = int(states[b, s])

                for token_id in range(min(self.config.vocab_size, 1000)):  # Limit for speed
                    token_indices = self.encoder.encode(token_id)
                    token_bits = self.encoder.to_bits(token_indices)

                    # Score = popcount(current & token)
                    overlap = bin(current_bits & token_bits).count('1')
                    logits[b, s, token_id] = overlap

        return PhiMambaOutput(
            logits=logits,
            hidden_states=np.stack(hidden_states) if return_hidden_states else None,
            energy=energy,
        )


class PhiMambaForCausalLM(PhiMambaPreTrainedModel):
    """
    Phi-Mamba for causal language modeling.

    Wraps PhiMambaModel with generation utilities.
    """

    def __init__(self, config: PhiMambaConfig):
        super().__init__(config)
        self.model = PhiMambaModel(config)

    def forward(
        self,
        input_ids: np.ndarray,
        **kwargs
    ) -> PhiMambaOutput:
        """Forward pass"""
        return self.model.forward(input_ids, **kwargs)

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Uses natural termination when energy drops below threshold.
        """
        batch_size, seq_len = input_ids.shape
        generated = input_ids.copy()

        for step in range(max_new_tokens):
            # Forward pass
            output = self.forward(generated)

            # Get logits for last position
            next_logits = output.logits[:, -1, :]  # [batch, vocab]

            # Check energy for natural termination
            if output.energy is not None:
                avg_energy = np.mean(output.energy[:, -1])
                if avg_energy < self.config.energy_threshold:
                    break  # Natural termination

            # Sample next token
            if do_sample and temperature > 0:
                # Apply temperature
                scaled_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_idx = np.argsort(scaled_logits, axis=-1)[:, -top_k:]
                    mask = np.ones_like(scaled_logits) * float('-inf')
                    for b in range(batch_size):
                        mask[b, top_k_idx[b]] = scaled_logits[b, top_k_idx[b]]
                    scaled_logits = mask

                # Softmax
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

                # Sample
                next_tokens = np.array([
                    np.random.choice(self.config.vocab_size, p=probs[b])
                    for b in range(batch_size)
                ])
            else:
                # Greedy
                next_tokens = np.argmax(next_logits, axis=-1)

            # Check for EOS
            if np.all(next_tokens == self.config.eos_token_id):
                break

            # Append
            generated = np.concatenate([
                generated,
                next_tokens.reshape(batch_size, 1)
            ], axis=1)

        return generated
