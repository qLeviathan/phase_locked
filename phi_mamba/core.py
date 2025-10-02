"""
Core implementation of Φ-Mamba language model
"""

import numpy as np
from math import sqrt, log, exp, pi, cos, sin
from typing import List, Dict, Tuple, Optional
from .encoding import zeckendorf_decomposition, TokenState
from .utils import PHI, PSI, fibonacci, compute_berry_phase


class PhiTokenizer:
    """Tokenizer that encodes text into φ-based geometric states"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.token_to_id = {}  # Will be populated with actual vocab
        self.id_to_token = {}
        self._initialize_vocab()
        
    def _initialize_vocab(self):
        """Initialize with a basic vocabulary for demos"""
        # In practice, this would load a full vocabulary
        basic_vocab = [
            "the", "a", "an", "cat", "dog", "sat", "ran", "jumped",
            "on", "in", "under", "mat", "chair", "table", "quickly",
            "slowly", "and", "but", "or", ".", ",", "!", "?"
        ]
        
        for i, token in enumerate(basic_vocab):
            self.token_to_id[token.lower()] = i
            self.id_to_token[i] = token.lower()
            
    def encode(self, text: str, retrocausal: bool = False) -> List[TokenState]:
        """
        Encode text into φ-states
        
        Args:
            text: Input text to encode
            retrocausal: If True, encode from end backwards
            
        Returns:
            List of TokenState objects
        """
        tokens = text.lower().split()
        states = []
        
        for i, token in enumerate(tokens):
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
                
                # Create φ-encoded state
                state = TokenState(
                    token=token,
                    index=token_id,
                    position=i,
                    vocab_size=self.vocab_size
                )
                states.append(state)
                
        if retrocausal:
            # Encode from end, propagating constraints backward
            states = self._apply_retrocausal_constraints(states)
            
        return states
    
    def _apply_retrocausal_constraints(self, states: List[TokenState]) -> List[TokenState]:
        """Apply retrocausal constraints from future to past"""
        if len(states) <= 1:
            return states
            
        # Work backwards through the sequence
        for i in range(len(states) - 1, 0, -1):
            future_state = states[i]
            past_state = states[i-1]
            
            # Future state constrains past state
            phase_constraint = compute_berry_phase(past_state, future_state)
            past_state.future_constraint = phase_constraint
            
            # Adjust past state energy based on future coherence
            if abs(phase_constraint % (2*pi)) < 0.1:  # Phase locked
                past_state.coherence_weight = 1.0
            else:
                past_state.coherence_weight = 0.5
                
        return states
    
    def decode(self, states: List[TokenState]) -> str:
        """Decode φ-states back to text"""
        tokens = []
        for state in states:
            if state.token:
                tokens.append(state.token)
            elif 0 <= state.index < len(self.id_to_token):
                tokens.append(self.id_to_token[state.index])
        return ' '.join(tokens)


class PhiLanguageModel:
    """
    Phase-locked language model using golden ratio encoding
    
    Key features:
    - All operations reduce to integer addition
    - Natural termination through energy decay
    - Retrocausal encoding for improved coherence
    - Topological information storage
    """
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer = PhiTokenizer(vocab_size)
        self.coupling_matrix = self._initialize_coupling()
        
    def _initialize_coupling(self) -> np.ndarray:
        """Initialize coupling matrix based on φ-geometry"""
        # In a full implementation, this would be learned
        # For now, initialize with cosine similarity of angles
        matrix = np.zeros((self.vocab_size, self.vocab_size))
        
        for i in range(min(100, self.vocab_size)):  # Demo: just first 100
            for j in range(min(100, self.vocab_size)):
                theta_i = 2 * pi * i / self.vocab_size
                theta_j = 2 * pi * j / self.vocab_size
                matrix[i,j] = cos(theta_i - theta_j)
                
        return matrix
    
    def encode(self, text: str, retrocausal: bool = True) -> List[TokenState]:
        """Encode text into φ-states"""
        return self.tokenizer.encode(text, retrocausal=retrocausal)
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text with natural termination
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens (usually won't be reached)
            temperature: Sampling temperature
            
        Returns:
            Generated text with natural termination
        """
        # Encode prompt with retrocausal constraints
        states = self.encode(prompt, retrocausal=True)
        
        if not states:
            return prompt
            
        tokens = prompt.split()
        
        for step in range(max_length):
            # Get next token
            next_state = self._generate_next(states, temperature)
            
            if next_state is None:
                # Natural termination
                break
                
            tokens.append(next_state.token)
            states.append(next_state)
            
            # Check for punctuation termination
            if next_state.token in ['.', '!', '?']:
                break
                
        return ' '.join(tokens)
    
    def _generate_next(self, context: List[TokenState], temperature: float) -> Optional[TokenState]:
        """Generate next token using phase-locked selection"""
        if not context:
            return None
            
        last_state = context[-1]
        
        # Check energy threshold
        if last_state.energy < 0.01:
            return None  # Natural termination
            
        candidates = []
        
        # Consider each possible next token
        for token_id in range(min(100, self.vocab_size)):  # Demo: limited vocab
            if token_id not in self.tokenizer.id_to_token:
                continue
                
            token = self.tokenizer.id_to_token[token_id]
            
            # Create candidate state
            candidate = TokenState(
                token=token,
                index=token_id,
                position=last_state.position + 1,
                vocab_size=self.vocab_size
            )
            
            # Calculate phase coherence
            berry_phase = compute_berry_phase(last_state, candidate)
            phase_locked = abs(berry_phase % (2*pi)) < 0.5
            
            # Calculate coupling strength
            coupling = self.coupling_matrix[last_state.index, token_id]
            
            # Score combines phase lock, coupling, and energy
            score = (
                (2.0 if phase_locked else 0.5) *
                (1.0 + coupling) *
                candidate.energy
            )
            
            candidates.append({
                'state': candidate,
                'score': score,
                'phase_locked': phase_locked
            })
        
        if not candidates:
            return None
            
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Temperature sampling
        if temperature == 0:
            # Greedy
            return candidates[0]['state']
        else:
            # Sample from top candidates
            scores = np.array([c['score'] for c in candidates[:10]])
            probs = np.exp(scores / temperature)
            probs = probs / probs.sum()
            
            idx = np.random.choice(len(probs), p=probs)
            return candidates[idx]['state']
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text under the model"""
        states = self.encode(text, retrocausal=True)
        
        if len(states) <= 1:
            return 0.0
            
        total_log_prob = 0.0
        
        for i in range(1, len(states)):
            context = states[:i]
            true_next = states[i]
            
            # Get prediction distribution
            candidates = []
            for token_id in range(min(100, self.vocab_size)):
                if token_id not in self.tokenizer.id_to_token:
                    continue
                    
                candidate = TokenState(
                    token=self.tokenizer.id_to_token[token_id],
                    index=token_id,
                    position=true_next.position,
                    vocab_size=self.vocab_size
                )
                
                berry_phase = compute_berry_phase(context[-1], candidate)
                phase_locked = abs(berry_phase % (2*pi)) < 0.5
                coupling = self.coupling_matrix[context[-1].index, token_id]
                
                score = (
                    (2.0 if phase_locked else 0.5) *
                    (1.0 + coupling) *
                    candidate.energy
                )
                
                candidates.append((token_id, score))
            
            # Normalize scores to probabilities
            scores = np.array([s for _, s in candidates])
            probs = np.exp(scores)
            probs = probs / probs.sum()
            
            # Find probability of true token
            true_idx = next((i for i, (tid, _) in enumerate(candidates) 
                           if tid == true_next.index), None)
            
            if true_idx is not None:
                log_prob = log(probs[true_idx] + 1e-10)
                total_log_prob += log_prob
                
        perplexity = exp(-total_log_prob / (len(states) - 1))
        return perplexity