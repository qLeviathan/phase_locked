"""
Encoding utilities for Φ-Mamba
Includes state representations and topological encoding
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from math import pi

# Import from centralized modules
from .constants import PHI, PSI, TWO_PI
from .math_core import fibonacci, zeckendorf_decomposition


@dataclass
class TokenState:
    """
    Representation of a token in φ-space
    
    Attributes:
        token: The actual token string
        index: Token ID in vocabulary
        position: Position in sequence
        vocab_size: Size of vocabulary
        theta_token: Angular position based on token identity
        theta_pos: Position-based angle (RoPE-like)
        energy: φ^(-position) energy level
        zeckendorf: Fibonacci decomposition of position
        future_constraint: Berry phase from future (retrocausal)
        coherence_weight: Weight based on phase locking
    """
    token: str
    index: int
    position: int
    vocab_size: int
    
    def __post_init__(self):
        # Compute derived properties
        self.theta_token = TWO_PI * (hash(self.token) % 1000) / 1000
        self.theta_pos = self.position * PHI**(-self.position / 10)
        self.theta_total = (self.theta_token + self.theta_pos) % TWO_PI
        
        self.energy = PHI**(-self.position)
        self.zeckendorf = zeckendorf_decomposition(self.position + 1)
        
        # Will be set by retrocausal encoding
        self.future_constraint: Optional[float] = None
        self.coherence_weight: float = 1.0
        
    @property
    def r(self) -> float:
        """Radial amplitude in cylinder encoding"""
        # Number of active Fibonacci terms determines amplitude
        return PHI ** len(self.zeckendorf)
    
    @property
    def active_shells(self) -> List[int]:
        """Which Fibonacci shells are active"""
        return self.zeckendorf
    
    def __repr__(self):
        return (f"TokenState('{self.token}', pos={self.position}, "
                f"θ={self.theta_total:.3f}, E={self.energy:.3f}, "
                f"shells={self.active_shells})")


# zeckendorf_decomposition is now imported from math_core (see imports at top)


def binary_from_zeckendorf(zeck: List[int], max_fib_index: int = 10) -> str:
    """
    Convert Zeckendorf decomposition to binary representation
    
    This shows the "emergent bit pattern" from topology:
    - 1 = hole exists at that scale
    - 0 = no hole at that scale
    - Gap constraint naturally enforced
    
    Args:
        zeck: Zeckendorf decomposition
        max_fib_index: Maximum Fibonacci index to consider
        
    Returns:
        Binary string representation
    """
    # Create binary representation
    binary = ['0'] * max_fib_index
    
    for fib in zeck:
        # Find which Fibonacci number this is
        idx = 1
        a, b = 1, 2
        while b < fib:
            a, b = b, a + b
            idx += 1
            
        if b == fib and idx < max_fib_index:
            binary[max_fib_index - idx - 1] = '1'
            
    return ''.join(binary)


def retrocausal_encode(tokens: List[str], vocab_size: int = 50000) -> List[TokenState]:
    """
    Encode tokens with retrocausal constraints
    
    Key insight: Future tokens constrain past tokens
    This creates more coherent sequences by ensuring
    consistency from both temporal directions.
    
    Args:
        tokens: List of token strings
        vocab_size: Size of vocabulary
        
    Returns:
        List of TokenStates with future constraints
    """
    if not tokens:
        return []
        
    # First, create states in forward direction
    states = []
    for i, token in enumerate(tokens):
        state = TokenState(
            token=token,
            index=hash(token) % vocab_size,  # Simple hash for demo
            position=i,
            vocab_size=vocab_size
        )
        states.append(state)
    
    # Then apply constraints from future to past
    for i in range(len(states) - 1, 0, -1):
        future = states[i]
        past = states[i-1]
        
        # Future constrains past through Berry phase
        phase_diff = future.theta_total - past.theta_total
        past.future_constraint = phase_diff
        
        # Adjust coherence weight based on phase alignment
        if abs(phase_diff % TWO_PI) < pi/4:  # Well aligned
            past.coherence_weight = 1.5
        elif abs(phase_diff % TWO_PI) > 3*pi/4:  # Poorly aligned
            past.coherence_weight = 0.5
            
    return states


def pentagon_reflection(state: TokenState) -> TokenState:
    """
    Apply pentagon reflection when phase lock fails
    
    This implements the "bounce" mechanism:
    - Mirror through conjugate path
    - Energy scales by 1/φ
    - After ~5 bounces, energy exhausted
    
    Args:
        state: Current token state
        
    Returns:
        Reflected state with reduced energy
    """
    reflected = TokenState(
        token=state.token,
        index=state.index,
        position=state.position,
        vocab_size=state.vocab_size
    )
    
    # Mirror the angle
    reflected.theta_token = pi - state.theta_token
    reflected.theta_total = (reflected.theta_token + reflected.theta_pos) % TWO_PI
    
    # Scale energy down by φ
    reflected.energy = state.energy / PHI
    
    # Mark as reflected
    reflected.coherence_weight = state.coherence_weight * 0.5
    
    return reflected


def calculate_betti_numbers(n: int, max_dim: int = 5) -> List[int]:
    """
    Calculate Betti numbers for shell depth n
    
    β_k = F_{n-k} (number of k-dimensional holes)
    
    This reveals the topological structure:
    - β_0: Connected components
    - β_1: Loops/cycles
    - β_2: Voids
    - etc.
    
    Args:
        n: Shell depth
        max_dim: Maximum dimension to calculate
        
    Returns:
        List of Betti numbers [β_0, β_1, ..., β_max_dim]
    """
    betti = []
    for k in range(min(max_dim + 1, n + 1)):
        beta_k = fibonacci(n - k)
        betti.append(beta_k)
    return betti


def encode_to_cylinder(states: List[TokenState]) -> List[Tuple[float, float, float]]:
    """
    Map token states to cylinder coordinates
    
    Returns (θ, r, z) coordinates for visualization:
    - θ: Angular position on circle
    - r: Radial distance (amplitude)
    - z: Vertical position (time/causality)
    
    Args:
        states: List of token states
        
    Returns:
        List of (theta, r, z) tuples
    """
    coords = []
    
    for state in states:
        theta = state.theta_total
        r = state.r
        z = state.position  # Could also use energy or other mapping
        
        coords.append((theta, r, z))
        
    return coords