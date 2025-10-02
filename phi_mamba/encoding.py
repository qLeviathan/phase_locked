"""
Encoding utilities for Φ-Mamba
Includes Zeckendorf decomposition and state representations
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from math import sqrt, pi, cos, sin, log
from .utils import PHI, PSI, fibonacci


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
        self.theta_token = 2 * pi * (hash(self.token) % 1000) / 1000
        self.theta_pos = self.position * PHI**(-self.position / 10)
        self.theta_total = (self.theta_token + self.theta_pos) % (2 * pi)
        
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


def zeckendorf_decomposition(n: int) -> List[int]:
    """
    Decompose integer n into non-consecutive Fibonacci numbers
    
    This is the key to the topological representation:
    - Each Fibonacci number represents a "hole" at that scale
    - Non-consecutive constraint = geometric requirement
    - Pattern emerges from φ² = φ + 1
    
    Args:
        n: Integer to decompose
        
    Returns:
        List of Fibonacci numbers that sum to n
        
    Example:
        >>> zeckendorf_decomposition(17)
        [13, 3, 1]  # F_7 + F_4 + F_2
    """
    if n == 0:
        return []
        
    # Build Fibonacci sequence up to n
    fibs = [1]  # Include F_1 = 1
    a, b = 1, 2
    while b <= n:
        fibs.append(b)
        a, b = b, a + b
        
    # Greedy algorithm: take largest possible
    result = []
    remaining = n
    
    i = len(fibs) - 1
    while i >= 0 and remaining > 0:
        if fibs[i] <= remaining:
            result.append(fibs[i])
            remaining -= fibs[i]
            i -= 2  # Skip next to ensure non-consecutive
        else:
            i -= 1
            
    return result


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
        if abs(phase_diff % (2*pi)) < pi/4:  # Well aligned
            past.coherence_weight = 1.5
        elif abs(phase_diff % (2*pi)) > 3*pi/4:  # Poorly aligned
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
    reflected.theta_total = (reflected.theta_token + reflected.theta_pos) % (2 * pi)
    
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