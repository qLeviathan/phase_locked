"""
Utility functions and constants for Φ-Mamba

This module provides convenience imports and utility functions.
Core constants and math functions have been moved to dedicated modules.
"""

import numpy as np
from math import pi, log
from typing import List, Tuple, Optional

# Import from centralized modules
from .constants import (
    PHI, PSI, LN_PHI, SQRT_5, UNITY, PHI_SQUARED, INV_PHI,
    TWO_PI, PENTAGON_ANGLE, GOLDEN_ANGLE,
    DEFAULT_PHASE_TOLERANCE, EPSILON
)
from .math_core import (
    fibonacci, lucas, zeckendorf_decomposition,
    cassini_identity, phi_multiply, phi_divide, phi_power,
    create_p_table, create_m_table, fibonacci_sequence,
    FibonacciCache
)
from .phase_dynamics import (
    compute_berry_phase, is_phase_locked, energy_decay_sequence,
    standing_wave as _standing_wave, pentagon_reflection_energy,
    compute_coherence_weight, phase_velocity, group_velocity
)

# Re-export for backward compatibility
__all__ = [
    # Constants
    'PHI', 'PSI', 'LN_PHI', 'SQRT_5', 'UNITY', 'PHI_SQUARED', 'INV_PHI',
    'TWO_PI', 'PENTAGON_ANGLE', 'GOLDEN_ANGLE',
    # Math functions
    'fibonacci', 'lucas', 'cassini_identity',
    'phi_multiply', 'phi_divide', 'phi_power',
    'create_p_table', 'create_m_table',
    # Phase dynamics functions
    'compute_berry_phase', 'is_phase_locked', 'energy_decay_sequence',
    'standing_wave',
    # Utility functions (defined below)
    'lagrangian_to_action', 'hamiltonian_from_lagrangian',
    'validate_recursive_unity'
]


def standing_wave(n: int, k: int) -> float:
    """
    Calculate standing wave amplitude at vertex k of n-gon

    Standing wave = φᵏ + ψᵏ

    At k=0, this equals 2 (constructive interference)
    This is where forward and backward paths meet

    Note: The 'n' parameter is kept for backward compatibility
    but is not used. The wave only depends on k.
    """
    return _standing_wave(k)


def lagrangian_to_action(weights: List[float], values: List[Tuple[float, float]]) -> float:
    """
    Convert Lagrangian (product form) to Action (sum form)
    
    L = ∏ᵢ |Tᵢ/Vᵢ|^(βᵢ) → S = Σᵢ βᵢ·ln|Tᵢ/Vᵢ|
    
    Args:
        weights: Fibonacci weights [F_{n-k}, ...]
        values: List of (T, V) tuples
        
    Returns:
        Action S
    """
    action = 0.0
    
    for i, (t, v) in enumerate(values):
        if i < len(weights) and v != 0:
            ratio = abs(t / v)
            action += weights[i] * log(ratio)
            
    return action


def hamiltonian_from_lagrangian(weights: List[float], values: List[Tuple[float, float]]) -> float:
    """
    Compute Hamiltonian from Lagrangian via Legendre transform
    
    H = Σᵢ Fᵢ(Tᵢ + Vᵢ)
    
    Args:
        weights: Fibonacci weights
        values: List of (T, V) tuples
        
    Returns:
        Hamiltonian H
    """
    h = 0.0
    
    for i, (t, v) in enumerate(values):
        if i < len(weights):
            h += weights[i] * (t + v)
            
    return h


def validate_recursive_unity():
    """
    Verify that 1 = φ² - φ
    
    This is the foundation of the entire framework:
    unity is not primitive but emerges from φ
    """
    unity_from_phi = PHI**2 - PHI
    print(f"φ² - φ = {unity_from_phi}")
    print(f"Error from 1.0: {abs(unity_from_phi - 1.0)}")
    assert abs(unity_from_phi - 1.0) < 1e-10, "Recursive unity validation failed!"
    
    # Also verify both φ and ψ satisfy x² - x = 1
    phi_check = PHI**2 - PHI - 1
    psi_check = PSI**2 - PSI - 1
    print(f"φ² - φ - 1 = {phi_check}")
    print(f"ψ² - ψ - 1 = {psi_check}")
    
    return True


# energy_decay_sequence is now imported from phase_dynamics (see imports at top)
# cassini_identity is now imported from math_core (see imports at top)