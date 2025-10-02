"""
Utility functions and constants for Φ-Mamba
"""

import numpy as np
from math import sqrt, log, exp, pi, cos, sin
from typing import List, Tuple, Optional


# Golden ratio constants
PHI = (1 + sqrt(5)) / 2
PSI = -1 / PHI
LN_PHI = log(PHI)
SQRT_5 = sqrt(5)

# Recursive unity
UNITY = PHI**2 - PHI  # Should equal 1.0


def fibonacci(n: int) -> int:
    """
    Compute nth Fibonacci number
    
    Uses Binet formula for efficiency:
    F_n = (φⁿ - ψⁿ) / √5
    """
    if n < 0:
        return 0
    if n == 0:
        return 0
    if n == 1:
        return 1
        
    # Binet formula
    return int(round((PHI**n - PSI**n) / SQRT_5))


def lucas(n: int) -> int:
    """
    Compute nth Lucas number
    
    L_n = φⁿ + ψⁿ
    """
    if n == 0:
        return 2
    if n == 1:
        return 1
        
    return int(round(PHI**n + PSI**n))


def compute_berry_phase(state1, state2) -> float:
    """
    Calculate Berry phase between two states
    
    This geometric phase accumulation determines
    whether transitions are "phase-locked" (coherent)
    
    Args:
        state1: First TokenState
        state2: Second TokenState
        
    Returns:
        Berry phase in radians
    """
    # Angular difference
    d_theta = state2.theta_total - state1.theta_total
    
    # Shell overlap (topological contribution)
    shells1 = set(state1.active_shells)
    shells2 = set(state2.active_shells)
    overlap = len(shells1.intersection(shells2))
    max_shells = max(len(shells1), len(shells2), 1)
    overlap_factor = overlap / max_shells
    
    # Position difference (causal contribution)
    d_pos = abs(state2.position - state1.position)
    
    # Berry phase combines all contributions
    gamma = d_theta * (1 + overlap_factor) + 2*pi*d_pos/100
    
    return gamma % (2*pi)


def is_phase_locked(gamma: float, tolerance: float = 0.5) -> bool:
    """
    Check if Berry phase indicates phase lock
    
    Phase lock occurs when γ ≡ 0 (mod 2π)
    within tolerance
    """
    reduced = abs(gamma % (2*pi))
    return reduced < tolerance or reduced > (2*pi - tolerance)


def phi_multiply(n: int, m: int) -> int:
    """
    Multiply using only addition in φ-space
    
    φⁿ × φᵐ = φⁿ⁺ᵐ
    
    This is the key computational advantage:
    all multiplication becomes addition
    """
    return n + m


def phi_power(base: int, exp: int) -> int:
    """
    Compute power using only addition
    
    (φⁿ)ᵐ = φⁿᵐ = φⁿ⁺ⁿ⁺...⁺ⁿ (m times)
    """
    result = 0
    for _ in range(exp):
        result += base
    return result


def phi_divide(n: int, m: int) -> int:
    """
    Division in φ-space
    
    φⁿ / φᵐ = φⁿ⁻ᵐ
    """
    return n - m


def create_p_table(max_n: int = 100) -> dict:
    """
    Create P-table for φⁿ values
    
    Maps integer n to actual value of φⁿ
    for verification and conversion
    """
    p_table = {}
    for n in range(-max_n, max_n + 1):
        p_table[n] = PHI**n
    return p_table


def create_m_table(max_n: int = 100) -> dict:
    """
    Create M-table for ψⁿ values
    
    Maps integer n to actual value of ψⁿ
    for conjugate operations
    """
    m_table = {}
    for n in range(-max_n, max_n + 1):
        m_table[n] = PSI**n
    return m_table


def standing_wave(n: int, k: int) -> float:
    """
    Calculate standing wave amplitude at vertex k of n-gon
    
    Standing wave = φᵏ + ψᵏ
    
    At k=0, this equals 2 (constructive interference)
    This is where forward and backward paths meet
    """
    return PHI**k + PSI**k


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


def energy_decay_sequence(initial: float = 1.0, steps: int = 10) -> List[float]:
    """
    Show energy decay through pentagon reflections
    
    E_n = E_0 / φⁿ
    
    This demonstrates natural termination:
    after ~5-7 reflections, energy < threshold
    """
    energies = [initial]
    
    for n in range(1, steps):
        energy = initial / (PHI**n)
        energies.append(energy)
        
    return energies


def cassini_identity(n: int) -> int:
    """
    Verify Cassini identity for Fibonacci numbers
    
    F_{n+1}·F_{n-1} - F_n² = (-1)ⁿ
    
    This shows the alternating area property
    that creates phase relationships
    """
    f_n = fibonacci(n)
    f_n_plus = fibonacci(n + 1)
    f_n_minus = fibonacci(n - 1)
    
    left = f_n_plus * f_n_minus - f_n**2
    right = (-1)**n
    
    assert left == right, f"Cassini identity failed for n={n}: {left} ≠ {right}"
    
    return left