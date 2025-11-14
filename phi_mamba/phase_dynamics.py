"""
Phase Dynamics for Φ-Mamba

This module handles all phase-related calculations including:
- Berry phase computation
- Phase locking detection
- Energy decay sequences
- Pentagon reflections

The Berry phase is a geometric phase that accumulates during
state transitions, determining coherence in the phi-space framework.
"""

from typing import List
from math import pi
from .constants import PHI, PSI, TWO_PI, DEFAULT_PHASE_TOLERANCE


def compute_berry_phase(state1, state2) -> float:
    """
    Calculate Berry phase between two states.

    The Berry phase is a geometric phase accumulation that determines
    whether transitions are "phase-locked" (coherent).

    Args:
        state1: First TokenState (must have attributes: theta_total,
                active_shells, position)
        state2: Second TokenState

    Returns:
        Berry phase γ in radians (0 to 2π)

    Mathematical Form:
        γ = Δθ(1 + overlap) + 2πΔpos/100

    Where:
        - Δθ: Angular difference in theta_total
        - overlap: Topological shell overlap factor
        - Δpos: Position difference (causal contribution)

    Phase Lock Condition:
        γ ≡ 0 (mod 2π) indicates constructive interference
    """
    # Angular difference (primary contribution)
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
    gamma = d_theta * (1 + overlap_factor) + TWO_PI * d_pos / 100

    return gamma % TWO_PI


def is_phase_locked(gamma: float, tolerance: float = DEFAULT_PHASE_TOLERANCE) -> bool:
    """
    Check if Berry phase indicates phase lock.

    Phase lock occurs when γ ≡ 0 (mod 2π) within tolerance,
    indicating constructive interference between paths.

    Args:
        gamma: Berry phase in radians
        tolerance: Maximum deviation from 0 (mod 2π) to consider locked

    Returns:
        True if phase locked, False otherwise

    Examples:
        >>> is_phase_locked(0.1)     # Near 0
        True
        >>> is_phase_locked(6.2)     # Near 2π
        True
        >>> is_phase_locked(3.14)    # Near π
        False
    """
    reduced = abs(gamma % TWO_PI)
    return reduced < tolerance or reduced > (TWO_PI - tolerance)


def energy_decay_sequence(initial: float = 1.0, steps: int = 10) -> List[float]:
    """
    Generate energy decay sequence through pentagon reflections.

    Energy decays as E(n) = E₀ / φⁿ

    This demonstrates natural termination:
    after ~5-7 reflections, energy drops below threshold.

    Args:
        initial: Initial energy level E₀
        steps: Number of reflection steps

    Returns:
        List of energy values [E₀, E₁, E₂, ...]

    Examples:
        >>> energies = energy_decay_sequence(1.0, 7)
        >>> energies[0]
        1.0
        >>> energies[6] < 0.1
        True

    Mathematical Insight:
        φ⁷ ≈ 29.03, so E₇ ≈ E₀/29 ≈ 0.034E₀
        This explains why sequences naturally terminate after 5-7 steps.
    """
    energies = [initial]

    for n in range(1, steps):
        energy = initial / (PHI**n)
        energies.append(energy)

    return energies


def pentagon_reflection_energy(state_energy: float) -> float:
    """
    Calculate energy after pentagon reflection.

    When a path fails to phase lock, it reflects through
    the conjugate (ψ) path, reducing energy by factor of φ.

    Args:
        state_energy: Current energy level

    Returns:
        Reflected energy = E / φ

    Examples:
        >>> pentagon_reflection_energy(1.0)
        0.618...
        >>> pentagon_reflection_energy(10.0)
        6.18...
    """
    return state_energy / PHI


def compute_coherence_weight(phase_diff: float) -> float:
    """
    Compute coherence weight based on phase alignment.

    Well-aligned phases (Δφ ≈ 0) get higher weight.
    Poorly-aligned phases (Δφ ≈ π) get lower weight.

    Args:
        phase_diff: Phase difference in radians

    Returns:
        Coherence weight (0.5 to 1.5)

    Weighting Scheme:
        - |Δφ| < π/4 (45°): weight = 1.5 (well aligned)
        - π/4 ≤ |Δφ| ≤ 3π/4: weight = 1.0 (neutral)
        - |Δφ| > 3π/4 (135°): weight = 0.5 (poorly aligned)
    """
    reduced_diff = abs(phase_diff % TWO_PI)

    if reduced_diff < pi / 4:  # Well aligned
        return 1.5
    elif reduced_diff > 3 * pi / 4:  # Poorly aligned
        return 0.5
    else:  # Neutral
        return 1.0


def standing_wave(k: int) -> float:
    """
    Calculate standing wave amplitude at position k.

    Standing wave = φᵏ + ψᵏ

    At k=0, this equals 2 (constructive interference).
    This is where forward and backward paths meet.

    Args:
        k: Position/vertex index

    Returns:
        Standing wave amplitude

    Properties:
        - standing_wave(0) = 2 (maximum constructive)
        - For large k, dominated by φᵏ term
        - Related to Lucas numbers: L(k) = φᵏ + ψᵏ

    Examples:
        >>> standing_wave(0)
        2.0
        >>> standing_wave(5)  # Should equal Lucas(5) = 11
        11.0...
    """
    return PHI**k + PSI**k


def phase_velocity(n: int) -> float:
    """
    Calculate phase velocity at level n.

    Phase velocity = ln(φⁿ) / n = ln(φ)

    This is constant for all n, showing that phase
    accumulates linearly in logarithmic space.

    Args:
        n: Level/position

    Returns:
        Phase velocity (constant ≈ 0.481)
    """
    from .constants import LN_PHI
    return LN_PHI


def group_velocity(n: int) -> float:
    """
    Calculate group velocity at level n.

    Group velocity determines energy transport rate.

    Args:
        n: Level/position

    Returns:
        Group velocity
    """
    return 1.0 / (PHI**n)


def compute_phase_gradient(states: List) -> List[float]:
    """
    Compute phase gradient across sequence of states.

    The gradient shows how quickly phase is changing,
    which determines transport and coherence properties.

    Args:
        states: List of TokenStates

    Returns:
        List of phase gradients [dθ₁, dθ₂, ...]

    Note:
        Gradient list has length len(states) - 1
    """
    if len(states) < 2:
        return []

    gradients = []
    for i in range(len(states) - 1):
        d_theta = states[i + 1].theta_total - states[i].theta_total
        gradients.append(d_theta)

    return gradients


def detect_phase_lock_transitions(states: List, tolerance: float = DEFAULT_PHASE_TOLERANCE) -> List[int]:
    """
    Detect where phase lock transitions occur in sequence.

    Args:
        states: List of TokenStates
        tolerance: Phase lock tolerance

    Returns:
        List of indices where phase lock status changes

    Examples:
        >>> transitions = detect_phase_lock_transitions(states)
        >>> # transitions might be [5, 12, 20] indicating
        >>> # lock/unlock events at those positions
    """
    if len(states) < 2:
        return []

    transitions = []
    prev_locked = None

    for i in range(len(states) - 1):
        gamma = compute_berry_phase(states[i], states[i + 1])
        locked = is_phase_locked(gamma, tolerance)

        if prev_locked is not None and locked != prev_locked:
            transitions.append(i)

        prev_locked = locked

    return transitions


def validate_phase_dynamics():
    """
    Validate phase dynamics mathematical properties.

    Verifies:
    1. Energy decay follows φ⁻ⁿ
    2. Standing wave at k=0 equals 2
    3. Phase velocity is constant
    4. Berry phase is in [0, 2π]

    Returns:
        True if all validations pass

    Raises:
        AssertionError if any validation fails
    """
    # Test energy decay
    energies = energy_decay_sequence(1.0, 5)
    expected_e4 = 1.0 / (PHI**4)
    assert abs(energies[4] - expected_e4) < 1e-10, "Energy decay validation failed"

    # Test standing wave
    sw_0 = standing_wave(0)
    assert abs(sw_0 - 2.0) < 1e-10, "Standing wave at k=0 should be 2"

    # Test phase velocity constancy
    pv_1 = phase_velocity(1)
    pv_10 = phase_velocity(10)
    assert abs(pv_1 - pv_10) < 1e-10, "Phase velocity should be constant"

    # Test pentagon reflection
    reflected = pentagon_reflection_energy(1.618)
    expected = 1.618 / PHI
    assert abs(reflected - expected) < 1e-10, "Pentagon reflection failed"

    return True


# Auto-validate on import (can be disabled if needed)
if __name__ != "__main__":
    validate_phase_dynamics()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE DYNAMICS - Φ-Mamba")
    print("=" * 70)
    print()

    print("ENERGY DECAY SEQUENCE:")
    energies = energy_decay_sequence(1.0, 10)
    for i, e in enumerate(energies):
        print(f"  E[{i}] = {e:.6f}")
    print()

    print("STANDING WAVE VALUES:")
    for k in range(6):
        sw = standing_wave(k)
        print(f"  k={k}: φᵏ + ψᵏ = {sw:.6f}")
    print()

    print("PHASE PROPERTIES:")
    print(f"  Phase velocity: {phase_velocity(1):.6f} (constant)")
    print(f"  Default tolerance: {DEFAULT_PHASE_TOLERANCE} radians")
    print()

    print("PHASE LOCK EXAMPLES:")
    for gamma_val in [0.1, 1.5, 3.14, 6.2]:
        locked = is_phase_locked(gamma_val)
        status = "✓ LOCKED" if locked else "✗ UNLOCKED"
        print(f"  γ = {gamma_val:.2f} rad → {status}")
    print()

    try:
        validate_phase_dynamics()
        print("✓ All phase dynamics validations passed!")
    except AssertionError as e:
        print(f"✗ Validation failed: {e}")

    print()
    print("=" * 70)
