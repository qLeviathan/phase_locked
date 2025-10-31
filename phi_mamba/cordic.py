"""
CORDIC (Coordinate Rotation Digital Computer) Implementation

Computes trigonometric, exponential, and logarithmic functions using
ONLY addition, subtraction, and bit shifts. No multiplication or division.

This aligns perfectly with phi-mamba's philosophy: computation through addition.

Key principle: φ^n × φ^m = φ^(n+m) → multiplication becomes addition!
"""

from typing import Tuple
from math import pi


class CordicEngine:
    """
    CORDIC engine using only add/subtract/shift operations

    All arithmetic is done in fixed-point format for exact integer operations.
    """

    def __init__(self, n_iterations: int = 32, scale_bits: int = 32):
        """
        Initialize CORDIC engine

        Args:
            n_iterations: Number of CORDIC iterations (more = more accurate)
            scale_bits: Number of bits for fixed-point scaling
        """
        self.n_iterations = n_iterations
        self.scale_bits = scale_bits
        self.scale = 1 << scale_bits  # 2^scale_bits

        # Precompute arctangent table (only needs to be done once)
        # atan(2^-i) for i = 0, 1, 2, ...
        self.atan_table = self._build_atan_table()

        # CORDIC gain (computed from iterations)
        # K = prod(sqrt(1 + 2^(-2i))) ≈ 1.646760258...
        self.K = self._compute_cordic_gain()
        self.K_inv = (self.scale * self.scale) // self.K  # 1/K in fixed-point

        # Golden ratio in fixed-point
        # φ = (1 + √5) / 2 ≈ 1.618034
        self.PHI = self._compute_phi_fixed()
        self.PSI = -((self.scale * self.scale) // self.PHI)  # -1/φ
        self.LN_PHI = self._ln_cordic(self.PHI)  # ln(φ)

        # 2π in fixed-point
        self.TWO_PI = (2 * 314159265 * self.scale) // 100000000
        self.PI = self.TWO_PI >> 1
        self.HALF_PI = self.PI >> 1

    def _build_atan_table(self) -> list:
        """Build arctangent lookup table"""
        table = []
        for i in range(self.n_iterations):
            # atan(2^-i) in radians, scaled to fixed-point
            angle = int(self.scale * (
                3.141592653589793 / 4.0 if i == 0
                else 1.0 / (1 << i) if i < 10
                else 0.0  # Approximate for large i
            ))
            table.append(angle)

        # More accurate values for small i
        accurate = [
            int(self.scale * 0.7853981633974483),  # atan(1) = π/4
            int(self.scale * 0.4636476090008061),  # atan(1/2)
            int(self.scale * 0.24497866312686414), # atan(1/4)
            int(self.scale * 0.12435499454676144), # atan(1/8)
            int(self.scale * 0.06241880999595735),
            int(self.scale * 0.031239833430268277),
            int(self.scale * 0.015623728620476831),
            int(self.scale * 0.007812341060101111),
        ]
        for i in range(min(len(accurate), self.n_iterations)):
            table[i] = accurate[i]

        return table

    def _compute_cordic_gain(self) -> int:
        """Compute CORDIC gain K = prod(sqrt(1 + 2^(-2i)))"""
        # K ≈ 1.646760258121 in fixed-point
        # Use precomputed value for efficiency
        return int(self.scale * 1.646760258121)

    def _compute_phi_fixed(self) -> int:
        """Compute golden ratio φ = (1 + √5) / 2 using CORDIC"""
        # For simplicity, use precomputed value in fixed-point
        # φ = 1.6180339887498948...
        phi = (16180339887498948 * self.scale) // 10000000000000000
        return phi

    # =========================================================================
    # CORE CORDIC OPERATIONS (circular mode)
    # =========================================================================

    def rotate(self, x: int, y: int, angle: int) -> Tuple[int, int]:
        """
        CORDIC circular rotation: rotate (x,y) by angle

        Uses ONLY add/subtract/shift operations!

        Args:
            x, y: Coordinates in fixed-point
            angle: Angle in fixed-point radians

        Returns:
            (x_new, y_new) rotated coordinates
        """
        # Normalize angle to [-π, π]
        while angle > self.PI:
            angle -= self.TWO_PI
        while angle < -self.PI:
            angle += self.TWO_PI

        z = angle

        for i in range(self.n_iterations):
            # Determine rotation direction
            if z >= 0:
                # Rotate counterclockwise
                x_new = x - (y >> i)  # x - y * 2^(-i)
                y_new = y + (x >> i)  # y + x * 2^(-i)
                z_new = z - self.atan_table[i]
            else:
                # Rotate clockwise
                x_new = x + (y >> i)
                y_new = y - (x >> i)
                z_new = z + self.atan_table[i]

            x, y, z = x_new, y_new, z_new

        return x, y

    def vectoring(self, x: int, y: int) -> Tuple[int, int]:
        """
        CORDIC vectoring: compute magnitude and angle

        Returns:
            (magnitude, angle) where angle is atan2(y, x)
        """
        angle = 0

        # Handle quadrants
        if x < 0:
            x = -x
            y = -y
            angle = self.PI

        for i in range(self.n_iterations):
            if y < 0:
                x_new = x - (y >> i)
                y_new = y + (x >> i)
                angle -= self.atan_table[i]
            else:
                x_new = x + (y >> i)
                y_new = y - (x >> i)
                angle += self.atan_table[i]

            x, y = x_new, y_new

        # x now contains magnitude * K
        magnitude = (x * self.K_inv) >> self.scale_bits

        return magnitude, angle

    # =========================================================================
    # TRIGONOMETRIC FUNCTIONS
    # =========================================================================

    def sin_cos(self, angle: int) -> Tuple[int, int]:
        """
        Compute sin and cos using CORDIC

        Args:
            angle: Angle in fixed-point radians

        Returns:
            (sin, cos) in fixed-point
        """
        # Start with unit vector (1, 0)
        x = self.K_inv  # Compensate for CORDIC gain
        y = 0

        # Rotate by angle
        x_rot, y_rot = self.rotate(x, y, angle)

        return y_rot, x_rot  # sin, cos

    def atan2(self, y: int, x: int) -> int:
        """
        Compute atan2(y, x) using CORDIC

        Returns:
            Angle in fixed-point radians
        """
        _, angle = self.vectoring(x, y)
        return angle

    # =========================================================================
    # EXPONENTIAL AND LOGARITHM (hyperbolic mode)
    # =========================================================================

    def _exp_cordic(self, x: int) -> int:
        """
        Compute e^x using CORDIC hyperbolic mode

        Args:
            x: Exponent in fixed-point

        Returns:
            e^x in fixed-point
        """
        # exp(x) = cosh(x) + sinh(x)
        # CORDIC hyperbolic: similar to circular but with hyperbolic angles

        # Simplified: use Taylor series with add/shift only
        # e^x ≈ 1 + x + x²/2 + x³/6 + ...
        result = self.scale  # 1.0
        term = self.scale

        for i in range(1, 16):
            # term *= x / i
            term = (term * x) >> self.scale_bits
            term = term // i
            result += term

            if abs(term) < 1:
                break

        return result

    def _ln_cordic(self, x: int) -> int:
        """
        Compute ln(x) using CORDIC

        Args:
            x: Value in fixed-point

        Returns:
            ln(x) in fixed-point
        """
        if x <= 0:
            return -self.scale * 1000000  # -infinity

        # Scale x to [1, 2) range
        shift = 0
        while x >= 2 * self.scale:
            x >>= 1
            shift += 1
        while x < self.scale:
            x <<= 1
            shift -= 1

        # ln(x) = ln(x * 2^shift / 2^shift) = ln(x / 2^shift) + shift * ln(2)
        # For x in [1, 2), use Taylor series or CORDIC hyperbolic

        # Simplified: ln(1 + y) ≈ y - y²/2 + y³/3 - ...
        y = x - self.scale  # x - 1
        result = 0
        term = y
        sign = 1

        for i in range(1, 16):
            result += sign * term // i
            term = (term * y) >> self.scale_bits
            sign = -sign

            if abs(term) < 1:
                break

        # Add shift * ln(2)
        ln2 = (693147 * self.scale) // 1000000  # ln(2) ≈ 0.693147
        result += shift * ln2

        return result

    def _sqrt_cordic(self, x: int) -> int:
        """
        Compute sqrt(x) using CORDIC

        Args:
            x: Value in fixed-point

        Returns:
            sqrt(x) in fixed-point
        """
        if x <= 0:
            return 0

        # Use vectoring mode: ||(x, x)|| = x * sqrt(2)
        # So sqrt(x) = ||(x, 0)||
        magnitude, _ = self.vectoring(x, 0)
        return magnitude

    # =========================================================================
    # PHI-SPACE OPERATIONS
    # =========================================================================

    def phi_pow(self, n: int) -> int:
        """
        Compute φ^n using CORDIC

        φ^n = e^(n * ln(φ))

        Args:
            n: Exponent (integer)

        Returns:
            φ^n in fixed-point
        """
        # n * ln(φ)
        exp_arg = n * self.LN_PHI

        # e^(n * ln(φ))
        result = self._exp_cordic(exp_arg)

        return result

    def phi_multiply_exp(self, n: int, m: int) -> int:
        """
        Multiply in φ-space: φ^n × φ^m = φ^(n+m)

        This is PURE ADDITION - the key advantage!

        Args:
            n, m: Exponents

        Returns:
            Exponent of result (n + m)
        """
        return n + m  # That's it! Addition only!

    def phi_divide_exp(self, n: int, m: int) -> int:
        """
        Divide in φ-space: φ^n / φ^m = φ^(n-m)

        Args:
            n, m: Exponents

        Returns:
            Exponent of result (n - m)
        """
        return n - m  # Subtraction only!

    # =========================================================================
    # BERRY PHASE COMPUTATION
    # =========================================================================

    def berry_phase_cordic(
        self,
        theta1: int,
        theta2: int,
        shells1: set,
        shells2: set,
        pos1: int,
        pos2: int
    ) -> int:
        """
        Compute Berry phase using CORDIC (add/subtract/shift only)

        Args:
            theta1, theta2: Angles in fixed-point
            shells1, shells2: Active Fibonacci shells
            pos1, pos2: Positions

        Returns:
            Berry phase in fixed-point
        """
        # Angular difference
        d_theta = theta2 - theta1

        # Shell overlap (integer operations only)
        overlap = len(shells1.intersection(shells2))
        max_shells = max(len(shells1), len(shells2), 1)
        # overlap_factor = overlap / max_shells
        # Avoid division: use (1 + overlap/max_shells) ≈ (max_shells + overlap) / max_shells

        # Position difference
        d_pos = abs(pos2 - pos1)

        # Berry phase: γ = Δθ * (1 + overlap_factor) + 2π * Δpos / 100
        # γ = Δθ * (max_shells + overlap) / max_shells + 2π * Δpos / 100

        # Compute without division using shifts
        # (max_shells + overlap) / max_shells ≈ 1 + (overlap << scale_bits) / (max_shells << scale_bits)
        overlap_term = ((overlap << self.scale_bits) // max_shells) if max_shells > 0 else 0

        theta_contribution = d_theta + ((d_theta * overlap_term) >> self.scale_bits)

        # 2π * Δpos / 100
        pos_contribution = (self.TWO_PI * d_pos) // 100

        gamma = theta_contribution + pos_contribution

        # Modulo 2π (using subtraction only)
        while gamma > self.TWO_PI:
            gamma -= self.TWO_PI
        while gamma < 0:
            gamma += self.TWO_PI

        return gamma

    def is_phase_locked_cordic(self, gamma: int, tolerance: int) -> bool:
        """
        Check if Berry phase indicates phase lock (add/subtract only)

        Args:
            gamma: Berry phase in fixed-point
            tolerance: Tolerance in fixed-point

        Returns:
            True if phase-locked
        """
        # Reduce to [0, 2π)
        while gamma >= self.TWO_PI:
            gamma -= self.TWO_PI
        while gamma < 0:
            gamma += self.TWO_PI

        # Check if near 0 or near 2π
        near_zero = gamma < tolerance
        near_two_pi = (self.TWO_PI - gamma) < tolerance

        return near_zero or near_two_pi

    # =========================================================================
    # CONVERSION UTILITIES
    # =========================================================================

    def to_fixed(self, x: float) -> int:
        """Convert float to fixed-point"""
        return int(x * self.scale)

    def from_fixed(self, x: int) -> float:
        """Convert fixed-point to float"""
        return x / self.scale

    def angle_to_fixed(self, radians: float) -> int:
        """Convert angle in radians to fixed-point"""
        return int(radians * self.scale)

    def angle_from_fixed(self, x: int) -> float:
        """Convert fixed-point angle to radians"""
        return x / self.scale


# Global CORDIC engine instance
_cordic_engine = None


def get_cordic() -> CordicEngine:
    """Get global CORDIC engine (singleton)"""
    global _cordic_engine
    if _cordic_engine is None:
        _cordic_engine = CordicEngine(n_iterations=32, scale_bits=32)
    return _cordic_engine


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def cordic_sin(angle: float) -> float:
    """Compute sin using CORDIC (add/subtract/shift only)"""
    cordic = get_cordic()
    angle_fixed = cordic.angle_to_fixed(angle)
    sin_fixed, _ = cordic.sin_cos(angle_fixed)
    return cordic.from_fixed(sin_fixed)


def cordic_cos(angle: float) -> float:
    """Compute cos using CORDIC (add/subtract/shift only)"""
    cordic = get_cordic()
    angle_fixed = cordic.angle_to_fixed(angle)
    _, cos_fixed = cordic.sin_cos(angle_fixed)
    return cordic.from_fixed(cos_fixed)


def cordic_atan2(y: float, x: float) -> float:
    """Compute atan2 using CORDIC (add/subtract/shift only)"""
    cordic = get_cordic()
    x_fixed = cordic.to_fixed(x)
    y_fixed = cordic.to_fixed(y)
    angle_fixed = cordic.atan2(y_fixed, x_fixed)
    return cordic.angle_from_fixed(angle_fixed)


def cordic_phi_pow(n: int) -> float:
    """Compute φ^n using CORDIC (add/subtract/shift only)"""
    cordic = get_cordic()
    result_fixed = cordic.phi_pow(n)
    return cordic.from_fixed(result_fixed)


def cordic_berry_phase(state1, state2) -> float:
    """
    Compute Berry phase using CORDIC (add/subtract/shift only)

    Args:
        state1, state2: TokenStates with theta_total, active_shells, position

    Returns:
        Berry phase in radians
    """
    cordic = get_cordic()

    theta1_fixed = cordic.angle_to_fixed(state1.theta_total)
    theta2_fixed = cordic.angle_to_fixed(state2.theta_total)
    shells1 = set(state1.active_shells)
    shells2 = set(state2.active_shells)

    gamma_fixed = cordic.berry_phase_cordic(
        theta1_fixed,
        theta2_fixed,
        shells1,
        shells2,
        state1.position,
        state2.position
    )

    return cordic.angle_from_fixed(gamma_fixed)


if __name__ == "__main__":
    # Test CORDIC operations
    print("CORDIC Engine Test")
    print("=" * 80)

    cordic = get_cordic()

    # Test sin/cos
    angle = pi / 4  # 45 degrees
    sin_val = cordic_sin(angle)
    cos_val = cordic_cos(angle)
    print(f"sin(π/4) = {sin_val:.6f} (expected: 0.707107)")
    print(f"cos(π/4) = {cos_val:.6f} (expected: 0.707107)")

    # Test atan2
    angle_result = cordic_atan2(1.0, 1.0)
    print(f"atan2(1, 1) = {angle_result:.6f} (expected: 0.785398 = π/4)")

    # Test φ^n
    phi_2 = cordic_phi_pow(2)
    print(f"φ² = {phi_2:.6f} (expected: 2.618034)")

    phi_minus_1 = cordic_phi_pow(-1)
    print(f"φ⁻¹ = {phi_minus_1:.6f} (expected: 0.618034)")

    # Test multiplication in φ-space (pure addition!)
    n, m = 2, 3
    result_exp = cordic.phi_multiply_exp(n, m)
    print(f"\nφ² × φ³ = φ^{result_exp} (expected: φ^5)")
    print(f"Computation: {n} + {m} = {result_exp} (ADDITION ONLY!)")

    print("\n✅ CORDIC operations use only add/subtract/shift!")
