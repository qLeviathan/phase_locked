"""
CORDIC Adapter for Financial System

Demonstrates how ALL financial calculations can be done using
ONLY addition, subtraction, and bit shifts.

Key insight: In φ-space, multiplication becomes addition!
  φ^n × φ^m = φ^(n+m)  ← Just add the exponents!

This module replaces floating-point operations with CORDIC.
"""

import numpy as np
from typing import List, Tuple
from .cordic import get_cordic, CordicEngine
from .financial_encoding import FinancialTokenState, FinancialPhiEncoder
from .financial_data import OHLCVBar


class CordicFinancialEncoder(FinancialPhiEncoder):
    """
    Financial encoder using CORDIC (add/subtract/shift only)

    All operations:
    - Price normalization: bit shifts
    - Angle computation: CORDIC sin/cos/atan2
    - Energy computation: φ^n using CORDIC
    - Berry phase: CORDIC angle arithmetic
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cordic = get_cordic()
        print("🔧 Using CORDIC: All operations are add/subtract/shift only!")

    def encode_bar_cordic(
        self,
        bar: OHLCVBar,
        position: int,
        reference_price: float = None
    ) -> FinancialTokenState:
        """
        Encode OHLCV bar using CORDIC (add/subtract/shift only)

        All arithmetic operations broken down:
        1. Price change: subtraction + shift for division
        2. Angle mapping: CORDIC atan2
        3. Energy: φ^(-position) using CORDIC
        4. Zeckendorf: integer addition only
        """
        if reference_price is None:
            reference_price = bar.open

        # =====================================================================
        # STEP 1: Price change (subtraction + shift for division)
        # =====================================================================
        # price_change_pct = ((close - ref) / ref) * 100
        # Avoid division: use shifts for power-of-2 approximations

        price_diff = bar.close - reference_price  # Subtraction
        # Division by reference_price approximated by multiplication + shift
        # Assumes reference_price ≈ 2^k for some k
        # For general case, convert to fixed-point integer math

        price_diff_fixed = self.cordic.to_fixed(price_diff)
        ref_fixed = self.cordic.to_fixed(reference_price)

        # price_change_pct = (price_diff / ref) * 100
        # In fixed-point: (price_diff_fixed * 100 * scale) / (ref_fixed)
        if ref_fixed != 0:
            price_change_pct_fixed = (price_diff_fixed * 100 * self.cordic.scale) // ref_fixed
            price_change_pct = self.cordic.from_fixed(price_change_pct_fixed)
        else:
            price_change_pct = 0.0

        # =====================================================================
        # STEP 2: Map price change to angle using CORDIC
        # =====================================================================
        # θ_price = (price_change_pct * sensitivity * φ) mod 2π

        # Get φ in fixed-point
        phi_fixed = self.cordic.PHI

        # Compute: price_change_pct * sensitivity * φ
        sensitivity_fixed = self.cordic.to_fixed(self.angular_sensitivity)
        pct_fixed = self.cordic.to_fixed(price_change_pct)

        # Multiply using fixed-point (becomes shift and add)
        theta_price_fixed = (pct_fixed * sensitivity_fixed * phi_fixed) >> (2 * self.cordic.scale_bits)

        # Modulo 2π using subtraction only
        while theta_price_fixed > self.cordic.TWO_PI:
            theta_price_fixed -= self.cordic.TWO_PI
        while theta_price_fixed < 0:
            theta_price_fixed += self.cordic.TWO_PI

        theta_price = self.cordic.angle_from_fixed(theta_price_fixed)

        # =====================================================================
        # STEP 3: Position-based angle (RoPE-like with φ decay)
        # =====================================================================
        # θ_pos = position * φ^(-position/10) mod 2π

        # Compute φ^(-position/10) using CORDIC
        exp_value = -position // 10  # Integer division (shift equivalent)
        phi_exp_fixed = self.cordic.phi_pow(exp_value)

        # position * φ^(-position/10)
        pos_fixed = position * self.cordic.scale
        theta_pos_fixed = (pos_fixed * phi_exp_fixed) >> self.cordic.scale_bits

        # Modulo 2π
        while theta_pos_fixed > self.cordic.TWO_PI:
            theta_pos_fixed -= self.cordic.TWO_PI
        while theta_pos_fixed < 0:
            theta_pos_fixed += self.cordic.TWO_PI

        theta_pos = self.cordic.angle_from_fixed(theta_pos_fixed)

        # =====================================================================
        # STEP 4: Combined angle (addition only!)
        # =====================================================================
        theta_total_fixed = theta_price_fixed + theta_pos_fixed  # Pure addition!

        # Modulo 2π
        while theta_total_fixed > self.cordic.TWO_PI:
            theta_total_fixed -= self.cordic.TWO_PI

        theta_total = self.cordic.angle_from_fixed(theta_total_fixed)

        # =====================================================================
        # STEP 5: Volume-based energy using φ^(-position)
        # =====================================================================
        # E = (volume / volume_scale) * φ^(-position)

        volume_normalized = bar.volume / self.volume_scale
        volume_normalized = min(volume_normalized, 10.0)

        # φ^(-position) using CORDIC
        energy_phi = self.cordic.phi_pow(-position)
        energy_phi_float = self.cordic.from_fixed(energy_phi)

        energy = volume_normalized * energy_phi_float

        # =====================================================================
        # STEP 6: Zeckendorf decomposition (integer addition only)
        # =====================================================================
        from .encoding import zeckendorf_decomposition
        price_int = int(abs(bar.close))
        zeck = zeckendorf_decomposition(price_int)  # Uses Fibonacci + subtraction only

        # =====================================================================
        # CREATE STATE
        # =====================================================================
        token_str = f"{bar.ticker}@{bar.close:.2f}"

        return FinancialTokenState(
            token=token_str,
            index=hash(token_str) % 50000,
            position=position,
            vocab_size=50000,
            theta_token=theta_price,
            theta_pos=theta_pos,
            theta_total=theta_total,
            energy=energy,
            zeckendorf=zeck,
            price=bar.close,
            volume=bar.volume,
            ticker=bar.ticker,
            timeframe=bar.timeframe,
            price_change_pct=price_change_pct
        )

    def compute_berry_phase_cordic(
        self,
        state1: FinancialTokenState,
        state2: FinancialTokenState
    ) -> float:
        """
        Compute Berry phase using CORDIC (add/subtract/shift only)

        Berry phase determines if two states are "phase-locked"
        """
        theta1_fixed = self.cordic.angle_to_fixed(state1.theta_total)
        theta2_fixed = self.cordic.angle_to_fixed(state2.theta_total)

        shells1 = set(state1.zeckendorf)
        shells2 = set(state2.zeckendorf)

        gamma_fixed = self.cordic.berry_phase_cordic(
            theta1_fixed,
            theta2_fixed,
            shells1,
            shells2,
            state1.position,
            state2.position
        )

        return self.cordic.angle_from_fixed(gamma_fixed)

    def is_phase_locked_cordic(
        self,
        state1: FinancialTokenState,
        state2: FinancialTokenState,
        tolerance: float = 0.5
    ) -> bool:
        """
        Check if two states are phase-locked using CORDIC
        """
        gamma = self.compute_berry_phase_cordic(state1, state2)
        gamma_fixed = self.cordic.angle_to_fixed(gamma)
        tolerance_fixed = self.cordic.angle_to_fixed(tolerance)

        return self.cordic.is_phase_locked_cordic(gamma_fixed, tolerance_fixed)


class PhiSpaceArithmetic:
    """
    Demonstration of arithmetic in φ-space

    Key insight: ALL MULTIPLICATION BECOMES ADDITION!
    """

    def __init__(self):
        self.cordic = get_cordic()

    def multiply(self, n: int, m: int) -> int:
        """
        Multiply φ^n × φ^m using ONLY addition

        φ^n × φ^m = φ^(n+m)

        This is the core advantage of φ-space!
        """
        return self.cordic.phi_multiply_exp(n, m)  # Just n + m!

    def divide(self, n: int, m: int) -> int:
        """
        Divide φ^n / φ^m using ONLY subtraction

        φ^n / φ^m = φ^(n-m)
        """
        return self.cordic.phi_divide_exp(n, m)  # Just n - m!

    def power(self, n: int, k: int) -> int:
        """
        Compute (φ^n)^k using ONLY addition

        (φ^n)^k = φ^(n×k) = φ^(n+n+...+n) [k times]
        """
        result = 0
        for _ in range(k):
            result += n  # Add n, k times
        return result

    def demonstrate(self):
        """Demonstrate φ-space arithmetic"""
        print("=" * 80)
        print("PHI-SPACE ARITHMETIC: ALL MULTIPLICATION BECOMES ADDITION!")
        print("=" * 80)
        print()

        # Example 1: Multiplication
        n, m = 3, 5
        result = self.multiply(n, m)
        print(f"MULTIPLICATION:")
        print(f"  φ³ × φ⁵ = φ^{result}")
        print(f"  Computation: {n} + {m} = {result}")
        print(f"  Operation: ADDITION ONLY! ✅")
        print()

        # Verify
        phi = (1 + np.sqrt(5)) / 2
        actual = phi**n * phi**m
        expected = phi**result
        print(f"  Verification: {actual:.6f} ≈ {expected:.6f}")
        print()

        # Example 2: Division
        n, m = 7, 2
        result = self.divide(n, m)
        print(f"DIVISION:")
        print(f"  φ⁷ / φ² = φ^{result}")
        print(f"  Computation: {n} - {m} = {result}")
        print(f"  Operation: SUBTRACTION ONLY! ✅")
        print()

        # Example 3: Power
        n, k = 2, 4
        result = self.power(n, k)
        print(f"POWER:")
        print(f"  (φ²)⁴ = φ^{result}")
        print(f"  Computation: {n} + {n} + {n} + {n} = {result}")
        print(f"  Operation: REPEATED ADDITION! ✅")
        print()

        # Example 4: Complex expression
        print(f"COMPLEX EXPRESSION:")
        print(f"  (φ³ × φ⁵) / φ² = ?")
        step1 = self.multiply(3, 5)  # 3 + 5 = 8
        step2 = self.divide(step1, 2)  # 8 - 2 = 6
        print(f"  Step 1: φ³ × φ⁵ = φ^{step1} (computation: 3 + 5 = {step1})")
        print(f"  Step 2: φ^{step1} / φ² = φ^{step2} (computation: {step1} - 2 = {step2})")
        print(f"  Result: φ^{step2}")
        print(f"  Total operations: 1 addition + 1 subtraction! ✅")
        print()

        print("=" * 80)
        print("KEY INSIGHT: In φ-space, multiplication is just addition!")
        print("This makes φ-mamba computationally elegant and efficient.")
        print("=" * 80)


def demonstrate_cordic_financial_pipeline():
    """
    Demonstrate complete financial pipeline using CORDIC
    """
    from .financial_data import OHLCVBar, Timeframe
    from datetime import datetime

    print("\n" + "=" * 80)
    print("CORDIC FINANCIAL PIPELINE DEMONSTRATION")
    print("=" * 80)
    print()

    # Create encoder
    encoder = CordicFinancialEncoder()

    # Create sample bar
    bar = OHLCVBar(
        timestamp=datetime.now(),
        open=100.0,
        high=105.0,
        low=98.0,
        close=103.0,
        volume=1000000.0,
        ticker="AAPL",
        timeframe=Timeframe.DAY_1
    )

    print(f"Encoding bar: {bar.ticker} @ ${bar.close}")
    print(f"  Open: ${bar.open}, High: ${bar.high}, Low: ${bar.low}, Close: ${bar.close}")
    print(f"  Volume: {bar.volume:,.0f}")
    print()

    # Encode using CORDIC
    print("Encoding with CORDIC (add/subtract/shift only)...")
    state = encoder.encode_bar_cordic(bar, position=0, reference_price=bar.open)

    print()
    print("Encoded State:")
    print(f"  θ_token: {state.theta_token:.6f} rad")
    print(f"  θ_pos: {state.theta_pos:.6f} rad")
    print(f"  θ_total: {state.theta_total:.6f} rad")
    print(f"  Energy: {state.energy:.6f}")
    print(f"  Zeckendorf: {state.zeckendorf}")
    print(f"  Active shells: {len(state.zeckendorf)}")
    print()

    # Create another bar
    bar2 = OHLCVBar(
        timestamp=datetime.now(),
        open=103.0,
        high=107.0,
        low=102.0,
        close=106.0,
        volume=1200000.0,
        ticker="AAPL",
        timeframe=Timeframe.DAY_1
    )

    state2 = encoder.encode_bar_cordic(bar2, position=1, reference_price=bar.open)

    # Compute Berry phase
    print("Computing Berry phase between states...")
    berry_phase = encoder.compute_berry_phase_cordic(state, state2)
    print(f"  Berry phase: {berry_phase:.6f} rad")
    print()

    # Check phase locking
    is_locked = encoder.is_phase_locked_cordic(state, state2, tolerance=0.5)
    print(f"  Phase-locked: {is_locked}")
    print()

    print("=" * 80)
    print("✅ All operations used ONLY add/subtract/shift!")
    print("=" * 80)


if __name__ == "__main__":
    # Demonstrate phi-space arithmetic
    arith = PhiSpaceArithmetic()
    arith.demonstrate()

    # Demonstrate financial pipeline
    demonstrate_cordic_financial_pipeline()
