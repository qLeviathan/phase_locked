#!/usr/bin/env python3
"""
ZISA Full System Diagnostic

Integrates:
1. IceStorm toolchain analysis
2. ZISA encoder/decoder verification
3. Discrete mathematics corpus statistics
4. RTL simulation validation
5. FPGA resource estimation

Target: iCE40 UP5K (Upduino v3.1)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))

from encoder import (
    ZISAEncoder, ZISAState, FIB, LUC,
    has_adjacent_ones, priority_cascade, bits_to_shells,
    fetch_gutenberg, strip_gutenberg_header_footer, tokenize
)
from rtl_sim import (
    simulate_sequence, verify_binet_decomposition,
    compute_cross_token_parity
)
from discrete_corpus_stats import (
    DiscreteNGramEngine, Rational, zeckendorf_encode,
    dominant_shell, build_log_shell_lut
)


# =============================================================================
# ICE40 UP5K SPECIFICATIONS
# =============================================================================

@dataclass
class ICE40UP5KSpec:
    """iCE40 UP5K FPGA specifications."""
    name: str = "iCE40 UP5K"
    lut4: int = 5280
    dff: int = 5280
    carry: int = 5280
    ebr_bits: int = 120 * 1024  # 120 Kbit
    spram_bits: int = 1024 * 1024  # 1 Mbit = 4 × 256 Kbit banks
    dsp: int = 8  # MAC16 units
    pll: int = 1
    io: int = 39
    global_buffers: int = 8

    def __str__(self):
        return f"""
    iCE40 UP5K Resources:
    ─────────────────────────────────
    LUT4:           {self.lut4:,}
    DFF:            {self.dff:,}
    Carry chains:   {self.carry:,}
    EBR:            {self.ebr_bits // 1024} Kbit
    SPRAM:          {self.spram_bits // 1024} Kbit ({self.spram_bits // (256*1024)} banks)
    DSP (MAC16):    {self.dsp}
    PLL:            {self.pll}
    IO:             {self.io}
    Global buffers: {self.global_buffers}
    """


# =============================================================================
# RESOURCE ESTIMATION
# =============================================================================

@dataclass
class ResourceEstimate:
    """Estimated FPGA resource usage."""
    lut4: int = 0
    dff: int = 0
    ebr_bits: int = 0
    spram_bits: int = 0
    dsp: int = 0

    def add(self, other: 'ResourceEstimate') -> 'ResourceEstimate':
        return ResourceEstimate(
            lut4=self.lut4 + other.lut4,
            dff=self.dff + other.dff,
            ebr_bits=self.ebr_bits + other.ebr_bits,
            spram_bits=self.spram_bits + other.spram_bits,
            dsp=self.dsp + other.dsp
        )

    def utilization(self, spec: ICE40UP5KSpec) -> Dict[str, float]:
        return {
            'lut4': self.lut4 / spec.lut4 * 100,
            'dff': self.dff / spec.dff * 100,
            'ebr': self.ebr_bits / spec.ebr_bits * 100,
            'spram': self.spram_bits / spec.spram_bits * 100,
            'dsp': self.dsp / spec.dsp * 100,
        }


def estimate_zisa_core() -> ResourceEstimate:
    """Estimate ZISA core resource usage."""
    return ResourceEstimate(
        lut4=300,      # FSM + cascade logic
        dff=200,       # State registers (48-bit rails + accumulators)
        ebr_bits=48 * (64 + 64 + 32),  # FIB + LUC + PSI LUTs
        spram_bits=0,  # Core doesn't use SPRAM
        dsp=0
    )


def estimate_ngram_engine() -> ResourceEstimate:
    """Estimate N-gram statistics engine resource usage."""
    return ResourceEstimate(
        lut4=520,      # Hash + shell comparison
        dff=128,       # Working registers
        ebr_bits=256 * 6,  # Shell LUT (256 entries × 6-bit, larger counts computed)
        spram_bits=4 * 256 * 1024,  # 4 SPRAM banks
        dsp=1          # Rational accumulator
    )


def estimate_zeckendorf_encoder() -> ResourceEstimate:
    """Estimate Zeckendorf encoder resource usage."""
    return ResourceEstimate(
        lut4=200,      # Greedy decomposition logic
        dff=64,        # Shell registers
        ebr_bits=48 * 64,  # Fibonacci LUT
        spram_bits=0,
        dsp=0
    )


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

def test_fibonacci_lucas_tables() -> Tuple[bool, str]:
    """Verify Fibonacci and Lucas table generation."""
    errors = []

    # Test Fibonacci recurrence
    for i in range(2, 45):
        if FIB[i] != FIB[i-1] + FIB[i-2]:
            errors.append(f"FIB[{i}] recurrence failed")

    # Test Lucas recurrence
    for i in range(2, 45):
        if LUC[i] != LUC[i-1] + LUC[i-2]:
            errors.append(f"LUC[{i}] recurrence failed")

    # Test Cassini identity: L_n² - 5·F_n² = 4·(-1)^n
    for n in range(1, 40):
        lhs = LUC[n]**2 - 5 * FIB[n]**2
        rhs = 4 * ((-1)**n)
        if lhs != rhs:
            errors.append(f"Cassini identity failed at n={n}")

    if errors:
        return False, "; ".join(errors[:3])
    return True, "Fibonacci/Lucas tables verified"


def test_binet_equations() -> Tuple[bool, str]:
    """Verify Binet field equations."""
    errors = []

    # Test up to n=20 where float precision is sufficient
    # Beyond n=20, |psi^n| < 1e-8 so numerical errors dominate
    for n in range(1, 21):
        if not verify_binet_decomposition(n):
            errors.append(f"Binet failed at n={n}")

    if errors:
        return False, "; ".join(errors[:3])
    return True, f"Binet equations verified (n=1..20)"


def test_zeckendorf_bijection() -> Tuple[bool, str]:
    """Verify Zeckendorf encoding is bijective."""
    errors = []

    for n in range(1, 1000):
        shells = zeckendorf_encode(n)

        # Verify non-consecutivity
        for i in range(len(shells) - 1):
            if shells[i] - shells[i+1] == 1:
                errors.append(f"Consecutive shells at n={n}")
                break

        # Verify reconstruction
        reconstructed = sum(FIB[k] for k in shells)
        if reconstructed != n:
            errors.append(f"Reconstruction failed at n={n}")

    if errors:
        return False, "; ".join(errors[:3])
    return True, f"Zeckendorf bijection verified (n=1..999)"


def test_cascade_normalization() -> Tuple[bool, str]:
    """Verify priority cascade normalization."""
    errors = []

    # Test cases: initial bits -> expected result
    test_cases = [
        (0b110, 0b1000),      # F_1 + F_2 = F_3
        (0b11000, 0b100000),  # F_3 + F_4 = F_5
        (0b1100, 0b10000),    # F_2 + F_3 = F_4
    ]

    for initial, expected in test_cases:
        result, tau = priority_cascade(initial)
        if result != expected:
            errors.append(f"Cascade {bin(initial)} -> {bin(result)}, expected {bin(expected)}")

    # Verify all results are Zeckendorf-legal
    for n in range(1, 100):
        shells = zeckendorf_encode(n)
        bits = sum(1 << k for k in shells)
        result, _ = priority_cascade(bits)
        if has_adjacent_ones(result):
            errors.append(f"Cascade left adjacent bits for n={n}")

    if errors:
        return False, "; ".join(errors[:3])
    return True, "Cascade normalization verified"


def test_rtl_simulation() -> Tuple[bool, str]:
    """Verify RTL simulation matches Python encoder."""
    try:
        # Create encoder
        text = "The quick brown fox jumps over the lazy dog"
        encoder = ZISAEncoder(max_vocab=1000)
        encoder.build_vocab_from_texts([text])

        seq = encoder.encode_text(text)
        if len(seq.tokens) < 3:
            return False, "Not enough tokens encoded"

        # Run RTL simulation
        token_pairs = [(t.i, t.j) for t in seq.tokens]
        rtl_state = simulate_sequence(token_pairs)

        # Compare
        mask = (1 << 48) - 1
        match_phi = (rtl_state.A_phi & mask) == (seq.final_state.A_phi & mask)
        match_psi = (rtl_state.A_psi & mask) == (seq.final_state.A_psi & mask)

        if not (match_phi and match_psi):
            return False, "RTL state mismatch"

        return True, f"RTL simulation verified ({len(seq.tokens)} tokens)"
    except Exception as e:
        return False, f"RTL test error: {e}"


def test_discrete_ngram() -> Tuple[bool, str]:
    """Verify discrete N-gram engine."""
    try:
        engine = DiscreteNGramEngine(vocab_size=256)

        # Count sequence
        tokens = [0, 1, 0, 2, 0, 1, 3, 0, 0, 0]
        engine.count_sequence(tokens)

        # Verify counts
        if engine.unigram_counts[0] != 6:
            return False, f"Unigram count wrong: {engine.unigram_counts[0]}"

        # Verify rational arithmetic
        ratio = engine.unigram_ratio(0)
        if ratio.p != 6 or ratio.q != 10:
            # May be reduced
            expected = Rational(6, 10)
            if ratio.p * expected.q != expected.p * ratio.q:
                return False, f"Ratio wrong: {ratio}"

        # Verify shell computation
        shell = engine._get_shell(6)
        if shell != dominant_shell(6):
            return False, f"Shell mismatch"

        return True, "Discrete N-gram engine verified"
    except Exception as e:
        return False, f"N-gram test error: {e}"


def test_round_trip_encoding() -> Tuple[bool, str]:
    """Verify round-trip encoding/decoding."""
    try:
        text = "Alice was beginning to get very tired of sitting by her sister"
        encoder = ZISAEncoder(max_vocab=5000)
        encoder.build_vocab_from_texts([text])

        seq = encoder.encode_text(text)
        decoded = encoder.decode_sequence(seq)

        # Compare filtered words
        original = [w for w in tokenize(text)
                   if encoder.vocab.encode_word(w) and
                      encoder.vocab.encode_word(w) in encoder.zeck]
        recovered = tokenize(decoded)

        if original == recovered:
            return True, f"Round-trip verified ({len(original)} tokens)"
        return False, f"Mismatch: {len(original)} vs {len(recovered)}"
    except Exception as e:
        return False, f"Round-trip error: {e}"


# =============================================================================
# FULL DIAGNOSTIC
# =============================================================================

def run_full_diagnostic():
    """Run complete system diagnostic."""
    spec = ICE40UP5KSpec()

    print("╔" + "═" * 70 + "╗")
    print("║" + "ZISA FULL SYSTEM DIAGNOSTIC".center(70) + "║")
    print("║" + "iCE40 UP5K Target Platform".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    # Target specs
    print("\n┌─ TARGET PLATFORM ─────────────────────────────────────────────────┐")
    print(spec)
    print("└───────────────────────────────────────────────────────────────────┘")

    # Run diagnostic tests
    print("\n┌─ DIAGNOSTIC TESTS ────────────────────────────────────────────────┐")

    tests = [
        ("Fibonacci/Lucas Tables", test_fibonacci_lucas_tables),
        ("Binet Field Equations", test_binet_equations),
        ("Zeckendorf Bijection", test_zeckendorf_bijection),
        ("Cascade Normalization", test_cascade_normalization),
        ("RTL Simulation", test_rtl_simulation),
        ("Discrete N-gram Engine", test_discrete_ngram),
        ("Round-trip Encoding", test_round_trip_encoding),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            success, msg = test_fn()
            status = "PASS" if success else "FAIL"
            if success:
                passed += 1
            else:
                failed += 1
            print(f"  [{status}] {name}")
            print(f"         {msg}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}")
            print(f"         Exception: {e}")

    print("└───────────────────────────────────────────────────────────────────┘")

    # Resource estimation
    print("\n┌─ RESOURCE ESTIMATION ─────────────────────────────────────────────┐")

    estimates = {
        'ZISA Core': estimate_zisa_core(),
        'N-gram Engine': estimate_ngram_engine(),
        'Zeckendorf Encoder': estimate_zeckendorf_encoder(),
    }

    total = ResourceEstimate()
    for name, est in estimates.items():
        total = total.add(est)
        print(f"\n  {name}:")
        print(f"    LUT4: {est.lut4}, DFF: {est.dff}, EBR: {est.ebr_bits//1024}Kb, DSP: {est.dsp}")

    print(f"\n  TOTAL ESTIMATED:")
    print(f"    LUT4:  {total.lut4:,} / {spec.lut4:,}")
    print(f"    DFF:   {total.dff:,} / {spec.dff:,}")
    print(f"    EBR:   {total.ebr_bits // 1024} Kb / {spec.ebr_bits // 1024} Kb")
    print(f"    SPRAM: {total.spram_bits // 1024} Kb / {spec.spram_bits // 1024} Kb")
    print(f"    DSP:   {total.dsp} / {spec.dsp}")

    util = total.utilization(spec)
    print(f"\n  UTILIZATION:")
    print(f"    LUT4:  {util['lut4']:.1f}%")
    print(f"    DFF:   {util['dff']:.1f}%")
    print(f"    EBR:   {util['ebr']:.1f}%")
    print(f"    SPRAM: {util['spram']:.1f}%")
    print(f"    DSP:   {util['dsp']:.1f}%")

    print("└───────────────────────────────────────────────────────────────────┘")

    # Architecture summary
    print("\n┌─ ARCHITECTURE SUMMARY ────────────────────────────────────────────┐")
    print("""
  ZISA State Machine:
  ┌─────────────────────────────────────────────────────────────────┐
  │  IDLE ──▶ ABSORB ──▶ CASCADE ──▶ UPDATE ──▶ DONE ──▶ IDLE     │
  │    ↑                    │  ↑                                   │
  │    │                    └──┘ (while adjacent bits)             │
  │    └───────────────────────────────────────────────────────────┘
  │                                                                 │
  │  Dual Rails: A_phi (even pos) │ A_psi (odd pos)                │
  │  Cascade: F_k + F_{k-1} = F_{k+2}  (τ = work quanta)           │
  │  Invariant: No adjacent 1-bits (Zeckendorf property)           │
  └─────────────────────────────────────────────────────────────────┘

  Discrete Math Engine:
  ┌─────────────────────────────────────────────────────────────────┐
  │  • Rational arithmetic (p/q) - exact, no floats                │
  │  • Shell indices - dominant Fibonacci approximation            │
  │  • Zeckendorf decomposition - discrete log_φ                   │
  │  • Shell difference - approximate log probability              │
  └─────────────────────────────────────────────────────────────────┘

  Key Relations:
    φ^n = F_n·φ + F_{n-1}          (Binet decomposition)
    L_n² - 5·F_n² = 4·(-1)^n       (Cassini identity)
    shell(c) - shell(T) ≈ log_φ(P) (probability approximation)
""")
    print("└───────────────────────────────────────────────────────────────────┘")

    # Summary
    print("\n╔" + "═" * 70 + "╗")
    if failed == 0:
        print("║" + f"ALL {passed} TESTS PASSED".center(70) + "║")
    else:
        print("║" + f"{passed} PASSED, {failed} FAILED".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    return failed == 0


if __name__ == '__main__':
    success = run_full_diagnostic()
    sys.exit(0 if success else 1)
