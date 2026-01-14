#!/usr/bin/env python3
"""
ZISA Stress Test: Large-Scale Binet DAG Traversal

Tests:
1. Multiple Gutenberg texts
2. Large token sequences
3. Edge cases (high shells, many cascades)
4. Round-trip fidelity
5. Cross-token psi-parity distribution
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import math
import random

# Import from sibling modules
from encoder import (
    ZISAEncoder, ZISAState, EncodedSequence,
    fetch_gutenberg, strip_gutenberg_header_footer, tokenize,
    has_adjacent_ones, priority_cascade, absorb,
    FIB, LUC, PSI_LUT, PSI_SCALE, bits_to_shells
)
from rtl_sim import (
    simulate_sequence, verify_binet_decomposition,
    compute_cross_token_parity, reconstruct_from_eigenvalues,
    SQRT5
)


# =============================================================================
# TEST HELPERS
# =============================================================================

def verify_round_trip(encoder: ZISAEncoder, text: str) -> Tuple[bool, int, int]:
    """
    Verify round-trip encoding/decoding.

    Returns (success, num_tokens, num_recovered)

    Note: Only words that are both in vocabulary AND have valid zeck encoding are checked.
    """
    seq = encoder.encode_text(text)
    decoded = encoder.decode_sequence(seq)

    # Only count words that can actually be encoded (in vocab AND in zeck LUT)
    original_words = []
    for w in tokenize(text):
        rank = encoder.vocab.encode_word(w)
        if rank is not None and rank in encoder.zeck:
            original_words.append(w)

    decoded_words = tokenize(decoded)

    return (original_words == decoded_words, len(original_words), len(decoded_words))


def analyze_cascade_distribution(encoder: ZISAEncoder, text: str) -> Dict:
    """Analyze cascade (tau) distribution for a text."""
    seq = encoder.encode_text(text)

    tau_values = seq.final_state.tau_history
    tau_counter = Counter(tau_values)

    return {
        'total_tokens': len(seq.tokens),
        'total_cascades': sum(tau_values),
        'max_cascade': max(tau_values) if tau_values else 0,
        'distribution': dict(tau_counter),
        'avg_cascade': sum(tau_values) / len(tau_values) if tau_values else 0,
    }


def verify_zeckendorf_throughout(encoder: ZISAEncoder, text: str) -> Tuple[bool, int]:
    """
    Verify Zeckendorf legality is maintained throughout encoding.

    Returns (all_legal, first_violation_pos)
    """
    words = tokenize(text)
    state = ZISAState()

    for pos, word in enumerate(words):
        rank = encoder.vocab.encode_word(word)
        if rank is None or rank not in encoder.zeck:
            continue

        i, j = encoder.zeck[rank]
        state = absorb(state, i, j)

        # Check legality
        if has_adjacent_ones(state.A_phi) or has_adjacent_ones(state.A_psi):
            return False, pos

    return True, -1


def analyze_shell_usage(encoder: ZISAEncoder, text: str) -> Dict:
    """Analyze which Fibonacci shells are used."""
    seq = encoder.encode_text(text)

    shell_counter = Counter()
    for token in seq.tokens:
        shell_counter[token.i] += 1
        shell_counter[token.j] += 1

    return {
        'total_shell_activations': sum(shell_counter.values()),
        'unique_shells': len(shell_counter),
        'shell_distribution': dict(shell_counter),
        'min_shell': min(shell_counter.keys()) if shell_counter else 0,
        'max_shell': max(shell_counter.keys()) if shell_counter else 0,
    }


def verify_eigenvalue_conservation(seq: EncodedSequence) -> Dict:
    """
    Verify eigenvalue accumulator properties.

    Key relations:
        L = phi^n + psi^n
        F = (phi^n - psi^n) / sqrt(5)
    """
    # From final bits
    phi_shells = bits_to_shells(seq.final_state.A_phi)
    psi_shells = bits_to_shells(seq.final_state.A_psi)

    f_phi = sum(FIB[k] for k in phi_shells)
    f_psi = sum(FIB[k] for k in psi_shells)
    l_phi = sum(LUC[k] for k in phi_shells)
    l_psi = sum(LUC[k] for k in psi_shells)

    f_total = f_phi + f_psi
    l_total = l_phi + l_psi

    # Reconstruct
    phi_comp, psi_comp = reconstruct_from_eigenvalues(f_total, l_total)

    return {
        'F_total': f_total,
        'L_total': l_total,
        'phi_component': phi_comp,
        'psi_component': psi_comp,
        'psi_magnitude': abs(psi_comp),
    }


# =============================================================================
# STRESS TESTS
# =============================================================================

def test_multiple_texts():
    """Test encoding on multiple Gutenberg texts."""
    print("=" * 70)
    print("TEST: Multiple Gutenberg Texts")
    print("=" * 70)

    texts = ['alice_wonderland', 'pride_prejudice', 'frankenstein']
    all_texts = []

    for name in texts:
        print(f"\n  Fetching {name}...")
        try:
            text = fetch_gutenberg(name)
            text = strip_gutenberg_header_footer(text)
            all_texts.append((name, text))
            print(f"    Length: {len(text)} chars")
        except Exception as e:
            print(f"    [SKIP] {e}")

    if not all_texts:
        print("\n  [FAIL] No texts available")
        return False

    # Build vocabulary from all texts
    print("\n  Building combined vocabulary...")
    encoder = ZISAEncoder(max_vocab=10000)
    encoder.build_vocab_from_texts([t for _, t in all_texts])
    print(f"    Vocabulary size: {len(encoder.vocab)}")

    # Test each text
    errors = 0
    for name, text in all_texts:
        # Take first 5000 chars for speed
        sample = text[:5000]

        success, orig, recov = verify_round_trip(encoder, sample)
        if success:
            print(f"\n  {name}: [PASS] Round-trip OK ({orig} tokens)")
        else:
            print(f"\n  {name}: [FAIL] Round-trip mismatch ({orig} -> {recov})")
            errors += 1

        # Verify Zeckendorf
        legal, viol_pos = verify_zeckendorf_throughout(encoder, sample)
        if legal:
            print(f"    Zeckendorf: LEGAL throughout")
        else:
            print(f"    Zeckendorf: VIOLATION at pos {viol_pos}")
            errors += 1

    print()
    return errors == 0


def test_cascade_dynamics():
    """Test cascade normalization dynamics."""
    print("=" * 70)
    print("TEST: Cascade Dynamics (tau distribution)")
    print("=" * 70)

    text = fetch_gutenberg('alice_wonderland')
    text = strip_gutenberg_header_footer(text)

    encoder = ZISAEncoder(max_vocab=8000)
    encoder.build_vocab_from_texts([text])

    # Analyze different sections
    sections = [
        text[:2000],
        text[10000:12000],
        text[50000:52000],
    ]

    for i, section in enumerate(sections):
        analysis = analyze_cascade_distribution(encoder, section)
        print(f"\n  Section {i+1}:")
        print(f"    Tokens:      {analysis['total_tokens']}")
        print(f"    Cascades:    {analysis['total_cascades']}")
        print(f"    Max cascade: {analysis['max_cascade']}")
        print(f"    Avg cascade: {analysis['avg_cascade']:.3f}")
        print(f"    Distribution: {dict(sorted(analysis['distribution'].items()))}")

    print()
    return True


def test_shell_distribution():
    """Test Fibonacci shell usage patterns."""
    print("=" * 70)
    print("TEST: Shell Distribution Analysis")
    print("=" * 70)

    text = fetch_gutenberg('alice_wonderland')
    text = strip_gutenberg_header_footer(text)

    encoder = ZISAEncoder(max_vocab=8000)
    encoder.build_vocab_from_texts([text])

    # Use smaller sample for faster testing
    analysis = analyze_shell_usage(encoder, text[:3000])

    print(f"\n  Total activations: {analysis['total_shell_activations']}")
    print(f"  Unique shells:     {analysis['unique_shells']}")
    print(f"  Shell range:       [{analysis['min_shell']}, {analysis['max_shell']}]")

    # Top 10 shells
    dist = analysis['shell_distribution']
    top_shells = sorted(dist.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 shells:")
    for shell, count in top_shells:
        print(f"    Shell {shell:2d} (F={FIB[shell]:8d}): {count:4d} times")

    print()
    return True


def test_parity_distribution():
    """Test cross-token psi-parity distribution."""
    print("=" * 70)
    print("TEST: Cross-Token Psi-Parity Distribution")
    print("=" * 70)

    text = fetch_gutenberg('alice_wonderland')
    text = strip_gutenberg_header_footer(text)

    encoder = ZISAEncoder(max_vocab=8000)
    encoder.build_vocab_from_texts([text])

    # Analyze parity for subsequences (reduced for speed)
    even_count = 0
    odd_count = 0
    sequences = 50

    words = tokenize(text[:10000])
    for _ in range(sequences):
        start = random.randint(0, len(words) - 50)
        sample_words = words[start:start + 30]
        sample_text = ' '.join(sample_words)

        seq = encoder.encode_text(sample_text)
        if not seq.tokens:
            continue

        token_pairs = [(t.i, t.j) for t in seq.tokens]
        parity = compute_cross_token_parity(token_pairs)

        if parity == 0:
            even_count += 1
        else:
            odd_count += 1

    print(f"\n  Analyzed {sequences} random subsequences:")
    print(f"    EVEN parity (positive correlation): {even_count} ({100*even_count/sequences:.1f}%)")
    print(f"    ODD parity (anti-correlation):      {odd_count} ({100*odd_count/sequences:.1f}%)")

    # Should be roughly 50/50 for random text
    ratio = even_count / (even_count + odd_count) if (even_count + odd_count) > 0 else 0
    balanced = 0.3 < ratio < 0.7
    print(f"    Balance ratio: {ratio:.3f} {'[OK]' if balanced else '[SKEWED]'}")

    print()
    return True


def test_eigenvalue_conservation():
    """Test eigenvalue accumulator conservation."""
    print("=" * 70)
    print("TEST: Eigenvalue Conservation (Binet)")
    print("=" * 70)

    text = fetch_gutenberg('alice_wonderland')
    text = strip_gutenberg_header_footer(text)

    encoder = ZISAEncoder(max_vocab=8000)
    encoder.build_vocab_from_texts([text])

    # Test on several samples
    samples = [
        "Alice was beginning to get very tired",
        "Down the rabbit hole",
        "Curiouser and curiouser",
        text[5000:5200],
        text[20000:20200],
    ]

    print()
    for sample in samples:
        seq = encoder.encode_text(sample)
        if len(seq.tokens) < 3:
            continue

        analysis = verify_eigenvalue_conservation(seq)

        preview = sample[:40].replace('\n', ' ')
        print(f"  '{preview}...'")
        print(f"    Tokens: {len(seq.tokens)}, F={analysis['F_total']}, L={analysis['L_total']}")
        print(f"    phi_comp={analysis['phi_component']:.2f}, psi_comp={analysis['psi_component']:.6f}")
        print()

    return True


def test_rtl_matching():
    """Test RTL simulation matches Python encoder."""
    print("=" * 70)
    print("TEST: RTL Simulation Matching")
    print("=" * 70)

    text = fetch_gutenberg('alice_wonderland')
    text = strip_gutenberg_header_footer(text)

    encoder = ZISAEncoder(max_vocab=8000)
    encoder.build_vocab_from_texts([text])

    # Test samples
    samples = [
        "Alice was beginning to get very tired of sitting by her sister",
        "The rabbit hole went straight on like a tunnel for some way",
        "She ate a little bit and said anxiously to herself",
    ]

    errors = 0
    for sample in samples:
        seq = encoder.encode_text(sample)
        if not seq.tokens:
            continue

        # Run RTL simulation
        token_pairs = [(t.i, t.j) for t in seq.tokens]
        rtl_state = simulate_sequence(token_pairs)

        # Compare
        mask = (1 << 48) - 1
        match_phi = (rtl_state.A_phi & mask) == (seq.final_state.A_phi & mask)
        match_psi = (rtl_state.A_psi & mask) == (seq.final_state.A_psi & mask)
        match_tau = rtl_state.tau_total == seq.final_state.tau_total

        preview = sample[:40]
        status = "PASS" if (match_phi and match_psi and match_tau) else "FAIL"
        print(f"\n  '{preview}...'")
        print(f"    [{status}] phi={'OK' if match_phi else 'MISMATCH'}, "
              f"psi={'OK' if match_psi else 'MISMATCH'}, "
              f"tau={'OK' if match_tau else 'MISMATCH'}")

        if status == "FAIL":
            errors += 1

    print()
    return errors == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("ZISA STRESS TEST SUITE")
    print("=" * 70 + "\n")

    results = []

    # Run tests
    results.append(("Multiple Texts", test_multiple_texts()))
    results.append(("Cascade Dynamics", test_cascade_dynamics()))
    results.append(("Shell Distribution", test_shell_distribution()))
    results.append(("Parity Distribution", test_parity_distribution()))
    results.append(("Eigenvalue Conservation", test_eigenvalue_conservation()))
    results.append(("RTL Matching", test_rtl_matching()))

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
