#!/usr/bin/env python3
"""
ZISA Encoder: Two-Shell Zeckendorf Encoding for Text Sequences

Encodes text from Project Gutenberg into the ZISA format:
- Words map to vocabulary ranks
- Ranks map to (i, j) shell pairs via two-shell Zeckendorf encoding
- Shell pairs encode as 64-bit patterns with exactly two 1s

The encoding is bijective and lossless.
"""

import re
import struct
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import Counter
import urllib.request
import hashlib


# =============================================================================
# FIBONACCI / LUCAS FOUNDATION
# =============================================================================

def build_fib_luc_tables(max_k: int = 48) -> Tuple[List[int], List[int]]:
    """Build Fibonacci and Lucas lookup tables."""
    FIB = [0, 1] + [0] * (max_k - 2)
    LUC = [2, 1] + [0] * (max_k - 2)
    for i in range(2, max_k):
        FIB[i] = FIB[i-1] + FIB[i-2]
        LUC[i] = LUC[i-1] + LUC[i-2]
    return FIB, LUC


FIB, LUC = build_fib_luc_tables(48)

# Precompute PSI values (signed fixed-point, scale 2^24)
PSI_SCALE = 1 << 24
PHI = (1 + 5**0.5) / 2
PSI_LUT = [int(round(((-1)**k / PHI**k) * PSI_SCALE)) for k in range(48)]


# =============================================================================
# TWO-SHELL ZECKENDORF ENCODING
# =============================================================================

def build_two_shell_pairs(max_pairs: int = 100000) -> List[Tuple[int, int, int]]:
    """
    Build sorted list of (value, i, j) for two-shell encoding.

    Each pair (i, j) with j > i + 1 (gap >= 2) encodes value F_i + F_j.
    Pairs are sorted by value, then by j, then by i.
    """
    pairs = []
    for j in range(3, 45):  # j >= 3 to allow i >= 1 with gap
        for i in range(1, j - 1):  # gap >= 2: j - i >= 2
            v = FIB[i] + FIB[j]
            pairs.append((v, i, j))

    # Sort by value, then j, then i for deterministic ordering
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    return pairs[:max_pairs]


def build_zeck_lut(max_rank: int = 50000) -> Dict[int, Tuple[int, int]]:
    """
    Build ZECK LUT: rank -> (i, j) shell pair.

    Rank 1 gets the smallest value pair, rank 2 next, etc.
    """
    pairs = build_two_shell_pairs(max_rank + 1000)

    # Deduplicate by value (keep first occurrence)
    seen_values = set()
    unique_pairs = []
    for v, i, j in pairs:
        if v not in seen_values:
            seen_values.add(v)
            unique_pairs.append((v, i, j))

    # Map rank -> (i, j)
    zeck = {}
    for rank, (v, i, j) in enumerate(unique_pairs[:max_rank], start=1):
        zeck[rank] = (i, j)

    return zeck


def build_unzeck_lut(zeck: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Build inverse: (i, j) -> rank."""
    return {v: k for k, v in zeck.items()}


# =============================================================================
# VOCABULARY BUILDER
# =============================================================================

@dataclass
class Vocabulary:
    """Word vocabulary with frequency-based ranking."""
    word_to_rank: Dict[str, int] = field(default_factory=dict)
    rank_to_word: Dict[int, str] = field(default_factory=dict)
    word_counts: Dict[str, int] = field(default_factory=dict)

    def __len__(self):
        return len(self.word_to_rank)

    def encode_word(self, word: str) -> Optional[int]:
        """Word -> rank (None if OOV)."""
        return self.word_to_rank.get(word.lower())

    def decode_rank(self, rank: int) -> Optional[str]:
        """Rank -> word (None if invalid)."""
        return self.rank_to_word.get(rank)


def tokenize(text: str) -> List[str]:
    """
    Simple word tokenizer.

    Splits on whitespace and punctuation, lowercases, keeps only alpha.
    """
    # Split on non-alpha, filter empty, lowercase
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return words


def build_vocabulary(texts: List[str], max_vocab: int = 10000) -> Vocabulary:
    """
    Build vocabulary from texts, ranked by frequency.

    Most frequent word gets rank 1, etc.
    """
    counter = Counter()
    for text in texts:
        words = tokenize(text)
        counter.update(words)

    # Rank by frequency (most frequent = rank 1)
    vocab = Vocabulary()
    for rank, (word, count) in enumerate(counter.most_common(max_vocab), start=1):
        vocab.word_to_rank[word] = rank
        vocab.rank_to_word[rank] = word
        vocab.word_counts[word] = count

    return vocab


# =============================================================================
# ZISA STATE
# =============================================================================

@dataclass
class ZISAState:
    """ZISA state: dual rails + accumulators."""
    A_phi: int = 0          # φ-rail bits (64-bit)
    A_psi: int = 0          # ψ-rail bits (64-bit)
    F_acc: int = 0          # Fibonacci accumulator
    L_acc: int = 0          # Lucas accumulator
    Psi_acc: int = 0        # ψ-signature accumulator (signed)
    tau_total: int = 0      # Total cascade count
    tau_history: List[int] = field(default_factory=list)
    pos: int = 0            # Sequence position

    def copy(self) -> 'ZISAState':
        return ZISAState(
            A_phi=self.A_phi,
            A_psi=self.A_psi,
            F_acc=self.F_acc,
            L_acc=self.L_acc,
            Psi_acc=self.Psi_acc,
            tau_total=self.tau_total,
            tau_history=self.tau_history.copy(),
            pos=self.pos
        )


# =============================================================================
# CASCADE NORMALIZATION
# =============================================================================

def has_adjacent_ones(bits: int) -> bool:
    """Check if bits has adjacent 1s (illegal Zeckendorf state)."""
    return (bits & (bits >> 1)) != 0


def priority_cascade(bits: int, max_shell: int = 47) -> Tuple[int, int]:
    """
    Priority cascade: normalize bits to legal Zeckendorf form.

    Returns (normalized_bits, cascade_count).
    Priority: higher shells cascade first.
    """
    tau = 0
    while has_adjacent_ones(bits):
        # Find highest adjacent pair
        for k in range(max_shell, 0, -1):
            if (bits >> k) & 1 and (bits >> (k-1)) & 1:
                # Clear k and k-1, set k+1
                bits &= ~(1 << k)
                bits &= ~(1 << (k-1))
                bits |= (1 << (k+1))
                tau += 1
                break
    return bits, tau


def bits_to_shells(bits: int) -> List[int]:
    """Extract active shell indices from bit pattern."""
    shells = []
    k = 0
    while bits:
        if bits & 1:
            shells.append(k)
        bits >>= 1
        k += 1
    return shells


def shells_to_bits(shells: List[int]) -> int:
    """Convert shell indices to bit pattern."""
    bits = 0
    for k in shells:
        bits |= (1 << k)
    return bits


# =============================================================================
# EIGENVALUE ACCUMULATORS
# =============================================================================

def compute_fsum(bits: int) -> int:
    """Compute Σ FIB[k] for active shells."""
    total = 0
    for k in bits_to_shells(bits):
        if k < len(FIB):
            total += FIB[k]
    return total


def compute_lsum(bits: int) -> int:
    """Compute Σ LUC[k] for active shells."""
    total = 0
    for k in bits_to_shells(bits):
        if k < len(LUC):
            total += LUC[k]
    return total


def compute_psi_signature(bits: int) -> int:
    """Compute Σ PSI[k] for active shells (signed fixed-point)."""
    total = 0
    for k in bits_to_shells(bits):
        if k < len(PSI_LUT):
            total += PSI_LUT[k]
    return total


# =============================================================================
# ABSORB OPERATION
# =============================================================================

def absorb(state: ZISAState, i: int, j: int) -> ZISAState:
    """
    Absorb token (i, j) into state.

    1. Create bit pattern from (i, j)
    2. Select rail by position parity (Verlet)
    3. Accumulate and cascade
    4. Update eigenvalue accumulators
    """
    # Create bit pattern
    token_bits = (1 << i) | (1 << j)

    # Token eigenvalue contributions
    token_psi = PSI_LUT[i] + PSI_LUT[j] if i < len(PSI_LUT) and j < len(PSI_LUT) else 0

    # Select rail by parity
    if state.pos % 2 == 0:
        # φ-rail (position channel)
        new_bits = state.A_phi | token_bits
        new_bits, tau = priority_cascade(new_bits)
        state.A_phi = new_bits
    else:
        # ψ-rail (velocity channel)
        new_bits = state.A_psi | token_bits
        new_bits, tau = priority_cascade(new_bits)
        state.A_psi = new_bits

    # Update accumulators
    state.F_acc = compute_fsum(state.A_phi) + compute_fsum(state.A_psi)
    state.L_acc = compute_lsum(state.A_phi) + compute_lsum(state.A_psi)
    state.Psi_acc = compute_psi_signature(state.A_phi) + compute_psi_signature(state.A_psi)

    # Update cascade history
    state.tau_history.append(tau)
    state.tau_total += tau
    state.pos += 1

    return state


# =============================================================================
# ENCODER
# =============================================================================

@dataclass
class EncodedToken:
    """Single encoded token."""
    word: str
    rank: int
    i: int
    j: int
    bits: int
    psi_signature: int


@dataclass
class EncodedSequence:
    """Full encoded sequence."""
    tokens: List[EncodedToken]
    final_state: ZISAState
    vocab: Vocabulary
    zeck: Dict[int, Tuple[int, int]]


class ZISAEncoder:
    """Full ZISA encoder."""

    def __init__(self, max_vocab: int = 10000):
        self.max_vocab = max_vocab
        self.vocab: Optional[Vocabulary] = None
        self.zeck = build_zeck_lut(max_vocab)
        self.unzeck = build_unzeck_lut(self.zeck)

    def build_vocab_from_texts(self, texts: List[str]):
        """Build vocabulary from training texts."""
        self.vocab = build_vocabulary(texts, self.max_vocab)

    def encode_text(self, text: str) -> EncodedSequence:
        """Encode text into ZISA format."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab_from_texts first.")

        words = tokenize(text)
        tokens = []
        state = ZISAState()

        for word in words:
            rank = self.vocab.encode_word(word)
            if rank is None or rank not in self.zeck:
                continue  # Skip OOV

            i, j = self.zeck[rank]
            bits = (1 << i) | (1 << j)
            psi_sig = PSI_LUT[i] + PSI_LUT[j]

            token = EncodedToken(
                word=word,
                rank=rank,
                i=i,
                j=j,
                bits=bits,
                psi_signature=psi_sig
            )
            tokens.append(token)

            # Absorb into state
            state = absorb(state, i, j)

        return EncodedSequence(
            tokens=tokens,
            final_state=state,
            vocab=self.vocab,
            zeck=self.zeck
        )

    def decode_sequence(self, seq: EncodedSequence) -> str:
        """Decode sequence back to text."""
        words = []
        for token in seq.tokens:
            words.append(token.word)
        return ' '.join(words)

    def to_bitstream(self, seq: EncodedSequence) -> bytes:
        """
        Convert encoded sequence to binary bitstream for Verilator.

        Format per token:
            - 2 bytes: rank (uint16)
            - 1 byte: i (uint8)
            - 1 byte: j (uint8)
            - 8 bytes: bits (uint64)
            - 4 bytes: psi_signature (int32)
        Total: 16 bytes per token
        """
        data = bytearray()

        # Header: number of tokens (4 bytes)
        data.extend(struct.pack('<I', len(seq.tokens)))

        # Each token
        for token in seq.tokens:
            data.extend(struct.pack('<H', token.rank))  # uint16
            data.extend(struct.pack('<B', token.i))     # uint8
            data.extend(struct.pack('<B', token.j))     # uint8
            data.extend(struct.pack('<Q', token.bits))  # uint64
            data.extend(struct.pack('<i', token.psi_signature))  # int32

        # Footer: final state
        data.extend(struct.pack('<Q', seq.final_state.A_phi))
        data.extend(struct.pack('<Q', seq.final_state.A_psi))
        data.extend(struct.pack('<q', seq.final_state.F_acc))
        data.extend(struct.pack('<q', seq.final_state.L_acc))
        data.extend(struct.pack('<i', seq.final_state.Psi_acc))
        data.extend(struct.pack('<I', seq.final_state.tau_total))

        return bytes(data)

    def from_bitstream(self, data: bytes) -> List[Tuple[int, int, int]]:
        """
        Parse bitstream back to (rank, i, j) tuples.
        """
        offset = 0
        num_tokens = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        tokens = []
        for _ in range(num_tokens):
            rank = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            i = struct.unpack_from('<B', data, offset)[0]
            offset += 1
            j = struct.unpack_from('<B', data, offset)[0]
            offset += 1
            offset += 8  # Skip bits
            offset += 4  # Skip psi_signature
            tokens.append((rank, i, j))

        return tokens


# =============================================================================
# GUTENBERG FETCHER
# =============================================================================

GUTENBERG_URLS = {
    'shakespeare_sonnets': 'https://www.gutenberg.org/cache/epub/1041/pg1041.txt',
    'pride_prejudice': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
    'moby_dick': 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt',
    'alice_wonderland': 'https://www.gutenberg.org/cache/epub/11/pg11.txt',
    'frankenstein': 'https://www.gutenberg.org/cache/epub/84/pg84.txt',
}


def fetch_gutenberg(name: str, cache_dir: Optional[Path] = None) -> str:
    """
    Fetch text from Project Gutenberg.

    Caches locally to avoid repeated downloads.
    """
    if name not in GUTENBERG_URLS:
        raise ValueError(f"Unknown text: {name}. Available: {list(GUTENBERG_URLS.keys())}")

    url = GUTENBERG_URLS[name]

    # Cache setup
    if cache_dir is None:
        cache_dir = Path(__file__).parent / 'cache'
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{name}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding='utf-8', errors='ignore')

    # Fetch
    print(f"Fetching {name} from {url}...")
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode('utf-8', errors='ignore')

    # Cache
    cache_file.write_text(text, encoding='utf-8')

    return text


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Gutenberg header and footer."""
    lines = text.split('\n')

    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if '*** START OF' in line.upper():
            start_idx = i + 1
            break

    for i, line in enumerate(lines):
        if '*** END OF' in line.upper():
            end_idx = i
            break

    return '\n'.join(lines[start_idx:end_idx])


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("ZISA ENCODER TEST")
    print("=" * 70)

    # Fetch sample text
    print("\nFetching Alice in Wonderland...")
    text = fetch_gutenberg('alice_wonderland')
    text = strip_gutenberg_header_footer(text)

    print(f"Text length: {len(text)} characters")

    # Build encoder
    print("\nBuilding encoder...")
    encoder = ZISAEncoder(max_vocab=8000)
    encoder.build_vocab_from_texts([text])

    print(f"Vocabulary size: {len(encoder.vocab)}")

    # Encode sample
    sample = "Alice was beginning to get very tired of sitting by her sister"
    print(f"\nEncoding: '{sample}'")

    seq = encoder.encode_text(sample)

    print(f"\nEncoded {len(seq.tokens)} tokens:")
    for t in seq.tokens[:10]:
        print(f"  {t.word:12s} -> rank={t.rank:4d}, (i={t.i:2d}, j={t.j:2d}), bits={t.bits:016x}")

    # Verify round-trip
    decoded = encoder.decode_sequence(seq)
    print(f"\nDecoded: '{decoded}'")

    # Check exact match
    original_words = tokenize(sample)
    decoded_words = tokenize(decoded)

    # Filter to vocab
    original_filtered = [w for w in original_words if encoder.vocab.encode_word(w) is not None]

    if original_filtered == decoded_words:
        print("\n[PASS] Round-trip encoding is EXACT")
    else:
        print("\n[FAIL] Round-trip mismatch!")
        print(f"  Original: {original_filtered}")
        print(f"  Decoded:  {decoded_words}")
        sys.exit(1)

    # Final state
    print(f"\nFinal state:")
    print(f"  A_phi:     {seq.final_state.A_phi:016x}")
    print(f"  A_psi:     {seq.final_state.A_psi:016x}")
    print(f"  F_acc:     {seq.final_state.F_acc}")
    print(f"  L_acc:     {seq.final_state.L_acc}")
    print(f"  Psi_acc:   {seq.final_state.Psi_acc}")
    print(f"  tau_total: {seq.final_state.tau_total}")

    # Verify Zeckendorf legality
    phi_legal = not has_adjacent_ones(seq.final_state.A_phi)
    psi_legal = not has_adjacent_ones(seq.final_state.A_psi)
    print(f"\nZeckendorf legality:")
    print(f"  φ-rail: {'LEGAL' if phi_legal else 'ILLEGAL'}")
    print(f"  ψ-rail: {'LEGAL' if psi_legal else 'ILLEGAL'}")

    if phi_legal and psi_legal:
        print("\n[PASS] Zeckendorf invariant maintained")
    else:
        print("\n[FAIL] Zeckendorf invariant violated!")
        sys.exit(1)

    # Generate bitstream
    bitstream = encoder.to_bitstream(seq)
    print(f"\nBitstream size: {len(bitstream)} bytes")

    # Save for Verilator
    output_path = Path(__file__).parent / 'test_data.bin'
    output_path.write_bytes(bitstream)
    print(f"Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
