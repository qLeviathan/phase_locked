"""
ZISA Test Suite: Two-Shell Zeckendorf Integer Sequence Accelerator

Components:
    - encoder.py: Python ZISA encoder with Gutenberg corpus support
    - rtl_sim.py: Pure Python RTL simulation for Binet DAG traversal
    - stress_test.py: Comprehensive test suite for all ZISA components
"""

from .encoder import (
    ZISAEncoder,
    ZISAState,
    EncodedSequence,
    EncodedToken,
    build_zeck_lut,
    build_vocabulary,
    tokenize,
    absorb,
    priority_cascade,
    has_adjacent_ones,
    fetch_gutenberg,
    strip_gutenberg_header_footer,
)

__all__ = [
    'ZISAEncoder',
    'ZISAState',
    'EncodedSequence',
    'EncodedToken',
    'build_zeck_lut',
    'build_vocabulary',
    'tokenize',
    'absorb',
    'priority_cascade',
    'has_adjacent_ones',
    'fetch_gutenberg',
    'strip_gutenberg_header_footer',
]
