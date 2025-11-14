"""
ZORDIC Core Engine - φ-Field Self-Organizing Lattice System
Leviathan AI Corporation

All operations reduce to integer arithmetic in φ-space.
No floating point in critical path.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Import from phi_mamba centralized modules
from phi_mamba.constants import PHI, PSI, SQRT_5 as SQRT5
from phi_mamba.math_core import fibonacci, lucas, zeckendorf_to_indices, fibonacci_sequence


@dataclass
class ZordicNode:
    """A node in the φ-lattice"""
    char: str
    index: int
    value: int
    shells: List[int]
    z_binary: List[int]
    phi: float
    psi: float
    lucas: float
    delta: float
    stability: float
    position: Optional[Tuple[float, float]] = None


class FibonacciCore:
    """Core Fibonacci/Lucas number generation using phi_mamba library"""

    def __init__(self, max_n=50):
        # Use phi_mamba's centralized functions
        self.F = fibonacci_sequence(max_n - 1, include_zero=True)
        self.L = [lucas(i) for i in range(max_n)]

    def zeckendorf_decompose(self, n):
        """
        Decompose integer into unique Fibonacci sum (Zeckendorf representation)

        Property: No consecutive Fibonacci numbers
        Example: 17 = F₇ + F₄ + F₂ = 13 + 3 + 1 → [7, 4, 2]

        Returns indices of Fibonacci numbers (not values)
        """
        return zeckendorf_to_indices(n)

    def to_z_binary(self, shells):
        """Convert shell indices to Z-binary representation"""
        if not shells:
            return [0]

        max_shell = max(shells)
        z_bin = [0] * (max_shell + 1)

        for shell in shells:
            z_bin[shell] = 1

        return z_bin

    def validate_zeckendorf(self, shells):
        """Check no adjacent Fibonacci indices (Zeckendorf constraint)"""
        sorted_shells = sorted(shells)
        for i in range(len(sorted_shells) - 1):
            if sorted_shells[i+1] - sorted_shells[i] == 1:
                return False
        return True

    def phi_power(self, k):
        """Compute φ^k via Binet formula"""
        return (PHI ** k - PSI ** k) / SQRT5

    def psi_power(self, k):
        """Compute ψ^k via Binet formula"""
        return (PHI ** k + PSI ** k)


class ZordicCorpus:
    """Character encoding system using Zeckendorf decomposition"""

    def __init__(self):
        self.fib = FibonacciCore()
        self.encoding = self._build_encoding()

    def _build_encoding(self):
        """Build complete ASCII encoding table"""
        encoding = {}

        # Alphabet + common symbols
        chars = 'abcdefghijklmnopqrstuvwxyz .,!?-:;\n'

        for i, char in enumerate(chars, start=1):
            shells = self.fib.zeckendorf_decompose(i)
            z_bin = self.fib.to_z_binary(shells)

            phi_sum = sum([PHI**k for k in shells]) if shells else 0
            psi_sum = sum([PSI**k for k in shells]) if shells else 0
            lucas = sum([self.fib.L[k] for k in shells if k < len(self.fib.L)]) if shells else 0

            encoding[char] = {
                'value': i,
                'shells': shells,
                'z_binary': z_bin,
                'phi': phi_sum,
                'psi': psi_sum,
                'lucas': lucas,
                'delta': phi_sum - psi_sum
            }

        return encoding

    def encode_char(self, char):
        """Encode single character"""
        return self.encoding.get(char.lower(), self.encoding.get(' ', {
            'value': 0, 'shells': [], 'z_binary': [0],
            'phi': 0, 'psi': 0, 'lucas': 0, 'delta': 0
        }))


class PhiLattice:
    """The self-organizing dual-field φ/ψ lattice"""

    def __init__(self, text: str, temperature: float = 1.0):
        self.text = text
        self.corpus = ZordicCorpus()
        self.nodes: List[ZordicNode] = []
        self.edges = defaultdict(list)
        self.stable_nodes = []
        self.temperature = temperature
        self.cascade_history = []
        self.log_entries = []

    def log(self, message: str, level: str = "info"):
        """Add log entry"""
        self.log_entries.append({
            'message': message,
            'level': level
        })

    def encode(self):
        """Phase 1: Encode text into φ-lattice"""
        self.log("=== PHASE 1: ENCODING ===", "header")

        for i, char in enumerate(self.text):
            data = self.corpus.encode_char(char)

            node = ZordicNode(
                char=char,
                index=i,
                value=data['value'],
                shells=data['shells'],
                z_binary=data['z_binary'],
                phi=data['phi'],
                psi=data['psi'],
                lucas=data['lucas'],
                delta=data['delta'],
                stability=abs(data['delta'])
            )

            self.nodes.append(node)
            self.log(f"[{i}] '{char}' → shells={data['shells']}, φ={data['phi']:.3f}, ψ={data['psi']:.3f}, Δ={data['delta']:.3f}")

        self.log(f"✓ Encoded {len(self.nodes)} characters", "success")

    def analyze_initial_state(self):
        """Analyze the initial field configuration"""
        self.log("=== INITIAL STATE ANALYSIS ===", "header")

        phi_total = sum(n.phi for n in self.nodes)
        psi_total = sum(n.psi for n in self.nodes)
        delta_total = phi_total - psi_total
        avg_stability = np.mean([n.stability for n in self.nodes])

        self.log(f"Total φ-field: {phi_total:.3f}")
        self.log(f"Total ψ-field: {psi_total:.3f}")
        self.log(f"Field delta: {delta_total:.3f}")
        self.log(f"Avg stability: {avg_stability:.3f}")

        stable_count = sum(1 for n in self.nodes if n.stability < 0.5)
        self.log(f"Initially stable nodes: {stable_count}/{len(self.nodes)} ({stable_count/len(self.nodes)*100:.1f}%)")

        return {
            'phi_total': phi_total,
            'psi_total': psi_total,
            'delta_total': delta_total,
            'stability': avg_stability,
            'stable_count': stable_count
        }

    def build_connectivity(self):
        """Phase 2: Build edges via Zeckendorf topology"""
        self.log("=== PHASE 2: CONNECTIVITY ===", "header")

        connections = {'j=0': 0, 'j=1': 0, 'j=2': 0, 'j=3': 0, 'forbidden': 0}

        for i in range(len(self.nodes) - 1):
            n1, n2 = self.nodes[i], self.nodes[i+1]

            # Check Zeckendorf validity
            combined = set(n1.shells) | set(n2.shells)
            is_valid = self.corpus.fib.validate_zeckendorf(list(combined))

            if not is_valid:
                self.log(f"[{i}→{i+1}] '{n1.char}'→'{n2.char}': FORBIDDEN (Zeckendorf violation)", "warning")
                connections['forbidden'] += 1
                continue

            # Determine connection type
            j_type = self._connection_type(n1, n2)
            self.edges[i].append((i+1, j_type))
            connections[j_type] += 1

            self.log(f"[{i}→{i+1}] '{n1.char}'→'{n2.char}': {j_type}")

        self.log(f"✓ Connectivity: {connections}", "success")
        return connections

    def _connection_type(self, n1: ZordicNode, n2: ZordicNode) -> str:
        """Determine j ∈ {0,1,2,3} based on field dynamics"""
        if n1.stability < 0.5 and n2.stability < 0.5:
            return 'j=3'  # Strong resonance
        elif n1.delta > 0 and n2.delta > 0:
            return 'j=1'  # Forward causality
        elif n1.delta < 0 and n2.delta < 0:
            return 'j=2'  # Backward causality
        else:
            return 'j=0'  # Mixed/disconnected

    def compute_interference(self):
        """Phase 3: Compute φ/ψ interference patterns"""
        self.log("=== PHASE 3: INTERFERENCE ===", "header")

        interference_map = []

        for i in range(len(self.nodes) - 1):
            phi1, phi2 = self.nodes[i].phi, self.nodes[i+1].phi
            psi1, psi2 = self.nodes[i].psi, self.nodes[i+1].psi

            # Interference metric
            denom = (np.sqrt(phi1**2 + psi1**2) * np.sqrt(phi2**2 + psi2**2) + 1e-9)
            interference = (phi1*phi2 + psi1*psi2) / denom

            interference_map.append(interference)

            if interference > 0.7:
                self.log(f"[{i},{i+1}] CONSTRUCTIVE: I={interference:.3f}", "success")
            elif interference < 0.3:
                self.log(f"[{i},{i+1}] DESTRUCTIVE: I={interference:.3f}", "warning")

        avg_interference = np.mean(interference_map) if interference_map else 0
        self.log(f"✓ Average interference: {avg_interference:.3f}", "success")

        return interference_map

    def cascade(self, max_iterations=20):
        """Phase 4: Dual cascade to stable configuration"""
        self.log("=== PHASE 4: CASCADE DYNAMICS ===", "header")

        iteration = 0
        cascade_rate = 0.3 / self.temperature

        while iteration < max_iterations:
            changed = False
            iteration_deltas = []

            for i, node in enumerate(self.nodes):
                old_delta = node.delta

                # Neighbor influence
                left_influence = self.nodes[i-1].delta * 0.2 if i > 0 else 0
                right_influence = self.nodes[i+1].delta * 0.2 if i < len(self.nodes) - 1 else 0

                # Cascade toward equilibrium
                target_delta = (left_influence + right_influence) / 2

                # Update with cascade rate
                node.delta = (1 - cascade_rate) * node.delta + cascade_rate * target_delta
                node.phi = node.delta + abs(node.psi)
                node.stability = abs(node.delta)

                delta_change = abs(node.delta - old_delta)
                iteration_deltas.append(delta_change)

                if delta_change > 0.01:
                    changed = True

            # Record history
            self.cascade_history.append({
                'iteration': iteration,
                'max_delta_change': max(iteration_deltas) if iteration_deltas else 0,
                'avg_stability': np.mean([n.stability for n in self.nodes])
            })

            if iteration % 5 == 0 or not changed:
                avg_change = np.mean(iteration_deltas) if iteration_deltas else 0
                avg_stab = np.mean([n.stability for n in self.nodes])
                self.log(f"Iteration {iteration}: Δchange={avg_change:.6f}, stability={avg_stab:.6f}")

            if not changed:
                self.log(f"✓ CONVERGED after {iteration} iterations", "success")
                break

            iteration += 1

        if iteration == max_iterations:
            self.log(f"⚠ Reached max iterations ({max_iterations})", "warning")

    def identify_stable_nodes(self):
        """Identify nodes that have collapsed to stable states"""
        self.stable_nodes = [i for i, node in enumerate(self.nodes) if node.stability < 0.5]
        return self.stable_nodes

    def analyze_regime(self):
        """Determine deterministic vs stochastic regime"""
        self.log("=== REGIME ANALYSIS ===", "header")

        stable_nodes = self.identify_stable_nodes()
        deterministic_ratio = len(stable_nodes) / len(self.nodes) if self.nodes else 0

        self.log(f"Stable nodes: {len(stable_nodes)}/{len(self.nodes)}")
        self.log(f"Unstable nodes: {len(self.nodes) - len(stable_nodes)}/{len(self.nodes)}")
        self.log(f"Deterministic ratio: {deterministic_ratio:.2%}")

        # Classify regime
        if deterministic_ratio > 0.75:
            regime = "DETERMINISTIC"
        elif deterministic_ratio < 0.35:
            regime = "STOCHASTIC"
        else:
            regime = "MIXED (Quantum-like)"

        self.log(f"⟹ REGIME: {regime}", "header")

        return {
            'regime': regime,
            'ratio': deterministic_ratio,
            'stable_count': len(stable_nodes),
            'total_nodes': len(self.nodes)
        }

    def full_analysis(self):
        """Run complete analysis pipeline"""
        self.encode()
        initial_state = self.analyze_initial_state()
        connections = self.build_connectivity()
        interference = self.compute_interference()
        self.cascade()
        regime_info = self.analyze_regime()

        return {
            'initial_state': initial_state,
            'connections': connections,
            'regime': regime_info,
            'final_stability': np.mean([n.stability for n in self.nodes]) if self.nodes else 0
        }
