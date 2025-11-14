"""
AI Engine Bridge - Connects real ZORDIC engine to HUD
Provides live data from actual φ-field calculations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../zordic_desktop/src'))

from zordic_core import PhiLattice
import threading
import time
import queue


class AIEngineBridge:
    """
    Bridges ZORDIC AI engine to sensorium HUD
    Provides real-time data feed from actual computations
    """

    def __init__(self):
        self.lattice = None
        self.data_queue = queue.Queue(maxsize=1000)
        self.processing = False
        self.current_text = ""

    def process_text_stream(self, text_stream):
        """
        Process streaming text input and feed to HUD
        Simulates real-time AI processing
        """
        self.processing = True
        self.lattice = PhiLattice(text_stream, temperature=1.0)

        # Encode in real-time
        for i, char in enumerate(text_stream):
            if not self.processing:
                break

            # Get character encoding
            data = self.lattice.corpus.encode_char(char)

            # Create event
            event = {
                'type': 'token',
                'char': char,
                'index': i,
                'shells': data['shells'],
                'phi': data['phi'],
                'psi': data['psi'],
                'delta': data['delta'],
                'stable': abs(data['delta']) < 0.5,
                'timestamp': time.time()
            }

            self.data_queue.put(event)
            time.sleep(0.05)  # Simulate processing time

        # Full analysis
        if self.processing:
            results = self.lattice.full_analysis()

            # Send regime update
            regime_event = {
                'type': 'regime',
                'regime': results['regime']['regime'],
                'ratio': results['regime']['ratio'],
                'stable_count': results['regime']['stable_count'],
                'total_nodes': results['regime']['total_nodes'],
                'timestamp': time.time()
            }

            self.data_queue.put(regime_event)

        self.processing = False

    def get_field_state(self):
        """Get current φ/ψ field state"""
        if self.lattice and self.lattice.nodes:
            phi_sum = sum(n.phi for n in self.lattice.nodes)
            psi_sum = sum(n.psi for n in self.lattice.nodes)
            return {
                'phi': phi_sum,
                'psi': psi_sum,
                'delta': abs(phi_sum - psi_sum),
                'node_count': len(self.lattice.nodes)
            }
        return None

    def get_metrics(self):
        """Get system metrics"""
        if self.lattice:
            return {
                'nodes': len(self.lattice.nodes),
                'stable_nodes': len(self.lattice.stable_nodes),
                'connections': len(self.lattice.edges),
                'cascade_iterations': len(self.lattice.cascade_history)
            }
        return {}

    def stop(self):
        """Stop processing"""
        self.processing = False
