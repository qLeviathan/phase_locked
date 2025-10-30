"""
Sequence Validator - Advanced sequence analysis and validation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter


class SequenceValidator:
    """
    Validates integer sequences and discovers mathematical properties
    """

    def __init__(self):
        self.analysis_cache = {}

    def compute_differences(self, seq: List[int], order: int = 1) -> List[int]:
        """Compute n-th order differences"""
        if order == 0:
            return seq

        diffs = [seq[i+1] - seq[i] for i in range(len(seq) - 1)]

        if order > 1:
            return self.compute_differences(diffs, order - 1)

        return diffs

    def find_difference_order(self, seq: List[int], max_order: int = 5) -> Optional[int]:
        """
        Find order of differences that becomes constant
        Returns None if no constant difference found
        """
        for order in range(1, max_order + 1):
            diffs = self.compute_differences(seq, order)

            if len(set(diffs)) == 1:  # All differences are the same
                return order

        return None

    def compute_ratios(self, seq: List[int]) -> List[float]:
        """Compute successive ratios a(n+1)/a(n)"""
        ratios = []
        for i in range(len(seq) - 1):
            if seq[i] != 0:
                ratios.append(seq[i+1] / seq[i])
        return ratios

    def detect_linear_recurrence(self, seq: List[int], max_order: int = 3) -> Optional[Dict]:
        """
        Detect linear recurrence relation
        e.g., a(n) = c₁*a(n-1) + c₂*a(n-2) + ...
        """
        n = len(seq)

        for order in range(1, min(max_order + 1, n // 2)):
            # Try to find coefficients
            # Build system of equations

            if n < 2 * order + 1:
                continue

            # Use least squares to find coefficients
            A = []
            b = []

            for i in range(order, n - 1):
                row = [seq[i - j] for j in range(1, order + 1)]
                A.append(row)
                b.append(seq[i])

            try:
                coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

                # Verify the recurrence
                errors = []
                for i in range(order, n - 1):
                    predicted = sum(coeffs[j] * seq[i - j - 1] for j in range(order))
                    error = abs(predicted - seq[i])
                    errors.append(error)

                avg_error = np.mean(errors)
                max_error = np.max(errors)

                # Check if it's a good fit
                if max_error < 0.1:  # Integer sequence, should be exact
                    return {
                        'order': order,
                        'coefficients': [round(c, 6) for c in coeffs],
                        'avg_error': avg_error,
                        'max_error': max_error,
                        'formula': self._format_recurrence(coeffs)
                    }

            except np.linalg.LinAlgError:
                continue

        return None

    def _format_recurrence(self, coeffs: List[float]) -> str:
        """Format recurrence relation as string"""
        terms = []
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-10:
                continue

            c_str = f"{c:.2f}" if abs(c - round(c)) > 1e-10 else str(int(round(c)))

            if abs(c - 1.0) < 1e-10:
                terms.append(f"a(n-{i+1})")
            elif abs(c + 1.0) < 1e-10:
                terms.append(f"-a(n-{i+1})")
            else:
                terms.append(f"{c_str}*a(n-{i+1})")

        return "a(n) = " + " + ".join(terms)

    def check_exponential_growth(self, seq: List[int]) -> Optional[Dict]:
        """
        Check if sequence has exponential growth
        i.e., a(n) ~ c * r^n
        """
        ratios = self.compute_ratios(seq)

        if not ratios or len(ratios) < 3:
            return None

        # Check if ratios converge to constant
        ratio_diffs = [abs(ratios[i+1] - ratios[i]) for i in range(len(ratios) - 1)]

        if len(ratio_diffs) > 0:
            avg_diff = np.mean(ratio_diffs)
            final_ratio = ratios[-1]

            # If ratios are converging
            if avg_diff < 0.01 * final_ratio:
                return {
                    'ratio': final_ratio,
                    'convergence': 'yes',
                    'formula': f'a(n) ~ c * {final_ratio:.6f}^n'
                }

        return None

    def compute_gcd_sequence(self, seq: List[int]) -> int:
        """Compute GCD of all terms"""
        if not seq:
            return 0

        result = seq[0]
        for x in seq[1:]:
            result = np.gcd(result, x)
            if result == 1:
                break

        return result

    def analyze_sequence(self, seq: List[int], name: str = "unknown") -> Dict:
        """Complete analysis of a sequence"""
        if not seq or len(seq) < 3:
            return {'error': 'Sequence too short for analysis'}

        analysis = {
            'name': name,
            'length': len(seq),
            'first_terms': seq[:10],
            'min': min(seq),
            'max': max(seq),
            'sum': sum(seq),
        }

        # Difference analysis
        const_diff_order = self.find_difference_order(seq)
        if const_diff_order:
            analysis['constant_difference_order'] = const_diff_order
            analysis['type_hint'] = f'polynomial of degree {const_diff_order}'

        # Ratio analysis
        ratios = self.compute_ratios(seq)
        if ratios:
            analysis['ratios'] = ratios[:10]
            analysis['ratio_mean'] = np.mean(ratios)
            analysis['ratio_std'] = np.std(ratios)

        # Check for exponential growth
        exp_growth = self.check_exponential_growth(seq)
        if exp_growth:
            analysis['exponential_growth'] = exp_growth

        # Check for linear recurrence
        recurrence = self.detect_linear_recurrence(seq, max_order=4)
        if recurrence:
            analysis['recurrence'] = recurrence

        # GCD analysis
        gcd = self.compute_gcd_sequence(seq)
        analysis['gcd'] = gcd

        # Statistical properties
        analysis['mean'] = np.mean(seq)
        analysis['std'] = np.std(seq)
        analysis['median'] = np.median(seq)

        return analysis

    def compare_sequences(self, seq1: List[int], seq2: List[int],
                         name1: str = "A", name2: str = "B") -> Dict:
        """Compare two sequences for relationships"""
        min_len = min(len(seq1), len(seq2))
        seq1_trunc = seq1[:min_len]
        seq2_trunc = seq2[:min_len]

        comparison = {
            'name1': name1,
            'name2': name2,
            'length_compared': min_len
        }

        # Check if identical
        comparison['identical'] = (seq1_trunc == seq2_trunc)

        # Check if one is shifted version of other
        for shift in range(1, min(5, min_len // 2)):
            if seq1[shift:shift+min_len-shift] == seq2[:min_len-shift]:
                comparison['shift_relationship'] = f"{name1} is {name2} shifted by {shift}"
                break

        # Check if one is multiple of other
        if seq2_trunc[0] != 0:
            ratios = [seq1_trunc[i] / seq2_trunc[i] for i in range(min_len) if seq2_trunc[i] != 0]
            if len(set([round(r, 6) for r in ratios])) == 1:
                comparison['multiple_relationship'] = f"{name1} = {ratios[0]:.6f} * {name2}"

        # Compute correlation
        if len(seq1_trunc) > 1 and len(seq2_trunc) > 1:
            correlation = np.corrcoef(seq1_trunc, seq2_trunc)[0, 1]
            comparison['correlation'] = correlation

        # Check for sum/difference relationships
        sum_seq = [seq1_trunc[i] + seq2_trunc[i] for i in range(min_len)]
        diff_seq = [seq1_trunc[i] - seq2_trunc[i] for i in range(min_len)]

        comparison['sum'] = sum_seq[:10]
        comparison['difference'] = diff_seq[:10]

        # Analyze sum and difference
        sum_analysis = self.analyze_sequence(sum_seq, f"{name1}+{name2}")
        diff_analysis = self.analyze_sequence(diff_seq, f"{name1}-{name2}")

        if 'recurrence' in sum_analysis:
            comparison['sum_recurrence'] = sum_analysis['recurrence']

        if 'recurrence' in diff_analysis:
            comparison['diff_recurrence'] = diff_analysis['recurrence']

        return comparison

    def validate_oeis_properties(self, seq: List[int], oeis_id: str,
                                  claimed_properties: List[str]) -> Dict:
        """
        Validate claimed OEIS properties
        """
        validation = {
            'oeis_id': oeis_id,
            'sequence': seq[:20],
            'properties_validated': []
        }

        analysis = self.analyze_sequence(seq, oeis_id)

        # Check each claimed property
        for prop in claimed_properties:
            prop_lower = prop.lower()

            result = {'property': prop, 'validated': False, 'notes': ''}

            # Check for Fibonacci recurrence
            if 'a(n) = a(n-1) + a(n-2)' in prop:
                if 'recurrence' in analysis:
                    coeffs = analysis['recurrence']['coefficients']
                    if len(coeffs) == 2 and abs(coeffs[0] - 1) < 0.01 and abs(coeffs[1] - 1) < 0.01:
                        result['validated'] = True
                        result['notes'] = 'Fibonacci recurrence confirmed'

            # Check for exponential growth
            if 'exponential' in prop_lower or 'geometric' in prop_lower:
                if 'exponential_growth' in analysis:
                    result['validated'] = True
                    result['notes'] = f"Ratio = {analysis['exponential_growth']['ratio']:.6f}"

            # Check for polynomial growth
            if 'polynomial' in prop_lower:
                if 'constant_difference_order' in analysis:
                    order = analysis['constant_difference_order']
                    result['validated'] = True
                    result['notes'] = f"Polynomial degree {order}"

            validation['properties_validated'].append(result)

        # Overall validation
        validation['all_validated'] = all(r['validated'] for r in validation['properties_validated'])

        return validation
