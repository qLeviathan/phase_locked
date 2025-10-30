#!/usr/bin/env python3
"""
ZORDIC Mathematical Validation Runner
Comprehensive OEIS-based validation with proof certificate generation

Run this to verify all mathematical properties of the system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from validation.oeis_core import OEISValidator
from validation.theorem_prover import PhiFieldTheoremProver
from validation.sequence_validator import SequenceValidator
from validation.proof_certificate import ProofCertificate


def main():
    """Run complete validation suite"""

    print("\n" + "="*90)
    print(" "*20 + "ZORDIC MATHEMATICAL VALIDATION SUITE")
    print(" "*25 + "OEIS-Based Verification")
    print(" "*24 + "Leviathan AI Corporation")
    print("="*90 + "\n")

    # Create output directory
    output_dir = Path("validation_outputs")
    output_dir.mkdir(exist_ok=True)

    print("Initializing validators...")
    print()

    # Initialize validators
    oeis_validator = OEISValidator()
    theorem_prover = PhiFieldTheoremProver()
    seq_validator = SequenceValidator()
    cert_generator = ProofCertificate()

    # Part 1: OEIS Validation
    print("╔" + "="*88 + "╗")
    print("║" + " "*25 + "PART 1: OEIS SEQUENCE VALIDATION" + " "*30 + "║")
    print("╚" + "="*88 + "╝")
    print()

    oeis_results = oeis_validator.run_full_validation()

    # Part 2: Theorem Proving
    print("\n")
    print("╔" + "="*88 + "╗")
    print("║" + " "*28 + "PART 2: THEOREM PROVING" + " "*36 + "║")
    print("╚" + "="*88 + "╝")
    print()

    theorem_results = theorem_prover.run_all_proofs()

    # Part 3: Additional Sequence Analysis
    print("\n")
    print("╔" + "="*88 + "╗")
    print("║" + " "*24 + "PART 3: ADVANCED SEQUENCE ANALYSIS" + " "*29 + "║")
    print("╚" + "="*88 + "╝")
    print()

    # Analyze Fibonacci sequence
    fib_seq = oeis_results['fibonacci']['computed']
    print("Analyzing Fibonacci Sequence:")
    print("-" * 88)
    fib_analysis = seq_validator.analyze_sequence(fib_seq, "Fibonacci")
    print(f"  Type: {fib_analysis.get('type_hint', 'N/A')}")
    if 'recurrence' in fib_analysis:
        print(f"  Recurrence: {fib_analysis['recurrence']['formula']}")
    if 'exponential_growth' in fib_analysis:
        print(f"  Growth: {fib_analysis['exponential_growth']['formula']}")
    print()

    # Analyze Lucas sequence
    lucas_seq = oeis_results['lucas']['computed']
    print("Analyzing Lucas Sequence:")
    print("-" * 88)
    lucas_analysis = seq_validator.analyze_sequence(lucas_seq, "Lucas")
    print(f"  Type: {lucas_analysis.get('type_hint', 'N/A')}")
    if 'recurrence' in lucas_analysis:
        print(f"  Recurrence: {lucas_analysis['recurrence']['formula']}")
    print()

    # Compare Fibonacci and Lucas
    print("Comparing Fibonacci and Lucas:")
    print("-" * 88)
    comparison = seq_validator.compare_sequences(fib_seq, lucas_seq, "Fibonacci", "Lucas")
    print(f"  Identical: {comparison['identical']}")
    if 'correlation' in comparison:
        print(f"  Correlation: {comparison['correlation']:.6f}")
    if 'shift_relationship' in comparison:
        print(f"  Shift: {comparison['shift_relationship']}")
    print()

    # Part 4: Generate Proof Certificate
    print()
    print("╔" + "="*88 + "╗")
    print("║" + " "*23 + "PART 4: GENERATING PROOF CERTIFICATE" + " "*28 + "║")
    print("╚" + "="*88 + "╝")
    print()

    print("Creating cryptographic proof certificate...")
    certificate = cert_generator.generate_certificate(oeis_results, theorem_results)

    # Export in multiple formats
    print()
    print("Exporting certificate:")
    print("-" * 88)

    # JSON export
    json_path = output_dir / "proof_certificate.json"
    cert_generator.export_json(certificate, str(json_path))
    print(f"  ✓ JSON: {json_path}")

    # Markdown export
    md_content = cert_generator.export_markdown(certificate)
    md_path = output_dir / "proof_certificate.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  ✓ Markdown: {md_path}")

    # LaTeX export
    latex_content = cert_generator.export_latex(certificate)
    latex_path = output_dir / "proof_certificate.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  ✓ LaTeX: {latex_path}")

    # Verify certificate integrity
    print()
    print("Verifying certificate integrity...")
    is_valid = cert_generator.verify_certificate(certificate)
    if is_valid:
        print("  ✓ Certificate cryptographic hash verified")
    else:
        print("  ✗ Certificate verification FAILED")

    # Final Summary
    print()
    print("="*90)
    print(" "*30 + "VALIDATION COMPLETE")
    print("="*90)
    print()

    total = certificate['summary']['total_validations']
    passed = certificate['summary']['passed']
    failed = certificate['summary']['failed']
    success_rate = (passed / total * 100) if total > 0 else 0

    print(f"  Total Validations:  {total}")
    print(f"  Passed:             {passed}")
    print(f"  Failed:             {failed}")
    print(f"  Success Rate:       {success_rate:.1f}%")
    print()

    if failed == 0:
        print("  " + "✓"*44)
        print("  " + " "*5 + "ALL MATHEMATICAL PROPERTIES VERIFIED")
        print("  " + " "*10 + "System is mathematically sound")
        print("  " + "✓"*44)
        print()
        print("  SHA-256 Hash:")
        print(f"  {certificate['cryptographic_hash']}")
        print()
        return 0
    else:
        print("  ✗ Some validations failed")
        print("  Review validation outputs for details")
        print()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
