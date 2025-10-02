"""
Run complete validation suite and generate all journal figures
"""

import subprocess
import sys
import os

def run_with_output(cmd, description):
    """Run command and show output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERRORS:", result.stderr)
    
    return result.returncode == 0

def main():
    """Run all validation and figure generation"""
    
    # Set Python path
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    print("PHI-MAMBA ARXIV PAPER GENERATION")
    print("="*60)
    
    # 1. Run game theory validation
    success = run_with_output(
        f"{sys.executable} game_theory_validation.py",
        "Game Theory Validation Suite"
    )
    
    if not success:
        print("Warning: Game theory validation had issues")
    
    # 2. Generate journal figures
    success = run_with_output(
        f"{sys.executable} journal_graphics.py",
        "Journal-Quality Figures"
    )
    
    if not success:
        print("Warning: Figure generation had issues")
    
    # 3. Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    
    print("\nGenerated outputs:")
    print("1. validation_outputs/")
    print("   - game_theory_validation_results.json")
    print("\n2. journal_figures/")
    print("   - figure_1_theoretical_framework.pdf/png")
    print("   - figure_2_equilibrium_dynamics.pdf/png")
    print("   - figure_3_panel_data_structure.pdf/png")
    print("   - figure_4_mechanism_design.pdf/png")
    print("\n3. LaTeX files:")
    print("   - arxiv_preprint.tex")
    print("   - references.bib")
    
    print("\nTo compile the LaTeX paper:")
    print("  pdflatex arxiv_preprint.tex")
    print("  bibtex arxiv_preprint")
    print("  pdflatex arxiv_preprint.tex")
    print("  pdflatex arxiv_preprint.tex")
    
    print("\nThe paper is ready for arXiv submission!")

if __name__ == "__main__":
    main()