#!/bin/bash
# Zeckendorf-CORDIC Ablation Study - Quick Runner
# Click and go: Just run ./run_ablation.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================="
echo -e "Zeckendorf-CORDIC Ablation Study"
echo -e "Integer-Only Compression"
echo -e "==================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

# Check numpy
if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${RED}Error: NumPy not installed${NC}"
    echo "Install with: pip install numpy"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Parse arguments
MODE="${1:-default}"

case "$MODE" in
    quick|q)
        echo "Running QUICK mode (2 layers, 95% sparsity)..."
        python3 main.py --quick
        ;;
    full|f)
        echo "Running FULL mode (32 layers, 99.5% sparsity)..."
        echo "⚠️  This will take 5-10 minutes!"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 main.py --full
        else
            echo "Cancelled"
            exit 0
        fi
        ;;
    extreme|e)
        echo "Running EXTREME mode (99.7% sparsity, 2-bit)..."
        python3 main.py --extreme
        ;;
    agentic|a)
        echo "Running AGENTIC mode (full ZORDIC lattice)..."
        echo "🧬 Dual Zeckendorf + φ-Cascade + Berry phase"
        python3 main.py --agentic
        ;;
    agentic-quick|aq)
        echo "Running AGENTIC QUICK mode..."
        python3 main.py --agentic --quick
        ;;
    clean|c)
        echo "Cleaning cache and running default..."
        python3 main.py --clean
        ;;
    help|h|-h|--help)
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  (none)         Default: 4 layers, 99.5% sparsity, 4-bit"
        echo "  quick          Fast: 2 layers, 95% sparsity"
        echo "  full           Complete: 32 layers, 99.5% sparsity (slow!)"
        echo "  extreme        Maximum: 99.7% sparsity, 2-bit"
        echo "  agentic        Full ZORDIC lattice complexity"
        echo "  agentic-quick  Agentic mode with 2 layers"
        echo "  clean          Clear cache and rerun"
        echo "  help           Show this message"
        echo ""
        echo "Advanced:"
        echo "  python3 main.py --layers 8 --sparsity 0.99 --bits 8"
        echo "  python3 main.py --agentic --experiment my-test"
        exit 0
        ;;
    *)
        echo "Running DEFAULT mode (4 layers, 99.5% sparsity, 4-bit)..."
        python3 main.py
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Ablation study complete!${NC}"
    echo ""
    echo "Results show:"
    echo "  • Memory reduction via compression"
    echo "  • Compute operations comparison"
    echo "  • Integer-only validation"
    echo "  • Performance trade-offs"
else
    echo ""
    echo -e "${RED}❌ Study failed with code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
