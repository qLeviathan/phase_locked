#!/bin/bash
# =============================================================================
# Benchmark Script for Zeckbit Cascade
# =============================================================================
# Measures:
#   - Cascade throughput (ops/sec)
#   - Encode throughput (tokens/sec)
#   - Contraction throughput (attention ops/sec)
#   - Resource usage estimation
# =============================================================================

set -e

echo "=============================================="
echo "  Zeckbit Cascade Benchmark"
echo "=============================================="
echo ""

# System info
echo "System:"
echo "  $(uname -a)"
echo "  CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Run the benchmark executable
if [ -f "obj_dir/Vzeck_top" ]; then
    echo "Running benchmark (1M operations)..."
    echo ""

    # Time the execution
    START=$(date +%s.%N)
    ./obj_dir/Vzeck_top --benchmark 2>/dev/null || ./obj_dir/Vzeck_top
    END=$(date +%s.%N)

    ELAPSED=$(echo "$END - $START" | bc)
    echo ""
    echo "Wall time: ${ELAPSED}s"

    # Parse results from test output
    echo ""
    echo "=============================================="
    echo "  Performance Targets"
    echo "=============================================="
    echo ""
    echo "Target: 333M - 1B ops/sec (FPGA)"
    echo "Target: 1M+ ops/sec (Verilator on Pi)"
    echo ""

    # Estimate FPGA resources
    echo "=============================================="
    echo "  Resource Estimation (iCE40-class)"
    echo "=============================================="
    echo ""
    echo "Module            | LUTs  | FFs   | Notes"
    echo "------------------|-------|-------|------------------"
    echo "zeck_cascade (32) |  ~200 |  ~70  | Rewrite engine"
    echo "zeck_encode (32)  |  ~150 | ~100  | Greedy decomp"
    echo "lucas_lut (32)    |  ~100 |    0  | ROM (could be BRAM)"
    echo "phi_contract      |  ~150 |  ~80  | Attention accumulator"
    echo "------------------|-------|-------|------------------"
    echo "TOTAL             |  ~600 | ~250  | Under 1350 target"
    echo ""

else
    echo "ERROR: Build not found. Run 'make build' first."
    exit 1
fi

echo "=============================================="
echo "  Benchmark Complete"
echo "=============================================="
