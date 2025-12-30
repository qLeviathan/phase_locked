# =============================================================================
# Makefile for φ-Subscript Calculus Primitives
# =============================================================================
# Target: Verilator on Raspberry Pi 3B / Kano
#
# Build: make
# Run:   make run
# Clean: make clean
# =============================================================================

VERILATOR = verilator
VERILATOR_FLAGS = --cc --exe --build -j 0 \
                  --trace \
                  -Wall \
                  -Wno-UNUSED \
                  -Wno-UNDRIVEN \
                  -Wno-WIDTH \
                  -Wno-CASEINCOMPLETE

TOP_MODULE = zeck_top
RTL_FILES = zeck_top.v \
            zeck_cascade.v \
            zeck_encode.v \
            zeck_merge.v \
            lucas_lut.v \
            phi_contract.v \
            phi_context.v \
            phi_decode.v \
            phi_infer.v

TB_FILE = sim/tb_zeck.cpp

OBJ_DIR = obj_dir
EXECUTABLE = $(OBJ_DIR)/V$(TOP_MODULE)

.PHONY: all build run clean wave lint check-deps benchmark

all: build

build: $(EXECUTABLE)

$(EXECUTABLE): $(RTL_FILES) $(TB_FILE)
	@echo "=== Building φ-Subscript Calculus Primitives ==="
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		$(RTL_FILES) \
		$(TB_FILE)

run: $(EXECUTABLE)
	@echo ""
	@echo "=== Running Zeckbit Cascade Tests ==="
	@echo ""
	./$(EXECUTABLE)

# Run with waveform capture
wave: $(EXECUTABLE)
	./$(EXECUTABLE) +trace
	@echo "Waveform saved to trace.vcd"
	@echo "View with: gtkwave trace.vcd"

clean:
	rm -rf $(OBJ_DIR) trace.vcd results/

# Quick syntax check
lint:
	$(VERILATOR) --lint-only -Wall $(RTL_FILES)

# Check Verilator is installed
check-deps:
	@which verilator > /dev/null || (echo "Install: sudo apt install verilator" && exit 1)
	@echo "Verilator: $$(verilator --version)"
	@echo "Ready to build."

# Run benchmark
benchmark: $(EXECUTABLE)
	@chmod +x benchmark.sh
	./benchmark.sh

# Quick test for specific module
test-cascade: $(EXECUTABLE)
	./$(EXECUTABLE) --test cascade

# Setup script
setup:
	@chmod +x setup_pi.sh benchmark.sh
	./setup_pi.sh

# Print Fibonacci/Lucas tables for reference
print-tables:
	@echo "Fibonacci (F_2 to F_20):"
	@python3 -c "f=[1,1]; [f.append(f[-1]+f[-2]) for _ in range(18)]; print(' '.join(f'F_{i+2}={v}' for i,v in enumerate(f)))"
	@echo ""
	@echo "Lucas (L_0 to L_15):"
	@python3 -c "l=[2,1]; [l.append(l[-1]+l[-2]) for _ in range(14)]; print(' '.join(f'L_{i}={v}' for i,v in enumerate(l)))"

# Resource estimation
estimate:
	@echo ""
	@echo "=== Resource Estimation (iCE40-class FPGA) ==="
	@echo ""
	@echo "Module            | LUTs  | FFs   | Notes"
	@echo "------------------|-------|-------|------------------"
	@echo "zeck_cascade (32) |  ~200 |  ~70  | Rewrite engine R1+R2"
	@echo "zeck_encode (32)  |  ~150 | ~100  | Greedy Zeckendorf"
	@echo "lucas_lut (32)    |  ~100 |    0  | ROM table"
	@echo "phi_contract      |  ~150 |  ~80  | Attention accumulator"
	@echo "zeck_merge        |   ~50 |  ~10  | Superposition + cascade"
	@echo "------------------|-------|-------|------------------"
	@echo "TOTAL             |  ~650 | ~260  | Under 1350 target"
	@echo ""
