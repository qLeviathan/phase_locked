# =============================================================================
# Makefile for Zeckbit Cascade - Verilator Build
# =============================================================================
# Target: Raspberry Pi 3B (ARM Cortex-A53)
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
                  -Wno-UNDRIVEN

TOP_MODULE = zeck_top
RTL_FILES = zeck_top.v zeck_cascade.v zeck_encode.v zeck_merge.v
TB_FILE = sim/tb_zeck.cpp

OBJ_DIR = obj_dir
EXECUTABLE = $(OBJ_DIR)/V$(TOP_MODULE)

.PHONY: all build run clean wave

all: build

build: $(EXECUTABLE)

$(EXECUTABLE): $(RTL_FILES) $(TB_FILE)
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		$(RTL_FILES) \
		$(TB_FILE)

run: $(EXECUTABLE)
	@echo "=== Running Zeckbit Cascade Demo ==="
	./$(EXECUTABLE)

# Run with waveform capture
wave: $(EXECUTABLE)
	./$(EXECUTABLE) +trace
	@echo "Waveform saved to trace.vcd"

clean:
	rm -rf $(OBJ_DIR) trace.vcd

# Quick test for syntax check
lint:
	$(VERILATOR) --lint-only -Wall $(RTL_FILES)

# For Raspberry Pi: check Verilator is installed
check-deps:
	@which verilator > /dev/null || (echo "Install: sudo apt install verilator" && exit 1)
	@echo "Verilator found: $$(verilator --version)"

# Print Fibonacci LUT for verification
print-fibs:
	@echo "Fibonacci sequence (F_2 to F_33):"
	@python3 -c "f=[1,2]; [f.append(f[-1]+f[-2]) for _ in range(30)]; print('\n'.join(f'F_{i+2} = {v}' for i,v in enumerate(f)))"
