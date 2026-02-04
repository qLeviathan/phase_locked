# ============================================================================
# SYNTHESIS MAKEFILE FOR UPDUINO 3.1
# ============================================================================
# Deterministic AI Apparatus - Patent Pending
# Target: Lattice iCE40 UP5K (UPduino 3.1)
#
# Requirements:
#   - Yosys (synthesis)
#   - nextpnr-ice40 (place and route)
#   - icepack (bitstream generation)
#   - iceprog (programming)
#
# Install on Ubuntu/Debian:
#   sudo apt install yosys nextpnr-ice40 fpga-icestorm
#
# Usage:
#   make           - Build bitstream
#   make prog      - Program UPduino
#   make sim       - Run simulation
#   make clean     - Clean build files
#   make report    - Show resource usage
# ============================================================================

# Project settings
PROJECT = zeck_ai
TOP = upduino_top
DEVICE = up5k
PACKAGE = sg48

# Directories
SRC_DIR = ../src
TB_DIR = ../tb
SIM_DIR = ../sim
BUILD_DIR = build

# Source files
SRCS = \
    $(SRC_DIR)/upduino_top.v \
    $(SRC_DIR)/zeck_fsm_controller.v \
    $(SRC_DIR)/zeckendorf_arith.v \
    $(SRC_DIR)/cascade_operator.v \
    $(SRC_DIR)/cordic_unit.v \
    $(SRC_DIR)/omega_memory.v

# Constraints
PCF = upduino3.pcf

# Tools
YOSYS = yosys
NEXTPNR = nextpnr-ice40
ICEPACK = icepack
ICEPROG = iceprog
IVERILOG = iverilog
VVP = vvp

# Flags
YOSYS_FLAGS = -q
NEXTPNR_FLAGS = --$(DEVICE) --package $(PACKAGE) --freq 48 --opt-timing
IVERILOG_FLAGS = -g2012 -Wall

# Output files
JSON = $(BUILD_DIR)/$(PROJECT).json
ASC = $(BUILD_DIR)/$(PROJECT).asc
BIN = $(BUILD_DIR)/$(PROJECT).bin
TIMING = $(BUILD_DIR)/$(PROJECT)_timing.txt
UTIL = $(BUILD_DIR)/$(PROJECT)_util.txt

# ============================================================================
# TARGETS
# ============================================================================

.PHONY: all clean prog sim report

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Synthesis with Yosys
$(JSON): $(SRCS) | $(BUILD_DIR)
	@echo "=== SYNTHESIS (Yosys) ==="
	$(YOSYS) $(YOSYS_FLAGS) -p "\
		read_verilog -DSYNTHESIS $(SRCS); \
		synth_ice40 -top $(TOP) -json $@" \
		2>&1 | tee $(BUILD_DIR)/synth.log
	@echo "Synthesis complete: $@"

# Place and Route with nextpnr
$(ASC): $(JSON) $(PCF)
	@echo "=== PLACE & ROUTE (nextpnr-ice40) ==="
	$(NEXTPNR) $(NEXTPNR_FLAGS) \
		--json $(JSON) \
		--pcf $(PCF) \
		--asc $@ \
		--report $(UTIL) \
		2>&1 | tee $(BUILD_DIR)/pnr.log
	@echo "Place & Route complete: $@"

# Bitstream generation
$(BIN): $(ASC)
	@echo "=== BITSTREAM GENERATION (icepack) ==="
	$(ICEPACK) $(ASC) $@
	@echo "Bitstream complete: $@"
	@ls -lh $@

# Program FPGA
prog: $(BIN)
	@echo "=== PROGRAMMING FPGA ==="
	$(ICEPROG) $<

# Simulation
sim: $(TB_DIR)/tb_zeck_top.v $(SRCS)
	@echo "=== SIMULATION ==="
	mkdir -p $(SIM_DIR)
	$(IVERILOG) $(IVERILOG_FLAGS) -o $(SIM_DIR)/tb_zeck_top.vvp \
		-I$(SRC_DIR) $(TB_DIR)/tb_zeck_top.v $(SRCS)
	$(VVP) $(SIM_DIR)/tb_zeck_top.vvp

# Resource report
report: $(ASC)
	@echo "=== RESOURCE UTILIZATION ==="
	@cat $(UTIL)
	@echo ""
	@echo "=== TIMING ANALYSIS ==="
	@grep -E "(Max frequency|Critical path)" $(BUILD_DIR)/pnr.log || true

# Clean
clean:
	rm -rf $(BUILD_DIR) $(SIM_DIR)/*.vvp $(SIM_DIR)/*.vcd

# ============================================================================
# ADDITIONAL TARGETS
# ============================================================================

# Timing analysis only
timing: $(ASC)
	@echo "=== TIMING ANALYSIS ==="
	icetime -d $(DEVICE) -c 48 $(ASC) 2>&1 | tee $(TIMING)

# View waveforms (requires GTKWave)
waves: sim
	gtkwave $(SIM_DIR)/tb_zeck_top.vcd &

# Lint check
lint: $(SRCS)
	@echo "=== LINT CHECK ==="
	verilator --lint-only -Wall $(SRCS) --top-module $(TOP) 2>&1 || true

# Format check (if verible is installed)
format:
	@echo "=== FORMAT CHECK ==="
	verible-verilog-format --inplace $(SRCS) 2>/dev/null || echo "verible not installed"

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "Deterministic AI FPGA Build System"
	@echo "=================================="
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build bitstream (default)"
	@echo "  prog     - Program UPduino 3.1"
	@echo "  sim      - Run simulation"
	@echo "  report   - Show resource utilization"
	@echo "  timing   - Detailed timing analysis"
	@echo "  waves    - View waveforms in GTKWave"
	@echo "  lint     - Run lint checks"
	@echo "  clean    - Remove build files"
	@echo ""
	@echo "Device: Lattice iCE40 UP5K"
	@echo "Package: SG48"
	@echo "Clock: 48MHz (internal oscillator)"
