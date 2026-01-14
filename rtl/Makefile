# Makefile for ZISA Core Verilator Testbench
#
# Usage:
#   make        - Build and run test
#   make build  - Build only
#   make run    - Run test
#   make clean  - Clean build artifacts

VERILATOR ?= verilator
PYTHON ?= python3

# Verilator flags
VFLAGS = --cc --exe --build
VFLAGS += -Wall
VFLAGS += --trace
VFLAGS += -CFLAGS "-std=c++14 -O2"

# Source files
VSRC = zisa_core.v
CPPSRC = tb_zisa_core.cpp

# Output
TARGET = obj_dir/Vzisa_core

# Test data
TESTDATA = ../tests/zisa/test_data.bin

.PHONY: all build run test clean generate_data

all: test

build: $(TARGET)

$(TARGET): $(VSRC) $(CPPSRC)
	$(VERILATOR) $(VFLAGS) $(VSRC) $(CPPSRC)

generate_data:
	cd .. && $(PYTHON) tests/zisa/encoder.py

run: $(TARGET)
	./$(TARGET) $(TESTDATA)

test: generate_data build run

clean:
	rm -rf obj_dir
	rm -f *.vcd

# Lint only (no build)
lint:
	$(VERILATOR) --lint-only -Wall $(VSRC)

# Generate VCD trace
trace: $(TARGET)
	./$(TARGET) $(TESTDATA) +trace
