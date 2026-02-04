# Deterministic AI RTL Implementation

## Patent Pending: Deterministic Artificial Intelligence Apparatus

Target: **Lattice iCE40 UP5K** (UPduino 3.1)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DETERMINISTIC AI APPARATUS                           │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   ZECKENDORF │    │   CASCADE    │    │    CORDIC    │                  │
│  │   ARITHMETIC │    │   OPERATOR   │    │  (sin/cos)   │                  │
│  │              │    │     (κ)      │    │              │                  │
│  │  INT→ZECK   │    │   11→100     │    │  Shift-Add   │                  │
│  │  ZECK→INT   │    │   F_k+F_k+1  │    │    Only      │                  │
│  │   OMEGA     │    │   =F_k+2     │    │              │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             │                                              │
│                    ┌────────▼────────┐                                     │
│                    │  FSM CONTROLLER │                                     │
│                    │   (14 States)   │                                     │
│                    │                 │                                     │
│                    │ INIT→DECOMPOSE  │                                     │
│                    │ →CASCADE        │                                     │
│                    │ →VALIDATE       │                                     │
│                    │ →ENCODE→OMEGA   │                                     │
│                    │ →INDEX→RECALL   │                                     │
│                    │ →SCORE          │                                     │
│                    │ →PHASE_CK       │                                     │
│                    │ →SELECT→ENERGY  │                                     │
│                    │ →EMIT→HALT      │                                     │
│                    └────────┬────────┘                                     │
│                             │                                              │
│                    ┌────────▼────────┐                                     │
│                    │  Ω-INDEXED      │                                     │
│                    │    MEMORY       │                                     │
│                    │   (SPRAM)       │                                     │
│                    │                 │                                     │
│                    │ Content-Address │                                     │
│                    │   via Omega     │                                     │
│                    └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles (Per Patent)

### 1. ZERO Floating Point
All arithmetic is integer-only:
- Fixed-point Q2.30 format (30 fractional bits)
- CORDIC uses only add/subtract/shift
- No IEEE 754, no rounding nondeterminism

### 2. Clock-Defined Immutable State
- Each clock cycle commits state to registers
- Committed state is never modified
- Complete audit trail via state hash chain

### 3. Hyperbolic Lattice Structure
- Zeckendorf representation creates negative curvature
- Exponential divergence of state paths
- Natural compression from phi-space geometry

### 4. Deterministic Inference
- Temperature = 0 (greedy selection)
- Phase-locked transitions only
- 100% reproducible across all runs

---

## Module Hierarchy

```
upduino_top.v                   # Top-level (UPduino 3.1)
├── zeck_fsm_controller.v       # 14-state FSM
├── zeckendorf_arith.v          # Zeckendorf arithmetic unit
├── cascade_operator.v          # κ operator (11→100)
├── cordic_unit.v               # Integer CORDIC (sin/cos/atan2)
└── omega_memory.v              # Ω-indexed SPRAM controller
```

---

## Resource Utilization (Estimated)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs     | ~2500| 5280      | ~47%        |
| Registers| ~800 | 5280      | ~15%        |
| SPRAM    | 1    | 4         | 25%         |
| DPRAM    | 0    | 30        | 0%          |
| DSP      | 0    | 8         | 0%          |

---

## Key Constants (Q2.30 Fixed Point)

```verilog
// Energy decay: 1/φ = 0.618...
localparam DECAY_FACTOR = 32'd663608942;   // (1/φ) × 2^30

// Golden angle: 2π/φ² = 2.399963...
localparam GOLDEN_ANGLE = 32'sd2577110170; // × 2^30

// Phase tolerance: ~0.094 rad = 5.4°
localparam PHASE_TOLERANCE = 32'd67108864; // 2^26

// Energy threshold: ~0.004
localparam ENERGY_THRESHOLD = 32'd4194304; // 2^22

// 2π
localparam TWO_PI_SCALED = 32'd6746518852; // 2π × 2^30
```

---

## CORDIC Accuracy

| Function | Error (vs IEEE 754) |
|----------|---------------------|
| sin(θ)   | < 0.0001 (16 iter)  |
| cos(θ)   | < 0.0001 (16 iter)  |
| atan2    | < 0.0001 (16 iter)  |

Each CORDIC iteration: 1 add + 1 sub + 2 shifts = 4 operations.
16 iterations = 64 operations total (no multiplier needed).

---

## Fibonacci Lookup Table

Precomputed F[0..46] covers integers up to 1,836,311,903 (< 2^31).

```verilog
FIB_ROM[0]  = 0;           FIB_ROM[10] = 55;
FIB_ROM[1]  = 1;           FIB_ROM[20] = 6765;
FIB_ROM[2]  = 1;           FIB_ROM[30] = 832040;
FIB_ROM[3]  = 2;           FIB_ROM[40] = 102334155;
FIB_ROM[4]  = 3;           FIB_ROM[46] = 1836311903;
FIB_ROM[5]  = 5;
```

---

## Build Instructions

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt install yosys nextpnr-ice40 fpga-icestorm

# macOS (Homebrew)
brew install yosys nextpnr icestorm
```

### Build
```bash
cd rtl/syn
make           # Build bitstream
make prog      # Program UPduino
make sim       # Run simulation
make report    # Resource utilization
```

### Simulation Only (no FPGA tools)
```bash
# Install iverilog
sudo apt install iverilog gtkwave

cd rtl/syn
make sim       # Runs testbench
make waves     # View waveforms
```

---

## UART Protocol

Baud: 115200, 8N1

| Command | Byte | Description |
|---------|------|-------------|
| NOP     | 0x00 | No operation |
| RESET   | 0x01 | Reset FSM |
| START   | 0x02 | Start inference |
| SEED    | 0x03 | Seed token (+ 4 bytes) |
| STATUS  | 0x04 | Get status |
| READ    | 0x05 | Read output |

---

## LED Status

| LED | State | Meaning |
|-----|-------|---------|
| Red | On | Halted (energy exhausted) |
| Green | Blink | Emitting token |
| Blue | On | Ready |

---

## Mathematical Foundations

### Zeckendorf's Theorem
Every positive integer has a unique representation as a sum of
non-consecutive Fibonacci numbers (OEIS A003714).

### Cascade Operator κ
```
κ(11) = 100    (F_k + F_{k+1} = F_{k+2})
```

### Cassini Identity (Checksum)
```
F[n+1] · F[n-1] - F[n]² = (-1)ⁿ
```

### Berry Phase Coherence
```
γ = Δθ × (1 + overlap) + 2π × Δpos/100
Phase-locked: γ ≡ 0 (mod 2π)
```

### Energy Decay
```
E(n) = E₀ / φⁿ → 0 (natural termination)
```

---

## License

Patent Pending. All rights reserved.

---

## References

- OEIS A003714: Fibbinary numbers
- OEIS A000045: Fibonacci sequence
- OEIS A000032: Lucas sequence
- CORDIC: Volder, 1959
- Zeckendorf's theorem: 1939
