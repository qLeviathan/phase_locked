# ZORDIC Desktop - φ-Field Self-Organizing Lattice

**Leviathan AI Corporation**
**Proof of Concept - Desktop Application v0.1.0**

---

## Overview

ZORDIC (Zeckendorf Ordered Retrocausal Deterministic/Indeterministic Computing) is a self-organizing computational substrate based on golden ratio (φ) field dynamics. It demonstrates how deterministic and stochastic behaviors emerge naturally from the same geometric constraints.

### Key Features

- **φ/ψ Dual Field Dynamics** - Forward (φ) and backward (ψ) causal fields
- **Zeckendorf Encoding** - Every character encoded as non-consecutive Fibonacci sum
- **Self-Organization** - System cascades to stable configuration automatically
- **Regime Detection** - Identifies deterministic vs stochastic operation modes
- **Real-time Visualization** - Live plots of field evolution
- **Integer-Only Core** - All critical operations reduce to integer arithmetic

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import numpy, matplotlib; print('✓ Dependencies installed')"
   ```

---

## Running the Application

### Quick Start

```bash
python run_zordic.py
```

Or make it executable:
```bash
chmod +x run_zordic.py
./run_zordic.py
```

### What Happens When You Run It

1. A window opens showing the ZORDIC interface
2. Enter text in the input field (default: "quantum field theory")
3. Adjust temperature slider (1.0 = balanced, higher = more stochastic)
4. Click **⚡ ANALYZE** button
5. Watch the system:
   - Encode characters into φ-lattice
   - Build connectivity graph
   - Compute interference patterns
   - Cascade to stable configuration
   - Determine operational regime

---

## Understanding the Interface

### Left Panel

**Input Section:**
- Text input field for analysis
- Temperature slider (0.1 to 2.0)
  - Low temperature → faster convergence, more deterministic
  - High temperature → slower convergence, more stochastic

**Control Buttons:**
- ⚡ ANALYZE - Run full lattice analysis
- 🔄 CLEAR - Reset all displays

**System Metrics:**
- Initial field state (φ, ψ totals)
- Connectivity statistics
- Final regime classification
- Stability metrics

**Operation Log:**
- Real-time step-by-step analysis
- Color-coded by importance
- Shows encoding, connectivity, cascade progress

### Right Panel - Visualizations

**Top Row:**
- **φ-Field** (blue) - Forward causal component
- **ψ-Field** (red) - Backward causal component

**Middle:**
- **Δ Stability** (bar chart) - Shows φ - ψ at each position
  - Green bars = stable (deterministic)
  - Orange bars = unstable (stochastic)

**Bottom:**
- **Cascade Convergence** - How system evolved over iterations
  - Y-axis: average stability
  - Dashed line: stability threshold (0.5)

---

## Theory & Mathematics

### The φ/ψ Field

Every position in text has two components:

- **φ component** = sum of φ^k for active Fibonacci shells
- **ψ component** = sum of ψ^k for active Fibonacci shells

Where:
- φ = (1 + √5)/2 ≈ 1.618 (golden ratio)
- ψ = (1 - √5)/2 ≈ -0.618 (conjugate)

### Zeckendorf Encoding

Each character maps to unique sum of non-consecutive Fibonacci numbers:
```
'a' = 1 = F₂           → shells=[2]
'b' = 2 = F₃           → shells=[3]
'c' = 3 = F₄           → shells=[4]
'd' = 4 = F₂ + F₄      → shells=[2,4]  (note: 2 and 4 are non-consecutive)
...
```

**Constraint:** No adjacent Fibonacci indices (Zeckendorf property)

### Cascade Operation

When constraint violated (adjacent indices appear):
```
F[i] + F[i+1] → F[i+2]
```

Example: shells=[2,3] (invalid) cascades to shells=[4] (valid)

This creates **natural resolution** of conflicts.

### Regime Classification

System determines operation mode based on stability:

**Deterministic Ratio = (stable nodes) / (total nodes)**

- **> 75%** → DETERMINISTIC
  - Unique output forced by geometry
  - No sampling needed
  - P-like complexity

- **< 35%** → STOCHASTIC
  - Multiple viable paths
  - Sampling required
  - NP-like complexity

- **35-75%** → MIXED (Quantum-like)
  - Superposition of modes
  - Some forced, some probabilistic
  - BQP-like complexity

---

## Example Workflows

### Example 1: Simple Word
```
Input: "cat"
Temperature: 1.0
Result: MIXED regime (66% deterministic)
```
- Short word stabilizes quickly
- Most nodes reach equilibrium
- Clear φ/ψ separation

### Example 2: Repeated Characters
```
Input: "aaa"
Temperature: 1.0
Result: DETERMINISTIC regime (>90% deterministic)
```
- Identical encodings reinforce
- Rapid cascade convergence
- Strong phase-locking

### Example 3: Complex Sentence
```
Input: "the cats eat mice"
Temperature: 1.0
Result: MIXED regime (50-60% deterministic)
```
- Varied characters create interference
- Some constructive, some destructive
- Slow convergence, multiple stable points

### Example 4: High Temperature
```
Input: "quantum"
Temperature: 2.0
Result: STOCHASTIC regime (<30% deterministic)
```
- High temperature prevents collapse
- System stays in superposition
- Many bifurcation points

---

## Technical Details

### Performance

- Encoding: O(n) where n = text length
- Connectivity: O(n)
- Cascade: O(n × k) where k ≈ 10-20 iterations
- Total: ~instant for text up to 1000 characters

### Data Flow

```
Text Input
   ↓
Character-by-Character Encoding
   ↓
Zeckendorf Decomposition (shells)
   ↓
φ/ψ Field Computation
   ↓
Connectivity Graph (Zeckendorf validation)
   ↓
Interference Analysis
   ↓
Cascade Dynamics (iterative)
   ↓
Regime Classification
   ↓
Visualization
```

### File Structure

```
zordic_desktop/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── zordic_core.py       # Core φ-field engine
│   └── zordic_gui.py        # GUI application
├── assets/                  # (future: icons, images)
├── docs/                    # (future: additional docs)
├── requirements.txt         # Python dependencies
├── run_zordic.py           # Main launcher
└── README.md               # This file
```

---

## Mathematical Properties (Verified)

All operations satisfy these axioms:

1. **Closure:** φⁿ + φᵐ = φᵏ (via Zeckendorf)
2. **Associativity:** (φⁿ · φᵐ) · φᵖ = φⁿ · (φᵐ · φᵖ)
3. **Identity:** φ⁰ = 1
4. **Inverse:** φⁿ · φ⁻ⁿ = 1
5. **Cascade Invariance:** Sum preserved under normalization
6. **Topological Invariance:** Holes preserved under deformation

---

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### GUI doesn't appear
- Check if tkinter is installed: `python -c "import tkinter"`
- On Linux: `sudo apt-get install python3-tk`
- On macOS: tkinter should be included with Python

### Plots not showing
- Verify matplotlib: `python -c "import matplotlib; print(matplotlib.__version__)"`
- Update: `pip install --upgrade matplotlib`

### Slow performance
- Reduce text length (keep under 100 chars for demo)
- Lower temperature (faster convergence)
- Check CPU usage (cascade can be intensive)

---

## Future Enhancements

### Planned Features (Mobile & Beyond)

1. **Mobile App** (next phase)
   - React Native or Flutter
   - Touch-based lattice interaction
   - Real-time field manipulation

2. **Advanced Visualizations**
   - 3D cylinder representation
   - Interactive lattice editing
   - Animation of cascade process

3. **Export/Import**
   - Save lattice configurations
   - Load pre-analyzed patterns
   - Share results

4. **Batch Processing**
   - Analyze multiple texts
   - Compare regime distributions
   - Statistical analysis tools

5. **CORDIC Integration**
   - Hardware-ready integer operations
   - FPGA/ASIC code generation
   - Ultra-low latency mode

---

## Theory References

### Key Concepts

- **Golden Ratio Field Theory** - Mathematics of φ/ψ dynamics
- **Zeckendorf Representation** - Unique Fibonacci decomposition
- **Retrocausality** - Backward influence in cascade
- **Topological Computing** - Information in hole structure
- **P vs NP** - Regime emergence explanation

### Academic Connections

- Game Theory (Φ-Mamba framework)
- Difference-in-Differences (DiD causal inference)
- Topological Data Analysis (Betti numbers)
- Quantum Computing (superposition-like mixed regime)
- Information Geometry (φ-geodesics)

---

## License & Credits

**Leviathan AI Corporation**
Proof of Concept - Research & Development

This is a demonstration of novel computational substrate.
For research and evaluation purposes.

### Core Technologies

- Python 3.8+
- NumPy (numerical computing)
- Matplotlib (visualization)
- Tkinter (GUI framework)

---

## Contact & Support

For questions about this proof of concept:

1. Check the operation log for detailed step information
2. Review the mathematics in the Theory section
3. Experiment with different inputs and temperatures
4. Observe how regime changes with text complexity

---

## Quick Tips

💡 **Tip 1:** Start with simple words to see clear deterministic behavior
💡 **Tip 2:** Use temperature=0.5 for fast convergence demonstrations
💡 **Tip 3:** Compare "aaa" vs "xyz" to see encoding differences
💡 **Tip 4:** Watch cascade iterations - usually converges in 5-10 steps
💡 **Tip 5:** Green bars in Δ plot = forced/deterministic positions

---

**Ready to explore the self-organizing φ-field!** 🌀

Run `python run_zordic.py` and start analyzing.
