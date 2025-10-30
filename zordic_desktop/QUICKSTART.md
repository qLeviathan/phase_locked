# ZORDIC Desktop - Quick Start Guide

**Get running in 2 minutes!**

---

## Step 1: Install Dependencies (30 seconds)

```bash
cd zordic_desktop
pip install -r requirements.txt
```

Expected output:
```
✓ Successfully installed numpy-1.x.x matplotlib-3.x.x scipy-1.x.x
```

---

## Step 2: Launch Application (10 seconds)

```bash
python run_zordic.py
```

Or:
```bash
./run_zordic.py
```

Expected output:
```
======================================================================
  ZORDIC - φ-Field Self-Organizing Lattice System
  Leviathan AI Corporation
======================================================================

Initializing application...

✓ Application started successfully
  Window should appear now...
```

---

## Step 3: Run Your First Analysis (1 minute)

1. **Window opens** - You'll see the ZORDIC interface with dark theme

2. **Input field** (top left) - Contains default text: "quantum field theory"

3. **Click ⚡ ANALYZE** button

4. **Watch the magic:**
   - Log updates in real-time (bottom left)
   - Metrics appear (middle left)
   - 4 graphs populate (right side)

5. **Results shown:**
   - Blue/red field plots (top right)
   - Stability bar chart (middle right) - green=stable, orange=unstable
   - Convergence curve (bottom right)
   - Regime classification in metrics

---

## What You're Seeing

### The Logs (bottom left)
```
=== PHASE 1: ENCODING ===
[0] 'q' → shells=[17], φ=1597.000, ψ=0.000, Δ=1597.000
[1] 'u' → shells=[8], φ=21.000, ψ=0.000, Δ=21.000
...
✓ Encoded 20 characters
```

Each character becomes a point in φ-space with specific "shells" (Fibonacci indices).

### The Metrics (middle left)
```
INITIAL STATE:
  φ-field total:    15234.567
  ψ-field total:    -234.567
  ...

FINAL REGIME:
  Type:             MIXED (Quantum-like)
  Deterministic:    67.4%
```

Shows how much of the system collapsed to deterministic vs stayed stochastic.

### The Graphs (right side)

**Top left (blue):** φ-field strength at each position
**Top right (red):** ψ-field strength at each position
**Middle (bars):** Stability - green bars = deterministic, orange = needs sampling
**Bottom (line):** How system converged over iterations

---

## Try These Examples

### Example 1: Highly Deterministic
```
Input: aaa
Temperature: 1.0
Expected: >90% deterministic (lots of green bars)
```

### Example 2: Highly Stochastic
```
Input: xyz
Temperature: 2.0
Expected: <30% deterministic (mostly orange bars)
```

### Example 3: Mixed Regime
```
Input: hello world
Temperature: 1.0
Expected: 50-70% deterministic (mix of green/orange)
```

---

## Temperature Slider Explained

- **0.1 - 0.5:** Fast convergence, more deterministic
- **1.0:** Balanced (default)
- **1.5 - 2.0:** Slow convergence, more stochastic

Try the same text at different temperatures to see regime shift!

---

## Interpreting Results

### DETERMINISTIC Regime (>75%)
- System "knows" the answer
- Output is geometrically forced
- Like solving 2+2 - only one answer possible
- **This is P-like behavior**

### STOCHASTIC Regime (<35%)
- Multiple valid paths exist
- Need to sample/search
- Like solving sudoku - must try options
- **This is NP-like behavior**

### MIXED Regime (35-75%)
- Some parts forced, some parts free
- Quantum-like superposition
- Like partial sudoku - some cells forced, others need search
- **This is BQP-like behavior**

---

## Common Questions

**Q: Why do some characters have high φ values?**
A: Characters map to integers, which decompose into Fibonacci sums. Higher character values = higher Fibonacci terms = exponentially larger φ powers.

**Q: What are "shells"?**
A: The Fibonacci indices in the Zeckendorf decomposition. E.g., 17 = F₇ + F₄ + F₂ means shells=[7,4,2].

**Q: Why does convergence take different iterations?**
A: Depends on text complexity and temperature. Simple patterns converge fast, complex patterns need more cascade iterations.

**Q: What's the point of this?**
A: Demonstrates that the SAME mathematical substrate naturally produces both deterministic and stochastic behavior - explaining why some problems are easy (P) and others hard (NP).

---

## Next Steps

1. ✅ You've run your first analysis
2. Try different texts and temperatures
3. Observe how regime changes
4. Read the full README.md for mathematical details
5. Wait for mobile version! (coming soon)

---

## Need Help?

**Application won't start?**
```bash
# Check Python version (need 3.8+)
python --version

# Check tkinter
python -c "import tkinter; print('✓ tkinter works')"

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Graphs not showing?**
```bash
# Check matplotlib
python -c "import matplotlib; print('✓ matplotlib works')"
```

**Something broken?**
Check the Operation Log (bottom left) for error messages.

---

**You're ready to explore! 🚀**

The self-organizing φ-field awaits your input...
