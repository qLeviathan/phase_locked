# 🚀 Quick Start Guide

## Run Your Integer-Only Phi-Mamba System in 5 Minutes

### Step 1: Setup Ollama (2 minutes)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Start server (in a separate terminal)
ollama serve
```

### Step 2: Install Python Dependencies (1 minute)

```bash
# Install required packages
pip install numpy requests psutil

# Optional: Build Rust bindings for 10x speedup
cd unified-zeckendorf-cordic
pip install maturin
maturin develop --release
cd ..
```

### Step 3: Run CLI (30 seconds)

```bash
# Interactive chat with both models
python concurrent_chat_cli.py --models phi-mamba,llama3
```

Example:
```
You: Explain quantum computing

[Phi-Mamba-Integer]
quantum computing is a new type of computing that...

[Ollama-llama3]
Quantum computing is a revolutionary approach to...

Total time: 1.85s (concurrent execution)
```

### Step 4: Or Run GUI (30 seconds)

```bash
# Launch desktop app
python concurrent_chat_gui.py
```

Features:
- ✅ Side-by-side comparison
- ✅ Visual status indicators
- ✅ Concurrent generation
- ✅ Easy model selection

### Step 5: Run Benchmarks (1 minute)

```bash
cd benchmarks
python run_benchmarks.py
```

---

## That's It!

You now have:
- ✅ Integer-only Phi-Mamba transformer
- ✅ Ollama integration
- ✅ CLI for concurrent chats
- ✅ Desktop GUI application
- ✅ Comprehensive benchmarks

## What Makes This Special?

**100% Integer-Only Operations**
- No floating point arithmetic
- Zeckendorf decomposition (Fibonacci representation)
- CORDIC shift-add trigonometry
- Formal algebraic axioms

**Concurrent Execution**
- Run multiple models simultaneously
- Compare outputs side-by-side
- Thread-based parallel generation

**Low Resource Usage**
- <50 MB memory footprint
- Faster than float operations
- Runs on CPU efficiently

---

## Troubleshooting

**"Ollama not available"**
```bash
# Make sure Ollama is running
curl http://localhost:11434/api/version

# If not running:
ollama serve
```

**"Phi-Mamba not available"**
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify installation
python -c "from phi_mamba_integer.integer_phi_mamba import IntegerPhiMamba; print('OK')"
```

**"Tkinter not found" (for GUI)**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (usually pre-installed)
# Windows (usually pre-installed)
```

---

## Next Steps

1. **Read Full Documentation**: See `ZECKENDORF_CORDIC_SETUP.md`
2. **Explore the Math**: Check `unified-zeckendorf-cordic/core/src/`
3. **Customize Models**: Edit model selection in CLI/GUI
4. **Run Benchmarks**: Compare performance metrics

---

## Quick Command Reference

```bash
# CLI - Interactive
python concurrent_chat_cli.py --models phi-mamba,llama3

# CLI - Benchmark
python concurrent_chat_cli.py --benchmark --models phi-mamba,llama3,mistral

# GUI
python concurrent_chat_gui.py

# Benchmarks
python benchmarks/run_benchmarks.py

# Build Rust core
cd unified-zeckendorf-cordic && cargo build --release

# Run tests
cargo test --release

# Demo integer-only model
python phi_mamba_integer/integer_phi_mamba.py
```

---

**"From chaos, mathematical order. No floats. Only truth."**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LEVIATHAN AI SYSTEMS
  Integer-Only Zeckendorf-CORDIC v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
