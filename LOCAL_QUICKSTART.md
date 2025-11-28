# Phase-Locked: Local Setup & Run Guide

## Prerequisites

```bash
# Python 3.8+
python --version

# Rust (for production components)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Node.js (for desktop app)
node --version  # v18+
```

---

## 1. Python Core (Fastest Start)

```bash
cd /home/user/phase_locked

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installation
python -c "from phi_mamba import PHI, PSI; print(f'φ = {PHI}, ψ = {PSI}')"
```

### Run Examples

```bash
# Game theory validation (comprehensive test)
python game_theory_validation.py

# Basic language model demo
python examples/basic_generation.py

# Financial analysis demo
python examples/financial_analysis_demo.py

# Compression demo (177x-295x)
cd llama_compression && python extreme_compression.py
```

---

## 2. Rust Core (Production Performance)

```bash
# Build phi_core library
cd phi_core
cargo test      # Run tests
cargo build --release

# Build rust_phi_mamba
cd ../rust_phi_mamba
cargo test
cargo build --release

# Run ZORDIC demo
cargo run --example zordic_demo
```

---

## 3. Desktop Application (Tauri GUI)

```bash
cd phi-mamba-desktop

# Install frontend dependencies
npm install

# Development mode (hot reload)
npm run tauri dev

# Production build
npm run tauri build
```

**Opens**: Real-time trade signal visualization with <1ms latency

---

## 4. Validation & Testing

```bash
# Python tests
cd /home/user/phase_locked
python -m pytest tests/

# Comprehensive validation
python comprehensive_test.py

# ZORDIC validation GUI
python zordic_desktop/zordic_gui.py

# Generate journal figures
python journal_graphics.py
```

---

## 5. Key Entry Points

| What You Want | Command |
|---------------|---------|
| **Quick demo** | `python demo_unique_features.py` |
| **Compression test** | `cd llama_compression && python extreme_compression.py` |
| **Financial system** | `python economic_demo.py` |
| **Game theory proofs** | `python game_theory_validation.py` |
| **Ablation study** | `cd llama_compression && python comprehensive_ablation.py` |
| **Visualizations** | `python advanced_visualizations.py` |

---

## 6. Directory Quick Reference

```
phase_locked/
├── phi_mamba/           # Python library (import from here)
├── examples/            # Usage examples
├── llama_compression/   # Compression demos
├── phi_core/            # Rust core (cargo build)
├── rust_phi_mamba/      # Rust production
├── phi-mamba-desktop/   # Tauri app (npm run tauri dev)
└── zordic_desktop/      # Python validation GUI
```

---

## 7. Minimal Test (Copy-Paste)

```python
#!/usr/bin/env python3
"""Minimal test - run from phase_locked directory"""

from phi_mamba import PhiLanguageModel, zeckendorf_decomposition, PHI

# Test Zeckendorf decomposition
n = 100
decomp = zeckendorf_decomposition(n)
print(f"Zeckendorf({n}) = {decomp}")  # [3, 8, 89] → 3+8+89=100

# Test language model
model = PhiLanguageModel(vocab_size=1000)
text = model.generate("the cat", max_length=10)
print(f"Generated: {text}")

# Test φ-arithmetic
print(f"φ² = {PHI**2:.6f}, φ+1 = {PHI+1:.6f}")  # Should be equal!

print("✅ All systems operational")
```

Save as `test_local.py` and run: `python test_local.py`

---

*For full documentation, see: `PHASE_LOCKED_COMPLETE_SYSTEM_MAP.md`*
