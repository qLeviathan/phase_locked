# Claude Code Configuration - Phase Locked Development Environment

## CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently

### GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message

### Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "Explore")
  Task("Coder agent", "Implement core features...", "general-purpose")
  Task("Architect agent", "Design system architecture...", "Plan")
```

### File Organization Rules

**NEVER save to root folder. Use these directories:**
```
/src              - Source code files
/tests            - Test files
/docs             - Documentation and markdown files
/config           - Configuration files
/scripts          - Utility scripts
/examples         - Example code
/benchmarks       - Benchmark code and results
/notebooks        - Jupyter notebooks
/output           - Generated outputs
```

**Existing Project Structure:**
```
/phi_core         - Core Phi-Mamba integer arithmetic
/phi_mamba        - Phi-Mamba SSM implementation
/phi_mamba_integer - Integer-only Phi-Mamba
/rust_phi_mamba   - Rust implementation
/llama_compression - LLM compression experiments
/ablation_study   - Ablation study results
/zordic_desktop   - Desktop application (Tauri)
/zordic_sensorium - Sensor integration
/unified-zeckendorf-cordic - ZORDIC unified implementation
/aurelia-core     - Aurelia framework core
/mobile-agent     - Mobile agent implementation
/phi-mamba-desktop - Desktop Phi-Mamba app
/phi-mamba-signals - Signal processing
/bit_cascade      - Bit cascade operations
/demo             - Demo implementations
```

---

## Project Overview

**Phase Locked** implements integer-only neural computation using:
- **ZORDIC**: Zeckendorf-CORDIC integer arithmetic system
- **Phi-Mamba**: State-space models with golden ratio dynamics
- **Bit Cascade**: Hierarchical bit operations for attention
- **Aurelia**: Unified agentic framework

### Core Technologies
- Zeckendorf representation (Fibonacci-based integers)
- CORDIC algorithms (rotation-only trigonometry)
- Phase-locked computation (deterministic state evolution)
- Integer-only inference (no floating point)

---

## Build Commands

### Python
```bash
pip install -e .                    # Install package
python -m pytest tests/             # Run tests
python comprehensive_test.py        # Full validation
python validate_zordic.py           # ZORDIC validation
```

### Rust
```bash
cd rust_phi_mamba && cargo build --release
cd rust_phi_mamba && cargo test
cd rust_phi_mamba/zordic && cargo run --release
```

### Desktop App (Tauri)
```bash
cd zordic_desktop && npm install && npm run tauri dev
cd phi-mamba-desktop && npm install && npm run tauri dev
```

---

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Integer Arithmetic**: Prefer integer ops over float when possible
- **Documentation**: Update docs in `/docs` not root

---

## Agent Execution with Claude Code

### The Correct Pattern:

1. **USE Claude Code's Task tool** to spawn agents for actual work
2. **BATCH all operations** in single messages
3. **ORGANIZE files** in proper subdirectories

### Example Development Workflow:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Explore codebase", "Find all CORDIC implementations and understand the pattern", "Explore")
  Task("Plan architecture", "Design the new integer attention module", "Plan")
  Task("Implement feature", "Build the CORDIC-based attention layer", "general-purpose")

  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }

  // All file operations together
  Write "src/cordic_attention.py"
  Write "tests/test_cordic_attention.py"
```

### WRONG (Multiple Messages):
```javascript
Message 1: Task("agent 1")
Message 2: TodoWrite { todos: [single todo] }
Message 3: Write "file.py"
// This breaks parallel coordination!
```

---

## Key Files Reference

### Core Implementation
- `phi_core/` - Integer arithmetic primitives
- `phi_mamba/phi_mamba.py` - Main Phi-Mamba implementation
- `bit_cascade/` - Bit cascade attention
- `rust_phi_mamba/zordic/` - Rust ZORDIC implementation

### Validation & Testing
- `validate_zordic.py` - ZORDIC validation suite
- `comprehensive_test.py` - Full system tests
- `tests/` - Unit tests
- `ablation_study/` - Ablation experiments

### Documentation (in /docs)
- `docs/math_foundations.md` - Mathematical foundations
- `docs/implementation.md` - Implementation guide

---

## Concurrent Execution Examples

### CORRECT: Batched Operations

```javascript
[Single Message]:
  // Parallel exploration
  Task("Find CORDIC", "Search for all CORDIC implementations", "Explore")
  Task("Find tests", "Search for test patterns used", "Explore")

  // Parallel reads
  Read "phi_core/cordic.py"
  Read "phi_core/zeckendorf.py"
  Read "tests/test_cordic.py"

  // Batched todos
  TodoWrite { todos: [
    {content: "Analyze CORDIC implementation", status: "in_progress"},
    {content: "Review Zeckendorf encoding", status: "pending"},
    {content: "Design new attention layer", status: "pending"},
    {content: "Implement integer attention", status: "pending"},
    {content: "Write unit tests", status: "pending"},
    {content: "Run validation suite", status: "pending"},
    {content: "Update documentation", status: "pending"},
    {content: "Benchmark performance", status: "pending"}
  ]}
```

### WRONG: Sequential Single Operations
```javascript
Message 1: Read "file1.py"
Message 2: Read "file2.py"
Message 3: TodoWrite {todos: [one todo]}
// Inefficient! Batch these!
```

---

## Performance Targets

- **Integer-only inference**: No FP operations in forward pass
- **Bit-exact reproducibility**: Same input = same output always
- **Low latency**: Sub-millisecond token processing
- **Memory efficient**: Minimal state overhead

---

## Important Reminders

1. **Do what has been asked; nothing more, nothing less**
2. **NEVER create files unless absolutely necessary**
3. **ALWAYS prefer editing existing files over creating new ones**
4. **NEVER proactively create documentation files unless requested**
5. **Never save working files, text/mds and tests to the root folder**
6. **Batch ALL related operations in a single message**
7. **Use Task tool for complex multi-step exploration**

---

## Quick Reference

| Task | Tool | Notes |
|------|------|-------|
| Explore codebase | `Task` with `Explore` | For open-ended search |
| Plan implementation | `Task` with `Plan` | For architecture design |
| Complex work | `Task` with `general-purpose` | For multi-step tasks |
| Read files | `Read` | Batch multiple reads |
| Search patterns | `Glob`, `Grep` | Batch multiple searches |
| Edit files | `Edit` | Batch related edits |
| Track progress | `TodoWrite` | Batch all todos at once |
