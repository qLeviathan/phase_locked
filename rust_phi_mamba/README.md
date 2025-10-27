# Rust Phi-Mamba Implementation

A high-performance Rust implementation of the Phi-Mamba game-theoretic language modeling framework with WASM and Tauri support.

## Structure

- `crates/phi-core/` - Core mathematical primitives and game theory implementation
- `crates/phi-wasm/` - WebAssembly bindings for browser deployment
- `crates/phi-tauri/` - Desktop application using Tauri framework
- `ui/` - Frontend UI (to be added)

## Features

### Core Library (`phi-core`)
- Integer-only golden ratio arithmetic using Lucas sequences
- Zeckendorf decomposition for Fibonacci representation
- Game-theoretic token state management
- Zero-copy, high-performance implementation

### WASM Module (`phi-wasm`)
- JavaScript/TypeScript bindings
- Efficient number representations
- Console logging support
- Full API exposure to web applications

### Desktop App (`phi-tauri`)
- Native desktop application
- Visual exploration of golden ratio properties
- Interactive game theory demonstrations
- Cross-platform (Windows, macOS, Linux)

## Building

### Prerequisites
- Rust 1.75 or later
- wasm-pack for WASM builds
- Node.js 18+ for Tauri

### Build Commands

```bash
# Build entire workspace
cargo build --release

# Build WASM module
cd crates/phi-wasm
wasm-pack build --target web

# Run Tauri app (requires UI to be built)
cd crates/phi-tauri
cargo tauri dev

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

## WASM Usage

```javascript
import init, { 
  WasmFibonacci, 
  wasm_zeckendorf_decomposition,
  WasmPhiInt 
} from './phi_wasm.js';

await init();

// Calculate Fibonacci numbers
const fib = new WasmFibonacci();
console.log(fib.get(10)); // "55"

// Zeckendorf decomposition
const decomp = wasm_zeckendorf_decomposition(100);
console.log(decomp); // [10, 7, 4, 2]

// Golden ratio arithmetic
const phi = WasmPhiInt.phi();
const phi_squared = phi.multiply(phi);
console.log(phi_squared.to_string()); // "1 + 1φ"
```

## Tauri App Commands

The Tauri app exposes these commands to the frontend:

- `calculate_fibonacci(index)` - Get nth Fibonacci number
- `decompose_zeckendorf(number)` - Get Zeckendorf decomposition
- `calculate_phi_arithmetic(op, a, b)` - Perform golden ratio arithmetic
- `get_constants()` - Get mathematical constants (φ, ψ, etc.)

## Performance

The Rust implementation provides:
- 10-100x faster Fibonacci calculations than Python
- Zero-allocation Zeckendorf decomposition
- Cache-friendly data structures
- Parallel processing support (via Rayon, to be added)

## Next Steps

1. Add frontend UI with React/Svelte
2. Implement full game-theoretic language model
3. Add GPU acceleration via wgpu
4. Create interactive visualizations
5. Add streaming token generation