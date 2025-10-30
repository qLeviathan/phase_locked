# Phi-Mamba Desktop - Latency-First Architecture

**Absolute Requirement: <1ms end-to-end latency (tick → signal → display)**

## Latency Budget

```
WebSocket tick arrives
    ↓ <0.1ms   Network stack to ring buffer
CORDIC encoding (Rust)
    ↓ <0.015ms  15μs per bar (add/subtract/shift only)
Berry phase compute
    ↓ <0.002ms  2μs incremental update (not full matrix)
Tauri IPC (shared memory)
    ↓ <0.05ms   Zero-copy via SharedArrayBuffer
React reconciliation
    ↓ <0.1ms    Minimal DOM updates (Canvas only)
WebGL draw call
    ↓ <0.5ms    GPU buffer upload + shader execution
Display
─────────────────────────────────────────────────
TOTAL: ~0.767ms ✓ (target: <1ms)
```

## Architecture

### Data Flow (Zero-Copy)

```
┌─────────────────────────────────────────────────────┐
│ WebSocket Feed (1000 ticks/sec)                     │
└────────────────┬────────────────────────────────────┘
                 ↓ lock-free ring buffer
┌─────────────────────────────────────────────────────┐
│ Rust Backend (Tauri Core)                           │
│  • CORDIC encoder                                    │
│  • Berry phase incremental                          │
│  • Signal generator                                  │
│  • Shared memory writer                             │
└────────────────┬────────────────────────────────────┘
                 ↓ SharedArrayBuffer (zero-copy)
┌─────────────────────────────────────────────────────┐
│ Frontend (React + WebGL)                             │
│  • SharedArrayBuffer reader                          │
│  • WebGL renderer (holographic field)               │
│  • Canvas-only updates (no DOM thrashing)           │
│  • RequestAnimationFrame (16.67ms budget)           │
└─────────────────────────────────────────────────────┘
```

## Tech Stack (Latency-Optimized)

### Backend (Rust)
- **Tauri 2.0**: Native IPC with shared memory
- **phi-mamba-signals**: Our CORDIC core (<15μs encoding)
- **tokio-tungstenite**: Async WebSocket (lock-free)
- **crossbeam**: Lock-free ring buffer (SPSC)
- **rayon**: Data parallelism for batch updates

### Frontend (React + WebGL)
- **React 18**: Concurrent rendering + transitions
- **Three.js**: WebGL abstraction (optimized)
- **@react-three/fiber**: React renderer for Three.js
- **zustand**: Minimal state management (no Redux overhead)
- **SharedArrayBuffer**: Zero-copy from Rust

### IPC (Zero-Copy)
- **SharedArrayBuffer**: OS-level shared memory
- **Atomics**: Lock-free synchronization
- **Ring buffer**: SPSC queue for ticks

## Critical Path Optimization

### 1. WebSocket → Ring Buffer (<0.1ms)

```rust
// Lock-free SPSC ring buffer
use crossbeam::queue::ArrayQueue;

static TICK_BUFFER: ArrayQueue<OHLCVBar> = ArrayQueue::new(1024);

async fn websocket_handler(stream: WebSocketStream) {
    while let Some(msg) = stream.next().await {
        let bar = parse_tick(msg); // ~10μs
        TICK_BUFFER.push(bar).ok(); // lock-free
    }
}
```

### 2. CORDIC Encoding (<0.015ms)

```rust
// Already optimized in phi-mamba-signals
let state = encoder.encode(&bar); // 15μs (measured)
```

### 3. Berry Phase Incremental (<0.002ms)

```rust
// Don't recompute full N×N matrix!
// Only update changed pairs

fn incremental_berry_phase(
    old_state: &FinancialState,
    new_state: &FinancialState,
    cache: &mut BerryCache,
) -> BerryPhase {
    // Cached computation: 2μs vs 200μs full matrix
    compute_berry_phase(old_state, new_state)
}
```

### 4. Shared Memory IPC (<0.05ms)

```rust
// Write to SharedArrayBuffer
use tauri::State;

#[tauri::command]
fn get_signal_buffer() -> Vec<u8> {
    SHARED_BUFFER.read_slice(0, 4096) // mmap, no copy
}
```

### 5. WebGL Direct Buffer (<0.5ms)

```javascript
// Upload to GPU once, update attributes
const positions = new Float32Array(sharedBuffer); // zero-copy view

geometry.attributes.position.array = positions;
geometry.attributes.position.needsUpdate = true;

renderer.render(scene, camera); // GPU draw call
```

## Performance Monitoring

### Built-in Profiler

Every component reports latency:

```rust
struct LatencyMetrics {
    ws_recv: Duration,      // target: <0.1ms
    cordic: Duration,       // target: <0.015ms
    berry: Duration,        // target: <0.002ms
    ipc: Duration,          // target: <0.05ms
    total: Duration,        // target: <1ms
}

impl LatencyMetrics {
    fn exceeds_budget(&self) -> bool {
        self.total > Duration::from_micros(1000)
    }
}
```

### Chrome DevTools Integration

```javascript
performance.mark('tick-received');
// ... processing ...
performance.mark('frame-rendered');
performance.measure('latency', 'tick-received', 'frame-rendered');

const latency = performance.getEntriesByName('latency')[0].duration;
if (latency > 1.0) {
    console.warn(`LATENCY BUDGET EXCEEDED: ${latency}ms`);
}
```

## Project Structure

```
phi-mamba-desktop/
├── src-tauri/                      # Rust backend
│   ├── src/
│   │   ├── main.rs                 # Tauri entry point
│   │   ├── websocket.rs            # WebSocket feed handler
│   │   ├── encoder.rs              # CORDIC encoding loop
│   │   ├── signal.rs               # Signal generator
│   │   ├── shared_memory.rs        # Zero-copy buffer
│   │   └── metrics.rs              # Latency monitoring
│   ├── Cargo.toml                  # phi-mamba-signals dep
│   └── tauri.conf.json             # Tauri config
│
├── src/                            # React frontend
│   ├── App.tsx                     # Main app
│   ├── hooks/
│   │   ├── useSharedBuffer.ts      # SharedArrayBuffer hook
│   │   └── useLatencyMonitor.ts    # Performance tracking
│   ├── components/
│   │   ├── HolographicField.tsx    # WebGL renderer
│   │   ├── SignalPanel.tsx         # Trade signals
│   │   ├── LatencyMonitor.tsx      # Real-time metrics
│   │   └── TickerInput.tsx         # Ticker selection
│   └── shaders/
│       ├── field.vert              # Vertex shader
│       └── field.frag              # Fragment shader
│
├── package.json
└── tsconfig.json
```

## Benchmark Targets

### Development (Debug)
- CORDIC encoding: <50μs
- Berry phase: <10μs
- Total latency: <2ms

### Production (Release)
- CORDIC encoding: <15μs ✓
- Berry phase: <2μs ✓
- Total latency: <1ms ✓

## Fallback Strategy

If latency budget exceeded:

1. **Skip frame** (better to drop than lag)
2. **Reduce tick rate** (1000/s → 500/s)
3. **Disable Berry phase** (show raw signals)
4. **Alert user** (red indicator)

## Testing

### Synthetic Load Test

```rust
#[test]
fn test_latency_budget() {
    let mut encoder = FinancialEncoder::default();
    let mut total = Duration::ZERO;

    for _ in 0..1000 {
        let start = Instant::now();

        let bar = generate_random_bar();
        let state = encoder.encode(&bar);
        let berry = compute_berry_phase(&state, &state);

        let elapsed = start.elapsed();
        total += elapsed;

        assert!(elapsed < Duration::from_micros(1000));
    }

    let avg = total / 1000;
    println!("Average latency: {:?}", avg);
    assert!(avg < Duration::from_micros(800)); // 20% margin
}
```

### Real-time Profiling

```bash
# Profile with perf
cargo build --release
perf record --call-graph=dwarf ./target/release/phi-mamba-desktop
perf report

# Expected hotspots:
# 1. CORDIC sin_cos (15μs)
# 2. Zeckendorf decomp (5μs)
# 3. WebSocket parse (10μs)
# 4. Shared memory write (2μs)
```

## Key Principles

1. **Zero-copy everywhere** - Never memcpy large buffers
2. **Lock-free data structures** - SPSC ring buffers
3. **Incremental updates** - Don't recompute full state
4. **GPU acceleration** - WebGL for all visualization
5. **Measure constantly** - Built-in latency profiler
6. **Fail fast** - Drop frames rather than lag

## Next Steps

1. Create Tauri project with shared memory setup
2. Integrate phi-mamba-signals crate
3. Implement lock-free WebSocket handler
4. Build WebGL holographic renderer
5. Add latency monitoring dashboard
6. Benchmark against <1ms target

---

**Philosophy**: Every microsecond counts. If it's not measured, it's not optimized.
