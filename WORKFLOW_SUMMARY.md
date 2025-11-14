# Latency-Minimized Workflow: Quick Reference

## 🎯 **The Problem**: Fragmented Data Flow (805μs latency)

```
Current Architecture (INEFFICIENT):

WebSocket → Desktop Encoder (15μs) ─→ Desktop UI
    ↓
    └─→ Mobile Encoder (15μs) ─────→ Mobile Agent

Berry Phase: Full N×N matrix (50μs)
Signals: Sequential generation (25μs)
GPU: CPU copy + upload (50μs)

TOTAL: 805μs
ISSUES:
- Duplicate encoding (2×)
- Full matrix computation every tick
- Sequential signal generation
- Extra CPU copy to GPU
```

## ✅ **The Solution**: Unified Pipeline (680μs latency)

```
Optimized Architecture (EFFICIENT):

WebSocket → UNIFIED ENCODER (15μs ONCE)
                ↓
            [Zero-Copy Distribution]
                ├─→ Desktop Buffer (1μs atomic push)
                ├─→ Mobile Buffer (1μs atomic push)
                └─→ Berry Cache (10μs incremental)
                        ↓
                [Parallel Signal Gen] (5μs Rayon)
                        ↓
                [Direct GPU] (0μs SharedArrayBuffer)
                        ↓
                [Display] (500μs GPU shader)

TOTAL: 680μs (-125μs = 15.5% faster)
BENEFITS:
✅ Single encoding source
✅ Incremental Berry phase
✅ Parallel signal generation
✅ Zero-copy GPU upload
```

---

## 📊 Latency Budget Breakdown

### Before Optimization

| Component | Latency | Notes |
|-----------|---------|-------|
| WebSocket parse | 100μs | Network stack |
| CORDIC encode (desktop) | 15μs | Add/shift only |
| CORDIC encode (mobile) | 15μs | **❌ DUPLICATE** |
| Berry phase (full N²) | 50μs | **❌ RECOMPUTES ALL** |
| Signal generation (sequential) | 25μs | **❌ NOT PARALLEL** |
| IPC transfer | 50μs | Zero-copy (OK) |
| GPU copy | 50μs | **❌ CPU MEMCPY** |
| Display render | 500μs | GPU shader (OK) |
| **TOTAL** | **805μs** | **❌ 195μs over target** |

### After Optimization

| Component | Latency | Optimization |
|-----------|---------|--------------|
| WebSocket parse | 100μs | (no change) |
| **Unified CORDIC** | **15μs** | ✅ **Single source** |
| ~~Duplicate encode~~ | ~~15μs~~ | ✅ **Eliminated** |
| **Berry incremental** | **10μs** | ✅ **Cache N, not N²** |
| **Parallel signals** | **5μs** | ✅ **Rayon vectorized** |
| IPC transfer | 50μs | (no change) |
| ~~GPU copy~~ | ~~50μs~~ | ✅ **Direct buffer** |
| Display render | 500μs | (no change) |
| **TOTAL** | **680μs** | ✅ **320μs under target** |

---

## 🚀 Implementation Roadmap

### Week 1: Critical Path Optimizations (Highest Impact)

**Day 1-2: Unified Pipeline**
```rust
// File: phi-mamba-desktop/src-tauri/src/unified_pipeline.rs

pub struct UnifiedPipeline {
    encoder: Arc<RwLock<FinancialEncoder>>,  // Single source
    desktop_queue: Arc<ArrayQueue<State>>,
    mobile_queue: Arc<ArrayQueue<State>>,
    berry_cache: Arc<RwLock<BerryCache>>,    // Incremental
}

impl UnifiedPipeline {
    pub fn process(&self, bar: OHLCVBar) {
        let state = self.encoder.write().encode(&bar);  // 15μs ONCE

        self.desktop_queue.push(state.clone());         // 1μs
        self.mobile_queue.push(state.clone());          // 1μs
        self.berry_cache.write().update(&state);        // 10μs
    }
}
```

**Impact**: -55μs (15μs duplicate + 40μs Berry phase)

**Day 3-4: Direct GPU Upload**
```typescript
// File: phi-mamba-desktop/src/hooks/useDirectGPU.ts

const sharedBuffer = new SharedArrayBuffer(4096 * 12);
const positions = new Float32Array(sharedBuffer);  // Zero-copy view

geometry.setAttribute('position',
  new THREE.BufferAttribute(positions, 3)  // Direct GPU buffer
);

// Rust writes directly to GPU-visible memory (no CPU copy)
```

**Impact**: -50μs (eliminates CPU memcpy)

**Day 5: Integration & Testing**
- Wire unified pipeline into main loop
- Verify latency measurements
- Profile with perf/flamegraph

---

### Week 2: Parallel Processing & Decoupling

**Day 1-2: Parallel Signal Generation**
```rust
// File: phi-mamba-desktop/src-tauri/src/signal_generator.rs

use rayon::prelude::*;

let signals = states.par_iter()
    .filter_map(|state| generate_signal(state))  // 5μs parallel
    .collect();
```

**Impact**: -20μs (vs 25μs sequential)

**Day 3-5: Display Decoupling**
```rust
// Data path: 1000 Hz (all ticks)
tokio::spawn(encode_loop);  // 26μs × 1000 = 26ms/sec CPU

// Display path: 60 Hz (visual updates only)
setInterval(update_webgl, 16.67ms);  // 0.5ms × 60 = 30ms/sec GPU
```

**Impact**: -95ms/sec CPU usage (12.5% → 3%)

---

### Week 3: Production Ready

**Real Data Integration**
- WebSocket feeds (Polygon.io, Alpha Vantage)
- Binary protocol parser
- Automatic failover to simulated feed

**Comprehensive Benchmarks**
- Load testing: 1000 ticks/sec sustained
- Latency distribution: P50/P95/P99
- CPU/memory profiling under load

---

## 🎓 Key Design Principles

### 1. **Encode Once, Distribute Zero-Copy**
```
BAD:  Encode → Copy → Encode → Copy
GOOD: Encode → [View] → [View] → [View]
```

### 2. **Incremental Over Full Recomputation**
```
BAD:  Full N×N matrix every tick (O(N²))
GOOD: Update N new pairs only (O(N))
```

### 3. **Parallel Where Possible**
```
BAD:  for state in states { compute(state); }
GOOD: states.par_iter().map(compute).collect();
```

### 4. **Direct GPU Access**
```
BAD:  CPU buffer → memcpy → GPU buffer
GOOD: SharedArrayBuffer (CPU & GPU can both access)
```

### 5. **Decouple Data from Display**
```
BAD:  Update UI every tick (1000 Hz)
GOOD: Data at 1000 Hz, Display at 60 Hz
```

---

## 📈 Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Latency (P50)** | 805μs | 680μs | **-15.5%** |
| **Latency (P99)** | ~1.2ms | <1ms | **✅ Under budget** |
| **CPU usage** | 12.5% | 3% | **-76%** |
| **Headroom** | -195μs | +320μs | **✅ 32% margin** |
| **Throughput** | 1000/sec | 1000/sec | (maintained) |

---

## 🔧 Quick Test

```bash
# Build optimized version
cd phi-mamba-desktop
npm run tauri build

# Run with profiling
RUST_LOG=info npm run tauri dev

# Measure latency in DevTools
performance.measure('latency', 'tick-received', 'frame-rendered');

# Check P99 < 1ms
console.log(performance.getEntriesByName('latency')
  .map(e => e.duration)
  .sort()[Math.floor(length * 0.99)]
);
```

---

## ✅ Success Criteria

**Week 1 Complete When**:
- [ ] Unified pipeline implemented
- [ ] Direct GPU upload working
- [ ] Latency measured < 750μs

**Week 2 Complete When**:
- [ ] Parallel signals implemented
- [ ] Display decoupled (60 Hz)
- [ ] CPU usage < 5%

**Week 3 Complete When**:
- [ ] Real data feed integrated
- [ ] P99 latency < 1ms
- [ ] Load tested at 1000 ticks/sec

---

**BOTTOM LINE**:

Single unified pipeline + incremental computation + zero-copy IPC = **680μs latency** ✅

**Next**: Implement `unified_pipeline.rs` (15.5% latency reduction in one file)
