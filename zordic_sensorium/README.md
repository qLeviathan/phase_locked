# ZORDIC SENSORIUM - AI HUD Interface

**Real-time visualization of AI internal state - A sensorium for artificial intelligence**

---

## Concept

The ZORDIC Sensorium is a **heads-up display (HUD)** that visualizes the internal state of an AI system in real-time. Think of it as a cockpit for AI - showing you what the AI "sees," "thinks," and "feels" as it processes information.

### What is a Sensorium?

A **sensorium** is the apparatus of an organism's perception considered as a whole. For AI, this means:

- **Visual representation** of computational processes
- **Real-time feedback** from φ-field dynamics
- **State awareness** - regime switching, stability, coherence
- **Sensory data streams** - tokens, fields, metrics
- **Operational monitoring** - like an aircraft cockpit but for AI

---

## Features

### 1. **φ-Field Dynamics Visualizer**

Real-time visualization of the dual φ/ψ conjugate field:

- **φ-wave** (cyan) - Forward causal component
- **ψ-wave** (pink) - Backward causal component
- **Δ stability** (orange) - Difference metric showing deterministic vs stochastic

**What it shows:**
- Field oscillations during processing
- Phase coherence between forward/backward causality
- Stability regions where AI is "certain" vs "uncertain"

### 2. **Token Processing Stream**

Live visualization of token-by-token processing:

- **Characters flowing** right-to-left (processing order)
- **Color coding:**
  - Cyan = Stable/deterministic token
  - Orange = Unstable/needs sampling
- **Glow effect** for high-uncertainty tokens
- **Fade over time** (5-second persistence)

**What it shows:**
- Real-time text processing
- Which tokens are "easy" (collapse immediately)
- Which tokens need exploration (sampling required)

### 3. **Regime State Monitor**

Shows current operational regime:

- **DETERMINISTIC** (green) - AI "knows" the answer
  - High confidence
  - Unique output path
  - Fast processing

- **STOCHASTIC** (pink) - AI needs to "search"
  - Multiple viable paths
  - Sampling required
  - Slower processing

- **MIXED** (orange) - Quantum-like superposition
  - Some parts certain, some uncertain
  - Partial determinism

**Includes:**
- Bar graph showing deterministic ratio
- Historical trend (last 100 updates)
- Real-time percentage

### 4. **System Metrics Panel**

Live system metrics:

- **φ-field** - Forward causal strength
- **ψ-field** - Backward causal strength
- **Δ** - Stability metric
- **FPS** - Rendering performance
- **Nodes** - Token count (when integrated)
- **Connections** - Edge count in lattice
- **Iterations** - Cascade steps taken

---

## Installation

### Requirements

- Python 3.8+
- pygame 2.5+
- numpy 1.21+

### Setup

```bash
cd zordic_sensorium
pip install -r requirements.txt
```

---

## Running

### Quick Start

```bash
python run_sensorium.py
```

### What You'll See

A full-screen HUD interface with:

**Top:**
- Large title "ZORDIC SENSORIUM"
- Subtitle "AI State Monitor - φ-Field Dynamics"

**Center-Top:**
- φ-field dynamics graph (800x300px)
- Real-time wave visualization

**Center:**
- Token stream (flowing characters)
- Color-coded by stability

**Right Side:**
- Regime indicator
- Current state (DETERMINISTIC/STOCHASTIC/MIXED)
- Progress bar showing ratio
- Historical trend

**Left Side:**
- Metrics panel
- Key values updated real-time

**Bottom:**
- Status bar
- System online indicator
- Current regime
- Timestamp

**Background:**
- Cyberpunk-style grid
- Dark theme for extended viewing
- HUD corner brackets

---

## Controls

| Key | Action |
|-----|--------|
| **ESC** | Exit sensorium |
| **SPACE** | Pause/Resume simulation |

---

## Architecture

### Data Flow

```
AI Engine
    ↓
Token Processing
    ↓
φ/ψ Field Calculation
    ↓
Data Queue
    ↓
Sensorium HUD
    ↓
Visual Display
```

### Components

**1. ZordicSensorium** (Main class)
- Manages entire HUD
- Coordinates all visualizers
- Handles rendering loop

**2. PhiFieldVisualizer**
- Real-time waveform display
- Dual φ/ψ streams
- Stability delta

**3. TokenStreamVisualizer**
- Character-by-character display
- Scrolling effect
- Stability color coding

**4. RegimeIndicator**
- Current state display
- Historical tracking
- Visual bar graph

**5. MetricsPanel**
- Key-value pairs
- Real-time updates
- Configurable metrics

### Threading Model

- **Main thread** - Pygame rendering loop (60 FPS)
- **Simulation thread** - Data generation (background)
- **Queue-based** communication (thread-safe)

---

## Integration with ZORDIC Engine

### Current State

**Simulation Mode:**
- Generates synthetic φ/ψ oscillations
- Random token stream
- Simulated regime switching

**Purpose:** Demonstrates HUD interface without requiring full engine

### Future Integration

**Real AI Engine:**
```python
from ai_engine_bridge import AIEngineBridge

# Create bridge
bridge = AIEngineBridge()

# Process text stream
bridge.process_text_stream("quantum field theory")

# Get real-time data
while processing:
    event = bridge.data_queue.get()

    if event['type'] == 'token':
        sensorium.token_stream.add_token(
            event['char'],
            event['phi'],
            event['psi'],
            event['stable']
        )
    elif event['type'] == 'regime':
        sensorium.regime_indicator.update(
            event['regime'],
            event['ratio']
        )
```

---

## Use Cases

### 1. **AI Development & Debugging**

Watch AI internal state during development:
- See where processing gets "stuck" (stochastic spikes)
- Identify unstable tokens
- Monitor regime transitions
- Debug φ-field calculations

### 2. **Live Demonstrations**

Show audiences what AI is "thinking":
- Visual explanation of uncertainty
- Real-time processing visualization
- Regime emergence demonstration

### 3. **Research & Analysis**

Study AI behavior:
- Record field dynamics
- Analyze regime patterns
- Compare different inputs
- Measure stability metrics

### 4. **Production Monitoring**

Monitor deployed AI systems:
- System health (FPS, memory)
- Processing load (token rate)
- Regime distribution
- Anomaly detection (unexpected patterns)

---

## Customization

### Color Schemes

Edit colors in `sensorium_hud.py`:

```python
# Default: Cyberpunk
COLOR_PRIMARY = (0, 255, 170)    # Cyan
COLOR_SECONDARY = (255, 60, 100) # Pink
COLOR_WARNING = (255, 170, 0)    # Orange

# Alternative: Matrix
COLOR_PRIMARY = (0, 255, 0)      # Green
COLOR_SECONDARY = (0, 200, 0)    # Dark green
COLOR_WARNING = (255, 255, 0)    # Yellow
```

### Layout

Adjust component positions:

```python
# φ-field position
self.screen.blit(phi_surf, (x, y))

# Resize components
self.phi_field = PhiFieldVisualizer(width, height)
```

### Metrics

Add custom metrics:

```python
self.metrics_panel.update_metric("My Metric", value, "units")
```

---

## Visual Design Philosophy

### Cyberpunk Aesthetic

- **Dark background** - Reduces eye strain
- **High contrast** - Clear visibility
- **Neon accents** - Futuristic feel
- **Grid overlay** - Spatial reference
- **Glowing effects** - Attention guidance

### Information Hierarchy

1. **Primary** - φ-field (largest, center)
2. **Secondary** - Token stream (middle)
3. **Tertiary** - Regime & metrics (sides)
4. **Status** - Bottom bar

### Real-time Feedback

- **60 FPS** - Smooth animation
- **Immediate updates** - No lag
- **Visual continuity** - Smooth transitions
- **Attention-grabbing** - Anomalies highlighted

---

## Performance

### Target Specs

- **FPS:** 60 (locked)
- **Latency:** <16ms per frame
- **CPU:** ~5-10% on modern hardware
- **Memory:** ~100MB

### Optimization

- **Pygame hardware acceleration** (if available)
- **Deque-based** data structures (O(1) operations)
- **Lazy rendering** (only redraw changed areas)
- **Thread pooling** (data generation off main thread)

---

## Future Enhancements

### Planned Features

1. **3D Visualization**
   - OpenGL-based φ-field rendering
   - Rotate/zoom camera controls
   - Depth perception for stability

2. **Audio Feedback**
   - Sonification of φ/ψ waves
   - Regime transition sounds
   - Alert tones for anomalies

3. **Network Monitor**
   - DID network activity (when integrated)
   - Peer connections
   - Data flow visualization

4. **Recording/Playback**
   - Save HUD sessions
   - Replay for analysis
   - Export to video

5. **Multi-Monitor Support**
   - Span across multiple displays
   - Dedicated screens per component
   - Mission control layout

6. **VR/AR Mode**
   - Immersive sensorium
   - 360° field visualization
   - Spatial audio

---

## Philosophy

### Why a Sensorium?

Traditional AI interfaces show:
- Input/output only
- No intermediate state
- No "thought process"

The sensorium reveals:
- **Internal dynamics** - What's happening inside
- **Uncertainty** - Where AI is confident vs uncertain
- **Process** - Step-by-step computation
- **Emergence** - How behavior arises

### AI Transparency

The sensorium makes AI **legible**:
- **Interpretable** - Visual metaphors for abstract math
- **Intuitive** - Colors/motion convey meaning
- **Real-time** - No post-hoc analysis needed
- **Honest** - Shows actual computation, not simplified

### Human-AI Collaboration

The sensorium enables:
- **Awareness** - Human knows AI state
- **Intervention** - Can pause/adjust when needed
- **Trust** - Transparency builds confidence
- **Learning** - Humans understand AI better

---

## Technical Details

### φ-Field Mathematics

The sensorium visualizes:

```
φ = (1 + √5) / 2 ≈ 1.618 (golden ratio)
ψ = (1 - √5) / 2 ≈ -0.618 (conjugate)

φ + ψ = 1
φ × ψ = -1

For token at position n:
  φ-component = Σ φᵏ for active shells
  ψ-component = Σ ψᵏ for active shells
  Δ = |φ - ψ| (stability metric)
```

### Regime Classification

```
Δ < 0.5  → DETERMINISTIC (green)
Δ > 1.5  → STOCHASTIC (pink)
0.5 ≤ Δ ≤ 1.5 → MIXED (orange)
```

### Update Rate

```
Rendering: 60 FPS (16.67ms per frame)
Data generation: 10 Hz (100ms per update)
Queue size: 1000 events (10 seconds buffer)
```

---

## Troubleshooting

### HUD Not Starting

**Issue:** Pygame fails to initialize

**Solution:**
```bash
pip install --upgrade pygame
python -c "import pygame; print(pygame.ver)"
```

### Low FPS

**Issue:** Performance below 60 FPS

**Solutions:**
- Reduce screen resolution
- Disable grid overlay
- Reduce history length (DataStream maxlen)
- Close other applications

### Display Issues

**Issue:** Colors look wrong, elements misaligned

**Solutions:**
- Check display DPI settings
- Update graphics drivers
- Try windowed mode instead of fullscreen

---

## Credits

**Design Philosophy:** Inspired by sci-fi HUDs, aircraft cockpits, and data visualization

**Color Scheme:** Cyberpunk aesthetic (cyan/pink/orange)

**Architecture:** Modular, extensible, real-time

**Purpose:** Make AI internal state visible and understandable

---

## Summary

The ZORDIC Sensorium is a **real-time HUD for AI** that makes internal computational processes visible. It transforms abstract mathematics (φ-field dynamics) into intuitive visual feedback, enabling:

- **Development** - Debug and optimize AI systems
- **Demonstration** - Show how AI works
- **Research** - Study emergence and dynamics
- **Monitoring** - Production system oversight

**Key Innovation:** Treating AI like a piloted system - giving it a "cockpit" where humans can observe what's happening inside.

---

**Run it now:**
```bash
python run_sensorium.py
```

Watch your AI think in real-time. 🌀
