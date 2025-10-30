# Tauri Desktop GUI with Holographic Overlay

## Overview

Desktop app with real-time trade signal overlay and holographic field visualization.

```
┌──────────────────────────────────────────────────────────────┐
│  Phi-Mamba Trade Signals - Desktop App                       │
├──────────────────────────────────────────────────────────────┤
│  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │   Price Chart          │  │  Signal Panel             │  │
│  │   with Overlays        │  │                           │  │
│  │                        │  │  ● AAPL: STRONG BUY (0.9) │  │
│  │   [Candlesticks]       │  │  ● MSFT: BUY (0.7)        │  │
│  │   [Berry Phase Lines]  │  │  ● GOOGL: HOLD (0.5)      │  │
│  │   [Phi Levels]         │  │  ● JPM: SELL (0.3)        │  │
│  └────────────────────────┘  └───────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Holographic Field View (WebGL)                         ││
│  │                                                          ││
│  │  [Live 3D visualization of phi-field]                   ││
│  │  [Color = coherence, Height = energy]                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌──────────────┬──────────────┬──────────────┬───────────┐│
│  │ Field:  0.73 │ Berry: 2.14  │ Signals: 15  │ Active: 3 ││
│  └──────────────┴──────────────┴──────────────┴───────────┘│
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
tauri-app/
├── src-tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── build.rs
│   └── src/
│       ├── main.rs              # Tauri entry point
│       ├── commands.rs          # IPC commands
│       ├── state.rs             # App state management
│       └── signals.rs           # Signal generation
├── src/
│   ├── App.tsx                  # Main React component
│   ├── main.tsx                 # React entry
│   ├── components/
│   │   ├── Chart.tsx            # Trading chart
│   │   ├── FieldView.tsx        # WebGL holographic field
│   │   ├── SignalPanel.tsx      # Signal list
│   │   └── StatusBar.tsx        # Status indicators
│   ├── hooks/
│   │   ├── useSignals.ts        # Signal subscription
│   │   ├── useFieldState.ts     # Field state hook
│   │   └── useMarketData.ts     # Market data hook
│   ├── shaders/
│   │   ├── field.vert.glsl      # Vertex shader
│   │   └── field.frag.glsl      # Fragment shader
│   └── wasm/
│       └── phi_mamba.ts         # WASM wrapper
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Tauri Backend (Rust)

### src-tauri/Cargo.toml

```toml
[package]
name = "phi-mamba-gui"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "1.5", features = [] }

[dependencies]
tauri = { version = "1.5", features = ["shell-open"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
phi-mamba-trade-signals = { path = "../../" }

# Real-time data
tokio-tungstenite = "0.20"  # WebSocket
```

### src-tauri/src/main.rs

```rust
// Prevents additional console window on Windows in release
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod commands;
mod state;
mod signals;

use tauri::Manager;
use state::AppState;

fn main() {
    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::encode_bar,
            commands::compute_signals,
            commands::get_field_state,
            commands::phi_multiply,
        ])
        .setup(|app| {
            // Initialize background tasks
            let app_handle = app.app_handle();
            tauri::async_runtime::spawn(async move {
                signals::run_signal_generator(app_handle).await;
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### src-tauri/src/commands.rs

```rust
//! Tauri IPC commands

use tauri::State;
use phi_mamba_trade_signals::encoding::financial::{FinancialEncoder, OHLCVBar};
use phi_mamba_trade_signals::cordic::CordicEngine;
use crate::state::AppState;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct EncodedState {
    pub ticker: String,
    pub theta: f64,
    pub energy: f64,
    pub zeckendorf: Vec<u32>,
}

/// Encode OHLCV bar to phi-space
#[tauri::command]
pub async fn encode_bar(
    ticker: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    position: usize,
) -> Result<EncodedState, String> {
    let encoder = FinancialEncoder::new();

    let bar = OHLCVBar {
        timestamp: 0,
        open,
        high,
        low,
        close,
        volume,
        ticker: ticker.clone(),
    };

    let state = encoder.encode_bar(&bar, position);

    Ok(EncodedState {
        ticker,
        theta: state.theta_total.to_num::<f64>(),
        energy: state.energy.to_num::<f64>(),
        zeckendorf: state.zeckendorf,
    })
}

#[derive(Serialize)]
pub struct TradeSignal {
    pub ticker: String,
    pub signal: String,  // "BUY", "SELL", "HOLD"
    pub strength: f64,
    pub confidence: f64,
    pub expected_return: f64,
}

/// Compute trade signals for all tickers
#[tauri::command]
pub async fn compute_signals(
    state: State<'_, AppState>,
) -> Result<Vec<TradeSignal>, String> {
    // Get current field state
    let field = state.get_field_state().await;

    // Generate signals using CORDIC + N-game theory
    let signals = field.generate_signals();

    Ok(signals.into_iter().map(|s| TradeSignal {
        ticker: s.ticker,
        signal: format!("{:?}", s.signal_type),
        strength: s.strength,
        confidence: s.confidence,
        expected_return: s.expected_return,
    }).collect())
}

#[derive(Serialize)]
pub struct FieldState {
    pub coherence: f64,
    pub berry_phase: f64,
    pub active_tickers: usize,
    pub timestamp: i64,
}

/// Get current holographic field state
#[tauri::command]
pub async fn get_field_state(
    state: State<'_, AppState>,
) -> Result<FieldState, String> {
    let field = state.get_field_state().await;

    Ok(FieldState {
        coherence: field.coherence,
        berry_phase: field.berry_phase,
        active_tickers: field.tickers.len(),
        timestamp: field.timestamp,
    })
}

/// Phi-space multiplication (ADDITION ONLY!)
#[tauri::command]
pub fn phi_multiply(n: i32, m: i32) -> i32 {
    CordicEngine::phi_multiply_exp(n, m)  // Just addition!
}
```

### src-tauri/src/signals.rs

```rust
//! Background signal generation

use tauri::{AppHandle, Manager};
use tokio::time::{interval, Duration};
use phi_mamba_trade_signals::signals::SignalGenerator;

/// Run continuous signal generation in background
pub async fn run_signal_generator(app: AppHandle) {
    let mut interval = interval(Duration::from_secs(1));
    let signal_gen = SignalGenerator::new();

    loop {
        interval.tick().await;

        // Generate signals
        match signal_gen.compute_latest_signals().await {
            Ok(signals) => {
                // Emit to frontend
                app.emit_all("signals-updated", &signals).ok();
            }
            Err(e) => {
                eprintln!("Error generating signals: {}", e);
            }
        }
    }
}
```

## React Frontend

### src/App.tsx

```typescript
import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import Chart from './components/Chart';
import FieldView from './components/FieldView';
import SignalPanel from './components/SignalPanel';
import StatusBar from './components/StatusBar';
import './App.css';

interface TradeSignal {
  ticker: string;
  signal: string;
  strength: number;
  confidence: number;
  expected_return: number;
}

interface FieldState {
  coherence: number;
  berry_phase: number;
  active_tickers: number;
  timestamp: number;
}

function App() {
  const [signals, setSignals] = useState<TradeSignal[]>([]);
  const [fieldState, setFieldState] = useState<FieldState | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>('AAPL');

  useEffect(() => {
    // Listen for signal updates from backend
    const unlisten = listen<TradeSignal[]>('signals-updated', (event) => {
      setSignals(event.payload);
    });

    // Poll field state
    const interval = setInterval(async () => {
      const state = await invoke<FieldState>('get_field_state');
      setFieldState(state);
    }, 1000);

    return () => {
      unlisten.then(f => f());
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="app">
      <div className="main-grid">
        <div className="chart-container">
          <Chart ticker={selectedTicker} />
        </div>

        <div className="signal-panel">
          <SignalPanel
            signals={signals}
            onSelectTicker={setSelectedTicker}
          />
        </div>

        <div className="field-view">
          <FieldView fieldState={fieldState} />
        </div>
      </div>

      <StatusBar fieldState={fieldState} signalCount={signals.length} />
    </div>
  );
}

export default App;
```

### src/components/FieldView.tsx (WebGL Holographic Visualization)

```typescript
import { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface Props {
  fieldState: FieldState | null;
}

export default function FieldView({ fieldState }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Create scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    containerRef.current.appendChild(renderer.domElement);

    // Create holographic field geometry
    const geometry = new THREE.PlaneGeometry(10, 10, 100, 100);

    // Custom shader for phi-field visualization
    const material = new THREE.ShaderMaterial({
      uniforms: {
        u_time: { value: 0.0 },
        u_coherence: { value: 0.0 },
        u_berry_phase: { value: 0.0 },
        u_phi: { value: 1.618034 },
      },
      vertexShader: `
        varying vec2 v_uv;
        varying float v_height;

        uniform float u_time;
        uniform float u_coherence;
        uniform float u_phi;

        void main() {
          v_uv = uv;

          // Phi-based wave displacement
          float wave = sin(uv.x * u_phi * 10.0 + u_time)
                     * cos(uv.y * u_phi * 10.0 - u_time);

          // Height based on coherence
          v_height = wave * u_coherence;

          vec3 newPosition = position;
          newPosition.z = v_height * 2.0;

          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 v_uv;
        varying float v_height;

        uniform float u_berry_phase;

        void main() {
          // Color based on height and Berry phase
          float r = 0.5 + 0.5 * v_height;
          float g = 0.5 + 0.5 * sin(u_berry_phase);
          float b = 0.5 + 0.5 * cos(u_berry_phase);

          gl_FragColor = vec4(r, g, b, 0.8);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    sceneRef.current = scene;
    rendererRef.current = renderer;

    // Animation loop
    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      // Update time uniform
      material.uniforms.u_time.value += 0.01;

      // Update field state
      if (fieldState) {
        material.uniforms.u_coherence.value = fieldState.coherence;
        material.uniforms.u_berry_phase.value = fieldState.berry_phase;
      }

      // Rotate for 3D effect
      mesh.rotation.x += 0.001;
      mesh.rotation.y += 0.002;

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      renderer.dispose();
    };
  }, [fieldState]);

  return (
    <div ref={containerRef} className="field-view-container" />
  );
}
```

### src/components/SignalPanel.tsx

```typescript
interface Props {
  signals: TradeSignal[];
  onSelectTicker: (ticker: string) => void;
}

export default function SignalPanel({ signals, onSelectTicker }: Props) {
  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'STRONG_BUY': return '#00ff00';
      case 'BUY': return '#66ff66';
      case 'HOLD': return '#ffff00';
      case 'SELL': return '#ff6666';
      case 'STRONG_SELL': return '#ff0000';
      default: return '#ffffff';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'STRONG_BUY': return '⬆⬆';
      case 'BUY': return '⬆';
      case 'HOLD': return '→';
      case 'SELL': return '⬇';
      case 'STRONG_SELL': return '⬇⬇';
      default: return '?';
    }
  };

  return (
    <div className="signal-panel">
      <h2>Trade Signals</h2>
      <div className="signal-list">
        {signals.map(signal => (
          <div
            key={signal.ticker}
            className="signal-item"
            onClick={() => onSelectTicker(signal.ticker)}
            style={{ borderLeft: `4px solid ${getSignalColor(signal.signal)}` }}
          >
            <div className="signal-header">
              <span className="ticker">{signal.ticker}</span>
              <span className="icon">{getSignalIcon(signal.signal)}</span>
            </div>
            <div className="signal-details">
              <div>
                <span className="label">Signal:</span>
                <span className="value">{signal.signal}</span>
              </div>
              <div>
                <span className="label">Strength:</span>
                <span className="value">{(signal.strength * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="label">Confidence:</span>
                <span className="value">{(signal.confidence * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="label">Expected Return:</span>
                <span className="value" style={{
                  color: signal.expected_return > 0 ? '#00ff00' : '#ff0000'
                }}>
                  {signal.expected_return > 0 ? '+' : ''}
                  {signal.expected_return.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Build & Run

```bash
# Install dependencies
cd tauri-app
npm install

# Development mode
npm run tauri dev

# Build production
npm run tauri build

# Output locations:
# macOS: src-tauri/target/release/bundle/macos/
# Windows: src-tauri/target/release/bundle/msi/
# Linux: src-tauri/target/release/bundle/appimage/
```

## Features Demonstrated

1. **Real-time Signal Updates**: Background task generates signals, pushes to frontend
2. **WebGL Visualization**: Holographic field rendered with Three.js
3. **Phi-space Computation**: All CORDIC operations run in Rust backend
4. **Cross-platform**: Works on Windows, macOS, Linux
5. **Native Performance**: Rust backend, no Electron overhead
6. **IPC Communication**: Efficient Tauri commands for frontend <-> backend

## Next: Holographic Memory Layer

See `HOLOGRAPHIC_MEMORY.md` for DID-based distributed state management.
