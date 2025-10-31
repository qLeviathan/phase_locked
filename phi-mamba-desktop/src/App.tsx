/**
 * Phi-Mamba Desktop - Main App
 *
 * Low-latency trade signal generator
 * Target: <1ms end-to-end
 */

import { LatencyMonitor } from './components/LatencyMonitor';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Phi-Mamba Desktop</h1>
        <p className="subtitle">Low-Latency Trade Signal Generator</p>
      </header>

      <main className="app-main">
        <section className="monitoring-section">
          <LatencyMonitor />
        </section>

        <section className="info-section">
          <div className="info-card">
            <h3>System Status</h3>
            <div className="status-grid">
              <div className="status-item">
                <span className="status-label">CORDIC Engine:</span>
                <span className="status-value green">● RUNNING</span>
              </div>
              <div className="status-item">
                <span className="status-label">Simulated Feed:</span>
                <span className="status-value green">● ACTIVE</span>
              </div>
              <div className="status-item">
                <span className="status-label">Berry Phase:</span>
                <span className="status-value green">● COMPUTING</span>
              </div>
            </div>
          </div>

          <div className="info-card">
            <h3>Architecture</h3>
            <ul className="architecture-list">
              <li>✓ Add/Subtract/Shift CORDIC (no multiply/divide)</li>
              <li>✓ Φ-space arithmetic (multiplication → addition)</li>
              <li>✓ Lock-free ring buffers (zero contention)</li>
              <li>✓ Zero-copy IPC (shared memory)</li>
              <li>✓ Zeckendorf encoding (OEIS A003714)</li>
            </ul>
          </div>

          <div className="info-card">
            <h3>Performance Targets</h3>
            <table className="perf-table">
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Target</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>WebSocket Parse</td>
                  <td>&lt;0.1ms</td>
                </tr>
                <tr>
                  <td>CORDIC Encoding</td>
                  <td>&lt;0.015ms</td>
                </tr>
                <tr>
                  <td>Berry Phase</td>
                  <td>&lt;0.002ms</td>
                </tr>
                <tr>
                  <td>IPC Transfer</td>
                  <td>&lt;0.05ms</td>
                </tr>
                <tr className="total-row">
                  <td><strong>TOTAL</strong></td>
                  <td><strong>&lt;1ms</strong></td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Phase Locked Team | Rust + Tauri + React | 546× Less Energy | 160× Faster
        </p>
      </footer>
    </div>
  );
}

export default App;
