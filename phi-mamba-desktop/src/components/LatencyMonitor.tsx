/**
 * Latency Monitor Component
 *
 * Real-time display of latency metrics
 * Shows RED if exceeding <1ms budget
 */

import { useLatencyMonitor } from '../hooks/useLatencyMonitor';
import './LatencyMonitor.css';

export function LatencyMonitor() {
  const { stats, exceedsBudget } = useLatencyMonitor(100);

  if (!stats) {
    return <div className="latency-monitor loading">Loading metrics...</div>;
  }

  const formatMicros = (us: number) => {
    if (us < 1000) {
      return `${us}μs`;
    } else {
      return `${(us / 1000).toFixed(2)}ms`;
    }
  };

  const getColor = (value: number, threshold: number) => {
    if (value > threshold) return 'red';
    if (value > threshold * 0.8) return 'yellow';
    return 'green';
  };

  return (
    <div className={`latency-monitor ${exceedsBudget ? 'budget-exceeded' : ''}`}>
      <h3>Latency Metrics</h3>

      <div className="metric-grid">
        <div className="metric">
          <div className="metric-label">P50 (Median)</div>
          <div
            className="metric-value"
            style={{ color: getColor(stats.p50_us, 500) }}
          >
            {formatMicros(stats.p50_us)}
          </div>
        </div>

        <div className="metric">
          <div className="metric-label">P95</div>
          <div
            className="metric-value"
            style={{ color: getColor(stats.p95_us, 800) }}
          >
            {formatMicros(stats.p95_us)}
          </div>
        </div>

        <div className="metric">
          <div className="metric-label">P99</div>
          <div
            className="metric-value"
            style={{ color: getColor(stats.p99_us, 1000) }}
          >
            {formatMicros(stats.p99_us)}
          </div>
        </div>

        <div className="metric">
          <div className="metric-label">Max</div>
          <div
            className="metric-value"
            style={{ color: getColor(stats.max_us, 1000) }}
          >
            {formatMicros(stats.max_us)}
          </div>
        </div>
      </div>

      <div className="metric-row">
        <span className="metric-label">Samples:</span>
        <span className="metric-value">{stats.count}</span>
      </div>

      <div className="metric-row">
        <span className="metric-label">Budget Violations:</span>
        <span
          className="metric-value"
          style={{ color: stats.budget_violations > 0 ? 'red' : 'green' }}
        >
          {stats.budget_violations} ({(stats.violation_rate * 100).toFixed(2)}%)
        </span>
      </div>

      <div className="budget-indicator">
        <div className="budget-label">LATENCY BUDGET: &lt;1ms</div>
        <div className={`budget-status ${exceedsBudget ? 'exceeded' : 'ok'}`}>
          {exceedsBudget ? '❌ EXCEEDED' : '✓ OK'}
        </div>
      </div>
    </div>
  );
}
