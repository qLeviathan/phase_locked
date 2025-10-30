/**
 * Latency Monitoring Hook
 *
 * Real-time latency statistics with <1ms target
 */

import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

export interface LatencyStats {
  count: number;
  mean_us: number;
  min_us: number;
  max_us: number;
  p50_us: number;
  p95_us: number;
  p99_us: number;
  budget_violations: number;
  violation_rate: number;
}

const LATENCY_BUDGET_US = 1000; // 1ms target

export function useLatencyMonitor(pollInterval: number = 100) {
  const [stats, setStats] = useState<LatencyStats | null>(null);
  const [exceedsBudget, setExceedsBudget] = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const latencyStats = await invoke<LatencyStats>('get_latency_stats');
        setStats(latencyStats);
        setExceedsBudget(latencyStats.p99_us > LATENCY_BUDGET_US);
      } catch (error) {
        console.error('Failed to fetch latency stats:', error);
      }
    }, pollInterval);

    return () => clearInterval(interval);
  }, [pollInterval]);

  return { stats, exceedsBudget };
}

/**
 * Measure round-trip latency (frontend → Rust → frontend)
 */
export async function measureRoundTripLatency(): Promise<number> {
  const start = performance.now();
  await invoke('ping');
  const end = performance.now();
  return end - start;
}
