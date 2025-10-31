//! WebSocket Feed Handler
//!
//! Receives real-time market data with <0.1ms latency
//! Lock-free queue for zero-contention

use crate::metrics::LatencyTimer;
use crossbeam::queue::ArrayQueue;
use futures_util::{SinkExt, StreamExt};
use phi_mamba_signals::OHLCVBar;
use std::sync::Arc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// WebSocket connection configuration
#[derive(Clone, Debug)]
pub struct WebSocketConfig {
    pub url: String,
    pub tickers: Vec<String>,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            url: "ws://localhost:8080".to_string(),
            tickers: vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()],
        }
    }
}

/// WebSocket handler
pub struct WebSocketHandler {
    config: WebSocketConfig,
    output_queue: Arc<ArrayQueue<OHLCVBar>>,
}

impl WebSocketHandler {
    pub fn new(config: WebSocketConfig, output_queue: Arc<ArrayQueue<OHLCVBar>>) -> Self {
        Self {
            config,
            output_queue,
        }
    }

    /// Connect and run WebSocket loop
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Connecting to WebSocket: {}", self.config.url);

        let (ws_stream, _) = connect_async(&self.config.url).await?;
        tracing::info!("WebSocket connected");

        let (mut write, mut read) = ws_stream.split();

        // Subscribe to tickers
        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "tickers": self.config.tickers,
        });

        write
            .send(Message::Text(subscribe_msg.to_string()))
            .await?;

        tracing::info!("Subscribed to tickers: {:?}", self.config.tickers);

        // Receive loop
        while let Some(msg) = read.next().await {
            let timer = LatencyTimer::new("websocket_parse");

            match msg {
                Ok(Message::Text(text)) => {
                    // Parse JSON message
                    if let Ok(bar) = parse_message(&text) {
                        // Push to lock-free queue
                        if self.output_queue.push(bar).is_err() {
                            tracing::warn!("Input queue full, dropping tick");
                        }
                    }
                }
                Ok(Message::Binary(data)) => {
                    // Binary format for lower latency
                    if let Ok(bar) = parse_binary(&data) {
                        if self.output_queue.push(bar).is_err() {
                            tracing::warn!("Input queue full, dropping tick");
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }

            let elapsed = timer.elapsed_us();
            if elapsed > 100 {
                // >100μs is slow for parsing
                tracing::warn!("Slow WebSocket parse: {}μs", elapsed);
            }
        }

        tracing::warn!("WebSocket disconnected");
        Ok(())
    }
}

/// Parse JSON message to OHLCV bar
fn parse_message(text: &str) -> Result<OHLCVBar, serde_json::Error> {
    #[derive(serde::Deserialize)]
    struct TickMessage {
        ticker: String,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
    }

    let msg: TickMessage = serde_json::from_str(text)?;

    Ok(OHLCVBar {
        timestamp: msg.timestamp,
        ticker: msg.ticker,
        open: msg.open,
        high: msg.high,
        low: msg.low,
        close: msg.close,
        volume: msg.volume,
    })
}

/// Parse binary message to OHLCV bar (for ultra-low latency)
///
/// Binary format (56 bytes):
/// - ticker (16 bytes, null-padded)
/// - timestamp (8 bytes, i64)
/// - open (8 bytes, f64)
/// - high (8 bytes, f64)
/// - low (8 bytes, f64)
/// - close (8 bytes, f64)
/// - volume (8 bytes, u64)
fn parse_binary(data: &[u8]) -> Result<OHLCVBar, Box<dyn std::error::Error>> {
    if data.len() < 56 {
        return Err("Binary message too short".into());
    }

    // Parse ticker (16 bytes)
    let ticker_bytes = &data[0..16];
    let ticker = std::str::from_utf8(ticker_bytes)?
        .trim_end_matches('\0')
        .to_string();

    // Parse fields (8 bytes each)
    let timestamp = i64::from_le_bytes(data[16..24].try_into()?);
    let open = f64::from_le_bytes(data[24..32].try_into()?);
    let high = f64::from_le_bytes(data[32..40].try_into()?);
    let low = f64::from_le_bytes(data[40..48].try_into()?);
    let close = f64::from_le_bytes(data[48..56].try_into()?);
    let volume = u64::from_le_bytes(data[56..64].try_into()?);

    Ok(OHLCVBar {
        timestamp,
        ticker,
        open,
        high,
        low,
        close,
        volume,
    })
}

/// Simulated feed for testing (generates synthetic ticks)
pub struct SimulatedFeed {
    output_queue: Arc<ArrayQueue<OHLCVBar>>,
    tickers: Vec<String>,
    tick_rate_hz: u64,
}

impl SimulatedFeed {
    pub fn new(output_queue: Arc<ArrayQueue<OHLCVBar>>, tickers: Vec<String>, tick_rate_hz: u64) -> Self {
        Self {
            output_queue,
            tickers,
            tick_rate_hz,
        }
    }

    /// Run simulated feed
    pub async fn run(self) {
        tracing::info!("Starting simulated feed at {}Hz", self.tick_rate_hz);

        let interval_us = 1_000_000 / self.tick_rate_hz;
        let mut interval = tokio::time::interval(tokio::time::Duration::from_micros(interval_us));

        let mut prices: std::collections::HashMap<String, f64> = self
            .tickers
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), 100.0 + i as f64 * 50.0))
            .collect();

        loop {
            interval.tick().await;

            for ticker in &self.tickers {
                let price = prices.get_mut(ticker).unwrap();

                // Random walk
                let change = (rand::random::<f64>() - 0.5) * 2.0;
                *price += change;
                *price = price.max(1.0); // Don't go negative

                let bar = OHLCVBar {
                    timestamp: chrono::Utc::now().timestamp_micros(),
                    ticker: ticker.clone(),
                    open: *price - 0.5,
                    high: *price + 0.5,
                    low: *price - 0.7,
                    close: *price,
                    volume: (rand::random::<u64>() % 10_000_000) + 1_000_000,
                };

                if self.output_queue.push(bar).is_err() {
                    tracing::warn!("Queue full, dropping simulated tick");
                }
            }
        }
    }
}

/// Tauri command to start simulated feed
#[tauri::command]
pub async fn start_simulated_feed(
    queue: tauri::State<'_, Arc<ArrayQueue<OHLCVBar>>>,
    tickers: Vec<String>,
    tick_rate_hz: u64,
) -> Result<(), String> {
    let feed = SimulatedFeed::new(queue.inner().clone(), tickers, tick_rate_hz);

    tokio::spawn(async move {
        feed.run().await;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_message() {
        let json = r#"{
            "ticker": "AAPL",
            "timestamp": 1700000000,
            "open": 180.0,
            "high": 182.0,
            "low": 179.0,
            "close": 181.5,
            "volume": 50000000
        }"#;

        let bar = parse_message(json).unwrap();

        assert_eq!(bar.ticker, "AAPL");
        assert_eq!(bar.open, 180.0);
        assert_eq!(bar.close, 181.5);
    }

    #[test]
    fn test_parse_binary() {
        let mut data = vec![0u8; 64];

        // Ticker "AAPL"
        data[0..4].copy_from_slice(b"AAPL");

        // Timestamp
        data[16..24].copy_from_slice(&1700000000i64.to_le_bytes());

        // Prices
        data[24..32].copy_from_slice(&180.0f64.to_le_bytes());
        data[32..40].copy_from_slice(&182.0f64.to_le_bytes());
        data[40..48].copy_from_slice(&179.0f64.to_le_bytes());
        data[48..56].copy_from_slice(&181.5f64.to_le_bytes());

        // Volume
        data[56..64].copy_from_slice(&50_000_000u64.to_le_bytes());

        let bar = parse_binary(&data).unwrap();

        assert_eq!(bar.ticker, "AAPL");
        assert_eq!(bar.open, 180.0);
        assert_eq!(bar.close, 181.5);
        assert_eq!(bar.volume, 50_000_000);
    }

    #[tokio::test]
    async fn test_simulated_feed() {
        let queue = Arc::new(ArrayQueue::new(1024));
        let tickers = vec!["AAPL".to_string()];

        let feed = SimulatedFeed::new(queue.clone(), tickers, 1000);

        // Run for short time
        tokio::select! {
            _ = feed.run() => {},
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {},
        }

        // Should have received ticks
        assert!(!queue.is_empty());
    }
}
