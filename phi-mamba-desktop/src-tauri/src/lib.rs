//! Phi-Mamba Desktop - Low-Latency Trade Signal Generator
//!
//! **LATENCY BUDGET: <1ms end-to-end**

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod encoder;
mod metrics;
mod shared_memory;
mod websocket;

use crossbeam::queue::ArrayQueue;
use metrics::LatencyTracker;
use phi_mamba_signals::OHLCVBar;
use shared_memory::SharedBuffer;
use std::sync::Arc;
use tracing_subscriber;

/// Initialize tracing
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .init();
}

/// Tauri command: Ping for latency measurement
#[tauri::command]
async fn ping() -> Result<i64, String> {
    Ok(chrono::Utc::now().timestamp_micros())
}

/// Tauri command: Get latency stats
#[tauri::command]
async fn get_latency_stats(
    tracker: tauri::State<'_, Arc<LatencyTracker>>,
) -> Result<metrics::LatencyStats, String> {
    Ok(tracker.stats(Some(1000)))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_tracing();

    tracing::info!("═══════════════════════════════════════════════════════");
    tracing::info!(" Phi-Mamba Desktop - Low-Latency Signal Generator");
    tracing::info!(" Target: <1ms end-to-end latency");
    tracing::info!("═══════════════════════════════════════════════════════");

    // Create shared state
    let input_queue = Arc::new(ArrayQueue::<OHLCVBar>::new(4096));
    let shared_buffer = Arc::new(SharedBuffer::new());
    let metrics_tracker = Arc::new(LatencyTracker::new());

    // Clone for each thread
    let input_queue_encoder = input_queue.clone();
    let shared_buffer_encoder = shared_buffer.clone();
    let metrics_encoder = metrics_tracker.clone();

    let shared_buffer_berry = shared_buffer.clone();
    let metrics_berry = metrics_tracker.clone();

    let input_queue_feed = input_queue.clone();

    tracing::info!("Starting background tasks...");

    // Start CORDIC encoding loop
    tokio::spawn(async move {
        let encoder = encoder::Encoder::new(
            input_queue_encoder,
            shared_buffer_encoder,
            metrics_encoder,
        );
        encoder.run().await;
    });

    // Start Berry phase computation loop
    tokio::spawn(async move {
        let berry_computer = encoder::BerryComputer::new(
            shared_buffer_berry,
            metrics_berry,
        );
        berry_computer.run().await;
    });

    // Start simulated feed (100Hz by default)
    tokio::spawn(async move {
        let tickers = vec![
            "AAPL".to_string(),
            "GOOGL".to_string(),
            "MSFT".to_string(),
            "TSLA".to_string(),
            "NVDA".to_string(),
        ];

        let feed = websocket::SimulatedFeed::new(
            input_queue_feed,
            tickers.clone(),
            100, // 100 ticks/sec
        );

        tracing::info!("Starting simulated feed for: {:?}", tickers);
        feed.run().await;
    });

    tracing::info!("All background tasks started");

    // Build Tauri app
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(input_queue)
        .manage(shared_buffer.clone())
        .manage(metrics_tracker)
        .invoke_handler(tauri::generate_handler![
            ping,
            get_latency_stats,
            shared_memory::get_states,
            shared_memory::get_signals,
            shared_memory::get_berry_matrix,
            shared_memory::get_buffer_stats,
            encoder::get_encoder_stats,
            websocket::start_simulated_feed,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
