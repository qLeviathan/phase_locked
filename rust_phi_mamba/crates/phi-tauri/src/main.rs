//! Tauri application for Phi-Mamba
//! 
//! Desktop application providing a visual interface for exploring
//! the game-theoretic language modeling framework

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use phi_core::{zeckendorf_decomposition, Fibonacci, PhiInt, TokenState, PHI, PSI};
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Mutex;

/// Application state
struct AppState {
    fibonacci: Mutex<Fibonacci>,
}

/// Response for Fibonacci calculation
#[derive(Serialize)]
struct FibonacciResponse {
    index: usize,
    value: String,
}

/// Response for Zeckendorf decomposition
#[derive(Serialize)]
struct ZeckendorfResponse {
    number: u64,
    decomposition: Vec<usize>,
    fibonacci_values: Vec<u64>,
}

/// Response for golden ratio calculation
#[derive(Serialize)]
struct PhiCalculation {
    expression: String,
    const_term: i64,
    phi_coeff: i64,
    approximate_value: f64,
}

/// Calculate Fibonacci number
#[tauri::command]
fn calculate_fibonacci(
    index: usize,
    state: State<AppState>,
) -> Result<FibonacciResponse, String> {
    let mut fib = state.fibonacci.lock().map_err(|e| e.to_string())?;
    
    if index > 93 {
        return Err("Fibonacci index too large for u64".to_string());
    }
    
    let value = fib.get(index);
    Ok(FibonacciResponse {
        index,
        value: value.to_string(),
    })
}

/// Perform Zeckendorf decomposition
#[tauri::command]
fn decompose_zeckendorf(number: u64) -> Result<ZeckendorfResponse, String> {
    let decomposition = zeckendorf_decomposition(number)
        .map_err(|e| e.to_string())?;
    
    let mut fib = Fibonacci::new();
    let fibonacci_values: Vec<u64> = decomposition
        .iter()
        .map(|&idx| fib.get(idx))
        .collect();
    
    Ok(ZeckendorfResponse {
        number,
        decomposition,
        fibonacci_values,
    })
}

/// Calculate golden ratio arithmetic
#[tauri::command]
fn calculate_phi_arithmetic(
    operation: String,
    a_const: i64,
    a_phi: i64,
    b_const: i64,
    b_phi: i64,
) -> Result<PhiCalculation, String> {
    let a = PhiInt::new(a_const, a_phi);
    let b = PhiInt::new(b_const, b_phi);
    
    let result = match operation.as_str() {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        _ => return Err("Invalid operation".to_string()),
    };
    
    let approximate = result.const_term as f64 + result.phi_coeff as f64 * PHI;
    
    let expression = if result.phi_coeff == 0 {
        result.const_term.to_string()
    } else if result.const_term == 0 {
        format!("{}φ", result.phi_coeff)
    } else {
        format!("{} + {}φ", result.const_term, result.phi_coeff)
    };
    
    Ok(PhiCalculation {
        expression,
        const_term: result.const_term,
        phi_coeff: result.phi_coeff,
        approximate_value: approximate,
    })
}

/// Get golden ratio constants
#[tauri::command]
fn get_constants() -> serde_json::Value {
    serde_json::json!({
        "phi": PHI,
        "psi": PSI,
        "phi_squared": PHI * PHI,
        "phi_inverse": 1.0 / PHI,
    })
}

fn main() {
    let app_state = AppState {
        fibonacci: Mutex::new(Fibonacci::new()),
    };
    
    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            calculate_fibonacci,
            decompose_zeckendorf,
            calculate_phi_arithmetic,
            get_constants,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}