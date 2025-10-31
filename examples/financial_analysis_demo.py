#!/usr/bin/env python3
"""
Phi-Mamba Financial Analysis Demo

Demonstrates the complete financial analysis system:
1. Multi-ticker data injection
2. Multiple timeframe analysis
3. Economic indicator integration
4. Forecasting at multiple horizons
5. Expected utility decision making
6. Opportunity screening
7. Nash equilibrium computation
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phi_mamba.financial_system import create_default_system
from phi_mamba.financial_data import Timeframe
from phi_mamba.financial_forecast import ForecastHorizon


def main():
    print("=" * 80)
    print("PHI-MAMBA FINANCIAL ANALYSIS SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print()

    # Create system with default parameters
    print("Initializing Phi-Mamba Financial System...")
    system = create_default_system(
        risk_aversion=1.2,  # Slightly risk-averse
        wealth=100000.0,    # $100k portfolio
        lookback=50         # 50 bars of history
    )
    print("✓ System initialized")
    print()

    # =========================================================================
    # STEP 1: Inject multiple tickers with synthetic data
    # =========================================================================
    print("STEP 1: Injecting ticker data into the field")
    print("-" * 80)

    tickers = {
        'AAPL': {'base_price': 150.0, 'volatility': 0.025, 'trend': 0.001},   # Tech growth
        'MSFT': {'base_price': 300.0, 'volatility': 0.022, 'trend': 0.0008},  # Tech stable
        'GOOGL': {'base_price': 120.0, 'volatility': 0.028, 'trend': 0.0012}, # Tech aggressive
        'JPM': {'base_price': 140.0, 'volatility': 0.018, 'trend': 0.0003},   # Finance stable
        'BAC': {'base_price': 30.0, 'volatility': 0.020, 'trend': 0.0002},    # Finance value
        'XOM': {'base_price': 110.0, 'volatility': 0.030, 'trend': -0.0002},  # Energy decline
        'CVX': {'base_price': 155.0, 'volatility': 0.028, 'trend': -0.0001},  # Energy flat
        'JNJ': {'base_price': 160.0, 'volatility': 0.012, 'trend': 0.0001},   # Healthcare defensive
        'PFE': {'base_price': 28.0, 'volatility': 0.025, 'trend': 0.0005},    # Healthcare growth
        'TSLA': {'base_price': 200.0, 'volatility': 0.045, 'trend': 0.002},   # High volatility growth
    }

    for ticker, params in tickers.items():
        system.inject_synthetic_ticker(
            ticker=ticker,
            timeframe=Timeframe.DAY_1,
            n_bars=252,  # 1 year of daily data
            base_price=params['base_price'],
            volatility=params['volatility'],
            trend=params['trend']
        )
        print(f"  ✓ Injected {ticker}: base=${params['base_price']:.2f}, vol={params['volatility']:.1%}, trend={params['trend']:.4f}")

    print()
    print(f"Total active tickers: {len(system.get_active_tickers())}")
    print()

    # =========================================================================
    # STEP 2: Add economic indicators
    # =========================================================================
    print("STEP 2: Adding economic indicators to the field")
    print("-" * 80)

    economic_data = [
        ('Fed_Funds_Rate', 5.25, 'rates', 5),
        ('CPI_YoY', 3.2, 'inflation', 5),
        ('Unemployment_Rate', 3.8, 'employment', 4),
        ('GDP_Growth', 2.1, 'growth', 4),
        ('Consumer_Sentiment', 68.5, 'sentiment', 3),
        ('VIX', 18.5, 'volatility', 3),
    ]

    for name, value, category, importance in economic_data:
        system.add_economic_indicator(name, value, category, importance=importance)
        print(f"  ✓ Added {name}: {value} ({category}, importance={importance})")

    print()

    # =========================================================================
    # STEP 3: Analyze field coherence
    # =========================================================================
    print("STEP 3: Analyzing field phase coherence")
    print("-" * 80)

    field_coherence = system.compute_field_coherence(Timeframe.DAY_1, lookback=20)
    print(f"  Field Coherence: {field_coherence:.4f}")
    print(f"  Interpretation: {'High synchronization' if field_coherence > 0.7 else 'Low synchronization' if field_coherence < 0.3 else 'Moderate synchronization'}")
    print()

    # Find phase-locked pairs
    print("  Phase-Locked Ticker Pairs (highly correlated):")
    locked_pairs = system.find_phase_locked_pairs(Timeframe.DAY_1, threshold=0.3)
    for ticker1, ticker2, berry_phase in locked_pairs[:5]:
        print(f"    {ticker1} <-> {ticker2}: Berry phase = {berry_phase:.4f}")
    print()

    # =========================================================================
    # STEP 4: Generate forecasts for multiple horizons
    # =========================================================================
    print("STEP 4: Generating multi-horizon forecasts")
    print("-" * 80)

    # Example: forecast AAPL at multiple horizons
    example_ticker = 'AAPL'
    print(f"  Forecasting {example_ticker} at multiple horizons:")
    print()

    horizons_forecasts = system.forecast_ticker_multiple_horizons(
        example_ticker, Timeframe.DAY_1
    )

    for horizon, forecast in horizons_forecasts.items():
        print(f"    {horizon.value} Horizon:")
        print(f"      Current Price: ${forecast.current_price:.2f}")
        print(f"      Forecast Close: ${forecast.forecast_close:.2f}")
        print(f"      Expected Return: {forecast.expected_return:+.2f}%")
        print(f"      Probability Up: {forecast.probability_up:.1%}")
        print(f"      Confidence: {forecast.confidence:.2f}")
        print(f"      Risk Score: {forecast.risk_score:.4f}")
        print()

    # =========================================================================
    # STEP 5: Complete field analysis
    # =========================================================================
    print("STEP 5: Performing complete field analysis")
    print("-" * 80)

    results = system.analyze(
        timeframe=Timeframe.DAY_1,
        horizon=ForecastHorizon.WEEK_1,  # 1-week forecast
        top_n=10,
        min_return=0.5,  # Minimum 0.5% expected return
        current_portfolio={},  # Starting with no positions
        cash=100000.0
    )

    print()
    system.print_analysis_summary(results)

    # =========================================================================
    # STEP 6: Screen opportunities by different criteria
    # =========================================================================
    print("STEP 6: Screening opportunities by Sharpe ratio")
    print("-" * 80)

    sharpe_opportunities = system.screener.screen_by_sharpe(
        results.field_forecast,
        top_n=5,
        min_sharpe=0.5
    )

    print(f"{'Rank':<6} {'Ticker':<10} {'Return%':<10} {'Risk':<10} {'Sharpe':<10} {'Conf':<8}")
    print("-" * 80)

    for i, (ticker, eu) in enumerate(sharpe_opportunities, 1):
        print(f"{i:<6} {ticker:<10} {eu.expected_return:<10.2f} {eu.risk:<10.4f} "
              f"{eu.sharpe_ratio:<10.2f} {eu.confidence:<8.2f}")

    print()

    # =========================================================================
    # STEP 7: Correlation analysis
    # =========================================================================
    print("STEP 7: Correlation matrix analysis")
    print("-" * 80)

    corr_matrix = system.analyze_correlation(Timeframe.DAY_1, lookback=20)
    active_tickers = system.get_active_tickers()

    print("  Correlation Matrix (top 5x5):")
    print(f"  {'Ticker':<10}", end='')
    for ticker in active_tickers[:5]:
        print(f"{ticker:>10}", end='')
    print()
    print("  " + "-" * 60)

    for i, ticker_i in enumerate(active_tickers[:5]):
        print(f"  {ticker_i:<10}", end='')
        for j, ticker_j in enumerate(active_tickers[:5]):
            if i < len(corr_matrix) and j < len(corr_matrix[i]):
                print(f"{corr_matrix[i][j]:>10.3f}", end='')
            else:
                print(f"{'N/A':>10}", end='')
        print()

    print()

    # =========================================================================
    # STEP 8: Export results
    # =========================================================================
    print("STEP 8: Exporting forecast data")
    print("-" * 80)

    forecast_data = system.export_forecasts_to_dict(results.field_forecast)
    print(f"  Exported {len(forecast_data)} forecasts to dictionary format")
    print(f"  Sample forecast (first ticker):")
    if len(forecast_data) > 0:
        sample = forecast_data[0]
        for key, value in sample.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    print()

    # =========================================================================
    # STEP 9: Dynamic ticker injection/removal
    # =========================================================================
    print("STEP 9: Demonstrating dynamic ticker management")
    print("-" * 80)

    # Add a new ticker
    print("  Adding new ticker: NVDA")
    system.inject_synthetic_ticker(
        ticker='NVDA',
        timeframe=Timeframe.DAY_1,
        n_bars=252,
        base_price=450.0,
        volatility=0.040,
        trend=0.003
    )
    print(f"  Active tickers: {system.get_active_tickers()}")
    print()

    # Remove a ticker
    print("  Removing ticker: XOM")
    system.remove_ticker('XOM')
    print(f"  Active tickers: {system.get_active_tickers()}")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Multi-ticker data injection (synthetic)")
    print("  ✓ Economic indicator integration")
    print("  ✓ Field phase coherence analysis")
    print("  ✓ Phase-locked pair detection")
    print("  ✓ Multi-horizon forecasting (1d, 1w, 1M)")
    print("  ✓ Expected utility decision framework")
    print("  ✓ Opportunity screening by utility and Sharpe ratio")
    print("  ✓ Nash equilibrium computation")
    print("  ✓ Correlation matrix analysis")
    print("  ✓ Dynamic ticker injection/removal")
    print("  ✓ Data export capabilities")
    print()
    print("Next Steps:")
    print("  - Replace synthetic data with real market data (CSV or API)")
    print("  - Integrate with live data feeds")
    print("  - Implement backtesting framework")
    print("  - Build visualization dashboard")
    print("  - Add portfolio optimization")
    print()


if __name__ == "__main__":
    main()
