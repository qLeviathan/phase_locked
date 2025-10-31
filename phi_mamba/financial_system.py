"""
Phi-Mamba Financial Analysis System

Main integration module that combines:
- Data loading
- Phi-space encoding
- Forecasting
- Decision making
- Opportunity screening

Provides high-level API for financial analysis using phi-mamba.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .financial_data import (
    FinancialDataLoader, MarketField, Timeframe,
    OHLCVBar, TickerData, EconomicIndicator
)
from .financial_encoding import (
    FinancialPhiEncoder, FinancialTokenState,
    compute_price_momentum_angle, analyze_field_topology
)
from .financial_forecast import (
    PhiMambaForecaster, PriceForecast, FieldForecast,
    ForecastHorizon, forecast_multiple_horizons
)
from .decision_framework import (
    DecisionMaker, MultiTickerGameSolver, OpportunityScreener,
    UtilityParameters, GameState, NashEquilibrium, ExpectedUtility
)


@dataclass
class AnalysisResults:
    """Complete analysis results"""
    field_forecast: FieldForecast
    top_opportunities: List[Tuple[str, ExpectedUtility]]
    nash_equilibrium: NashEquilibrium
    field_metrics: Dict[str, float]
    field_topology: Dict[str, any]
    timestamp: datetime = field(default_factory=datetime.now)


class PhiMambaFinancialSystem:
    """
    Complete financial analysis system using Phi-Mamba

    Integrates all components for end-to-end analysis:
    1. Load multi-ticker data
    2. Encode to phi-space
    3. Generate forecasts
    4. Apply decision framework
    5. Screen opportunities
    """

    def __init__(
        self,
        lookback: int = 50,
        n_monte_carlo: int = 100,
        utility_params: Optional[UtilityParameters] = None
    ):
        """
        Initialize financial system

        Args:
            lookback: Historical bars for encoding
            n_monte_carlo: Monte Carlo simulations for forecasting
            utility_params: Utility function parameters
        """
        # Initialize components
        self.data_loader = FinancialDataLoader()
        self.encoder = FinancialPhiEncoder()
        self.forecaster = PhiMambaForecaster(
            encoder=self.encoder,
            lookback=lookback,
            n_monte_carlo=n_monte_carlo
        )

        # Decision making
        if utility_params is None:
            utility_params = UtilityParameters()
        self.decision_maker = DecisionMaker(utility_params)
        self.game_solver = MultiTickerGameSolver(self.decision_maker)
        self.screener = OpportunityScreener(self.decision_maker)

    def inject_ticker(
        self,
        ticker: str,
        timeframe: Timeframe,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ):
        """
        Inject ticker data into the system

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            data: Nx5 OHLCV array
            timestamps: Optional timestamps
        """
        self.data_loader.load_ticker_data(ticker, timeframe, data, timestamps)

    def inject_ticker_csv(
        self,
        ticker: str,
        filepath: str,
        timeframe: Timeframe,
        date_column: str = "Date",
        date_format: str = "%Y-%m-%d"
    ):
        """
        Inject ticker from CSV file

        Args:
            ticker: Ticker symbol
            filepath: Path to CSV file
            timeframe: Data timeframe
            date_column: Name of date column
            date_format: Date format string
        """
        self.data_loader.load_csv(ticker, filepath, timeframe, date_column, date_format)

    def inject_synthetic_ticker(
        self,
        ticker: str,
        timeframe: Timeframe,
        n_bars: int = 252,
        base_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0005
    ):
        """
        Inject synthetic ticker for testing

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            n_bars: Number of bars
            base_price: Starting price
            volatility: Price volatility
            trend: Trend per bar
        """
        self.data_loader.generate_synthetic_data(
            ticker, timeframe, n_bars, base_price, volatility, trend
        )

    def add_economic_indicator(
        self,
        name: str,
        value: float,
        category: str,
        timestamp: Optional[datetime] = None,
        importance: int = 3
    ):
        """
        Add economic indicator to field

        Args:
            name: Indicator name
            value: Indicator value
            category: Category (e.g., "inflation", "rates")
            timestamp: Timestamp
            importance: Importance (1-5)
        """
        self.data_loader.add_economic_indicator(
            name, value, category, timestamp, importance
        )

    def remove_ticker(self, ticker: str):
        """Remove ticker from field"""
        self.data_loader.market_field.remove_ticker(ticker)

    def get_active_tickers(self) -> List[str]:
        """Get list of active tickers"""
        return self.data_loader.market_field.get_active_tickers()

    def analyze(
        self,
        timeframe: Timeframe,
        horizon: ForecastHorizon,
        top_n: int = 10,
        min_return: float = 1.0,
        current_portfolio: Optional[Dict[str, float]] = None,
        cash: float = 100000.0
    ) -> AnalysisResults:
        """
        Complete analysis of the market field

        Args:
            timeframe: Data timeframe to analyze
            horizon: Forecast horizon
            top_n: Number of top opportunities to return
            min_return: Minimum expected return threshold
            current_portfolio: Current positions {ticker: quantity}
            cash: Available cash

        Returns:
            AnalysisResults with complete analysis
        """
        market_field = self.data_loader.get_field()

        # 1. Generate field forecast
        field_forecast = self.forecaster.forecast_field(
            market_field, timeframe, horizon
        )

        # 2. Screen for top opportunities
        top_opportunities = self.screener.screen_opportunities(
            field_forecast, top_n, min_return
        )

        # 3. Solve multi-ticker game for Nash equilibrium
        nash_equilibrium = self.game_solver.solve_field_game(
            field_forecast, current_portfolio, cash
        )

        # 4. Compute field metrics
        field_metrics = self.screener.compute_field_metrics(field_forecast)

        # 5. Analyze field topology
        encoded_field = self.encoder.encode_field(market_field, timeframe)
        field_topology = analyze_field_topology(encoded_field)

        return AnalysisResults(
            field_forecast=field_forecast,
            top_opportunities=top_opportunities,
            nash_equilibrium=nash_equilibrium,
            field_metrics=field_metrics,
            field_topology=field_topology
        )

    def forecast_ticker(
        self,
        ticker: str,
        timeframe: Timeframe,
        horizon: ForecastHorizon
    ) -> Optional[PriceForecast]:
        """
        Forecast single ticker

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            horizon: Forecast horizon

        Returns:
            PriceForecast or None if ticker not found
        """
        market_field = self.data_loader.get_field()

        if ticker not in market_field.tickers:
            return None

        ticker_data = market_field.tickers[ticker]
        return self.forecaster.forecast_ticker(ticker_data, timeframe, horizon)

    def forecast_ticker_multiple_horizons(
        self,
        ticker: str,
        timeframe: Timeframe
    ) -> Dict[ForecastHorizon, PriceForecast]:
        """
        Forecast ticker at multiple horizons

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe

        Returns:
            Dict mapping horizon -> forecast
        """
        market_field = self.data_loader.get_field()

        if ticker not in market_field.tickers:
            return {}

        ticker_data = market_field.tickers[ticker]
        return forecast_multiple_horizons(self.forecaster, ticker_data, timeframe)

    def analyze_correlation(
        self,
        timeframe: Timeframe,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Compute correlation matrix across tickers

        Args:
            timeframe: Timeframe
            lookback: Lookback period

        Returns:
            NxN correlation matrix
        """
        market_field = self.data_loader.get_field()
        return market_field.get_correlation_matrix(timeframe, lookback)

    def find_phase_locked_pairs(
        self,
        timeframe: Timeframe,
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Find ticker pairs that are phase-locked (correlated)

        Args:
            timeframe: Timeframe
            threshold: Berry phase threshold

        Returns:
            List of (ticker1, ticker2, berry_phase) tuples
        """
        market_field = self.data_loader.get_field()
        encoded_field = self.encoder.encode_field(market_field, timeframe)
        return self.encoder.find_phase_locked_pairs(encoded_field, threshold)

    def compute_field_coherence(
        self,
        timeframe: Timeframe,
        lookback: int = 20
    ) -> float:
        """
        Compute overall field coherence

        Args:
            timeframe: Timeframe
            lookback: Lookback period

        Returns:
            Coherence score [0, 1]
        """
        market_field = self.data_loader.get_field()
        encoded_field = self.encoder.encode_field(market_field, timeframe)
        return self.encoder.compute_field_coherence(encoded_field, lookback)

    def print_analysis_summary(self, results: AnalysisResults):
        """
        Print formatted analysis summary

        Args:
            results: AnalysisResults object
        """
        print("=" * 80)
        print("PHI-MAMBA FINANCIAL ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {results.timestamp}")
        print(f"Forecast Horizon: {results.field_forecast.horizon.value}")
        print()

        print("FIELD METRICS:")
        print("-" * 80)
        for metric, value in results.field_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print()

        print(f"TOP {len(results.top_opportunities)} OPPORTUNITIES (by Expected Utility):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Ticker':<10} {'Return%':<10} {'Risk':<10} {'Sharpe':<10} {'Utility':<12} {'Conf':<8}")
        print("-" * 80)

        for i, (ticker, eu) in enumerate(results.top_opportunities, 1):
            print(f"{i:<6} {ticker:<10} {eu.expected_return:<10.2f} {eu.risk:<10.4f} "
                  f"{eu.sharpe_ratio:<10.2f} {eu.utility:<12.4f} {eu.confidence:<8.2f}")
        print()

        print("NASH EQUILIBRIUM ALLOCATION:")
        print("-" * 80)
        print(f"Total Utility: {results.nash_equilibrium.total_utility:.4f}")
        print(f"Strategy Type: {'Pure' if results.nash_equilibrium.is_pure_strategy else 'Mixed'}")
        print()
        print(f"{'Ticker':<10} {'Action':<10} {'Quantity':<12} {'Price':<10}")
        print("-" * 80)

        for ticker, action in results.nash_equilibrium.optimal_actions.items():
            print(f"{ticker:<10} {action.action.value:<10} {action.quantity:<12.2f} {action.price:<10.2f}")
        print()

        print("FIELD TOPOLOGY:")
        print("-" * 80)
        for key, value in results.field_topology.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}: {value[:5]}...")  # Print first 5 items
            else:
                print(f"  {key}: {value}")
        print()

        print("=" * 80)

    def export_forecasts_to_dict(
        self,
        field_forecast: FieldForecast
    ) -> List[Dict[str, any]]:
        """
        Export forecasts to list of dictionaries (for JSON/CSV export)

        Args:
            field_forecast: Field forecast

        Returns:
            List of forecast dictionaries
        """
        forecasts_list = []

        for ticker, forecast in field_forecast.forecasts.items():
            forecasts_list.append({
                'ticker': ticker,
                'horizon': forecast.horizon.value,
                'current_price': forecast.current_price,
                'forecast_open': forecast.forecast_open,
                'forecast_close': forecast.forecast_close,
                'forecast_high': forecast.forecast_high,
                'forecast_low': forecast.forecast_low,
                'expected_return': forecast.expected_return,
                'probability_up': forecast.probability_up,
                'confidence': forecast.confidence,
                'risk_score': forecast.risk_score,
                'phase_coherence': forecast.phase_coherence,
                'timestamp': forecast.timestamp.isoformat()
            })

        return forecasts_list

    def clear_all_data(self):
        """Clear all loaded data"""
        self.data_loader.clear()


def create_default_system(
    risk_aversion: float = 1.0,
    wealth: float = 100000.0,
    lookback: int = 50
) -> PhiMambaFinancialSystem:
    """
    Create a default financial system with standard parameters

    Args:
        risk_aversion: Risk aversion parameter (1.0 = risk-neutral)
        wealth: Total wealth for position sizing
        lookback: Historical lookback period

    Returns:
        PhiMambaFinancialSystem instance
    """
    utility_params = UtilityParameters(
        risk_aversion=risk_aversion,
        wealth=wealth,
        max_position_size=0.2,
        transaction_cost=0.001,
        time_preference=0.95
    )

    return PhiMambaFinancialSystem(
        lookback=lookback,
        n_monte_carlo=100,
        utility_params=utility_params
    )
