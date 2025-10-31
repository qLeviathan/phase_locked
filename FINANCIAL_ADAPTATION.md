# Phi-Mamba Financial Analysis System

## Overview

This document describes the adaptation of the Phi-Mamba model for financial time series analysis, forecasting, and trading decision-making. The system leverages the golden ratio (φ) and phase-locking dynamics to analyze market behavior and make probabilistic forecasts.

## Architecture

The financial adaptation consists of several integrated modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Phi-Mamba Financial System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Data Layer  │  │ Encoding     │  │ Forecasting  │          │
│  │              │──>│ Layer        │──>│ Layer        │          │
│  │              │  │              │  │              │          │
│  │ Multi-Ticker │  │ Phi-Space    │  │ Phase-Locked │          │
│  │ OHLCV Data   │  │ TokenStates  │  │ Prediction   │          │
│  │              │  │              │  │              │          │
│  │ Economic     │  │ Field        │  │ Multi-Horizon│          │
│  │ Indicators   │  │ Representation│ │ Forecasts    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                             │                    │
│                                             ▼                    │
│  ┌──────────────────────────────────────────────────┐          │
│  │         Decision Framework (Game Theory)          │          │
│  ├──────────────────────────────────────────────────┤          │
│  │  • Expected Utility Maximization                  │          │
│  │  • N-Game Nash Equilibrium                        │          │
│  │  • Backward Induction                             │          │
│  │  • Opportunity Screening                          │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Financial Data Module (`financial_data.py`)

**Purpose**: Ingests and manages multi-ticker financial data with economic indicators.

**Key Features**:
- Multi-ticker support with dynamic injection/removal
- Multiple timeframes (1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w, 1M)
- OHLCV (Open, High, Low, Close, Volume) data structure
- Economic indicator integration
- Field-based representation (all tickers as a unified "field")
- Correlation matrix computation
- Technical indicator calculation

**Classes**:
- `Timeframe`: Enum for supported timeframes
- `OHLCVBar`: Single price bar with timestamp
- `EconomicIndicator`: Economic data point
- `TickerData`: Complete ticker data across timeframes
- `MarketField`: Multi-ticker field representation
- `FinancialDataLoader`: Data loading and management

**Example Usage**:
```python
from phi_mamba.financial_data import FinancialDataLoader, Timeframe

loader = FinancialDataLoader()

# Load ticker from CSV
loader.load_csv('AAPL', 'aapl_data.csv', Timeframe.DAY_1)

# Inject economic indicator
loader.add_economic_indicator(
    name='Fed_Funds_Rate',
    value=5.25,
    category='rates',
    importance=5
)

# Get market field
field = loader.get_field()
tickers = field.get_active_tickers()
```

### 2. Financial Encoding Module (`financial_encoding.py`)

**Purpose**: Converts financial time series into phi-space TokenStates.

**Key Mappings**:
- **Price movements** → Angular positions (θ)
- **Volume** → Energy levels
- **Time** → Position in sequence
- **Multi-ticker field** → Coupled TokenStates

**Encoding Process**:

1. **Price to Angle**:
   ```
   θ_price = (price_change_% × sensitivity × φ) mod 2π
   ```

2. **Position to Angle** (RoPE-like):
   ```
   θ_pos = (position × φ^(-position/10)) mod 2π
   ```

3. **Volume to Energy**:
   ```
   E = (volume / volume_scale) × φ^(-position)
   ```

4. **Zeckendorf Decomposition**:
   - Price integer → Non-consecutive Fibonacci sum
   - Creates topological structure

**Classes**:
- `FinancialTokenState`: Extended TokenState with price/volume
- `FinancialPhiEncoder`: Main encoder class

**Key Functions**:
- `encode_bar()`: Encode single OHLCV bar
- `encode_sequence()`: Encode time series
- `encode_field()`: Encode entire multi-ticker field
- `compute_field_coherence()`: Measure phase alignment
- `find_phase_locked_pairs()`: Find correlated tickers

**Example Usage**:
```python
from phi_mamba.financial_encoding import FinancialPhiEncoder

encoder = FinancialPhiEncoder()

# Encode ticker data
states = encoder.encode_ticker_data(ticker_data, Timeframe.DAY_1)

# Compute field coherence (0-1, higher = more synchronized)
coherence = encoder.compute_field_coherence(encoded_field)

# Find phase-locked pairs (correlated tickers)
pairs = encoder.find_phase_locked_pairs(encoded_field, threshold=0.3)
```

### 3. Financial Forecasting Module (`financial_forecast.py`)

**Purpose**: Forecasts price movements using phase-locking dynamics.

**Forecast Horizons**:
- **1 Day**: Next day open/close
- **1 Week**: 5-day forecast
- **1 Month**: 21-day forecast

**Forecasting Method**:

1. **Phase-Locked Generation**:
   - Generate future states that minimize Berry phase
   - Uses Monte Carlo simulation for uncertainty

2. **Retrocausal Constraints**:
   - Future states influence past (backward induction)
   - Improves coherence

3. **Probability Estimation**:
   - P(up) from future state transitions
   - Confidence from phase coherence

**Classes**:
- `ForecastHorizon`: Enum for time horizons
- `PriceForecast`: Single ticker forecast
- `FieldForecast`: Multi-ticker forecast
- `PhiMambaForecaster`: Main forecasting engine

**Forecast Attributes**:
- `forecast_open`: Forecast opening price
- `forecast_close`: Forecast closing price
- `forecast_high/low`: Range forecast
- `expected_return`: Expected return %
- `probability_up`: Probability of price increase
- `confidence`: Forecast confidence [0, 1]
- `risk_score`: Volatility estimate
- `phase_coherence`: Phase-locking strength

**Example Usage**:
```python
from phi_mamba.financial_forecast import PhiMambaForecaster, ForecastHorizon

forecaster = PhiMambaForecaster(lookback=50, n_monte_carlo=100)

# Forecast single ticker
forecast = forecaster.forecast_ticker(
    ticker_data,
    Timeframe.DAY_1,
    ForecastHorizon.WEEK_1
)

print(f"Expected Return: {forecast.expected_return:.2f}%")
print(f"Probability Up: {forecast.probability_up:.1%}")
print(f"Confidence: {forecast.confidence:.2f}")

# Forecast entire field
field_forecast = forecaster.forecast_field(
    market_field,
    Timeframe.DAY_1,
    ForecastHorizon.DAY_1
)
```

### 4. Decision Framework Module (`decision_framework.py`)

**Purpose**: N-game decision-making using expected utility theory.

**Core Concepts**:

1. **Expected Utility**:
   ```
   EU = Σ p_i × U(wealth_i)
   ```
   Where U is a utility function (CRRA, CARA, log, or quadratic)

2. **Nash Equilibrium**:
   - Finds optimal allocation across all tickers
   - No single ticker can improve by unilateral deviation

3. **Backward Induction**:
   - Multi-stage game solving
   - Future optimal actions constrain present

4. **Opportunity Screening**:
   - Rank by expected utility
   - Filter by Sharpe ratio
   - Threshold by confidence

**Utility Functions**:
- **CRRA** (Constant Relative Risk Aversion): U(w) = w^(1-γ) / (1-γ)
- **CARA** (Constant Absolute Risk Aversion): U(w) = -exp(-α·w)
- **Logarithmic**: U(w) = log(w)
- **Quadratic**: U(w) = w - (γ/2)·w²

**Classes**:
- `Action`: Enum (BUY, SELL, HOLD, SHORT, COVER)
- `UtilityParameters`: Risk aversion, wealth, position limits
- `GameState`: State in trading game
- `GameAction`: Trading action
- `ExpectedUtility`: Expected utility of action
- `NashEquilibrium`: Optimal multi-ticker allocation
- `UtilityFunction`: Utility computation
- `DecisionMaker`: Action evaluation
- `MultiTickerGameSolver`: Nash equilibrium solver
- `OpportunityScreener`: Opportunity filtering

**Example Usage**:
```python
from phi_mamba.decision_framework import (
    DecisionMaker, UtilityParameters, OpportunityScreener
)

# Create decision maker
params = UtilityParameters(
    risk_aversion=1.2,  # Slightly risk-averse
    wealth=100000.0,
    max_position_size=0.2,  # Max 20% per position
    transaction_cost=0.001
)
decision_maker = DecisionMaker(params, utility_type='CRRA')

# Screen opportunities
screener = OpportunityScreener(decision_maker)
top_opportunities = screener.screen_opportunities(
    field_forecast,
    top_n=10,
    min_return=1.0  # Min 1% expected return
)

for ticker, eu in top_opportunities:
    print(f"{ticker}: Return={eu.expected_return:.2f}%, "
          f"Utility={eu.utility:.4f}, Sharpe={eu.sharpe_ratio:.2f}")
```

### 5. Financial System Module (`financial_system.py`)

**Purpose**: High-level API integrating all components.

**Main Class**: `PhiMambaFinancialSystem`

**Workflow**:

1. **Data Injection**:
   ```python
   system = create_default_system(risk_aversion=1.0, wealth=100000)

   # Inject tickers
   system.inject_ticker_csv('AAPL', 'aapl.csv', Timeframe.DAY_1)
   system.inject_synthetic_ticker('TSLA', Timeframe.DAY_1)

   # Add economic data
   system.add_economic_indicator('CPI_YoY', 3.2, 'inflation')
   ```

2. **Analysis**:
   ```python
   results = system.analyze(
       timeframe=Timeframe.DAY_1,
       horizon=ForecastHorizon.WEEK_1,
       top_n=10,
       min_return=1.0
   )
   ```

3. **Results**:
   - `field_forecast`: Forecasts for all tickers
   - `top_opportunities`: Top N trades by utility
   - `nash_equilibrium`: Optimal allocation
   - `field_metrics`: Aggregate statistics
   - `field_topology`: Topological analysis

4. **Utilities**:
   - `print_analysis_summary()`: Pretty-print results
   - `export_forecasts_to_dict()`: Export to JSON/CSV
   - `find_phase_locked_pairs()`: Correlation detection
   - `compute_field_coherence()`: Synchronization measure

**Example Usage**:
```python
from phi_mamba.financial_system import create_default_system
from phi_mamba.financial_data import Timeframe
from phi_mamba.financial_forecast import ForecastHorizon

# Create system
system = create_default_system(risk_aversion=1.2, wealth=100000)

# Load data
for ticker in ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TSLA']:
    system.inject_synthetic_ticker(ticker, Timeframe.DAY_1, n_bars=252)

# Analyze
results = system.analyze(
    timeframe=Timeframe.DAY_1,
    horizon=ForecastHorizon.WEEK_1,
    top_n=5
)

# Print summary
system.print_analysis_summary(results)
```

## Theoretical Foundation

### Phase-Locking and Market Synchronization

Markets exhibit phase-locking behavior where assets move in synchrony:

- **High Coherence** (φ ≈ 0): Assets move together (risk-on/risk-off)
- **Low Coherence** (φ ≈ π): Decorrelated movement (diversification)

### Berry Phase for Transitions

The Berry phase between states determines transition probability:

```
Berry_Phase = √[(Δθ)² + α·(shell_overlap)² + β·(Δposition)²]
```

Phase-locked transitions (Berry ≈ 0 mod 2π) are more probable.

### Expected Utility Maximization

For each action `a` in state `s`:

```
EU(a, s) = Σ p(outcome_i | a, s) × U(wealth_i)
```

Optimal action: `a* = argmax EU(a, s)`

### Nash Equilibrium for Portfolio

For N assets, find allocation (w₁, w₂, ..., wₙ) such that:

```
No single w_i can be changed to improve total utility
```

Solved using pure strategy Nash equilibrium.

## Key Features

### 1. Dynamic Multi-Ticker Support

Add/remove tickers dynamically without recomputing the entire field:

```python
system.inject_ticker('NVDA', Timeframe.DAY_1, data)
system.remove_ticker('XOM')
```

### 2. Multi-Timeframe Analysis

Analyze the same tickers at different scales:

```python
# Daily analysis
results_daily = system.analyze(timeframe=Timeframe.DAY_1, ...)

# Hourly analysis
results_hourly = system.analyze(timeframe=Timeframe.HOUR_1, ...)

# Weekly analysis
results_weekly = system.analyze(timeframe=Timeframe.WEEK_1, ...)
```

### 3. Economic Indicator Integration

Incorporate macro data into the field:

```python
system.add_economic_indicator('Fed_Funds_Rate', 5.25, 'rates', importance=5)
system.add_economic_indicator('CPI_YoY', 3.2, 'inflation', importance=5)
system.add_economic_indicator('VIX', 18.5, 'volatility', importance=3)
```

### 4. Multi-Horizon Forecasting

Forecast at multiple time horizons simultaneously:

```python
forecasts = system.forecast_ticker_multiple_horizons('AAPL', Timeframe.DAY_1)

day_forecast = forecasts[ForecastHorizon.DAY_1]
week_forecast = forecasts[ForecastHorizon.WEEK_1]
month_forecast = forecasts[ForecastHorizon.MONTH_1]
```

### 5. Probability Screening

Screen for high-probability setups:

```python
# By expected utility
opportunities = screener.screen_opportunities(
    field_forecast,
    top_n=10,
    min_return=2.0  # Min 2% expected return
)

# By Sharpe ratio
sharpe_opportunities = screener.screen_by_sharpe(
    field_forecast,
    top_n=10,
    min_sharpe=1.0  # Min Sharpe ratio of 1.0
)
```

### 6. Phase-Locked Pair Detection

Find correlated assets (similar to pairs trading):

```python
pairs = system.find_phase_locked_pairs(Timeframe.DAY_1, threshold=0.3)

for ticker1, ticker2, berry_phase in pairs:
    print(f"{ticker1} <-> {ticker2}: Berry phase = {berry_phase:.4f}")
```

## Usage Examples

### Basic Workflow

```python
from phi_mamba.financial_system import create_default_system
from phi_mamba.financial_data import Timeframe
from phi_mamba.financial_forecast import ForecastHorizon

# 1. Create system
system = create_default_system(risk_aversion=1.0, wealth=100000)

# 2. Inject data
system.inject_ticker_csv('AAPL', 'data/aapl.csv', Timeframe.DAY_1)
system.inject_ticker_csv('MSFT', 'data/msft.csv', Timeframe.DAY_1)

# 3. Add economic data
system.add_economic_indicator('CPI_YoY', 3.2, 'inflation')

# 4. Analyze
results = system.analyze(
    timeframe=Timeframe.DAY_1,
    horizon=ForecastHorizon.WEEK_1,
    top_n=5
)

# 5. View results
system.print_analysis_summary(results)

# 6. Get top opportunity
if len(results.top_opportunities) > 0:
    ticker, eu = results.top_opportunities[0]
    print(f"Best opportunity: {ticker}")
    print(f"Expected return: {eu.expected_return:.2f}%")
    print(f"Confidence: {eu.confidence:.2f}")
```

### Advanced: Custom Utility Function

```python
from phi_mamba.decision_framework import (
    DecisionMaker, UtilityParameters
)

# Create custom utility parameters
params = UtilityParameters(
    risk_aversion=1.5,  # More risk-averse
    wealth=250000.0,
    max_position_size=0.15,  # Max 15% per position
    transaction_cost=0.0005,  # 5 bps transaction cost
    time_preference=0.98  # Higher time preference
)

# Use CARA utility instead of default CRRA
decision_maker = DecisionMaker(params, utility_type='CARA')

# Create system with custom decision maker
system = PhiMambaFinancialSystem(
    lookback=100,
    n_monte_carlo=200,
    utility_params=params
)
```

### Advanced: Backtesting Framework

```python
# Pseudo-code for backtesting
for date in backtest_dates:
    # Update data up to date
    system.inject_ticker_data_up_to(ticker, date)

    # Generate forecast
    forecast = system.forecast_ticker(ticker, Timeframe.DAY_1, ForecastHorizon.DAY_1)

    # Make decision
    if forecast.expected_return > 1.0 and forecast.confidence > 0.7:
        # Execute trade
        execute_trade('BUY', ticker, quantity)

    # Track performance
    record_performance(date, forecast, actual_return)
```

## Performance Considerations

### Computational Complexity

- **Data Ingestion**: O(N) for N bars
- **Encoding**: O(N × M) for N bars, M tickers
- **Forecasting**: O(N × K) for N steps, K Monte Carlo samples
- **Decision Making**: O(M²) for M tickers (Nash equilibrium)

### Optimization Tips

1. **Limit lookback**: Use 50-100 bars instead of all history
2. **Reduce Monte Carlo samples**: 50-100 samples often sufficient
3. **Batch processing**: Process multiple tickers in parallel
4. **Cache encoded states**: Reuse encodings when possible

## Future Enhancements

### Planned Features

1. **Live Data Integration**: Real-time API feeds (Yahoo Finance, Alpha Vantage)
2. **Backtesting Framework**: Historical performance testing
3. **Portfolio Optimization**: Multi-period optimization
4. **Visualization Dashboard**: Interactive charts and graphs
5. **Model Training**: Learn coupling matrix from data
6. **Risk Management**: Position sizing, stop-loss, drawdown limits
7. **Alternative Data**: Sentiment, options flow, insider trading

### Research Directions

1. **Adaptive φ**: Learn optimal golden ratio variant per market regime
2. **Deep Integration**: Combine with neural networks
3. **High-Frequency**: Extend to sub-second timeframes
4. **Options Pricing**: Apply to derivatives
5. **Market Regimes**: Detect regime shifts using topology

## References

1. Main Phi-Mamba paper: `arxiv_preprint.tex`
2. Mathematical foundations: `docs/math_foundations.md`
3. Game theory validation: `game_theory_validation.py`
4. Implementation details: `docs/implementation.md`

## Contact and Support

For questions or issues:
- GitHub Issues: https://github.com/[your-repo]/issues
- Documentation: See `docs/` directory
- Examples: See `examples/` directory

---

**Note**: This is a research prototype. Do not use for actual trading without thorough validation and risk management. Financial markets are complex and past performance does not guarantee future results.
