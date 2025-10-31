"""
Financial Data Ingestion Module for Phi-Mamba

Supports multi-ticker data loading with:
- Multiple timeframes (1min, 5min, 15min, 1h, 1d, 1w, 1M)
- OHLCV data structure
- Economic indicators integration
- Dynamic ticker injection
- Field-based multi-asset representation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import warnings


class Timeframe(Enum):
    """Supported timeframes for financial data"""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes"""
        mapping = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
            "1M": 43200  # Approximate
        }
        return mapping[self.value]


@dataclass
class OHLCVBar:
    """Single OHLCV bar with timestamp"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: str
    timeframe: Timeframe

    @property
    def price_range(self) -> float:
        """High - Low"""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Close - Open (can be negative)"""
        return self.close - self.open

    @property
    def body_pct(self) -> float:
        """Body as percentage of open"""
        if self.open == 0:
            return 0.0
        return (self.body / self.open) * 100.0

    @property
    def is_bullish(self) -> bool:
        """Close > Open"""
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        """Distance from top of body to high"""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Distance from low to bottom of body"""
        return min(self.open, self.close) - self.low


@dataclass
class EconomicIndicator:
    """Economic indicator data point"""
    timestamp: datetime
    name: str
    value: float
    category: str  # e.g., "inflation", "employment", "sentiment", "rates"
    importance: int = 3  # 1-5 scale
    unit: str = ""


@dataclass
class TickerData:
    """Complete ticker data with multiple timeframes"""
    ticker: str
    bars: Dict[Timeframe, List[OHLCVBar]] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def add_bar(self, bar: OHLCVBar):
        """Add a bar to the appropriate timeframe"""
        if bar.timeframe not in self.bars:
            self.bars[bar.timeframe] = []
        self.bars[bar.timeframe].append(bar)

    def get_latest(self, timeframe: Timeframe) -> Optional[OHLCVBar]:
        """Get most recent bar for timeframe"""
        if timeframe not in self.bars or len(self.bars[timeframe]) == 0:
            return None
        return self.bars[timeframe][-1]

    def get_range(self, timeframe: Timeframe, start: datetime, end: datetime) -> List[OHLCVBar]:
        """Get bars within time range"""
        if timeframe not in self.bars:
            return []
        return [bar for bar in self.bars[timeframe]
                if start <= bar.timestamp <= end]


@dataclass
class MarketField:
    """
    Multi-ticker field representation
    Enables the 'field' to update as tickers are injected
    """
    tickers: Dict[str, TickerData] = field(default_factory=dict)
    economic_indicators: List[EconomicIndicator] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def inject_ticker(self, ticker: str, data: Optional[TickerData] = None):
        """
        Dynamically inject a ticker into the field
        Field updates accordingly
        """
        if data is None:
            data = TickerData(ticker=ticker)
        self.tickers[ticker] = data

    def remove_ticker(self, ticker: str):
        """Remove ticker from field"""
        if ticker in self.tickers:
            del self.tickers[ticker]

    def add_economic_indicator(self, indicator: EconomicIndicator):
        """Add economic indicator to field"""
        self.economic_indicators.append(indicator)

    def get_active_tickers(self) -> List[str]:
        """Get list of active ticker symbols"""
        return list(self.tickers.keys())

    def get_field_state(self, timeframe: Timeframe) -> Dict[str, OHLCVBar]:
        """
        Get current field state across all tickers at timeframe
        Returns dict of ticker -> latest bar
        """
        return {
            ticker: data.get_latest(timeframe)
            for ticker, data in self.tickers.items()
            if data.get_latest(timeframe) is not None
        }

    def get_correlation_matrix(self, timeframe: Timeframe, lookback: int = 20) -> np.ndarray:
        """
        Compute correlation matrix of close prices across tickers

        Args:
            timeframe: Which timeframe to analyze
            lookback: Number of bars to use

        Returns:
            NxN correlation matrix for N tickers
        """
        tickers = self.get_active_tickers()
        n = len(tickers)

        if n == 0:
            return np.array([])

        # Collect close prices
        price_matrix = []
        valid_tickers = []

        for ticker in tickers:
            bars = self.tickers[ticker].bars.get(timeframe, [])
            if len(bars) >= lookback:
                closes = [bar.close for bar in bars[-lookback:]]
                price_matrix.append(closes)
                valid_tickers.append(ticker)

        if len(price_matrix) < 2:
            return np.eye(len(valid_tickers))

        price_array = np.array(price_matrix)
        correlation = np.corrcoef(price_array)

        return correlation


class FinancialDataLoader:
    """
    Loads and manages financial data from various sources
    """

    def __init__(self):
        self.market_field = MarketField()

    def load_ticker_data(
        self,
        ticker: str,
        timeframe: Timeframe,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ):
        """
        Load ticker data from numpy array

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            data: Nx5 array [open, high, low, close, volume]
            timestamps: Optional timestamps (generated if not provided)
        """
        if ticker not in self.market_field.tickers:
            self.market_field.inject_ticker(ticker)

        n_bars = len(data)

        # Generate timestamps if not provided
        if timestamps is None:
            end_time = datetime.now()
            delta = timedelta(minutes=timeframe.minutes)
            timestamps = [end_time - delta * (n_bars - i - 1) for i in range(n_bars)]

        # Create OHLCV bars
        for i, (ts, ohlcv) in enumerate(zip(timestamps, data)):
            bar = OHLCVBar(
                timestamp=ts,
                open=float(ohlcv[0]),
                high=float(ohlcv[1]),
                low=float(ohlcv[2]),
                close=float(ohlcv[3]),
                volume=float(ohlcv[4]),
                ticker=ticker,
                timeframe=timeframe
            )
            self.market_field.tickers[ticker].add_bar(bar)

    def load_csv(
        self,
        ticker: str,
        filepath: str,
        timeframe: Timeframe,
        date_column: str = "Date",
        date_format: str = "%Y-%m-%d"
    ):
        """
        Load ticker data from CSV file

        Expected columns: Date, Open, High, Low, Close, Volume
        """
        try:
            import pandas as pd

            df = pd.read_csv(filepath)
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)

            data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            timestamps = df[date_column].tolist()

            self.load_ticker_data(ticker, timeframe, data, timestamps)

        except ImportError:
            warnings.warn("pandas not installed. Install with: pip install pandas")
        except Exception as e:
            raise ValueError(f"Error loading CSV for {ticker}: {e}")

    def generate_synthetic_data(
        self,
        ticker: str,
        timeframe: Timeframe,
        n_bars: int = 252,
        base_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0005
    ) -> np.ndarray:
        """
        Generate synthetic OHLCV data for testing

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            n_bars: Number of bars to generate
            base_price: Starting price
            volatility: Price volatility (std dev)
            trend: Trend per bar (can be negative)

        Returns:
            Nx5 array of OHLCV data
        """
        np.random.seed(hash(ticker) % (2**32))

        data = []
        current_price = base_price

        for i in range(n_bars):
            # Generate daily return with trend
            daily_return = np.random.normal(trend, volatility)

            # Open
            open_price = current_price

            # Close with trend
            close_price = open_price * (1 + daily_return)

            # High/Low with intraday volatility
            intraday_range = abs(np.random.normal(0, volatility * 0.5))
            high_price = max(open_price, close_price) * (1 + intraday_range)
            low_price = min(open_price, close_price) * (1 - intraday_range)

            # Volume (log-normal distribution)
            volume = np.random.lognormal(mean=15, sigma=0.5)

            data.append([open_price, high_price, low_price, close_price, volume])
            current_price = close_price

        data_array = np.array(data)
        self.load_ticker_data(ticker, timeframe, data_array)

        return data_array

    def add_economic_indicator(
        self,
        name: str,
        value: float,
        category: str,
        timestamp: Optional[datetime] = None,
        importance: int = 3,
        unit: str = ""
    ):
        """Add economic indicator to market field"""
        if timestamp is None:
            timestamp = datetime.now()

        indicator = EconomicIndicator(
            timestamp=timestamp,
            name=name,
            value=value,
            category=category,
            importance=importance,
            unit=unit
        )
        self.market_field.add_economic_indicator(indicator)

    def get_field(self) -> MarketField:
        """Get the current market field"""
        return self.market_field

    def clear(self):
        """Clear all data"""
        self.market_field = MarketField()


def calculate_technical_indicators(bars: List[OHLCVBar]) -> Dict[str, float]:
    """
    Calculate common technical indicators from OHLCV bars

    Returns dict with:
        - sma_20, sma_50, sma_200: Simple moving averages
        - rsi_14: Relative Strength Index
        - atr_14: Average True Range
        - bb_upper, bb_lower: Bollinger Bands
    """
    if len(bars) < 2:
        return {}

    closes = np.array([bar.close for bar in bars])
    highs = np.array([bar.high for bar in bars])
    lows = np.array([bar.low for bar in bars])

    indicators = {}

    # Simple Moving Averages
    for period in [20, 50, 200]:
        if len(closes) >= period:
            indicators[f'sma_{period}'] = np.mean(closes[-period:])

    # RSI
    if len(closes) >= 15:
        deltas = np.diff(closes)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])

        if avg_loss == 0:
            indicators['rsi_14'] = 100.0
        else:
            rs = avg_gain / avg_loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs))

    # ATR
    if len(bars) >= 15:
        tr_list = []
        for i in range(1, len(bars)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        indicators['atr_14'] = np.mean(tr_list[-14:])

    # Bollinger Bands
    if len(closes) >= 20:
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        indicators['bb_upper'] = sma_20 + 2 * std_20
        indicators['bb_lower'] = sma_20 - 2 * std_20
        indicators['bb_middle'] = sma_20

    return indicators
