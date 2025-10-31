"""
Financial Forecasting Module for Phi-Mamba

Forecasts price movements using phi-space phase dynamics:
- Day-ahead forecasts (1-day open/close)
- Week-ahead forecasts (5-day)
- Month-ahead forecasts (21-day)

Uses phase-locking and retrocausal constraints for prediction.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .financial_encoding import FinancialPhiEncoder, FinancialTokenState
from .financial_data import OHLCVBar, TickerData, MarketField, Timeframe
from .encoding import pentagon_reflection, retrocausal_encode
from .utils import PHI, compute_berry_phase, is_phase_locked
from .generation import generate_with_phase_lock


class ForecastHorizon(Enum):
    """Forecast time horizons"""
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    @property
    def bars(self) -> int:
        """Number of bars to forecast (daily timeframe)"""
        mapping = {
            "1d": 1,
            "1w": 5,
            "1M": 21
        }
        return mapping[self.value]


@dataclass
class PriceForecast:
    """Price forecast for a single ticker"""
    ticker: str
    horizon: ForecastHorizon
    current_price: float
    forecast_open: float
    forecast_close: float
    forecast_high: Optional[float] = None
    forecast_low: Optional[float] = None
    confidence: float = 0.5  # [0, 1]
    probability_up: float = 0.5  # Probability price increases
    expected_return: float = 0.0  # Expected return %
    risk_score: float = 0.0  # Risk metric
    phase_coherence: float = 0.0  # Phase-locking strength
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def forecast_return(self) -> float:
        """Forecast return % (close vs current)"""
        return ((self.forecast_close - self.current_price) / self.current_price) * 100.0

    @property
    def forecast_range(self) -> float:
        """Forecast high-low range"""
        if self.forecast_high is None or self.forecast_low is None:
            return 0.0
        return self.forecast_high - self.forecast_low


@dataclass
class FieldForecast:
    """Forecast for entire market field"""
    forecasts: Dict[str, PriceForecast] = field(default_factory=dict)
    horizon: ForecastHorizon = ForecastHorizon.DAY_1
    field_coherence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_forecast(self, ticker: str) -> Optional[PriceForecast]:
        """Get forecast for specific ticker"""
        return self.forecasts.get(ticker)

    def get_top_opportunities(self, n: int = 10, min_return: float = 1.0) -> List[PriceForecast]:
        """
        Get top N opportunities ranked by expected utility

        Args:
            n: Number of top opportunities
            min_return: Minimum expected return % threshold

        Returns:
            List of top forecasts
        """
        candidates = [
            f for f in self.forecasts.values()
            if f.expected_return >= min_return
        ]

        # Sort by expected return / risk ratio
        candidates.sort(
            key=lambda f: f.expected_return / max(f.risk_score, 0.01),
            reverse=True
        )

        return candidates[:n]


class PhiMambaForecaster:
    """
    Forecasting engine using Phi-Mamba phase dynamics
    """

    def __init__(
        self,
        encoder: Optional[FinancialPhiEncoder] = None,
        lookback: int = 50,
        n_monte_carlo: int = 100
    ):
        """
        Initialize forecaster

        Args:
            encoder: Financial encoder (creates default if None)
            lookback: Number of historical bars to use
            n_monte_carlo: Number of Monte Carlo simulations for uncertainty
        """
        self.encoder = encoder if encoder is not None else FinancialPhiEncoder()
        self.lookback = lookback
        self.n_monte_carlo = n_monte_carlo

    def forecast_ticker(
        self,
        ticker_data: TickerData,
        timeframe: Timeframe,
        horizon: ForecastHorizon
    ) -> PriceForecast:
        """
        Forecast price for a single ticker

        Args:
            ticker_data: Historical ticker data
            timeframe: Data timeframe
            horizon: Forecast horizon

        Returns:
            PriceForecast object
        """
        # Get historical bars
        bars = ticker_data.bars.get(timeframe, [])

        if len(bars) < self.lookback:
            # Not enough data - return neutral forecast
            current_price = bars[-1].close if len(bars) > 0 else 100.0
            return PriceForecast(
                ticker=ticker_data.ticker,
                horizon=horizon,
                current_price=current_price,
                forecast_open=current_price,
                forecast_close=current_price,
                confidence=0.0
            )

        # Use recent bars for encoding
        recent_bars = bars[-self.lookback:]
        current_price = recent_bars[-1].close

        # Encode historical sequence
        states = self.encoder.encode_sequence(recent_bars)

        # Generate future states using phase dynamics
        future_states = self._generate_future_states(
            states,
            n_steps=horizon.bars
        )

        # Convert future states to price forecast
        forecast = self._states_to_forecast(
            ticker=ticker_data.ticker,
            current_price=current_price,
            current_states=states,
            future_states=future_states,
            horizon=horizon
        )

        return forecast

    def forecast_field(
        self,
        market_field: MarketField,
        timeframe: Timeframe,
        horizon: ForecastHorizon
    ) -> FieldForecast:
        """
        Forecast entire market field

        Args:
            market_field: Market field with multiple tickers
            timeframe: Data timeframe
            horizon: Forecast horizon

        Returns:
            FieldForecast object
        """
        field_forecast = FieldForecast(horizon=horizon)

        # Forecast each ticker
        for ticker, ticker_data in market_field.tickers.items():
            forecast = self.forecast_ticker(ticker_data, timeframe, horizon)
            field_forecast.forecasts[ticker] = forecast

        # Compute field coherence
        encoded_field = self.encoder.encode_field(market_field, timeframe, self.lookback)
        field_forecast.field_coherence = self.encoder.compute_field_coherence(encoded_field)

        return field_forecast

    def _generate_future_states(
        self,
        current_states: List[FinancialTokenState],
        n_steps: int
    ) -> List[FinancialTokenState]:
        """
        Generate future states using phase-locked transitions

        Args:
            current_states: Current state sequence
            n_steps: Number of steps to forecast

        Returns:
            List of future states
        """
        if len(current_states) == 0:
            return []

        future_states = []
        last_state = current_states[-1]

        for step in range(n_steps):
            position = last_state.position + step + 1

            # Generate next state using phase-locking
            next_state = self._generate_next_state(
                current_states + future_states,
                position
            )

            future_states.append(next_state)

        return future_states

    def _generate_next_state(
        self,
        history: List[FinancialTokenState],
        position: int
    ) -> FinancialTokenState:
        """
        Generate next state using phase-locking dynamics

        Args:
            history: Historical states
            position: Position of new state

        Returns:
            Next FinancialTokenState
        """
        if len(history) == 0:
            # Default state
            return FinancialTokenState(
                token="forecast",
                index=0,
                position=position,
                vocab_size=50000,
                theta_token=0.0,
                theta_pos=0.0,
                theta_total=0.0,
                energy=PHI ** (-position)
            )

        last_state = history[-1]

        # Monte Carlo simulation for uncertainty
        angles = []
        energies = []

        for _ in range(self.n_monte_carlo):
            # Sample angle based on phase-locking
            candidate_angle = self._sample_next_angle(history)
            angles.append(candidate_angle)

            # Energy decay
            energy = last_state.energy * PHI ** (-1)
            energies.append(energy)

        # Aggregate Monte Carlo results
        mean_angle = np.angle(np.mean(np.exp(1j * np.array(angles))))
        mean_angle = (mean_angle + 2 * np.pi) % (2 * np.pi)
        mean_energy = np.mean(energies)

        # Position-based angle
        theta_pos = (position * PHI ** (-position / 10.0)) % (2 * np.pi)

        # Price from angle (inverse of encoding)
        # Assume angle represents price change
        price_change_pct = (mean_angle / (self.encoder.angular_sensitivity * PHI)) % 100.0
        if price_change_pct > 50:
            price_change_pct -= 100  # Map to [-50, 50]

        forecast_price = last_state.price * (1 + price_change_pct / 100.0) if last_state.price else 100.0

        # Zeckendorf
        price_int = int(abs(forecast_price))
        from .encoding import zeckendorf_decomposition
        zeck = zeckendorf_decomposition(price_int)

        return FinancialTokenState(
            token=f"forecast@{forecast_price:.2f}",
            index=hash(f"forecast_{position}") % 50000,
            position=position,
            vocab_size=50000,
            theta_token=mean_angle,
            theta_pos=theta_pos,
            theta_total=(mean_angle + theta_pos) % (2 * np.pi),
            energy=mean_energy,
            zeckendorf=zeck,
            price=forecast_price,
            ticker=last_state.ticker
        )

    def _sample_next_angle(
        self,
        history: List[FinancialTokenState]
    ) -> float:
        """
        Sample next angle using phase-locking probability

        Args:
            history: Historical states

        Returns:
            Sampled angle
        """
        if len(history) == 0:
            return np.random.uniform(0, 2 * np.pi)

        last_state = history[-1]
        last_angle = last_state.theta_total

        # Sample from distribution centered on phase-locked continuation
        # Use von Mises distribution (circular normal)
        kappa = 2.0  # Concentration parameter

        sampled_angle = np.random.vonmises(last_angle, kappa)
        return (sampled_angle + 2 * np.pi) % (2 * np.pi)

    def _states_to_forecast(
        self,
        ticker: str,
        current_price: float,
        current_states: List[FinancialTokenState],
        future_states: List[FinancialTokenState],
        horizon: ForecastHorizon
    ) -> PriceForecast:
        """
        Convert future states to PriceForecast

        Args:
            ticker: Ticker symbol
            current_price: Current price
            current_states: Historical states
            future_states: Forecast states
            horizon: Forecast horizon

        Returns:
            PriceForecast
        """
        if len(future_states) == 0:
            return PriceForecast(
                ticker=ticker,
                horizon=horizon,
                current_price=current_price,
                forecast_open=current_price,
                forecast_close=current_price,
                confidence=0.0
            )

        # First future state = forecast open
        forecast_open = future_states[0].price if future_states[0].price else current_price

        # Last future state = forecast close
        forecast_close = future_states[-1].price if future_states[-1].price else current_price

        # Forecast high/low from all future states
        future_prices = [s.price for s in future_states if s.price is not None]
        forecast_high = max(future_prices) if len(future_prices) > 0 else forecast_close
        forecast_low = min(future_prices) if len(future_prices) > 0 else forecast_close

        # Compute confidence from phase coherence
        confidence = self._compute_forecast_confidence(current_states, future_states)

        # Probability of upward movement
        prob_up = self._compute_probability_up(future_states)

        # Expected return
        expected_return = ((forecast_close - current_price) / current_price) * 100.0

        # Risk score (volatility of future states)
        risk_score = self._compute_risk_score(future_states)

        # Phase coherence
        phase_coherence = self._compute_phase_coherence(future_states)

        return PriceForecast(
            ticker=ticker,
            horizon=horizon,
            current_price=current_price,
            forecast_open=forecast_open,
            forecast_close=forecast_close,
            forecast_high=forecast_high,
            forecast_low=forecast_low,
            confidence=confidence,
            probability_up=prob_up,
            expected_return=expected_return,
            risk_score=risk_score,
            phase_coherence=phase_coherence
        )

    def _compute_forecast_confidence(
        self,
        current_states: List[FinancialTokenState],
        future_states: List[FinancialTokenState]
    ) -> float:
        """
        Compute confidence in forecast based on phase coherence

        Args:
            current_states: Historical states
            future_states: Forecast states

        Returns:
            Confidence score [0, 1]
        """
        if len(current_states) == 0 or len(future_states) == 0:
            return 0.0

        # Check phase-locking between last current and first future
        last_current = current_states[-1]
        first_future = future_states[0]

        berry_phase = compute_berry_phase(last_current, first_future)
        phase_locked = is_phase_locked(berry_phase, tolerance=0.5)

        # Base confidence on phase lock
        if phase_locked:
            base_confidence = 0.8
        else:
            base_confidence = 0.5

        # Adjust by energy level (higher energy = more confidence)
        energy_factor = min(last_current.energy, 1.0)

        confidence = base_confidence * (0.5 + 0.5 * energy_factor)

        return np.clip(confidence, 0.0, 1.0)

    def _compute_probability_up(
        self,
        future_states: List[FinancialTokenState]
    ) -> float:
        """
        Compute probability of upward price movement

        Args:
            future_states: Future states

        Returns:
            Probability [0, 1]
        """
        if len(future_states) < 2:
            return 0.5

        # Count upward transitions
        up_count = 0
        total_count = 0

        for i in range(1, len(future_states)):
            if future_states[i].price is not None and future_states[i-1].price is not None:
                if future_states[i].price > future_states[i-1].price:
                    up_count += 1
                total_count += 1

        if total_count == 0:
            return 0.5

        return up_count / total_count

    def _compute_risk_score(
        self,
        future_states: List[FinancialTokenState]
    ) -> float:
        """
        Compute risk score from forecast volatility

        Args:
            future_states: Future states

        Returns:
            Risk score (higher = more risky)
        """
        if len(future_states) < 2:
            return 0.0

        prices = [s.price for s in future_states if s.price is not None]

        if len(prices) < 2:
            return 0.0

        # Compute returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

        # Risk = standard deviation of returns
        risk = np.std(returns) * 100.0  # Convert to percentage

        return risk

    def _compute_phase_coherence(
        self,
        states: List[FinancialTokenState]
    ) -> float:
        """
        Compute phase coherence of state sequence

        Args:
            states: State sequence

        Returns:
            Coherence [0, 1]
        """
        if len(states) < 2:
            return 1.0

        # Count phase-locked transitions
        locked_count = 0
        total_count = 0

        for i in range(1, len(states)):
            bp = compute_berry_phase(states[i-1], states[i])
            if is_phase_locked(bp, tolerance=0.5):
                locked_count += 1
            total_count += 1

        if total_count == 0:
            return 1.0

        return locked_count / total_count


def forecast_multiple_horizons(
    forecaster: PhiMambaForecaster,
    ticker_data: TickerData,
    timeframe: Timeframe
) -> Dict[ForecastHorizon, PriceForecast]:
    """
    Generate forecasts for multiple horizons

    Args:
        forecaster: PhiMambaForecaster instance
        ticker_data: Ticker data
        timeframe: Data timeframe

    Returns:
        Dict mapping horizon -> forecast
    """
    forecasts = {}

    for horizon in ForecastHorizon:
        forecast = forecaster.forecast_ticker(ticker_data, timeframe, horizon)
        forecasts[horizon] = forecast

    return forecasts
