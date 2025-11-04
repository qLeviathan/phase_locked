"""
Financial Encoding Module for Phi-Mamba

Converts financial time series (OHLCV data) into phi-space TokenStates
for processing by the Phi-Mamba model.

Key mappings:
- Price movements → Angular positions (theta)
- Volume → Energy levels
- Time → Position in sequence
- Multi-ticker field → Coupled TokenStates
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings

from .encoding import TokenState, calculate_betti_numbers
from .math_core import zeckendorf_decomposition
from .constants import PHI, PSI
from .phase_dynamics import compute_berry_phase, is_phase_locked
from .financial_data import (
    OHLCVBar, TickerData, MarketField, Timeframe,
    EconomicIndicator, calculate_technical_indicators
)


@dataclass
class FinancialTokenState(TokenState):
    """
    TokenState extended with financial-specific attributes.

    Inherits from TokenState:
        - token, index, position, vocab_size
        - theta_token, theta_pos, theta_total
        - energy, zeckendorf
        - future_constraint, coherence_weight
        - r, active_shells (properties)

    Adds financial fields:
        - price, volume, ticker, timeframe
        - price_change_pct, technical_indicators
    """
    price: Optional[float] = None
    volume: Optional[float] = None
    ticker: Optional[str] = None
    timeframe: Optional[Timeframe] = None
    price_change_pct: Optional[float] = None
    technical_indicators: Dict[str, float] = field(default_factory=dict)


class FinancialPhiEncoder:
    """
    Encodes financial data into phi-space representations
    """

    def __init__(
        self,
        price_scale: float = 100.0,
        volume_scale: float = 1e6,
        angular_sensitivity: float = 1.0
    ):
        """
        Initialize encoder

        Args:
            price_scale: Reference price for normalization
            volume_scale: Reference volume for normalization
            angular_sensitivity: Multiplier for price → angle mapping
        """
        self.price_scale = price_scale
        self.volume_scale = volume_scale
        self.angular_sensitivity = angular_sensitivity

    def encode_bar(
        self,
        bar: OHLCVBar,
        position: int,
        reference_price: Optional[float] = None
    ) -> FinancialTokenState:
        """
        Encode a single OHLCV bar into a FinancialTokenState

        Args:
            bar: OHLCV bar to encode
            position: Position in sequence
            reference_price: Reference price for calculating returns

        Returns:
            FinancialTokenState in phi-space
        """
        if reference_price is None:
            reference_price = bar.open

        # Calculate price change
        price_change_pct = ((bar.close - reference_price) / reference_price) * 100.0

        # Map price change to angular position
        # Positive returns → positive angles, negative returns → negative angles
        # Scale by golden ratio for natural spacing
        theta_price = (price_change_pct * self.angular_sensitivity * PHI) % (2 * np.pi)

        # Map position using phi-based spacing (similar to RoPE)
        theta_pos = (position * PHI ** (-position / 10.0)) % (2 * np.pi)

        # Combined angle
        theta_total = (theta_price + theta_pos) % (2 * np.pi)

        # Volume-based energy
        # Higher volume = higher energy = more "active" state
        volume_normalized = bar.volume / self.volume_scale
        energy_volume = min(volume_normalized, 10.0)  # Cap at 10x

        # Combine with position-based decay
        energy = energy_volume * (PHI ** (-position))

        # Zeckendorf decomposition of price (integer part)
        price_int = int(abs(bar.close))
        zeck = zeckendorf_decomposition(price_int)

        # Create token representation
        token_str = f"{bar.ticker}@{bar.close:.2f}"

        return FinancialTokenState(
            token=token_str,
            index=hash(token_str) % 50000,  # Map to vocab space
            position=position,
            vocab_size=50000,
            theta_token=theta_price,
            theta_pos=theta_pos,
            theta_total=theta_total,
            energy=energy,
            zeckendorf=zeck,
            price=bar.close,
            volume=bar.volume,
            ticker=bar.ticker,
            timeframe=bar.timeframe,
            price_change_pct=price_change_pct
        )

    def encode_sequence(
        self,
        bars: List[OHLCVBar],
        reference_price: Optional[float] = None
    ) -> List[FinancialTokenState]:
        """
        Encode a sequence of OHLCV bars

        Args:
            bars: List of OHLCV bars in chronological order
            reference_price: Reference price (uses first bar close if not provided)

        Returns:
            List of FinancialTokenStates
        """
        if len(bars) == 0:
            return []

        if reference_price is None:
            reference_price = bars[0].close

        states = []
        for i, bar in enumerate(bars):
            state = self.encode_bar(bar, position=i, reference_price=reference_price)
            states.append(state)

        return states

    def encode_ticker_data(
        self,
        ticker_data: TickerData,
        timeframe: Timeframe,
        max_length: Optional[int] = None
    ) -> List[FinancialTokenState]:
        """
        Encode all data for a ticker at a specific timeframe

        Args:
            ticker_data: TickerData object
            timeframe: Which timeframe to encode
            max_length: Maximum number of bars to encode (most recent)

        Returns:
            List of FinancialTokenStates
        """
        bars = ticker_data.bars.get(timeframe, [])

        if len(bars) == 0:
            return []

        if max_length is not None and len(bars) > max_length:
            bars = bars[-max_length:]

        return self.encode_sequence(bars)

    def encode_field(
        self,
        market_field: MarketField,
        timeframe: Timeframe,
        max_length: Optional[int] = None
    ) -> Dict[str, List[FinancialTokenState]]:
        """
        Encode entire market field (all tickers) at a timeframe

        Args:
            market_field: MarketField containing multiple tickers
            timeframe: Which timeframe to encode
            max_length: Maximum bars per ticker

        Returns:
            Dict mapping ticker -> list of states
        """
        encoded_field = {}

        for ticker, ticker_data in market_field.tickers.items():
            states = self.encode_ticker_data(ticker_data, timeframe, max_length)
            if len(states) > 0:
                encoded_field[ticker] = states

        return encoded_field

    def encode_field_snapshot(
        self,
        market_field: MarketField,
        timeframe: Timeframe
    ) -> List[FinancialTokenState]:
        """
        Encode current snapshot of all tickers as a single sequence

        This creates a "field state" where all tickers at the current time
        are represented as a sequence, preserving their phase relationships.

        Args:
            market_field: MarketField
            timeframe: Timeframe

        Returns:
            List of states (one per ticker at current time)
        """
        field_state = market_field.get_field_state(timeframe)

        states = []
        position = 0

        # Sort tickers for consistent ordering
        for ticker in sorted(field_state.keys()):
            bar = field_state[ticker]
            if bar is not None:
                state = self.encode_bar(bar, position)
                states.append(state)
                position += 1

        return states

    def compute_field_coherence(
        self,
        encoded_field: Dict[str, List[FinancialTokenState]],
        lookback: int = 20
    ) -> float:
        """
        Compute overall phase coherence across the field

        Measures how "aligned" different tickers are in phi-space.
        High coherence = synchronized movement.

        Args:
            encoded_field: Dict of ticker -> states
            lookback: How many recent bars to analyze

        Returns:
            Coherence score [0, 1]
        """
        if len(encoded_field) < 2:
            return 1.0

        # Collect recent states from each ticker
        recent_states = []
        for ticker, states in encoded_field.items():
            if len(states) >= lookback:
                recent_states.append(states[-lookback:])

        if len(recent_states) < 2:
            return 1.0

        # Compute pairwise Berry phases
        berry_phases = []
        for i in range(len(recent_states)):
            for j in range(i + 1, len(recent_states)):
                states_i = recent_states[i]
                states_j = recent_states[j]

                for k in range(min(len(states_i), len(states_j))):
                    bp = compute_berry_phase(states_i[k], states_j[k])
                    berry_phases.append(bp)

        if len(berry_phases) == 0:
            return 1.0

        # Coherence = how many pairs are phase-locked
        locked_count = sum(1 for bp in berry_phases if abs(bp % (2 * np.pi)) < 0.5)
        coherence = locked_count / len(berry_phases)

        return coherence

    def find_phase_locked_pairs(
        self,
        encoded_field: Dict[str, List[FinancialTokenState]],
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Find ticker pairs that are phase-locked (strongly correlated)

        Args:
            encoded_field: Dict of ticker -> states
            threshold: Berry phase threshold for "locked" status

        Returns:
            List of (ticker1, ticker2, berry_phase) tuples
        """
        tickers = list(encoded_field.keys())
        locked_pairs = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker_i = tickers[i]
                ticker_j = tickers[j]

                states_i = encoded_field[ticker_i]
                states_j = encoded_field[ticker_j]

                if len(states_i) == 0 or len(states_j) == 0:
                    continue

                # Compare most recent states
                state_i = states_i[-1]
                state_j = states_j[-1]

                bp = compute_berry_phase(state_i, state_j)

                if abs(bp % (2 * np.pi)) < threshold:
                    locked_pairs.append((ticker_i, ticker_j, bp))

        return locked_pairs

    def encode_economic_indicator(
        self,
        indicator: EconomicIndicator,
        position: int
    ) -> FinancialTokenState:
        """
        Encode economic indicator as a TokenState

        Args:
            indicator: Economic indicator
            position: Position in sequence

        Returns:
            FinancialTokenState
        """
        # Map indicator value to angle
        # Normalize by typical range (assume -10 to 10 for percentages)
        normalized_value = np.clip(indicator.value / 10.0, -1, 1)
        theta_indicator = (normalized_value * np.pi) % (2 * np.pi)

        # Position-based angle
        theta_pos = (position * PHI ** (-position / 10.0)) % (2 * np.pi)

        theta_total = (theta_indicator + theta_pos) % (2 * np.pi)

        # Importance-based energy
        energy = (indicator.importance / 5.0) * (PHI ** (-position))

        # Zeckendorf of value (use absolute value, scaled)
        value_int = int(abs(indicator.value * 100))  # Convert to basis points
        zeck = zeckendorf_decomposition(value_int)

        token_str = f"{indicator.name}@{indicator.value:.2f}"

        return FinancialTokenState(
            token=token_str,
            index=hash(token_str) % 50000,
            position=position,
            vocab_size=50000,
            theta_token=theta_indicator,
            theta_pos=theta_pos,
            theta_total=theta_total,
            energy=energy,
            zeckendorf=zeck,
            price=indicator.value,
            ticker=indicator.name,
            technical_indicators={"category": indicator.category}
        )

    def create_combined_sequence(
        self,
        market_field: MarketField,
        timeframe: Timeframe,
        include_economic: bool = True,
        max_bars: int = 50
    ) -> List[FinancialTokenState]:
        """
        Create a combined sequence of ticker data + economic indicators

        This interleaves market data with economic data to create a
        comprehensive field representation.

        Args:
            market_field: MarketField
            timeframe: Timeframe
            include_economic: Whether to include economic indicators
            max_bars: Max bars per ticker

        Returns:
            Combined sequence of states
        """
        sequence = []
        position = 0

        # Add ticker snapshots (most recent)
        field_snapshot = self.encode_field_snapshot(market_field, timeframe)
        sequence.extend(field_snapshot)
        position += len(field_snapshot)

        # Add economic indicators
        if include_economic and len(market_field.economic_indicators) > 0:
            for indicator in market_field.economic_indicators[-10:]:  # Last 10
                state = self.encode_economic_indicator(indicator, position)
                sequence.append(state)
                position += 1

        return sequence


def compute_price_momentum_angle(
    bars: List[OHLCVBar],
    lookback: int = 20
) -> float:
    """
    Compute momentum direction as an angle

    Args:
        bars: List of OHLCV bars
        lookback: Lookback period

    Returns:
        Angle in radians [0, 2π)
        - 0 = strong upward momentum
        - π/2 = sideways with bullish bias
        - π = strong downward momentum
        - 3π/2 = sideways with bearish bias
    """
    if len(bars) < lookback:
        return 0.0

    recent_bars = bars[-lookback:]

    # Calculate total return
    start_price = recent_bars[0].close
    end_price = recent_bars[-1].close
    total_return = (end_price - start_price) / start_price

    # Calculate volatility (std of returns)
    returns = []
    for i in range(1, len(recent_bars)):
        ret = (recent_bars[i].close - recent_bars[i-1].close) / recent_bars[i-1].close
        returns.append(ret)

    volatility = np.std(returns) if len(returns) > 0 else 0.01

    # Map to angle
    # Return → radial component (r)
    # Volatility → angular spread
    angle = np.arctan2(total_return, volatility)
    angle = (angle + 2 * np.pi) % (2 * np.pi)

    return angle


def analyze_field_topology(
    encoded_field: Dict[str, List[FinancialTokenState]]
) -> Dict[str, any]:
    """
    Analyze topological structure of the field

    Args:
        encoded_field: Encoded field data

    Returns:
        Dict with topological metrics
    """
    analysis = {
        'n_tickers': len(encoded_field),
        'total_states': sum(len(states) for states in encoded_field.values()),
        'betti_numbers': {},
        'active_shells': {},
        'energy_distribution': []
    }

    # Aggregate Zeckendorf decompositions
    all_zeck = []
    all_energies = []

    for ticker, states in encoded_field.items():
        for state in states:
            all_zeck.append(state.zeckendorf)
            all_energies.append(state.energy)

    # Compute aggregate Betti numbers
    if len(all_zeck) > 0:
        # Average number of active shells
        avg_shells = np.mean([len(z) for z in all_zeck])
        analysis['avg_active_shells'] = avg_shells

        # Most common Fibonacci numbers
        from collections import Counter
        all_fibs = []
        for z in all_zeck:
            all_fibs.extend(z)
        fib_counts = Counter(all_fibs)
        analysis['common_fibonacci_scales'] = fib_counts.most_common(5)

    # Energy distribution
    if len(all_energies) > 0:
        analysis['energy_distribution'] = {
            'mean': np.mean(all_energies),
            'std': np.std(all_energies),
            'min': np.min(all_energies),
            'max': np.max(all_energies)
        }

    return analysis
