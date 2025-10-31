"""
N-Game Decision Framework with Expected Utility

Implements game-theoretic decision making for financial markets:
- Expected utility maximization
- Multi-stage game analysis
- Nash equilibrium computation
- Backward induction
- Probability-based screening

Based on the phi-mamba framework's game-theoretic foundations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

from .financial_forecast import PriceForecast, FieldForecast, ForecastHorizon
from .financial_data import MarketField, TickerData
from .utils import PHI


class Action(Enum):
    """Trading actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SHORT = "short"
    COVER = "cover"


@dataclass
class UtilityParameters:
    """Parameters for utility function"""
    risk_aversion: float = 1.0  # 0 = risk-neutral, >1 = risk-averse, <1 = risk-seeking
    wealth: float = 100000.0  # Current wealth
    max_position_size: float = 0.2  # Max % of wealth per position
    transaction_cost: float = 0.001  # 0.1% transaction cost
    time_preference: float = 0.95  # Discount factor for future utility


@dataclass
class GameState:
    """State in the trading game"""
    ticker: str
    forecast: PriceForecast
    current_position: float = 0.0  # Current position size (shares)
    cash: float = 0.0  # Available cash
    timestamp: int = 0  # Game stage


@dataclass
class GameAction:
    """Action in the trading game"""
    action: Action
    ticker: str
    quantity: float  # Number of shares
    price: float  # Execution price

    @property
    def value(self) -> float:
        """Monetary value of action"""
        return self.quantity * self.price


@dataclass
class ExpectedUtility:
    """Expected utility of an action"""
    action: GameAction
    state: GameState
    expected_return: float
    risk: float
    utility: float
    probability_success: float
    confidence: float

    @property
    def sharpe_ratio(self) -> float:
        """Risk-adjusted return (Sharpe-like ratio)"""
        if self.risk == 0:
            return 0.0
        return self.expected_return / self.risk


@dataclass
class NashEquilibrium:
    """Nash equilibrium for multi-ticker game"""
    optimal_actions: Dict[str, GameAction]
    total_utility: float
    field_forecast: FieldForecast
    is_pure_strategy: bool = True


class UtilityFunction:
    """
    Utility function for trading decisions

    Implements various utility formulations:
    - CARA (Constant Absolute Risk Aversion)
    - CRRA (Constant Relative Risk Aversion)
    - Logarithmic
    - Quadratic
    """

    def __init__(self, params: UtilityParameters, utility_type: str = "CRRA"):
        """
        Initialize utility function

        Args:
            params: Utility parameters
            utility_type: Type of utility ("CARA", "CRRA", "log", "quadratic")
        """
        self.params = params
        self.utility_type = utility_type

    def compute(self, wealth: float) -> float:
        """
        Compute utility of wealth level

        Args:
            wealth: Wealth level

        Returns:
            Utility value
        """
        if self.utility_type == "CARA":
            # U(w) = -exp(-α * w)
            return -np.exp(-self.params.risk_aversion * wealth)

        elif self.utility_type == "CRRA":
            # U(w) = (w^(1-γ)) / (1-γ)
            gamma = self.params.risk_aversion
            if gamma == 1.0:
                return np.log(wealth)
            return (wealth ** (1 - gamma)) / (1 - gamma)

        elif self.utility_type == "log":
            # U(w) = log(w)
            return np.log(max(wealth, 1e-10))

        elif self.utility_type == "quadratic":
            # U(w) = w - (γ/2) * w²
            return wealth - (self.params.risk_aversion / 2) * (wealth ** 2)

        else:
            raise ValueError(f"Unknown utility type: {self.utility_type}")

    def compute_expected_utility(
        self,
        wealth_distribution: List[Tuple[float, float]]
    ) -> float:
        """
        Compute expected utility over wealth distribution

        Args:
            wealth_distribution: List of (wealth, probability) tuples

        Returns:
            Expected utility
        """
        eu = 0.0
        for wealth, prob in wealth_distribution:
            eu += prob * self.compute(wealth)
        return eu


class DecisionMaker:
    """
    Makes trading decisions using expected utility maximization
    """

    def __init__(
        self,
        utility_params: Optional[UtilityParameters] = None,
        utility_type: str = "CRRA"
    ):
        """
        Initialize decision maker

        Args:
            utility_params: Utility parameters (uses defaults if None)
            utility_type: Type of utility function
        """
        self.utility_params = utility_params if utility_params else UtilityParameters()
        self.utility_function = UtilityFunction(self.utility_params, utility_type)

    def evaluate_action(
        self,
        state: GameState,
        action: GameAction
    ) -> ExpectedUtility:
        """
        Evaluate expected utility of an action in a state

        Args:
            state: Current game state
            action: Proposed action

        Returns:
            ExpectedUtility object
        """
        forecast = state.forecast

        # Current wealth
        current_wealth = state.cash + state.current_position * forecast.current_price

        # Transaction cost
        transaction_cost = action.value * self.utility_params.transaction_cost

        # Compute wealth distribution after action
        wealth_distribution = self._compute_wealth_distribution(
            state, action, transaction_cost
        )

        # Expected utility
        eu = self.utility_function.compute_expected_utility(wealth_distribution)

        # Expected return
        expected_return = forecast.expected_return

        # Risk
        risk = forecast.risk_score

        return ExpectedUtility(
            action=action,
            state=state,
            expected_return=expected_return,
            risk=risk,
            utility=eu,
            probability_success=forecast.probability_up if action.action == Action.BUY else (1 - forecast.probability_up),
            confidence=forecast.confidence
        )

    def _compute_wealth_distribution(
        self,
        state: GameState,
        action: GameAction,
        transaction_cost: float
    ) -> List[Tuple[float, float]]:
        """
        Compute wealth distribution after taking action

        Args:
            state: Current state
            action: Action to take
            transaction_cost: Transaction cost

        Returns:
            List of (wealth, probability) tuples
        """
        forecast = state.forecast
        current_wealth = state.cash + state.current_position * forecast.current_price

        # Simplified: use forecast to estimate outcome distribution
        # Assume three outcomes: up, flat, down

        if action.action == Action.BUY:
            # Buy action
            new_position = state.current_position + action.quantity
            new_cash = state.cash - action.value - transaction_cost

            # Up scenario (forecast close)
            wealth_up = new_cash + new_position * forecast.forecast_close
            prob_up = forecast.probability_up

            # Flat scenario
            wealth_flat = new_cash + new_position * forecast.current_price
            prob_flat = 0.2

            # Down scenario
            down_price = forecast.current_price * (1 - abs(forecast.forecast_return) / 100.0)
            wealth_down = new_cash + new_position * down_price
            prob_down = 1 - prob_up - prob_flat

            return [
                (wealth_up, prob_up),
                (wealth_flat, prob_flat),
                (wealth_down, max(prob_down, 0.0))
            ]

        elif action.action == Action.SELL:
            # Sell action
            new_position = state.current_position - action.quantity
            new_cash = state.cash + action.value - transaction_cost

            # After selling, wealth is mostly cash, less sensitive to price
            wealth = new_cash + new_position * forecast.forecast_close

            return [(wealth, 1.0)]

        elif action.action == Action.HOLD:
            # Hold - wealth depends on price movement
            prob_up = forecast.probability_up

            wealth_up = state.cash + state.current_position * forecast.forecast_close
            wealth_flat = state.cash + state.current_position * forecast.current_price
            down_price = forecast.current_price * 0.95
            wealth_down = state.cash + state.current_position * down_price

            return [
                (wealth_up, prob_up),
                (wealth_flat, 0.2),
                (wealth_down, 1 - prob_up - 0.2)
            ]

        else:
            # Default: no change
            return [(current_wealth, 1.0)]

    def find_optimal_action(
        self,
        state: GameState,
        allowed_actions: Optional[List[Action]] = None
    ) -> ExpectedUtility:
        """
        Find optimal action for a state

        Args:
            state: Current game state
            allowed_actions: List of allowed actions (all if None)

        Returns:
            ExpectedUtility of optimal action
        """
        if allowed_actions is None:
            allowed_actions = [Action.BUY, Action.SELL, Action.HOLD]

        # Generate candidate actions
        candidate_actions = self._generate_candidate_actions(state, allowed_actions)

        # Evaluate each action
        evaluations = [
            self.evaluate_action(state, action)
            for action in candidate_actions
        ]

        # Select action with maximum expected utility
        optimal = max(evaluations, key=lambda eu: eu.utility)

        return optimal

    def _generate_candidate_actions(
        self,
        state: GameState,
        allowed_actions: List[Action]
    ) -> List[GameAction]:
        """
        Generate candidate actions for a state

        Args:
            state: Current game state
            allowed_actions: Allowed action types

        Returns:
            List of GameAction candidates
        """
        candidates = []
        forecast = state.forecast
        max_position = self.utility_params.max_position_size * self.utility_params.wealth / forecast.current_price

        if Action.BUY in allowed_actions:
            # Buy at various quantities
            for fraction in [0.25, 0.5, 0.75, 1.0]:
                quantity = max_position * fraction
                candidates.append(GameAction(
                    action=Action.BUY,
                    ticker=state.ticker,
                    quantity=quantity,
                    price=forecast.current_price
                ))

        if Action.SELL in allowed_actions and state.current_position > 0:
            # Sell at various quantities
            for fraction in [0.25, 0.5, 0.75, 1.0]:
                quantity = state.current_position * fraction
                candidates.append(GameAction(
                    action=Action.SELL,
                    ticker=state.ticker,
                    quantity=quantity,
                    price=forecast.current_price
                ))

        if Action.HOLD in allowed_actions:
            # Hold
            candidates.append(GameAction(
                action=Action.HOLD,
                ticker=state.ticker,
                quantity=0,
                price=forecast.current_price
            ))

        return candidates


class MultiTickerGameSolver:
    """
    Solves multi-ticker games to find Nash equilibria
    """

    def __init__(
        self,
        decision_maker: Optional[DecisionMaker] = None,
        max_tickers: int = 10
    ):
        """
        Initialize game solver

        Args:
            decision_maker: DecisionMaker instance
            max_tickers: Maximum number of tickers to consider simultaneously
        """
        self.decision_maker = decision_maker if decision_maker else DecisionMaker()
        self.max_tickers = max_tickers

    def solve_field_game(
        self,
        field_forecast: FieldForecast,
        current_portfolio: Optional[Dict[str, float]] = None,
        cash: float = 100000.0
    ) -> NashEquilibrium:
        """
        Solve multi-ticker game to find optimal allocation

        Args:
            field_forecast: Forecasts for all tickers
            current_portfolio: Current positions {ticker: quantity}
            cash: Available cash

        Returns:
            NashEquilibrium with optimal actions
        """
        if current_portfolio is None:
            current_portfolio = {}

        # Create game states for each ticker
        states = {}
        for ticker, forecast in field_forecast.forecasts.items():
            states[ticker] = GameState(
                ticker=ticker,
                forecast=forecast,
                current_position=current_portfolio.get(ticker, 0.0),
                cash=cash / len(field_forecast.forecasts)  # Divide cash equally
            )

        # Find optimal action for each ticker independently (pure strategy Nash)
        optimal_actions = {}
        total_utility = 0.0

        for ticker, state in states.items():
            optimal_eu = self.decision_maker.find_optimal_action(state)
            optimal_actions[ticker] = optimal_eu.action
            total_utility += optimal_eu.utility

        return NashEquilibrium(
            optimal_actions=optimal_actions,
            total_utility=total_utility,
            field_forecast=field_forecast,
            is_pure_strategy=True
        )

    def backward_induction(
        self,
        states: List[GameState],
        n_stages: int = 3
    ) -> List[GameAction]:
        """
        Solve game using backward induction

        Args:
            states: Sequence of game states (one per stage)
            n_stages: Number of stages to look ahead

        Returns:
            Optimal action sequence
        """
        if len(states) == 0:
            return []

        # Work backward from final stage
        optimal_actions = [None] * n_stages

        for stage in range(n_stages - 1, -1, -1):
            if stage < len(states):
                state = states[stage]

                # Find optimal action at this stage given future optimality
                optimal_eu = self.decision_maker.find_optimal_action(state)
                optimal_actions[stage] = optimal_eu.action

        return [a for a in optimal_actions if a is not None]


class OpportunityScreener:
    """
    Screens trading opportunities using expected utility
    """

    def __init__(
        self,
        decision_maker: Optional[DecisionMaker] = None,
        min_utility: float = 0.0,
        min_confidence: float = 0.5
    ):
        """
        Initialize screener

        Args:
            decision_maker: DecisionMaker instance
            min_utility: Minimum utility threshold
            min_confidence: Minimum confidence threshold
        """
        self.decision_maker = decision_maker if decision_maker else DecisionMaker()
        self.min_utility = min_utility
        self.min_confidence = min_confidence

    def screen_opportunities(
        self,
        field_forecast: FieldForecast,
        top_n: int = 10,
        min_return: float = 1.0
    ) -> List[Tuple[str, ExpectedUtility]]:
        """
        Screen field for top opportunities

        Args:
            field_forecast: Field forecast
            top_n: Number of top opportunities to return
            min_return: Minimum expected return %

        Returns:
            List of (ticker, ExpectedUtility) tuples sorted by utility
        """
        opportunities = []

        for ticker, forecast in field_forecast.forecasts.items():
            # Skip if below minimum return
            if forecast.expected_return < min_return:
                continue

            # Skip if below minimum confidence
            if forecast.confidence < self.min_confidence:
                continue

            # Create game state
            state = GameState(
                ticker=ticker,
                forecast=forecast,
                current_position=0.0,
                cash=self.decision_maker.utility_params.wealth
            )

            # Evaluate BUY action
            buy_action = GameAction(
                action=Action.BUY,
                ticker=ticker,
                quantity=1000,  # Standard lot size
                price=forecast.current_price
            )

            eu = self.decision_maker.evaluate_action(state, buy_action)

            # Filter by minimum utility
            if eu.utility >= self.min_utility:
                opportunities.append((ticker, eu))

        # Sort by utility (descending)
        opportunities.sort(key=lambda x: x[1].utility, reverse=True)

        return opportunities[:top_n]

    def screen_by_sharpe(
        self,
        field_forecast: FieldForecast,
        top_n: int = 10,
        min_sharpe: float = 0.5
    ) -> List[Tuple[str, ExpectedUtility]]:
        """
        Screen opportunities by Sharpe ratio

        Args:
            field_forecast: Field forecast
            top_n: Number of top opportunities
            min_sharpe: Minimum Sharpe ratio

        Returns:
            List of (ticker, ExpectedUtility) sorted by Sharpe ratio
        """
        opportunities = []

        for ticker, forecast in field_forecast.forecasts.items():
            state = GameState(
                ticker=ticker,
                forecast=forecast,
                current_position=0.0,
                cash=self.decision_maker.utility_params.wealth
            )

            buy_action = GameAction(
                action=Action.BUY,
                ticker=ticker,
                quantity=1000,
                price=forecast.current_price
            )

            eu = self.decision_maker.evaluate_action(state, buy_action)

            if eu.sharpe_ratio >= min_sharpe:
                opportunities.append((ticker, eu))

        # Sort by Sharpe ratio
        opportunities.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)

        return opportunities[:top_n]

    def compute_field_metrics(
        self,
        field_forecast: FieldForecast
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics for the field

        Args:
            field_forecast: Field forecast

        Returns:
            Dict of metric name -> value
        """
        all_forecasts = list(field_forecast.forecasts.values())

        if len(all_forecasts) == 0:
            return {}

        returns = [f.expected_return for f in all_forecasts]
        risks = [f.risk_score for f in all_forecasts]
        confidences = [f.confidence for f in all_forecasts]
        probs_up = [f.probability_up for f in all_forecasts]

        return {
            'mean_expected_return': np.mean(returns),
            'median_expected_return': np.median(returns),
            'mean_risk': np.mean(risks),
            'mean_confidence': np.mean(confidences),
            'mean_prob_up': np.mean(probs_up),
            'bullish_count': sum(1 for r in returns if r > 0),
            'bearish_count': sum(1 for r in returns if r < 0),
            'high_confidence_count': sum(1 for c in confidences if c > 0.7),
            'field_coherence': field_forecast.field_coherence
        }
