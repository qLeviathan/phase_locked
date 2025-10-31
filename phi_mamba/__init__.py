"""
Φ-Mamba: Phase-Locked Language Modeling with Golden Ratio Encoding

A revolutionary approach to language modeling that uses:
- Golden ratio (φ) as the fundamental primitive
- Retrocausal encoding for improved coherence
- Topological information storage
- Natural termination through energy decay

Financial Adaptation:
- Multi-ticker time series analysis
- Phase-locked forecasting
- Expected utility decision framework
- N-game Nash equilibrium computation
"""

from .core import PhiLanguageModel, PhiTokenizer
from .encoding import retrocausal_encode, zeckendorf_decomposition
from .generation import generate_with_phase_lock
from .utils import PHI, PSI, compute_berry_phase

# Financial modules
from .financial_data import (
    FinancialDataLoader, MarketField, Timeframe,
    OHLCVBar, TickerData, EconomicIndicator
)
from .financial_encoding import (
    FinancialPhiEncoder, FinancialTokenState
)
from .financial_forecast import (
    PhiMambaForecaster, PriceForecast, FieldForecast, ForecastHorizon
)
from .decision_framework import (
    DecisionMaker, OpportunityScreener, UtilityParameters,
    MultiTickerGameSolver, NashEquilibrium, ExpectedUtility
)
from .financial_system import (
    PhiMambaFinancialSystem, create_default_system, AnalysisResults
)

__version__ = "0.1.0"
__author__ = "Marc Castillo"

__all__ = [
    # Core
    "PhiLanguageModel",
    "PhiTokenizer",
    "retrocausal_encode",
    "zeckendorf_decomposition",
    "generate_with_phase_lock",
    "PHI",
    "PSI",
    "compute_berry_phase",
    # Financial
    "FinancialDataLoader",
    "MarketField",
    "Timeframe",
    "OHLCVBar",
    "TickerData",
    "EconomicIndicator",
    "FinancialPhiEncoder",
    "FinancialTokenState",
    "PhiMambaForecaster",
    "PriceForecast",
    "FieldForecast",
    "ForecastHorizon",
    "DecisionMaker",
    "OpportunityScreener",
    "UtilityParameters",
    "MultiTickerGameSolver",
    "NashEquilibrium",
    "ExpectedUtility",
    "PhiMambaFinancialSystem",
    "create_default_system",
    "AnalysisResults"
]