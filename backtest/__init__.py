"""Backtesting engines: vectorized (HTF->M1 via Numba) and event-driven.

Both engines consume the same StrategySignals and share the same Numba M1
execution core (backtest/_m1_core.run_m1), so they produce identical trades
and metrics by construction.
"""

from backtest.engine import BacktestEngine, BacktestResult
from backtest.event_engine import EventEngine, validate_with_event_engine

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "EventEngine",
    "validate_with_event_engine",
]