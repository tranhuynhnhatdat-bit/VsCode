"""Strategy skeleton: pure, stateless strategies producing vectorized signals."""

from strategy.base import Strategy, StrategySignals, require_ohlcv, OHLCV_COLUMNS

__all__ = ["Strategy", "StrategySignals", "require_ohlcv", "OHLCV_COLUMNS"]