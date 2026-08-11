"""Event-driven validation engine: bar-by-bar processing with M1 OHLC semantics.

Design (grill session):
- Matches the MQL5 Strategy Tester's "1 minute OHLC" mode.
- SL/TP evaluated against each M1 bar's High/Low; a stop within the bar's
  range fills at the stop level, a bar that opens beyond the stop (gap) fills
  at the bar's open, and when both SL and TP are in the same bar the level
  closer to the open fires first.
- Stateful position manager mirrors MQL5 CTrade behavior.
- Consumes the SAME StrategySignals as the vectorized engine, so the GA
  conditions are evaluated once and executed identically by both engines.
- Execution is delegated to the shared Numba M1 core (backtest/_m1_core.py),
  which the vectorized engine also uses — guaranteeing exact parity and
  giving a large speedup over a pure-Python loop.

Output metrics (used as final validation filters):
  - profit_factor > 1.3
  - win_rate > 35%
  - max_drawdown < 15%
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest._m1_core import run_m1
from data_manager import DataManager
from strategy.base import StrategySignals
from symbol_info import SymbolInfo

# Results directory for equity curve PNGs.
RESULTS_DIR = Path(r"C:\Users\DAT\Desktop\VsCode\results")


@dataclass
class EventTrade:
    """A single trade record (mirrors vectorbt trade output format)."""

    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    size: float
    direction: int  # 0 = long, 1 = short
    pnl: float
    return_pct: float
    status: int = 1  # 1 = closed


@dataclass
class OpenPosition:
    """State of an open position (mirrors MQL5 PositionSelect)."""

    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float  # ASK for longs, BID for shorts
    size: float
    direction: int  # 0 = long, 1 = short
    sl_price: float | None = None
    sl_distance: float | None = None
    tp_price: float | None = None


class EventEngine:
    """Bar-by-bar event-driven backtest engine (MQL5 "1 minute OHLC" mode).

    Consumes the same StrategySignals as the vectorized engine and delegates
    execution to the shared Numba M1 core, so it is behaviorally identical to
    the vectorized engine's M1 path (backtest/engine.py run()).

    Args:
        symbol: Trading symbol (e.g. "XAUUSD").
        htf_timeframe: Strategy's native timeframe (e.g. "H1").
        initial_capital: Starting capital in deposit currency.
        risk_money: Fixed risk per trade in USD.
    """

    def __init__(
        self,
        symbol: str,
        htf_timeframe: str = "H1",
        initial_capital: float = 10_000.0,
        risk_money: float = 100.0,
    ) -> None:
        self.symbol = symbol
        self.htf_timeframe = htf_timeframe
        self.initial_capital = initial_capital
        self.risk_money = risk_money
        self._dm = DataManager()
        self._si = SymbolInfo()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        signals: StrategySignals,
        htf_df: pd.DataFrame,
        m1_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run the event-driven backtest using MQL5 "1 minute OHLC" semantics.

        Consumes the same StrategySignals as the vectorized engine and shares
        the same Numba M1 execution core, so results are identical to
        BacktestEngine.run().

        Args:
            signals: StrategySignals indexed by the strategy's HTF DateTime.
            htf_df: The HTF (H1) OHLCV DataFrame used to generate signals.
            m1_df: M1 OHLCV DataFrame with DateTime index.

        Returns:
            Dict with keys:
                - 'metrics': dict of performance metrics
                - 'equity_curve': daily equity Series
                - 'trades': DataFrame of trade records
                - 'n_trades': int
                - 'profit_factor': float
                - 'win_rate': float
                - 'max_drawdown': float
        """
        info = self._si.get(self.symbol)
        result = run_m1(
            signals, htf_df, m1_df, self.htf_timeframe, info,
            self.initial_capital, self.risk_money,
        )
        metrics = result["metrics"]
        return {
            "metrics": metrics,
            "equity_curve": result["equity_curve"],
            "trades": result["trades"],
            "n_trades": metrics["n_trades"],
            "profit_factor": metrics["profit_factor"],
            "win_rate": metrics["win_rate"],
            "max_drawdown": metrics["max_drawdown_pct"],
        }


def apply_filters(
    result: dict[str, Any], filters: dict[str, float] | None = None
) -> tuple[bool, list[str]]:
    """Apply pass/fail filters to an event-engine result dict.

    Returns (passed, fail_reasons). Pure function — no strategy knowledge.
    """
    if filters is None:
        filters = {
            "profit_factor": 1.3,
            "win_rate": 35.0,
            "max_drawdown_pct": -15.0,  # drawdown is negative; -15% means 15% DD
        }

    passed = True
    fail_reasons: list[str] = []

    for metric, threshold in filters.items():
        value = result.get(metric, 0.0)
        if isinstance(value, (int, float)):
            if metric == "max_drawdown_pct":
                # Drawdown is negative; we want drawdown less than threshold
                # (e.g., -15% means 15% max drawdown -> value >= -15 passes)
                if value < threshold:
                    passed = False
                    fail_reasons.append(
                        f"{metric}={value:.2f} < {threshold:.2f}"
                    )
            else:
                if value <= threshold:
                    passed = False
                    fail_reasons.append(
                        f"{metric}={value:.4f} <= {threshold:.4f}"
                    )
        else:
            if value == float("inf") and metric == "profit_factor":
                continue
            if value <= threshold:
                passed = False
                fail_reasons.append(
                    f"{metric}={value} <= {threshold}"
                )

    return passed, fail_reasons


def validate_with_event_engine(
    signals: StrategySignals,
    h1_df: pd.DataFrame,
    m1_df: pd.DataFrame,
    symbol: str = "XAUUSD",
    initial_capital: float = 10_000.0,
    risk_money: float = 100.0,
    filters: dict[str, float] | None = None,
    engine: EventEngine | None = None,
    htf_timeframe: str = "H1",
) -> dict[str, Any]:
    """Run a strategy's signals through the event engine and apply filters.

    Takes already-generated StrategySignals (the caller builds the strategy
    and calls generate) — the backtest package stays decoupled from any
    specific strategy class.

    Args:
        signals: StrategySignals for the strategy (indexed by HTF DateTime).
        h1_df: Full H1 (HTF) DataFrame the signals were generated from.
        m1_df: Full M1 DataFrame.
        symbol: Trading symbol.
        initial_capital: Starting capital.
        risk_money: Fixed risk per trade.
        filters: Dict of {metric_name: min_threshold}. Default:
            {"profit_factor": 1.3, "win_rate": 35.0, "max_drawdown_pct": -15.0}
            (max_drawdown is negative, so threshold is upper bound)
        engine: Optional pre-built EventEngine to reuse across calls (avoids
            re-constructing DataManager/SymbolInfo for every strategy).
        htf_timeframe: Strategy's native timeframe.

    Returns:
        Dict with keys:
            - 'passed': bool
            - 'result': event engine result dict
            - 'fail_reasons': list of str describing which filters failed
    """
    if engine is None:
        engine = EventEngine(
            symbol=symbol,
            htf_timeframe=htf_timeframe,
            initial_capital=initial_capital,
            risk_money=risk_money,
        )

    result = engine.run(signals, h1_df, m1_df)
    passed, fail_reasons = apply_filters(result, filters)

    return {
        "passed": passed,
        "result": result,
        "fail_reasons": fail_reasons,
    }
