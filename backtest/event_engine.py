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
from composable.composable import ComposableStrategy
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


class TickSimulator:
    """Generates synthetic intra-bar ticks from M1 OHLC.

    Uses a configurable number of ticks per bar with a controlled random walk
    from Open -> High -> Low -> Close. This approximates the MT5 tester's
    tick generation for SL/TP hit detection.

    Retained for API compatibility; the M1 OHLC mode (EventEngine.run) does
    not use synthetic ticks.
    """

    def __init__(self, ticks_per_bar: int = 20, seed: int = 42) -> None:
        self.ticks_per_bar = ticks_per_bar
        self._rng = np.random.default_rng(seed)

    def generate_ticks(
        self, open_: float, high: float, low: float, close: float
    ) -> list[float]:
        """Generate synthetic tick prices for one M1 bar.

        Produces `ticks_per_bar` ticks that traverse Open -> High -> Low ->
        Close. The path respects the bar's OHLC extremes.
        """
        if self.ticks_per_bar <= 0:
            return [close]

        n = self.ticks_per_bar
        ticks: list[float] = []

        # First tick is the open price.
        ticks.append(open_)

        if n <= 1:
            return ticks

        remaining = n - 1

        # Split into two phases: Open -> extreme, extreme -> Close.
        phase1_ticks = max(1, int(remaining * self._rng.uniform(0.3, 0.7)))
        phase2_ticks = remaining - phase1_ticks

        direction = 1 if close >= open_ else -1
        extreme = high if direction > 0 else low

        # Phase 1: walk from open toward the extreme.
        for i in range(1, phase1_ticks + 1):
            progress = i / phase1_ticks
            target = open_ + (extreme - open_) * progress
            noise = self._rng.uniform(-0.0005, 0.0005) * (high - low)
            price = max(min(target + noise, high), low)
            ticks.append(price)

        # Phase 2: walk from where we are to the close.
        start_price = ticks[-1]
        for i in range(1, phase2_ticks + 1):
            progress = i / phase2_ticks
            target = start_price + (close - start_price) * progress
            noise = self._rng.uniform(-0.0005, 0.0005) * (high - low)
            price = max(min(target + noise, high), low)
            ticks.append(price)

        # Ensure we end at (or very near) the close.
        ticks[-1] = close

        return ticks


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
        ticks_per_bar: Retained for API compatibility; unused in M1 OHLC mode.
    """

    def __init__(
        self,
        symbol: str,
        htf_timeframe: str = "H1",
        initial_capital: float = 10_000.0,
        risk_money: float = 100.0,
        ticks_per_bar: int = 20,
    ) -> None:
        self.symbol = symbol
        self.htf_timeframe = htf_timeframe
        self.initial_capital = initial_capital
        self.risk_money = risk_money
        self._dm = DataManager()
        self._si = SymbolInfo()
        self._tick_sim = TickSimulator(ticks_per_bar=ticks_per_bar)

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


def validate_with_event_engine(
    params: dict[str, Any],
    m1_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    symbol: str = "XAUUSD",
    initial_capital: float = 10_000.0,
    risk_money: float = 100.0,
    ticks_per_bar: int = 20,
    filters: dict[str, float] | None = None,
    engine: EventEngine | None = None,
    htf_timeframe: str = "H1",
) -> dict[str, Any]:
    """Run a single strategy through the event engine and apply filters.

    Builds the ComposableStrategy from `params`, generates its StrategySignals
    from h1_df, and runs them through the event engine — so the GA conditions
    are actually evaluated (the event engine no longer ignores them).

    Args:
        params: ComposableStrategy constructor params (entry_hour, exit_hour,
            session_days, sl_atr, atr_period, connective, cond genes, ...).
        m1_df: Full M1 DataFrame.
        h1_df: Full H1 (HTF) DataFrame.
        symbol: Trading symbol.
        initial_capital: Starting capital.
        risk_money: Fixed risk per trade.
        ticks_per_bar: Synthetic ticks per M1 bar (unused in M1 OHLC mode).
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
    if filters is None:
        filters = {
            "profit_factor": 1.3,
            "win_rate": 35.0,
            "max_drawdown_pct": -15.0,  # drawdown is negative; -15% means 15% DD
        }

    if engine is None:
        engine = EventEngine(
            symbol=symbol,
            htf_timeframe=htf_timeframe,
            initial_capital=initial_capital,
            risk_money=risk_money,
            ticks_per_bar=ticks_per_bar,
        )

    # Build the strategy from params and generate its signals. The full GA
    # params dict drops into the constructor (conditions decoded from genes).
    strategy = ComposableStrategy(**params)
    signals = strategy.generate(h1_df)

    result = engine.run(signals, h1_df, m1_df)

    # Apply filters.
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

    return {
        "passed": passed,
        "result": result,
        "fail_reasons": fail_reasons,
    }