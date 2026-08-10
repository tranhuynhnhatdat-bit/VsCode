"""Shared HTF -> M1 signal mapping helpers used by both backtest engines.

Both the vectorized engine (`backtest/engine.py`) and the event-driven engine
(`backtest/event_engine.py`) must map HTF entry/exit signals onto M1 execution
bars *identically*, otherwise the two engines disagree on trade timing. This
module centralizes that mapping so both engines stay in lock-step.

The HTF bar timestamp is the bar's OPEN time; a signal computed from that
bar's close is only known one period later. The fill is the first M1 bar at
or after T + htf_period — the first tick of the target M1 bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import StrategySignals

# HTF timeframe -> pandas offset for the signal->fill lookahead.
HTF_OFFSET = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1ME",
}


def htf_period(timeframe: str) -> pd.Timedelta:
    """Timedelta for the HTF timeframe's signal->fill lookahead."""
    return pd.Timedelta(HTF_OFFSET[timeframe])


def map_signals_to_m1(
    signals: StrategySignals, m1_index: pd.DatetimeIndex, timeframe: str
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Map HTF entry/exit signals to M1 bars at/after each actionable time.

    A signal at HTF time T is known at T + htf_period. The fill is the first
    M1 bar at or after T + htf_period.

    Returns (entries, exits, short_entries, short_exits) boolean Series on
    the M1 index.
    """
    delta = htf_period(timeframe).to_timedelta64()
    n = len(m1_index)
    m1_vals = m1_index.values  # datetime64[ns]
    out: dict[str, np.ndarray] = {}

    for name in ("entries", "exits", "short_entries", "short_exits"):
        target = np.zeros(n, dtype=bool)
        sig = getattr(signals, name)
        ts = sig.index[sig.values].values  # datetime64[ns] of True bars
        if len(ts):
            actionable = ts + delta
            pos = np.searchsorted(m1_vals, actionable, side="left")
            pos = pos[pos < n]
            target[pos] = True
        out[name] = target

    return (
        pd.Series(out["entries"], index=m1_index),
        pd.Series(out["exits"], index=m1_index),
        pd.Series(out["short_entries"], index=m1_index),
        pd.Series(out["short_exits"], index=m1_index),
    )


def map_stops_to_m1(
    htf_stops: pd.Series, m1_index: pd.DatetimeIndex, timeframe: str
) -> pd.Series:
    """Map HTF SL/TP to M1 via forward-fill, shifted one period ahead.

    A stop value at HTF bar T is known only at T + htf_period (the bar's
    close). We shift the stop index forward by one HTF period *before* the
    forward-fill, so each M1 bar sees only stops whose close has already
    happened. This keeps stops exactly consistent with ``map_signals_to_m1``,
    which shifts entries/exits by the same period.

    Without the shift, an M1 bar sitting exactly on an HTF boundary would
    pick up the stop of the bar *opening* at that instant (whose close has not
    happened yet) — a one-period lookahead. This matters for trailing stops,
    which rewrite their value each bar; it is harmless for a fixed stop that
    keeps its entry value (the value is identical across adjacent held bars).
    """
    shifted = htf_stops.copy()
    shifted.index = shifted.index + htf_period(timeframe)
    return shifted.reindex(m1_index, method="ffill")