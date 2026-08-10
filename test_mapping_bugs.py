"""Regression tests for the HTF->M1 mapping and swap bug fixes.

Covered:
1. map_stops_to_m1 shifts SL/TP by one HTF period so a stop value is only
   visible to M1 bars after its HTF bar closes (no one-period lookahead).
2. map_signals_to_m1 entry/exit fill timing is correct (signal at T fills at
   the first M1 bar >= T + htf_period).
3. _holding_swap skips non-trading days (weekends) and applies the 3x
   rollover on the correctly-converted weekday (MT5 Sun=0 -> Python Mon=0).

Run: python test_mapping_bugs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest._mapping import map_signals_to_m1, map_stops_to_m1
from backtest._m1_core import build_day_arrays, _holding_swap
from strategy.base import StrategySignals


def make_m1_index(start: str, n: int, freq: str = "1min") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq)


def test_stops_shifted_one_period() -> None:
    """A stop value must only apply to M1 bars at/after its bar's close."""
    htf_idx = pd.date_range("2024-01-01", periods=4, freq="1h")
    stops = pd.Series([10.0, 11.0, 12.0, 13.0], index=htf_idx)
    m1 = make_m1_index("2024-01-01 00:00", 4 * 60, "1min")

    mapped = map_stops_to_m1(stops, m1, "H1")

    # Bar 00:00's stop (10.0) is only known at 01:00 (its close). So M1 bars
    # before 01:00 must be NaN; from 01:00 the value is 10.0; bar 01:00's
    # stop (11.0) is only known from 02:00.
    assert np.isnan(mapped.loc["2024-01-01 00:00"]), "stop leaked early"
    assert np.isnan(mapped.loc["2024-01-01 00:59"]), "stop leaked before close"
    assert mapped.loc["2024-01-01 01:00"] == 10.0, "next bar's stop leaked"
    assert mapped.loc["2024-01-01 01:59"] == 10.0
    assert mapped.loc["2024-01-01 02:00"] == 11.0
    print("  [PASS] map_stops_to_m1 shifts SL/TP by one period")


def test_signals_fill_timing() -> None:
    """Entry at H1 bar T fills at the first M1 bar >= T + 1h."""
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")
    m1 = make_m1_index("2024-01-01 00:00", 3 * 60, "1min")

    entries = pd.Series(False, index=idx)
    entries.iloc[0] = True
    signals = StrategySignals(
        entries=entries,
        exits=pd.Series(False, index=idx),
        short_entries=pd.Series(False, index=idx),
        short_exits=pd.Series(False, index=idx),
        sl_stop=pd.Series(np.nan, index=idx),
        tp_stop=pd.Series(np.nan, index=idx),
    )

    e, _, _, _ = map_signals_to_m1(signals, m1, "H1")
    true_times = m1[e.values]
    assert list(true_times) == [pd.Timestamp("2024-01-01 01:00")]
    print("  [PASS] map_signals_to_m1 fills at T + htf_period")


def _weekday_m1(start: str, end: str) -> pd.DatetimeIndex:
    """M1 index with no weekend bars, like real market data."""
    raw = pd.date_range(start, end, freq="1min")
    return raw[raw.dayofweek < 5]


def test_holding_swap_skips_weekends_and_rollover() -> None:
    """Swap skips weekends and 3x applies on the converted rollover weekday.

    Entry Wed 2024-01-03 -> exit Mon 2024-01-08 (through the weekend).
    swap_rollover3days = 3 means MT5 Wednesday (Sun=0); Python Wed = 2.
    """
    m1 = _weekday_m1("2024-01-03 00:00", "2024-01-08 23:59")
    day_ids, day_weekday, _ = build_day_arrays(m1)

    # Long, swap_long=1, swap_short=1, rollover=MT5 Wed(3).
    swap = _holding_swap(
        0, len(m1) - 1, day_ids, day_weekday, 1.0, 0, 1.0, 1.0, 3
    )
    # Days held: Wed(3x) + Thu(1x) + Fri(1x) = 5x. Sat/Sun skipped, Mon not held.
    assert swap == 5.0, f"expected 5.0 (Wed 3x + Thu + Fri), got {swap}"
    print("  [PASS] _holding_swap skips weekends + converts rollover weekday")


def test_holding_swap_no_rollover() -> None:
    """swap_rollover3days = -1 means no 3x anywhere."""
    m1 = _weekday_m1("2024-01-03 00:00", "2024-01-08 23:59")
    day_ids, day_weekday, _ = build_day_arrays(m1)
    swap = _holding_swap(
        0, len(m1) - 1, day_ids, day_weekday, 1.0, 0, 1.0, 1.0, -1
    )
    # Wed + Thu + Fri = 3x (no 3x multiplier).
    assert swap == 3.0, f"expected 3.0 (no rollover), got {swap}"
    print("  [PASS] _holding_swap handles swap_rollover3days = -1")


if __name__ == "__main__":
    print("=== HTF->M1 mapping + swap regression tests ===")
    test_stops_shifted_one_period()
    test_signals_fill_timing()
    test_holding_swap_skips_weekends_and_rollover()
    test_holding_swap_no_rollover()
    print("\nAll mapping/swap regression tests passed!")
