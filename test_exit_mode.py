"""Unit tests for the ComposableStrategy `exit_mode` feature.

The `end_of_week` exit mode force-closes any still-held position at the exit
hour on literal Friday (Python weekday 4), regardless of whether Friday is a
configured session day. This guards against a position carrying indefinitely
when the normal same-day exit's data is missing (e.g. a holiday/data gap).

Run: python -m test_exit_mode
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from composable.composable import ComposableStrategy


def _make_h1_df(day_times, n_hours=24, seed=0):
    """Build an H1 OHLCV frame over the given starting timestamps.

    Each element of `day_times` contributes H1 bars from 00:00 through
    (n_hours-1):00.
    """
    rng = np.random.default_rng(seed)
    idx = []
    for dt in day_times:
        idx.extend(pd.date_range(dt, periods=n_hours, freq="h"))
    idx = pd.DatetimeIndex(sorted(idx))
    n = len(idx)
    close_p = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close_p + np.abs(rng.normal(0, 0.5, n))
    low = close_p - np.abs(rng.normal(0, 0.5, n))
    open_p = np.concatenate([[close_p[0]], close_p[:-1]])
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close_p,
            "Volume": vol,
        },
        index=idx,
    )


def _week_days():
    # Jan 1 2024 is a Monday, so Wed=03, Thu=04, Fri=05.
    return [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]


def test_same_day_keeps_position_open_when_exit_missing():
    """same_day: with the Wed 22:00 exit bar missing, no exit fires."""
    df = _make_h1_df(_week_days())
    # Remove Wednesday 22:00 (hour == exit_hour) to simulate missing exit data.
    df = df.drop(pd.Timestamp("2024-01-03 22:00"))
    strat = ComposableStrategy(
        entry_hour=1, exit_hour=22, session_days=(2,), max_conditions=0
    )
    sig = strat.generate(df)
    assert sig.entries.sum() == 1, "one Wednesday entry expected"
    assert sig.exits.sum() == 0, "same_day should leave the position open"
    print("test_same_day_keeps_position_open_when_exit_missing: OK")


def test_end_of_week_closes_on_friday():
    """end_of_week: with the Wed 22:00 exit bar missing, Friday 22:00 closes."""
    df = _make_h1_df(_week_days())
    df = df.drop(pd.Timestamp("2024-01-03 22:00"))
    strat = ComposableStrategy(
        entry_hour=1,
        exit_hour=22,
        session_days=(2,),
        max_conditions=0,
        exit_mode="end_of_week",
    )
    sig = strat.generate(df)
    assert sig.entries.sum() == 1, "one Wednesday entry expected"
    assert sig.exits.sum() == 1, "end_of_week should force one Friday close"
    exit_idx = sig.exits[sig.exits].index
    assert len(exit_idx) == 1
    assert exit_idx[0] == pd.Timestamp("2024-01-05 22:00"), (
        f"expected Friday 22:00 close, got {exit_idx[0]}"
    )
    print("test_end_of_week_closes_on_friday: OK")


def test_end_of_week_no_effect_when_same_day_exit_ok():
    """end_of_week: when the same-day exit fires, no extra Friday close."""
    df = _make_h1_df(_week_days())  # Wed 22:00 present -> same-day exit fires
    strat = ComposableStrategy(
        entry_hour=1,
        exit_hour=22,
        session_days=(2,),
        max_conditions=0,
        exit_mode="end_of_week",
    )
    sig = strat.generate(df)
    assert sig.entries.sum() == 1, "one Wednesday entry expected"
    assert sig.exits.sum() == 1, "one exit expected (the same-day close)"
    exit_idx = sig.exits[sig.exits].index
    assert exit_idx[0] == pd.Timestamp("2024-01-03 22:00"), (
        "exit should be the Wednesday same-day close"
    )
    print("test_end_of_week_no_effect_when_same_day_exit_ok: OK")


def test_invalid_exit_mode_rejected():
    """Invalid exit_mode is rejected at construction."""
    try:
        ComposableStrategy(exit_mode="bogus")
    except ValueError:
        print("test_invalid_exit_mode_rejected: OK")
        return
    raise AssertionError("invalid exit_mode should raise ValueError")


if __name__ == "__main__":
    test_same_day_keeps_position_open_when_exit_missing()
    test_end_of_week_closes_on_friday()
    test_end_of_week_no_effect_when_same_day_exit_ok()
    test_invalid_exit_mode_rejected()
    print("\nAll exit-mode tests passed.")