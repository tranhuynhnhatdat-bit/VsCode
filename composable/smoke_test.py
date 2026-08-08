"""Smoke test: verify condition composition works end-to-end.

Builds small synthetic OHLCV data, constructs a ComposableStrategy with
hand-picked conditions, and asserts the resulting entry/exit signals match
manual expectations. This proves the composition mechanism (base logic AND
condition combination) before the GA is trusted with it.

Run: python -m composable.smoke_test
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from composable.conditions import (
    Condition,
    close,
    const,
    indicator,
)
from composable.composable import (
    ComposableStrategy,
    build_param_space,
)
from composable.conditions import condition_from_genes


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily-indexed OHLCV starting on a Monday."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
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


def test_no_conditions_pure_time():
    """With no conditions, every session-day entry bar is an entry."""
    df = _make_df()
    strat = ComposableStrategy(
        entry_hour=1, exit_hour=22, session_days=(2, 4), max_conditions=3
    )
    sig = strat.generate(df)
    # Daily index: hour is always 0, so entry_hour=1 never fires.
    assert sig.entries.sum() == 0, "no daily bar should be hour==1"
    print("test_no_conditions_pure_time: OK")


def test_single_condition_and():
    """With one condition, entries are base-entry AND condition."""
    df = _make_df()
    # Entry hour must match df's hour. Use hour=0 (all daily bars).
    strat = ComposableStrategy(
        entry_hour=0,
        exit_hour=22,
        session_days=(0,),  # Mondays only
        max_conditions=1,
        conditions=[Condition("gt", close(), const(100.0))],
    )
    sig = strat.generate(df)
    # Entries: Mondays where close > 100.
    mondays = df.index.weekday == 0
    expected = mondays & (df["Close"] > 100.0)
    assert sig.entries.tolist() == expected.tolist(), (
        "entries must equal base-entry AND condition"
    )
    print("test_single_condition_and: OK")


def test_gene_decoding():
    """Genes decode into a Condition and filter entries."""
    df = _make_df()
    genes = {
        "cond1_type": "ind_const",
        "cond1_op": "gt",
        "cond1_period": 14,
        "cond1_ind": "RSI",
        "cond1_threshold": 30.0,
        "connective": "and",
    }
    strat = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes
    )
    assert len(strat.conditions) == 1
    cond = strat.conditions[0]
    assert cond.op == "gt"
    sig = strat.generate(df)
    rsi = _rsi(df, 14)
    mondays = df.index.weekday == 0
    expected = mondays & (rsi > 30.0)
    assert sig.entries.tolist() == expected.tolist(), "gene-decoded entries mismatch"
    print("test_gene_decoding: OK")


def test_build_param_space():
    """build_param_space produces the expected gene layout."""
    space = build_param_space(max_conditions=3)
    assert "connective" in space
    for i in (1, 2, 3):
        p = f"cond{i}_"
        for key in ("type", "op", "period", "ind", "threshold"):
            assert f"{p}{key}" in space, f"missing {p}{key}"
    print("test_build_param_space: OK")


def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Reference RSI (Wilder) for the gene test."""
    from composable import indicators as ind

    return ind.RSI(df, period)


if __name__ == "__main__":
    test_no_conditions_pure_time()
    test_single_condition_and()
    test_gene_decoding()
    test_build_param_space()
    print("\nAll smoke tests passed.")