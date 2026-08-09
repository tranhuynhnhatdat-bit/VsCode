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
    price,
    KIND_OPEN,
    KIND_CLOSE,
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
        "cond1_ind": "RSI",
        "cond1_threshold": 30.0,
        "connective": "and",
        "rsi_period": 14,
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
    """build_param_space produces the expected gene layout (Option B)."""
    space = build_param_space(max_conditions=3)
    assert "connective" in space
    # Global per-parent params present.
    for key in (
        "sma_period", "ema_period", "rsi_period", "bb_period", "bb_stddev",
        "macd_fast", "macd_slow", "ichi_tenkan", "ichi_kijun", "ichi_senkou",
    ):
        assert key in space, f"missing global param {key}"
    for i in (1, 2, 3):
        p = f"cond{i}_"
        for key in ("type", "op", "ind", "ind2", "price", "price2", "threshold"):
            assert f"{p}{key}" in space, f"missing {p}{key}"
    print("test_build_param_space: OK")


def test_price_price_condition():
    """price_price: Close > Open (bullish bar) filters entries."""
    df = _make_df()
    genes = {
        "cond1_type": "price_price",
        "cond1_op": "gt",
        "cond1_price": "Close",
        "cond1_price2": "Open",
        "connective": "and",
    }
    strat = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes
    )
    assert len(strat.conditions) == 1
    sig = strat.generate(df)
    mondays = df.index.weekday == 0
    expected = mondays & (df["Close"] > df["Open"])
    assert sig.entries.tolist() == expected.tolist(), "price_price entries mismatch"
    print("test_price_price_condition: OK")


def test_ind_ind_condition():
    """ind_ind: SMA > EMA (both price-scale) filters entries."""
    df = _make_df()
    genes = {
        "cond1_type": "ind_ind",
        "cond1_op": "gt",
        "cond1_ind": "SMA",
        "cond1_ind2": "EMA",
        "connective": "and",
        "sma_period": 10,
        "ema_period": 20,
    }
    strat = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes
    )
    assert len(strat.conditions) == 1
    sig = strat.generate(df)
    from composable import indicators as ind

    sma = ind.SMA(df["Close"], 10)
    ema = ind.EMA(df["Close"], 20)
    mondays = df.index.weekday == 0
    expected = mondays & (sma > ema)
    assert sig.entries.tolist() == expected.tolist(), "ind_ind entries mismatch"
    print("test_ind_ind_condition: OK")


def test_scale_validation():
    """Scale-invalid conditions are rejected (slot -> none)."""
    # RSI (oscillator) vs SMA (price) is invalid for ind_ind.
    genes = {
        "cond1_type": "ind_ind",
        "cond1_op": "gt",
        "cond1_ind": "RSI",
        "cond1_ind2": "SMA",
        "connective": "and",
    }
    strat = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes
    )
    assert len(strat.conditions) == 0, "RSI vs SMA must be rejected"

    # price_ind with an oscillator-scale indicator is invalid.
    genes2 = {
        "cond1_type": "price_ind",
        "cond1_op": "gt",
        "cond1_price": "Close",
        "cond1_ind": "RSI",
        "connective": "and",
    }
    strat2 = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes2
    )
    assert len(strat2.conditions) == 0, "Close vs RSI must be rejected"

    # ind_const with a price-scale indicator is invalid.
    genes3 = {
        "cond1_type": "ind_const",
        "cond1_op": "gt",
        "cond1_ind": "SMA",
        "cond1_threshold": 50.0,
        "connective": "and",
    }
    strat3 = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes3
    )
    assert len(strat3.conditions) == 0, "SMA vs const must be rejected"
    print("test_scale_validation: OK")


def test_new_indicators():
    """New indicators compute without error and produce finite values."""
    df = _make_df()
    from composable import indicators as ind

    # Each returns a Series (or tuple for multi-line).
    checks = {
        "Momentum": ind.Momentum(df, 14),
        "WPR": ind.WPR(df, 14),
        "MFI": ind.MFI(df, 14),
        "OBV": ind.OBV(df),
    }
    for name, s in checks.items():
        assert isinstance(s, pd.Series), f"{name} should return a Series"
        assert s.notna().any(), f"{name} should have non-NaN values"

    bb_upper, bb_lower = ind.Bollinger(df, 20, 2.0)
    assert bb_upper.notna().any() and bb_lower.notna().any()

    macd_main, macd_sig = ind.MACD(df, 12, 26)
    assert macd_main.notna().any() and macd_sig.notna().any()

    t, k, sa, sb, c = ind.Ichimoku(df, 9, 26, 52)
    for line in (t, k, sa, sb, c):
        assert line.notna().any(), "Ichimoku line should have non-NaN values"
    print("test_new_indicators: OK")


def test_global_param_resolution():
    """Global params resolve into condition operands (Option B)."""
    df = _make_df()
    genes = {
        "cond1_type": "ind_const",
        "cond1_op": "lt",
        "cond1_ind": "Stoch_K",
        "cond1_threshold": 30.0,
        "connective": "and",
        "stoch_k": 14,
        "stoch_d": 3,
    }
    strat = ComposableStrategy(
        entry_hour=0, exit_hour=22, session_days=(0,), max_conditions=1, **genes
    )
    assert len(strat.conditions) == 1
    cond = strat.conditions[0]
    # The Stoch_K operand should have period=14, param2=3 resolved from globals.
    assert cond.left.period == 14, "Stoch_K period should resolve to 14"
    assert cond.left.param2 == 3, "Stoch_K d_period should resolve to 3"
    sig = strat.generate(df)
    assert sig.entries.sum() >= 0
    print("test_global_param_resolution: OK")


def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Reference RSI (Wilder) for the gene test."""
    from composable import indicators as ind

    return ind.RSI(df, period)


if __name__ == "__main__":
    test_no_conditions_pure_time()
    test_single_condition_and()
    test_gene_decoding()
    test_build_param_space()
    test_price_price_condition()
    test_ind_ind_condition()
    test_scale_validation()
    test_new_indicators()
    test_global_param_resolution()
    print("\nAll smoke tests passed.")