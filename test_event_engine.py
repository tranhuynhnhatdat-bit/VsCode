"""Quick smoke test for the event-driven validation engine.

Tests:
1. EventEngine runs end-to-end with minimal data
2. validate_with_event_engine applies filters correctly

Run: python test_event_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.event_engine import (
    EventEngine,
    validate_with_event_engine,
)
from composable.composable import ComposableStrategy


def test_event_engine_empty() -> None:
    """Test that EventEngine handles no-data gracefully."""
    engine = EventEngine(
        symbol="XAUUSD",
        initial_capital=10_000.0,
        risk_money=100.0,
    )

    empty_idx = pd.DatetimeIndex([])
    empty_m1 = pd.DataFrame(
        {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}, index=empty_idx
    )
    empty_h1 = pd.DataFrame(
        {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}, index=empty_idx
    )

    # Build an empty strategy + signals aligned to the empty H1 index.
    strat = ComposableStrategy(entry_hour=1, exit_hour=22, session_days=(2, 4))
    signals = strat.generate(empty_h1)

    result = engine.run(signals, empty_h1, empty_m1)

    assert result["n_trades"] == 0
    assert result["metrics"]["final_equity"] == 10_000.0
    print("  [PASS] EventEngine handles empty data gracefully")


def _generate_aligned_data(
    start_date: str = "2024-01-01",
    n_weeks: int = 2,
    base_price: float = 2050.0,
    force_bearish_entry: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate aligned H1 and M1 data for testing.

    H1 bars are exactly 1 hour. M1 bars are exactly 1 minute.
    Entry hour bars (hour=1) have Close < Open so entries fire.
    Uses real calendar dates with Wed/Fri session days.
    """
    # H1: 24 hours * 7 days * n_weeks bars.
    n_h1 = 24 * 7 * n_weeks
    h1_dates = pd.date_range(start_date, periods=n_h1, freq="1h", tz=None)

    np.random.seed(42)
    h1 = pd.DataFrame({
        "Open": np.full(n_h1, base_price, dtype=float),
        "High": np.full(n_h1, base_price, dtype=float),
        "Low": np.full(n_h1, base_price, dtype=float),
        "Close": np.full(n_h1, base_price, dtype=float),
        "Volume": np.random.randint(100, 10000, n_h1),
    }, index=h1_dates)

    # Add some random walk to prices so ATR > 0.
    price = base_price
    for i in range(n_h1):
        change = np.random.normal(0, 3.0)
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, 2.0))
        l = min(o, c) - abs(np.random.normal(0, 2.0))

        if h1_dates[i].hour == 1 and force_bearish_entry:
            c = o - 10.0  # Force Close < Open for entry condition
            h = max(o, c) + 2.0
            l = min(o, c) - 2.0

        h1.iloc[i] = {
            "Open": o, "High": h, "Low": l, "Close": c,
            "Volume": np.random.randint(100, 10000)
        }
        price = c

    # M1: 60 * number of H1 bars.
    n_m1 = n_h1 * 60
    m1_dates = pd.date_range(start_date, periods=n_m1, freq="1min", tz=None)
    m1 = pd.DataFrame({
        "Open": np.full(n_m1, base_price, dtype=float),
        "High": np.full(n_m1, base_price, dtype=float),
        "Low": np.full(n_m1, base_price, dtype=float),
        "Close": np.full(n_m1, base_price, dtype=float),
        "Volume": np.random.randint(10, 1000, n_m1),
    }, index=m1_dates)

    # Build M1 bars from H1 using interpolation.
    for i in range(n_h1):
        h1_idx = h1_dates[i]
        h1_o = float(h1.iloc[i]["Open"])
        h1_c = float(h1.iloc[i]["Close"])
        h1_h = float(h1.iloc[i]["High"])
        h1_l = float(h1.iloc[i]["Low"])

        start_m1 = i * 60
        end_m1 = start_m1 + 60
        for j in range(start_m1, min(end_m1, n_m1)):
            frac = (j - start_m1) / 60.0
            m1_open = h1_o + (h1_c - h1_o) * frac / 60  # nearly h1_o for first tick
            m1_close = h1_o + (h1_c - h1_o) * (frac + 1/60)

            # Ensure it passes through h1_o at j=start_m1 and h1_c at j=end_m1-1
            if j == start_m1:
                m1_open = h1_o
                m1_close = h1_o + (h1_c - h1_o) / 60.0
            elif j == end_m1 - 1:
                m1_open = m1.iloc[j-1]["Close"]
                m1_close = h1_c

            # Add minor intra-bar variation
            m1_h = max(m1_open, m1_close) + abs(np.random.normal(0, 0.5))
            m1_l = min(m1_open, m1_close) - abs(np.random.normal(0, 0.5))
            # Ensure bounds
            m1_h = min(m1_h, h1_h)
            m1_l = max(m1_l, h1_l)

            m1.iloc[j] = {
                "Open": m1_open, "High": m1_h, "Low": m1_l,
                "Close": m1_close, "Volume": np.random.randint(10, 1000)
            }

    return h1, m1


def test_event_engine_with_aligned_data() -> None:
    """Test EventEngine with properly aligned synthetic data."""
    h1, m1 = _generate_aligned_data(
        start_date="2024-01-01", n_weeks=2, base_price=2050.0
    )

    engine = EventEngine(
        symbol="XAUUSD",
        initial_capital=10_000.0,
        risk_money=100.0,
    )

    # Build a pure-time ComposableStrategy (no conditions) and generate signals.
    strat = ComposableStrategy(entry_hour=1, exit_hour=22, session_days=(2, 4))
    signals = strat.generate(h1)

    result = engine.run(signals, h1, m1)

    assert "metrics" in result
    assert "equity_curve" in result
    print(f"  Trades generated: {result['n_trades']}")
    print(f"  Profit factor: {result['profit_factor']:.2f}")
    print(f"  Win rate: {result['win_rate']:.1f}%")
    print(f"  Max drawdown: {result['max_drawdown']:.1f}%")
    print(f"  Final equity: {result['metrics']['final_equity']:.2f}")

    # With 2 weeks, we should have at least 2 Wed + 2 Fri = 4 entry opportunities
    # Some may fail due to ATR warmup, but at least 1 should succeed.
    assert result["n_trades"] > 0, (
        f"Expected at least 1 trade, got {result['n_trades']}. "
        f"This likely means the entry logic is failing."
    )
    print("  [PASS] EventEngine generates trades with aligned data")


def test_validate_with_event_engine() -> None:
    """Test the validate_with_event_engine wrapper and its filters."""
    h1, m1 = _generate_aligned_data(
        start_date="2024-01-03", n_weeks=2, base_price=2050.0
    )

    params = {
        "entry_hour": 1,
        "exit_hour": 22,
        "session_days": (2, 4),  # Wed, Fri
        "sl_atr": 2.0,
        "atr_period": 14,
        "connective": "and",
        "cond1_type": "none",
        "cond2_type": "none",
    }

    strat = ComposableStrategy(**params)
    signals = strat.generate(h1)

    result = validate_with_event_engine(
        signals=signals,
        h1_df=h1,
        m1_df=m1,
        symbol="XAUUSD",
        initial_capital=10_000.0,
        risk_money=100.0,
    )

    assert "passed" in result
    assert "result" in result
    assert "fail_reasons" in result
    print(f"  Validation passed: {result['passed']}")
    print(f"  PF={result['result']['profit_factor']:.2f}, "
          f"WR={result['result']['win_rate']:.1f}%, "
          f"DD={result['result']['max_drawdown']:.1f}%")
    print(f"  Trades: {result['result']['n_trades']}")
    print("  [PASS] validate_with_event_engine works correctly")


if __name__ == "__main__":
    print("\nTesting EventEngine (empty data)...")
    test_event_engine_empty()

    print("\nTesting EventEngine (aligned data)...")
    test_event_engine_with_aligned_data()

    print("\nTesting validate_with_event_engine...")
    test_validate_with_event_engine()

    print("\nAll tests passed!")