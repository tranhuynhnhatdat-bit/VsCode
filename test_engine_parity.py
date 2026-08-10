"""Parity test: the vectorized engine and the event-driven engine should agree.

The event engine is the source of truth (a faithful MQL5 "1 minute OHLC" model
of the ComposableStrategy). The vectorized engine's M1 `run()` is aligned to it
via the shared HTF->M1 mapping and shared SL/TP resolution. This test asserts
the two engines produce *similar* metrics (trade count equal, PF/win-rate/
drawdown within tolerance) on a real XAUUSD window.

Run: python test_engine_parity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestEngine
from backtest.event_engine import EventEngine
from composable.composable import ComposableStrategy
from data_manager import DataManager

# Short window so the test is fast but has enough entries (H1 Wed/Fri).
START = "2024-01-01"
END = "2024-06-01"

# A pure-time strategy (no conditions) + a condition-filtered one.
PURE_TIME_PARAMS = {
    "entry_hour": 1,
    "exit_hour": 22,
    "session_days": (2, 4),
    "sl_atr": 2.0,
    "atr_period": 14,
    "connective": "and",
    "cond1_type": "none",
    "cond2_type": "none",
}

CONDITION_PARAMS = {
    "entry_hour": 1,
    "exit_hour": 22,
    "session_days": (2, 4),
    "sl_atr": 2.0,
    "atr_period": 14,
    "connective": "and",
    "cond1_type": "ind_const",
    "cond1_op": "gt",
    "cond1_ind": "RSI",
    "cond1_threshold": 30.0,
    "rsi_period": 14,
    "cond2_type": "none",
}


def assert_similar(v: dict, e: dict, label: str) -> None:
    """Assert the two engines' metrics are similar within tolerance."""
    print(f"\n  --- {label} ---")
    print(f"    {'metric':<22}{'vectorized':>14}{'event':>14}")
    for k in [
        "n_trades", "profit_factor", "win_rate", "max_drawdown_pct",
        "total_return_pct", "final_equity",
    ]:
        vv = v.get(k)
        ev = e.get(k)
        print(f"    {k:<22}{str(vv):>14}{str(ev):>14}")

    # Trade count must match exactly (same signals + same mapping).
    assert v["n_trades"] == e["n_trades"], (
        f"{label}: n_trades mismatch vectorized={v['n_trades']} event={e['n_trades']}"
    )
    n = max(v["n_trades"], 1)

    # Profit factor within 15% (or both inf / both 0).
    vpf, epf = v["profit_factor"], e["profit_factor"]
    if vpf != float("inf") and epf != float("inf"):
        if vpf > 0 or epf > 0:
            ratio = max(vpf, epf) / max(min(vpf, epf), 1e-9)
            assert ratio < 1.15, (
                f"{label}: profit_factor too far apart v={vpf} e={epf}"
            )

    # Win rate within 5 percentage points.
    assert abs(v["win_rate"] - e["win_rate"]) <= 5.0, (
        f"{label}: win_rate too far apart v={v['win_rate']} e={e['win_rate']}"
    )

    # Max drawdown within 5 percentage points.
    assert abs(v["max_drawdown_pct"] - e["max_drawdown_pct"]) <= 5.0, (
        f"{label}: max_drawdown too far apart "
        f"v={v['max_drawdown_pct']} e={e['max_drawdown_pct']}"
    )

    # Final equity within 15%.
    vf, ef = v["final_equity"], e["final_equity"]
    if ef != 0:
        assert abs(vf - ef) / abs(ef) < 0.15, (
            f"{label}: final_equity too far apart v={vf} e={ef}"
        )

    print(f"  [PASS] {label}: metrics are similar")


def run_case(params: dict, dm: DataManager, label: str) -> None:
    """Run both engines on the same data and compare."""
    h1 = dm.load("XAUUSD", "H1", start=START, end=END)
    m1 = dm.load("XAUUSD", "M1", start=h1.index[0], end=h1.index[-1] + pd.Timedelta(days=1))

    strat = ComposableStrategy(**params)
    signals = strat.generate(h1)

    # Vectorized engine (M1).
    ve = BacktestEngine(
        symbol="XAUUSD", timeframe="H1", risk_money=100.0, initial_capital=10_000.0,
    )
    t0 = time.time()
    v_result = ve.run(signals, h1)
    t1 = time.time()

    # Event engine.
    ee = EventEngine(
        symbol="XAUUSD", htf_timeframe="H1", initial_capital=10_000.0, risk_money=100.0,
    )
    t2 = time.time()
    e_result = ee.run(signals, h1, m1)
    t3 = time.time()

    print(f"\n  [{label}] vectorized {t1-t0:.2f}s, event {t3-t2:.2f}s")
    assert_similar(v_result.metrics, e_result["metrics"], label)


def main() -> None:
    dm = DataManager()
    print("=== Engine Parity Test (XAUUSD H1/M1) ===")
    run_case(PURE_TIME_PARAMS, dm, "pure-time")
    run_case(CONDITION_PARAMS, dm, "conditioned")
    print("\nAll engine parity tests passed!")


if __name__ == "__main__":
    main()