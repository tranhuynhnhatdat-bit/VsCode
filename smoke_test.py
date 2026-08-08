"""Smoke test for DataManager, SymbolInfo, the strategy skeleton, and the
backtesting engine. Verifies M1 load, resampling, filtering, errors, symbol
metadata retrieval, strategy signal generation, and end-to-end backtest."""

import pandas as pd

from data_manager import DataManager, OUTPUT_COLUMNS
from symbol_info import SymbolInfo, FIELDS, SYMBOL_MAP
from strategy.examples.sma_crossover import SmaCrossover
from strategy.base import StrategySignals
from backtest.engine import BacktestEngine, BacktestResult
from optimization.engine import TestEngine, OptimizationResult
from optimization.genetic import GAConfig

dm = DataManager()

# 1. Symbols list
syms = dm.symbols()
print(f"symbols() -> {len(syms)} symbols")
assert "EURUSD" in syms and "XAUUSD" in syms

# 2. M1 load (reads the full file once, cached after)
m1 = dm.load("EURUSD", "M1")
print(f"M1 EURUSD: {len(m1)} rows, cols={list(m1.columns)}")
assert list(m1.columns) == OUTPUT_COLUMNS
assert len(m1) > 0

# 3. Resample to H1 (writes parquet on first call)
h1 = dm.load("EURUSD", "H1")
print(f"H1 EURUSD: {len(h1)} rows, first={h1.index[0]}, last={h1.index[-1]}")
assert len(h1) > 0
assert len(h1) < len(m1)

# 4. Resample to D1
d1 = dm.load("EURUSD", "D1")
print(f"D1 EURUSD: {len(d1)} rows")
assert len(d1) > 0

# 5. Date filtering (filter M1 first, then resample)
h1_2023 = dm.load("EURUSD", "H1", start="2023-01-01", end="2023-12-31")
print(f"H1 EURUSD 2023: {len(h1_2023)} rows")
assert len(h1_2023) > 0
assert h1_2023.index[0] >= pd.Timestamp("2023-01-01")
assert h1_2023.index[-1] <= pd.Timestamp("2023-12-31")

# 6. Empty result -> empty DataFrame with correct columns
empty = dm.load("EURUSD", "H1", start="1990-01-01", end="1990-01-02")
print(f"Empty result: {len(empty)} rows, cols={list(empty.columns)}")
assert empty.empty
assert list(empty.columns) == OUTPUT_COLUMNS

# 7. Error handling
try:
    dm.load("NOTASYMBOL", "H1")
    raise AssertionError("expected ValueError for unknown symbol")
except ValueError as e:
    print(f"Unknown symbol -> ValueError: {e}")

try:
    dm.load("EURUSD", "M2")
    raise AssertionError("expected ValueError for unsupported timeframe")
except ValueError as e:
    print(f"Unsupported timeframe -> ValueError: {e}")

# 8. force_recompute works (rebuilds parquet)
h1_force = dm.load("EURUSD", "H1", force_recompute=True)
print(f"H1 force_recompute: {len(h1_force)} rows")
assert len(h1_force) == len(h1)

# ------------------------------------------------------------------ #
# SymbolInfo tests
# ------------------------------------------------------------------ #
si = SymbolInfo()

# 9. Symbols list matches SYMBOL_MAP
si_syms = si.symbols()
print(f"SymbolInfo.symbols() -> {len(si_syms)} symbols")
assert set(si_syms) == set(SYMBOL_MAP.keys())

# 10. get() returns a dict with all expected fields.
#     If no disk cache and MT5 is unavailable, skip gracefully.
try:
    eur_info = si.get("EURUSD")
    print(f"SymbolInfo.get('EURUSD') -> {len(eur_info)} fields")
    assert isinstance(eur_info, dict)
    assert set(eur_info.keys()) == set(FIELDS)
    assert eur_info["trade_tick_value"] is not None
    assert eur_info["trade_contract_size"] is not None
    assert eur_info["digits"] is not None
except RuntimeError as e:
    print(f"SymbolInfo.get('EURUSD') skipped (no MT5 / no cache): {e}")

# 11. Unknown symbol -> ValueError
try:
    si.get("NOTASYMBOL")
    raise AssertionError("expected ValueError for unknown symbol")
except ValueError as e:
    print(f"SymbolInfo unknown symbol -> ValueError: {e}")

# ------------------------------------------------------------------ #
# Strategy skeleton tests
# ------------------------------------------------------------------ #

# 12. SmaCrossover on H1 EURUSD data
strat = SmaCrossover(fast=10, slow=50, atr_period=14, sl_atr=2.0, tp_atr=3.0)
sig = strat.generate(h1)
print(
    f"SmaCrossover on H1: {int(sig.entries.sum())} long entries, "
    f"{int(sig.short_entries.sum())} short entries, "
    f"{int(sig.exits.sum())} long exits, {int(sig.short_exits.sum())} short exits"
)

# 13. Signals align with the input index and are well-formed
assert isinstance(sig, StrategySignals)
assert sig.entries.index.equals(h1.index)
assert sig.exits.index.equals(h1.index)
assert sig.short_entries.index.equals(h1.index)
assert sig.short_exits.index.equals(h1.index)
assert sig.sl_stop.index.equals(h1.index)
assert sig.tp_stop.index.equals(h1.index)
assert sig.entries.dtype == bool
assert sig.exits.dtype == bool
assert sig.short_entries.dtype == bool
assert sig.short_exits.dtype == bool

# 14. No bar enters both directions at once
assert not (sig.entries & sig.short_entries).any()

# 15. SL/TP are NaN on flat bars, set while holding
held = sig.sl_stop.notna()
print(
    f"SL bars: {int(held.sum())} of {len(held)} "
    f"({(held.sum() / len(held)):.1%} of the time)"
)
assert held.sum() > 0  # the strategy actually holds sometimes

# 16. A trend-following crossover keeps you in the market most of the time.
#     (The old buggy version reported 0.8% — nearly always flat.)
assert held.sum() > 0.9 * len(held)

# 17. Pairing invariants: exits never outnumber entries for either side,
#     and at most one "open" trade remains (the last one, if data ends
#     mid-trend).
n_entries = int(sig.entries.sum())
n_exits = int(sig.exits.sum())
n_s_entries = int(sig.short_entries.sum())
n_s_exits = int(sig.short_exits.sum())
print(
    f"Pairing: long {n_entries}E/{n_exits}X, "
    f"short {n_s_entries}E/{n_s_exits}X"
)
assert n_exits <= n_entries
assert n_s_exits <= n_s_entries
assert n_entries - n_exits <= 1
assert n_s_entries - n_s_exits <= 1

# 18. Long and short positions never overlap in time: SL/TP never switch
#     side mid-hold.
assert not (sig.exits & sig.entries).any()
assert not (sig.short_exits & sig.short_entries).any()
assert not (sig.entries & sig.short_entries).any()

# 19. SL and TP are on the correct side of the entry bar's close for every
#     long entry (SL below, TP above) and short entry (SL above, TP below).
long_entries = sig.entries[sig.entries].index
if len(long_entries):
    lc = h1["Close"].loc[long_entries]
    assert (sig.sl_stop.loc[long_entries] < lc).all()
    assert (sig.tp_stop.loc[long_entries] > lc).all()
short_entries = sig.short_entries[sig.short_entries].index
if len(short_entries):
    sc = h1["Close"].loc[short_entries]
    assert (sig.sl_stop.loc[short_entries] > sc).all()
    assert (sig.tp_stop.loc[short_entries] < sc).all()
print("Strategy skeleton tests passed.")

# ------------------------------------------------------------------ #
# Backtesting engine tests
# ------------------------------------------------------------------ #

# 20. End-to-end backtest on a 1-year H1 window (keeps M1 load tractable).
h1_win = dm.load("EURUSD", "H1", start="2023-01-01", end="2023-12-31")
sig_win = strat.generate(h1_win)
engine = BacktestEngine(
    symbol="EURUSD",
    timeframe="H1",
    risk_money=100.0,
    initial_capital=10_000.0,
    strategy_name="1",
)
result = engine.run(sig_win, h1_win)
print(f"Backtest: {result.metrics['n_trades']} trades, "
      f"return {result.metrics['total_return_pct']:.2f}%, "
      f"final equity ${result.metrics['final_equity']:.2f}")

# 21. Result shape
assert isinstance(result, BacktestResult)
assert isinstance(result.metrics, dict)
assert isinstance(result.equity_curve, pd.Series)
assert len(result.equity_curve) > 0
assert result.metrics["n_trades"] > 0
assert result.metrics["final_equity"] > 0

# 22. Equity curve is daily and monotonic in time
assert result.equity_curve.index.is_monotonic_increasing
assert (result.equity_curve.index.to_series().diff().dropna() >= pd.Timedelta("1D")).all()

# 23. Save the equity curve PNG
png_path = result.save_equity_curve()
print(f"Saved equity curve: {png_path}")
assert png_path.exists()
assert png_path.suffix == ".png"

print("Backtesting engine tests passed.")

# ------------------------------------------------------------------ #
# TestEngine (optimization) tests
# ------------------------------------------------------------------ #

# 24. Tiny GA on a 6-month window: population 3, 2 generations, early stop
#     on budget. Keeps the smoke test fast.
small_config = GAConfig(
    population=3,
    generations=2,
    tournament_k=2,
    elitism=1,
    mutation_rate=0.20,
    early_stop_generations=1,
    max_evaluations=4,
    seed=42,
)
optimizer = TestEngine(
    symbol="EURUSD",
    timeframe="H1",
    strategy_class=SmaCrossover,
    param_space={
        "fast": {"min": 5, "max": 15, "step": 5},
        "slow": {"min": 20, "max": 40, "step": 10},
        "atr_period": [14],
        "sl_atr": [2.0],
        "tp_atr": [3.0],
    },
    split=(0.30, 0.50, 0.20),
    constraints=[("fast", "<", "slow")],
    ga_config=small_config,
    strategy_name="smoke_opt",
    start="2023-07-01",
    end="2023-12-31",
)
opt_result = optimizer.optimize()

# 25. Result structure (budget is generation-granular, so the full 2x3
#     population may evaluate: max 6).
assert isinstance(opt_result, OptimizationResult)
assert opt_result.report.history
assert 0 < len(opt_result.report.history) <= 6
assert opt_result.summary_path is not None
assert opt_result.summary_path.exists()

# 26. GA never produced invalid (fast >= slow) individuals.
for rec in opt_result.report.history:
    assert rec.params["fast"] < rec.params["slow"]

print("TestEngine (optimization) tests passed.")

print("\nAll smoke tests passed.")
