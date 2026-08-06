"""Smoke test for DataManager. Verifies M1 load, resampling, filtering, and errors."""

import pandas as pd

from data_manager import DataManager, OUTPUT_COLUMNS

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

print("\nAll smoke tests passed.")