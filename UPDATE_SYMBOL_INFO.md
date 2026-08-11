# Update Symbol Info from the MT5 API

This document tells an agent how to update the symbol metadata file
(`symbol_info.json`) used by the trading engine for realistic backtesting.

## Purpose

Symbol metadata — tick size/value, contract size, swap rates, margin, spread,
volume limits, etc. — is fetched once from the MT5 API and persisted to a single
JSON file. Backtests and the engine load this file from disk, so **MT5 is only
needed when you explicitly refresh** this data.

- **Module**: `symbol_info.py` (`SymbolInfo` class)
- **Output file**: `C:\Users\DAT\Desktop\Strategy_Factory\Data\symbol_info.json`
- **Symbols**: 13 internal symbols mapped to FTMO/MT5 instrument names
  (e.g. `AUDUSD`, `XAUUSD`, `US30.cash`, `US100.cash`). See `SYMBOL_MAP` in
  `symbol_info.py`.

## When to run this

Run this whenever the symbol metadata needs to be current, for example:

- After the broker changes contract specifications, swap, spread, or margin.
- When a new symbol is added to `SYMBOL_MAP` in `symbol_info.py`.
- After switching to a different MT5/FTMO account with different settings.

## Prerequisites

1. **MetaTrader5 Python package** installed:
   ```bash
   pip install MetaTrader5
   ```
2. **MT5 terminal open and logged in** to the FTMO account. The internal symbols
   map to FTMO/MT5 instrument names (e.g. `US30.cash`), so the terminal must be
   logged into the account that exposes those instruments.

## How to run

Use the ready-to-run script (recommended):

```bash
python refresh_symbol_info.py
```

This force re-fetches all 13 symbols from MT5, overwrites
`symbol_info.json`, and prints a per-symbol summary plus the output path.

Equivalent one-liner:

```bash
python -c "from symbol_info import SymbolInfo; SymbolInfo().refresh()"
```

## What it writes

- **File**: `C:\Users\DAT\Desktop\Strategy_Factory\Data\symbol_info.json`
- **Contents**: one key per internal symbol, each holding the fields listed in
  `FIELDS` in `symbol_info.py` (trade_tick_value, trade_tick_size,
  trade_contract_size, point, digits, volume_min/max/step, trade_calc_mode,
  swap_mode, swap_long/short, swap_rollover3days, margin_initial,
  margin_maintenance, trade_stops_level, trade_freeze_level, spread,
  spread_float, currency_base, currency_profit, description).

## Verification

After running, confirm the refresh succeeded:

1. The script should print `Refreshed 13 symbols.` and
   `Symbol info written to: C:\Users\DAT\Desktop\Strategy_Factory\Data\symbol_info.json`.
2. Verify the file exists and contains all 13 symbols:
   ```bash
   python -c "import json; d = json.load(open(r'C:\Users\DAT\Desktop\Strategy_Factory\Data\symbol_info.json')); print(len(d), sorted(d))"
   ```
   Expected: `13` and the 13 internal symbol names.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `RuntimeError: MetaTrader5 package not installed` | Package missing | `pip install MetaTrader5` |
| `RuntimeError: MT5 initialize() failed` | MT5 terminal not open / not logged in | Start MT5 terminal and log in, then retry |
| `RuntimeError: MT5 symbol 'X' not found` | Instrument name mismatch for a symbol | Check `SYMBOL_MAP` in `symbol_info.py` against the account's instrument list |

## Notes

- On subsequent runs, `SymbolInfo` loads from disk — no MT5 needed for backtests.
- `SymbolInfo().get("EURUSD")` auto-fetches from MT5 only if the symbol is not
  already cached on disk.
- The data directory in `symbol_info.py` and `data_manager.py` is hardcoded to
  `C:\Users\DAT\Desktop\Strategy_Factory\Data\`, matching the actual data
  folder. If you move the Data folder, update `DATA_DIR` in both modules.
- The runnable script `refresh_symbol_info.py` is available both in the repo
  root and in the Data folder; it adds the repo root to `sys.path` so it works
  from either location.
