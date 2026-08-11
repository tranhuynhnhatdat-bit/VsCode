"""Refresh symbol info from the MT5 API and re-persist it to disk.

This is a ready-to-run script for updating Data/symbol_info.json.
It force re-fetches all 13 symbols from MT5 and overwrites the JSON.

Usage:
    python refresh_symbol_info.py

Prerequisites:
    - MetaTrader5 package installed (pip install MetaTrader5)
    - The MT5 terminal is open and logged into the FTMO account
      (the internal symbols map to FTMO/MT5 instrument names, e.g. US30.cash)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `symbol_info.py` importable regardless of the current working directory
# (the script may be run from the repo root or from a copy in Data/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from symbol_info import SYMBOL_INFO_PATH, SymbolInfo


def main() -> int:
    print("Refreshing symbol info from MT5 API ...")
    try:
        info = SymbolInfo()
        info.refresh()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Make sure the MT5 terminal is open and logged in, and that "
            "'pip install MetaTrader5' has been run.",
            file=sys.stderr,
        )
        return 1

    symbols = info.symbols()
    print(f"Refreshed {len(symbols)} symbols.")
    for symbol in symbols:
        entry = info.get(symbol)
        print(f"  {symbol}: spread={entry.get('spread')} "
              f"swap_long={entry.get('swap_long')} "
              f"swap_short={entry.get('swap_short')} "
              f"volume_step={entry.get('volume_step')}")

    print(f"Symbol info written to: {SYMBOL_INFO_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())