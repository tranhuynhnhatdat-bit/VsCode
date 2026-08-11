"""SymbolInfo: fetches MT5 symbol metadata (point value, swap, contract size, etc.)
for lot size calculation and realistic backtesting, and persists it to disk.

Design decisions (from grilling session):
- Separate module from DataManager (different domain: trading/risk data vs OHLCV)
- Fetch from MT5 once, persist to Data/symbol_info.json
- Load from disk on subsequent runs (no MT5 dependency for backtests)
- Dict return type: get("EURUSD") -> {"trade_tick_value": ..., ...}
- Hardcoded internal -> FTMO MT5 symbol mapping
- refresh() to force re-fetch from MT5
- ValueError on unknown symbol
- RuntimeError if MT5 unavailable and no disk cache
- 24/5 trading assumption (no session fields captured)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Absolute path to the data directory (hardcoded to prevent errors).
DATA_DIR = Path(r"C:\Users\DAT\Desktop\Strategy_Factory\Data")
# Single JSON file holding symbol info for all symbols.
SYMBOL_INFO_PATH = DATA_DIR / "symbol_info.json"

# Internal symbol name -> FTMO MT5 symbol name.
SYMBOL_MAP: dict[str, str] = {
    "AUDUSD": "AUDUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "NZDUSD": "NZDUSD",
    "USA30IDXUSD": "US30.cash",
    "USA500IDXUSD": "US500.cash",
    "USATECHIDXUSD": "US100.cash",
    "USSC2000IDXUSD": "US2000.cash",
    "USDCAD": "USDCAD",
    "USDCHF": "USDCHF",
    "USDJPY": "USDJPY",
    "XAGUSD": "XAGUSD",
    "XAUUSD": "XAUUSD",
}

# Fields captured from mt5.symbol_info() for realistic backtesting.
# Grouped by purpose for readability; the dict is flat.
# NOTE: Field names match the MetaTrader5 Python package's SymbolInfo
# namedtuple (e.g. volume_min, swap_long, margin_initial), not the MQL5
# SYMBOL_INFO_* constants. Commission fields are not exposed by this
# package version, so they are omitted.
FIELDS = [
    # Lot size calculation
    "trade_tick_value",
    "trade_tick_size",
    "trade_contract_size",
    "point",
    "digits",
    "volume_min",
    "volume_max",
    "volume_step",
    "trade_calc_mode",
    # Swap / holding cost
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    # Margin
    "margin_initial",
    "margin_maintenance",
    # Backtest realism
    "trade_stops_level",
    "trade_freeze_level",
    "spread",
    "spread_float",
    # Currency
    "currency_base",
    "currency_profit",
    # Metadata
    "description",
]


class SymbolInfo:
    """Manages MT5 symbol metadata, persisted to disk as JSON."""

    def __init__(self) -> None:
        # In-memory cache: internal symbol -> dict of fields.
        self._info: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def symbols(self) -> list[str]:
        """Return the list of available symbols (keys of SYMBOL_MAP)."""
        return list(SYMBOL_MAP.keys())

    def get(self, symbol: str) -> dict[str, Any]:
        """Return symbol metadata as a dict.

        Args:
            symbol: One of the keys in SYMBOL_MAP.

        Returns:
            A dict of MT5 symbol_info fields (see FIELDS).

        Raises:
            ValueError: If symbol is unknown.
            RuntimeError: If no disk cache exists and MT5 is unavailable.
        """
        self._validate(symbol)
        if symbol not in self._info:
            self._fetch_from_mt5()
        return dict(self._info[symbol])

    def refresh(self) -> None:
        """Force re-fetch all symbol info from MT5 and re-persist to disk."""
        self._fetch_from_mt5()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _validate(self, symbol: str) -> None:
        if symbol not in SYMBOL_MAP:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Available: {', '.join(sorted(SYMBOL_MAP))}"
            )

    def _load(self) -> None:
        """Load symbol info from disk if present."""
        if not SYMBOL_INFO_PATH.exists():
            return
        with open(SYMBOL_INFO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._info = {sym: dict(info) for sym, info in data.items()}

    def _persist(self) -> None:
        """Write symbol info to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SYMBOL_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(self._info, f, indent=2, ensure_ascii=False)

    def _fetch_from_mt5(self) -> None:
        """Fetch symbol info from MT5 for all symbols, persist, and cache.

        Raises:
            RuntimeError: If MT5 cannot be initialized or a symbol is missing.
        """
        try:
            import MetaTrader5 as mt5
        except ImportError as e:
            raise RuntimeError(
                "MetaTrader5 package not installed. Install it with "
                "'pip install MetaTrader5' and run once with the MT5 terminal "
                "open to generate Data/symbol_info.json."
            ) from e

        if not mt5.initialize():
            raise RuntimeError(
                f"MT5 initialize() failed: {mt5.last_error()}. "
                "Start the MT5 terminal and log in, then retry."
            )

        try:
            for internal, mt5_name in SYMBOL_MAP.items():
                info = mt5.symbol_info(mt5_name)
                if info is None:
                    raise RuntimeError(
                        f"MT5 symbol '{mt5_name}' not found (internal name "
                        f"'{internal}'). Check SYMBOL_MAP in symbol_info.py."
                    )
                self._info[internal] = {
                    field: getattr(info, field) for field in FIELDS
                }
        finally:
            mt5.shutdown()

        self._persist()