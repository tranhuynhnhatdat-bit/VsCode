"""DataManager: manages OHLCV CSV files in Data/, resamples M1 to any timeframe,
and persists resampled data to parquet on disk.

Design decisions (from grilling session):
- Python + pandas
- Resample from M1 for all popular timeframes
- Hardcoded symbol -> filename dict
- Absolute DATA_DIR constant
- DataFrame output (DateTime index, OHLCV columns)
- Optional start/end date filtering (filter M1 first, then resample)
- In-memory M1 cache + in-memory resampled cache
- Disk persistence to Data/resampled/<SYMBOL>_<TF>.parquet
- mtime staleness check (recompute if source newer)
- Disk-first resolution, fall back to resample+write
- Keep partial final bar
- ValueError on unknown symbol/timeframe
- Empty DataFrame on empty result
- force_recompute=True param
- Ignore timezone (naive datetime)

Data format (MQL5 export, fixed spread, no header):
- 9 columns: Date,Time,Open,High,Low,Close,Volume,RealVolume,Spread
- Spread is in points and constant per symbol (e.g. XAUUSD=30, EURUSD=15)
- The Spread column is carried through resampling so the engine can apply
  the per-bar spread to entry fills (matching the MQL5 engine).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Absolute path to the data directory (hardcoded to prevent errors).
DATA_DIR = Path(r"C:\Users\DAT\Desktop\VsCode\Data")
# Subdirectory where resampled parquet files are persisted.
RESAMPLED_DIR = DATA_DIR / "resampled"

# Hardcoded symbol -> filename mapping. Covers all 13 files in Data/.
SYMBOLS: dict[str, str] = {
    "AUDUSD": "AUDUSD_ftmo.csv",
    "EURUSD": "EURUSD_ftmo.csv",
    "GBPUSD": "GBPUSD_ftmo.csv",
    "NZDUSD": "NZDUSD_ftmo.csv",
    "USA30IDXUSD": "USA30IDXUSD_ftmo.csv",
    "USA500IDXUSD": "USA500IDXUSD_ftmo.csv",
    "USATECHIDXUSD": "USATECHIDXUSD_ftmo.csv",
    "USDCAD": "USDCAD_ftmo.csv",
    "USDCHF": "USDCHF_ftmo.csv",
    "USDJPY": "USDJPY_ftmo.csv",
    "USSC2000IDXUSD": "USSC2000IDXUSD_ftmo.csv",
    "XAGUSD": "XAGUSD_ftmo.csv",
    "XAUUSD": "XAUUSD_ftmo.csv",
}

# Supported timeframes. M1 is the source; the rest are resampled from it.
TIMEFRAMES: set[str] = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"}

# Resampling rule: Open=first, High=max, Low=min, Close=last, Volume=sum.
# Spread is constant per symbol, so "first" is correct.
RESAMPLE_RULE = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "Spread": "first",
}

# pandas offset alias for each timeframe.
_TIMEFRAME_OFFSET = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1ME",
}

# Columns of the output DataFrame (DateTime is the index).
OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Spread"]

# Raw CSV column names (no header in the MQL5 export).
_RAW_COLUMNS = [
    "Date",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RealVolume",
    "Spread",
]


class DataManager:
    """Manages OHLCV data for all symbols, resampling M1 to any timeframe."""

    def __init__(self) -> None:
        # Cache of raw M1 DataFrames keyed by symbol.
        self._m1_cache: dict[str, pd.DataFrame] = {}
        # Cache of resampled DataFrames keyed by (symbol, timeframe).
        self._resampled_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def symbols(self) -> list[str]:
        """Return the list of available symbols (keys of the hardcoded dict)."""
        return list(SYMBOLS.keys())

    def load(
        self,
        symbol: str,
        timeframe: str = "M1",
        start: str | None = None,
        end: str | None = None,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Load OHLCV data for a symbol at a given timeframe.

        Args:
            symbol: One of the keys in SYMBOLS.
            timeframe: One of TIMEFRAMES (default "M1").
            start: Optional start datetime string (inclusive), e.g. "2023-01-01".
            end: Optional end datetime string (inclusive), e.g. "2023-12-31".
            force_recompute: If True, ignore the on-disk parquet and rebuild.

        Returns:
            A DataFrame indexed by DateTime with columns
            Open, High, Low, Close, Volume, Spread. Empty DataFrame if no
            rows match.
        """
        self._validate(symbol, timeframe)

        if timeframe == "M1":
            df = self._get_m1(symbol)
            return self._filter(df, start, end)

        # Check in-memory resampled cache.
        key = (symbol, timeframe)
        if not force_recompute and key in self._resampled_cache:
            return self._filter(self._resampled_cache[key], start, end)

        # Check on-disk parquet (unless force_recompute).
        parquet_path = self._parquet_path(symbol, timeframe)
        if not force_recompute and self._is_fresh(parquet_path, symbol):
            df = pd.read_parquet(parquet_path)
            self._resampled_cache[key] = df
            return self._filter(df, start, end)

        # Resample from M1, write to disk, cache, return.
        m1 = self._get_m1(symbol)
        df = m1.resample(_TIMEFRAME_OFFSET[timeframe]).agg(RESAMPLE_RULE).dropna()
        self._write_parquet(df, parquet_path)
        self._resampled_cache[key] = df
        return self._filter(df, start, end)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _validate(self, symbol: str, timeframe: str) -> None:
        if symbol not in SYMBOLS:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Available: {', '.join(sorted(SYMBOLS))}"
            )
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Available: {', '.join(sorted(TIMEFRAMES))}"
            )

    def _get_m1(self, symbol: str) -> pd.DataFrame:
        """Return the raw M1 DataFrame for a symbol, reading + caching on first use."""
        if symbol in self._m1_cache:
            return self._m1_cache[symbol]

        path = DATA_DIR / SYMBOLS[symbol]
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found for '{symbol}': {path}"
            )

        df = pd.read_csv(
            path,
            header=None,
            names=_RAW_COLUMNS,
            dtype={
                "Open": float,
                "High": float,
                "Low": float,
                "Close": float,
                "Volume": float,
                "RealVolume": float,
                "Spread": float,
            },
        )
        # Combine Date + Time into a single DateTime index.
        df["DateTime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M"
        )
        df = df.drop(columns=["Date", "Time", "RealVolume"])
        df = df.set_index("DateTime").sort_index()
        self._m1_cache[symbol] = df
        return df

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        return RESAMPLED_DIR / f"{symbol}_{timeframe}.parquet"

    def _is_fresh(self, parquet_path: Path, symbol: str) -> bool:
        """True if the parquet exists and is newer than the source M1 CSV."""
        if not parquet_path.exists():
            return False
        source_path = DATA_DIR / SYMBOLS[symbol]
        if not source_path.exists():
            return False
        return parquet_path.stat().st_mtime >= source_path.stat().st_mtime

    def _write_parquet(self, df: pd.DataFrame, parquet_path: Path) -> None:
        RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)

    def _filter(
        self, df: pd.DataFrame, start: str | None, end: str | None
    ) -> pd.DataFrame:
        """Apply start/end filtering, returning an empty frame with correct columns."""
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        if df.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        return df