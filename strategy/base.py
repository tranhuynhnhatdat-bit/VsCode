"""Strategy base classes and signal contract.

A Strategy is a pure, stateless function from an OHLCV DataFrame to a
StrategySignals dataclass. StrategySignals mirrors vectorbt
Portfolio.from_signals() arguments, so a thin adapter can feed the
backtesting engine later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import pandas as pd

# Columns expected in the DataFrame passed to Strategy.generate().
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def require_ohlcv(df: pd.DataFrame) -> None:
    """Raise ValueError if df is missing any OHLCV column."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {missing}. Got: {list(df.columns)}"
        )


@dataclass
class StrategySignals:
    """Vectorized entry/exit/SL/TP contract for a Strategy.

    Field names mirror vectorbt's Portfolio.from_signals() arguments:
      entries, exits, short_entries, short_exits, sl_stop, tp_stop

    Semantics:
    - entries/exits: booleans indexed by the strategy's HTF DateTime.
      True -> enter long / exit to flat.
    - short_entries/short_exits: booleans. True -> enter short / exit to flat.
    - sl_stop/tp_stop: floats, NaN except bars while a position is open.
      A fixed stop keeps its entry value; a trailing stop rewrites each bar.

    Long and short are exclusive: entries and short_entries must never both
    be True at the same bar. Exit + opposite entry on the same bar is a
    "flip" — the engine decides whether that requires a flat bar.
    """

    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    sl_stop: pd.Series
    tp_stop: pd.Series

    def validate(self, df: pd.DataFrame) -> None:
        """Verify all series align with df and are internally consistent.

        Raises:
            TypeError: If any field is not a pd.Series.
            ValueError: If indexes mismatch, dtypes are wrong, or entry/exit
                signals overlap illegally.
        """
        index = df.index
        for f in fields(self):
            s = getattr(self, f.name)
            if not isinstance(s, pd.Series):
                raise TypeError(
                    f"{f.name} must be a pd.Series, got {type(s).__name__}"
                )
            if not s.index.equals(index):
                raise ValueError(
                    f"{f.name} index does not match the input df index"
                )

        for name in ("entries", "exits", "short_entries", "short_exits"):
            s = getattr(self, name)
            if s.dtype != bool:
                raise ValueError(f"{name} must have dtype bool, got {s.dtype}")

        for name in ("sl_stop", "tp_stop"):
            s = getattr(self, name)
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError(f"{name} must be numeric, got {s.dtype}")

        # Cannot enter both directions at the same bar.
        both_entry = self.entries & self.short_entries
        if both_entry.any():
            n = int(both_entry.sum())
            first = both_entry.idxmax()
            raise ValueError(
                f"entries and short_entries overlap at {n} bar(s), "
                f"first at {first}. A signal must exit to flat before "
                f"entering the other direction."
            )

        # A bar cannot be both an entry and an exit for the same side.
        for entry_name, exit_name in (
            ("entries", "exits"),
            ("short_entries", "short_exits"),
        ):
            overlap = getattr(self, entry_name) & getattr(self, exit_name)
            if overlap.any():
                n = int(overlap.sum())
                first = overlap.idxmax()
                raise ValueError(
                    f"{entry_name} and {exit_name} overlap at {n} bar(s), "
                    f"first at {first}."
                )


class Strategy(ABC):
    """Pure, stateless strategy: OHLCV DataFrame in, StrategySignals out."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> StrategySignals:
        """Generate entry/exit/SL/TP signals for an OHLCV DataFrame.

        Must be a pure function of df — no internal state, no side effects,
        no access to symbol metadata (that is the engine's concern). Params
        are set in the constructor.
        """