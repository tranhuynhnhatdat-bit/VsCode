"""ComposableStrategy: fixed time/session skeleton + GA-composed conditions.

The strategy mirrors GoldSession's structure:
- A fixed base: enter at `entry_hour` bar on `session_days`, exit at
  `exit_hour` bar, with an optional ATR stop loss.
- Up to `max_conditions` conditions chosen by the GA, combined with a
  single global connective (AND or OR), ANDed with the base logic.

The base logic always gates first: a bar must be a session-day entry bar
AND the (possibly negated) condition combination must hold.

Conditions are decoded from GA genes via `condition_from_genes`; the
constructor accepts either pre-built Condition objects (for tests) or the
gene dict (for the GA). See `build_param_space()` for the GA gene layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from composable import indicators as ind
from composable.conditions import (
    Condition,
    condition_from_genes,
)
from strategy.base import Strategy, StrategySignals, require_ohlcv

# Connective choices for the GA.
CONNECTIVES = ("and", "or")

# Default base skeleton (matches GoldSession's session days: Wed=2, Fri=4).
DEFAULT_ENTRY_HOUR = 1  # H1 bar whose close is known at 02:00 fill
DEFAULT_EXIT_HOUR = 22  # H1 bar whose close is known at 23:00 fill
DEFAULT_SESSION_DAYS = (2, 4)


class ComposableStrategy(Strategy):
    """Fixed time/session base + up to N GA-composed conditions.

    Parameters (constructor):
      entry_hour, exit_hour, session_days: base skeleton (fixed for GA).
      max_conditions: how many condition slots (0..max_conditions).
      connective: 'and' or 'or' — how conditions combine.
      conditions: optional pre-built list of Condition objects (tests).
      sl_atr: ATR stop-loss multiplier (0 = no SL).
      atr_period: ATR period for the stop.
      **genes: remaining kwargs are GA condition genes (cond1_type,
        cond1_op, cond1_period, cond1_ind, cond1_threshold, ...). The GA
        drops its whole params dict into the constructor, so these arrive
        as flat kwargs and are decoded via `_decode_conditions`.
    """

    def __init__(
        self,
        entry_hour: int = DEFAULT_ENTRY_HOUR,
        exit_hour: int = DEFAULT_EXIT_HOUR,
        session_days: tuple[int, ...] = DEFAULT_SESSION_DAYS,
        max_conditions: int = 3,
        connective: str = "and",
        conditions: list[Condition] | None = None,
        sl_atr: float = 2.0,
        atr_period: int = 14,
        **genes: Any,
    ) -> None:
        self.entry_hour = entry_hour
        self.exit_hour = exit_hour
        self.session_days = tuple(session_days)
        self.max_conditions = max_conditions
        if connective not in CONNECTIVES:
            raise ValueError(f"connective must be one of {CONNECTIVES}")
        self.connective = connective
        self.sl_atr = sl_atr
        self.atr_period = atr_period

        # Resolve conditions: explicit list wins over gene decoding.
        if conditions is not None:
            self.conditions = list(conditions)
        else:
            self.conditions = self._decode_conditions(genes)

    # ------------------------------------------------------------------ #
    # Strategy interface
    # ------------------------------------------------------------------ #
    def generate(self, df: pd.DataFrame) -> StrategySignals:
        require_ohlcv(df)

        # Base time/session logic.
        is_session = pd.Series(
            df.index.weekday.isin(self.session_days), index=df.index
        )
        is_entry_hour = pd.Series(
            df.index.hour == self.entry_hour, index=df.index
        )
        base_entry = (is_session & is_entry_hour).fillna(False).astype(bool)

        # Condition combination.
        cond_ok = self._evaluate_conditions(df)

        entries = (base_entry & cond_ok).fillna(False).astype(bool)

        # Exit: session day at exit hour, only on days that had an entry.
        entry_days = entries.index.normalize()[entries]
        is_exit_hour = pd.Series(
            df.index.hour == self.exit_hour, index=df.index
        )
        day_has_entry = pd.Series(
            df.index.normalize().isin(entry_days), index=df.index
        )
        exits = (
            is_session & is_exit_hour & day_has_entry
        ).fillna(False).astype(bool)

        # Holding state.
        held = (
            entries.astype(int).cumsum()
            - exits.astype(int).cumsum().shift(1).fillna(0)
        ) > 0

        # Optional ATR stop distance, set at entry, carried while held.
        entry_sl = pd.Series(np.nan, index=df.index, dtype=float)
        long_entry = held & ~held.shift(1).fillna(False)
        if self.sl_atr > 0:
            atr = ind.ATR(df, self.atr_period)
            entry_sl.loc[long_entry] = self.sl_atr * atr.loc[long_entry]
        sl_stop = entry_sl.ffill().where(held)

        signals = StrategySignals(
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=df.index),
            short_exits=pd.Series(False, index=df.index),
            sl_stop=sl_stop,
            tp_stop=pd.Series(np.nan, index=df.index, dtype=float),
            sl_is_distance=True,
        )
        signals.validate(df)
        return signals

    # ------------------------------------------------------------------ #
    # Condition evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_conditions(self, df: pd.DataFrame) -> pd.Series:
        """AND/OR the conditions into one boolean Series (defaults True)."""
        if not self.conditions:
            # No conditions: pure base time trade.
            return pd.Series(True, index=df.index)
        series = [c.evaluate(df) for c in self.conditions]
        if self.connective == "and":
            out = series[0]
            for s in series[1:]:
                out = out & s
        else:  # or
            out = series[0]
            for s in series[1:]:
                out = out | s
        return out.fillna(False)

    # ------------------------------------------------------------------ #
    # Gene decoding
    # ------------------------------------------------------------------ #
    def _decode_conditions(
        self, genes: dict[str, Any]
    ) -> list[Condition]:
        """Decode GA slot genes into a list of Condition objects."""
        result: list[Condition] = []
        for i in range(1, self.max_conditions + 1):
            prefix = f"cond{i}_"
            c = condition_from_genes(genes, prefix)
            if c is not None:
                result.append(c)
        return result


def build_param_space(
    max_conditions: int = 3,
    indicators: tuple[str, ...] | None = None,
    periods: tuple[int, ...] = (5, 10, 14, 20, 50),
    thresholds: tuple[float, ...] = (20.0, 30.0, 50.0, 70.0, 80.0),
) -> dict[str, Any]:
    """Build the GA ParamSpace dict for condition slots.

    Each slot has:
      <i>_type:      none | price_ind | ind_const
      <i>_op:        gt | lt | crosses_above | crosses_below
      <i>_period:    indicator period (range)
      <i>_ind:       indicator name
      <i>_threshold: constant threshold
    Plus a global `connective: [and, or]`.

    The GA drops its whole params dict into `ComposableStrategy(**params)`;
    `max_conditions` is NOT part of the space (fixed at its constructor
    default of 3), while `connective` and the per-slot genes arrive as flat
    kwargs and are decoded by the strategy constructor.
    """
    indicators = indicators or ("SMA", "EMA", "ATR", "RSI", "CCI", "Stochastic", "ADX")
    space: dict[str, Any] = {
        "connective": list(CONNECTIVES),
        # ATR stop-loss multiplier (>= 1.0 so positions get a non-zero SL
        # distance, which the engine needs to size lots). Optimized by GA.
        "sl_atr": {"min": 1.0, "max": 5.0, "step": 0.5},
    }
    for i in range(1, max_conditions + 1):
        p = f"cond{i}_"
        space[f"{p}type"] = [
            "none",
            "price_ind",
            "ind_const",
        ]
        space[f"{p}op"] = list(("gt", "lt", "crosses_above", "crosses_below"))
        space[f"{p}period"] = {"min": periods[0], "max": periods[-1], "step": 1}
        space[f"{p}ind"] = list(indicators)
        space[f"{p}threshold"] = {"min": thresholds[0], "max": thresholds[-1], "step": 5.0}
    return space