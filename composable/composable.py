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

Param layout (Option B — MQL5-handle style):
- Global per-parent indicator params shared across all condition slots
  (e.g. `sma_period`, `rsi_period`, `bb_period` + `bb_stddev`, ...).
- Per-slot genes only pick WHICH indicator line + op + threshold.
- The strategy resolves each condition's indicator operand period/param2
  from the global params at construction time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from composable import indicators as ind
from composable.conditions import (
    INDICATOR_REGISTRY,
    KIND_INDICATOR,
    Operand,
    Condition,
    condition_from_genes,
)
from strategy.base import Strategy, StrategySignals, require_ohlcv

# Connective choices for the GA.
CONNECTIVES = ("and", "or")

# Exit modes: how a held position is eventually closed.
#   same_day   - close at exit_hour on session days only (current behavior).
#   end_of_week- close at exit_hour on session days; if still holding at the
#                end of the trading week (literal Friday, weekday 4), force
#                close there regardless of session-day membership.
EXIT_MODES = ("same_day", "end_of_week")
# Python weekday number for Friday (pandas: Mon=0 .. Sun=6).
FRIDAY_WEEKDAY = 4

# Default base skeleton (matches GoldSession's session days: Wed=2, Fri=4).
DEFAULT_ENTRY_HOUR = 1  # H1 bar whose close is known at 02:00 fill
DEFAULT_EXIT_HOUR = 22  # H1 bar whose close is known at 23:00 fill
DEFAULT_SESSION_DAYS = (2, 4)

# Global param keys per indicator parent: (period_key, param2_key_or_None).
# param2 is the second shared param (e.g. bb_stddev, macd_slow, stoch_d).
PARENT_PARAMS: dict[str, tuple[str, str | None]] = {
    "SMA": ("sma_period", None),
    "EMA": ("ema_period", None),
    "ATR": ("atr_period", None),
    "RSI": ("rsi_period", None),
    "CCI": ("cci_period", None),
    "Stochastic": ("stoch_k", "stoch_d"),
    "ADX": ("adx_period", None),
    "Bollinger": ("bb_period", "bb_stddev"),
    "MACD": ("macd_fast", "macd_slow"),
    "Momentum": ("mom_period", None),
    "WPR": ("wpr_period", None),
    "MFI": ("mfi_period", None),
    "OBV": (None, None),
    "Ichimoku": ("ichi_tenkan", "ichi_kijun"),
}


class ComposableStrategy(Strategy):
    """Fixed time/session base + up to N GA-composed conditions.

    Parameters (constructor):
      entry_hour, exit_hour, session_days: base skeleton (fixed for GA).
      max_conditions: how many condition slots (0..max_conditions).
      connective: 'and' or 'or' — how conditions combine.
      conditions: optional pre-built list of Condition objects (tests).
      sl_atr: ATR stop-loss multiplier (0 = no SL).
      atr_period: ATR period for the stop.
      exit_mode: how to eventually close a held position — 'same_day' or
        'end_of_week'. Manual per-strategy setting, NOT a GA gene.
      **genes: remaining kwargs are GA genes — global indicator params
        (sma_period, rsi_period, ...) plus per-slot genes (cond1_type,
        cond1_op, cond1_ind, cond1_ind2, cond1_price, cond1_price2,
        cond1_threshold, ...). The GA drops its whole params dict into the
        constructor, so these arrive as flat kwargs and are decoded via
        `_decode_conditions`.
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
        exit_mode: str = "same_day",
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
        if exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")
        self.exit_mode = exit_mode

        # Store the global indicator params (Option B) for operand resolution.
        self.global_params = {
            k: v for k, v in genes.items() if k in _GLOBAL_PARAM_KEYS
        }

        # Resolve conditions: explicit list wins over gene decoding.
        if conditions is not None:
            self.conditions = list(conditions)
        else:
            self.conditions = self._decode_conditions(genes)
        # Inject global params into each condition's indicator operands.
        for c in self.conditions:
            self._resolve_operand(c.left)
            self._resolve_operand(c.right)

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

        # Same-day exit: session day at exit hour, only on days that had an
        # entry that day. This is the primary/intended close.
        entry_days = entries.index.normalize()[entries]
        is_exit_hour = pd.Series(
            df.index.hour == self.exit_hour, index=df.index
        )
        day_has_entry = pd.Series(
            df.index.normalize().isin(entry_days), index=df.index
        )
        same_day_exits = (
            is_session & is_exit_hour & day_has_entry
        ).fillna(False).astype(bool)

        # Holding state after same-day exits (before the end-of-week fallback).
        held_before_fallback = (
            entries.astype(int).cumsum()
            - same_day_exits.astype(int).cumsum().shift(1).fillna(0)
        ) > 0

        # End-of-week fallback: if still holding at the end of the trading
        # week (literal Friday, weekday 4), force-close at exit_hour regardless
        # of whether Friday is a configured session day. Bounded hard deadline
        # — no further fallback after this.
        fallback_exits = pd.Series(False, index=df.index)
        if self.exit_mode == "end_of_week":
            is_friday = pd.Series(
                df.index.weekday == FRIDAY_WEEKDAY, index=df.index
            )
            fallback_exits = (
                is_friday & is_exit_hour & held_before_fallback
            ).fillna(False).astype(bool)

        exits = (same_day_exits | fallback_exits).fillna(False).astype(bool)

        # Final holding state (drives SL carry and position tracking).
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

    def _resolve_operand(self, op: Operand) -> None:
        """Fill an indicator operand's period/param2 from global params."""
        if op.kind != KIND_INDICATOR or op.indicator is None:
            return
        parent = INDICATOR_REGISTRY[op.indicator][0]
        period_key, param2_key = PARENT_PARAMS[parent]
        if period_key is not None:
            op.period = int(self.global_params.get(period_key, 14))
        if param2_key is not None:
            op.param2 = self.global_params.get(param2_key)


# All global param keys (for filtering genes in the constructor).
_GLOBAL_PARAM_KEYS = {
    "sma_period", "ema_period", "atr_period", "rsi_period", "cci_period",
    "stoch_k", "stoch_d", "stoch_slowing", "adx_period",
    "bb_period", "bb_stddev", "macd_fast", "macd_slow",
    "mom_period", "wpr_period", "mfi_period",
    "ichi_tenkan", "ichi_kijun", "ichi_senkou",
}


def build_param_space(
    max_conditions: int = 3,
    indicators: tuple[str, ...] | None = None,
    periods: tuple[int, ...] = (5, 10, 14, 20, 50),
    thresholds: tuple[float, ...] = (20.0, 30.0, 50.0, 70.0, 80.0),
) -> dict[str, Any]:
    """Build the GA ParamSpace dict (Option B — global per-parent params).

    Global params (shared across all slots, MQL5-handle style):
      sma_period, ema_period, atr_period, rsi_period, cci_period,
      stoch_k, stoch_d, stoch_slowing, adx_period, bb_period, bb_stddev,
      macd_fast, macd_slow, mom_period, wpr_period, mfi_period,
      ichi_tenkan, ichi_kijun, ichi_senkou
    Plus `connective` and `sl_atr`.

    Per-slot genes (each slot just picks which indicator + op + threshold):
      <i>_type:      none | price_ind | price_price | ind_const | ind_ind
      <i>_op:        gt | lt | crosses_above | crosses_below
      <i>_ind:       indicator line name
      <i>_ind2:      indicator line name (ind_ind right side)
      <i>_price:     Open | High | Low | Close (price_price / price_ind left)
      <i>_price2:    Open | High | Low | Close (price_price right)
      <i>_threshold: float (ind_const; validated per-indicator scale)

    The GA drops its whole params dict into `ComposableStrategy(**params)`;
    `max_conditions` is NOT part of the space (fixed at its constructor
    default of 3), while `connective`, the global params, and the per-slot
    genes arrive as flat kwargs and are decoded by the strategy constructor.
    """
    indicators = indicators or tuple(INDICATOR_REGISTRY.keys())
    space: dict[str, Any] = {
        "connective": list(CONNECTIVES),
        # ATR stop-loss multiplier (>= 1.0 so positions get a non-zero SL
        # distance, which the engine needs to size lots). Optimized by GA.
        "sl_atr": {"min": 1.0, "max": 5.0, "step": 0.5},
        # Global per-parent indicator params (Option B).
        "sma_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "ema_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "atr_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "rsi_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "cci_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "stoch_k": {"min": periods[0], "max": periods[-1], "step": 1},
        "stoch_d": {"min": 3, "max": 10, "step": 1},
        "stoch_slowing": {"min": 1, "max": 5, "step": 1},
        "adx_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "bb_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "bb_stddev": [1.0, 2.0, 3.0],
        "macd_fast": [5, 8, 12, 20],
        "macd_slow": [20, 26, 40, 50],
        "mom_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "wpr_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "mfi_period": {"min": periods[0], "max": periods[-1], "step": 1},
        "ichi_tenkan": [9, 12, 20],
        "ichi_kijun": [26, 30, 40],
        "ichi_senkou": [52, 60, 80],
    }
    for i in range(1, max_conditions + 1):
        p = f"cond{i}_"
        space[f"{p}type"] = [
            "none",
            "price_ind",
            "price_price",
            "ind_const",
            "ind_ind",
        ]
        space[f"{p}op"] = list(("gt", "lt", "crosses_above", "crosses_below"))
        space[f"{p}ind"] = list(indicators)
        space[f"{p}ind2"] = list(indicators)
        space[f"{p}price"] = ["Open", "High", "Low", "Close"]
        space[f"{p}price2"] = ["Open", "High", "Low", "Close"]
        space[f"{p}threshold"] = {
            "min": thresholds[0],
            "max": thresholds[-1],
            "step": 5.0,
        }
    return space