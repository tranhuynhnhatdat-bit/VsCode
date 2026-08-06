"""SMA crossover example strategy: long/short with fixed ATR-based SL/TP.

Demonstrates the Strategy contract:
- Pure function of an HTF OHLCV DataFrame
- Long and short are exclusive; direction changes pass through a
  mandatory flat bar (exit, then re-enter the opposite side next bar)
- Fixed SL/TP set at entry from ATR multiples, carried forward while
  the position is open
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategySignals, require_ohlcv


class SmaCrossover(Strategy):
    """Fast/slow SMA crossover with fixed ATR-based stops.

    Regime: fast SMA above slow SMA -> long regime; below -> short regime.

    Signals are derived from a state machine over regime runs:
    - The first sustained regime (>= 2 bars) opens a position on the next
      bar's close.
    - A sustained regime flip closes the position on the flip bar's close
      and re-opens the opposite side on the next bar's close -> one
      mandatory flat bar between.
    - A single-bar regime (immediate flip-back) is suppressed entirely:
      no exit, no re-entry. Whipsaw noise does not churn the position.
    - SL/TP are fixed at entry from ATR multiples and carried forward.
    """

    def __init__(
        self,
        fast: int = 10,
        slow: int = 50,
        atr_period: int = 14,
        sl_atr: float = 2.0,
        tp_atr: float = 3.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr

    def generate(self, df: pd.DataFrame) -> StrategySignals:
        require_ohlcv(df)

        fast_sma = df["Close"].rolling(self.fast).mean()
        slow_sma = df["Close"].rolling(self.slow).mean()
        atr = self._atr(df)

        # Regime: +1 long, -1 short, NaN warmup / exact-equal.
        direction = np.sign(fast_sma - slow_sma)
        direction = direction.where(direction != 0)

        states = self._state_machine(direction)

        held = self._holding(
            states["entries"],
            states["exits"],
            states["short_entries"],
            states["short_exits"],
        )
        sl_stop, tp_stop = self._fixed_stops(df, atr, held["long"], held["short"])

        signals = StrategySignals(
            entries=states["entries"],
            exits=states["exits"],
            short_entries=states["short_entries"],
            short_exits=states["short_exits"],
            sl_stop=sl_stop,
            tp_stop=tp_stop,
        )
        signals.validate(df)
        return signals

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _state_machine(self, direction: pd.Series) -> dict[str, pd.Series]:
        """Derive entry/exit series from the regime series.

        Runs of constant direction are processed in order. Entries execute
        one bar after the signal bar (shift(1) at the end); exits execute
        at the flip bar's close (no shift). This guarantees every entry has
        a matching exit and no bar is both an entry and an exit.
        """
        entries_at = pd.Series(False, index=direction.index)
        exits = pd.Series(False, index=direction.index)
        short_entries_at = pd.Series(False, index=direction.index)
        short_exits = pd.Series(False, index=direction.index)

        valid = direction.notna()
        d = direction.loc[valid]
        if d.empty:
            return {
                "entries": entries_at,
                "exits": exits,
                "short_entries": short_entries_at,
                "short_exits": short_exits,
            }

        # Group consecutive equal directions into runs.
        run_id = (d != d.shift(1)).cumsum()

        pos = 0  # current held direction: 0 flat, 1 long, -1 short
        for _, grp in d.groupby(run_id):
            run_dir = 1 if grp.iloc[0] > 0 else -1
            start = grp.index[0]
            length = len(grp)

            if pos == 0:
                # Flat: open the first sustained regime.
                if length == 1:
                    continue  # single-bar blip, wait for a real regime
                if run_dir == 1:
                    entries_at.loc[start] = True
                else:
                    short_entries_at.loc[start] = True
                pos = run_dir

            elif pos == run_dir:
                # Already holding this direction: nothing to do.
                continue

            else:
                # Flip into the opposite regime.
                if length == 1:
                    continue  # single-bar blip: suppress exit and re-entry
                if pos == 1:
                    exits.loc[start] = True
                else:
                    short_exits.loc[start] = True
                if run_dir == 1:
                    entries_at.loc[start] = True
                else:
                    short_entries_at.loc[start] = True
                pos = run_dir

        # Entries execute on the close of the NEXT bar (one flat bar after
        # the exit). Exits stay on the flip bar (close of the flip bar).
        return {
            "entries": entries_at.shift(1).fillna(False).astype(bool),
            "exits": exits,
            "short_entries": short_entries_at.shift(1).fillna(False).astype(bool),
            "short_exits": short_exits,
        }

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    @staticmethod
    def _holding(
        entries: pd.Series,
        exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
    ) -> dict[str, pd.Series]:
        """Boolean series: True on bars where the position is open.

        A position is open from its entry bar through its exit bar
        (inclusive) — the exit happens at the close of the exit bar.
        """
        long = (
            entries.astype(int).cumsum()
            - exits.astype(int).cumsum().shift(1).fillna(0)
        ) > 0
        short = (
            short_entries.astype(int).cumsum()
            - short_exits.astype(int).cumsum().shift(1).fillna(0)
        ) > 0
        return {"long": long, "short": short}

    def _fixed_stops(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        long_held: pd.Series,
        short_held: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Fixed SL/TP: stop values set at each entry bar, filled forward
        while the position is open, NaN on flat bars."""
        close = df["Close"]
        entry_sl = pd.Series(np.nan, index=df.index)
        entry_tp = pd.Series(np.nan, index=df.index)

        # Long: SL below entry, TP above entry.
        long_entry = long_held & ~long_held.shift(1).fillna(False)
        entry_sl.loc[long_entry] = close.loc[long_entry] - self.sl_atr * atr.loc[long_entry]
        entry_tp.loc[long_entry] = close.loc[long_entry] + self.tp_atr * atr.loc[long_entry]

        # Short: SL above entry, TP below entry.
        short_entry = short_held & ~short_held.shift(1).fillna(False)
        entry_sl.loc[short_entry] = close.loc[short_entry] + self.sl_atr * atr.loc[short_entry]
        entry_tp.loc[short_entry] = close.loc[short_entry] - self.tp_atr * atr.loc[short_entry]

        held = long_held | short_held
        sl_stop = entry_sl.ffill().where(held)
        tp_stop = entry_tp.ffill().where(held)
        return sl_stop, tp_stop