"""GoldSession: long-only daily session strategy on XAUUSD.

Buys at 02:00 and closes at 23:00 on Wednesday and Friday, with an
ATR-based stop loss and volume/trend/regime entry filters.

Design (from grilling session):
- H1 computation, M1 execution via the engine's one-period shift
- Entry signal on the 01:00 H1 bar -> engine fills at ~02:00 M1
- Exit signal on the 22:00 H1 bar -> engine fills at ~23:00 M1
- Long-only; Wednesday and Friday only
- ATR(14) stop loss, fixed at entry, carried while held
- No take profit: the 23:00 close is the profit exit
- Filters (all must pass at the entry bar):
    volume: H1 volume > volume_mult * hour-aligned rolling mean
            (same hour-of-day over the last volume_lookback days)
    trend:  ADX(adx_period) > adx_threshold
    regime: ATR(14) > rolling mean(ATR(14), regime_lookback)
- If the SL closes the position intraday, no re-entry that day
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategySignals, require_ohlcv


class GoldSession(Strategy):
    """Long-only Wed/Fri 02:00-23:00 session with ATR SL and filters."""

    def __init__(
        self,
        sl_atr: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        volume_lookback: int = 20,
        volume_mult: float = 1.5,
        regime_lookback: int = 20,
        atr_period: int = 14,
        entry_hour: int = 1,
        exit_hour: int = 22,
        session_days: tuple[int, ...] = (2, 4),
    ) -> None:
        self.sl_atr = sl_atr
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_lookback = volume_lookback
        self.volume_mult = volume_mult
        self.regime_lookback = regime_lookback
        self.atr_period = atr_period
        self.entry_hour = entry_hour
        self.exit_hour = exit_hour
        self.session_days = session_days

    def generate(self, df: pd.DataFrame) -> StrategySignals:
        require_ohlcv(df)

        atr = self._atr(df)
        adx = self._adx(df, self.adx_period)

        # Hour-aligned rolling mean: average volume at the same hour-of-day
        # over the last volume_lookback days. At 01:00 this compares against
        # the typical 01:00 volume, not the 24h average (which is dominated
        # by London/NY peak hours and would never pass at 02:00).
        vol_mean = (
            df["Volume"]
            .groupby(df.index.hour)
            .transform(lambda s: s.rolling(self.volume_lookback).mean())
        )

        # Filters (all evaluated at the entry bar).
        volume_ok = df["Volume"] > self.volume_mult * vol_mean
        trend_ok = adx > self.adx_threshold
        regime_ok = atr > atr.rolling(self.regime_lookback).mean()

        # Entry: session day at the entry hour, all filters pass.
        is_session = pd.Series(
            df.index.weekday.isin(self.session_days), index=df.index
        )
        is_entry_hour = pd.Series(
            df.index.hour == self.entry_hour, index=df.index
        )
        entries = (
            is_session & is_entry_hour & volume_ok & trend_ok & regime_ok
        ).fillna(False).astype(bool)

        # Exit: session day at the exit hour, but only on days that had an
        # entry. Firing an unconditional exit every session would let exit
        # counts accumulate in the holding-state cumsums and break the
        # entry/SL pairing, even though a flat-day exit is a harmless no-op
        # for vectorbt itself.
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

        # Holding state: from entry bar through exit bar (inclusive).
        held = (
            entries.astype(int).cumsum()
            - exits.astype(int).cumsum().shift(1).fillna(0)
        ) > 0

        # Fixed ATR stop, set at entry, carried while held. No TP.
        entry_sl = pd.Series(np.nan, index=df.index)
        long_entry = held & ~held.shift(1).fillna(False)
        entry_sl.loc[long_entry] = (
            df["Close"].loc[long_entry] - self.sl_atr * atr.loc[long_entry]
        )
        sl_stop = entry_sl.ffill().where(held)
        tp_stop = pd.Series(np.nan, index=df.index)

        signals = StrategySignals(
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=df.index),
            short_exits=pd.Series(False, index=df.index),
            sl_stop=sl_stop,
            tp_stop=tp_stop,
        )
        signals.validate(df)
        return signals

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

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Wilder's ADX."""
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        return dx.ewm(alpha=alpha, adjust=False).mean()