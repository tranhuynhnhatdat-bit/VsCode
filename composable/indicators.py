"""Pure indicator functions matching MQL5 built-in indicator semantics.

Each function takes an OHLCV DataFrame (columns Open, High, Low, Close,
Volume) and returns a pandas Series indexed identically to the input. The
formulas follow MQL5's built-in indicator implementations (`iMA`, `iRSI`,
`iATR`, `iCCI`, `iStochastic`, `iADX`) so that conditions built on top of
them match what the MQL5 backtesting engine computes.

Notes on MQL5 parity:
- All moving averages are computed on CLOSE prices (PRICE_CLOSE, the default).
- EMA uses the classic k = 2/(period+1) and is seeded by the SMA of the
  first `period` bars.
- ATR and RSI use Wilder's smoothing (alpha = 1/period), seeded by the SMA
  of the first `period` values.
- Stochastic uses MODE_SMA for both %K and %D (the MQL5 default).
- ADX follows Wilder's original recursive smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import require_ohlcv


def SMA(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average on `series` (MQL5 MODE_SMA)."""
    return series.rolling(period).mean()


def EMA(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (MQL5 MODE_EMA, PRICE_CLOSE).

    k = 2 / (period + 1); seeded by the SMA of the first `period` values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    return series.ewm(span=period, adjust=False).mean()


def _wilders_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: EMA with alpha = 1/period, seeded by SMA."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (MQL5 iATR, Wilder's smoothing)."""
    require_ohlcv(df)
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
    return _wilders_smooth(tr, period)


def RSI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index (MQL5 iRSI, PRICE_CLOSE, Wilder's)."""
    require_ohlcv(df)
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilders_smooth(gain, period)
    avg_loss = _wilders_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss is 0 (price never fell), RSI = 100.
    rsi = rsi.where(avg_loss.fillna(0.0) > 0.0, 100.0)
    return rsi


def CCI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Commodity Channel Index (MQL5 iCCI).

    TP = (H + L + C) / 3
    CCI = (TP - SMA(TP, period)) / (0.015 * mean_deviation)
    """
    require_ohlcv(df)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(period).mean()
    # Mean deviation over the same window as the SMA.
    mean_dev = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma) / (0.015 * mean_dev.replace(0.0, np.nan))


def Stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator (MQL5 iStochastic, MODE_SMA for both).

    Returns (K, D) where K is the %K line and D is the %D line.
    %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    %D = SMA(%K, d_period)
    """
    require_ohlcv(df)
    low = df["Low"].rolling(k_period).min()
    high = df["High"].rolling(k_period).max()
    rng = (high - low).replace(0.0, np.nan)
    k_raw = 100.0 * (df["Close"] - low) / rng
    # Slowing: MQL5 applies a simple average of the raw %K over `slowing`.
    k = k_raw.rolling(slowing).mean()
    d = k.rolling(d_period).mean()
    return k, d


# ------------------------------------------------------------------ #
# ADX (Wilder's) — matches GoldSession._adx() and MQL5 iADX.
# ------------------------------------------------------------------ #
def _plus_minus_di(df: pd.DataFrame, period: int):
    """Return (atr, plus_di, minus_di) using Wilder's smoothing."""
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

    atr = _wilders_smooth(tr, period)
    plus_di = 100.0 * _wilders_smooth(plus_dm, period) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilders_smooth(minus_dm, period) / atr.replace(0.0, np.nan)
    return atr, plus_di, minus_di


def ADX(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (MQL5 iADX, Wilder's)."""
    require_ohlcv(df)
    _atr, plus_di, minus_di = _plus_minus_di(df, period)
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
    return _wilders_smooth(dx, period)


def PlusDI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """+DI line (MQL5 iADX main buffer 1)."""
    require_ohlcv(df)
    _atr, plus_di, _minus_di = _plus_minus_di(df, period)
    return plus_di


def MinusDI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """-DI line (MQL5 iADX main buffer 2)."""
    require_ohlcv(df)
    _atr, _plus_di, minus_di = _plus_minus_di(df, period)
    return minus_di