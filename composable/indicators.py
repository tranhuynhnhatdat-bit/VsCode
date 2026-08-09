"""Pure indicator functions matching MQL5 built-in indicator semantics.

Each function takes an OHLCV DataFrame (columns Open, High, Low, Close,
Volume) and returns a pandas Series indexed identically to the input. The
formulas follow MQL5's built-in indicator implementations (`iMA`, `iRSI`,
`iATR`, `iCCI`, `iStochastic`, `iADX`, `iMomentum`, `iWPR`, `iMFI`, `iOBV`,
`iBands`, `iMACD`, `iIchimoku`) so that conditions built on top of them
match what the MQL5 backtesting engine computes.

Notes on MQL5 parity:
- All moving averages are computed on CLOSE prices (PRICE_CLOSE, the default).
- EMA uses the classic k = 2/(period+1) and is seeded by the SMA of the
  first `period` bars.
- ATR and RSI use Wilder's smoothing (alpha = 1/period), seeded by the SMA
  of the first `period` values.
- Stochastic uses MODE_SMA for both %K and %D (the MQL5 default).
- ADX follows Wilder's original recursive smoothing.
- Bollinger Bands use the population standard deviation (ddof=0), matching
  MQL5's iBands.
- MACD signal line is a simple MA of the main line (MODE_SMA), matching
  MQL5's iMACD.
- Ichimoku lines follow iIchimoku: Senkou spans are shifted forward by the
  Kijun period (read at bar i they are the values computed `kijun` bars
  earlier — no lookahead); Chikou is the close shifted back by Kijun.
- Momentum uses the difference form `Close - Close[period]` (per design
  decision; MQL5's iMomentum is the ratio form `Close/Close[period]*100`).
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


# ------------------------------------------------------------------ #
# New indicators (MQL5 parity)
# ------------------------------------------------------------------ #
def Momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Momentum: Close - Close[period] (difference form, per design).

    NOTE: MQL5's built-in iMomentum is the ratio form
    `Close / Close[period] * 100`. The difference form was chosen during
    design (thresholds around 0); it is still trivially expressible in MQL5
    as `Close[i] - Close[i + period]`.
    """
    require_ohlcv(df)
    return df["Close"] - df["Close"].shift(period)


def WPR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams Percent Range (MQL5 iWPR), range -100..0.

    WPR = -100 * (HighestHigh - Close) / (HighestHigh - LowestLow)
    """
    require_ohlcv(df)
    high = df["High"].rolling(period).max()
    low = df["Low"].rolling(period).min()
    rng = (high - low).replace(0.0, np.nan)
    return -100.0 * (high - df["Close"]) / rng


def MFI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (MQL5 iMFI), range 0..100.

    MFI = 100 - 100 / (1 + positive_money_flow / negative_money_flow)
    where positive/negative money flows are summed over `period` bars
    (classic MFI, matching MQL5's iMFI).
    """
    require_ohlcv(df)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw = tp * df["Volume"]
    prev_tp = tp.shift(1)
    pos = raw.where(tp > prev_tp, 0.0)
    neg = raw.where(tp < prev_tp, 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    # When negative money flow is 0, MFI = 100.
    mfi = mfi.where(neg_sum.fillna(0.0) > 0.0, 100.0)
    return mfi


def OBV(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume (MQL5 iOBV).

    OBV[i] = OBV[i-1] + Volume[i] if Close[i] > Close[i-1]
           = OBV[i-1] - Volume[i] if Close[i] < Close[i-1]
           = OBV[i-1]             otherwise
    """
    require_ohlcv(df)
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def Bollinger(
    df: pd.DataFrame, period: int = 20, stddev: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Bollinger Bands (MQL5 iBands): returns (upper, lower).

    middle = SMA(Close, period); bands use the population standard
    deviation (ddof=0), matching MQL5.
    """
    require_ohlcv(df)
    close = df["Close"]
    mid = SMA(close, period)
    std = close.rolling(period).std(ddof=0)
    upper = mid + stddev * std
    lower = mid - stddev * std
    return upper, lower


def MACD(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD (MQL5 iMACD): returns (main, signal).

    main = EMA(fast) - EMA(slow); signal = SMA(main, signal) (MODE_SMA).
    """
    require_ohlcv(df)
    main = EMA(df["Close"], fast) - EMA(df["Close"], slow)
    sig = SMA(main, signal)
    return main, sig


def Ichimoku(
    df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku Kinko Hyo (MQL5 iIchimoku).

    Returns (tenkan, kijun, senkou_a, senkou_b, chikou).

    MQL5 semantics:
    - Tenkan = (HH(tenkan) + LL(tenkan)) / 2
    - Kijun  = (HH(kijun) + LL(kijun)) / 2
    - Senkou A = (Tenkan + Kijun) / 2, shifted FORWARD by kijun bars.
      Read at bar i it is the value computed kijun bars earlier (no lookahead).
    - Senkou B = (HH(senkou) + LL(senkou)) / 2, shifted forward by kijun bars.
    - Chikou = Close shifted BACK by kijun bars (a lagging line).
    """
    require_ohlcv(df)
    high, low, close = df["High"], df["Low"], df["Close"]
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    senkou_a = ((tenkan_line + kijun_line) / 2.0).shift(kijun)
    senkou_b = (
        (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2.0
    ).shift(kijun)
    chikou = close.shift(kijun)
    return tenkan_line, kijun_line, senkou_a, senkou_b, chikou