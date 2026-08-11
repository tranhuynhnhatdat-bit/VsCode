# Composable Strategy Reference

## Overview

The composable strategy system builds trading strategies from three layers:

1. **Base skeleton** — fixed time/session logic (entry hour, exit hour, session days)
2. **Conditions** — boolean expressions combining prices, indicators, and constants
3. **Indicators** — pure MQL5-parity indicator functions

The GA (genetic algorithm) composes conditions from a pool of indicators and operators, then ANDs/ORs them with the base skeleton.

---

## 1. Base Skeleton (fixed, not optimized by GA)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entry_hour` | 1 | H1 bar whose close triggers entry (entry fires at hour+1) |
| `exit_hour` | 22 | H1 bar whose close triggers exit (exit fires at hour+1) |
| `session_days` | (2, 4) | Python weekday numbers: Mon=0, Tue=1, **Wed=2**, Thu=3, **Fri=4** |
| `sl_atr` | 2.0 | Stop-loss = ATR × this multiplier |
| `atr_period` | 14 | ATR lookback period |
| `max_conditions` | 3 | Number of condition slots (0 = pure time) |
| `connective` | "and" | How conditions combine: `"and"` or `"or"` |
| `exit_mode` | "same_day" | How a held position is eventually closed: `"same_day"` or `"end_of_week"`. Manual per-strategy setting, **not** a GA gene. |

**Entry logic:** Session day AND entry_hour bar Close < Open (bearish) → BUY at ASK on the first M1 tick at hour+1.

**Exit logic:** Session day at exit_hour → close position at BID on the first M1 tick at hour+1.

**Exit modes:**
- `same_day` (default): close at `exit_hour` on session days only. No fallback.
- `end_of_week`: same-day close stays primary; if still holding at the end of the
  trading week, force-close at `exit_hour` on **literal Friday** (Python weekday 4),
  regardless of whether Friday is a session day. Bounded hard deadline — a position
  never carries past the week. Mirrors the MQL5 `InpExitMode` input.

---

## 2. Condition Types

Each condition slot can be one of 5 types. The GA selects the type per slot.

| Type | Left Side | Right Side | Example |
|------|-----------|------------|---------|
| `none` | — | — | Slot disabled (no condition) |
| `price_ind` | Price (O/H/L/C) | Price-scale indicator | `Close > SMA(14)` |
| `price_price` | Price (O/H/L/C) | Price (O/H/L/C) | `Close > Open` |
| `ind_const` | Oscillator-scale indicator | Constant threshold | `RSI(14) < 30` |
| `ind_ind` | Indicator | Indicator (same scale) | `SMA(10) > EMA(20)` |

### Operators

| Operator | Meaning |
|----------|---------|
| `gt` | Greater than (`>`) |
| `lt` | Less than (`<`) |
| `crosses_above` | Crossed above (previous bar ≤, current bar >) |
| `crosses_below` | Crossed below (previous bar ≥, current bar <) |

### Scale Rules (enforced at construction)

- **price_ind**: Right side must be a **price-scale** indicator (SMA, EMA, BB_Upper, BB_Lower, Tenkan, Kijun, SenkouA, SenkouB, Chikou)
- **ind_const**: Left side must be an **oscillator-scale** indicator (RSI, CCI, Stoch_K, Stoch_D, ADX, PlusDI, MinusDI, Momentum, WPR, MFI)
- **ind_ind**: Both sides must share the **same scale** (both price, both oscillator, or both distance)
- **price_price**: Always valid (any price vs any price)

---

## 3. Price Operands

Available for `price_ind` and `price_price` condition types.

| Operand | Column | Description |
|---------|--------|-------------|
| `Open` | `df["Open"]` | Bar open price |
| `High` | `df["High"]` | Bar high price |
| `Low` | `df["Low"]` | Bar low price |
| `Close` | `df["Close"]` | Bar close price |

---

## 4. Indicator Registry

> **Source of truth:** the tables below are derived from the single
> `INDICATOR_REGISTRY` in `composable/conditions.py` (each line is an
> `IndicatorSpec` carrying parent, scale, threshold, global param keys, and
> the compute callable). Add or change an indicator line there only; the
> GA param space, condition scale checks, and the strategy's global param
> resolution all derive from it.

### 4.1 Price-Scale Indicators

These produce values in the same range as price. Used in `price_ind` and `ind_ind` conditions.

| Indicator Line | Parent | Period Param | Param2 | MQL5 Function |
|----------------|--------|-------------|--------|---------------|
| `SMA` | SMA | `sma_period` | — | `iMA(MODE_SMA, PRICE_CLOSE)` |
| `EMA` | EMA | `ema_period` | — | `iMA(MODE_EMA, PRICE_CLOSE)` |
| `BB_Upper` | Bollinger | `bb_period` | `bb_stddev` | `iBands(upper)` |
| `BB_Lower` | Bollinger | `bb_period` | `bb_stddev` | `iBands(lower)` |
| `Tenkan` | Ichimoku | `ichi_tenkan` | `ichi_kijun` | `iIchimoku(Tenkan)` |
| `Kijun` | Ichimoku | `ichi_tenkan` | `ichi_kijun` | `iIchimoku(Kijun)` |
| `SenkouA` | Ichimoku | `ichi_tenkan` | `ichi_kijun` | `iIchimoku(SenkouA)` |
| `SenkouB` | Ichimoku | `ichi_tenkan` | `ichi_kijun` | `iIchimoku(SenkouB)` |
| `Chikou` | Ichimoku | `ichi_tenkan` | `ichi_kijun` | `iIchimoku(Chikou)` |

### 4.2 Oscillator-Scale Indicators

These produce values in a bounded range (typically 0–100 or -100–100). Used in `ind_const` and `ind_ind` conditions.

| Indicator Line | Parent | Period Param | Param2 | Range | Thresholds | MQL5 Function |
|----------------|--------|-------------|--------|-------|------------|---------------|
| `RSI` | RSI | `rsi_period` | — | 0–100 | 20, 30, 50, 70, 80 | `iRSI(PRICE_CLOSE)` |
| `CCI` | CCI | `cci_period` | — | -200–+200 | -200, -100, 100, 200 | `iCCI` |
| `Stoch_K` | Stochastic | `stoch_k` | `stoch_d` | 0–100 | 20, 30, 50, 70, 80 | `iStochastic(%K)` |
| `Stoch_D` | Stochastic | `stoch_k` | `stoch_d` | 0–100 | 20, 30, 50, 70, 80 | `iStochastic(%D)` |
| `ADX` | ADX | `adx_period` | — | 0–100 | 20, 30, 50, 70, 80 | `iADX(main)` |
| `PlusDI` | ADX | `adx_period` | — | 0–100 | 20, 30, 50, 70, 80 | `iADX(+DI)` |
| `MinusDI` | ADX | `adx_period` | — | 0–100 | 20, 30, 50, 70, 80 | `iADX(-DI)` |
| `Momentum` | Momentum | `mom_period` | — | unbounded | -100, -50, 0, 50, 100 | `iMomentum` (difference form) |
| `WPR` | WPR | `wpr_period` | — | -100–0 | -80, -50, -20 | `iWPR` |
| `MFI` | MFI | `mfi_period` | — | 0–100 | 20, 30, 50, 70, 80 | `iMFI` |

### 4.3 Distance-Scale Indicators

These produce values in price-distance units. Used in `ind_ind` conditions (both sides must be distance-scale).

| Indicator Line | Parent | Period Param | Param2 | MQL5 Function |
|----------------|--------|-------------|--------|---------------|
| `ATR` | ATR | `atr_period` | — | `iATR` |
| `MACD_Main` | MACD | `macd_fast` | `macd_slow` | `iMACD(main)` |
| `MACD_Signal` | MACD | `macd_fast` | `macd_slow` | `iMACD(signal)` |
| `OBV` | OBV | — | — | `iOBV` |

---

## 5. Global Parameter Keys (GA Option B)

These are shared across all condition slots (MQL5-handle style). The GA optimizes them once, and all conditions that use the same parent indicator inherit the same period.

| Key | Type | Default | Used By |
|-----|------|---------|---------|
| `sma_period` | int (5–50) | 14 | SMA |
| `ema_period` | int (5–50) | 14 | EMA |
| `atr_period` | int (5–50) | 14 | ATR |
| `rsi_period` | int (5–50) | 14 | RSI |
| `cci_period` | int (5–50) | 14 | CCI |
| `stoch_k` | int (5–50) | 14 | Stochastic %K period |
| `stoch_d` | int (3–10) | 3 | Stochastic %D period |
| `stoch_slowing` | int (1–5) | 3 | Stochastic slowing |
| `adx_period` | int (5–50) | 14 | ADX, PlusDI, MinusDI |
| `bb_period` | int (5–50) | 20 | Bollinger Bands |
| `bb_stddev` | float [1.0, 2.0, 3.0] | 2.0 | Bollinger Bands stddev |
| `macd_fast` | int [5, 8, 12, 20] | 12 | MACD fast EMA |
| `macd_slow` | int [20, 26, 40, 50] | 26 | MACD slow EMA |
| `mom_period` | int (5–50) | 14 | Momentum |
| `wpr_period` | int (5–50) | 14 | WPR |
| `mfi_period` | int (5–50) | 14 | MFI |
| `ichi_tenkan` | int [9, 12, 20] | 9 | Ichimoku Tenkan |
| `ichi_kijun` | int [26, 30, 40] | 26 | Ichimoku Kijun |
| `ichi_senkou` | int [52, 60, 80] | 52 | Ichimoku Senkou B |

---

## 6. Per-Slot Genes (GA)

Each condition slot `i` (1..max_conditions) has these genes:

| Gene | Values | Description |
|------|--------|-------------|
| `cond{i}_type` | `none`, `price_ind`, `price_price`, `ind_const`, `ind_ind` | Condition type |
| `cond{i}_op` | `gt`, `lt`, `crosses_above`, `crosses_below` | Comparison operator |
| `cond{i}_ind` | Any indicator line name | Left indicator (for indicator-based types) |
| `cond{i}_ind2` | Any indicator line name | Right indicator (for `ind_ind` only) |
| `cond{i}_price` | `Open`, `High`, `Low`, `Close` | Left price (for `price_ind`, `price_price`) |
| `cond{i}_price2` | `Open`, `High`, `Low`, `Close` | Right price (for `price_price` only) |
| `cond{i}_threshold` | float (20–80, step 5) | Constant threshold (for `ind_const` only) |

---

## 7. MQL5 Parity Notes

All indicators are implemented to match MQL5 built-in behavior:

| Indicator | MQL5 Parity Detail |
|-----------|-------------------|
| **SMA** | Simple moving average on Close (PRICE_CLOSE) |
| **EMA** | k = 2/(period+1), seeded by SMA of first `period` bars |
| **ATR** | Wilder's smoothing (alpha = 1/period), seeded by SMA |
| **RSI** | Wilder's smoothing, RSI = 100 when avg_loss = 0 |
| **CCI** | TP = (H+L+C)/3, 0.015 × mean deviation |
| **Stochastic** | MODE_SMA for both %K and %D |
| **ADX** | Wilder's original recursive smoothing |
| **Bollinger** | Population stddev (ddof=0) |
| **MACD** | Signal line = SMA of main line (MODE_SMA) |
| **Ichimoku** | Senkou spans shifted forward by Kijun period; Chikou shifted back |
| **Momentum** | Difference form: `Close - Close[period]` (MQL5 uses ratio form) |

---

## 8. Example Conditions

```
# Pure time (no conditions)
Close < Open on entry_hour bar → BUY

# Price vs indicator
Close > SMA(20) AND Close < BB_Upper(20, 2.0)

# Price vs price
Close > Open (bullish bar)

# Oscillator vs threshold
RSI(14) < 30 (oversold) OR Stoch_K(14,3) > 80 (overbought)

# Indicator vs indicator (same scale)
SMA(10) > EMA(20)  (both price-scale)
ADX(14) > PlusDI(14)  (both oscillator-scale)
MACD_Main(12,26) > MACD_Signal(12,26)  (both distance-scale)

# Cross detection
RSI(14) crosses_above 30  (just exited oversold)
Close crosses_above SMA(20)  (price broke above moving average)