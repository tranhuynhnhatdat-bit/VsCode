# Composable Strategy — XAUUSD H1

**Base:** Long-only session trade — **BUY at 02:00, CLOSE at 23:00** on **Wednesday and Friday**, symbol **XAUUSD**.

**Gating conditions:** AT LEAST ONE of the following conditions must be true:
- `CCI(32) < 20.00`
- `Close crosses below EMA(42)`

**Stop loss:** `4.5 × ATR(14)`. No take-profit; the session close is the profit exit.

## Parameters

| Parameter | Value |
|-----------|-------|
| `connective` | `or` |
| `sl_atr` | `4.5` |
| `cond1_type` | `ind_const` |
| `cond1_op` | `lt` |
| `cond1_period` | `32` |
| `cond1_ind` | `CCI` |
| `cond1_threshold` | `20.0` |
| `cond2_type` | `price_ind` |
| `cond2_op` | `crosses_below` |
| `cond2_period` | `42` |
| `cond2_ind` | `EMA` |
| `cond2_threshold` | `40.0` |

## Strategy Logic (pseudo-code)

```
ON each closed H1 bar:
  ENTRY: day IN (Wed, Fri)
         AND hour == 01:00            # signal bar -> fill ~02:00 M1
         AND any of:
             CCI(32) < 20.00
             Close crosses below EMA(42)
         -> BUY (long) — fill at ASK (open + spread)

  EXIT : day IN (Wed, Fri)
         AND hour == 22:00            # signal bar -> fill ~23:00 M1
         AND a position was opened this session
         -> CLOSE — fill at BID (open)

  RISK : on BUY, SL = entry_ask − 4.5 × ATR(14)
```

## Narrative

### Entry — 02:00 (Wed & Fri)
- The entry signal is computed on the closed `01:00` H1 bar
  (close known at 02:00) and filled at the first M1 tick at/after
  02:00, buying at ASK (open + spread) — matching the MQL5 engine fill timing.
- Conditions (evaluated on that same bar):
  - `CCI(32) < 20.00`
  - `Close crosses below EMA(42)`

### Exit — 23:00
- The exit signal is computed on the closed `22:00` H1 bar and
  filled at the first M1 tick at/after 23:00, selling at BID (open).
- Only fires on a day that actually opened a position.

### Risk management
- **Stop loss:** fixed ATR multiple, set at entry and carried while held:
  `SL = entry_ask − 4.5 × ATR(14)`.
- **Position sizing:** fixed risk per trade (risk_money = 100 USD) →
  lots = risk_money / (SL distance × tick_value / tick_size), floored to
  volume_step; rejected if outside the broker's min/max.
- **No take-profit:** the 23:00 session close is the profit exit.
- **Direction:** long only.

## Performance

| Stage | Profit Factor | Trades | Trades/Month | Win Rate % |
|-------|:-------------:|:------:|:------------:|:----------:|
| H1 train (Stage 1) | 1.18 | 634 | 4.55 | 56.94 |
| M1 train (Stage 2) | 1.18 | 634 | 4.55 | 56.94 |
| M1 OOS1 (Stage 3) | 1.41 | 379 | 4.53 | 58.84 |
| M1 OOS2 (Stage 3) | 1.40 | 216 | 3.87 | 54.63 |

*Stage 1 = H1 fast screen (GA fitness). Stage 2 = M1 confirmation on the train
window. Stage 3 = M1 out-of-sample windows OOS1/OOS2.*
