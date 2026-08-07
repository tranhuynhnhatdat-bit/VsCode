# GoldSession — Wed/Fri 02:00–23:00 Long Session on XAUUSD

## Strategy logic

A long-only, intraday session strategy on gold (XAUUSD). It buys at
**02:00** and closes at **23:00** on **Wednesday and Friday** only, with an
ATR-based stop loss and three entry filters.

### Entry

- **Day:** Wednesday and Friday only.
- **Time:** 02:00 (M1 fill).
- **Direction:** long only.
- **Filters** (all must pass at the entry bar, computed on H1):
  1. **Volume filter** — H1 volume at the entry bar is above
     `volume_mult` × the **hour-aligned** rolling mean of H1 volume:
     the average volume at the same hour-of-day over the last
     `volume_lookback` days (a raw 24-hour rolling mean would be
     dominated by London/NY peak hours and would never pass at 02:00).
  2. **Trend filter** — ADX(`adx_period`) is above `adx_threshold`
     (only trade when the market is trending).
  3. **Regime filter** — ATR(14) is above its own rolling mean over the
     last `regime_lookback` bars (only trade when volatility is
     expanding).

### Exit

- **Time:** 23:00 same day (M1 fill).
- **Stop loss:** fixed at entry at `sl_atr` × ATR(14) below the entry
  price, carried while the position is open.
- **No take profit** — the 23:00 close is the profit exit.

### Re-entry

If the stop loss closes the position intraday, there is **no re-entry
that day**. The next opportunity is the next session day (Friday, or the
following Wednesday).

## Fill mapping (H1 signals → M1 execution)

The backtesting engine maps H1 signals to M1 fills with a one-period
shift (an H1 bar's close is only known one hour later). To buy at
**02:00 M1**, the strategy sets its entry signal on the **01:00 H1 bar**;
the engine fills at the first M1 bar after 02:00. Similarly, the exit
signal is set on the **22:00 H1 bar** so the fill lands at **23:00 M1**.

## Parameters

| Parameter          | Default | Description                                        |
| ------------------ | ------- | -------------------------------------------------- |
| `sl_atr`           | 2.0     | Stop loss = `sl_atr` × ATR(14) below entry         |
| `adx_period`       | 14      | ADX lookback period                                |
| `adx_threshold`    | 20.0    | ADX must exceed this to trade                       |
| `volume_lookback`  | 20      | Rolling volume mean window (H1 bars)               |
| `volume_mult`      | 1.5     | Volume must exceed `volume_mult` × mean            |
| `regime_lookback`  | 20      | ATR rolling mean window (H1 bars)                  |
| `atr_period`       | 14      | ATR period (fixed)                                 |
| `entry_hour`       | 1       | H1 bar whose close is known at 02:00 (fill hour)   |
| `exit_hour`        | 22      | H1 bar whose close is known at 23:00 (fill hour)   |
| `session_days`     | (2, 4)  | Weekdays: 2 = Wednesday, 4 = Friday                |

## Optimization

The genetic optimizer (`TestEngine`) sweeps `sl_atr`, `adx_period`,
`adx_threshold`, `volume_lookback`, `volume_mult`, and
`regime_lookback`. Entry/close times and session days are fixed.

Passing strategies are saved to `results/` as equity-curve PNGs plus a
summary JSON (`*_optimization_summary.json`) containing each passing
strategy's parameters and train/OOS1/OOS2 metrics.