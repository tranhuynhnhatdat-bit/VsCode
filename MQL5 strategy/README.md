# MQL5 Strategy 31 — Cross-Engine Notes (Agent Memory)

This file exists so **no agent ever forgets** the following hard-won facts about
matching the Python backtest engine to the MQL5 Strategy Tester.

---

## 1. Weekday mapping — MQL5 vs Python (CRITICAL)

`InpSessionDays` is a comma-separated list of **developer-day numbers**.
The two engines number the days differently:

| Day | Python (pandas `dayofweek`) | MQL5 (`TimeDayOfWeek`/`dt.day_of_week`) |
|-----|:---------------------------:|:----------------------------------------:|
| Sunday | 6 | 0 |
| Monday | 0 | 1 |
| Tuesday | 1 | 2 |
| **Wednesday** | **2** | **3** |
| Thursday | 3 | 4 |
| **Friday** | **4** | **5** |
| Saturday | 5 | 6 |

- strategy_31 uses **Wed + Fri**.
- In MQL5 that is **`InpSessionDays=3,5`** — NOT `2,4`.
- The user's backtest confirms: `InpSessionDays=3,5` is correct.
- **Do NOT "fix" it back to 2,4.**

---

## 2. Python vs MQL5 — reported performance (strategy_31, XAUUSD)

| Metric | Python engine | MQL5 (report as first run) | MQL5 (expected after fix) |
|--------|:-------------:|:--------------------------:|:-------------------------:|
| Trades | 897 | 895 | ~897 |
| Win rate | 56.6% | 38.88% | ~56.6% |
| Profit factor | 1.44 | 1.13 | ~1.44 |
| Max drawdown | −10.0% | −39.30% | ~−10% |
| Max holding | ~21 h | 305:42:40 | ~21 h |

The first MQL5 run had **895 trades** (entry logic correct) but the
**23:00 same-day exit was unreliable** — many positions rode for days until
their SL, crushing win rate / PF / drawdown.

---

## 3. Known bug (FIXED in `Composable_Strategy_31.mq5`)

### Root cause
The original EA detected entry/exit hours from `CopyRates(_Symbol, _Period, 0, 2, mrate)`
with `mrate[1]` treated as the "last closed bar". Depending on the tester
timeframe (report ran on **H1**, not M1 as the header comment assumed) and
tick timing, this made the `hour == InpExitHour + 1` check unreliable, so the
23:00 exit sometimes never fired.

### Fix
OnTick now reads the **current forming M1 bar** directly:
```mql5
datetime curTime = iTime(_Symbol, PERIOD_M1, 0);
```
and derives `day_of_week` / `hour` from it. This is **timeframe-agnostic** —
the EA behaves identically whether the tester chart is M1, H1, or anything else.

- **Entry:** first tick at/after 02:00 (hour == InpEntryHour + 1), no open position.
  Signal H1 bar = today 01:00 (built from `StructToTime`, not `curTime - 3600`):
  `Close < Open` → BUY at ASK, SL = 1.5 × ATR(17) below.
- **Exit:** first tick at/after 23:00 (hour == InpExitHour + 1), position open
  **and opened the same calendar day** (matches Python engine `session` logic).
- Entry/Exit no longer depend on the fragile `_sqIsBarOpen`/`lastBarTime` gate
  (kept only as an early-out optimization).

### Run settings to use (unchanged from before)
- Symbol: XAUUSD (report used XAUUSD_QDM on FTMO-Demo)
- Any tester period (M1 or H1 both work now)
- Model: Every tick (or 1 minute OHLC for speed)
- From 2003.05.05 → To 2026.07.31, deposit 10000, 1:100
- Inputs: InpRiskMoney=100, InpSlAtr=1.5, InpAtrPeriod=17,
  InpEntryHour=1, InpExitHour=22, **InpSessionDays=3,5**, InpMagicNumber=31

---

## 4. Expected equivalence guarantees

To keep the two engines matched, the MQL5 EA must:
- BUY at **ASK** (open + spread), close at **BID** — matches engine fills.
- SL = ask − 1.5 × ATR(17); SL trailing/gap check was dropped in the SQX-style
  rewrite (fixed SL only, matching the Python `sl` column semantics).
- Sizing: `lots = risk_money / (sl_distance × tick_value / tick_size)`,
  floored to `volume_step`, **rejected** (never clamped) if outside
  `[volume_min, volume_max]`.
- One position at a time; positions only close at 23:00 same session or by SL.

---

## 5. Exit mode input (`InpExitMode`)

Every EA has an `input ENUM_EXIT_MODE InpExitMode` (default `EXIT_SAME_DAY`):

| Value | Behavior |
|-------|----------|
| `EXIT_SAME_DAY` (default) | Close at the session's exit hour (23:00) on session days only. No fallback. Matches the original behavior. |
| `EXIT_END_OF_WEEK` | Same-day 23:00 close stays primary. **If still holding at the end of the trading week, force-close at 23:00 on literal Friday** (MQL5 weekday `5`), regardless of whether Friday is a configured session day. Bounded hard deadline — a position never carries past the week. |

Key implementation note: the end-of-week close is checked **before** the
`IsSessionDay` gate in `OnTick`, so it fires even on a non-session Friday.
Entries and the normal same-day exit remain session-day-gated.

This mirrors the Python `ComposableStrategy(exit_mode=...)` setting, keeping
the two engines in parity. It is a manual per-strategy input — **not** a
GA-optimized gene.
