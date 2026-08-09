"""Render a passing strategy's params + metrics into a human-readable,
MQL5-portable strategy.md document.

The document is human-readable AND portable to MQL5: it includes the full
base skeleton with parameter values, the composed conditions (lossless
from `params`), narrative entry/exit/risk sections, a compact pseudo-code
block, and the per-stage performance summary. An LLM can regenerate the
exact params dict from the document without the JSON.
"""

from __future__ import annotations

from pathlib import Path


def describe_condition(c) -> str:
    """Human-readable description of a Condition."""
    from composable.conditions import (
        KIND_CLOSE,
        KIND_CONST,
        KIND_HIGH,
        KIND_INDICATOR,
        KIND_LOW,
        KIND_OPEN,
    )

    def side_str(side) -> str:
        if side.kind == KIND_CONST:
            return f"{side.value:.2f}"
        if side.kind == KIND_OPEN:
            return "Open"
        if side.kind == KIND_HIGH:
            return "High"
        if side.kind == KIND_LOW:
            return "Low"
        if side.kind == KIND_CLOSE:
            return "Close"
        # Indicator: format with its shared params.
        if side.period is not None and side.param2 is not None:
            return f"{side.indicator}({side.period}, {side.param2})"
        if side.period is not None:
            return f"{side.indicator}({side.period})"
        return f"{side.indicator}"

    op_names = {
        "gt": ">",
        "lt": "<",
        "crosses_above": "crosses above",
        "crosses_below": "crosses below",
    }
    return f"{side_str(c.left)} {op_names[c.op]} {side_str(c.right)}"


def render_strategy_md(
    folder: Path,
    params: dict,
    strat,
    metrics_map: dict,
) -> Path:
    """Write a self-contained strategy.md next to a passing strategy's outputs."""
    path = folder / "strategy.md"

    # Human-readable conditions.
    cond_lines = [f"- `{describe_condition(c)}`" for c in strat.conditions]
    if not cond_lines:
        cond_lines.append("- *(none — pure time/session entry)*")
    cond_md = "\n".join(cond_lines)
    cond_nested_md = "\n".join(f"  {l}" for l in cond_lines)

    connective_word = (
        "ALL of the following conditions must be true"
        if strat.connective == "and"
        else "AT LEAST ONE of the following conditions must be true"
    )
    cond_any = "all of" if strat.connective == "and" else "any of"

    # Parameter table (lossless: every GA param with its value).
    params_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in params.items())

    # Performance table.
    perf_rows = [
        "| Stage | Profit Factor | Trades | Trades/Month | Win Rate % |",
        "|-------|:-------------:|:------:|:------------:|:----------:|",
    ]
    for label, m in (
        ("H1 train (Stage 1)", metrics_map["htf_train"]),
        ("M1 train (Stage 2)", metrics_map["m1_train"]),
        ("M1 OOS1 (Stage 3)", metrics_map["m1_oos1"]),
        ("M1 OOS2 (Stage 3)", metrics_map["m1_oos2"]),
    ):

        def _fmt(v):
            return f"{v:.2f}" if isinstance(v, float) else str(v)

        perf_rows.append(
            f"| {label} | {_fmt(m.get('profit_factor'))} | "
            f"{_fmt(m.get('n_trades'))} | {_fmt(m.get('trades_per_month'))} | "
            f"{_fmt(m.get('win_rate'))} |"
        )
    perf_md = "\n".join(perf_rows)

    entry_fill = strat.entry_hour + 1  # H1 signal -> M1 fill at +1h
    exit_fill = strat.exit_hour + 1

    cond_block = "\n".join(
        f"             {describe_condition(c)}" for c in strat.conditions
    ) or "             (none)"

    md = f"""# Composable Strategy — XAUUSD H1

**Base:** Long-only session trade — **BUY at {entry_fill:02d}:00, CLOSE at {exit_fill:02d}:00** on **Wednesday and Friday**, symbol **XAUUSD**.

**Gating conditions:** {connective_word}:
{cond_md}

**Stop loss:** `{strat.sl_atr:.1f} × ATR({strat.atr_period})`. No take-profit; the session close is the profit exit.

## Parameters

| Parameter | Value |
|-----------|-------|
{params_md}

## Strategy Logic (pseudo-code)

```
ON each closed H1 bar:
  ENTRY: day IN (Wed, Fri)
         AND hour == {strat.entry_hour:02d}:00            # signal bar -> fill ~{entry_fill:02d}:00 M1
         AND {cond_any}:
{cond_block}
         -> BUY (long) — fill at ASK (open + spread)

  EXIT : day IN (Wed, Fri)
         AND hour == {strat.exit_hour:02d}:00            # signal bar -> fill ~{exit_fill:02d}:00 M1
         AND a position was opened this session
         -> CLOSE — fill at BID (open)

  RISK : on BUY, SL = entry_ask − {strat.sl_atr:.1f} × ATR({strat.atr_period})
```

## Narrative

### Entry — {entry_fill:02d}:00 (Wed & Fri)
- The entry signal is computed on the closed `{strat.entry_hour:02d}:00` H1 bar
  (close known at {entry_fill:02d}:00) and filled at the first M1 tick at/after
  {entry_fill:02d}:00, buying at ASK (open + spread) — matching the MQL5 engine fill timing.
- Conditions (evaluated on that same bar):
{cond_nested_md}

### Exit — {exit_fill:02d}:00
- The exit signal is computed on the closed `{strat.exit_hour:02d}:00` H1 bar and
  filled at the first M1 tick at/after {exit_fill:02d}:00, selling at BID (open).
- Only fires on a day that actually opened a position.

### Risk management
- **Stop loss:** fixed ATR multiple, set at entry and carried while held:
  `SL = entry_ask − {strat.sl_atr:.1f} × ATR({strat.atr_period})`.
- **Position sizing:** fixed risk per trade (risk_money = 100 USD) →
  lots = risk_money / (SL distance × tick_value / tick_size), floored to
  volume_step; rejected if outside the broker's min/max.
- **No take-profit:** the {exit_fill:02d}:00 session close is the profit exit.
- **Direction:** long only.

## Performance

{perf_md}

*Stage 1 = H1 fast screen (GA fitness). Stage 2 = M1 confirmation on the train
window. Stage 3 = M1 out-of-sample windows OOS1/OOS2.*
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path