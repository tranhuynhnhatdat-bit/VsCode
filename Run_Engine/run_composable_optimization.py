"""Run the ComposableStrategy 3-stage optimization on XAUUSD and save passing strategies.

The base skeleton is fixed (02:00 entry / 23:00 exit, Wed/Fri long-only),
matching GoldSession. The GA composes up to 3 conditions drawn from the
indicator pool (SMA, EMA, ATR, RSI, CCI, Stochastic, ADX), combined with a
single global AND/OR connective, ANDed with the base time logic.

Stage 1: GA on H1 (train window only) — fast screen.
Stage 2: M1 confirmation (same train window).
Stage 3: M1 OOS1/OOS2 gates.

Saves to results/:
- One equity-curve PNG per passing strategy
- optimization summary JSON (params + metrics)
- a metrics CSV for quick review
- a human-readable strategy.md per passing strategy

Run from the repo root:
    python Run_Engine/run_composable_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root even though this is in a subdir.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv

from composable.composable import ComposableStrategy, build_param_space
from optimization.engine import TestEngine
from optimization.genetic import GAConfig
from Render.strategy_md import describe_condition, render_strategy_md

RESULTS_DIR = ROOT / "results"

# Full data range (2003-05-05 -> 2026-08-03).
START = "2003-05-05"
END = "2026-08-03"

# Stage 1: H1 train gates (loosened; M1 confirmation is the real filter).
HTF_TRAIN_GATES = {
    "profit_factor": 1.1,
    "n_trades": 100.0,
    "win_rate": 35.0,
}

# Stage 2: M1 confirmation ratio.
M1_CONFIRM_RATIO = 0.9
M1_PF_CAP = 10.0

# Stage 3: M1 OOS gates.
M1_OOS_GATES = {
    "oos1": {"profit_factor": 1.0, "trades_per_month": 2.0},
    "oos2": {"profit_factor": 1.1, "trades_per_month": 2.0},
}

# GA param space: condition slots + global connective. Base skeleton fixed.
# 2 condition slots (was 3) so random AND-combinations can actually trade.
PARAM_SPACE = build_param_space(
    max_conditions=2,
    indicators=("SMA", "EMA", "ATR", "RSI", "CCI", "Stochastic", "ADX"),
    periods=(5, 10, 14, 20, 50),
    thresholds=(20.0, 30.0, 50.0, 70.0, 80.0),
)

# Seed the GA with the pure-time individual (all condition slots = none) so
# a baseline trade always exists and the GA has a non-zero fitness to build on.
SEED_INDIVIDUAL = {
    "connective": "and",
    "sl_atr": 2.0,
    "cond1_type": "none",
    "cond1_op": "gt",
    "cond1_period": 14,
    "cond1_ind": "RSI",
    "cond1_threshold": 50.0,
    "cond2_type": "none",
    "cond2_op": "gt",
    "cond2_period": 14,
    "cond2_ind": "RSI",
    "cond2_threshold": 50.0,
}


def save_metrics_csv(
    result, path: Path = RESULTS_DIR / "composable_passing_metrics.csv"
) -> Path:
    """Write one row per passing strategy: params + all stage metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in result.passing:
        row = {}
        for k, v in p.params.items():
            row[f"param_{k}"] = v
        for window, metrics in (
            ("htf_train", p.htf_train_metrics),
            ("m1_train", p.m1_train_metrics),
            ("m1_oos1", p.m1_oos1_metrics),
            ("m1_oos2", p.m1_oos2_metrics),
        ):
            for k, v in metrics.items():
                row[f"{window}_{k}"] = v
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("no_passing_strategies\n")
    return path


def main() -> None:
    optimizer = TestEngine(
        symbol="XAUUSD",
        timeframe="H1",
        strategy_class=ComposableStrategy,
        param_space=PARAM_SPACE,
        split=(0.30, 0.50, 0.20),
        fitness_criterion="pf",
        htf_train_gates=HTF_TRAIN_GATES,
        m1_confirm_ratio=M1_CONFIRM_RATIO,
        m1_pf_cap=M1_PF_CAP,
        m1_oos_gates=M1_OOS_GATES,
        strategy_name="composable",
        results_dir=RESULTS_DIR,
        start=START,
        end=END,
        risk_money=100.0,
        ga_config=GAConfig(
            population=50,
            generations=20,
            tournament_k=3,
            elitism=2,
            mutation_rate=0.10,
            early_stop_generations=3,
            seed=None,
            workers=6,
            initial_population=[SEED_INDIVIDUAL],
        ),
    )
    result = optimizer.optimize()

    csv_path = save_metrics_csv(result)
    print(f"Passing strategies: {len(result.passing)}")
    print(f"Metrics CSV: {csv_path}")

    if result.stage_results:
        n1 = sum(1 for s in result.stage_results if s.htf_train_pass)
        n2 = sum(1 for s in result.stage_results if s.m1_confirm_pass)
        n3 = sum(1 for s in result.stage_results if s.m1_oos_pass)
        print(
            f"Stage funnel: {len(result.stage_results)} unique -> "
            f"{n1} pass H1 -> {n2} pass M1 confirm -> {n3} pass M1 OOS"
        )

    # Human-readable description of each passing strategy's conditions.
    for i, p in enumerate(result.passing):
        strat = ComposableStrategy(**p.params)
        print(f"\n--- Passing #{i} ---")
        print(f"  connective: {strat.connective}")
        for c in strat.conditions:
            print(f"    cond: {describe_condition(c)}")
        print(f"  htf_train: {p.htf_train_metrics}")
        print(f"  m1_train:  {p.m1_train_metrics}")
        print(f"  m1_oos1:   {p.m1_oos1_metrics}")
        print(f"  m1_oos2:   {p.m1_oos2_metrics}")
        print(f"  folder:    {p.equity_curve_png.parent}")
        md_path = render_strategy_md(
            p.equity_curve_png.parent,
            p.params,
            strat,
            {
                "htf_train": p.htf_train_metrics,
                "m1_train": p.m1_train_metrics,
                "m1_oos1": p.m1_oos1_metrics,
                "m1_oos2": p.m1_oos2_metrics,
            },
        )
        print(f"  strategy.md: {md_path}")


if __name__ == "__main__":
    main()