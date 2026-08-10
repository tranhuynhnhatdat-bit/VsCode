"""Run the GoldSession 3-stage optimization on XAUUSD and save passing strategies.

Stage 1: GA on H1 (train window only) — fast screen.
Stage 2: M1 confirmation (same train window) — M1 PF > 1.1.
Stage 3: M1 OOS1/OOS2 gates.

Saves to results/:
- One equity-curve PNG per passing strategy
- optimization summary JSON (params + H1/M1 train + M1 OOS metrics)
- a CSV of all passing strategies' params and metrics for quick review
"""

from __future__ import annotations

import csv
from pathlib import Path

from optimization.engine import TestEngine
from optimization.genetic import GAConfig
from strategy.examples.gold_session import GoldSession

RESULTS_DIR = Path(r"C:\Users\DAT\Desktop\VsCode\results")

# Full data range (2003-05-05 -> 2026-08-03).
START = "2003-05-05"
END = "2026-08-03"

# Stage 1: H1 train gates (loosened; M1 confirmation is the real filter).
HTF_TRAIN_GATES = {
    "profit_factor": 1.1,
    "n_trades": 100.0,
    "win_rate": 35.0,
}

# Stage 2: M1 confirmation — simple absolute profit-factor gate.
M1_CONFIRM_PF = 1.1

# Stage 3: M1 OOS gates.
M1_OOS_GATES = {
    "oos1": {"profit_factor": 1.0, "trades_per_month": 1.0},
    "oos2": {"profit_factor": 1.1, "trades_per_month": 1.0},
}

# GA param space: entry/close times and session days are fixed.
PARAM_SPACE = {
    "sl_atr": {"min": 1.0, "max": 5.0, "step": 0.5},
    "adx_period": {"min": 14, "max": 28, "step": 2},
    "adx_threshold": {"min": 15, "max": 35, "step": 5},
    "volume_lookback": {"min": 10, "max": 50, "step": 5},
    "volume_mult": {"min": 1.0, "max": 2.0, "step": 0.1},
    "regime_lookback": {"min": 10, "max": 50, "step": 5},
    # Per-filter on/off toggles: the GA decides whether to use each filter
    # (all off = pure time-based session trade).
    "use_volume_filter": [True, False],
    "use_trend_filter": [True, False],
    "use_regime_filter": [True, False],
}


def save_metrics_csv(
    result, path: Path = RESULTS_DIR / "gold_session_passing_metrics.csv"
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
        strategy_class=GoldSession,
        param_space=PARAM_SPACE,
        split=(0.30, 0.50, 0.20),
        fitness_criterion="pf",
        htf_train_gates=HTF_TRAIN_GATES,
        m1_confirm_pf=M1_CONFIRM_PF,
        m1_oos_gates=M1_OOS_GATES,
        strategy_name="gold_session",
        results_dir=RESULTS_DIR,
        start=START,
        end=END,
        # Fixed risk money per trade (matches MQL5 InpRiskMoney).
        risk_money=100.0,
        # H1 train backtests are fast (~1-2s each), so a real GA is tractable.
        ga_config=GAConfig(
            population=50,
            generations=20,
            tournament_k=3,
            elitism=2,
            mutation_rate=0.10,
            early_stop_generations=3,
            seed=None,
            workers=6,
        ),
    )
    result = optimizer.optimize()

    csv_path = save_metrics_csv(result)
    print(f"Passing strategies: {len(result.passing)}")
    print(f"Metrics CSV: {csv_path}")

    # Per-stage funnel summary.
    if result.stage_results:
        n1 = sum(1 for s in result.stage_results if s.htf_train_pass)
        n2 = sum(1 for s in result.stage_results if s.m1_confirm_pass)
        n3 = sum(1 for s in result.stage_results if s.m1_oos_pass)
        print(
            f"Stage funnel: {len(result.stage_results)} unique -> "
            f"{n1} pass H1 -> {n2} pass M1 confirm -> {n3} pass M1 OOS"
        )

    for i, p in enumerate(result.passing):
        print(f"\n--- Passing #{i} ---")
        print(f"  params: {p.params}")
        print(f"  htf_train: {p.htf_train_metrics}")
        print(f"  m1_train:  {p.m1_train_metrics}")
        print(f"  m1_oos1:   {p.m1_oos1_metrics}")
        print(f"  m1_oos2:   {p.m1_oos2_metrics}")
        print(f"  folder:    {p.equity_curve_png.parent}")
        print(f"  png:       {p.equity_curve_png}")


if __name__ == "__main__":
    main()