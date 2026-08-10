"""Run the ComposableStrategy 4-stage optimization on XAUUSD and save passing strategies.

This is a VALIDATION variant of run_composable_optimization.py. It applies the
same pipeline but with:
  - Reduced GA cost (population, generations, islands, workers, collection target)
  - Behavioral diversity (collect only strategies whose actual H1-train trade
    set differs from every already-collected strategy) instead of the vacuous
    gene-Hamming gate
  - No-op rejection (drop individuals with no active conditions, so we stop
    saving pure-time baseline look-alikes as "passing" strategies)

The event-driven engine (Stage 4) still runs on the FULL data range, and all
filter thresholds are IDENTICAL to run_composable_optimization.py (no tightening).

Writes to results_validation/ so the original results/ are not clobbered.
Includes a verification block comparing inert-strategy counts before/after.

Run from the repo root:
    python Run_Engine/run_composable_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root even though this is in a subdir.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json
import time

import pandas as pd

from composable.composable import ComposableStrategy, build_param_space
from optimization.engine import TestEngine, PassingStrategy
from optimization.genetic import GAConfig
from Render.strategy_md import describe_condition, render_strategy_md
from backtest.event_engine import validate_with_event_engine
from data_manager import DataManager

# Separate output dir so the original results/ are untouched.
RESULTS_DIR = ROOT / "results_validation"

# Full data range (2003-05-05 -> 2026-08-03).
START = "2003-05-05"
END = "2026-08-03"

# ------------------------------------------------------------------ #
# Filters IDENTICAL to run_composable_optimization.py (no tightening).
# ------------------------------------------------------------------ #
# Phase A: H1 train gates.
HTF_TRAIN_GATES = {
    "profit_factor": 1.1,
    "n_trades": 100.0,
    "win_rate": 35.0,
}

# Phase B: M1 confirmation — simple absolute profit-factor gate.
M1_CONFIRM_PF = 1.1

# Phase B: M1 OOS gates.
M1_OOS_GATES = {
    "oos1": {"profit_factor": 1.0, "trades_per_month": 2.0},
    "oos2": {"profit_factor": 1.1, "trades_per_month": 2.0},
}

# Phase B: event-driven final gate (Stage 4).
EVENT_FILTERS = {
    "profit_factor": 1.3,
    "win_rate": 35.0,
    "max_drawdown_pct": -15.0,
}

# ------------------------------------------------------------------ #
# Reduced GA cost (the anti-overfit lever requested).
# ------------------------------------------------------------------ #
COLLECT_TARGET = 60
MAX_COLLECT_EVALUATIONS = 6_000

# GA param space: global per-parent indicator params (Option B) + per-slot
# condition genes. Base skeleton fixed. 2 condition slots.
PARAM_SPACE = build_param_space(
    max_conditions=2,
    periods=(5, 10, 14, 20, 50),
    thresholds=(20.0, 30.0, 50.0, 70.0, 80.0),
)

# Seed the GA with the pure-time individual (all condition slots = none) so
# a baseline trade always exists and the GA has a non-zero fitness to build on.
SEED_INDIVIDUAL = {
    "connective": "and",
    "sl_atr": 2.0,
    # Global per-parent indicator params (Option B).
    "sma_period": 14,
    "ema_period": 14,
    "atr_period": 14,
    "rsi_period": 14,
    "cci_period": 14,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_slowing": 3,
    "adx_period": 14,
    "bb_period": 20,
    "bb_stddev": 2.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "mom_period": 14,
    "wpr_period": 14,
    "mfi_period": 14,
    "ichi_tenkan": 9,
    "ichi_kijun": 26,
    "ichi_senkou": 52,
    # Per-slot genes (all none = pure time).
    "cond1_type": "none",
    "cond1_op": "gt",
    "cond1_ind": "RSI",
    "cond1_ind2": "SMA",
    "cond1_price": "Close",
    "cond1_price2": "Open",
    "cond1_threshold": 50.0,
    "cond2_type": "none",
    "cond2_op": "gt",
    "cond2_ind": "RSI",
    "cond2_ind2": "SMA",
    "cond2_price": "Close",
    "cond2_price2": "Open",
    "cond2_threshold": 50.0,
}


class BehaviorTestEngine(TestEngine):
    """TestEngine that collects by BEHAVIORAL diversity and rejects no-ops.

    The base TestEngine's `_collect_stage1` uses a gene-level Hamming gate
    (`diversity_threshold=1`), which counts gene differences even when they do
    not change behavior. Two strategies differing only in `cond1_ind` while
    `cond1_type="none"` are "diverse" but trade identically. This subclass
    replaces that gate with a check on the ACTUAL H1-train trade set, and
    refuses to collect individuals with no active conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Collected strategy key -> behavioral trade signature.
        self._collected_sigs: dict[str, tuple] = {}

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #
    def _collect_stage1(self, params: dict, fitness: float) -> None:
        """Called by the GA for each evaluated individual.

        Adds `params` to the shared collected set only if it:
          - passes the H1 train gate, AND
          - has at least one active condition (no-op rejection), AND
          - is behaviorally diverse vs every collected strategy.
        """
        if len(self._collected) >= self.collect_target:
            return
        key = json.dumps(params, sort_keys=True, default=str)
        if key in self._collected_keys:
            return

        # No-op rejection: require at least one real (decoded) condition.
        if self._active_condition_count(params) < 1:
            return

        # H1 train gate check (cached backtest).
        wm = self._window_metrics(
            self._evaluate_htf(params),
            self._train_start,
            self._train_end,
            self._htf_train.index,
        )
        passed, _reason = self._htf_train_check(wm)
        if not passed:
            return

        # Behavioral diversity: must differ from every collected strategy in
        # its actual H1-train trade set (entry timestamps + count).
        sig = self._trade_signature(params)
        for csig in self._collected_sigs.values():
            if csig == sig:
                return

        self._collected.append(dict(params))
        self._collected_keys.add(key)
        self._collected_metrics[key] = wm
        self._collected_sigs[key] = sig
        print(
            f"  [COLLECT] {len(self._collected)}/{self.collect_target} "
            f"behaviorally-diverse H1-train survivors"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _active_condition_count(self, params: dict) -> int:
        """Number of real (decoded) conditions in the strategy."""
        strat = self.strategy_class(**params)
        return len(getattr(strat, "conditions", []) or [])

    def _trade_signature(self, params: dict) -> tuple:
        """Hashable signature of the H1-train closed-trade set.

        Two strategies with identical behavior produce the same closed-trade
        entry times, so comparing this signature is a true behavioral check.
        """
        result = self._evaluate_htf(params)
        trades = result.trades
        if trades is None or trades.empty:
            return ("empty",)
        closed = trades[trades["status"] == 1]
        if closed.empty:
            return ("empty",)
        entry_times = self._htf_train.index[
            closed["entry_idx"].astype(int).values
        ]
        # (n_trades, first-50 entry times, last-50 entry times) as strings.
        times = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in entry_times]
        return (len(closed), tuple(times[:50]), tuple(times[-50:]))


# ------------------------------------------------------------------ #
# Shared helpers (same as run_composable_optimization.py)
# ------------------------------------------------------------------ #
def _jsonable(value):
    """Convert inf/nan to None for JSON serialization."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return None
    return value


def save_strategy_folder(p: PassingStrategy, index: int) -> Path:
    """Write one folder per event-passing strategy."""
    folder = RESULTS_DIR / f"strategy_{index}"
    folder.mkdir(parents=True, exist_ok=True)

    ev = p.event_metrics
    eq = p.event_equity_curve
    trades = p.event_trades

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = folder / "equity_curve.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    if eq is not None and len(eq):
        ax.plot(eq.index, eq.values, lw=1.2)
    ax.set_title(
        f"Event Engine Equity — {p.params.get('connective', '')} "
        f"XAUUSD H1 | params={p.params}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    trades_path = folder / "trades.csv"
    if trades is not None and not trades.empty:
        trades.to_csv(trades_path, index=False)
    else:
        with open(trades_path, "w", newline="", encoding="utf-8") as f:
            f.write("no_trades\n")

    strategy_path = folder / "strategy.json"
    data = {
        "params": _jsonable(p.params),
        "htf_train": _jsonable(p.htf_train_metrics),
        "m1_train": _jsonable(p.m1_train_metrics),
        "m1_oos1": _jsonable(p.m1_oos1_metrics),
        "m1_oos2": _jsonable(p.m1_oos2_metrics),
        "event": _jsonable(ev),
        "event_pass": p.event_pass,
        "event_fail_reasons": p.event_fail_reasons,
    }
    with open(strategy_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return folder


def save_metrics_csv(passing: list[PassingStrategy]) -> Path:
    """Write one row per event-passing strategy: params + all stage metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "composable_passing_metrics.csv"
    rows = []
    for p in passing:
        row = {}
        for k, v in p.params.items():
            row[f"param_{k}"] = v
        for window, metrics in (
            ("htf_train", p.htf_train_metrics),
            ("m1_train", p.m1_train_metrics),
            ("m1_oos1", p.m1_oos1_metrics),
            ("m1_oos2", p.m1_oos2_metrics),
            ("event", p.event_metrics),
        ):
            if metrics:
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


# ------------------------------------------------------------------ #
# Verification: inert / duplicate strategy detection
# ------------------------------------------------------------------ #
def _is_pure_time(row: dict) -> bool:
    """True if the strategy row has no active (non-none) condition slots."""
    return (
        str(row.get("param_cond1_type", "none")) == "none"
        and str(row.get("param_cond2_type", "none")) == "none"
    )


def count_inert_strategies(csv_path: Path) -> dict:
    """Read a passing-metrics CSV and report inert/duplicate strategies."""
    report = {"path": str(csv_path), "n_rows": 0, "pure_time": 0, "dup_trade_sets": 0}
    if not csv_path.exists():
        report["path"] += " (missing)"
        return report

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    report["n_rows"] = len(rows)
    if not rows:
        return report

    report["pure_time"] = sum(1 for r in rows if _is_pure_time(r))

    # Behavioral duplicates: same htf_train_n_trades + m1_oos1_n_trades +
    # event_n_trades (identical underlying trade behavior).
    seen: dict[tuple, int] = {}
    for r in rows:
        sig = (
            r.get("htf_train_n_trades"),
            r.get("m1_oos1_n_trades"),
            r.get("event_n_trades"),
        )
        seen[sig] = seen.get(sig, 0) + 1
    report["dup_trade_sets"] = sum(max(0, c - 1) for c in seen.values())
    return report


def verify_before_after() -> None:
    """Compare old results/ vs new results_validation/ for inert strategies."""
    old_csv = ROOT / "results" / "composable_passing_metrics.csv"
    new_csv = RESULTS_DIR / "composable_passing_metrics.csv"

    print("\n=== Verification: inert / duplicate strategies ===")
    old_r = count_inert_strategies(old_csv)
    new_r = count_inert_strategies(new_csv)
    print(f"  BEFORE ({Path(old_r['path']).name}):")
    print(f"    rows={old_r['n_rows']}, pure_time={old_r['pure_time']}, "
          f"dup_trade_sets={old_r['dup_trade_sets']}")
    print(f"  AFTER ({Path(new_r['path']).name}):")
    print(f"    rows={new_r['n_rows']}, pure_time={new_r['pure_time']}, "
          f"dup_trade_sets={new_r['dup_trade_sets']}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main() -> None:
    t_total = time.time()

    optimizer = BehaviorTestEngine(
        symbol="XAUUSD",
        timeframe="H1",
        strategy_class=ComposableStrategy,
        param_space=PARAM_SPACE,
        constraints=[("macd_fast", "<", "macd_slow")],
        split=(0.30, 0.50, 0.20),
        fitness_criterion="pf",
        htf_train_gates=HTF_TRAIN_GATES,
        m1_confirm_pf=M1_CONFIRM_PF,
        m1_oos_gates=M1_OOS_GATES,
        strategy_name="composable",
        results_dir=RESULTS_DIR,
        start=START,
        end=END,
        risk_money=100.0,
        ga_config=GAConfig(
            population=20,
            generations=8,
            tournament_k=3,
            elitism=2,
            mutation_rate=0.10,
            early_stop_generations=3,
            seed=None,
            workers=3,
            initial_population=[SEED_INDIVIDUAL],
            islands=2,
            migration_interval=8,
            migration_count=2,
            restart_stagnation=3,
        ),
        collect_target=COLLECT_TARGET,
        diversity_threshold=1,  # unused by BehaviorTestEngine._collect_stage1
        max_collect_evaluations=MAX_COLLECT_EVALUATIONS,
    )

    # ------------------------------------------------------------------ #
    # Phase A: GA collection (Stage 1 only).
    # ------------------------------------------------------------------ #
    print(f"\n=== Phase A: GA collection (Stage 1) — target {COLLECT_TARGET} ===")
    result = optimizer.optimize()
    collected = result.passing
    print(f"  Collected {len(collected)} unique diverse H1-train survivors")

    if not collected:
        print("  No Stage-1 survivors collected — aborting.")
        return

    # ------------------------------------------------------------------ #
    # Phase B: M1 funnel (Stage 2 + Stage 3).
    # ------------------------------------------------------------------ #
    print(f"\n=== Phase B-1: M1 funnel (Stage 2 + Stage 3) — {len(collected)} candidates ===")
    t2_start = time.time()
    m1_survivors = optimizer.run_m1_funnel(collected)
    t2_end = time.time()
    print(f"  [TIMING] {t2_end-t2_start:.1f}s")
    print(f"  [PASSING] {len(m1_survivors)} / {len(collected)} passed M1 confirm + OOS")

    if not m1_survivors:
        print("  No M1-survivors — aborting.")
        return

    # ------------------------------------------------------------------ #
    # Phase B-2: Event-driven validation (Stage 4) on FULL data + saving.
    # ------------------------------------------------------------------ #
    print(f"\n=== Phase B-2: Event-Driven Validation (Stage 4) — {len(m1_survivors)} candidates ===")
    t3_start = time.time()
    dm = DataManager()
    h1_df = dm.load("XAUUSD", "H1", start=START, end=END)
    m1_df = dm.load("XAUUSD", "M1", start=START, end=END)

    event_passing: list[PassingStrategy] = []
    for i, p in enumerate(m1_survivors):
        print(f"  Validating #{i} ... ", end="", flush=True)
        ev_result = validate_with_event_engine(
            params=p.params,
            m1_df=m1_df,
            h1_df=h1_df,
            symbol="XAUUSD",
            initial_capital=10_000.0,
            risk_money=100.0,
            ticks_per_bar=20,
            filters=EVENT_FILTERS,
        )
        p.event_metrics = ev_result["result"]["metrics"]
        p.event_equity_curve = ev_result["result"]["equity_curve"]
        p.event_trades = ev_result["result"]["trades"]
        p.event_pass = ev_result["passed"]
        p.event_fail_reasons = ev_result["fail_reasons"]

        if ev_result["passed"]:
            event_passing.append(p)
            print(
                f"PASSED (PF={ev_result['result']['profit_factor']:.2f}, "
                f"WR={ev_result['result']['win_rate']:.1f}%, "
                f"DD={ev_result['result']['max_drawdown']:.1f}%)"
            )
        else:
            reasons = "; ".join(ev_result["fail_reasons"])
            print(f"FAILED: {reasons}")

    t3_end = time.time()
    print(f"\n=== Phase B-2 Summary ===")
    print(f"  [TIMING] {t3_end-t3_start:.1f}s")
    print(
        f"  [PASSING] {len(event_passing)} / {len(m1_survivors)} "
        f"passed event validation"
    )

    # ------------------------------------------------------------------ #
    # Save folders ONLY for event-passing strategies.
    # ------------------------------------------------------------------ #
    print(f"\n=== Saving event-passing strategies ===")
    for idx, p in enumerate(event_passing):
        folder = save_strategy_folder(p, idx)
        p.equity_curve_png = folder / "equity_curve.png"
        print(f"  Saved #{idx} -> {folder}")

    csv_path = save_metrics_csv(event_passing)
    print(f"  Metrics CSV: {csv_path}")

    # ------------------------------------------------------------------ #
    # Final summary + strategy.md for event-passing strategies.
    # ------------------------------------------------------------------ #
    print(f"\n=== Total ===")
    print(f"  [TIMING] {time.time()-t_total:.1f}s (all phases)")
    print(f"  [PASSING] {len(event_passing)} strategies passed all 4 stages")

    for i, p in enumerate(event_passing):
        strat = ComposableStrategy(**p.params)
        print(f"\n--- Passing #{i} ---")
        print(f"  connective: {strat.connective}")
        for c in strat.conditions:
            print(f"    cond: {describe_condition(c)}")
        print(f"  htf_train: {p.htf_train_metrics}")
        print(f"  m1_train:  {p.m1_train_metrics}")
        print(f"  m1_oos1:   {p.m1_oos1_metrics}")
        print(f"  m1_oos2:   {p.m1_oos2_metrics}")
        print(f"  event:     {p.event_metrics}")
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
                "event": p.event_metrics,
            },
        )
        print(f"  strategy.md: {md_path}")

    # ------------------------------------------------------------------ #
    # Verification: inert / duplicate strategy comparison.
    # ------------------------------------------------------------------ #
    verify_before_after()


if __name__ == "__main__":
    main()