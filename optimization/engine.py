"""TestEngine: two-phase optimization pipeline.

Phase A — GA collection (Stage 1 only):
  Island-model GA on the strategy's native timeframe (HTF), train window only.
  - Fitness = profit factor on HTF train-window closed trades.
  - Each evaluated individual is checked against the H1 train gate
    (profit_factor, n_trades, win_rate) AND a behavioral-diversity gate
    (Hamming distance over condition-slot genes vs the shared collected set).
  - Survivors are collected into a SHARED set across all islands.
  - The GA stops when the shared set reaches `collect_target` (default 500)
    or the evaluation budget backstop is exhausted.
  - No strategy folders are saved here — saving happens only after the
    event-driven validation (Stage 4).

Phase B — M1 funnel (fixed collected set):
  `run_m1_funnel()` runs the collected survivors through:
    - M1 confirmation on the train window (ratio gate)
    - M1 OOS1/OOS2 gates (per-window PF and trades/month)
  Returns the survivors with their stage metrics attached. The caller
  (run_composable_optimization.py) then runs event-driven validation and
  saves folders only for event-passing strategies.

Windows are split by timestamp once (30% OOS1 / 50% train / 20% OOS2 of the
date range) and used identically for HTF and M1.
"""

from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, RESULTS_DIR
from data_manager import DataManager
from optimization.genetic import GAConfig, GAReport, GeneticOptimizer, ParamSpace
from strategy.base import Strategy

# Profit factor cap so fitness can never be infinite.
PF_CAP = 10.0
# Supported fitness criteria.
CRITERIA = ("pf_n_trades", "pf", "return")
# Default H1 train gates (all strictly-greater-than).
DEFAULT_HTF_TRAIN_GATES: dict[str, float] = {
    "profit_factor": 1.1,
    "n_trades": 100.0,
    "win_rate": 35.0,
}
# Default M1 OOS gates.
DEFAULT_M1_OOS_GATES: dict[str, dict[str, float]] = {
    "oos1": {"profit_factor": 1.0, "trades_per_month": 1.0},
    "oos2": {"profit_factor": 1.1, "trades_per_month": 1.0},
}
# Condition-slot genes used for behavioral diversity (the composed filters).
CONDITION_GENES = (
    "connective",
    "cond1_type", "cond1_op", "cond1_ind", "cond1_ind2",
    "cond1_price", "cond1_price2", "cond1_threshold",
    "cond2_type", "cond2_op", "cond2_ind", "cond2_ind2",
    "cond2_price", "cond2_price2", "cond2_threshold",
    "cond3_type", "cond3_op", "cond3_ind", "cond3_ind2",
    "cond3_price", "cond3_price2", "cond3_threshold",
)
DAYS_PER_MONTH = 30.44


# ------------------------------------------------------------------ #
# Module-level helpers (also used by parallel worker processes)
# ------------------------------------------------------------------ #
def _compute_window_metrics(
    result: BacktestResult,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    index: pd.DatetimeIndex,
    initial_capital: float,
) -> dict[str, float]:
    """Gate metrics from closed trades whose ENTRY falls in the window."""
    zero = {
        "n_trades": 0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "trades_per_month": 0.0,
        "avg_trade_pct": 0.0,
        "total_return_pct": 0.0,
    }
    trades = result.trades
    if trades is None or trades.empty:
        return zero
    closed = trades[trades["status"] == 1]
    if closed.empty:
        return zero

    entry_times = index[closed["entry_idx"].astype(int).values]
    mask = (entry_times >= start_ts) & (entry_times <= end_ts)
    sub = closed[mask]
    n = int(len(sub))
    if n == 0:
        return zero

    pnls = sub["pnl"].astype(float)
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    span_days = max((end_ts - start_ts).days, 1)
    return {
        "n_trades": n,
        "profit_factor": float(pf),
        "win_rate": float((pnls > 0).mean() * 100),
        "trades_per_month": n / (span_days / DAYS_PER_MONTH),
        "avg_trade_pct": float(sub["return"].astype(float).mean() * 100),
        "total_return_pct": float(pnls.sum() / initial_capital * 100),
    }


def _compute_criterion_score(wm: dict[str, float], criterion: str) -> float:
    """Fitness score from window metrics (module-level for workers)."""
    pf = wm["profit_factor"]
    if pf == float("inf"):
        pf = PF_CAP
    if criterion == "pf":
        return float(pf)
    if criterion == "return":
        return float(wm["total_return_pct"])
    # Smooth penalty for trading fewer than 100 times.
    return float(pf * min(1.0, wm["n_trades"] / 100.0))


def _condition_signature(params: dict[str, Any]) -> tuple[Any, ...]:
    """Behavioral-diversity signature: the condition-slot genes."""
    return tuple(params.get(g) for g in CONDITION_GENES)


def _condition_hamming(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Number of differing condition-slot genes between two param sets."""
    sa, sb = _condition_signature(a), _condition_signature(b)
    return sum(1 for x, y in zip(sa, sb) if x != y)


# ------------------------------------------------------------------ #
# Parallel worker state (per-process, set by _worker_init)
# ------------------------------------------------------------------ #
_worker_state: dict[str, Any] = {}


def _worker_init(payload: dict[str, Any]) -> None:
    """Initialize a worker process: engine + sliced HTF + window bounds."""
    _worker_state["engine"] = BacktestEngine(
        symbol=payload["symbol"],
        timeframe=payload["timeframe"],
        risk_money=payload["risk_money"],
        initial_capital=payload["initial_capital"],
        strategy_name=payload["strategy_name"],
    )
    _worker_state["strategy_class"] = payload["strategy_class"]
    _worker_state["htf"] = payload["htf_train"]
    _worker_state["train_start"] = payload["train_start"]
    _worker_state["train_end"] = payload["train_end"]
    _worker_state["fitness_criterion"] = payload["fitness_criterion"]


def _worker_fitness(params: dict[str, Any]) -> float:
    """Evaluate one individual's H1-train fitness in a worker process."""
    strategy = _worker_state["strategy_class"](**params)
    signals = strategy.generate(_worker_state["htf"])
    result = _worker_state["engine"].run_htf(signals, _worker_state["htf"])
    wm = _compute_window_metrics(
        result,
        _worker_state["train_start"],
        _worker_state["train_end"],
        _worker_state["htf"].index,
        _worker_state["engine"].initial_capital,
    )
    return _compute_criterion_score(wm, _worker_state["fitness_criterion"])


@dataclass
class PassingStrategy:
    """A strategy's journey through the pipeline.

    Stage metrics are populated only as the strategy passes each stage.
    `hm_train_metrics` is set during Phase A; the M1 metrics are set during
    Phase B; event fields are set by the caller after event validation.
    """

    params: dict[str, Any]
    fitness: float
    htf_train_metrics: dict[str, float] | None = None
    m1_train_metrics: dict[str, float] | None = None
    m1_oos1_metrics: dict[str, float] | None = None
    m1_oos2_metrics: dict[str, float] | None = None
    event_metrics: dict[str, Any] | None = None
    event_pass: bool | None = None
    event_fail_reasons: list[str] = field(default_factory=list)
    event_equity_curve: Any = None  # pd.Series from the event engine
    event_trades: Any = None  # pd.DataFrame from the event engine
    equity_curve_png: Path | None = None


@dataclass
class OptimizationResult:
    """Result of a TestEngine Phase-A optimization run (Stage-1 survivors)."""

    report: GAReport
    passing: list[PassingStrategy]
    summary_path: Path | None = None


class TestEngine:
    """Runs Phase A (GA collection) + provides Phase B (M1 funnel)."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        strategy_class: type[Strategy],
        param_space: dict[str, Any],
        split: tuple[float, float, float] = (0.30, 0.50, 0.20),
        constraints: list[tuple[str, str, str]] | None = None,
        initial_capital: float = 10_000.0,
        risk_money: float = 100.0,
        strategy_name: str = "optimized",
        ga_config: GAConfig | None = None,
        fitness_criterion: str = "pf",
        htf_train_gates: dict[str, float] | None = None,
        m1_confirm_ratio: float = 0.9,
        m1_pf_cap: float = 10.0,
        m1_confirm_pf: float = 1.1,
        m1_oos_gates: dict[str, dict[str, float]] | None = None,
        results_dir: Path = RESULTS_DIR,
        start: str | None = None,
        end: str | None = None,
        collect_target: int = 500,
        diversity_threshold: int = 1,
        max_collect_evaluations: int = 50_000,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.split = tuple(split)
        self.constraints = constraints
        self.initial_capital = initial_capital
        self.risk_money = risk_money
        self.strategy_name = strategy_name
        self.ga_config = ga_config or GAConfig()
        self.results_dir = Path(results_dir)
        self.start = start
        self.end = end
        self.collect_target = collect_target
        self.diversity_threshold = diversity_threshold
        self.max_collect_evaluations = max_collect_evaluations

        if (
            len(self.split) != 3
            or any(f <= 0 for f in self.split)
            or abs(sum(self.split) - 1.0) > 1e-9
        ):
            raise ValueError(
                f"split must be 3 positive fractions summing to 1, "
                f"got {self.split}"
            )
        if fitness_criterion not in CRITERIA:
            raise ValueError(
                f"fitness_criterion must be one of {CRITERIA}"
            )
        self.fitness_criterion = fitness_criterion

        self.htf_train_gates = htf_train_gates or DEFAULT_HTF_TRAIN_GATES
        self.m1_confirm_ratio = m1_confirm_ratio
        self.m1_pf_cap = m1_pf_cap
        self.m1_confirm_pf = m1_confirm_pf
        self.m1_oos_gates = m1_oos_gates or DEFAULT_M1_OOS_GATES
        bad = set(self.m1_oos_gates) - {"oos1", "oos2"}
        if bad:
            raise ValueError(
                f"m1_oos_gates keys must be oos1/oos2, got {sorted(bad)}"
            )

        # Filled by _load_data() during optimize().
        self._htf: pd.DataFrame | None = None
        self._m1: pd.DataFrame | None = None
        self._htf_train: pd.DataFrame | None = None
        self._htf_m1_oos1: pd.DataFrame | None = None
        self._htf_m1_oos2: pd.DataFrame | None = None
        self._m1_train_index: pd.DatetimeIndex | None = None
        self._m1_oos1_index: pd.DatetimeIndex | None = None
        self._m1_oos2_index: pd.DatetimeIndex | None = None
        self._oos1_start = self._oos1_end = pd.Timestamp(0)
        self._train_start = self._train_end = pd.Timestamp(0)
        self._oos2_start = self._oos2_end = pd.Timestamp(0)
        self._htf_cache: dict[str, BacktestResult] = {}
        self._m1_train_cache: dict[str, BacktestResult] = {}
        self._m1_oos1_cache: dict[str, BacktestResult] = {}
        self._m1_oos2_cache: dict[str, BacktestResult] = {}
        self._engine: BacktestEngine | None = None
        self._pool: ProcessPoolExecutor | None = None

        # Phase A shared collection state.
        self._collected: list[dict[str, Any]] = []
        self._collected_keys: set[str] = set()
        self._collected_metrics: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimize(self) -> OptimizationResult:
        """Phase A: island GA collecting Stage-1 survivors. No folders saved."""
        t_start = time.time()

        self._load_data()
        self._engine = BacktestEngine(
            symbol=self.symbol,
            timeframe=self.timeframe,
            risk_money=self.risk_money,
            initial_capital=self.initial_capital,
            strategy_name=self.strategy_name,
        )
        self._verify_slicing()

        # Reset collection state (in case optimize() is called twice).
        self._collected = []
        self._collected_keys = set()
        self._collected_metrics = {}

        # Stage 1: island GA with shared collection.
        t0 = time.time()
        ga_config = GAConfig(
            population=self.ga_config.population,
            generations=self.ga_config.generations,
            tournament_k=self.ga_config.tournament_k,
            elitism=self.ga_config.elitism,
            mutation_rate=self.ga_config.mutation_rate,
            early_stop_generations=self.ga_config.early_stop_generations,
            max_evaluations=min(
                self.max_collect_evaluations,
                self.ga_config.max_evaluations or self.max_collect_evaluations,
            ),
            initial_population=self.ga_config.initial_population,
            seed=self.ga_config.seed,
            workers=self.ga_config.workers,
            islands=self.ga_config.islands,
            migration_interval=self.ga_config.migration_interval,
            migration_count=self.ga_config.migration_count,
            restart_stagnation=self.ga_config.restart_stagnation,
        )
        ga = GeneticOptimizer(
            param_space=ParamSpace(self.param_space),
            fitness_fn=self._fitness,
            config=ga_config,
            constraints=self.constraints,
            batch_fitness_fn=(
                self._batch_fitness if ga_config.workers > 1 else None
            ),
            collect_fn=self._collect_stage1,
            stop_fn=self._collection_done,
        )
        try:
            report = ga.run()
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
        t1 = time.time()

        print(f"=== Phase A: GA collection (H1 train) ===")
        print(f"  [TIMING] {t1-t0:.1f}s")
        print(f"  [EVALS] {len(report.history)} evaluations")
        print(
            f"  [COLLECTED] {len(self._collected)} / {self.collect_target} "
            f"diverse H1-train survivors"
        )

        # Build PassingStrategy list from the collected set.
        passing: list[PassingStrategy] = []
        for params in self._collected:
            key = json.dumps(params, sort_keys=True, default=str)
            passing.append(
                PassingStrategy(
                    params=params,
                    fitness=0.0,
                    htf_train_metrics=self._collected_metrics.get(key),
                )
            )

        summary_path = self._save_collection_summary(report, passing)

        print(f"  [SUMMARY] {summary_path}")

        return OptimizationResult(
            report=report,
            passing=passing,
            summary_path=summary_path,
        )

    def run_m1_funnel(
        self, passing: list[PassingStrategy]
    ) -> list[PassingStrategy]:
        """Phase B (M1 part): M1-confirm + M1-OOS on Stage-1 survivors.

        Returns the survivors that pass both M1 stages, with their stage
        metrics attached. Real-time pass/fail printed per strategy.
        """
        if self._engine is None:
            raise RuntimeError("call optimize() before run_m1_funnel()")

        survivors: list[PassingStrategy] = []
        for i, p in enumerate(passing):
            # Stage 2: M1 confirmation (same train window).
            m1_result = self._evaluate_m1_train(p.params)
            m1_train = self._window_metrics(
                m1_result, self._train_start, self._train_end, self._m1_train_index
            )
            p.m1_train_metrics = m1_train
            m1_pf = m1_train["profit_factor"]
            if m1_pf < self.m1_confirm_pf:
                print(
                    f"  [M1-CONFIRM] #{i} FAIL: m1_train pf={m1_pf:.4f} < "
                    f"{self.m1_confirm_pf}"
                )
                continue
            print(
                f"  [M1-CONFIRM] #{i} PASS (pf={m1_pf:.4f}, n={m1_train['n_trades']})"
            )

            # Stage 3: M1 OOS gates (separate OOS1/OOS2 slices).
            m1_oos1_result = self._evaluate_m1_oos1(p.params)
            m1_oos1 = self._window_metrics(
                m1_oos1_result, self._oos1_start, self._oos1_end, self._m1_oos1_index
            )
            m1_oos2_result = self._evaluate_m1_oos2(p.params)
            m1_oos2 = self._window_metrics(
                m1_oos2_result, self._oos2_start, self._oos2_end, self._m1_oos2_index
            )
            p.m1_oos1_metrics = m1_oos1
            p.m1_oos2_metrics = m1_oos2
            passed, reason = self._m1_oos_check(m1_oos1, m1_oos2)
            if not passed:
                print(f"  [M1-OOS] #{i} FAIL: {reason}")
                continue
            print(
                f"  [M1-OOS] #{i} PASS "
                f"(oos1 pf={m1_oos1['profit_factor']:.4f}, "
                f"oos2 pf={m1_oos2['profit_factor']:.4f})"
            )
            survivors.append(p)

        return survivors

    # ------------------------------------------------------------------ #
    # Phase A collection callbacks
    # ------------------------------------------------------------------ #
    def _collect_stage1(self, params: dict[str, Any], fitness: float) -> None:
        """Called by the GA for each evaluated individual.

        Adds `params` to the shared collected set if it passes the H1 train
        gate AND is behaviorally diverse vs the already-collected set.
        """
        if len(self._collected) >= self.collect_target:
            return
        key = json.dumps(params, sort_keys=True, default=str)
        if key in self._collected_keys:
            return

        # H1 train gate check (cached backtest).
        wm = self._window_metrics(
            self._evaluate_htf(params),
            self._train_start,
            self._train_end,
            self._htf_train.index,
        )
        passed, reason = self._htf_train_check(wm)
        if not passed:
            return

        # Behavioral diversity: must differ from EVERY collected strategy
        # in at least `diversity_threshold` condition-slot genes.
        for existing in self._collected:
            if _condition_hamming(params, existing) < self.diversity_threshold:
                return

        self._collected.append(dict(params))
        self._collected_keys.add(key)
        self._collected_metrics[key] = wm
        print(
            f"  [COLLECT] {len(self._collected)}/{self.collect_target} "
            f"diverse H1-train survivors"
        )

    def _collection_done(self) -> bool:
        """Called by the GA to decide whether to stop collection."""
        return len(self._collected) >= self.collect_target

    # ------------------------------------------------------------------ #
    # Data loading / timestamp split / slicing
    # ------------------------------------------------------------------ #
    def _load_data(self) -> None:
        """Load HTF + M1; split by timestamp; build warmup slices."""
        dm = DataManager()
        htf = dm.load(
            self.symbol, self.timeframe, start=self.start, end=self.end
        )
        if htf.empty:
            raise ValueError(
                f"No {self.timeframe} data for {self.symbol} in "
                f"[{self.start}, {self.end}]"
            )
        m1 = dm.load(
            self.symbol,
            "M1",
            start=htf.index[0],
            end=htf.index[-1] + pd.Timedelta(days=1),
        )
        if m1.empty:
            raise ValueError(f"No M1 data for {self.symbol} in the range")

        self._htf = htf
        self._m1 = m1

        # Timestamp-based split (same dates for HTF and M1).
        span = htf.index[-1] - htf.index[0]
        f1, f2, _f3 = self.split
        self._oos1_start = htf.index[0]
        self._oos1_end = htf.index[0] + span * f1
        self._train_start = self._oos1_end
        self._train_end = htf.index[0] + span * (f1 + f2)
        self._oos2_start = self._train_end
        self._oos2_end = htf.index[-1]

        # Warmup: max lookback among lookback/period params (in days).
        warmup_days = 0
        for name, s in self.param_space.items():
            if (
                isinstance(s, dict)
                and "max" in s
                and ("lookback" in name or "period" in name)
            ):
                warmup_days = max(warmup_days, int(s["max"]))
        warmup_days = max(warmup_days, 365) + 5
        warmup_td = pd.Timedelta(days=warmup_days)
        # Tail buffer past each window end so trades entered inside the
        # window can close.
        tail_td = pd.Timedelta(days=2)

        # Stage 1 + Stage 2: train window + warmup + tail.
        self._htf_train = htf.loc[
            self._train_start - warmup_td : self._train_end + tail_td
        ]
        self._m1_train_index = dm.load(
            self.symbol,
            "M1",
            start=self._htf_train.index[0],
            end=self._htf_train.index[-1] + pd.Timedelta(days=1),
        ).index

        # Stage 3 OOS1: data start -> oos1_end + tail (no warmup needed).
        self._htf_m1_oos1 = htf.loc[
            self._oos1_start : self._oos1_end + tail_td
        ]
        self._m1_oos1_index = dm.load(
            self.symbol,
            "M1",
            start=self._htf_m1_oos1.index[0],
            end=self._htf_m1_oos1.index[-1] + pd.Timedelta(days=1),
        ).index

        # Stage 3 OOS2: oos2_start - warmup -> oos2_end + tail.
        self._htf_m1_oos2 = htf.loc[
            self._oos2_start - warmup_td : self._oos2_end + tail_td
        ]
        self._m1_oos2_index = dm.load(
            self.symbol,
            "M1",
            start=self._htf_m1_oos2.index[0],
            end=self._htf_m1_oos2.index[-1] + pd.Timedelta(days=1),
        ).index

    # ------------------------------------------------------------------ #
    # Correctness guard: sliced vs full H1 train metrics
    # ------------------------------------------------------------------ #
    def _verify_slicing(self) -> None:
        """Verify sliced-H1 train metrics match full-H1 train metrics."""
        rng = random.Random(123)
        space = ParamSpace(self.param_space)
        samples = [space.random_individual(rng) for _ in range(2)]

        for params in samples:
            strategy = self.strategy_class(**params)

            # Full H1.
            signals_full = strategy.generate(self._htf)
            result_full = self._engine.run_htf(signals_full, self._htf)
            wm_full = self._window_metrics(
                result_full, self._train_start, self._train_end, self._htf.index
            )

            # Sliced H1 (train + warmup).
            signals_sliced = strategy.generate(self._htf_train)
            result_sliced = self._engine.run_htf(signals_sliced, self._htf_train)
            wm_sliced = self._window_metrics(
                result_sliced,
                self._train_start,
                self._train_end,
                self._htf_train.index,
            )

            for k in ("profit_factor", "n_trades", "win_rate", "trades_per_month"):
                vf, vs = wm_full[k], wm_sliced[k]
                if isinstance(vf, float) and isinstance(vs, float):
                    if math.isinf(vf) and math.isinf(vs):
                        continue
                    tol = 1e-2 * max(1.0, abs(vf))
                    if abs(vf - vs) > tol:
                        raise RuntimeError(
                            f"Slicing mismatch for '{k}': full={vf}, "
                            f"sliced={vs}, params={params}"
                        )
        print(
            "Slicing verification passed: sliced-H1 train metrics "
            "match full-H1"
        )

    # ------------------------------------------------------------------ #
    # GA fitness (Phase A: HTF train)
    # ------------------------------------------------------------------ #
    def _fitness(self, params: dict[str, Any]) -> float:
        """Fitness: criterion score on HTF-train-window closed trades."""
        result = self._evaluate_htf(params)
        wm = self._window_metrics(
            result, self._train_start, self._train_end, self._htf_train.index
        )
        return self._criterion_score(wm)

    def _criterion_score(self, wm: dict[str, float]) -> float:
        return _compute_criterion_score(wm, self.fitness_criterion)

    def _batch_fitness(self, params_list: list[dict[str, Any]]) -> list[float]:
        """Evaluate a batch of params in parallel across worker processes."""
        if self._pool is None:
            payload = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "risk_money": self.risk_money,
                "initial_capital": self.initial_capital,
                "strategy_name": self.strategy_name,
                "strategy_class": self.strategy_class,
                "htf_train": self._htf_train,
                "train_start": self._train_start,
                "train_end": self._train_end,
                "fitness_criterion": self.fitness_criterion,
            }
            self._pool = ProcessPoolExecutor(
                max_workers=self.ga_config.workers,
                initializer=_worker_init,
                initargs=(payload,),
            )
        futures = [
            self._pool.submit(_worker_fitness, p) for p in params_list
        ]
        return [f.result() for f in futures]

    def _evaluate_htf(self, params: dict[str, Any]) -> BacktestResult:
        """HTF backtest on the train slice (fast screen), cached."""
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in self._htf_cache:
            strategy = self.strategy_class(**params)
            signals = strategy.generate(self._htf_train)
            self._htf_cache[key] = self._engine.run_htf(
                signals, self._htf_train
            )
        return self._htf_cache[key]

    # ------------------------------------------------------------------ #
    # M1 evaluations (sliced windows)
    # ------------------------------------------------------------------ #
    def _evaluate_m1_train(self, params: dict[str, Any]) -> BacktestResult:
        """M1 backtest on the train slice (Stage 2), cached."""
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in self._m1_train_cache:
            strategy = self.strategy_class(**params)
            signals = strategy.generate(self._htf_train)
            self._m1_train_cache[key] = self._engine.run(
                signals, self._htf_train
            )
        return self._m1_train_cache[key]

    def _evaluate_m1_oos1(self, params: dict[str, Any]) -> BacktestResult:
        """M1 backtest on the OOS1 slice (Stage 3), cached."""
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in self._m1_oos1_cache:
            strategy = self.strategy_class(**params)
            signals = strategy.generate(self._htf_m1_oos1)
            self._m1_oos1_cache[key] = self._engine.run(
                signals, self._htf_m1_oos1
            )
        return self._m1_oos1_cache[key]

    def _evaluate_m1_oos2(self, params: dict[str, Any]) -> BacktestResult:
        """M1 backtest on the OOS2 slice (Stage 3), cached."""
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in self._m1_oos2_cache:
            strategy = self.strategy_class(**params)
            signals = strategy.generate(self._htf_m1_oos2)
            self._m1_oos2_cache[key] = self._engine.run(
                signals, self._htf_m1_oos2
            )
        return self._m1_oos2_cache[key]

    # ------------------------------------------------------------------ #
    # Window metrics (trades split by entry timestamp)
    # ------------------------------------------------------------------ #
    def _window_metrics(
        self,
        result: BacktestResult,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        index: pd.DatetimeIndex,
    ) -> dict[str, float]:
        """Gate metrics from closed trades whose ENTRY falls in the window."""
        return _compute_window_metrics(
            result, start_ts, end_ts, index, self.initial_capital
        )

    # ------------------------------------------------------------------ #
    # Pass gates
    # ------------------------------------------------------------------ #
    def _htf_train_check(
        self, wm: dict[str, float]
    ) -> tuple[bool, str | None]:
        """Stage 1 check: (passed, fail_reason)."""
        for metric, min_value in self.htf_train_gates.items():
            val = wm.get(metric, 0.0)
            if val <= min_value:
                return False, (
                    f"htf_train {metric}={val:.4f} <= {min_value}"
                )
        return True, None

    def _m1_oos_check(
        self, oos1: dict[str, float], oos2: dict[str, float]
    ) -> tuple[bool, str | None]:
        """Stage 3 check: (passed, fail_reason)."""
        for window, metrics in (("oos1", oos1), ("oos2", oos2)):
            for metric, min_value in self.m1_oos_gates[window].items():
                val = metrics.get(metric, 0.0)
                if val <= min_value:
                    return False, (
                        f"m1_{window} {metric}={val:.4f} <= {min_value}"
                    )
        return True, None

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def _save_collection_summary(
        self, report: GAReport, passing: list[PassingStrategy]
    ) -> Path:
        """Write a CSV of the collected Stage-1 survivors."""
        import csv

        self.results_dir.mkdir(parents=True, exist_ok=True)
        base = f"{self.strategy_name}_{self.symbol}_{self.timeframe}"
        path = self.results_dir / f"{base}_stage1_collected.csv"

        rows = []
        for p in passing:
            row = {}
            for k, v in p.params.items():
                row[f"param_{k}"] = v
            for k, v in (p.htf_train_metrics or {}).items():
                row[f"htf_train_{k}"] = v
            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("no_collected\n")
        return path

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Convert inf/nan to None for JSON serialization."""
        if isinstance(value, dict):
            return {k: TestEngine._jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [TestEngine._jsonable(v) for v in value]
        if isinstance(value, float) and (
            math.isinf(value) or math.isnan(value)
        ):
            return None
        return value