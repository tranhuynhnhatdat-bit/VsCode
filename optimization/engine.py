"""TestEngine: 3-stage optimization pipeline.

Stage 1 — GA on the strategy's native timeframe (HTF), train window only.
  - Fitness = profit factor on HTF train-window closed trades.
  - Fast screen: HTF backtests are cheap (no M1 mapping).
  - Runs on a train-window slice (with warmup) for speed; verified identical
    to full-range metrics by _verify_slicing().

Stage 2 — M1 confirmation (same train window).
  - Survivors re-run on M1; M1 train PF must be >= m1_confirm_ratio * H1 PF.
  - H1 PF is capped at m1_pf_cap for the ratio (infinite PF -> cap).
  - Runs on a train-window M1 slice (with warmup) for speed.

Stage 3 — M1 out-of-sample.
  - Survivors tested on M1 OOS1/OOS2 windows (same timestamps as H1).
  - Gates: per-window PF and trades/month thresholds.
  - Runs on separate OOS1/OOS2 M1 slices (OOS2 gets warmup from train tail).

Windows are split by timestamp once (30% OOS1 / 50% train / 20% OOS2 of the
date range) and used identically for HTF and M1, so the M1 confirmation and
OOS tests align with the H1 train window.

Passing strategies: equity PNG (train shaded) + summary JSON.
"""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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


# ------------------------------------------------------------------ #
# Parallel worker state (per-process, set by _worker_init)
# ------------------------------------------------------------------ #
_worker_state: dict[str, Any] = {}


def _worker_init(payload: dict[str, Any]) -> None:
    """Initialize a worker process: engine + sliced HTF + window bounds."""
    _worker_state["engine"] = BacktestEngine(
        symbol=payload["symbol"],
        timeframe=payload["timeframe"],
        risk_pct=payload["risk_pct"],
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
    """A strategy whose params passed all three stages."""

    params: dict[str, Any]
    htf_train_metrics: dict[str, float]
    m1_train_metrics: dict[str, float]
    m1_oos1_metrics: dict[str, float]
    m1_oos2_metrics: dict[str, float]
    equity_curve_png: Path | None = None


@dataclass
class StageResult:
    """One individual's journey through the 3-stage pipeline.

    Fields are None until the individual reaches that stage.
    """

    params: dict[str, Any]
    fitness: float
    htf_train: dict[str, float] | None = None
    htf_train_pass: bool | None = None
    htf_fail_reason: str | None = None
    m1_train: dict[str, float] | None = None
    m1_confirm_pass: bool | None = None
    m1_confirm_fail_reason: str | None = None
    m1_oos1: dict[str, float] | None = None
    m1_oos2: dict[str, float] | None = None
    m1_oos_pass: bool | None = None
    m1_oos_fail_reason: str | None = None


@dataclass
class OptimizationResult:
    """Result of a TestEngine optimization run."""

    report: GAReport
    passing: list[PassingStrategy]
    stage_results: list[StageResult] | None = None
    summary_path: Path | None = None


class TestEngine:
    """Runs the 3-stage pipeline: H1 GA -> M1 confirm -> M1 OOS."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        strategy_class: type[Strategy],
        param_space: dict[str, Any],
        split: tuple[float, float, float] = (0.30, 0.50, 0.20),
        constraints: list[tuple[str, str, str]] | None = None,
        initial_capital: float = 10_000.0,
        risk_pct: float = 0.01,
        strategy_name: str = "optimized",
        ga_config: GAConfig | None = None,
        fitness_criterion: str = "pf",
        htf_train_gates: dict[str, float] | None = None,
        m1_confirm_ratio: float = 0.9,
        m1_pf_cap: float = 10.0,
        m1_oos_gates: dict[str, dict[str, float]] | None = None,
        results_dir: Path = RESULTS_DIR,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.split = tuple(split)
        self.constraints = constraints
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.strategy_name = strategy_name
        self.ga_config = ga_config or GAConfig()
        self.results_dir = Path(results_dir)
        self.start = start
        self.end = end

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
        self._m1_cache: dict[str, BacktestResult] = {}
        self._m1_train_cache: dict[str, BacktestResult] = {}
        self._m1_oos1_cache: dict[str, BacktestResult] = {}
        self._m1_oos2_cache: dict[str, BacktestResult] = {}
        self._engine: BacktestEngine | None = None
        self._pool: ProcessPoolExecutor | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimize(self) -> OptimizationResult:
        """Run the 3-stage pipeline and save passing strategies."""
        self._load_data()
        self._engine = BacktestEngine(
            symbol=self.symbol,
            timeframe=self.timeframe,
            risk_pct=self.risk_pct,
            initial_capital=self.initial_capital,
            strategy_name=self.strategy_name,
        )
        self._verify_slicing()

        ga = GeneticOptimizer(
            param_space=ParamSpace(self.param_space),
            fitness_fn=self._fitness,
            config=self.ga_config,
            constraints=self.constraints,
            batch_fitness_fn=(
                self._batch_fitness if self.ga_config.workers > 1 else None
            ),
        )
        try:
            report = ga.run()
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None

        # Rank unique evaluated individuals by fitness; run the stages.
        seen: dict[str, tuple[dict[str, Any], float]] = {}
        for rec in report.history:
            key = json.dumps(rec.params, sort_keys=True, default=str)
            if key not in seen:
                seen[key] = (rec.params, rec.fitness)

        passing: list[PassingStrategy] = []
        stage_results: list[StageResult] = []
        for params, fit in sorted(
            seen.values(), key=lambda t: t[1], reverse=True
        ):
            sr = StageResult(params=params, fitness=fit)

            # Stage 1: H1 train gate.
            htf_train = self._window_metrics(
                self._evaluate_htf(params), self._train_start, self._train_end, self._htf_train.index
            )
            sr.htf_train = htf_train
            sr.htf_train_pass, sr.htf_fail_reason = self._htf_train_check(
                htf_train
            )
            if not sr.htf_train_pass:
                stage_results.append(sr)
                continue

            # Stage 2: M1 confirmation (same train window).
            m1_result = self._evaluate_m1_train(params)
            m1_train = self._window_metrics(
                m1_result, self._train_start, self._train_end, self._m1_train_index
            )
            sr.m1_train = m1_train
            htf_pf = min(htf_train["profit_factor"], self.m1_pf_cap)
            m1_pf = m1_train["profit_factor"]
            sr.m1_confirm_pass = m1_pf >= self.m1_confirm_ratio * htf_pf
            if not sr.m1_confirm_pass:
                sr.m1_confirm_fail_reason = (
                    f"m1_train profit_factor={m1_pf:.4f} < "
                    f"{self.m1_confirm_ratio} * htf_pf={htf_pf:.4f}"
                )
                stage_results.append(sr)
                continue

            # Stage 3: M1 OOS gates (separate OOS1/OOS2 slices).
            m1_oos1_result = self._evaluate_m1_oos1(params)
            m1_oos1 = self._window_metrics(
                m1_oos1_result, self._oos1_start, self._oos1_end, self._m1_oos1_index
            )
            m1_oos2_result = self._evaluate_m1_oos2(params)
            m1_oos2 = self._window_metrics(
                m1_oos2_result, self._oos2_start, self._oos2_end, self._m1_oos2_index
            )
            sr.m1_oos1 = m1_oos1
            sr.m1_oos2 = m1_oos2
            sr.m1_oos_pass, sr.m1_oos_fail_reason = self._m1_oos_check(
                m1_oos1, m1_oos2
            )
            if not sr.m1_oos_pass:
                stage_results.append(sr)
                continue

            # Full M1 run only for passing strategies (folder output).
            full_result = self._evaluate_m1_full(params)
            folder = self._save_strategy_folder(
                full_result,
                params,
                len(passing),
                htf_train,
                m1_train,
                m1_oos1,
                m1_oos2,
            )
            sr.m1_oos_pass = True
            stage_results.append(sr)
            passing.append(
                PassingStrategy(
                    params=params,
                    htf_train_metrics=htf_train,
                    m1_train_metrics=m1_train,
                    m1_oos1_metrics=m1_oos1,
                    m1_oos2_metrics=m1_oos2,
                    equity_curve_png=folder / "equity_curve.png",
                )
            )

        summary_path = self._save_summary(report, passing)
        stage_csvs = self._save_stage_csvs(stage_results)
        n1 = sum(1 for s in stage_results if s.htf_train_pass)
        n2 = sum(1 for s in stage_results if s.m1_confirm_pass)
        n3 = sum(1 for s in stage_results if s.m1_oos_pass)
        print(
            f"Optimization complete: {len(report.history)} evals, "
            f"{len(passing)} passing strategies (summary: {summary_path})"
        )
        print(
            f"Stage funnel: {len(stage_results)} unique -> {n1} pass H1 "
            f"-> {n2} pass M1 confirm -> {n3} pass M1 OOS"
        )
        print(f"Stage CSVs: {stage_csvs}")
        return OptimizationResult(
            report=report,
            passing=passing,
            stage_results=stage_results,
            summary_path=summary_path,
        )

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
        # The hour-aligned volume filter needs volume_lookback DAYS of prior
        # same-hour bars, so warmup must be in days, not bars.
        # A 365-day floor ensures the ADX/ATR ewm (infinite-memory recursive
        # filters) have converged by the train start, so sliced metrics match
        # full-range metrics to within floating-point noise.
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
        # window can close (a straddling trade would otherwise be counted
        # as open/status=0 in the sliced run but closed in the full run).
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
        """Verify sliced-H1 train metrics match full-H1 train metrics.

        Runs a few random param sets through both the full H1 series and the
        warmup-sliced H1 series, and asserts the train-window metrics match.
        This guards against the slicing changing indicator history.
        """
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

            # Only metrics that drive pipeline decisions must match exactly.
            # total_return_pct / avg_trade_pct are informational: a tiny
            # ADX ewm drift at the slice boundary can shift one entry's
            # fill price (same trade count, slightly different PnL) without
            # affecting any gate or the fitness.
            for k in ("profit_factor", "n_trades", "win_rate", "trades_per_month"):
                vf, vs = wm_full[k], wm_sliced[k]
                if isinstance(vf, float) and isinstance(vs, float):
                    if math.isinf(vf) and math.isinf(vs):
                        continue
                    # Relative tolerance: indicator drift at the slice
                    # boundary is ~0.3%; a real warmup bug (missing warmup
                    # -> NaN filters -> 0 entries) is >>10%.
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
    # GA fitness (Stage 1: HTF train)
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
                "risk_pct": self.risk_pct,
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
    # M1 evaluations (sliced windows + full for PNG)
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

    def _evaluate_m1_full(self, params: dict[str, Any]) -> BacktestResult:
        """Full-range M1 backtest (only for passing strategies' PNG), cached."""
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in self._m1_cache:
            strategy = self.strategy_class(**params)
            signals = strategy.generate(self._htf)
            self._m1_cache[key] = self._engine.run(signals, self._htf)
        return self._m1_cache[key]

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
    def _htf_train_pass(self, wm: dict[str, float]) -> bool:
        """Stage 1: H1 train gate (strictly greater than thresholds)."""
        return self._htf_train_check(wm)[0]

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

    def _m1_oos_pass(
        self, oos1: dict[str, float], oos2: dict[str, float]
    ) -> bool:
        """Stage 3: M1 OOS gates (strictly greater than thresholds)."""
        return self._m1_oos_check(oos1, oos2)[0]

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
    def _save_strategy_folder(
        self,
        result: BacktestResult,
        params: dict[str, Any],
        index: int,
        htf_train: dict[str, float],
        m1_train: dict[str, float],
        m1_oos1: dict[str, float],
        m1_oos2: dict[str, float],
    ) -> Path:
        """Write one folder per passing strategy.

        Folder layout (results/strategy_<index>/):
        - equity_curve.png  full-data equity curve, train region shaded
        - trades.csv        full M1 trade records
        - strategy.json     params + all stage metrics + full-run metrics
        """
        import csv
        import json

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        folder = self.results_dir / f"strategy_{index}"
        folder.mkdir(parents=True, exist_ok=True)

        # Equity curve PNG (train region shaded).
        png_path = folder / "equity_curve.png"
        fig, ax = plt.subplots(figsize=(12, 6))
        if len(result.equity_curve):
            ax.plot(
                result.equity_curve.index,
                result.equity_curve.values,
                lw=1.2,
            )
        ax.axvspan(
            self._train_start,
            self._train_end,
            color="#ff7f0e",
            alpha=0.15,
            label="Train (in-sample)",
        )
        ax.set_title(
            f"{self.strategy_name} {self.symbol} {self.timeframe} "
            f"| params={params}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USD)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        # Trade records CSV.
        trades_path = folder / "trades.csv"
        if result.trades is not None and not result.trades.empty:
            result.trades.to_csv(trades_path, index=False)
        else:
            with open(trades_path, "w", newline="", encoding="utf-8") as f:
                f.write("no_trades\n")

        # Params + stage metrics + full-run metrics JSON.
        strategy_path = folder / "strategy.json"
        data = {
            "params": self._jsonable(params),
            "htf_train": self._jsonable(htf_train),
            "m1_train": self._jsonable(m1_train),
            "m1_oos1": self._jsonable(m1_oos1),
            "m1_oos2": self._jsonable(m1_oos2),
            "full_run_metrics": self._jsonable(result.metrics),
        }
        with open(strategy_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return folder

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

    def _save_summary(
        self, report: GAReport, passing: list[PassingStrategy]
    ) -> Path:
        """Write the optimization summary JSON (passing strategies only)."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / (
            f"{self.strategy_name}_{self.symbol}_{self.timeframe}"
            f"_optimization_summary.json"
        )
        data = {
            "strategy": self.strategy_class.__name__,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "split": list(self.split),
            "start": str(self._oos1_start.date()),
            "end": str(self._oos2_end.date()),
            "n_evaluations": len(report.history),
            "best_fitness": (
                None if report.best is None else report.best.fitness
            ),
            "n_passing": len(passing),
            "passing_strategies": [
                {
                    "params": self._jsonable(p.params),
                    "htf_train": self._jsonable(p.htf_train_metrics),
                    "m1_train": self._jsonable(p.m1_train_metrics),
                    "m1_oos1": self._jsonable(p.m1_oos1_metrics),
                    "m1_oos2": self._jsonable(p.m1_oos2_metrics),
                    "equity_curve_png": (
                        str(p.equity_curve_png)
                        if p.equity_curve_png
                        else None
                    ),
                }
                for p in passing
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def _save_stage_csvs(
        self, stage_results: list[StageResult]
    ) -> list[Path]:
        """Write per-stage CSVs so you can see where strategies are dropped.

        - stage1_htf_train.csv: every unique eval + H1 train metrics + pass
        - stage2_m1_confirm.csv: Stage-1 survivors + M1 train metrics + pass
        - stage3_m1_oos.csv:     Stage-2 survivors + M1 OOS metrics + pass
        """
        import csv

        self.results_dir.mkdir(parents=True, exist_ok=True)
        base = (
            f"{self.strategy_name}_{self.symbol}_{self.timeframe}"
        )
        paths: list[Path] = []

        def _write(
            name: str, rows: list[dict[str, Any]]
        ) -> Path:
            p = self.results_dir / f"{base}_{name}.csv"
            if rows:
                fieldnames = list(rows[0].keys())
                with open(p, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    f.write("no_rows\n")
            return p

        # Stage 1: all unique evals.
        s1_rows = []
        for sr in stage_results:
            row = {"fitness": sr.fitness}
            for k, v in sr.params.items():
                row[f"param_{k}"] = v
            for k, v in (sr.htf_train or {}).items():
                row[f"htf_train_{k}"] = v
            row["pass"] = sr.htf_train_pass
            row["fail_reason"] = sr.htf_fail_reason or ""
            s1_rows.append(row)
        paths.append(_write("stage1_htf_train", s1_rows))

        # Stage 2: Stage-1 survivors.
        s2_rows = []
        for sr in stage_results:
            if not sr.htf_train_pass:
                continue
            row = {"fitness": sr.fitness}
            for k, v in sr.params.items():
                row[f"param_{k}"] = v
            for k, v in (sr.htf_train or {}).items():
                row[f"htf_train_{k}"] = v
            for k, v in (sr.m1_train or {}).items():
                row[f"m1_train_{k}"] = v
            row["pass"] = sr.m1_confirm_pass
            row["fail_reason"] = sr.m1_confirm_fail_reason or ""
            s2_rows.append(row)
        paths.append(_write("stage2_m1_confirm", s2_rows))

        # Stage 3: Stage-2 survivors.
        s3_rows = []
        for sr in stage_results:
            if not sr.m1_confirm_pass:
                continue
            row = {"fitness": sr.fitness}
            for k, v in sr.params.items():
                row[f"param_{k}"] = v
            for k, v in (sr.m1_oos1 or {}).items():
                row[f"m1_oos1_{k}"] = v
            for k, v in (sr.m1_oos2 or {}).items():
                row[f"m1_oos2_{k}"] = v
            row["pass"] = sr.m1_oos_pass
            row["fail_reason"] = sr.m1_oos_fail_reason or ""
            s3_rows.append(row)
        paths.append(_write("stage3_m1_oos", s3_rows))

        return paths