"""TestEngine: genetic parameter optimization + 3-window validation.

Split data by index position: OOS1 (30%) / TRAIN (50%) / OOS2 (20%).
GA evolves params on TRAIN only. One full-range backtest per individual;
trades split by entry window serve fitness, OOS gates, and the chart.
Passing strategies: equity PNG (train shaded) + summary JSON.
"""

from __future__ import annotations

import json
import math
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
# Default pass gates (all strictly-greater-than). Constructor-configurable.
DEFAULT_GATES: dict[str, dict[str, float]] = {
    "train": {"profit_factor": 1.2, "n_trades": 100.0, "win_rate": 35.0},
    "oos1": {"profit_factor": 1.0, "trades_per_month": 2.0},
    "oos2": {"profit_factor": 1.1, "trades_per_month": 2.0},
}
DAYS_PER_MONTH = 30.44


@dataclass
class PassingStrategy:
    """A strategy whose params passed all three gates."""

    params: dict[str, Any]
    train_metrics: dict[str, float]
    oos1_metrics: dict[str, float]
    oos2_metrics: dict[str, float]
    equity_curve_png: Path | None = None


@dataclass
class OptimizationResult:
    """Result of a TestEngine optimization run."""

    report: GAReport
    passing: list[PassingStrategy]
    summary_path: Path | None = None


class TestEngine:
    """Runs a GA over strategy params and validates across three windows."""

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
        fitness_criterion: str = "pf_n_trades",
        pass_thresholds: dict[str, dict[str, float]] | None = None,
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

        self.pass_thresholds = pass_thresholds or DEFAULT_GATES
        bad = set(self.pass_thresholds) - {"train", "oos1", "oos2"}
        if bad:
            raise ValueError(
                f"pass_thresholds keys must be train/oos1/oos2, "
                f"got {sorted(bad)}"
            )

        # Filled by _load_data() during optimize().
        self._htf: pd.DataFrame | None = None
        self._m1_index: pd.DatetimeIndex | None = None
        self._oos1_start = self._oos1_end = pd.Timestamp(0)
        self._train_start = self._train_end = pd.Timestamp(0)
        self._oos2_start = self._oos2_end = pd.Timestamp(0)
        self._cache: dict[str, BacktestResult] = {}
        self._engine: BacktestEngine | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimize(self) -> OptimizationResult:
        """Run the genetic search, gate survivors, save outputs."""
        self._load_data()
        self._engine = BacktestEngine(
            symbol=self.symbol,
            timeframe=self.timeframe,
            risk_pct=self.risk_pct,
            initial_capital=self.initial_capital,
            strategy_name=self.strategy_name,
        )
        ga = GeneticOptimizer(
            param_space=ParamSpace(self.param_space),
            fitness_fn=self._fitness,
            config=self.ga_config,
            constraints=self.constraints,
        )
        report = ga.run()

        # Rank unique evaluated individuals by fitness; gate each.
        seen: dict[str, tuple[dict[str, Any], float, BacktestResult]] = {}
        for rec in report.history:
            key = json.dumps(rec.params, sort_keys=True, default=str)
            if key not in seen:
                seen[key] = (rec.params, rec.fitness, self._cache[key])

        passing: list[PassingStrategy] = []
        for params, _fit, result in sorted(
            seen.values(), key=lambda t: t[1], reverse=True
        ):
            windows = self._window_metrics_map(result)
            if self._all_pass(windows):
                png = self._save_png(
                    result.equity_curve, params, len(passing)
                )
                passing.append(
                    PassingStrategy(
                        params=params,
                        train_metrics=windows["train"],
                        oos1_metrics=windows["oos1"],
                        oos2_metrics=windows["oos2"],
                        equity_curve_png=png,
                    )
                )

        summary_path = self._save_summary(report, passing)
        print(
            f"Optimization complete: {len(report.history)} evals, "
            f"{len(passing)} passing strategies (summary: {summary_path})"
        )
        return OptimizationResult(
            report=report, passing=passing, summary_path=summary_path
        )

    # ------------------------------------------------------------------ #
    # Data loading / splitting
    # ------------------------------------------------------------------ #
    def _load_data(self) -> None:
        """Load M1 + HTF; split M1 by index position into the three windows."""
        dm = DataManager()
        htf = dm.load(
            self.symbol, self.timeframe, start=self.start, end=self.end
        )
        if htf.empty:
            raise ValueError(
                f"No {self.timeframe} data for {self.symbol} in "
                f"[{self.start}, {self.end}]"
            )
        # M1 range must match exactly what BacktestEngine.run() loads.
        m1 = dm.load(
            self.symbol,
            "M1",
            start=htf.index[0],
            end=htf.index[-1] + pd.Timedelta(days=1),
        )
        if m1.empty:
            raise ValueError(f"No M1 data for {self.symbol} in the range")

        self._htf = htf
        self._m1_index = m1.index
        n = len(m1)
        f1, f2, _f3 = self.split
        idx1 = int(n * f1)
        idx2 = int(n * (f1 + f2))
        if idx1 < 1 or idx2 <= idx1 or idx2 >= n:
            raise ValueError(
                f"Split {self.split} produces an empty window on "
                f"{n} M1 bars; enlarge the data range"
            )
        self._oos1_start, self._oos1_end = m1.index[0], m1.index[idx1 - 1]
        self._train_start, self._train_end = m1.index[idx1], m1.index[idx2 - 1]
        self._oos2_start, self._oos2_end = m1.index[idx2], m1.index[-1]

    # ------------------------------------------------------------------ #
    # GA fitness
    # ------------------------------------------------------------------ #
    def _fitness(self, params: dict[str, Any]) -> float:
        """Fitness: criterion score on TRAIN-window closed trades only."""
        result = self._evaluate(params)
        wm = self._window_metrics(result, self._train_start, self._train_end)
        return self._criterion_score(wm)

    def _criterion_score(self, wm: dict[str, float]) -> float:
        pf = wm["profit_factor"]
        if pf == float("inf"):
            pf = PF_CAP
        if self.fitness_criterion == "pf":
            return float(pf)
        if self.fitness_criterion == "return":
            return float(wm["total_return_pct"])
        # Smooth penalty for trading fewer than 100 times.
        return float(pf * min(1.0, wm["n_trades"] / 100.0))

    def _evaluate(self, params: dict[str, Any]) -> BacktestResult:
        """Full-range backtest for params, cached by canonical JSON key."""
        key = json.dumps(params, sort_keys=True, default=str)
        if key not in self._cache:
            strategy = self.strategy_class(**params)
            signals = strategy.generate(self._htf)
            self._cache[key] = self._engine.run(signals, self._htf)
        return self._cache[key]

    # ------------------------------------------------------------------ #
    # Window metrics (trades split by entry timestamp)
    # ------------------------------------------------------------------ #
    def _window_metrics_map(
        self, result: BacktestResult
    ) -> dict[str, dict[str, float]]:
        return {
            "train": self._window_metrics(
                result, self._train_start, self._train_end
            ),
            "oos1": self._window_metrics(
                result, self._oos1_start, self._oos1_end
            ),
            "oos2": self._window_metrics(
                result, self._oos2_start, self._oos2_end
            ),
        }

    def _window_metrics(
        self, result: BacktestResult, start_ts: pd.Timestamp, end_ts: pd.Timestamp
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

        entry_times = self._m1_index[
            closed["entry_idx"].astype(int).values
        ]
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
            "total_return_pct": float(
                pnls.sum() / self.initial_capital * 100
            ),
        }

    # ------------------------------------------------------------------ #
    # Pass gate
    # ------------------------------------------------------------------ #
    def _all_pass(self, windows: dict[str, dict[str, float]]) -> bool:
        """True if every window meets its thresholds (strictly greater)."""
        for window, thresholds in self.pass_thresholds.items():
            metrics = windows[window]
            for metric, min_value in thresholds.items():
                if metrics.get(metric, 0.0) <= min_value:
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def _save_png(
        self, equity_curve: pd.Series, params: dict[str, Any], index: int
    ) -> Path:
        """Full-data equity PNG with the train region shaded."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / (
            f"{self.strategy_name}_{self.symbol}_{self.timeframe}"
            f"_pass{index}.png"
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        if len(equity_curve):
            ax.plot(equity_curve.index, equity_curve.values, lw=1.2)
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
        fig.savefig(path, dpi=150)
        plt.close(fig)
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
                    "train": self._jsonable(p.train_metrics),
                    "oos1": self._jsonable(p.oos1_metrics),
                    "oos2": self._jsonable(p.oos2_metrics),
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