"""Lightweight genetic algorithm for strategy parameter optimization.

Design (from grilling session):
- Mixed param space: numeric ranges {min, max, step} and categorical lists
- An individual is a dict[str, Any] dropped into StrategyClass(**params)
- Tournament selection (k=3), elitism (top N), uniform crossover,
  per-gene mutation with bounded re-draw on invalid individuals
- Structural constraints e.g. ("fast", "<", "slow") reject invalid births
- Early stop after N stagnant generations; optional evaluation budget
- Pure numeric engine: fitness is a caller-supplied float
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

# Fitness function maps a params dict to a score (float).
FitnessFn = Callable[[dict[str, Any]], float]

# Structural constraint: (param_name, operator, other_param_name).
# Example: ("fast", "<", "slow") means fast must be < slow.
CONSTRAINT_OPS = ("<", "<=", ">", ">=")

# Max mutation re-draw attempts before falling back to a fresh random draw.
_MAX_REDRAW_TRIES = 20


@dataclass
class ParamSpace:
    """Validated strategy parameter search space.

    spec format (mixed types allowed):
        "fast":   {"min": 2,   "max": 30,   "step": 1}   # int range
        "sl_atr": {"min": 1.0, "max": 5.0,  "step": 0.5} # float range
        "mode":   ["ema", "sma"]                          # categorical
    """

    spec: dict[str, Any]

    def __post_init__(self) -> None:
        self._names: list[str] = []
        for name, s in self.spec.items():
            if isinstance(s, list):
                if not s:
                    raise ValueError(f"Categorical param '{name}' is empty")
                for v in s:
                    if not isinstance(v, (int, float, str, bool)):
                        raise TypeError(
                            f"Categorical param '{name}' value {v!r} must be "
                            "int, float, str, or bool"
                        )
            elif isinstance(s, dict):
                missing = {"min", "max", "step"} - set(s)
                if missing:
                    raise ValueError(
                        f"Numeric param '{name}' missing keys {sorted(missing)}; "
                        "expected {'min', 'max', 'step'}"
                    )
                for k in ("min", "max", "step"):
                    if not isinstance(s[k], (int, float)):
                        raise TypeError(
                            f"Numeric param '{name}' {k} must be int or float"
                        )
                if s["min"] > s["max"]:
                    raise ValueError(f"Numeric param '{name}' min > max")
                if s["step"] <= 0:
                    raise ValueError(f"Numeric param '{name}' step must be > 0")
            else:
                raise TypeError(
                    f"Param '{name}' must be a list (categorical) or a dict "
                    f"(numeric range), got {type(s).__name__}"
                )
            self._names.append(name)

    @property
    def names(self) -> list[str]:
        """Ordered list of gene names."""
        return self._names

    def random_individual(self, rng: random.Random) -> dict[str, Any]:
        """Draw a fresh random individual (may violate constraints)."""
        return {
            name: self._random_gene(name, self.spec[name], rng)
            for name in self._names
        }

    def mutate(
        self, individual: dict[str, Any], rate: float, rng: random.Random
    ) -> dict[str, Any]:
        """Mutate each gene independently with probability `rate`."""
        out = dict(individual)
        for name in self._names:
            if rng.random() < rate:
                out[name] = self._random_gene(name, self.spec[name], rng)
        return out

    def crossover(
        self, a: dict[str, Any], b: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Uniform crossover: each gene comes from either parent."""
        return {
            name: (a[name] if rng.random() < 0.5 else b[name])
            for name in self._names
        }

    def _random_gene(self, name: str, s: Any, rng: random.Random) -> Any:
        if isinstance(s, list):
            return rng.choice(s)
        lo, hi, step = s["min"], s["max"], s["step"]
        if isinstance(step, int):
            n_steps = (hi - lo) // step
            return lo + rng.randint(0, n_steps) * step
        # Float range: snap to step grid, round, clamp.
        n_steps = int(round((hi - lo) / step))
        val = lo + rng.randint(0, n_steps) * step
        val = min(float(hi), max(float(lo), val))
        return round(val, 10)


def make_validator(
    constraints: list[tuple[str, str, str]] | None,
) -> Callable[[dict[str, Any]], bool]:
    """Build a pure function that checks structural constraints.

    constraints: list of (param_a, op, param_b) tuples, e.g.
        [("fast", "<", "slow")]

    Raises ValueError on malformed constraint definitions.
    """
    constraints = constraints or []
    for c in constraints:
        if len(c) != 3:
            raise ValueError(f"Constraint must be (a, op, b), got {c!r}")
        a, op, b = c
        if op not in CONSTRAINT_OPS:
            raise ValueError(
                f"Constraint op must be one of {CONSTRAINT_OPS}, got {op!r}"
            )

    def is_valid(ind: dict[str, Any]) -> bool:
        for a, op, b in constraints:
            va, vb = ind.get(a), ind.get(b)
            if va is None or vb is None:
                return False
            try:
                if op == "<" and not (va < vb):
                    return False
                if op == "<=" and not (va <= vb):
                    return False
                if op == ">" and not (va > vb):
                    return False
                if op == ">=" and not (va >= vb):
                    return False
            except TypeError:
                return False
        return True

    return is_valid


@dataclass
class GAConfig:
    """Genetic algorithm hyper-parameters."""

    population: int = 50
    generations: int = 15
    tournament_k: int = 3
    elitism: int = 2
    mutation_rate: float = 0.10
    early_stop_generations: int = 3
    max_evaluations: int | None = None
    initial_population: list[dict[str, Any]] | None = None
    seed: int | None = None
    workers: int = 1


@dataclass
class IndividualRecord:
    """One evaluated individual."""

    params: dict[str, Any]
    fitness: float
    evaluated_at: int  # 0-based evaluation order


@dataclass
class GAReport:
    """Result of a genetic search."""

    best: IndividualRecord | None
    history: list[IndividualRecord]  # every evaluated individual, in order


class GeneticOptimizer:
    """Runs the genetic search over a ParamSpace.

    Fitness is computed by a caller-supplied callable; the optimizer only
    orchestrates selection / crossover / mutation / survival.
    """

    def __init__(
        self,
        param_space: ParamSpace,
        fitness_fn: FitnessFn,
        config: GAConfig,
        constraints: list[tuple[str, str, str]] | None = None,
        batch_fitness_fn: Callable[[list[dict[str, Any]]], list[float]]
        | None = None,
    ) -> None:
        if config.population < 1:
            raise ValueError("population must be >= 1")
        if config.generations < 1:
            raise ValueError("generations must be >= 1")
        if config.elitism > config.population:
            raise ValueError("elitism cannot exceed population")
        if not (0.0 < config.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in (0, 1]")
        self.space = param_space
        self.fitness_fn = fitness_fn
        self.batch_fitness_fn = batch_fitness_fn
        self.config = config
        self.is_valid = make_validator(constraints)
        self.rng = random.Random(config.seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> GAReport:
        """Run the search and return the best individual + full history."""
        cfg = self.config
        rng = self.rng
        history: list[IndividualRecord] = []
        order = 0

        def _eval(ind: dict[str, Any], idx: int) -> float:
            nonlocal order
            fit = self.fitness_fn(ind)
            history.append(IndividualRecord(dict(ind), fit, order))
            order += 1
            return fit

        # Initial population: seeded individuals first, then random fill.
        pop: list[dict[str, Any]] = []
        for s in cfg.initial_population or []:
            seed = {k: v for k, v in s.items() if k in self.space.names}
            if self.is_valid(seed):
                pop.append(seed)
        while len(pop) < cfg.population:
            ind = self.space.random_individual(rng)
            if self.is_valid(ind):
                pop.append(ind)
        pop = pop[: cfg.population]

        best_fitness = float("-inf")
        stagnant = 0
        fitness: list[float] = []

        for gen in range(cfg.generations):
            if self.batch_fitness_fn is not None and cfg.workers > 1:
                fits = self.batch_fitness_fn(pop)
                fitness = []
                for ind, fit in zip(pop, fits):
                    history.append(IndividualRecord(dict(ind), fit, order))
                    order += 1
                    fitness.append(fit)
            else:
                fitness = [_eval(ind, i) for i, ind in enumerate(pop)]
            gen_best = max(fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best
                stagnant = 0
            else:
                stagnant += 1
            best = max(history, key=lambda r: r.fitness)

            print(
                f"  GA gen {gen + 1}/{cfg.generations} | "
                f"best fitness {best_fitness:.4f} | evals {order}"
            )

            if cfg.max_evaluations is not None and order >= cfg.max_evaluations:
                print(
                    f"  GA stopping: evaluation budget "
                    f"{cfg.max_evaluations} reached"
                )
                break
            if gen == cfg.generations - 1:
                break
            if stagnant >= cfg.early_stop_generations:
                print(
                    f"  GA stopping: no improvement for "
                    f"{stagnant} generations"
                )
                break

            # Build next generation: elitism + children.
            ranked = sorted(
                range(len(pop)), key=lambda i: fitness[i], reverse=True
            )
            elite = [dict(pop[i]) for i in ranked[: cfg.elitism]]
            next_pop = list(elite)
            while len(next_pop) < cfg.population:
                parent_a = self._tournament(
                    pop, fitness, rng, cfg.tournament_k
                )
                parent_b = self._tournament(
                    pop, fitness, rng, cfg.tournament_k
                )
                child = self.space.crossover(parent_a, parent_b, rng)
                child = self.space.mutate(child, cfg.mutation_rate, rng)
                if not self.is_valid(child):
                    child = self._valid_clone(child, cfg.mutation_rate, rng)
                next_pop.append(child)
            pop = next_pop

        best = max(history, key=lambda r: r.fitness)
        return GAReport(best=best, history=history)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _valid_clone(
        self, ind: dict[str, Any], rate: float, rng: random.Random
    ) -> dict[str, Any]:
        """Re-mutate an invalid individual until valid (bounded tries)."""
        for _ in range(_MAX_REDRAW_TRIES):
            ind = self.space.mutate(ind, rate, rng)
            if self.is_valid(ind):
                return ind
        # Fall back to a fresh random draw.
        for _ in range(_MAX_REDRAW_TRIES):
            fresh = self.space.random_individual(rng)
            if self.is_valid(fresh):
                return fresh
        raise RuntimeError("could not generate a valid individual")

    @staticmethod
    def _tournament(
        pop: list[dict[str, Any]],
        fitness: list[float],
        rng: random.Random,
        k: int = 3,
    ) -> dict[str, Any]:
        """Tournament selection: pick the fittest of k random candidates."""
        idxs = [rng.randrange(len(pop)) for _ in range(k)]
        best_i = max(idxs, key=lambda i: fitness[i])
        return pop[best_i]
