"""Lightweight genetic algorithm for strategy parameter optimization.

Design (from grilling session):
- Mixed param space: numeric ranges {min, max, step} and categorical lists
- An individual is a dict[str, Any] dropped into StrategyClass(**params)
- Tournament selection (k=3), elitism (top N), uniform crossover,
  per-gene mutation with bounded re-draw on invalid individuals
- Structural constraints e.g. ("fast", "<", "slow") reject invalid births
- Early stop after N stagnant generations; optional evaluation budget
- Pure numeric engine: fitness is a caller-supplied float

Island model (added for diversity):
- N independent populations (islands) evolve in parallel
- Ring migration: every `migration_interval` generations, each island sends
  its top-`migration_count` fittest individuals to the ring neighbor,
  replacing the worst in the destination
- Restart-on-stagnation: an island that stagnates for
  `restart_stagnation` generations resets its population to fresh seeds
  (shared collected set + global best persist across restarts)
- A caller-supplied `collect_fn` is called for every evaluated individual;
  it may add the individual to a shared "collected" set. The run stops
  when `stop_fn()` returns True (e.g. collected set reached target size)
  or the evaluation budget is exhausted.
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
    # Island model.
    islands: int = 1
    migration_interval: int = 5
    migration_count: int = 2
    restart_stagnation: int = 3


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

    Supports an island model: `config.islands` independent populations that
    migrate on a ring topology. A caller-supplied `collect_fn` is called for
    each evaluated individual; a `stop_fn` decides when to halt (e.g. a
    shared collected set reached its target size).
    """

    def __init__(
        self,
        param_space: ParamSpace,
        fitness_fn: FitnessFn,
        config: GAConfig,
        constraints: list[tuple[str, str, str]] | None = None,
        batch_fitness_fn: Callable[[list[dict[str, Any]]], list[float]]
        | None = None,
        collect_fn: Callable[[dict[str, Any], float], None] | None = None,
        stop_fn: Callable[[], bool] | None = None,
    ) -> None:
        if config.population < 1:
            raise ValueError("population must be >= 1")
        if config.generations < 1:
            raise ValueError("generations must be >= 1")
        if config.elitism > config.population:
            raise ValueError("elitism cannot exceed population")
        if not (0.0 < config.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in (0, 1]")
        if config.islands < 1:
            raise ValueError("islands must be >= 1")
        self.space = param_space
        self.fitness_fn = fitness_fn
        self.batch_fitness_fn = batch_fitness_fn
        self.config = config
        self.is_valid = make_validator(constraints)
        self.rng = random.Random(config.seed)
        self.collect_fn = collect_fn
        self.stop_fn = stop_fn
        # params-key -> last fitness, for migration ranking.
        self._fitness_cache: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> GAReport:
        """Run the search and return the best individual + full history."""
        cfg = self.config
        rng = self.rng
        history: list[IndividualRecord] = []
        order = 0
        best_fitness = float("-inf")
        best_individual: dict[str, Any] | None = None

        def _eval(ind: dict[str, Any]) -> float:
            nonlocal order, best_fitness, best_individual
            fit = self.fitness_fn(ind)
            history.append(IndividualRecord(dict(ind), fit, order))
            self._fitness_cache[self._key(ind)] = fit
            order += 1
            if self.collect_fn is not None:
                self.collect_fn(ind, fit)
            if fit > best_fitness:
                best_fitness = fit
                best_individual = dict(ind)
            return fit

        def _eval_batch(pop: list[dict[str, Any]]) -> list[float]:
            nonlocal order, best_fitness, best_individual
            fits = self.batch_fitness_fn(pop)
            out: list[float] = []
            for ind, fit in zip(pop, fits):
                history.append(IndividualRecord(dict(ind), fit, order))
                self._fitness_cache[self._key(ind)] = fit
                order += 1
                if self.collect_fn is not None:
                    self.collect_fn(ind, fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = dict(ind)
                out.append(fit)
            return out

        def _should_stop() -> bool:
            if self.stop_fn is not None and self.stop_fn():
                return True
            if cfg.max_evaluations is not None and order >= cfg.max_evaluations:
                return True
            return False

        # Initial populations: one per island.
        pop_islands: list[list[dict[str, Any]]] = [
            self._initial_population(rng) for _ in range(cfg.islands)
        ]
        # Per-island stagnation + best-fitness tracking.
        island_best: list[float] = [float("-inf")] * cfg.islands
        island_stagnant: list[int] = [0] * cfg.islands

        for gen in range(cfg.generations):
            for i_island, pop in enumerate(pop_islands):
                if self.batch_fitness_fn is not None and cfg.workers > 1:
                    fitness = _eval_batch(pop)
                else:
                    fitness = [_eval(ind) for ind in pop]

                gen_best = max(fitness)
                if gen_best > island_best[i_island]:
                    island_best[i_island] = gen_best
                    island_stagnant[i_island] = 0
                else:
                    island_stagnant[i_island] += 1

                # Evaluate stagnation / restart for this island.
                if island_stagnant[i_island] >= cfg.restart_stagnation:
                    print(
                        f"  Island {i_island + 1} stagnated "
                        f"{island_stagnant[i_island]} gens — restarting "
                        f"population (collected set preserved)"
                    )
                    pop_islands[i_island] = self._initial_population(
                        rng, warm=best_individual
                    )
                    island_best[i_island] = float("-inf")
                    island_stagnant[i_island] = 0
                    continue

                # Build next generation for this island.
                pop_islands[i_island] = self._next_generation(
                    pop, fitness, rng
                )

            # Global print.
            print(
                f"  GA gen {gen + 1}/{cfg.generations} | "
                f"best fitness {best_fitness:.4f} | evals {order}"
            )

            # Migration (ring topology) after each generation.
            if cfg.islands > 1 and (gen + 1) % cfg.migration_interval == 0:
                self._migrate(pop_islands, rng)

            # Stop conditions.
            if _should_stop():
                if self.stop_fn is not None and self.stop_fn():
                    print(
                        f"  GA stopping: collection target reached "
                        f"({order} evals)"
                    )
                elif cfg.max_evaluations is not None:
                    print(
                        f"  GA stopping: evaluation budget "
                        f"{cfg.max_evaluations} reached ({order} evals)"
                    )
                break
            if gen == cfg.generations - 1:
                break

        best = (
            IndividualRecord(best_individual, best_fitness, 0)
            if best_individual is not None
            else None
        )
        return GAReport(best=best, history=history)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _initial_population(
        self,
        rng: random.Random,
        warm: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a fresh population; optionally warm-start with `warm`."""
        cfg = self.config
        pop: list[dict[str, Any]] = []
        for s in cfg.initial_population or []:
            seed = {k: v for k, v in s.items() if k in self.space.names}
            if self.is_valid(seed):
                pop.append(seed)
        if warm is not None and self.is_valid(warm):
            pop.append(dict(warm))
        while len(pop) < cfg.population:
            ind = self.space.random_individual(rng)
            if self.is_valid(ind):
                pop.append(ind)
        return pop[: cfg.population]

    def _next_generation(
        self,
        pop: list[dict[str, Any]],
        fitness: list[float],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        """Build the next generation: elitism + children."""
        cfg = self.config
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
        return next_pop[: cfg.population]

    def _migrate(
        self,
        pop_islands: list[list[dict[str, Any]]],
        rng: random.Random,
    ) -> None:
        """Ring migration: each island sends its top-K to the ring neighbor."""
        cfg = self.config
        n = len(pop_islands)
        if n < 2:
            return

        # Compute the top-K emigrants per island in advance.
        emigrants: list[list[dict[str, Any]]] = []
        for pop in pop_islands:
            ranked = sorted(
                pop,
                key=lambda ind: self._fitness_cache.get(
                    self._key(ind), float("-inf")
                ),
                reverse=True,
            )
            emigrants.append([dict(ind) for ind in ranked[: cfg.migration_count]])

        # Replace the worst-K in each destination island.
        for i in range(n):
            dest = (i + 1) % n
            dest_pop = pop_islands[dest]
            # Find the worst-K indices by last fitness.
            scored = [
                (idx, self._fitness_cache.get(self._key(ind), float("inf")))
                for idx, ind in enumerate(dest_pop)
            ]
            scored.sort(key=lambda t: t[1], reverse=True)  # worst first
            worst_idx = [idx for idx, _ in scored[: cfg.migration_count]]
            for idx, emigrant in zip(worst_idx, emigrants[i]):
                dest_pop[idx] = emigrant

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(ind: dict[str, Any]) -> str:
        import json

        return json.dumps(ind, sort_keys=True, default=str)

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