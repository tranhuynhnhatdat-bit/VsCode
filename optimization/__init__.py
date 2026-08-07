"""TestEngine: genetic parameter optimization + multi-window validation."""

from optimization.engine import TestEngine, OptimizationResult, PassingStrategy
from optimization.genetic import (
    GAConfig,
    GAReport,
    GeneticOptimizer,
    IndividualRecord,
    ParamSpace,
)

__all__ = [
    "TestEngine",
    "OptimizationResult",
    "PassingStrategy",
    "GAConfig",
    "GAReport",
    "GeneticOptimizer",
    "IndividualRecord",
    "ParamSpace",
]