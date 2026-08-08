"""Condition primitives for composable strategies.

A Condition is a serializable `left [op] right` expression evaluated on an
OHLCV DataFrame. Both operands can be a constant, the close price, or an
indicator value. Operators are the MQL5-common comparisons plus one-bar
cross detection.

The GA emits conditions via categorical genes in a ParamSpace; the strategy
constructor expands those genes into Condition objects and ANDs/ORs their
boolean Series together with the base time/session logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from composable import indicators as ind

# Operator names available to conditions.
OPS = ("gt", "lt", "crosses_above", "crosses_below")

# Operand kinds.
KIND_CONST = "const"
KIND_CLOSE = "close"
KIND_INDICATOR = "indicator"


@dataclass
class Operand:
    """One side of a condition expression.

    kind:
      const     -> `value` is the constant
      close     -> the bar's Close price
      indicator -> `indicator` name + `period` (and optional `param2`)
    """

    kind: str
    value: float | None = None
    indicator: str | None = None
    period: int | None = None
    param2: int | None = None

    def series(self, df: pd.DataFrame) -> pd.Series:
        """Return the operand's numeric series over `df`."""
        if self.kind == KIND_CONST:
            if self.value is None:
                raise ValueError("const operand requires a value")
            return pd.Series(
                float(self.value), index=df.index, dtype=float
            )
        if self.kind == KIND_CLOSE:
            return df["Close"].astype(float)
        if self.kind == KIND_INDICATOR:
            return _indicator_series(
                df,
                self.indicator,
                self.period,
                self.param2,
            )
        raise ValueError(f"Unknown operand kind: {self.kind!r}")


@dataclass
class Condition:
    """A `left [op] right` boolean condition over an OHLCV DataFrame."""

    op: str
    left: Operand
    right: Operand

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series over `df` where the condition holds."""
        if self.op not in OPS:
            raise ValueError(f"Unknown operator: {self.op!r}")
        l = self.left.series(df)
        r = self.right.series(df)
        if self.op == "gt":
            out = l > r
        elif self.op == "lt":
            out = l < r
        elif self.op == "crosses_above":
            out = (l.shift(1) <= r.shift(1)) & (l > r)
        else:  # crosses_below
            out = (l.shift(1) >= r.shift(1)) & (l < r)
        return out.fillna(False).astype(bool)


# ------------------------------------------------------------------ #
# Operand builders
# ------------------------------------------------------------------ #
def const(value: float) -> Operand:
    return Operand(kind=KIND_CONST, value=value)


def close() -> Operand:
    return Operand(kind=KIND_CLOSE)


def indicator(
    name: str, period: int, param2: int | None = None
) -> Operand:
    return Operand(
        kind=KIND_INDICATOR, indicator=name, period=period, param2=param2
    )


# ------------------------------------------------------------------ #
# Indicator value series
# ------------------------------------------------------------------ #
def _indicator_series(
    df: pd.DataFrame,
    name: str | None,
    period: int | None,
    param2: int | None = None,
) -> pd.Series:
    """Compute a single indicator value series by name.

    Stochastic returns its %K line (the primary/default buffer).
    """
    if name is None or period is None:
        raise ValueError("indicator operand requires name and period")
    if name == "SMA":
        return ind.SMA(df["Close"], period)
    if name == "EMA":
        return ind.EMA(df["Close"], period)
    if name == "ATR":
        return ind.ATR(df, period)
    if name == "RSI":
        return ind.RSI(df, period)
    if name == "CCI":
        return ind.CCI(df, period)
    if name == "Stochastic":
        k, _d = ind.Stochastic(df, k_period=period)
        return k
    if name == "ADX":
        return ind.ADX(df, period)
    if name == "PlusDI":
        return ind.PlusDI(df, period)
    if name == "MinusDI":
        return ind.MinusDI(df, period)
    raise ValueError(f"Unknown indicator: {name!r}")


# ------------------------------------------------------------------ #
# GA gene expansion
# ------------------------------------------------------------------ #
# The set of indicator names the GA can choose from.
INDICATOR_NAMES = ("SMA", "EMA", "ATR", "RSI", "CCI", "Stochastic", "ADX")

# Operator choices for the GA.
OP_CHOICES = ("gt", "lt", "crosses_above", "crosses_below")

# The "empty" indicator sentinel in a condition slot (no condition).
NONE = "none"


def condition_from_genes(
    genes: dict[str, Any], prefix: str
) -> Condition | None:
    """Build a Condition from GA slot genes, or None if the slot is empty.

    Expected genes (prefixed by `prefix`):
      <prefix>_type: 'none' | 'price_ind' | 'price_const' | 'ind_const'
      <prefix>_op:   one of OP_CHOICES
      <prefix>_period: int (for the indicator side)
      <prefix>_ind:  indicator name (for indicator sides)
      <prefix>_threshold: float (for const sides)
    """
    ctype = genes.get(f"{prefix}type", NONE)
    if ctype == NONE:
        return None
    op = genes.get(f"{prefix}op")
    period = genes.get(f"{prefix}period")
    ind_name = genes.get(f"{prefix}ind")
    threshold = genes.get(f"{prefix}threshold")

    if ctype == "price_ind":
        # Close [op] Indicator(period)
        return Condition(
            op=op,
            left=close(),
            right=indicator(ind_name, period),
        )
    if ctype == "ind_const":
        # Indicator(period) [op] constant
        return Condition(
            op=op,
            left=indicator(ind_name, period),
            right=const(threshold),
        )
    if ctype == "ind_ind":
        # Indicator(period) [op] Indicator(period2)
        period2 = genes.get(f"{prefix}period2")
        return Condition(
            op=op,
            left=indicator(ind_name, period),
            right=indicator(genes.get(f"{prefix}ind2"), period2),
        )
    raise ValueError(f"Unknown condition type: {ctype!r}")