"""Condition primitives for composable strategies.

A Condition is a serializable `left [op] right` expression evaluated on an
OHLCV DataFrame. Both operands can be a constant, a price (Open/High/Low/
Close), or an indicator value. Operators are the MQL5-common comparisons
plus one-bar cross detection.

The GA emits conditions via categorical genes in a ParamSpace; the strategy
constructor expands those genes into Condition objects and ANDs/ORs their
boolean Series together with the base time/session logic.

Scale model (single source of truth):
- Each selectable indicator line belongs to a parent indicator and a scale
  (price / oscillator / distance). The scale determines which condition
  types are legal:
    * price_ind   : price vs price-scale indicator
    * price_price : price vs price (e.g. Close > Open)
    * ind_const   : oscillator-scale indicator vs constant (per-indicator
                    threshold range)
    * ind_ind     : indicator vs indicator, both sides SAME scale
- Thresholds are only meaningful for oscillator-scale indicators; the
  per-indicator threshold range is used for validation and rendering.
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
KIND_OPEN = "open"
KIND_HIGH = "high"
KIND_LOW = "low"
KIND_INDICATOR = "indicator"

# Price operand kinds (all price-scale).
PRICE_KINDS = (KIND_OPEN, KIND_HIGH, KIND_LOW, KIND_CLOSE)

# Scales.
SCALE_PRICE = "price"
SCALE_OSCILLATOR = "oscillator"
SCALE_DISTANCE = "distance"

# Condition types the GA can emit.
TYPE_NONE = "none"
TYPE_PRICE_IND = "price_ind"
TYPE_PRICE_PRICE = "price_price"
TYPE_IND_CONST = "ind_const"
TYPE_IND_IND = "ind_ind"
CONDITION_TYPES = (
    TYPE_NONE,
    TYPE_PRICE_IND,
    TYPE_PRICE_PRICE,
    TYPE_IND_CONST,
    TYPE_IND_IND,
)

# Indicator registry: line_name -> (parent, scale, threshold_range_or_None).
# threshold_range is the set of legal constant thresholds for oscillator-scale
# indicators (used for validation + rendering). None = no threshold allowed.
INDICATOR_REGISTRY: dict[str, tuple[str, str, tuple[float, ...] | None]] = {
    "SMA": ("SMA", SCALE_PRICE, None),
    "EMA": ("EMA", SCALE_PRICE, None),
    "ATR": ("ATR", SCALE_DISTANCE, None),
    "RSI": ("RSI", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "CCI": ("CCI", SCALE_OSCILLATOR, (-200.0, -100.0, 100.0, 200.0)),
    "Stoch_K": ("Stochastic", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "Stoch_D": ("Stochastic", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "ADX": ("ADX", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "PlusDI": ("ADX", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "MinusDI": ("ADX", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "BB_Upper": ("Bollinger", SCALE_PRICE, None),
    "BB_Lower": ("Bollinger", SCALE_PRICE, None),
    "MACD_Main": ("MACD", SCALE_DISTANCE, None),
    "MACD_Signal": ("MACD", SCALE_DISTANCE, None),
    "Momentum": ("Momentum", SCALE_OSCILLATOR, (-100.0, -50.0, 0.0, 50.0, 100.0)),
    "WPR": ("WPR", SCALE_OSCILLATOR, (-80.0, -50.0, -20.0)),
    "MFI": ("MFI", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0)),
    "OBV": ("OBV", SCALE_DISTANCE, None),
    "Tenkan": ("Ichimoku", SCALE_PRICE, None),
    "Kijun": ("Ichimoku", SCALE_PRICE, None),
    "SenkouA": ("Ichimoku", SCALE_PRICE, None),
    "SenkouB": ("Ichimoku", SCALE_PRICE, None),
    "Chikou": ("Ichimoku", SCALE_PRICE, None),
}

# The set of indicator line names the GA can choose from.
INDICATOR_NAMES = tuple(INDICATOR_REGISTRY.keys())

# Operator choices for the GA.
OP_CHOICES = ("gt", "lt", "crosses_above", "crosses_below")

# The "empty" indicator sentinel in a condition slot (no condition).
NONE = "none"


@dataclass
class Operand:
    """One side of a condition expression.

    kind:
      const     -> `value` is the constant
      open/high/low/close -> the bar's price
      indicator -> `indicator` line name + shared params (period, param2)
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
        if self.kind in PRICE_KINDS:
            col = {
                KIND_OPEN: "Open",
                KIND_HIGH: "High",
                KIND_LOW: "Low",
                KIND_CLOSE: "Close",
            }[self.kind]
            return df[col].astype(float)
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


def price(kind: str) -> Operand:
    if kind not in PRICE_KINDS:
        raise ValueError(f"price kind must be one of {PRICE_KINDS}, got {kind!r}")
    return Operand(kind=kind)


def close() -> Operand:
    return Operand(kind=KIND_CLOSE)


def indicator(
    name: str, period: int | None = None, param2: int | None = None
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
    """Compute a single indicator line series by name.

    Multi-line indicators return the requested line; shared params are read
    from `period` / `param2` (the GA's global per-parent params).
    """
    if name is None:
        raise ValueError("indicator operand requires a name")
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
    if name == "Stoch_K":
        k, _d = ind.Stochastic(df, k_period=period, d_period=param2 or 3)
        return k
    if name == "Stoch_D":
        k, d = ind.Stochastic(df, k_period=period, d_period=param2 or 3)
        return d
    if name == "ADX":
        return ind.ADX(df, period)
    if name == "PlusDI":
        return ind.PlusDI(df, period)
    if name == "MinusDI":
        return ind.MinusDI(df, period)
    if name == "BB_Upper":
        upper, _lower = ind.Bollinger(df, period, param2 or 2.0)
        return upper
    if name == "BB_Lower":
        _upper, lower = ind.Bollinger(df, period, param2 or 2.0)
        return lower
    if name == "MACD_Main":
        main, _sig = ind.MACD(df, period, param2 or 26)
        return main
    if name == "MACD_Signal":
        _main, sig = ind.MACD(df, period, param2 or 26)
        return sig
    if name == "Momentum":
        return ind.Momentum(df, period)
    if name == "WPR":
        return ind.WPR(df, period)
    if name == "MFI":
        return ind.MFI(df, period)
    if name == "OBV":
        return ind.OBV(df)
    if name == "Tenkan":
        t, _k, _a, _b, _c = ind.Ichimoku(df, period, param2 or 26, 52)
        return t
    if name == "Kijun":
        _t, k, _a, _b, _c = ind.Ichimoku(df, period, param2 or 26, 52)
        return k
    if name == "SenkouA":
        _t, _k, a, _b, _c = ind.Ichimoku(df, period, param2 or 26, 52)
        return a
    if name == "SenkouB":
        _t, _k, _a, b, _c = ind.Ichimoku(df, period, param2 or 26, 52)
        return b
    if name == "Chikou":
        _t, _k, _a, _b, c = ind.Ichimoku(df, period, param2 or 26, 52)
        return c
    raise ValueError(f"Unknown indicator: {name!r}")


# ------------------------------------------------------------------ #
# GA gene expansion
# ------------------------------------------------------------------ #
def condition_from_genes(
    genes: dict[str, Any], prefix: str
) -> Condition | None:
    """Build a Condition from GA slot genes, or None if the slot is empty.

    Expected genes (prefixed by `prefix`):
      <prefix>_type: 'none' | 'price_ind' | 'price_price' | 'ind_const' | 'ind_ind'
      <prefix>_op:   one of OP_CHOICES
      <prefix>_ind:  indicator line name (for indicator sides)
      <prefix>_ind2: indicator line name (for ind_ind right side)
      <prefix>_price: price kind (for price_price / price_ind left)
      <prefix>_price2: price kind (for price_price right)
      <prefix>_threshold: float (for ind_const)

    Scale validation:
      - price_ind : right side must be price-scale
      - ind_const : left side must be oscillator-scale
      - ind_ind   : both sides must share the same scale
    If a slot's genes violate a scale rule, the slot is treated as `none`
    (the GA wastes no fitness on structurally meaningless conditions).
    """
    ctype = genes.get(f"{prefix}type", NONE)
    if ctype == NONE:
        return None
    op = genes.get(f"{prefix}op")
    ind_name = genes.get(f"{prefix}ind")
    ind2_name = genes.get(f"{prefix}ind2")
    price_kind = genes.get(f"{prefix}price")
    price2_kind = genes.get(f"{prefix}price2")
    threshold = genes.get(f"{prefix}threshold")

    if ctype == TYPE_PRICE_IND:
        # price [op] price-scale indicator
        if ind_name not in INDICATOR_REGISTRY:
            return None
        if INDICATOR_REGISTRY[ind_name][1] != SCALE_PRICE:
            return None
        return Condition(
            op=op,
            left=price(price_kind if price_kind in PRICE_KINDS else KIND_CLOSE),
            right=indicator(ind_name),
        )
    if ctype == TYPE_PRICE_PRICE:
        # price [op] price (e.g. Close > Open)
        return Condition(
            op=op,
            left=price(price_kind if price_kind in PRICE_KINDS else KIND_CLOSE),
            right=price(price2_kind if price2_kind in PRICE_KINDS else KIND_OPEN),
        )
    if ctype == TYPE_IND_CONST:
        # oscillator-scale indicator [op] constant
        if ind_name not in INDICATOR_REGISTRY:
            return None
        if INDICATOR_REGISTRY[ind_name][1] != SCALE_OSCILLATOR:
            return None
        return Condition(
            op=op,
            left=indicator(ind_name),
            right=const(float(threshold) if threshold is not None else 50.0),
        )
    if ctype == TYPE_IND_IND:
        # indicator [op] indicator, both sides SAME scale
        if ind_name not in INDICATOR_REGISTRY or ind2_name not in INDICATOR_REGISTRY:
            return None
        if INDICATOR_REGISTRY[ind_name][1] != INDICATOR_REGISTRY[ind2_name][1]:
            return None
        return Condition(
            op=op,
            left=indicator(ind_name),
            right=indicator(ind2_name),
        )
    raise ValueError(f"Unknown condition type: {ctype!r}")