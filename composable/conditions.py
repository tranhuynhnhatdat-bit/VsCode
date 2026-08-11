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

The indicator line registry (``INDICATOR_REGISTRY``) is the single source of
truth for every selectable line: its parent, scale, threshold range, the
global GA param keys that feed it, and the compute callable that produces its
series. ``conditions.py``, ``composable.py``, and the GA param space all
derive from it — add a new indicator line here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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

# Operator choices for the GA.
OP_CHOICES = ("gt", "lt", "crosses_above", "crosses_below")

# The "empty" indicator sentinel in a condition slot (no condition).
NONE = "none"


# ------------------------------------------------------------------ #
# Indicator line registry (single source of truth)
# ------------------------------------------------------------------ #
@dataclass(frozen=True)
class IndicatorSpec:
    """One selectable indicator line's full specification.

    This is the single source of truth for everything a line needs across
    the codebase:

    - ``name``: line name used in condition genes / operands.
    - ``parent``: owning indicator (SMA, RSI, Stochastic, ...).
    - ``scale``: price / oscillator / distance (drives condition legality).
    - ``threshold``: legal constant thresholds for oscillator-scale lines,
      or None if a constant threshold is meaningless for this line.
    - ``period_key`` / ``param2_key``: the global GA param keys that feed
      this line's period and second parameter (None if the line has no such
      param). OBV has neither.
    - ``compute``: callable(df, period, param2) -> Series for this single
      line. Multi-line indicators (Stochastic, Bollinger, MACD, Ichimoku)
      each get a thin wrapper that calls the shared function and returns the
      requested line.
    """

    name: str
    parent: str
    scale: str
    threshold: tuple[float, ...] | None
    period_key: str | None
    param2_key: str | None
    compute: Callable[[pd.DataFrame, int | None, int | None], pd.Series]


def _sma(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.SMA(df["Close"], period)


def _ema(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.EMA(df["Close"], period)


def _atr(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.ATR(df, period)


def _rsi(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.RSI(df, period)


def _cci(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.CCI(df, period)


def _stoch_k(df: pd.DataFrame, period: int | None, param2: int | None = None):
    k, _d = ind.Stochastic(df, k_period=period, d_period=param2 or 3)
    return k


def _stoch_d(df: pd.DataFrame, period: int | None, param2: int | None = None):
    k, d = ind.Stochastic(df, k_period=period, d_period=param2 or 3)
    return d


def _adx(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.ADX(df, period)


def _plusdi(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.PlusDI(df, period)


def _minusdi(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.MinusDI(df, period)


def _bb_upper(df: pd.DataFrame, period: int | None, param2: int | None = None):
    upper, _lower = ind.Bollinger(df, period, param2 or 2.0)
    return upper


def _bb_lower(df: pd.DataFrame, period: int | None, param2: int | None = None):
    _upper, lower = ind.Bollinger(df, period, param2 or 2.0)
    return lower


def _macd_main(df: pd.DataFrame, period: int | None, param2: int | None = None):
    main, _sig = ind.MACD(df, period, param2 or 26)
    return main


def _macd_signal(df: pd.DataFrame, period: int | None, param2: int | None = None):
    _main, sig = ind.MACD(df, period, param2 or 26)
    return sig


def _momentum(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.Momentum(df, period)


def _wpr(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.WPR(df, period)


def _mfi(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.MFI(df, period)


def _obv(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return ind.OBV(df)


def _ichimoku_line(
    df: pd.DataFrame, period: int | None, param2: int | None, which: int
) -> pd.Series:
    t, k, a, b, c = ind.Ichimoku(df, period, param2 or 26, 52)
    return (t, k, a, b, c)[which]


def _tenkan(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return _ichimoku_line(df, period, param2, 0)


def _kijun(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return _ichimoku_line(df, period, param2, 1)


def _senkou_a(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return _ichimoku_line(df, period, param2, 2)


def _senkou_b(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return _ichimoku_line(df, period, param2, 3)


def _chikou(df: pd.DataFrame, period: int | None, param2: int | None = None):
    return _ichimoku_line(df, period, param2, 4)


# line_name -> IndicatorSpec. Add a new indicator line here ONLY.
INDICATOR_REGISTRY: dict[str, IndicatorSpec] = {
    "SMA": IndicatorSpec("SMA", "SMA", SCALE_PRICE, None, "sma_period", None, _sma),
    "EMA": IndicatorSpec("EMA", "EMA", SCALE_PRICE, None, "ema_period", None, _ema),
    "ATR": IndicatorSpec("ATR", "ATR", SCALE_DISTANCE, None, "atr_period", None, _atr),
    "RSI": IndicatorSpec(
        "RSI", "RSI", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0),
        "rsi_period", None, _rsi,
    ),
    "CCI": IndicatorSpec(
        "CCI", "CCI", SCALE_OSCILLATOR, (-200.0, -100.0, 100.0, 200.0),
        "cci_period", None, _cci,
    ),
    "Stoch_K": IndicatorSpec(
        "Stoch_K", "Stochastic", SCALE_OSCILLATOR,
        (20.0, 30.0, 50.0, 70.0, 80.0), "stoch_k", "stoch_d", _stoch_k,
    ),
    "Stoch_D": IndicatorSpec(
        "Stoch_D", "Stochastic", SCALE_OSCILLATOR,
        (20.0, 30.0, 50.0, 70.0, 80.0), "stoch_k", "stoch_d", _stoch_d,
    ),
    "ADX": IndicatorSpec(
        "ADX", "ADX", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0),
        "adx_period", None, _adx,
    ),
    "PlusDI": IndicatorSpec(
        "PlusDI", "ADX", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0),
        "adx_period", None, _plusdi,
    ),
    "MinusDI": IndicatorSpec(
        "MinusDI", "ADX", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0),
        "adx_period", None, _minusdi,
    ),
    "BB_Upper": IndicatorSpec(
        "BB_Upper", "Bollinger", SCALE_PRICE, None, "bb_period", "bb_stddev",
        _bb_upper,
    ),
    "BB_Lower": IndicatorSpec(
        "BB_Lower", "Bollinger", SCALE_PRICE, None, "bb_period", "bb_stddev",
        _bb_lower,
    ),
    "MACD_Main": IndicatorSpec(
        "MACD_Main", "MACD", SCALE_DISTANCE, None, "macd_fast", "macd_slow",
        _macd_main,
    ),
    "MACD_Signal": IndicatorSpec(
        "MACD_Signal", "MACD", SCALE_DISTANCE, None, "macd_fast", "macd_slow",
        _macd_signal,
    ),
    "Momentum": IndicatorSpec(
        "Momentum", "Momentum", SCALE_OSCILLATOR,
        (-100.0, -50.0, 0.0, 50.0, 100.0), "mom_period", None, _momentum,
    ),
    "WPR": IndicatorSpec(
        "WPR", "WPR", SCALE_OSCILLATOR, (-80.0, -50.0, -20.0),
        "wpr_period", None, _wpr,
    ),
    "MFI": IndicatorSpec(
        "MFI", "MFI", SCALE_OSCILLATOR, (20.0, 30.0, 50.0, 70.0, 80.0),
        "mfi_period", None, _mfi,
    ),
    "OBV": IndicatorSpec(
        "OBV", "OBV", SCALE_DISTANCE, None, None, None, _obv,
    ),
    "Tenkan": IndicatorSpec(
        "Tenkan", "Ichimoku", SCALE_PRICE, None, "ichi_tenkan", "ichi_kijun",
        _tenkan,
    ),
    "Kijun": IndicatorSpec(
        "Kijun", "Ichimoku", SCALE_PRICE, None, "ichi_tenkan", "ichi_kijun",
        _kijun,
    ),
    "SenkouA": IndicatorSpec(
        "SenkouA", "Ichimoku", SCALE_PRICE, None, "ichi_tenkan", "ichi_kijun",
        _senkou_a,
    ),
    "SenkouB": IndicatorSpec(
        "SenkouB", "Ichimoku", SCALE_PRICE, None, "ichi_tenkan", "ichi_kijun",
        _senkou_b,
    ),
    "Chikou": IndicatorSpec(
        "Chikou", "Ichimoku", SCALE_PRICE, None, "ichi_tenkan", "ichi_kijun",
        _chikou,
    ),
}

# The set of indicator line names the GA can choose from.
INDICATOR_NAMES = tuple(INDICATOR_REGISTRY.keys())

# Global GA param keys that exist in the param space but are not consumed by
# any single line's period/param2 (reserved for future wiring). Kept next to
# the registry so the global-param-key set stays in one place.
EXTRA_GLOBAL_PARAM_KEYS = frozenset({"stoch_slowing", "ichi_senkou"})


def build_parent_params() -> dict[str, tuple[str | None, str | None]]:
    """parent -> (period_key, param2_key), derived from the registry."""
    out: dict[str, tuple[str | None, str | None]] = {}
    for spec in INDICATOR_REGISTRY.values():
        if spec.parent not in out:
            out[spec.parent] = (spec.period_key, spec.param2_key)
    return out


def build_global_param_keys() -> set[str]:
    """All global GA param keys (period/param2 across lines + extras)."""
    keys = set(EXTRA_GLOBAL_PARAM_KEYS)
    for spec in INDICATOR_REGISTRY.values():
        if spec.period_key:
            keys.add(spec.period_key)
        if spec.param2_key:
            keys.add(spec.param2_key)
    return keys


# ------------------------------------------------------------------ #
# Condition-slot gene layout (single source of truth)
# ------------------------------------------------------------------ #
# The per-slot fields that make up a condition's GA genes. Both the GA
# param space (build_param_space) and the optimizer's behavioral-diversity
# signature (CONDITION_GENES) derive the slot gene names from this list, so
# the layout lives in exactly one place.
CONDITION_GENE_FIELDS = (
    "type", "op", "ind", "ind2", "price", "price2", "threshold",
)


def condition_gene_names(max_conditions: int) -> tuple[str, ...]:
    """All condition-slot gene names for `max_conditions` slots.

    e.g. max_conditions=2 ->
      cond1_type, cond1_op, ..., cond2_type, ..., cond2_threshold
    """
    names: list[str] = []
    for i in range(1, max_conditions + 1):
        prefix = f"cond{i}_"
        for f in CONDITION_GENE_FIELDS:
            names.append(f"{prefix}{f}")
    return tuple(names)


def _indicator_series(
    df: pd.DataFrame,
    name: str | None,
    period: int | None,
    param2: int | None = None,
) -> pd.Series:
    """Compute a single indicator line series by name (data-driven).

    Multi-line indicators return the requested line; shared params are read
    from `period` / `param2` (the GA's global per-parent params).
    """
    if name is None:
        raise ValueError("indicator operand requires a name")
    try:
        spec = INDICATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown indicator: {name!r}") from None
    return spec.compute(df, period, param2)


# ------------------------------------------------------------------ #
# Operands and conditions
# ------------------------------------------------------------------ #
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
        if INDICATOR_REGISTRY[ind_name].scale != SCALE_PRICE:
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
        if INDICATOR_REGISTRY[ind_name].scale != SCALE_OSCILLATOR:
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
        if INDICATOR_REGISTRY[ind_name].scale != INDICATOR_REGISTRY[ind2_name].scale:
            return None
        return Condition(
            op=op,
            left=indicator(ind_name),
            right=indicator(ind2_name),
        )
    raise ValueError(f"Unknown condition type: {ctype!r}")