"""Backtesting engine: vectorbt-based, HTF signals mapped to M1 execution.

Design (from grilling session):
- Thin wrapper around vectorbt Portfolio.from_signals()
- Entries/exits fill at the M1 bar open at/after the HTF actionable time
  (the first tick of the target M1 bar, matching the MQL5 engine)
- Long entries fill at ASK (open + spread); exits fill at BID (open).
  The spread cost is captured through the ask/bid fill differential of the
  shared M1 core (a long buys at ask, exits at bid), matching the H1 screen's
  single spread cost (spread_points * tick_value * size). The H1 screen
  (run_htf) charges the spread as a single fixed fee on entry from the
  per-bar Spread column instead — both paths model the same one-time cost.
- SL-first on simultaneous SL/TP hit; gap-through-SL fills at the worse price
- Fixed risk money per trade -> lot per entry (matches MQL5 RiskToLots:
  lots = riskMoney / (stopDistance * tickValue/tickSize), rounded down to
  volume_step, rejected if outside [volume_min, volume_max])
- SL can be expressed as an absolute price or as a DISTANCE (sl_is_distance).
  Distance-based SL is converted to an absolute stop relative to the entry
  fill price (ask for longs, bid for shorts), matching the MQL5 engine.
- All costs: spread (fixed_fees on entry), swap (post-processed per day held,
  Wed 3-day rule), commission (hardcoded 0)
- Daily equity curve; matplotlib PNG save
- Single position: new entries while in a position are ignored
- Forex PnL: prices scaled by tick_value/tick_size so vectorbt's
  (exit-entry)*size equals real PnL in deposit currency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest._m1_core import compute_metrics, run_m1
from backtest._mapping import HTF_OFFSET as _HTF_OFFSET
from data_manager import DataManager
from symbol_info import SymbolInfo
from strategy.base import StrategySignals

# Results directory for equity curve PNGs.
RESULTS_DIR = Path(r"C:\Users\DAT\Desktop\VsCode\results")

# Commission is modeled but set to 0 for now (FTMO default).
COMMISSION = 0.0

# Cash ceiling passed to vectorbt so position sizes are never reduced by
# available cash. The MQL5 engine sizes positions purely from risk money
# (margin is a separate concern), so the backtest must not let cash cap
# the lot size. The equity curve is shifted back to start at
# initial_capital after the run.
_CASH_CEILING = 1e15


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    metrics: dict[str, float | str]
    equity_curve: pd.Series
    trades: pd.DataFrame
    symbol: str
    timeframe: str
    strategy_name: str

    def save_equity_curve(self) -> Path:
        """Save the daily equity curve as a PNG in results/."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{self.strategy_name}_{self.symbol}_{self.timeframe}.png"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.equity_curve.index, self.equity_curve.values, lw=1.2)
        ax.set_title(
            f"Equity Curve — {self.strategy_name} {self.symbol} {self.timeframe}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USD)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


class BacktestEngine:
    """Runs a StrategySignals set against M1 data via vectorbt."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        risk_money: float = 100.0,
        initial_capital: float = 10_000.0,
        strategy_name: str = "1",
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_money = risk_money
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name

        self._dm = DataManager()
        self._si = SymbolInfo()

        # Cache of precomputed M1 frames + scaled arrays keyed by
        # (start, end). Loading multi-year M1 and re-scaling prices on every
        # run() call is the dominant cost in the M1 funnel; this cache makes
        # repeated runs over the same window near-free.
        self._prep_cache: dict[tuple, dict | None] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, signals: StrategySignals, htf_df: pd.DataFrame) -> BacktestResult:
        """Backtest the given signals on M1 data.

        Uses the shared Numba M1 core (backtest/_m1_core.run_m1), which is the
        same execution path the event-driven engine uses, so the two engines
        produce identical trades and metrics. This is both faster than vectorbt
        and exactly matches the event engine's MQL5 "1 minute OHLC" semantics.

        Args:
            signals: StrategySignals indexed by the strategy's HTF DateTime.
            htf_df: The OHLCV DataFrame that was passed to strategy.generate().

        Returns:
            BacktestResult with metrics, daily equity curve, and trades.
        """
        # Pull M1 data for the signal range (plus a buffer for fills), cached
        # along with the symbol metadata and spread.
        start = htf_df.index[0]
        end = htf_df.index[-1] + pd.Timedelta(days=1)
        prep = self._prep_m1(start, end)
        if prep is None:
            return self._empty_result(start, end)

        result = run_m1(
            signals, htf_df, prep["m1"], self.timeframe, prep["info"],
            self.initial_capital, self.risk_money,
        )

        return BacktestResult(
            metrics=result["metrics"],
            equity_curve=result["equity_curve"],
            trades=result["trades"],
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy_name=self.strategy_name,
        )

    def run_htf(self, signals: StrategySignals, htf_df: pd.DataFrame) -> BacktestResult:
        """Backtest on the strategy's native timeframe directly (no M1 mapping).

        Fills at the next HTF bar open after the signal bar; SL/TP evaluated
        on HTF bars only. Faster but less realistic than run() (no intra-bar
        fills). Used by the optimizer's fast H1 screen.
        """
        signals.validate(htf_df)

        info = self._si.get(self.symbol)
        tick_value = float(info["trade_tick_value"])
        tick_size = float(info["trade_tick_size"])
        volume_step = float(info["volume_step"])
        volume_min = float(info["volume_min"])
        volume_max = float(info["volume_max"])
        swap_long = float(info["swap_long"])
        swap_short = float(info["swap_short"])
        swap_rollover3days = int(info["swap_rollover3days"])

        # Per-bar spread (points) from the data; fall back to symbol_info.
        spread_points = self._spread_series(htf_df, info)

        # Fill at the next HTF bar open (no lookahead).
        entries = signals.entries.shift(1).fillna(False).astype(bool)
        exits = signals.exits.shift(1).fillna(False).astype(bool)
        s_entries = signals.short_entries.shift(1).fillna(False).astype(bool)
        s_exits = signals.short_exits.shift(1).fillna(False).astype(bool)

        # Resolve SL: absolute prices for vectorbt + distances for sizing.
        sl_abs, sl_dist = self._resolve_sl_htf(
            signals, entries, s_entries, htf_df["Open"], spread_points, tick_size
        )

        # Per-entry lot sizes (fixed risk money per trade).
        size = self._compute_sizes(
            entries,
            s_entries,
            sl_dist,
            tick_value,
            tick_size,
            volume_step,
            volume_min,
            volume_max,
        )

        # Scale prices so vectorbt PnL = real forex PnL in deposit currency.
        scale = tick_value / tick_size
        close = htf_df["Close"] * scale
        open_ = htf_df["Open"] * scale
        high = htf_df["High"] * scale
        low = htf_df["Low"] * scale
        sl_scaled = sl_abs * scale
        tp_scaled = signals.tp_stop * scale

        # Spread as a fixed fee on entry.
        fixed_fees = pd.Series(0.0, index=htf_df.index)
        entry_bars = entries | s_entries
        fixed_fees.loc[entry_bars] = (
            spread_points.loc[entry_bars] * tick_value * size.loc[entry_bars]
        )

        # Run vectorbt on the HTF bars directly. init_cash is a huge
        # ceiling so position sizes are never reduced by cash.
        import vectorbt as vb

        pf = vb.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            short_entries=s_entries,
            short_exits=s_exits,
            size=size,
            price=open_,
            open=open_,
            high=high,
            low=low,
            sl_stop=sl_scaled,
            tp_stop=tp_scaled,
            fixed_fees=fixed_fees,
            init_cash=_CASH_CEILING,
            lock_cash=False,
            accumulate=False,
            upon_opposite_entry="ignore",
            freq=_HTF_OFFSET[self.timeframe],
        )

        # Post-process swap and build equity curve.
        trades = pf.trades.records
        htf_equity = pf.value()
        swap_series = self._compute_swap(
            trades, htf_df.index, swap_long, swap_short, swap_rollover3days
        )
        # Shift the curve from the cash ceiling back to initial_capital.
        htf_equity_adj = (
            htf_equity - (_CASH_CEILING - self.initial_capital)
            - swap_series.cumsum()
        )
        daily_equity = htf_equity_adj.resample("D").last().dropna()

        metrics = compute_metrics(
            daily_equity, trades, htf_df.index, self.initial_capital
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=daily_equity,
            trades=trades,
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy_name=self.strategy_name,
        )

    # ------------------------------------------------------------------ #
    # M1 preparation (cached)
    # ------------------------------------------------------------------ #
    def _prep_m1(self, start, end) -> dict | None:
        """Load M1 for [start, end] and cache it with the symbol metadata.

        Returns a dict with the M1 frame and SymbolInfo metadata, or None if
        there is no M1 data in the range. Cached by (start, end) so repeated
        runs over the same window skip the expensive M1 load.
        """
        key = (start, end)
        if key in self._prep_cache:
            return self._prep_cache[key]

        m1 = self._dm.load(self.symbol, "M1", start=start, end=end)
        if m1.empty:
            self._prep_cache[key] = None
            return None

        prep = {"m1": m1, "info": self._si.get(self.symbol)}
        self._prep_cache[key] = prep
        return prep

    # ------------------------------------------------------------------ #
    # SL resolution (absolute prices vs distances)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _spread_series(df: pd.DataFrame, info: dict) -> pd.Series:
        """Per-bar spread in points from the data; fall back to symbol_info."""
        if "Spread" in df.columns:
            return df["Spread"].astype(float)
        return pd.Series(float(info["spread"]), index=df.index)

    def _resolve_sl_htf(
        self,
        signals: StrategySignals,
        entries: pd.Series,
        s_entries: pd.Series,
        open_: pd.Series,
        spread_points: pd.Series,
        tick_size: float,
    ) -> tuple[pd.Series, pd.Series]:
        """Resolve SL on the strategy's native timeframe (fill = next bar open)."""
        if signals.sl_is_distance:
            # The distance at the fill bar equals the distance at the signal
            # bar (fixed while held, so shift(1) is a no-op for the value).
            sl_dist = signals.sl_stop
            spread_price = spread_points * tick_size
            sl_abs = pd.Series(np.nan, index=open_.index)

            long_bars = entries & sl_dist.notna()
            sl_abs.loc[long_bars] = (
                open_.loc[long_bars]
                + spread_price.loc[long_bars]
                - sl_dist.loc[long_bars]
            )
            short_bars = s_entries & sl_dist.notna()
            sl_abs.loc[short_bars] = (
                open_.loc[short_bars] + sl_dist.loc[short_bars]
            )
            return sl_abs.ffill(), sl_dist

        sl_abs = signals.sl_stop
        sl_dist = pd.Series(np.nan, index=open_.index)
        long_bars = entries & sl_abs.notna()
        sl_dist.loc[long_bars] = open_.loc[long_bars] - sl_abs.loc[long_bars]
        short_bars = s_entries & sl_abs.notna()
        sl_dist.loc[short_bars] = sl_abs.loc[short_bars] - open_.loc[short_bars]
        return sl_abs, sl_dist

    # ------------------------------------------------------------------ #
    # Lot sizing
    # ------------------------------------------------------------------ #
    def _compute_sizes(
        self,
        m1_entries: pd.Series,
        m1_s_entries: pd.Series,
        m1_sl_dist: pd.Series,
        tick_value: float,
        tick_size: float,
        volume_step: float,
        volume_min: float,
        volume_max: float,
    ) -> pd.Series:
        """Lot per entry = risk_money / (SL distance in dollars per lot).

        Matches MQL5 RiskToLots:
        - lots = risk_money / (stopDistance * tickValue/tickSize)
        - rounded down to volume_step
        - REJECTED (size 0) if outside [volume_min, volume_max]
        """
        size = pd.Series(np.nan, index=m1_entries.index)

        entry_bars = (m1_entries | m1_s_entries) & m1_sl_dist.notna()
        dist = m1_sl_dist.loc[entry_bars]
        dist = dist[dist > 0]
        if len(dist):
            lots = self.risk_money / (dist * tick_value / tick_size)
            size.loc[dist.index] = lots

        # Round down to volume_step (epsilon avoids float floor artifacts).
        size = np.floor(size / volume_step + 1e-9) * volume_step
        # Reject if outside [min, max] (MQL5 returns 0, never clamps).
        size = size.where((size >= volume_min) & (size <= volume_max), 0.0)
        return size.fillna(0.0)

    # ------------------------------------------------------------------ #
    # Swap
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_swap(
        trades: pd.DataFrame,
        m1_index: pd.DatetimeIndex,
        swap_long: float,
        swap_short: float,
        swap_rollover3days: int,
    ) -> pd.Series:
        """M1-indexed swap charges (USD) at each day boundary while held.

        Swap accrues per calendar day held (from entry day through the day
        before exit). The configured rollover weekday carries a 3x charge.

        Vectorized: the last-M1-bar-of-each-day lookup is precomputed once
        into a dict, avoiding an O(n) mask creation per (trade, day).
        """
        swap = pd.Series(0.0, index=m1_index)
        if trades.empty:
            return swap

        # Map each calendar day -> index of its last M1 bar (single pass).
        dates = m1_index.normalize()
        is_last = np.empty(len(dates), dtype=bool)
        is_last[:-1] = dates[:-1].values != dates[1:].values
        is_last[-1] = True
        last_of_day = {}
        for pos in np.flatnonzero(is_last):
            last_of_day[dates[pos]] = pos

        # MT5 SYMBOL_SWAP_ROLLOVER3DAYS uses 0=Sunday..6=Saturday; Python's
        # weekday() uses 0=Monday..6=Sunday. Convert: 3x day in Python terms
        # is (swap_rollover3days - 1) % 7. Negative means "no 3x rollover".
        rollover_py = (swap_rollover3days - 1) % 7 if swap_rollover3days >= 0 else -2

        for _, tr in trades.iterrows():
            entry_idx = int(tr["entry_idx"])
            exit_idx = int(tr["exit_idx"])
            size = float(tr["size"])
            direction = int(tr["direction"])  # 0=long, 1=short
            rate = swap_long if direction == 0 else swap_short

            entry_time = m1_index[entry_idx]
            exit_time = m1_index[exit_idx]
            days_held = (exit_time.date() - entry_time.date()).days
            if days_held <= 0:
                continue

            # Days charged: entry_date .. exit_date-1 (rollover at end of day).
            for d in range(days_held):
                day = entry_time.date() + pd.Timedelta(days=d)
                mult = 3 if day.weekday() == rollover_py else 1
                last_pos = last_of_day.get(pd.Timestamp(day))
                if last_pos is not None:
                    swap.iloc[last_pos] += rate * size * mult
        return swap

    def _empty_result(self, start, end) -> BacktestResult:
        """Return a zeroed result when there is no M1 data."""
        metrics = {
            "total_return_pct": 0.0,
            "cagr": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "n_trades": 0,
            "avg_trade_pct": 0.0,
            "exposure_pct": 0.0,
            "final_equity": self.initial_capital,
            "start_date": str(pd.Timestamp(start).date()),
            "end_date": str(pd.Timestamp(end).date()),
        }
        return BacktestResult(
            metrics=metrics,
            equity_curve=pd.Series(dtype=float),
            trades=pd.DataFrame(),
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy_name=self.strategy_name,
        )