"""Backtesting engine: vectorbt-based, HTF signals mapped to M1 execution.

Design (from grilling session):
- Thin wrapper around vectorbt Portfolio.from_signals()
- Entries/exits fill at the M1 bar open at/after the HTF actionable time
  (the first tick of the target M1 bar, matching the MQL5 engine)
- Long entries fill at ASK (open + spread); exits fill at BID (open).
  The spread is charged as a fixed fee on entry using the per-bar Spread
  column from the data (fixed spread, MQL5 export format).
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

# HTF timeframe -> pandas offset. The HTF bar timestamp is the bar's OPEN
# time; a signal computed from that bar's close is only known one period
# later. Used to shift signal timestamps before mapping to M1 fills.
_HTF_OFFSET = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1ME",
}


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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, signals: StrategySignals, htf_df: pd.DataFrame) -> BacktestResult:
        """Backtest the given signals on M1 data.

        Args:
            signals: StrategySignals indexed by the strategy's HTF DateTime.
            htf_df: The OHLCV DataFrame that was passed to strategy.generate().

        Returns:
            BacktestResult with metrics, daily equity curve, and trades.
        """
        # 1. Validate signals against the HTF df.
        signals.validate(htf_df)

        # 2. Pull M1 data for the signal range (plus a buffer for fills).
        start = htf_df.index[0]
        end = htf_df.index[-1] + pd.Timedelta(days=1)
        m1 = self._dm.load(self.symbol, "M1", start=start, end=end)
        if m1.empty:
            return self._empty_result(start, end)

        # 3. Load symbol metadata.
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
        spread_points = self._spread_series(m1, info)

        # 4. Map HTF signals to M1.
        m1_entries, m1_exits, m1_s_entries, m1_s_exits = self._map_signals(
            signals, m1.index
        )

        # 5. Resolve SL: absolute prices for vectorbt + distances for sizing.
        m1_sl_abs, m1_sl_dist = self._resolve_sl(
            signals,
            m1_entries,
            m1_s_entries,
            m1["Open"],
            spread_points,
            tick_size,
            m1.index,
        )
        m1_tp = self._map_stops(signals.tp_stop, m1.index)

        # 6. Compute per-entry lot sizes (fixed risk money per trade).
        m1_size = self._compute_sizes(
            m1_entries,
            m1_s_entries,
            m1_sl_dist,
            tick_value,
            tick_size,
            volume_step,
            volume_min,
            volume_max,
        )

        # 7. Scale prices so vectorbt PnL = real forex PnL in deposit currency.
        scale = tick_value / tick_size
        close = m1["Close"] * scale
        open_ = m1["Open"] * scale
        high = m1["High"] * scale
        low = m1["Low"] * scale
        sl_scaled = m1_sl_abs * scale
        tp_scaled = m1_tp * scale

        # Spread as a fixed fee on entry (buy at ask = bid + spread).
        # fee = spread_points * tick_value * size (tick_size cancels).
        fixed_fees = pd.Series(0.0, index=m1.index)
        entry_bars = m1_entries | m1_s_entries
        fixed_fees.loc[entry_bars] = (
            spread_points.loc[entry_bars] * tick_value * m1_size.loc[entry_bars]
        )

        # 8. Run vectorbt. init_cash is a huge ceiling so vectorbt never
        #    reduces the position size to fit cash (MQL5 sizes purely from
        #    risk money). The equity curve is shifted back to start at
        #    initial_capital below.
        import vectorbt as vb

        pf = vb.Portfolio.from_signals(
            close=close,
            entries=m1_entries,
            exits=m1_exits,
            short_entries=m1_s_entries,
            short_exits=m1_s_exits,
            size=m1_size,
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
            freq="1min",
        )

        # 9. Post-process swap and build equity curve.
        trades = pf.trades.records
        m1_equity = pf.value()
        swap_series = self._compute_swap(
            trades, m1.index, swap_long, swap_short, swap_rollover3days
        )
        # Shift the curve from the cash ceiling back to initial_capital.
        m1_equity_adj = (
            m1_equity - (_CASH_CEILING - self.initial_capital)
            - swap_series.cumsum()
        )
        daily_equity = m1_equity_adj.resample("D").last().dropna()

        # 10. Compute metrics.
        metrics = self._compute_metrics(
            daily_equity, trades, m1.index, m1_equity_adj
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=daily_equity,
            trades=trades,
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

        metrics = self._compute_metrics(
            daily_equity, trades, htf_df.index, htf_equity_adj
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
    # HTF -> M1 mapping
    # ------------------------------------------------------------------ #
    def _map_signals(
        self, signals: StrategySignals, m1_index: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Map HTF entry/exit signals to the M1 bar at/after each actionable time.

        The HTF bar timestamp is the bar's OPEN time. A signal at HTF time T
        is computed from that bar's close, known at T + htf_period. The fill
        is the first M1 bar at or after T + htf_period — the first tick of
        the target M1 bar, matching the MQL5 engine's fill timing.
        """
        period = pd.Timedelta(_HTF_OFFSET[self.timeframe])
        n = len(m1_index)
        entries = pd.Series(False, index=m1_index)
        exits = pd.Series(False, index=m1_index)
        s_entries = pd.Series(False, index=m1_index)
        s_exits = pd.Series(False, index=m1_index)

        for name, target in (
            ("entries", entries),
            ("exits", exits),
            ("short_entries", s_entries),
            ("short_exits", s_exits),
        ):
            sig_series = getattr(signals, name)
            for t in sig_series[sig_series].index:
                actionable = t + period
                pos = m1_index.searchsorted(actionable, side="left")
                if pos < n:
                    target.iloc[pos] = True
        return entries, exits, s_entries, s_exits

    @staticmethod
    def _map_stops(
        htf_stops: pd.Series, m1_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Map HTF SL/TP to M1 via forward-fill.

        Each M1 bar gets the stop value of the most recent HTF bar whose
        close has already happened. Since positions open strictly after the
        HTF entry bar, there is no lookahead.
        """
        return htf_stops.reindex(m1_index, method="ffill")

    # ------------------------------------------------------------------ #
    # SL resolution (absolute prices vs distances)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _spread_series(df: pd.DataFrame, info: dict) -> pd.Series:
        """Per-bar spread in points from the data; fall back to symbol_info."""
        if "Spread" in df.columns:
            return df["Spread"].astype(float)
        return pd.Series(float(info["spread"]), index=df.index)

    def _resolve_sl(
        self,
        signals: StrategySignals,
        m1_entries: pd.Series,
        m1_s_entries: pd.Series,
        m1_open: pd.Series,
        spread_points: pd.Series,
        tick_size: float,
        m1_index: pd.DatetimeIndex,
    ) -> tuple[pd.Series, pd.Series]:
        """Resolve SL into (absolute prices for vectorbt, distances for sizing).

        Distance-based SL (sl_is_distance=True) is converted to an absolute
        stop relative to the entry fill price:
          - Long:  fill at ASK = open + spread; SL = ask - distance
          - Short: fill at BID = open;          SL = bid + distance
        This matches the MQL5 engine's SL placement (sl = ask - sl_atr*ATR).
        """
        if signals.sl_is_distance:
            m1_sl_dist = self._map_stops(signals.sl_stop, m1_index)
            spread_price = spread_points * tick_size
            m1_sl_abs = pd.Series(np.nan, index=m1_index)

            long_bars = m1_entries & m1_sl_dist.notna()
            m1_sl_abs.loc[long_bars] = (
                m1_open.loc[long_bars]
                + spread_price.loc[long_bars]
                - m1_sl_dist.loc[long_bars]
            )
            short_bars = m1_s_entries & m1_sl_dist.notna()
            m1_sl_abs.loc[short_bars] = (
                m1_open.loc[short_bars] + m1_sl_dist.loc[short_bars]
            )
            return m1_sl_abs.ffill(), m1_sl_dist

        m1_sl_abs = self._map_stops(signals.sl_stop, m1_index)
        m1_sl_dist = pd.Series(np.nan, index=m1_index)
        long_bars = m1_entries & m1_sl_abs.notna()
        m1_sl_dist.loc[long_bars] = m1_open.loc[long_bars] - m1_sl_abs.loc[long_bars]
        short_bars = m1_s_entries & m1_sl_abs.notna()
        m1_sl_dist.loc[short_bars] = m1_sl_abs.loc[short_bars] - m1_open.loc[short_bars]
        return m1_sl_abs, m1_sl_dist

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
        """
        swap = pd.Series(0.0, index=m1_index)
        if trades.empty:
            return swap

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
                mult = 3 if day.weekday() == swap_rollover3days else 1
                # Place the charge at the last M1 bar of that day.
                day_mask = m1_index.date == day
                if day_mask.any():
                    last_pos = int(np.where(day_mask)[0][-1])
                    swap.iloc[last_pos] += rate * size * mult
        return swap

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _compute_metrics(
        self,
        daily_equity: pd.Series,
        trades: pd.DataFrame,
        m1_index: pd.DatetimeIndex,
        m1_equity: pd.Series,
    ) -> dict[str, float | str]:
        final_equity = float(daily_equity.iloc[-1]) if len(daily_equity) else self.initial_capital
        start_date = daily_equity.index[0] if len(daily_equity) else m1_index[0]
        end_date = daily_equity.index[-1] if len(daily_equity) else m1_index[-1]

        total_return_pct = (final_equity / self.initial_capital - 1) * 100

        years = (end_date - start_date).days / 365.25
        cagr = (
            (final_equity / self.initial_capital) ** (1 / years) - 1
            if years > 0 and final_equity > 0
            else 0.0
        )

        # Max drawdown from daily equity.
        running_max = daily_equity.cummax()
        drawdown = (daily_equity - running_max) / running_max
        max_drawdown_pct = float(drawdown.min() * 100) if len(drawdown) else 0.0

        # Sharpe / Sortino from daily returns.
        daily_returns = daily_equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = float(daily_returns.mean() / downside.std() * np.sqrt(252))
        else:
            sortino = 0.0

        # Trade-based metrics.
        closed = trades[trades["status"] == 1] if not trades.empty else trades
        n_trades = int(len(closed))
        if n_trades > 0:
            pnls = closed["pnl"].astype(float)
            win_rate = float((pnls > 0).mean() * 100)
            gross_profit = float(pnls[pnls > 0].sum())
            gross_loss = float(-pnls[pnls < 0].sum())
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )
            avg_trade_pct = float(closed["return"].astype(float).mean() * 100)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_pct = 0.0

        # Exposure: fraction of M1 bars in a position.
        if not trades.empty and len(m1_index) > 0:
            held_bars = sum(
                int(tr["exit_idx"]) - int(tr["entry_idx"]) + 1
                for _, tr in trades.iterrows()
            )
            exposure_pct = float(held_bars / len(m1_index) * 100)
        else:
            exposure_pct = 0.0

        return {
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe": sharpe,
            "sortino": sortino,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "avg_trade_pct": avg_trade_pct,
            "exposure_pct": exposure_pct,
            "final_equity": final_equity,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
        }

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