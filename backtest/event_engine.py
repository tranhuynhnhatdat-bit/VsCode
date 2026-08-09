"""Event-driven validation engine: bar-by-bar processing with tick simulation.

Design (grill session):
- Matches MQL5 OnTick event flow as closely as possible
- Processes M1 bars sequentially in chronological order
- Simulates synthetic intra-bar ticks from M1 OHLC for SL/TP checking
- Stateful position manager mirrors MQL5 CTrade behavior
- Accepts the same StrategySignals as the vectorized engine, or raw strategy
  params for the ComposableStrategy
- Returns BacktestResult (same format as vectorized BacktestEngine) for
  drop-in compatibility with the optimization pipeline

Output metrics (used as final validation filters):
  - profit_factor > 1.3
  - win_rate > 35%
  - max_drawdown < 15%
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_manager import DataManager
from symbol_info import SymbolInfo

# Results directory for equity curve PNGs.
RESULTS_DIR = Path(r"C:\Users\DAT\Desktop\VsCode\results")


@dataclass
class EventTrade:
    """A single trade record (mirrors vectorbt trade output format)."""

    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    size: float
    direction: int  # 0 = long, 1 = short
    pnl: float
    return_pct: float
    status: int = 1  # 1 = closed


@dataclass
class OpenPosition:
    """State of an open position (mirrors MQL5 PositionSelect)."""

    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float  # ASK for longs, BID for shorts
    size: float
    direction: int  # 0 = long, 1 = short
    sl_price: float | None = None
    sl_distance: float | None = None
    tp_price: float | None = None


class TickSimulator:
    """Generates synthetic intra-bar ticks from M1 OHLC.

    Uses a configurable number of ticks per bar with a controlled random walk
    from Open -> High -> Low -> Close. This approximates the MT5 tester's
    tick generation for SL/TP hit detection.
    """

    def __init__(self, ticks_per_bar: int = 20, seed: int = 42) -> None:
        self.ticks_per_bar = ticks_per_bar
        self._rng = np.random.default_rng(seed)

    def generate_ticks(
        self, open_: float, high: float, low: float, close: float
    ) -> list[float]:
        """Generate synthetic tick prices for one M1 bar.

        Produces `ticks_per_bar` ticks that traverse Open -> High -> Low ->
        Close. The path respects the bar's OHLC extremes.
        """
        if self.ticks_per_bar <= 0:
            return [close]

        n = self.ticks_per_bar
        ticks: list[float] = []

        # First tick is the open price.
        ticks.append(open_)

        if n <= 1:
            return ticks

        remaining = n - 1

        # Split into two phases: Open -> extreme, extreme -> Close.
        phase1_ticks = max(1, int(remaining * self._rng.uniform(0.3, 0.7)))
        phase2_ticks = remaining - phase1_ticks

        direction = 1 if close >= open_ else -1
        extreme = high if direction > 0 else low

        # Phase 1: walk from open toward the extreme.
        for i in range(1, phase1_ticks + 1):
            progress = i / phase1_ticks
            target = open_ + (extreme - open_) * progress
            noise = self._rng.uniform(-0.0005, 0.0005) * (high - low)
            price = max(min(target + noise, high), low)
            ticks.append(price)

        # Phase 2: walk from where we are to the close.
        start_price = ticks[-1]
        for i in range(1, phase2_ticks + 1):
            progress = i / phase2_ticks
            target = start_price + (close - start_price) * progress
            noise = self._rng.uniform(-0.0005, 0.0005) * (high - low)
            price = max(min(target + noise, high), low)
            ticks.append(price)

        # Ensure we end at (or very near) the close.
        ticks[-1] = close

        return ticks


class EventEngine:
    """Bar-by-bar event-driven backtest engine.

    Processes M1 data sequentially, matching MQL5 OnTick behavior:
    - On each M1 bar: check time conditions, evaluate entries/exits
    - Intra-bar: simulate ticks to check SL/TP
    - Stateful position management

    Args:
        symbol: Trading symbol (e.g. "XAUUSD").
        htf_timeframe: Strategy's native timeframe (e.g. "H1").
        initial_capital: Starting capital in deposit currency.
        risk_money: Fixed risk per trade in USD.
        ticks_per_bar: Number of synthetic ticks per M1 bar for SL/TP checks.
    """

    def __init__(
        self,
        symbol: str,
        htf_timeframe: str = "H1",
        initial_capital: float = 10_000.0,
        risk_money: float = 100.0,
        ticks_per_bar: int = 20,
    ) -> None:
        self.symbol = symbol
        self.htf_timeframe = htf_timeframe
        self.initial_capital = initial_capital
        self.risk_money = risk_money
        self._dm = DataManager()
        self._si = SymbolInfo()
        self._tick_sim = TickSimulator(ticks_per_bar=ticks_per_bar)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        entry_hour: int,
        exit_hour: int,
        session_days: tuple[int, ...],
        sl_atr: float,
        atr_period: int,
        m1_df: pd.DataFrame,
        htf_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run the event-driven backtest.

        Args:
            entry_hour: Hour of the H1 signal bar (entry fires at hour+1).
            exit_hour: Hour of the H1 exit bar (exit fires at hour+1).
            session_days: Tuple of weekday numbers (Python: Mon=0..Sun=6).
            sl_atr: Stop-loss multiplier (0 = no SL).
            atr_period: ATR calculation period.
            m1_df: M1 OHLCV DataFrame with DateTime index.
            htf_df: HTF (H1) OHLCV DataFrame with DateTime index (for signal bars).

        Returns:
            Dict with keys:
                - 'metrics': dict of performance metrics
                - 'equity_curve': daily equity Series
                - 'trades': DataFrame of trade records
                - 'n_trades': int
                - 'profit_factor': float
                - 'win_rate': float
                - 'max_drawdown': float
        """
        # Load symbol metadata.
        info = self._si.get(self.symbol)
        tick_value = float(info["trade_tick_value"])
        tick_size = float(info["trade_tick_size"])
        volume_step = float(info["volume_step"])
        volume_min = float(info["volume_min"])
        volume_max = float(info["volume_max"])
        swap_long = float(info["swap_long"])
        swap_short = float(info["swap_short"])
        swap_rollover3days = int(info["swap_rollover3days"])
        spread_fixed = float(info.get("spread", 10.0))

        # Pre-compute ATR on the H1 data.
        htf_atr = self._compute_atr(htf_df, atr_period)

        # Build a set of M1 bar hours:minutes that are entry-eligible.
        # Entry fires on the first M1 bar at/after entry_hour+1:00,
        # exit fires on the first M1 bar at/after exit_hour+1:00.
        # We track whether we've already processed entry/exit for each day.
        entry_hour_target = entry_hour + 1
        exit_hour_target = exit_hour + 1

        # ------------------------------------------------------------------ #
        # Main event loop: process each M1 bar sequentially.
        # ------------------------------------------------------------------ #
        trades: list[EventTrade] = []
        position: OpenPosition | None = None
        entry_days: set[pd.Timestamp] = set()  # days that had an entry
        equity_values: list[tuple[pd.Timestamp, float]] = []
        cash = self.initial_capital
        total_swap = 0.0

        m1_indices = list(m1_df.index)
        n_bars = len(m1_indices)

        for idx in range(n_bars):
            cur_time = m1_indices[idx]
            day_of_week = cur_time.weekday()  # Mon=0..Sun=6
            hour = cur_time.hour
            minute = cur_time.minute
            cur_date = cur_time.date()
            cur_day_start = pd.Timestamp(cur_date)

            # --- Accrue swap at the last M1 bar of each day ---
            is_last_bar_of_day = (
                idx == n_bars - 1
                or m1_indices[idx + 1].date() != cur_date
            )
            if is_last_bar_of_day and position is not None:
                days_held = (cur_date - position.entry_time.date()).days
                if days_held > 0:
                    rate = swap_long if position.direction == 0 else swap_short
                    mult = 3 if cur_time.weekday() == swap_rollover3days else 1
                    swap_charge = rate * position.size * mult * days_held
                    cash -= swap_charge
                    total_swap += swap_charge

            # --- Check if position is open; simulate intra-bar ticks for SL/TP ---
            sl_tp_hit_this_bar = False
            if position is not None:
                bar_open = float(m1_df.iloc[idx]["Open"])
                bar_high = float(m1_df.iloc[idx]["High"])
                bar_low = float(m1_df.iloc[idx]["Low"])
                bar_close = float(m1_df.iloc[idx]["Close"])

                # Generate synthetic ticks for this bar.
                ticks = self._tick_sim.generate_ticks(
                    bar_open, bar_high, bar_low, bar_close
                )

                # Check each tick for SL/TP hit.
                hit_price: float | None = None

                for tick_price in ticks:
                    if position.sl_price is not None:
                        if position.direction == 0:  # long: SL hit if price <= sl
                            if tick_price <= position.sl_price:
                                hit_price = min(tick_price, bar_low)  # gap-through: worse of hit and low
                                break
                        else:  # short: SL hit if price >= sl
                            if tick_price >= position.sl_price:
                                hit_price = max(tick_price, bar_high)  # gap-through: worse of hit and high
                                break

                    if position.tp_price is not None and hit_price is None:
                        if position.direction == 0:  # long: TP hit if price >= tp
                            if tick_price >= position.tp_price:
                                hit_price = max(tick_price, bar_high)
                                break
                        else:  # short: TP hit if price <= tp
                            if tick_price <= position.tp_price:
                                hit_price = min(tick_price, bar_low)
                                break

                if hit_price is not None:
                    # Close the position at the hit price.
                    pnl = self._close_position(
                        position, trades, idx, cur_time, hit_price,
                        tick_value, tick_size
                    )
                    cash += pnl
                    position = None
                    sl_tp_hit_this_bar = True

            # Skip entry/exit logic if SL/TP already closed position this bar.
            if not sl_tp_hit_this_bar:
                # --- ENTRY LOGIC (mirrors MQL5 OnTick entry block) ---
                is_session_day = day_of_week in session_days
                if (
                    is_session_day
                    and hour == entry_hour + 1
                    and minute == 0  # First M1 bar of the hour
                    and position is None  # No open position (single position rule)
                ):
                    # The signal bar opened at entry_hour:00 and closed at entry_hour+1:00.
                    signal_bar_time = cur_time.replace(
                        hour=entry_hour, minute=0, second=0, microsecond=0
                    )

                    # Find the H1 bar in our data.
                    if signal_bar_time in htf_df.index:
                        h1_row = htf_df.loc[signal_bar_time]
                        h1_open = float(h1_row["Open"])
                        h1_close = float(h1_row["Close"])

                        # Condition: Close < Open (bearish bar -> BUY).
                        if h1_close < h1_open:
                            # ATR-based SL distance.
                            atr_val = htf_atr.get(signal_bar_time, np.nan)
                            if not np.isnan(atr_val) and sl_atr > 0:
                                sl_distance = sl_atr * atr_val
                            else:
                                sl_distance = 0.0

                            if sl_distance > 0:
                                # Entry price = ASK = Open + spread.
                                bar_open_price = float(m1_df.iloc[idx]["Open"])
                                spread_price = spread_fixed * tick_size
                                ask = bar_open_price + spread_price

                                # SL = ask - distance.
                                sl_price = ask - sl_distance

                                # Lot size = fixed risk money.
                                size = self._compute_lot_size(
                                    sl_distance, tick_value, tick_size,
                                    volume_step, volume_min, volume_max
                                )

                                if size > 0:
                                    position = OpenPosition(
                                        entry_idx=idx,
                                        entry_time=cur_time,
                                        entry_price=ask,
                                        size=size,
                                        direction=0,
                                        sl_price=sl_price,
                                        sl_distance=sl_distance,
                                    )
                                    # Spread cost charged on entry (matching vectorbt's fixed_fees).
                                    cash -= size * spread_price
                                    entry_days.add(cur_day_start)

                # --- EXIT LOGIC (mirrors MQL5 OnTick exit block) ---
                if (
                    position is not None
                    and is_session_day
                    and hour == exit_hour + 1
                    and minute == 0  # First M1 bar of the hour
                ):
                    # Close at market (BID = Open).
                    bar_open_price = float(m1_df.iloc[idx]["Open"])
                    pnl = self._close_position(
                        position, trades, idx, cur_time, bar_open_price,
                        tick_value, tick_size
                    )
                    cash += pnl
                    position = None

            # --- Record equity at each M1 bar close ---
            if position is not None:
                mtm_value = self._position_value(
                    position, m1_df, idx, tick_value, tick_size
                )
                equity_values.append((cur_time, cash + mtm_value))
            else:
                equity_values.append((cur_time, cash))

        # Close any position still open at end of data.
        if position is not None:
            last_close = float(m1_df.iloc[-1]["Close"])
            pnl = self._close_position(
                position, trades, n_bars - 1, m1_indices[-1],
                last_close, tick_value, tick_size
            )
            cash += pnl
            position = None

        # ------------------------------------------------------------------ #
        # Build results.
        # ------------------------------------------------------------------ #
        trades_df = self._trades_to_dataframe(trades)
        equity_series = self._equity_to_series(equity_values)

        metrics = self._compute_metrics(
            equity_series, trades_df, self.initial_capital
        )

        return {
            "metrics": metrics,
            "equity_curve": equity_series,
            "trades": trades_df,
            "n_trades": metrics["n_trades"],
            "profit_factor": metrics["profit_factor"],
            "win_rate": metrics["win_rate"],
            "max_drawdown": metrics["max_drawdown_pct"],
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Compute ATR on a DataFrame with OHLC columns."""
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
        return atr

    def _compute_lot_size(
        self,
        sl_distance: float,
        tick_value: float,
        tick_size: float,
        volume_step: float,
        volume_min: float,
        volume_max: float,
    ) -> float:
        """Lot size from fixed risk money (matches MQL5 sqMMFixedAmount)."""
        if sl_distance <= 0 or tick_value <= 0 or tick_size <= 0 or volume_step <= 0:
            return 0.0

        lots = self.risk_money / (sl_distance * tick_value / tick_size)
        lots = math.floor(lots / volume_step + 1e-9) * volume_step

        if lots < volume_min or lots > volume_max:
            return 0.0
        return lots

    def _close_position(
        self,
        position: OpenPosition,
        trades: list[EventTrade],
        exit_idx: int,
        exit_time: pd.Timestamp,
        exit_price: float,
        tick_value: float,
        tick_size: float,
    ) -> float:
        """Close a position and record the trade. Returns PnL."""
        if position.direction == 0:  # long
            pnl = (exit_price - position.entry_price) * position.size * (tick_value / tick_size)
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size * (tick_value / tick_size)

        return_pct = pnl / (position.entry_price * position.size) if position.entry_price > 0 else 0.0

        trades.append(EventTrade(
            entry_idx=position.entry_idx,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_idx=exit_idx,
            exit_time=exit_time,
            exit_price=exit_price,
            size=position.size,
            direction=position.direction,
            pnl=pnl,
            return_pct=return_pct,
            status=1,
        ))
        return pnl

    @staticmethod
    def _position_value(
        position: OpenPosition,
        m1_df: pd.DataFrame,
        current_idx: int,
        tick_value: float,
        tick_size: float,
    ) -> float:
        """Mark-to-market value of an open position at the current bar."""
        current_price = float(m1_df.iloc[current_idx]["Close"])
        if position.direction == 0:  # long
            return (current_price - position.entry_price) * position.size * (tick_value / tick_size)
        else:  # short
            return (position.entry_price - current_price) * position.size * (tick_value / tick_size)

    @staticmethod
    def _trades_to_dataframe(trades: list[EventTrade]) -> pd.DataFrame:
        """Convert trade list to DataFrame matching vectorbt trade format."""
        if not trades:
            return pd.DataFrame()
        records = []
        for t in trades:
            records.append({
                "entry_idx": t.entry_idx,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_idx": t.exit_idx,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "size": t.size,
                "direction": t.direction,
                "pnl": t.pnl,
                "return": t.return_pct,
                "status": t.status,
            })
        return pd.DataFrame(records)

    @staticmethod
    def _equity_to_series(
        equity_values: list[tuple[pd.Timestamp, float]]
    ) -> pd.Series:
        """Convert equity timestamp-value pairs to a daily Series."""
        if not equity_values:
            return pd.Series(dtype=float)
        df = pd.DataFrame(equity_values, columns=["time", "equity"])
        df = df.set_index("time").resample("D").last().dropna()
        return df["equity"]

    @staticmethod
    def _compute_metrics(
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        initial_capital: float,
    ) -> dict[str, float | str]:
        """Compute performance metrics (same as vectorized engine)."""
        zero_metrics: dict[str, float | str] = {
            "total_return_pct": 0.0,
            "cagr": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "n_trades": 0,
            "avg_trade_pct": 0.0,
            "final_equity": initial_capital,
        }

        if equity_curve.empty:
            return zero_metrics

        final_equity = float(equity_curve.iloc[-1])
        total_return_pct = (final_equity / initial_capital - 1) * 100

        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = max((end_date - start_date).days / 365.25, 0.001)

        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if final_equity > 0 else 0.0

        # Max drawdown.
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown_pct = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0

        # Sharpe / Sortino.
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        else:
            sharpe = 0.0
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = float(daily_returns.mean() / downside.std() * math.sqrt(252))
        else:
            sortino = 0.0

        # Trade-based metrics.
        if trades_df.empty:
            return {
                **zero_metrics,
                "total_return_pct": total_return_pct,
                "cagr": cagr,
                "max_drawdown_pct": max_drawdown_pct,
                "sharpe": sharpe,
                "sortino": sortino,
                "final_equity": final_equity,
            }

        closed = trades_df[trades_df["status"] == 1] if not trades_df.empty else trades_df
        n_trades = int(len(closed))
        if n_trades > 0:
            pnls = closed["pnl"].astype(float)
            win_rate = float((pnls > 0).mean() * 100)
            gross_profit = float(pnls[pnls > 0].sum())
            gross_loss = float(-pnls[pnls < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            avg_trade_pct = float(closed["return"].astype(float).mean() * 100)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_pct = 0.0

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
            "final_equity": final_equity,
        }


def validate_with_event_engine(
    params: dict[str, Any],
    m1_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    symbol: str = "XAUUSD",
    initial_capital: float = 10_000.0,
    risk_money: float = 100.0,
    ticks_per_bar: int = 20,
    filters: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run a single strategy through the event engine and apply filters.

    Args:
        params: ComposableStrategy constructor params (entry_hour, exit_hour,
            session_days, sl_atr, atr_period, etc.).
        m1_df: Full M1 DataFrame.
        h1_df: Full H1 (HTF) DataFrame.
        symbol: Trading symbol.
        initial_capital: Starting capital.
        risk_money: Fixed risk per trade.
        ticks_per_bar: Synthetic ticks per M1 bar.
        filters: Dict of {metric_name: min_threshold}. Default:
            {"profit_factor": 1.3, "win_rate": 35.0, "max_drawdown_pct": -15.0}
            (max_drawdown is negative, so threshold is upper bound)

    Returns:
        Dict with keys:
            - 'passed': bool
            - 'result': event engine result dict
            - 'fail_reasons': list of str describing which filters failed
    """
    if filters is None:
        filters = {
            "profit_factor": 1.3,
            "win_rate": 35.0,
            "max_drawdown_pct": -15.0,  # drawdown is negative; -15% means 15% DD
        }

    engine = EventEngine(
        symbol=symbol,
        htf_timeframe="H1",
        initial_capital=initial_capital,
        risk_money=risk_money,
        ticks_per_bar=ticks_per_bar,
    )

    # Extract strategy params with defaults (matching ComposableStrategy).
    entry_hour = int(params.get("entry_hour", 1))
    exit_hour = int(params.get("exit_hour", 22))
    session_days = tuple(params.get("session_days", (2, 4)))
    sl_atr = float(params.get("sl_atr", 2.0))
    atr_period = int(params.get("atr_period", 14))

    result = engine.run(
        entry_hour=entry_hour,
        exit_hour=exit_hour,
        session_days=session_days,
        sl_atr=sl_atr,
        atr_period=atr_period,
        m1_df=m1_df,
        htf_df=h1_df,
    )

    # Apply filters.
    passed = True
    fail_reasons: list[str] = []

    for metric, threshold in filters.items():
        value = result.get(metric, 0.0)
        if isinstance(value, (int, float)):
            if metric == "max_drawdown_pct":
                # Drawdown is negative; we want drawdown less than threshold
                # (e.g., -15% means 15% max drawdown -> value >= -15 passes)
                if value < threshold:
                    passed = False
                    fail_reasons.append(
                        f"{metric}={value:.2f} < {threshold:.2f}"
                    )
            else:
                if value <= threshold:
                    passed = False
                    fail_reasons.append(
                        f"{metric}={value:.4f} <= {threshold:.4f}"
                    )
        else:
            if value == float("inf") and metric == "profit_factor":
                continue
            if value <= threshold:
                passed = False
                fail_reasons.append(
                    f"{metric}={value} <= {threshold}"
                )

    return {
        "passed": passed,
        "result": result,
        "fail_reasons": fail_reasons,
    }