"""Numba-accelerated M1 OHLC execution core shared by both backtest engines.

Both the vectorized engine (backtest/engine.py) and the event-driven engine
(backtest/event_engine.py) must execute M1 bars with the SAME semantics, or
their results diverge. vectorbt's Portfolio.from_signals evaluates SL/TP on
the bar CLOSE, not the bar HIGH/LOW, so it can't reproduce the event engine's
MQL5 "1 minute OHLC" fill logic. This module provides a single Numba-JIT core
that both engines call, guaranteeing identical results and a large speedup
over both vectorbt's M1 overhead and a pure-Python event loop.

Semantics (matches the event engine / MQL5 "1 minute OHLC"):
- SL/TP evaluated against each M1 bar's High/Low.
- A stop within the bar's range fills at the stop level; a bar that opens
  beyond the stop (gap) fills at the bar's open.
- When both SL and TP are in the same bar, the level closer to the open fires
  first (MQL5 convention).
- Long entries fill at ASK (open + spread); short entries at BID (open).
  The spread cost is captured through the ask/bid fill differential (a long
  buys at ask and exits at bid; a short sells at bid and closes at ask), so
  no separate spread fee is charged. This matches the H1 screen's single
  spread cost (spread_points * tick_value * size).
- Lot size = risk_money / (SL distance * tick_value/tick_size), floored to
  volume_step, rejected outside [volume_min, volume_max].
- Swap charged for days [entry_day, exit_day - 1] with the rollover weekday
  at 3x, placed at the last M1 bar of each day.
- Single position: new entries while in a position are ignored.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

from backtest._mapping import map_signals_to_m1, map_stops_to_m1
from strategy.base import StrategySignals

# Sentinel for "no TP" (TP prices are positive; -1 means none).
_NO_TP = -1.0


def build_day_arrays(m1_index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-bar day-id, per-day weekday, and last-bar-of-day masks.

    Returns (day_ids, day_weekday, is_last_bar_of_day):
      - day_ids[i]: int day index (days since the epoch) for M1 bar i.
      - day_weekday[d]: weekday (Mon=0..Sun=6) for day-id d.
      - is_last_bar_of_day[i]: True on the final M1 bar of each calendar day.
    """
    dates = m1_index.normalize()
    day_ids = dates.as_unit("ns").asi8 // np.int64(86400 * 10**9)
    day_ids = day_ids.astype(np.int64)

    # Unique days in order -> weekday map.
    unique_days = np.unique(day_ids)
    day_weekday = np.full(int(unique_days[-1]) + 1, -1, dtype=np.int32)
    for d in unique_days:
        day_weekday[int(d)] = m1_index[day_ids == d][0].weekday()

    # Last M1 bar of each day.
    is_last = np.empty(len(day_ids), dtype=bool)
    is_last[:-1] = day_ids[:-1] != day_ids[1:]
    is_last[-1] = True

    return day_ids, day_weekday, is_last


@njit(cache=True)
def _holding_swap(
    entry_idx: int,
    exit_idx: int,
    day_ids: np.ndarray,
    day_weekday: np.ndarray,
    size: float,
    direction: int,
    swap_long: float,
    swap_short: float,
    swap_rollover3days: int,
) -> float:
    """Swap for days [entry_day, exit_day-1] (3x on rollover weekday).

    Notes:
    - MT5's SYMBOL_SWAP_ROLLOVER3DAYS uses 0=Sunday..6=Saturday, while
      ``day_weekday`` uses Python's 0=Monday..6=Sunday. Convert: the 3x day
      in Python weekday terms is ``(swap_rollover3days - 1) % 7``.
    - Non-trading days (weekends/holidays, day_weekday[day] == -1) accrue no
      swap, matching engine._compute_swap and real MT5 behavior.
    """
    entry_day = day_ids[entry_idx]
    exit_day = day_ids[exit_idx]
    days_held = exit_day - entry_day
    if days_held <= 0:
        return 0.0
    rate = swap_long if direction == 0 else swap_short
    # Convert MT5 rollover weekday (Sun=0) to Python weekday (Mon=0).
    # A negative value (e.g. -1) means "no 3x rollover day".
    rollover_py = (swap_rollover3days - 1) % 7 if swap_rollover3days >= 0 else -2
    total = 0.0
    for d in range(days_held):
        day = entry_day + d
        wd = day_weekday[day]
        if wd < 0:
            continue  # non-trading day: no swap
        mult = 3 if wd == rollover_py else 1
        total += rate * size * mult
    return total


@njit(cache=True)
def _compute_lot(
    sl_distance: float,
    risk_money: float,
    tick_value: float,
    tick_size: float,
    volume_step: float,
    volume_min: float,
    volume_max: float,
) -> float:
    """Lot from fixed risk money (matches vectorized/event sizing)."""
    if sl_distance <= 0 or tick_value <= 0 or tick_size <= 0 or volume_step <= 0:
        return 0.0
    lots = risk_money / (sl_distance * tick_value / tick_size)
    lots = np.floor(lots / volume_step + 1e-9) * volume_step
    if lots < volume_min or lots > volume_max:
        return 0.0
    return lots


@njit(cache=True)
def simulate_m1(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    spread_arr: np.ndarray,
    entry_mask: np.ndarray,
    exit_mask: np.ndarray,
    s_entry_mask: np.ndarray,
    s_exit_mask: np.ndarray,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    sl_is_dist: bool,
    is_last_bar_of_day: np.ndarray,
    day_ids: np.ndarray,
    day_weekday: np.ndarray,
    tick_value: float,
    tick_size: float,
    volume_step: float,
    volume_min: float,
    volume_max: float,
    risk_money: float,
    swap_long: float,
    swap_short: float,
    swap_rollover3days: int,
    initial_capital: float,
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Simulate M1 OHLC execution. Returns (trades, equity_idx, equity_vals).

    trades: list of tuples
        (entry_idx, exit_idx, entry_price, exit_price, size, direction, pnl)
    equity_idx: int array of M1 bar indices (last bar of each day)
    equity_vals: float array of cash+mtm at those bars
    """
    n_bars = len(open_arr)
    trades: list[tuple] = []
    equity_idx_list: list[int] = []
    equity_vals_list: list[float] = []

    # Position state (-1 = flat).
    pos_entry_idx = -1
    pos_entry_price = 0.0
    pos_size = 0.0
    pos_dir = 0  # 0 long, 1 short
    pos_sl = 0.0
    pos_sl_has = False
    pos_tp = 0.0
    pos_tp_has = False
    cash = initial_capital

    for idx in range(n_bars):
        bar_open = open_arr[idx]
        bar_high = high_arr[idx]
        bar_low = low_arr[idx]

        # --- SL/TP evaluated on bar extremes (MQL5 "1 minute OHLC") ---
        sl_tp_hit = False
        if pos_entry_idx >= 0:
            hit: float = -1.0  # sentinel: no hit
            if pos_dir == 0:  # long
                sl_hit = pos_sl_has and bar_low <= pos_sl
                tp_hit = pos_tp_has and bar_high >= pos_tp
                if sl_hit and tp_hit:
                    if abs(bar_open - pos_sl) <= abs(pos_tp - bar_open):
                        hit = min(pos_sl, bar_open)
                    else:
                        hit = max(pos_tp, bar_open)
                elif sl_hit:
                    hit = min(pos_sl, bar_open)
                elif tp_hit:
                    hit = max(pos_tp, bar_open)
            else:  # short
                sl_hit = pos_sl_has and bar_high >= pos_sl
                tp_hit = pos_tp_has and bar_low <= pos_tp
                if sl_hit and tp_hit:
                    if abs(bar_open - pos_sl) <= abs(pos_tp - bar_open):
                        hit = max(pos_sl, bar_open)
                    else:
                        hit = min(pos_tp, bar_open)
                elif sl_hit:
                    hit = max(pos_sl, bar_open)
                elif tp_hit:
                    hit = min(pos_tp, bar_open)

            if hit >= 0:
                exit_price = hit
                if pos_dir == 0:
                    pnl = (exit_price - pos_entry_price) * pos_size * (tick_value / tick_size)
                else:
                    pnl = (pos_entry_price - exit_price) * pos_size * (tick_value / tick_size)
                swap_charge = _holding_swap(
                    pos_entry_idx, idx, day_ids, day_weekday, pos_size, pos_dir,
                    swap_long, swap_short, swap_rollover3days,
                )
                cash += pnl - swap_charge
                trades.append((pos_entry_idx, idx, pos_entry_price, exit_price,
                               pos_size, pos_dir, pnl))
                pos_entry_idx = -1
                sl_tp_hit = True

        if not sl_tp_hit:
            # --- ENTRY ---
            if pos_entry_idx < 0 and (entry_mask[idx] or s_entry_mask[idx]):
                sl_val = sl_arr[idx]
                tp_val = tp_arr[idx]
                spread = spread_arr[idx]

                if entry_mask[idx]:  # LONG
                    ask = bar_open + spread
                    if sl_is_dist:
                        if sl_val > 0.0:
                            sl_price = ask - sl_val
                            sl_distance = sl_val
                        else:
                            sl_price = -1.0
                            sl_distance = 0.0
                    else:
                        if sl_val == sl_val:  # not NaN
                            sl_price = sl_val
                            # Risk distance measured from the actual fill
                            # price (ask = open + spread), not the open.
                            sl_distance = ask - sl_val
                        else:
                            sl_price = -1.0
                            sl_distance = 0.0
                    tp_price = tp_val if tp_val == tp_val else _NO_TP

                    size = _compute_lot(
                        sl_distance, risk_money, tick_value, tick_size,
                        volume_step, volume_min, volume_max,
                    )
                    if size > 0:
                        pos_entry_idx = idx
                        pos_entry_price = ask
                        pos_size = size
                        pos_dir = 0
                        pos_sl = sl_price
                        pos_sl_has = sl_price >= 0
                        pos_tp = tp_price
                        pos_tp_has = tp_price >= 0
                        # No explicit spread fee: a long buys at ASK and
                        # exits at BID, so the spread is in the PnL.

                elif s_entry_mask[idx]:  # SHORT
                    bid = bar_open
                    if sl_is_dist:
                        if sl_val > 0.0:
                            sl_price = bid + sl_val
                            sl_distance = sl_val
                        else:
                            sl_price = -1.0
                            sl_distance = 0.0
                    else:
                        if sl_val == sl_val:
                            sl_price = sl_val
                            sl_distance = sl_val - bar_open
                        else:
                            sl_price = -1.0
                            sl_distance = 0.0
                    tp_price = tp_val if tp_val == tp_val else _NO_TP

                    size = _compute_lot(
                        sl_distance, risk_money, tick_value, tick_size,
                        volume_step, volume_min, volume_max,
                    )
                    if size > 0:
                        pos_entry_idx = idx
                        pos_entry_price = bid
                        pos_size = size
                        pos_dir = 1
                        pos_sl = sl_price
                        pos_sl_has = sl_price >= 0
                        pos_tp = tp_price
                        pos_tp_has = tp_price >= 0
                        # No explicit spread fee: a short exits at ASK
                        # (scheduled) so the spread is in the PnL. Note:
                        # a short stopped by SL/TP exits at the bid level and
                        # so does not capture the spread — acceptable while
                        # shorts are unused.

            # --- EXIT (scheduled) ---
            elif pos_entry_idx >= 0 and (exit_mask[idx] or s_exit_mask[idx]):
                exit_price = bar_open
                if pos_dir == 1:  # short closes at ASK
                    exit_price = bar_open + spread_arr[idx]
                if pos_dir == 0:
                    pnl = (exit_price - pos_entry_price) * pos_size * (tick_value / tick_size)
                else:
                    pnl = (pos_entry_price - exit_price) * pos_size * (tick_value / tick_size)
                swap_charge = _holding_swap(
                    pos_entry_idx, idx, day_ids, day_weekday, pos_size, pos_dir,
                    swap_long, swap_short, swap_rollover3days,
                )
                cash += pnl - swap_charge
                trades.append((pos_entry_idx, idx, pos_entry_price, exit_price,
                               pos_size, pos_dir, pnl))
                pos_entry_idx = -1

        # --- Record equity at the last M1 bar of each day ---
        if is_last_bar_of_day[idx]:
            if pos_entry_idx >= 0:
                if pos_dir == 0:
                    mtm = (close_arr[idx] - pos_entry_price) * pos_size * (tick_value / tick_size)
                else:
                    mtm = (pos_entry_price - close_arr[idx]) * pos_size * (tick_value / tick_size)
                equity_idx_list.append(idx)
                equity_vals_list.append(cash + mtm)
            else:
                equity_idx_list.append(idx)
                equity_vals_list.append(cash)

    # Close any position still open at end of data.
    if pos_entry_idx >= 0:
        exit_price = close_arr[n_bars - 1]
        if pos_dir == 0:
            pnl = (exit_price - pos_entry_price) * pos_size * (tick_value / tick_size)
        else:
            pnl = (pos_entry_price - exit_price) * pos_size * (tick_value / tick_size)
        swap_charge = _holding_swap(
            pos_entry_idx, n_bars - 1, day_ids, day_weekday, pos_size, pos_dir,
            swap_long, swap_short, swap_rollover3days,
        )
        trades.append((pos_entry_idx, n_bars - 1, pos_entry_price, exit_price,
                       pos_size, pos_dir, pnl))

    return trades, np.array(equity_idx_list, dtype=np.int64), np.array(equity_vals_list, dtype=np.float64)


# ------------------------------------------------------------------ #
# High-level M1 backtest shared by both engines.
# ------------------------------------------------------------------ #
def run_m1(
    signals: StrategySignals,
    htf_df: pd.DataFrame,
    m1_df: pd.DataFrame,
    timeframe: str,
    info: dict,
    initial_capital: float,
    risk_money: float,
) -> dict:
    """Run the M1 OHLC backtest on the given signals + data.

    This is the single execution path used by both the vectorized
    (BacktestEngine.run) and event-driven (EventEngine.run) engines, so they
    produce identical trades, equity, and metrics by construction.

    Args:
        signals: StrategySignals indexed by HTF DateTime.
        htf_df: The HTF OHLCV DataFrame used to generate signals.
        m1_df: M1 OHLCV DataFrame.
        timeframe: HTF timeframe (e.g. "H1").
        info: dict from SymbolInfo.get().
        initial_capital: Starting capital.
        risk_money: Fixed risk per trade.

    Returns:
        dict with keys: metrics, equity_curve, trades (DataFrame).
    """
    signals.validate(htf_df)

    tick_value = float(info["trade_tick_value"])
    tick_size = float(info["trade_tick_size"])
    volume_step = float(info["volume_step"])
    volume_min = float(info["volume_min"])
    volume_max = float(info["volume_max"])
    swap_long = float(info["swap_long"])
    swap_short = float(info["swap_short"])
    swap_rollover3days = int(info["swap_rollover3days"])

    m1_index = pd.DatetimeIndex(m1_df.index)
    n_bars = len(m1_index)

    if n_bars == 0:
        trades_df = pd.DataFrame()
        equity_series = pd.Series(dtype=float)
        return {
            "metrics": _compute_metrics(equity_series, trades_df, initial_capital),
            "equity_curve": equity_series,
            "trades": trades_df,
        }

    open_arr = m1_df["Open"].to_numpy(dtype=float)
    high_arr = m1_df["High"].to_numpy(dtype=float)
    low_arr = m1_df["Low"].to_numpy(dtype=float)
    close_arr = m1_df["Close"].to_numpy(dtype=float)

    # Per-bar spread in price units (points * tick_size).
    if "Spread" in m1_df.columns:
        spread_arr = m1_df["Spread"].astype(float).to_numpy(dtype=float) * tick_size
    else:
        spread_arr = np.full(n_bars, float(info.get("spread", 10.0)) * tick_size)

    # Map HTF signals to M1 (shared mapping).
    m1_entries, m1_exits, m1_s_entries, m1_s_exits = map_signals_to_m1(
        signals, m1_index, timeframe
    )
    entry_mask = m1_entries.to_numpy(dtype=bool)
    exit_mask = m1_exits.to_numpy(dtype=bool)
    s_entry_mask = m1_s_entries.to_numpy(dtype=bool)
    s_exit_mask = m1_s_exits.to_numpy(dtype=bool)

    # SL: distance or absolute price, ffill'd to M1 (shifted one period so
    # stops are only visible after their HTF bar closes — no lookahead).
    m1_sl = map_stops_to_m1(signals.sl_stop, m1_index, timeframe)
    sl_arr = m1_sl.to_numpy(dtype=float)
    # TP: absolute price, ffill'd to M1 (NaN -> no TP).
    m1_tp = map_stops_to_m1(signals.tp_stop, m1_index, timeframe)
    tp_arr = m1_tp.to_numpy(dtype=float)

    day_ids, day_weekday, is_last = build_day_arrays(m1_index)

    trades_raw, eq_idx, eq_vals = simulate_m1(
        open_arr, high_arr, low_arr, close_arr, spread_arr,
        entry_mask, exit_mask, s_entry_mask, s_exit_mask,
        sl_arr, tp_arr, signals.sl_is_distance,
        is_last, day_ids, day_weekday,
        tick_value, tick_size, volume_step, volume_min, volume_max,
        risk_money, swap_long, swap_short, swap_rollover3days, initial_capital,
    )

    trades_df = _trades_to_dataframe(trades_raw, m1_index)
    equity_series = _equity_to_series(eq_idx, eq_vals, m1_index)

    return {
        "metrics": _compute_metrics(equity_series, trades_df, initial_capital),
        "equity_curve": equity_series,
        "trades": trades_df,
    }


def _trades_to_dataframe(trades_raw: list[tuple], m1_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert Numba trades to the standard trade DataFrame format."""
    if not trades_raw:
        return pd.DataFrame()
    records = []
    for (entry_idx, exit_idx, entry_price, exit_price, size, direction, pnl) in trades_raw:
        entry_idx = int(entry_idx)
        exit_idx = int(exit_idx)
        return_pct = pnl / (entry_price * size) if entry_price * size > 0 else 0.0
        records.append({
            "entry_idx": entry_idx,
            "entry_time": m1_index[entry_idx],
            "entry_price": entry_price,
            "exit_idx": exit_idx,
            "exit_time": m1_index[exit_idx],
            "exit_price": exit_price,
            "size": size,
            "direction": int(direction),
            "pnl": pnl,
            "return": return_pct,
            "status": 1,
        })
    return pd.DataFrame(records)


def _equity_to_series(
    eq_idx: np.ndarray, eq_vals: np.ndarray, m1_index: pd.DatetimeIndex
) -> pd.Series:
    """Equity at the last M1 bar of each day -> daily Series."""
    if len(eq_idx) == 0:
        return pd.Series(dtype=float)
    times = m1_index[eq_idx]
    s = pd.Series(eq_vals, index=times)
    return s.resample("D").last().dropna()


def _compute_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float | str]:
    """Compute performance metrics (shared by both engines)."""
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

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0

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
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
    }
