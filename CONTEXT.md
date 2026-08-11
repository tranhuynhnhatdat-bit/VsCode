# Trading Engine

A Python engine for developing, backtesting, and genetic-optimizing trading strategies for forex/CFD markets, with an MQL5 port for live MT5 deployment. It turns a strategy into OHLCV signals, executes those signals with MQL5-faithful "1 minute OHLC" semantics, and validates candidates through a GA-driven, out-of-sample funnel.

## Layers

**Strategy** (`strategy/`, `composable/`)
Pure, stateless signal generators: an OHLCV DataFrame in, `StrategySignals` out. The canonical output contract holds entry/exit/short booleans plus stop price series. `ComposableStrategy` is the flagship — a fixed time/session skeleton (enter/exit at configured hours on configured session days) combined with up to N indicator conditions chosen by the GA, joined by a single AND/OR connective.

**Backtest engine** (`backtest/`)
Two engines sharing one execution core. `_m1_core.py` is a Numba-JIT `simulate_m1` that walks every M1 bar with single-position, "1 minute OHLC" semantics matching the MQL5 engine. `BacktestEngine.run()` uses that shared core; `run_htf()` is a fast vectorbt screen on the strategy's native higher timeframe. `_mapping.py` maps HTF signals/stops onto M1 so both engines agree on timing and fills.

**Optimization** (`optimization/`)
`TestEngine` runs a two-phase pipeline: an island-model GA (`genetic.py`) collects diverse strategies on the higher-timeframe train window, then a fast HTF screen gates them, followed by an M1 confirmation funnel with out-of-sample (OOS1/OOS2) validation.

**Data & execution** (`data_manager.py`, `symbol_info.py`, `Run_Engine/`, `MQL5 strategy/`)
Data loading, per-symbol metadata (tick size/value, spread, swap, volume limits), runnable scripts, and the MQL5 reference implementation used for live trading.

## Key terms

**StrategySignals**:
The vectorized signal contract a strategy emits — entries, exits, short entries/exits, stop prices, and whether the stop is an absolute price or a distance.
_Avoid_: trade list, orders

**M1 core**:
The shared Numba execution path that fills signals on M1 bars with MQL5-faithful OHLC semantics; both the vectorbt and event-driven engines use it so results are identical.
_Avoid_: simulator, M1 loop

**HTF→M1 mapping**:
The shared vectorized timing that maps higher-timeframe signals and stops onto M1 bars, guaranteeing the two engines agree on when fills happen.
_Avoid_: resampling, alignment

**ComposableStrategy**:
A strategy skeleton of fixed session/time logic gated by GA-composed indicator conditions.
_Avoid_: template strategy

**Genetic optimizer**:
The island-model GA plus two-phase validation funnel (HTF collection → M1 confirmation → OOS gates) that selects robust strategies.
_Avoid_: search, brute force

**MQL5 port**:
The reference implementation that mirrors the Python strategy and engine semantics for live MT5 deployment.