# Project Handover: TQQQ Trading & Momentum Regime Terminal

Last updated: 2026-11-19

## Read This First

This is the continuity record for future human and LLM maintainers. Read it before editing. Do not revert existing work, preserve dual-platform launch, and update this file whenever code, behavior, dependencies, tests, or known issues change.

## Mission

Maximize exposure to 3x Nasdaq upside (TQQQ) during sustained bull market regimes while systematically sidestepping catastrophic drawdowns (-70% to -85%) during bear regimes, mitigate volatility drag in choppy sideways markets, and time recovery bounces using regime switches and quantitative indicators.

## Current Application

- Primary UI entrypoint: `app.py` (thin Streamlit shell, base port 8505).
- Shared execution engine: `engine.py` (400-day warm-up loader, backtest pipeline, Strategy Arena, Stress-Test Suite, dynamic per-strategy Trade Log).
- Chart builders: `charts.py` (Plotly candlestick/signals, Arena, equity curve, drawdown).
- UI cards & safe HTML: `ui_components.py` (`html.escape()`-safe banners, playbooks, KPI cards, live vs offline Gemini badges).
- Strategy registry & metadata: `strategy.py` (`StrategyRegistry`, `STRATEGY_METADATA`, `TQQQStrategy`; NumPy-accelerated).
- Backtesting engine: `backtester.py` (fractional position weights, inner-joined trading dates >= inception, Day-0 rebase, slippage/RF support, real TQQQ & QQQ B&H benchmarks).
- Market data layer & canonical cache: `data_handler.py` (1 canonical `{ticker}_1d.csv` per ticker, incremental refresh, in-memory slicing, inclusive `end_date`).
- Technical indicators: `indicators.py` (Wilder's smoothed EMA RSI with zero-loss guard, SMA, EMA, MACD).
- Gemini advisor: `llm_advisor.py` (`google-genai`, `GEMINI_MODEL` override, explicit `is_fallback` / `error_detail` metadata).
- CLI execution: `main.py` (`--mode`, `--start`, `--end`, `--strategy`, `--buffer`, `--capital`, `--slippage-bps`, `--rf`).
- Test suite: `pytest.ini` and `tests/` (`test_indicators.py`, `test_strategy.py`, `test_backtester.py`, `test_data_handler.py`).

## AI Contract

- Keep Gemini as the in-app AI provider unless explicitly told otherwise.
- Environment variables: `GEMINI_API_KEY` (required for live calls) and optional `GEMINI_MODEL` override via `.env`.
- Default model order: `gemini-3.8-flash`, then `gemini-3.7-flash`.
- Target SDK: `google-genai` (deprecated `google-generativeai` removed).
- Quantitative backtesting, strategy signals, and charts remain 100% operational without an API key; offline fallbacks are explicitly flagged in UI and payloads.

## Launch

- Windows: `run.bat` (hash-gated pip sync, socket port probe starting at 8505)
- macOS/Linux: `sh run.sh` or `./run.sh` (POSIX-safe `.` activation, hash-gated pip sync, socket port probe starting at 8505)
- Direct: `.venv/bin/streamlit run app.py --server.port 8505 --server.headless true`

## Verification

- Syntax: `.venv/bin/python -m py_compile *.py`
- Unit tests: `.venv/bin/pytest -v`
- SDK check: `.venv/bin/python -W error::FutureWarning -c "from google import genai; import llm_advisor"`
- CLI smoke: `.venv/bin/python main.py --mode signals`

## Known Risks

- Yahoo Finance historical feeds are external failure points; canonical CSV cache provides offline resilience once populated.
- 3x leveraged ETF backtests depend on dividend/split-adjusted Yahoo Finance closes (`auto_adjust=True`); full refresh (`force_refresh=True`) should be triggered periodically after ETF splits.

## Workspace State

This is an independent Git repository on branch `main`.

## Change Log

- 2026-11-19: Fixed fractional 0.5 position accounting, pre-inception date alignment, Day-0 equity rebase, Wilder RSI, `rsi_mean_reversion` dip-buyer logic, price-normalized `dual_momentum` MACD threshold, single-file canonical CSV caching (removed 72 legacy CSVs), modularized `app.py` into `engine.py`/`charts.py`/`ui_components.py`/`app.py`, upgraded `llm_advisor.py` to `google-genai` with honest fallback badges & HTML escaping, standardized `run.sh`/`run.bat`, and added `pytest` suite.
- 2026-09-11: Added the initial cross-model handover record and Windows launcher.
