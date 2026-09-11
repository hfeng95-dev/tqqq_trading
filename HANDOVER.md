# Project Handover: TQQQ Trading & Momentum Regime Terminal

Last updated: 2026-09-11

## Read This First

This is the continuity record for future human and LLM maintainers. Read it before editing. Do not revert existing work, preserve dual-platform launch, and update this file whenever code, behavior, dependencies, tests, or known issues change.

## Mission

Maximize exposure to 3x Nasdaq upside (TQQQ) during sustained bull market regimes while systematically sidestepping catastrophic drawdowns (-70% to -85%) during bear regimes, mitigate volatility drag in choppy sideways markets, and time recovery bounces using regime switches and quantitative indicators.

## Current Application

- Primary UI: `app.py` (Streamlit, base port 8505).
- Strategy registry and core models: `strategy.py` (`StrategyRegistry`, `TQQQStrategy`).
- Backtesting engine: `backtester.py`.
- Market data layer & cache: `data_handler.py`.
- Technical indicators: `indicators.py`.
- Gemini advisor: `llm_advisor.py`.
- CLI execution: `main.py`.
- Important decisions remain deterministic; Gemini provides narrative regime analysis, risk assessment, and trade execution plans.

## AI Contract

- Keep Gemini as the in-app AI provider unless explicitly told otherwise.
- Environment variable: `GEMINI_API_KEY` via `.env`.
- Model order: `gemini-3.8-flash`, then `gemini-3.7-flash`.
- Target SDK: `google-genai`.
- Quantitative backtesting, strategy signals, and charts must remain 100% operational without an API key.

## Launch

- Windows: `run.bat`
- macOS/Linux: `sh run.sh`
- Direct: `streamlit run app.py --server.port 8505 --server.headless true`
- Both launchers search for or use port 8505.

## Verification

- Install: `python -m pip install -r requirements.txt`
- Syntax: `python -m py_compile app.py backtester.py data_handler.py indicators.py llm_advisor.py main.py strategy.py`
- CLI test: `python main.py`
- UI smoke test: launch, run strategy backtests, examine performance metrics, and test Gemini analysis both with and without `GEMINI_API_KEY`.

## Known Risks

- Yahoo Finance historical and intraday feeds are external failure points.
- 3x leveraged ETF simulation requires accurate handling of expense ratios, volatility drag, and split adjustments.
- `data_cache/` contains cached parquet/CSV files; test with and without cache to verify fresh signal generation.
- Python package versions (`pandas`, `numpy`, `plotly`) should remain aligned with other workspace tools.

## Workspace State

This is an independent Git repository on branch `main`.

## Maintenance Contract

After every meaningful change:

1. Update `Last updated` and the relevant sections.
2. Record only verification that was actually run.
3. Add unresolved provider, data-shape, or package failures to `Known Risks`.
4. Preserve Gemini, deterministic no-key behavior, and Windows/macOS launch support unless explicitly redirected.
5. Do not erase prior user changes or silently alter strategy logic.

## Change Log

- 2026-09-11: Added the initial cross-model handover record and Windows launcher.
