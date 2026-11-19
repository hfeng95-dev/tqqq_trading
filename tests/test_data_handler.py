import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import engine
import llm_advisor
import ui_components
from data_handler import DataHandler


def _fake_yf_df(start="2020-01-02", periods=10, base=100.0):
    idx = pd.bdate_range(start, periods=periods)
    vals = np.linspace(base, base + periods - 1, periods)
    return pd.DataFrame(
        {"Open": vals, "High": vals + 1, "Low": vals - 1, "Close": vals, "Volume": 1000},
        index=idx,
    )


def test_mutable_default_tickers_not_shared(tmp_path):
    h1 = DataHandler(cache_dir=str(tmp_path))
    h2 = DataHandler(cache_dir=str(tmp_path))
    h1.tickers.append("SPY")
    assert "SPY" not in h2.tickers
    assert h2.tickers == ["TQQQ", "SQQQ", "QQQ"]


def test_exclusive_end_date_and_canonical_cache_slicing(tmp_path):
    calls = []

    def fake_download(ticker, start=None, end=None, interval="1d", auto_adjust=True, progress=False):
        calls.append((ticker, start, end))
        return _fake_yf_df("2023-01-02", periods=30, base=100.0)

    with patch("data_handler.yf.download", side_effect=fake_download):
        handler = DataHandler(tickers=["QQQ"], cache_dir=str(tmp_path))
        res1 = handler.fetch_data(start_date="2023-01-02", end_date="2023-01-20", use_cache=True)

    # yf.download received inclusive end + 1 day
    assert len(calls) == 1
    assert calls[0][2] == "2023-01-21"
    # Canonical single CSV file created
    csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
    assert csv_files == ["QQQ_1d.csv"]
    assert res1["QQQ"].index.max() <= pd.Timestamp("2023-01-20")

    # Slicing another sub-range within cached max should NOT call yf.download again
    with patch("data_handler.yf.download", side_effect=AssertionError("Should hit cache")):
        res2 = handler.fetch_data(start_date="2023-01-05", end_date="2023-01-13", use_cache=True)
    assert res2["QQQ"].index.min() >= pd.Timestamp("2023-01-05")
    assert res2["QQQ"].index.max() <= pd.Timestamp("2023-01-13")


def test_incremental_refresh_extends_canonical_csv(tmp_path):
    initial = _fake_yf_df("2023-01-02", periods=10, base=100.0)  # through 2023-01-13
    handler = DataHandler(tickers=["QQQ"], cache_dir=str(tmp_path))
    initial.to_csv(tmp_path / "QQQ_1d.csv")

    inc_calls = []

    def fake_inc_download(ticker, start=None, end=None, interval="1d", auto_adjust=True, progress=False):
        inc_calls.append((start, end))
        return _fake_yf_df("2023-01-16", periods=5, base=110.0)

    with patch("data_handler.yf.download", side_effect=fake_inc_download):
        res = handler.fetch_data(start_date="2023-01-02", end_date="2023-01-20", use_cache=True)

    assert len(inc_calls) == 1
    assert inc_calls[0][0] == "2023-01-14"
    assert inc_calls[0][1] == "2023-01-21"
    assert len(res["QQQ"]) == 15
    assert not res["QQQ"].index.duplicated().any()


def test_llm_advisor_offline_fallback_and_html_escape(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-test")

    models = llm_advisor.get_available_models()
    assert models[0] == "gemini-custom-test"
    assert "gemini-3.8-flash" in models
    assert "gemini-3.7-flash" in models
    # When model_name is None, GEMINI_MODEL leads; explicit model_name always leads
    assert llm_advisor._resolve_models(None)[0] == "gemini-custom-test"
    assert llm_advisor._resolve_models("gemini-3.8-flash")[0] == "gemini-3.8-flash"
    assert llm_advisor._resolve_models("gemini-3.8-flash")[1] == "gemini-custom-test"

    res = llm_advisor.analyze_market_regime(
        {
            "qqq_price": 500.0,
            "sma_200": 480.0,
            "buffer_pct": 5.0,
            "lower_buffer": 456.0,
            "current_stance": "50% LONG TQQQ / 50% CASH",
        }
    )
    assert res["is_fallback"] is True
    assert res["error_detail"]
    assert res["allocation"]["TQQQ_pct"] == 50

    # Verify HTML escaping prevents XSS injection in Gemini card
    malicious = dict(res)
    malicious["regime"] = "<script>alert(1)</script>"
    malicious["regime_thesis"] = "<img src=x onerror=alert(2)>"
    card_html = ui_components.render_gemini_analysis_card_html(malicious, 456.0)
    assert "<script>" not in card_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in card_html
    assert "Offline / Heuristic Fallback" in card_html


def test_engine_arena_stress_and_trade_log():
    idx = pd.bdate_range("2017-01-02", periods=1600)
    rng = np.random.default_rng(99)
    rets = rng.normal(0.0004, 0.012, len(idx))
    qqq_close = 120.0 * np.cumprod(1.0 + rets)
    tqqq_close = 20.0 * np.cumprod(1.0 + 3.0 * rets)
    sqqq_close = 80.0 * np.cumprod(1.0 - 3.0 * rets)

    raw_data = {
        "QQQ": pd.DataFrame(
            {"Open": qqq_close, "High": qqq_close * 1.01, "Low": qqq_close * 0.99, "Close": qqq_close, "Volume": 1000},
            index=idx,
        ),
        "TQQQ": pd.DataFrame({"Close": tqqq_close}, index=idx),
        "SQQQ": pd.DataFrame({"Close": sqqq_close}, index=idx),
    }

    arena_rows, arena_curves = engine.run_strategy_arena(
        raw_data, buffer_pct=5.0, initial_capital=50000.0, start_date="2018-01-01", end_date="2022-12-31"
    )
    # 6 strategies + 2 real baselines (TQQQ B&H, QQQ B&H)
    assert len(arena_rows) == 8
    assert len(arena_curves) == 6
    tqqq_row = [r for r in arena_rows if "TQQQ Buy & Hold" in r["Strategy Name"]][0]
    assert tqqq_row["Sharpe Ratio"] != "N/A"
    assert tqqq_row["Max Drawdown"] != "-82.0%"

    stress_rows = engine.run_stress_test_suite(
        raw_data,
        strategy_key="vol_managed",
        buffer_pct=5.0,
        initial_capital=25000.0,
        regimes=[("2018-2020", "2018-01-01", "2020-12-31"), ("2022", "2022-01-01", "2022-12-31")],
    )
    assert len(stress_rows) == 2
    assert "Final Wealth ($25,000 Init)" in stress_rows[0]

    _, bt_df = engine.run_backtest_pipeline(
        raw_data, strategy_key="vol_managed", buffer_pct=5.0, start_date="2018-01-01", end_date="2022-12-31"
    )
    trade_log = engine.build_trade_log(bt_df, "vol_managed", 5.0, tail_n=10)
    assert isinstance(trade_log, pd.DataFrame)
    assert engine.format_position_label(0.5) == "50% LONG TQQQ / 50% CASH"
    assert engine.format_position_label(-1.0) == "100% SHORT / LONG SQQQ"
