import numpy as np
import pandas as pd
import pytest

from indicators import Indicators


def _make_df(closes):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": s, "High": s * 1.01, "Low": s * 0.99, "Close": s, "Volume": 1_000_000}
    )


def test_wilder_rsi_monotonic_up_is_100():
    df = _make_df(np.linspace(100.0, 150.0, 40))
    out = Indicators.add_rsi(df.copy(), period=14)
    # Warm-up should be NaN, post-warmup should be exactly 100.0 (zero-loss guard)
    assert out["RSI"].iloc[:14].isna().all()
    assert (out["RSI"].iloc[14:] == 100.0).all()


def test_wilder_rsi_monotonic_down_is_0():
    df = _make_df(np.linspace(150.0, 100.0, 40))
    out = Indicators.add_rsi(df.copy(), period=14)
    assert out["RSI"].iloc[:14].isna().all()
    assert np.allclose(out["RSI"].iloc[14:], 0.0)


def test_wilder_rsi_flat_series_is_50():
    df = _make_df([100.0] * 40)
    out = Indicators.add_rsi(df.copy(), period=14)
    assert out["RSI"].iloc[:14].isna().all()
    assert np.allclose(out["RSI"].iloc[14:], 50.0)


def test_wilder_rsi_differs_from_cutler_sma_and_matches_ewm():
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0.1, 1.5, 80))
    df = _make_df(closes)
    out = Indicators.add_rsi(df.copy(), period=14)

    delta = df["Close"].diff()
    exp_gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    exp_loss = (-delta).clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    expected = 100.0 - (100.0 / (1.0 + exp_gain / exp_loss))

    pd.testing.assert_series_equal(out["RSI"], expected, check_names=False)
    assert out["RSI"].iloc[14:].between(0.0, 100.0).all()


def test_apply_all_adds_required_columns():
    df = _make_df(np.linspace(100.0, 200.0, 250))
    out = Indicators.apply_all(df.copy())
    for col in ("SMA_200", "EMA_200", "SMA_50", "EMA_50", "EMA_10", "RSI", "MACD", "MACD_Signal", "MACD_Hist"):
        assert col in out.columns
    assert pd.notna(out["SMA_200"].iloc[-1])
