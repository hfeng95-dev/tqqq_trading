import numpy as np
import pandas as pd
import pytest

from strategy import StrategyRegistry, TQQQStrategy


def _make_ohlcv(closes):
    idx = pd.bdate_range("2018-01-01", periods=len(closes))
    s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": s, "High": s * 1.01, "Low": s * 0.99, "Close": s, "Volume": 1_000_000}
    )


def test_strategy_metadata_covers_all_strategies():
    for key in StrategyRegistry.STRATEGIES:
        assert key in StrategyRegistry.STRATEGY_METADATA
        meta = StrategyRegistry.get_metadata(key)
        assert meta["entry_rules"]
        assert meta["exit_rules"]


@pytest.mark.parametrize("strategy_key", list(StrategyRegistry.STRATEGIES.keys()))
def test_all_strategies_run_cleanly(strategy_key):
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.015, 350)
    closes = 100.0 * np.cumprod(1.0 + rets)
    df = _make_ohlcv(closes)

    out = StrategyRegistry.calculate(df, strategy_key=strategy_key, buffer_pct=5.0)
    assert "Position" in out.columns
    assert "Trade_Action" in out.columns
    assert not out["Position"].isna().any()
    assert set(out["Trade_Action"].unique()).issubset({-1, 0, 1})
    assert out["Position"].between(-1.0, 1.0).all()


def test_vol_managed_emits_fractional_half_and_cash():
    # 210 calm bull bars -> low vol (1.0), then 40 medium-vol bars (~26% ann -> 0.5),
    # then 40 high-vol bars (~50% ann -> 0.0)
    calm = np.linspace(100.0, 130.0, 220)
    med_moves = np.array([0.0165 if i % 2 == 0 else -0.0162 for i in range(40)])
    med = calm[-1] * np.cumprod(1.0 + med_moves)
    high_moves = np.array([0.035 if i % 2 == 0 else -0.034 for i in range(40)])
    high = med[-1] * np.cumprod(1.0 + high_moves)

    df = _make_ohlcv(np.concatenate([calm, med, high]))
    out = StrategyRegistry.calculate(df, strategy_key="vol_managed", buffer_pct=5.0)

    post_warmup = out["Position"].iloc[200:]
    assert 1.0 in post_warmup.values
    assert 0.5 in post_warmup.values
    assert 0.0 in post_warmup.values


def test_rsi_mean_reversion_is_true_dip_buyer():
    # Build a bull trend where RSI stays ~55-65 (should NOT enter while flat),
    # then a sharp pullback above lower_band pushing RSI < 40 (MUST enter 1.0),
    # then a rally pushing RSI > 70 (MUST exit to 0.0).
    base = np.linspace(100.0, 140.0, 210)
    # Oscillate gently around 140 so RSI stays neutral (~50-55) above SMA_200 (~120)
    neutral = 140.0 + np.array([0.4 if i % 2 == 0 else -0.35 for i in range(25)])
    # Sharp pullback from 140 -> 126 (still > lower_band ~117) to push RSI < 40
    dip = np.linspace(139.0, 126.0, 10)
    # Mild bounce keeping RSI between 45 and 60 (should stay long)
    hold = np.linspace(127.0, 133.0, 10)
    # Strong rip pushing RSI > 70 (should take profit to 0.0)
    rip = np.linspace(135.0, 168.0, 18)

    closes = np.concatenate([base, neutral, dip, hold, rip])
    df = _make_ohlcv(closes)
    out = StrategyRegistry.calculate(df, strategy_key="rsi_mean_reversion", buffer_pct=5.0)

    # During neutral phase after warmup, while flat and RSI >= 40, must stay 0.0 (not always-long!)
    neutral_slice = out.iloc[215:234]
    assert (neutral_slice["RSI"] >= 40.0).all()
    assert (neutral_slice["Position"] == 0.0).all()

    # Dip phase pushes RSI < 40 -> enters 1.0
    dip_slice = out.iloc[235:255]
    assert (dip_slice["Position"] == 1.0).any()

    # Rip phase pushes RSI > 70 -> exits back to 0.0
    rip_slice = out.iloc[255:]
    assert (rip_slice["RSI"] > 70.0).any()
    assert rip_slice["Position"].iloc[-1] == 0.0


def test_dual_momentum_macd_threshold_is_price_scale_invariant():
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0002, 0.018, 320)
    closes_1x = 50.0 * np.cumprod(1.0 + rets)
    closes_10x = closes_1x * 10.0

    out_1x = StrategyRegistry.calculate(_make_ohlcv(closes_1x), strategy_key="dual_momentum", buffer_pct=5.0)
    out_10x = StrategyRegistry.calculate(_make_ohlcv(closes_10x), strategy_key="dual_momentum", buffer_pct=5.0)

    pd.testing.assert_series_equal(out_1x["Position"], out_10x["Position"])


def test_long_short_hedged_switches_to_short_below_lower_band():
    up = np.linspace(100.0, 150.0, 220)
    crash = np.linspace(148.0, 80.0, 40)
    df = _make_ohlcv(np.concatenate([up, crash]))
    out = TQQQStrategy.calculate_positions(df, strategy_key="long_short_hedged", buffer_pct=5.0)

    assert out["Position"].iloc[215] == 1.0
    assert out["Position"].iloc[-1] == -1.0
