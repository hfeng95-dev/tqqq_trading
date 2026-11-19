import math

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester


def _make_series(dates, closes, positions=None):
    df = pd.DataFrame({"Close": closes}, index=dates)
    if positions is not None:
        df["Position"] = positions
    return df


def test_fractional_half_position_earns_half_tqqq_plus_half_rf():
    dates = pd.bdate_range("2023-01-02", periods=6)
    # TQQQ moves +10% every day
    tqqq_closes = [100.0 * (1.10 ** i) for i in range(6)]
    sqqq_closes = [100.0 * (0.90 ** i) for i in range(6)]
    qqq_closes = [100.0 * (1.03 ** i) for i in range(6)]
    positions = [0.5] * 6

    df_map = {
        "QQQ": _make_series(dates, qqq_closes, positions),
        "TQQQ": _make_series(dates, tqqq_closes),
        "SQQQ": _make_series(dates, sqqq_closes),
    }

    rf = 0.05
    daily_rf = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    bt = Backtester(df_map, initial_capital=10000.0, risk_free_rate=rf)
    metrics = bt.run()

    # Day 0 rebased to 0.0; Days 1..5 earn 0.5*0.10 + 0.5*daily_rf
    assert bt.df["Strategy_Cum_Returns"].iloc[0] == 1.0
    assert bt.df["Portfolio_Value"].iloc[0] == 10000.0
    expected_daily = 0.5 * 0.10 + 0.5 * daily_rf
    assert np.allclose(bt.df["Strategy_Returns"].iloc[1:], expected_daily, rtol=1e-6)
    expected_total = (1.0 + expected_daily) ** 5 - 1.0
    assert math.isclose(metrics["Total Return"], expected_total, rel_tol=1e-6)


def test_short_position_earns_sqqq_return():
    dates = pd.bdate_range("2023-01-02", periods=5)
    qqq_closes = [100.0, 99.0, 98.0, 97.0, 96.0]
    tqqq_closes = [100.0, 97.0, 94.09, 91.2673, 88.529281]
    sqqq_closes = [100.0, 103.0, 106.09, 109.2727, 112.550881]
    positions = [-1.0] * 5

    bt = Backtester(
        {
            "QQQ": _make_series(dates, qqq_closes, positions),
            "TQQQ": _make_series(dates, tqqq_closes),
            "SQQQ": _make_series(dates, sqqq_closes),
        },
        initial_capital=10000.0,
    )
    metrics = bt.run()
    assert np.allclose(bt.df["Strategy_Returns"].iloc[1:], 0.03, atol=1e-5)
    assert math.isclose(metrics["Total Return"], (1.03 ** 4) - 1.0, rel_tol=1e-5)


def test_pre_inception_alignment_and_day0_rebase_no_prior_leak():
    qqq_dates = pd.bdate_range("2010-01-04", periods=40)
    # TQQQ/SQQQ start later (inception)
    lev_dates = qqq_dates[20:]

    qqq_closes = np.linspace(40.0, 50.0, len(qqq_dates))
    tqqq_closes = np.linspace(10.0, 20.0, len(lev_dates))
    sqqq_closes = np.linspace(100.0, 60.0, len(lev_dates))

    bt = Backtester(
        {
            "QQQ": _make_series(qqq_dates, qqq_closes, [1.0] * len(qqq_dates)),
            "TQQQ": _make_series(lev_dates, tqqq_closes),
            "SQQQ": _make_series(lev_dates, sqqq_closes),
        },
        initial_capital=25000.0,
        start_date="2010-01-01",
    )
    metrics = bt.run()

    assert "Error" not in metrics
    assert metrics["Effective Start Date"] == lev_dates[0].strftime("%Y-%m-%d")
    assert not bt.df["Strategy_Returns"].isna().any()
    assert bt.df["Strategy_Cum_Returns"].iloc[0] == 1.0
    assert bt.df["TQQQ_B_H_Returns"].iloc[0] == 1.0
    assert bt.df["Market_Cum_Returns"].iloc[0] == 1.0
    assert bt.df["Portfolio_Value"].iloc[0] == 25000.0

    # Also test slicing mid-series: Day-0 of slice must NOT leak prior-day return
    mid_date = lev_dates[5].strftime("%Y-%m-%d")
    bt_mid = Backtester(
        {
            "QQQ": _make_series(qqq_dates, qqq_closes, [1.0] * len(qqq_dates)),
            "TQQQ": _make_series(lev_dates, tqqq_closes),
            "SQQQ": _make_series(lev_dates, sqqq_closes),
        },
        initial_capital=10000.0,
        start_date=mid_date,
    )
    m_mid = bt_mid.run()
    assert m_mid["Effective Start Date"] == mid_date
    assert bt_mid.df["Strategy_Returns"].iloc[0] == 0.0
    assert bt_mid.df["Strategy_Cum_Returns"].iloc[0] == 1.0
    assert bt_mid.df["Portfolio_Value"].iloc[0] == 10000.0


def test_slippage_deducted_on_position_turnover():
    dates = pd.bdate_range("2023-01-02", periods=6)
    flat = [100.0] * 6
    # Position flips 0 -> 1 on day 1 (realized day 2), then 1 -> -1 on day 3 (realized day 4, 2x turnover)
    positions = [0.0, 1.0, 1.0, -1.0, -1.0, -1.0]

    bt = Backtester(
        {
            "QQQ": _make_series(dates, flat, positions),
            "TQQQ": _make_series(dates, flat),
            "SQQQ": _make_series(dates, flat),
        },
        initial_capital=10000.0,
        slippage_bps=10.0,  # 0.10% per unit turnover
    )
    bt.run()

    # Day 2 turnover = |1 - 0| = 1 -> -0.001; Day 4 turnover = |-1 - 1| = 2 -> -0.002
    assert math.isclose(bt.df["Strategy_Returns"].iloc[2], -0.001, abs_tol=1e-9)
    assert math.isclose(bt.df["Strategy_Returns"].iloc[4], -0.002, abs_tol=1e-9)
    assert math.isclose(bt.df["Strategy_Returns"].iloc[3], 0.0, abs_tol=1e-9)


def test_real_benchmarks_and_wipeout_short_window_guards():
    dates = pd.bdate_range("2022-01-03", periods=10)
    # TQQQ peaks at 100 then drops to 50 (-50% MDD) then recovers to 60
    tqqq = [100.0, 100.0, 90.0, 80.0, 70.0, 50.0, 52.0, 55.0, 58.0, 60.0]
    qqq = [100.0, 100.0, 97.0, 94.0, 91.0, 85.0, 86.0, 87.0, 88.0, 89.0]
    sqqq = [100.0] * 10
    bt = Backtester(
        {
            "QQQ": _make_series(dates, qqq, [0.0] * 10),
            "TQQQ": _make_series(dates, tqqq),
            "SQQQ": _make_series(dates, sqqq),
        },
        initial_capital=10000.0,
    )
    m = bt.run()
    assert math.isclose(m["TQQQ B&H Max Drawdown"], -0.50, rel_tol=1e-6)
    assert math.isclose(m["QQQ B&H Max Drawdown"], -0.15, rel_tol=1e-6)
    assert math.isfinite(m["TQQQ B&H CAGR"])
    assert math.isfinite(m["TQQQ B&H Sharpe"])
    assert math.isfinite(m["CAGR Alpha vs TQQQ"])

    # 1-day short window & -100% wipeout guard
    d2 = pd.bdate_range("2022-01-03", periods=2)
    bt_wipe = Backtester(
        {
            "QQQ": _make_series(d2, [100.0, 50.0], [1.0, 1.0]),
            "TQQQ": _make_series(d2, [100.0, 0.0]),
            "SQQQ": _make_series(d2, [100.0, 100.0]),
        }
    )
    m_wipe = bt_wipe.run()
    assert m_wipe["Total Return"] == -1.0
    assert m_wipe["Annualized Return"] == -1.0
    assert math.isfinite(m_wipe["Sharpe Ratio"])
