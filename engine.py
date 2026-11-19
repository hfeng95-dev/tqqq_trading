import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtester import Backtester
from data_handler import DataHandler
from strategy import StrategyRegistry

logger = logging.getLogger(__name__)

WARMUP_DAYS = 400
CANONICAL_HISTORY_START = "2008-01-01"

STRESS_REGIMES: List[Tuple[str, str, Optional[str]]] = [
    ("Full History (2010 - Present)", "2010-01-01", None),
    ("2023 - Present AI Mega-Cap Bull Market", "2023-01-01", None),
    ("2022 Fed Rate Hike Bear Market", "2022-01-01", "2022-12-31"),
    ("2020 Covid Shock & Recovery", "2020-01-01", "2020-12-31"),
    ("2018 Trade War & Fed Tightening", "2018-01-01", "2018-12-31"),
]


def load_canonical_data(
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Loads QQQ, TQQQ, and SQQQ with a 400-calendar-day warm-up buffer before `start_date`
    using the canonical single-file-per-ticker DataHandler.
    """
    handler = DataHandler(cache_dir=cache_dir)
    actual_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=WARMUP_DAYS)
    ).strftime("%Y-%m-%d")
    data = handler.fetch_data(
        start_date=actual_start,
        end_date=end_date,
        use_cache=True,
        force_refresh=force_refresh,
    )
    return data, dict(handler.last_errors)


def run_backtest_pipeline(
    raw_data: Dict[str, pd.DataFrame],
    strategy_key: str = "alpha_locked",
    buffer_pct: float = 5.0,
    initial_capital: float = 10000.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Runs strategy calculation on QQQ and executes the Backtester over [start_date, end_date].
    Does not mutate `raw_data`.
    """
    if not raw_data:
        return {"Error": "No market data loaded"}, None
    for req in ("QQQ", "TQQQ", "SQQQ"):
        if req not in raw_data or raw_data[req] is None or raw_data[req].empty:
            return {"Error": f"Missing market data for {req}"}, None

    df_qqq = StrategyRegistry.calculate(
        raw_data["QQQ"], strategy_key=strategy_key, buffer_pct=buffer_pct
    )
    bt_data = {
        "QQQ": df_qqq,
        "TQQQ": raw_data["TQQQ"],
        "SQQQ": raw_data["SQQQ"],
    }
    backtester = Backtester(
        bt_data,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )
    metrics = backtester.run()
    return metrics, backtester.df


def run_strategy_arena(
    raw_data: Dict[str, pd.DataFrame],
    buffer_pct: float = 5.0,
    initial_capital: float = 10000.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """
    Compares all 6 registered strategies plus real TQQQ B&H and QQQ B&H baselines.
    """
    arena_results: List[Dict[str, Any]] = []
    arena_curves: Dict[str, pd.Series] = {}
    ref_metrics: Optional[Dict[str, Any]] = None

    for s_key, s_name in StrategyRegistry.STRATEGIES.items():
        m, bt_df = run_backtest_pipeline(
            raw_data,
            strategy_key=s_key,
            buffer_pct=buffer_pct,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            slippage_bps=slippage_bps,
            risk_free_rate=risk_free_rate,
        )
        if not m or "Error" in m or bt_df is None or bt_df.empty:
            continue

        if ref_metrics is None:
            ref_metrics = m
        arena_curves[s_name] = bt_df["Strategy_Cum_Returns"]
        arena_results.append(
            {
                "Strategy Name": s_name,
                "Total Return": f"{m['Total Return']:+.1%}",
                "Annualized Return (CAGR)": f"{m['Annualized Return']:+.1%}",
                "CAGR Alpha vs TQQQ": f"{m['CAGR Alpha vs TQQQ']:+.1%}",
                "Total Spread vs TQQQ": f"{m['Alpha vs TQQQ']:+.1%}",
                "Sharpe Ratio": f"{m['Sharpe Ratio']:.2f}",
                "Max Drawdown": f"{m['Max Drawdown']:.1%}",
                "Final Wealth ($)": f"${m['Final Wealth']:,.0f}",
            }
        )

    if ref_metrics is not None:
        arena_results.append(
            {
                "Strategy Name": "Baseline: TQQQ Buy & Hold (No Timing)",
                "Total Return": f"{ref_metrics['TQQQ B&H Return']:+.1%}",
                "Annualized Return (CAGR)": f"{ref_metrics['TQQQ B&H CAGR']:+.1%}",
                "CAGR Alpha vs TQQQ": "+0.0%",
                "Total Spread vs TQQQ": "+0.0%",
                "Sharpe Ratio": f"{ref_metrics['TQQQ B&H Sharpe']:.2f}",
                "Max Drawdown": f"{ref_metrics['TQQQ B&H Max Drawdown']:.1%}",
                "Final Wealth ($)": f"${ref_metrics['TQQQ B&H Final Wealth']:,.0f}",
            }
        )
        arena_results.append(
            {
                "Strategy Name": "Baseline: QQQ 1x Buy & Hold",
                "Total Return": f"{ref_metrics['QQQ B&H Return']:+.1%}",
                "Annualized Return (CAGR)": f"{ref_metrics['QQQ B&H CAGR']:+.1%}",
                "CAGR Alpha vs TQQQ": f"{ref_metrics['QQQ B&H CAGR'] - ref_metrics['TQQQ B&H CAGR']:+.1%}",
                "Total Spread vs TQQQ": f"{ref_metrics['QQQ B&H Return'] - ref_metrics['TQQQ B&H Return']:+.1%}",
                "Sharpe Ratio": f"{ref_metrics['QQQ B&H Sharpe']:.2f}",
                "Max Drawdown": f"{ref_metrics['QQQ B&H Max Drawdown']:.1%}",
                "Final Wealth ($)": f"${ref_metrics['QQQ B&H Final Wealth']:,.0f}",
            }
        )

    return arena_results, arena_curves


def run_stress_test_suite(
    full_raw_data: Dict[str, pd.DataFrame],
    strategy_key: str = "alpha_locked",
    buffer_pct: float = 5.0,
    initial_capital: float = 10000.0,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    regimes: Optional[List[Tuple[str, str, Optional[str]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs multi-regime stress tests by reusing ONE loaded canonical dataset and computing
    strategy positions once before slicing across each historical cycle.
    Honors `initial_capital`.
    """
    if not full_raw_data:
        return []
    for req in ("QQQ", "TQQQ", "SQQQ"):
        if req not in full_raw_data or full_raw_data[req] is None or full_raw_data[req].empty:
            return []

    df_qqq_signals = StrategyRegistry.calculate(
        full_raw_data["QQQ"], strategy_key=strategy_key, buffer_pct=buffer_pct
    )
    shared_data = {
        "QQQ": df_qqq_signals,
        "TQQQ": full_raw_data["TQQQ"],
        "SQQQ": full_raw_data["SQQQ"],
    }

    wealth_col = f"Final Wealth (${initial_capital:,.0f} Init)"
    stress_results: List[Dict[str, Any]] = []
    for name, s_dt, e_dt in (regimes or STRESS_REGIMES):
        bt = Backtester(
            shared_data,
            initial_capital=initial_capital,
            start_date=s_dt,
            end_date=e_dt,
            slippage_bps=slippage_bps,
            risk_free_rate=risk_free_rate,
        )
        m_res = bt.run()
        if not m_res or "Error" in m_res:
            continue

        stress_results.append(
            {
                "Market Regime / Cycle": name,
                "Effective Period": f"{m_res['Effective Start Date']} to {m_res['Effective End Date']}",
                "Strategy Return": f"{m_res['Total Return']:+.1%}",
                "TQQQ B&H Return": f"{m_res['TQQQ B&H Return']:+.1%}",
                "CAGR Alpha vs TQQQ": f"{m_res['CAGR Alpha vs TQQQ']:+.1%}",
                "Strategy Max Drawdown": f"{m_res['Max Drawdown']:.1%}",
                "TQQQ B&H Max Drawdown": f"{m_res['TQQQ B&H Max Drawdown']:.1%}",
                "Sharpe Ratio": f"{m_res['Sharpe Ratio']:.2f}",
                wealth_col: f"${m_res['Final Wealth']:,.0f}",
            }
        )

    return stress_results


def format_position_label(pos: float) -> str:
    """Human-readable stance label supporting 1.0, 0.5, 0.0, -1.0, and any fractional weight."""
    if pos >= 0.99:
        return "100% LONG TQQQ"
    if pos > 0.01:
        long_pct = int(round(pos * 100))
        cash_pct = max(0, 100 - long_pct)
        return f"{long_pct}% LONG TQQQ / {cash_pct}% CASH"
    if pos <= -0.99:
        return "100% SHORT / LONG SQQQ"
    if pos < -0.01:
        short_pct = int(round(abs(pos) * 100))
        cash_pct = max(0, 100 - short_pct)
        return f"{short_pct}% SQQQ / {cash_pct}% CASH"
    return "100% CASH / FLAT"


def _describe_trade_action(prev_pos: float, new_pos: float) -> str:
    if new_pos >= 0.99 and prev_pos <= -0.01:
        return "COVER SQQQ -> 100% BUY TQQQ"
    if new_pos >= 0.99 and prev_pos > 0.01:
        return "SCALE UP -> 100% TQQQ"
    if new_pos >= 0.99:
        return "BUY 100% TQQQ"
    if 0.01 < new_pos < 0.99:
        pct = int(round(new_pos * 100))
        if new_pos > prev_pos:
            return f"BUY {pct}% TQQQ / {100 - pct}% CASH"
        return f"REDUCE -> {pct}% TQQQ / {100 - pct}% CASH"
    if new_pos <= -0.01:
        pct = int(round(abs(new_pos) * 100))
        if prev_pos > 0.01:
            return f"SELL TQQQ -> {pct}% LONG SQQQ"
        return f"SHORT / {pct}% LONG SQQQ"
    if prev_pos <= -0.01:
        return "COVER SQQQ -> 100% CASH"
    return "SELL TQQQ -> 100% CASH"


def _describe_trade_reason(
    strategy_key: str,
    row: pd.Series,
    prev_pos: float,
    new_pos: float,
    buffer_pct: float,
) -> str:
    close = float(row.get("Close", 0.0))
    sma200 = float(row.get("SMA_200", 0.0))
    lower_buf = float(row.get("SMA_200_Lower_Buffer", 0.0))
    ema10 = float(row.get("EMA_10", 0.0))
    ema9 = float(row.get("EMA_9", 0.0))
    ema30 = float(row.get("EMA_30", 0.0))
    macd_hist = float(row.get("MACD_Hist", 0.0))
    rsi = float(row.get("RSI", 50.0)) if pd.notna(row.get("RSI")) else 50.0
    rvol = float(row.get("Realized_Vol_20", 20.0)) if pd.notna(row.get("Realized_Vol_20")) else 20.0

    if strategy_key == "alpha_locked":
        if new_pos > prev_pos:
            if close > sma200:
                return f"QQQ (${close:.2f}) reclaimed 200 SMA (${sma200:.2f})"
            return (
                f"Bear bounce: QQQ (${close:.2f}) > {buffer_pct:.1f}% floor (${lower_buf:.2f}) "
                f"& MACD Hist ({macd_hist:+.2f}) > 0"
            )
        return (
            f"QQQ (${close:.2f}) < {buffer_pct:.1f}% floor (${lower_buf:.2f}) "
            f"& MACD Hist ({macd_hist:+.2f}) <= 0"
        )

    if strategy_key == "dual_momentum":
        macd_pct = (macd_hist / close * 100.0) if close > 0 else 0.0
        if new_pos > prev_pos:
            if close > sma200:
                return (
                    f"QQQ (${close:.2f}) > 200 SMA (${sma200:.2f}) & 10 EMA (${ema10:.2f}), "
                    f"RSI {rsi:.1f} > 45"
                )
            return f"Normalized MACD recovery: MACD_H/Close {macd_pct:+.2f}% > +0.20%"
        if close < lower_buf:
            return f"QQQ (${close:.2f}) breached {buffer_pct:.1f}% floor (${lower_buf:.2f})"
        return f"Momentum break: QQQ (${close:.2f}) < 10 EMA (${ema10:.2f}) & RSI {rsi:.1f} < 40"

    if strategy_key == "vol_managed":
        if close > lower_buf:
            if new_pos >= 0.99:
                return f"QQQ > floor (${lower_buf:.2f}) & 20d Vol {rvol:.1f}% < 22% (Full 3x)"
            if new_pos > 0.01:
                return f"QQQ > floor (${lower_buf:.2f}) & 20d Vol {rvol:.1f}% in 22-32% (50% scale)"
            return f"High-vol shield: 20d Vol {rvol:.1f}% >= 32% -> Cash"
        if new_pos > 0.01:
            return f"Below floor but MACD Hist ({macd_hist:+.2f}) > 0 & Vol {rvol:.1f}% < 28% (50%)"
        return f"Below floor (${lower_buf:.2f}) with MACD ({macd_hist:+.2f}) / Vol {rvol:.1f}% -> Cash"

    if strategy_key == "triple_ma":
        if new_pos > prev_pos:
            return f"QQQ (${close:.2f}) > floor (${lower_buf:.2f}) & 9 EMA (${ema9:.2f}) > 30 EMA (${ema30:.2f})"
        if close <= lower_buf:
            return f"QQQ (${close:.2f}) <= {buffer_pct:.1f}% floor (${lower_buf:.2f})"
        return f"EMA death cross: 9 EMA (${ema9:.2f}) <= 30 EMA (${ema30:.2f})"

    if strategy_key == "rsi_mean_reversion":
        if new_pos > prev_pos:
            if close > lower_buf:
                return f"Bull dip buy: QQQ > floor (${lower_buf:.2f}) & Wilder RSI {rsi:.1f} < 40"
            return f"Deep oversold bounce: QQQ <= floor (${lower_buf:.2f}) & Wilder RSI {rsi:.1f} < 30"
        if close > lower_buf and rsi > 70.0:
            return f"Overbought exit: Wilder RSI {rsi:.1f} > 70"
        return f"Floor exit: QQQ (${close:.2f}) <= floor (${lower_buf:.2f}) & RSI {rsi:.1f} >= 30"

    if strategy_key == "long_short_hedged":
        if new_pos > 0:
            return f"Bull reclaim: QQQ (${close:.2f}) > 200 SMA (${sma200:.2f}) -> Long TQQQ"
        return f"Bear flip: QQQ (${close:.2f}) < {buffer_pct:.1f}% floor (${lower_buf:.2f}) -> Long SQQQ"

    return f"Position shifted from {prev_pos:+.1f} to {new_pos:+.1f} (QQQ ${close:.2f})"


def build_trade_log(
    backtest_df: pd.DataFrame,
    strategy_key: str,
    buffer_pct: float,
    tail_n: int = 15,
) -> pd.DataFrame:
    """Builds a reverse-chronological trade transition table with dynamic per-strategy reasons."""
    if backtest_df is None or backtest_df.empty or "Trade_Action" not in backtest_df.columns:
        return pd.DataFrame()

    prev_positions = backtest_df["Position"].shift(1).fillna(0.0)
    trade_mask = backtest_df["Trade_Action"] != 0
    trades = backtest_df.loc[trade_mask].tail(tail_n)
    if trades.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for dt, row in trades.iterrows():
        prev_pos = float(prev_positions.loc[dt])
        new_pos = float(row["Position"])
        records.append(
            {
                "Date": dt.strftime("%Y-%m-%d"),
                "Executed Action": _describe_trade_action(prev_pos, new_pos),
                "Target Stance": format_position_label(new_pos),
                "QQQ Price": f"${float(row['Close']):.2f}",
                "QQQ 200 SMA": f"${float(row.get('SMA_200', 0.0)):.2f}",
                f"QQQ {buffer_pct:.1f}% Floor": f"${float(row.get('SMA_200_Lower_Buffer', 0.0)):.2f}",
                "RSI (14)": f"{float(row.get('RSI', 50.0)):.1f}" if pd.notna(row.get("RSI")) else "N/A",
                "MACD Hist": f"{float(row.get('MACD_Hist', 0.0)):+.2f}" if pd.notna(row.get("MACD_Hist")) else "N/A",
                "Decision Reason": _describe_trade_reason(
                    strategy_key, row, prev_pos, new_pos, buffer_pct
                ),
            }
        )

    return pd.DataFrame(records).iloc[::-1].reset_index(drop=True)
