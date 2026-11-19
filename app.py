from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import charts
import engine
import llm_advisor
import ui_components
from strategy import StrategyRegistry

st.set_page_config(
    page_title="TQQQ Alpha-Locked Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui_components.CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_full_market_data(
    end_dt_str: str, force_refresh: bool = False
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Loads full canonical history (2008..end_dt) once for all 5 tabs."""
    return engine.load_canonical_data(
        start_date="2010-01-01",
        end_date=end_dt_str,
        force_refresh=force_refresh,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_arena(
    end_dt_str: str,
    start_dt_str: str,
    buffer_pct: float,
    init_capital: float,
    slippage_bps: float,
    risk_free_rate: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    raw_data, _ = load_full_market_data(end_dt_str, force_refresh=False)
    return engine.run_strategy_arena(
        raw_data,
        buffer_pct=buffer_pct,
        initial_capital=init_capital,
        start_date=start_dt_str,
        end_date=end_dt_str,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stress_suite(
    end_dt_str: str,
    strategy_key: str,
    buffer_pct: float,
    init_capital: float,
    slippage_bps: float,
    risk_free_rate: float,
) -> List[Dict[str, Any]]:
    raw_data, _ = load_full_market_data(end_dt_str, force_refresh=False)
    return engine.run_stress_test_suite(
        raw_data,
        strategy_key=strategy_key,
        buffer_pct=buffer_pct,
        initial_capital=init_capital,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )


def _safe_pct_dist(price: float, ref: float) -> float:
    if pd.notna(ref) and ref > 0.0 and pd.notna(price):
        return float(((price / ref) - 1.0) * 100.0)
    return 0.0


def _aligned_spot(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0
    if ts in df.index:
        return float(df.loc[ts, "Close"])
    val = df["Close"].asof(ts)
    return float(val) if pd.notna(val) else float(df["Close"].iloc[-1])


def _init_session_state() -> None:
    defaults = {
        "start_date_val": date(2010, 1, 1),
        "end_date_val": date.today(),
        "ai_analysis": None,
        "ai_chat_history": [],
        "portfolio_capital": 50000.0,
        "force_refresh_once": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    _init_session_state()
    today_dt = date.today()

    with st.sidebar:
        st.markdown("### ⚡ **Strategy Engine Config**")
        strategy_keys = list(StrategyRegistry.STRATEGIES.keys())
        selected_strat_key = st.selectbox(
            "Select Strategy Algorithm",
            options=strategy_keys,
            format_func=lambda k: StrategyRegistry.STRATEGIES[k],
            index=0,
        )

        st.markdown("---")
        buffer_val = st.slider(
            "QQQ Price Buffer (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Hysteresis Band on QQQ 200-day SMA to filter noise and reduce whipsaw churn.",
        )

        st.markdown("---")
        preset_choice = st.selectbox(
            "Select Historical Period",
            [
                "Full History (2010 - Present)",
                "Past 5 Years (2021 - Present)",
                "Past 3 Years (2023 - Present)",
                "Past 1 Year (Last 12 Months)",
                "2022 Fed Rate Hike Bear Market",
                "2020 Covid Shock & Monster Recovery",
                "Custom Range",
            ],
            index=0,
        )

        preset_ranges = {
            "Full History (2010 - Present)": (date(2010, 1, 1), today_dt),
            "Past 5 Years (2021 - Present)": (date(2021, 1, 1), today_dt),
            "Past 3 Years (2023 - Present)": (date(2023, 1, 1), today_dt),
            "Past 1 Year (Last 12 Months)": (today_dt - timedelta(days=365), today_dt),
            "2022 Fed Rate Hike Bear Market": (date(2022, 1, 1), date(2022, 12, 31)),
            "2020 Covid Shock & Monster Recovery": (date(2020, 1, 1), date(2020, 12, 31)),
        }
        if preset_choice in preset_ranges:
            start_d, end_d = preset_ranges[preset_choice]
        else:
            start_d = st.session_state.start_date_val
            end_d = st.session_state.end_date_val

        c_cal1, c_cal2 = st.columns(2)
        with c_cal1:
            sel_start_date = st.date_input(
                "Start Date", value=start_d, min_value=date(2010, 1, 1), max_value=today_dt
            )
        with c_cal2:
            sel_end_date = st.date_input(
                "End Date", value=end_d, min_value=date(2010, 1, 1), max_value=today_dt
            )

        st.session_state.start_date_val = sel_start_date
        st.session_state.end_date_val = sel_end_date

        init_capital = st.number_input(
            "Starting Capital ($)",
            min_value=100.0,
            value=float(st.session_state.portfolio_capital),
            step=10000.0,
        )
        st.session_state.portfolio_capital = init_capital

        with st.expander("Execution Friction (Optional)", expanded=False):
            slippage_bps = st.number_input(
                "Slippage / Cost (bps per turnover)", min_value=0.0, max_value=100.0, value=0.0, step=1.0
            )
            rf_pct = st.number_input(
                "Cash Risk-Free Rate (% annual)", min_value=0.0, max_value=15.0, value=0.0, step=0.5
            )
        risk_free_rate = rf_pct / 100.0

        st.markdown("---")
        model_options = llm_advisor.get_available_models()
        model_choice = st.selectbox("Gemini Model", model_options, index=0)

        st.markdown("---")
        if st.button("🚀 Refresh Data & Re-Run", width="stretch"):
            load_full_market_data.clear()
            cached_arena.clear()
            cached_stress_suite.clear()
            st.session_state.force_refresh_once = True
            st.rerun()

    st.markdown(
        ui_components.render_header_html(
            model_choice=model_choice,
            ai_analysis=st.session_state.ai_analysis,
            api_configured=llm_advisor.is_api_configured(),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui_components.render_architecture_banner_html(buffer_val, selected_strat_key),
        unsafe_allow_html=True,
    )

    if sel_start_date > sel_end_date:
        st.error(
            f"Invalid date range: Start Date ({sel_start_date}) must be on or before End Date ({sel_end_date})."
        )
        st.stop()

    start_date_str = sel_start_date.strftime("%Y-%m-%d")
    end_date_str = sel_end_date.strftime("%Y-%m-%d")
    force_flag = bool(st.session_state.pop("force_refresh_once", False))

    with st.spinner("Loading canonical price feeds for QQQ, TQQQ, SQQQ..."):
        raw_data, load_errors = load_full_market_data(end_date_str, force_refresh=force_flag)

    if load_errors:
        st.warning(
            "Data feed notices: " + "; ".join(f"{k}: {v}" for k, v in load_errors.items())
        )

    metrics, backtest_df = engine.run_backtest_pipeline(
        raw_data,
        strategy_key=selected_strat_key,
        buffer_pct=buffer_val,
        initial_capital=init_capital,
        start_date=start_date_str,
        end_date=end_date_str,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )

    if not metrics or "Error" in metrics or backtest_df is None or backtest_df.empty:
        err_msg = metrics.get("Error") if isinstance(metrics, dict) else "No backtest data available."
        st.error(f"Backtest could not run for {start_date_str} to {end_date_str}: {err_msg}")
        st.stop()

    last_ts = backtest_df.index[-1]
    last_row = backtest_df.iloc[-1]
    as_of_date = last_ts.strftime("%Y-%m-%d")

    qqq_close = float(last_row["Close"])
    sma200 = float(last_row["SMA_200"]) if pd.notna(last_row.get("SMA_200")) else 0.0
    sma50 = float(last_row["SMA_50"]) if pd.notna(last_row.get("SMA_50")) else 0.0
    ema10 = float(last_row["EMA_10"]) if pd.notna(last_row.get("EMA_10")) else 0.0
    ema9 = float(last_row["EMA_9"]) if pd.notna(last_row.get("EMA_9")) else 0.0
    ema30 = float(last_row["EMA_30"]) if pd.notna(last_row.get("EMA_30")) else 0.0
    lower_buffer = (
        float(last_row["SMA_200_Lower_Buffer"])
        if pd.notna(last_row.get("SMA_200_Lower_Buffer"))
        else sma200 * (1.0 - buffer_val / 100.0)
    )
    macd_val = float(last_row["MACD"]) if pd.notna(last_row.get("MACD")) else 0.0
    macd_sig = float(last_row["MACD_Signal"]) if pd.notna(last_row.get("MACD_Signal")) else 0.0
    macd_hist = float(last_row["MACD_Hist"]) if pd.notna(last_row.get("MACD_Hist")) else 0.0
    rsi_val = float(last_row["RSI"]) if pd.notna(last_row.get("RSI")) else 50.0
    rvol20 = float(last_row["Realized_Vol_20"]) if pd.notna(last_row.get("Realized_Vol_20")) else 20.0
    curr_pos = float(last_row.get("Position", 0.0))

    # Align spot prices strictly to backtest_df.index[-1]
    tqqq_close = _aligned_spot(raw_data.get("TQQQ"), last_ts)
    sqqq_close = _aligned_spot(raw_data.get("SQQQ"), last_ts)

    dist_200 = _safe_pct_dist(qqq_close, sma200)
    dist_50 = _safe_pct_dist(qqq_close, sma50)
    dist_lower_buffer = _safe_pct_dist(qqq_close, lower_buffer)
    strat_display_name = StrategyRegistry.STRATEGIES.get(selected_strat_key, selected_strat_key)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📡 Live Buy/Sell Decision Terminal",
            "⚔️ Multi-Strategy Comparative Arena",
            "🧪 Strategy Backtester & Log Curves",
            "🧠 Gemini Strategic Advisor & Co-Pilot",
            "📊 Multi-Regime Historical Stress Test",
        ]
    )

    with tab1:
        st.markdown(
            ui_components.render_signal_banner_html(
                curr_pos,
                strat_display_name,
                qqq_close,
                sma200,
                lower_buffer,
                dist_200,
                dist_lower_buffer,
                buffer_val,
                tqqq_close,
                sqqq_close,
                as_of_date,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui_components.render_timeframe_scorecard_html(metrics, init_capital),
            unsafe_allow_html=True,
        )

        entry_html, exit_html = ui_components.render_strategy_playbook_html(
            selected_strat_key,
            sma200,
            lower_buffer,
            buffer_val,
            tqqq_close,
            sqqq_close,
            ema10,
            ema9,
            ema30,
            rsi_val,
            rvol20,
        )
        c_rule1, c_rule2 = st.columns(2)
        with c_rule1:
            st.markdown(entry_html, unsafe_allow_html=True)
        with c_rule2:
            st.markdown(exit_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        spot_cards = ui_components.render_spot_price_cards_html(
            qqq_close,
            sma200,
            lower_buffer,
            buffer_val,
            tqqq_close,
            sqqq_close,
            curr_pos,
            dist_lower_buffer,
            as_of_date,
        )
        for col, card_html in zip(st.columns(4), spot_cards):
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        fig_price = charts.build_signal_candlestick_chart(
            backtest_df,
            metrics["Effective Start Date"],
            metrics["Effective End Date"],
            buffer_val,
            strategy_key=selected_strat_key,
        )
        st.plotly_chart(fig_price, width="stretch", config={"scrollZoom": True, "displayModeBar": True})

        st.markdown("### 📜 **Recent Trade Execution History Log**")
        trade_log_df = engine.build_trade_log(backtest_df, selected_strat_key, buffer_val, tail_n=15)
        if not trade_log_df.empty:
            st.dataframe(trade_log_df, width="stretch", hide_index=True)
        else:
            st.info("No position transitions occurred during the selected period.")

    with tab2:
        st.markdown("### ⚔️ **Multi-Strategy Quantitative Benchmark Arena**")
        st.markdown(
            f"All 6 strategies evaluated over **{metrics['Effective Start Date']} to {metrics['Effective End Date']}** "
            f"(Buffer: **{buffer_val:.1f}%**, Capital: **${init_capital:,.0f}**):"
        )
        arena_results, arena_curves = cached_arena(
            end_date_str, start_date_str, buffer_val, init_capital, slippage_bps, risk_free_rate
        )
        st.dataframe(pd.DataFrame(arena_results), width="stretch", hide_index=True)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        fig_arena = charts.build_arena_comparison_chart(arena_curves, backtest_df)
        st.plotly_chart(fig_arena, width="stretch", config={"scrollZoom": True, "displayModeBar": True})

    with tab3:
        st.markdown(
            f"### 🧪 **Active Strategy Backtest ({metrics['Effective Start Date']} to {metrics['Effective End Date']} • Buffer: {buffer_val:.1f}%)**"
        )
        kpi_cards = ui_components.render_backtest_kpi_cards_html(metrics, init_capital)
        for col, card_html in zip(st.columns(5), kpi_cards):
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(
            charts.build_equity_curve_chart(backtest_df, buffer_val),
            width="stretch",
            config={"scrollZoom": True, "displayModeBar": True},
        )
        st.plotly_chart(
            charts.build_drawdown_chart(backtest_df),
            width="stretch",
            config={"scrollZoom": True},
        )

    with tab4:
        st.markdown(f"### 🧠 **Gemini Strategic Intelligence ({model_choice})**")
        market_ctx = {
            "date": as_of_date,
            "strategy_name": strat_display_name,
            "qqq_price": qqq_close,
            "tqqq_price": tqqq_close,
            "sqqq_price": sqqq_close,
            "sma_200": sma200,
            "sma_50": sma50,
            "ema_10": ema10,
            "buffer_pct": float(buffer_val),
            "lower_buffer": lower_buffer,
            "dist_200_sma_pct": dist_200,
            "dist_50_sma_pct": dist_50,
            "dist_lower_buffer_pct": dist_lower_buffer,
            "macd": macd_val,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
            "rsi": rsi_val,
            "rvol_20": rvol20,
            "current_stance": engine.format_position_label(curr_pos),
        }

        col_ai1, col_ai2 = st.columns([1.3, 1])
        with col_ai1:
            st.markdown("#### ⚡ **Macro Regime Assessment**")
            if st.button("🤖 Run Gemini Leveraged Regime Audit", width="stretch"):
                with st.spinner(f"{model_choice} is evaluating Nasdaq regime, volatility drag & triggers..."):
                    st.session_state.ai_analysis = llm_advisor.analyze_market_regime(
                        market_ctx, backtest_df.tail(60), model_name=model_choice
                    )

            if st.session_state.ai_analysis:
                st.markdown(
                    ui_components.render_gemini_analysis_card_html(
                        st.session_state.ai_analysis, lower_buffer
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info("Click the button above to run on-demand Gemini regime analysis.")

        with col_ai2:
            st.markdown("#### 💬 **Ask Gemini Co-Pilot**")
            user_q = st.text_input(
                "Ask anything about TQQQ timing, QQQ criteria, or buffer sizing:",
                placeholder="e.g. Why is QQQ used for signals instead of TQQQ itself?",
            )
            if st.button("Submit Question", key="btn_ask_tqqq_copilot", width="stretch"):
                if user_q.strip():
                    with st.spinner(f"{model_choice} is thinking..."):
                        res = llm_advisor.ask_gemini_copilot(
                            user_q.strip(),
                            market_ctx,
                            model_name=model_choice,
                            recent_history=backtest_df.tail(20),
                        )
                        st.session_state.ai_chat_history.append(
                            {
                                "q": user_q.strip(),
                                "a": res["text"],
                                "model": res["model_used"],
                                "is_fallback": res["is_fallback"],
                            }
                        )

            for chat in reversed(st.session_state.ai_chat_history[-5:]):
                label = "Offline Fallback" if chat.get("is_fallback") else chat.get("model", model_choice)
                st.markdown(f"**Q:** *{chat['q']}*")
                st.markdown(f"**{label}:** {chat['a']}")
                st.markdown("---")

    with tab5:
        st.markdown(
            f"### 📊 **Multi-Regime Historical Stress Test ({strat_display_name} • Buffer: {buffer_val:.1f}% • Capital: ${init_capital:,.0f})**"
        )
        stress_results = cached_stress_suite(
            end_date_str, selected_strat_key, buffer_val, init_capital, slippage_bps, risk_free_rate
        )
        if stress_results:
            st.dataframe(pd.DataFrame(stress_results), width="stretch", hide_index=True)
        else:
            st.warning("No stress-test regimes had sufficient overlapping data.")


if __name__ == "__main__":
    main()
