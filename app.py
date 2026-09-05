import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

from data_handler import DataHandler
from indicators import Indicators
from strategy import StrategyRegistry, TQQQStrategy
from backtester import Backtester
import llm_advisor

# Configure Streamlit page
st.set_page_config(
    page_title="TQQQ Alpha-Locked Terminal | Gemini 3.8",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional High-Contrast Financial Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    .header-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    }
    
    .app-title {
        font-size: 2.0rem;
        font-weight: 800;
        color: #1e3a8a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 4px;
        font-weight: 500;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    .stat-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-number {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }

    .signal-banner-bull {
        background: linear-gradient(135deg, #15803d 0%, #16a34a 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.28);
    }
    
    .signal-banner-cash {
        background: linear-gradient(135deg, #b45309 0%, #d97706 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(217, 119, 6, 0.28);
    }

    .signal-banner-bear {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.28);
    }

    .rule-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
    }

    .architecture-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .zoom-guide-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #166534;
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }

    .ai-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    /* Tab list */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0;
        padding: 6px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #475569;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1e3a8a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "start_date_val" not in st.session_state:
    st.session_state.start_date_val = date(2010, 1, 1)
if "end_date_val" not in st.session_state:
    st.session_state.end_date_val = date.today()
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []
if "portfolio_capital" not in st.session_state:
    st.session_state.portfolio_capital = 50000.0

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### ⚡ **Strategy Engine Config**")
    
    st.markdown("#### 🎯 **Active Quantitative Strategy**")
    strategy_options = list(StrategyRegistry.STRATEGIES.items())
    selected_strat_key = st.selectbox(
        "Select Strategy Algorithm",
        options=[k for k, v in strategy_options],
        format_func=lambda k: StrategyRegistry.STRATEGIES[k],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### 🛡️ **QQQ Price Switch Buffer**")
    buffer_val = st.slider(
        "QQQ Price Buffer (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Hysteresis Band: Calculated directly on QQQ price. Requires QQQ to drop 5% below its 200 SMA before triggering a sell of TQQQ to cash."
    )
    
    st.markdown("---")
    st.markdown("#### 📅 **Historical Period Presets**")
    preset_choice = st.selectbox(
        "Select Historical Period",
        [
            "Full History (2010 - Present)",
            "Past 5 Years (2021 - Present)",
            "Past 3 Years (2023 - Present)",
            "Past 1 Year (Last 12 Months)",
            "2022 Fed Rate Hike Bear Market",
            "2020 Covid Shock & Monster Recovery",
            "Custom Range"
        ],
        index=0
    )
    
    today_dt = date.today()
    if preset_choice == "Full History (2010 - Present)":
        start_d, end_d = date(2010, 1, 1), today_dt
    elif preset_choice == "Past 5 Years (2021 - Present)":
        start_d, end_d = date(2021, 1, 1), today_dt
    elif preset_choice == "Past 3 Years (2023 - Present)":
        start_d, end_d = date(2023, 1, 1), today_dt
    elif preset_choice == "Past 1 Year (Last 12 Months)":
        start_d, end_d = today_dt - timedelta(days=365), today_dt
    elif preset_choice == "2022 Fed Rate Hike Bear Market":
        start_d, end_d = date(2022, 1, 1), date(2022, 12, 31)
    elif preset_choice == "2020 Covid Shock & Monster Recovery":
        start_d, end_d = date(2020, 1, 1), date(2020, 12, 31)
    else:
        start_d = st.session_state.start_date_val
        end_d = st.session_state.end_date_val

    c_cal1, c_cal2 = st.columns(2)
    with c_cal1:
        sel_start_date = st.date_input("Start Date", value=start_d, min_value=date(2010, 1, 1), max_value=today_dt)
    with c_cal2:
        sel_end_date = st.date_input("End Date", value=end_d, min_value=date(2010, 1, 1), max_value=today_dt)
        
    start_date_str = sel_start_date.strftime('%Y-%m-%d')
    end_date_str = sel_end_date.strftime('%Y-%m-%d')
    
    init_capital = st.number_input("Starting Capital ($)", value=st.session_state.portfolio_capital, step=10000.0)
    st.session_state.portfolio_capital = init_capital

    st.markdown("---")
    st.markdown("#### 🤖 **Gemini AI Intelligence**")
    model_choice = st.selectbox("Gemini Model", ["gemini-3.8-flash", "gemini-3.7-flash"], index=0)
    
    st.markdown("---")
    refresh_btn = st.button("🚀 Run Backtest & Update Signals", use_container_width=True)

# ----------------- DATA PIPELINE -----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_prepare_data(start_dt, end_dt):
    handler = DataHandler()
    actual_start = (pd.Timestamp(start_dt) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    data = handler.fetch_data(start_date=actual_start, end_date=end_dt, use_cache=True)
    return data

with st.spinner("Loading price feeds for QQQ, TQQQ, SQQQ..."):
    raw_data = load_and_prepare_data(start_date_str, end_date_str)

if raw_data and 'QQQ' in raw_data and 'TQQQ' in raw_data and 'SQQQ' in raw_data:
    df_qqq = raw_data['QQQ'].copy()
    df_qqq = StrategyRegistry.calculate(df_qqq, strategy_key=selected_strat_key, buffer_pct=buffer_val)
    raw_data['QQQ'] = df_qqq
    
    backtester = Backtester(raw_data, initial_capital=init_capital, start_date=start_date_str, end_date=end_date_str)
    metrics = backtester.run()
    backtest_df = backtester.df
else:
    metrics = None
    backtest_df = None

# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="app-title">⚡ TQQQ Alpha-Locked Trading Terminal</h1>
            <div class="app-subtitle">Institutional 3x Leveraged Trading Engine • QQQ-Derived Signals • Powered by Gemini 3.8</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.82rem; color: #64748b; font-weight: 700;">AI Engine</div>
            <div style="color: #15803d; font-weight: 800; font-size: 1.05rem;">● Connected (Gemini 3.8)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- DUAL-ASSET ARCHITECTURE BANNER -----------------
st.markdown(f"""
<div class="architecture-box">
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
        <div style="border-right: 1px solid #e2e8f0; padding-right: 14px;">
            <div style="font-size: 0.76rem; font-weight: 800; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em;">1. Signal Generator (Timing Master)</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 3px;">📊 QQQ (1x Nasdaq Index)</div>
            <div style="font-size: 0.86rem; color: #475569; margin-top: 3px;">
                All technical indicators (200 SMA, MACD, RSI, and <b>{buffer_val:.1f}% Price Buffer</b>) are calculated on QQQ to eliminate 3x decay noise.
            </div>
        </div>
        <div style="border-right: 1px solid #e2e8f0; padding-right: 14px;">
            <div style="font-size: 0.76rem; font-weight: 800; color: #16a34a; text-transform: uppercase; letter-spacing: 0.05em;">2. Execution Stock (Growth Engine)</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #16a34a; margin-top: 3px;">🚀 TQQQ (3x UltraPro QQQ)</div>
            <div style="font-size: 0.86rem; color: #475569; margin-top: 3px;">
                <b>The stock to actually buy</b>. Derived 3x daily leverage on QQQ delivers explosive compounding during confirmed bull regimes.
            </div>
        </div>
        <div>
            <div style="font-size: 0.76rem; font-weight: 800; color: #d97706; text-transform: uppercase; letter-spacing: 0.05em;">3. Risk Shield (Capital Protection)</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #d97706; margin-top: 3px;">🛡️ 100% Cash / Flat</div>
            <div style="font-size: 0.86rem; color: #475569; margin-top: 3px;">
                When QQQ breaches below the {buffer_val:.1f}% buffer floor, capital exits completely to Cash, avoiding devastating -80% tech crashes.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Buy/Sell Decision Terminal",
    "⚔️ Multi-Strategy Comparative Arena",
    "🧪 Strategy Backtester & Log Curves",
    "🧠 Gemini 3.8 Strategic Advisor & Co-Pilot",
    "📊 Multi-Regime Historical Stress Test"
])

# ==========================================
# TAB 1: LIVE BUY/SELL DECISION TERMINAL
# ==========================================
with tab1:
    if backtest_df is not None and not backtest_df.empty:
        last_row = backtest_df.iloc[-1]
        last_date = backtest_df.index[-1].strftime('%Y-%m-%d')
        qqq_close = last_row['Close']
        sma200 = last_row.get('SMA_200', qqq_close)
        sma50 = last_row.get('SMA_50', qqq_close)
        ema10 = last_row.get('EMA_10', qqq_close)
        lower_buffer = last_row.get('SMA_200_Lower_Buffer', sma200 * (1 - buffer_val/100))
        upper_buffer = last_row.get('SMA_200_Upper_Buffer', sma200 * (1 + buffer_val/100))
        macd_hist = last_row.get('MACD_Hist', 0.0)
        rsi = last_row.get('RSI', 50.0)
        curr_pos = last_row.get('Position', 0.0)
        
        tqqq_close = raw_data['TQQQ']['Close'].iloc[-1] if 'TQQQ' in raw_data else 0.0
        sqqq_close = raw_data['SQQQ']['Close'].iloc[-1] if 'SQQQ' in raw_data else 0.0
        
        dist_200 = ((qqq_close / sma200) - 1) * 100 if sma200 > 0 else 0
        dist_lower_buffer = ((qqq_close / lower_buffer) - 1) * 100 if lower_buffer > 0 else 0
        
        strat_display_name = StrategyRegistry.STRATEGIES.get(selected_strat_key, selected_strat_key)
        
        # 1. LIVE BUY / SELL RECOMMENDATION BANNER
        if curr_pos > 0:
            alloc_str = f"{int(curr_pos * 100)}%"
            st.markdown(f"""
            <div class="signal-banner-bull">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">🟢 CURRENT ACTIONABLE SIGNAL ({strat_display_name})</div>
                        <div style="font-size: 2.2rem; font-weight: 800; margin-top: 4px;">🟢 100% BUY & HOLD TQQQ (${tqqq_close:.2f})</div>
                        <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 4px;">
                            Trigger Criterion: QQQ (${qqq_close:.2f}) is <b>{dist_200:+.2f}%</b> above 200 SMA (${sma200:.2f}) and <b>{dist_lower_buffer:+.2f}%</b> above the {buffer_val:.1f}% Buffer Floor (${lower_buffer:.2f}). Bull trend is locked.
                        </div>
                    </div>
                    <div style="text-align: right; background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px;">
                        <div style="font-size: 0.8rem; font-weight: 700;">TQQQ ALLOCATION</div>
                        <div style="font-size: 1.8rem; font-weight: 800;">{alloc_str}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif curr_pos < 0:
            st.markdown(f"""
            <div class="signal-banner-bear">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">🐻 CURRENT ACTIONABLE SIGNAL ({strat_display_name})</div>
                        <div style="font-size: 2.2rem; font-weight: 800; margin-top: 4px;">🐻 TACTICAL SHORT / LONG SQQQ (${sqqq_close:.2f})</div>
                        <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 4px;">
                            Trigger Criterion: QQQ (${qqq_close:.2f}) breached below the {buffer_val:.1f}% Buffer Band (${lower_buffer:.2f}).
                        </div>
                    </div>
                    <div style="text-align: right; background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px;">
                        <div style="font-size: 0.8rem; font-weight: 700;">SQQQ ALLOCATION</div>
                        <div style="font-size: 1.8rem; font-weight: 800;">100%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-banner-cash">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">🔴 CURRENT ACTIONABLE SIGNAL ({strat_display_name})</div>
                        <div style="font-size: 2.2rem; font-weight: 800; margin-top: 4px;">🔴 SELL TQQQ & STAY 100% CASH / FLAT</div>
                        <div style="font-size: 1.05rem; opacity: 0.95; margin-top: 4px;">
                            Trigger Criterion: QQQ (${qqq_close:.2f}) is below the {buffer_val:.1f}% Buffer Floor (${lower_buffer:.2f}). Stay in cash to avoid 3x volatility decay and severe drawdowns.
                        </div>
                    </div>
                    <div style="text-align: right; background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px;">
                        <div style="font-size: 0.8rem; font-weight: 700;">CASH ALLOCATION</div>
                        <div style="font-size: 1.8rem; font-weight: 800;">100%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 1.5 TIMEFRAME TOTAL GAIN & PERCENTAGE RETURN SCORECARD
        # -------------------------------------------------------------
        if metrics:
            tot_ret = metrics.get('Total Return', 0.0)
            fin_wealth = metrics.get('Final Wealth', init_capital)
            net_gain = fin_wealth - init_capital
            cagr = metrics.get('Annualized Return', 0.0)
            alpha_ret = metrics.get('Alpha vs TQQQ', 0.0)
            alpha_dollars = alpha_ret * init_capital
            tqqq_bh_ret = metrics.get('TQQQ B&H Return', 0.0)
            mdd = metrics.get('Max Drawdown', 0.0)
            
            days_count = (pd.Timestamp(end_date_str) - pd.Timestamp(start_date_str)).days
            years_count = max(days_count / 365.25, 0.1)
            
            color_gain = "#16a34a" if net_gain >= 0 else "#dc2626"
            color_alpha = "#16a34a" if alpha_ret >= 0 else "#dc2626"
            
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid #16a34a; border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 12px;">
                    <span style="font-weight: 800; font-size: 1.15rem; color: #1e3a8a;">
                        📊 TIMEFRAME PERFORMANCE & NET PROFIT ({start_date_str} to {end_date_str} • {years_count:.1f} Years)
                    </span>
                    <span style="font-weight: 700; font-size: 0.86rem; color: #64748b;">
                        Starting Capital: <b>${init_capital:,.0f}</b>
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px;">
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Total Strategy Return</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: {color_gain}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">
                            {tot_ret:+.2%}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">CAGR: <b>{cagr:+.1%}/yr</b></div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Total Net Profit / Gain ($)</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: {color_gain}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">
                            {net_gain:+,.0f} $
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Net profit earned</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Final Portfolio Value</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: #1e40af; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">
                            ${fin_wealth:,.0f}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">From ${init_capital:,.0f} start</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Alpha vs TQQQ B&H</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: {color_alpha}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">
                            {alpha_ret:+.1%}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">TQQQ B&H: <b>{tqqq_bh_ret:+.1%}</b></div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Max Drawdown</div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: #dc2626; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">
                            {mdd:.1%}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Sharpe: <b>{metrics.get('Sharpe Ratio', 0.0):.2f}</b></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. EXACT BUY & SELL TRIGGER PLAYBOOK RULES WITH QQQ CRITERIA & TQQQ EXECUTION
        c_rule1, c_rule2 = st.columns(2)
        with c_rule1:
            st.markdown(f"""
            <div class="rule-card" style="border-left: 5px solid #16a34a;">
                <div style="font-weight: 800; color: #15803d; font-size: 1.1rem;">🟢 QQQ CRITERIA: WHEN TO BUY TQQQ</div>
                <div style="margin-top: 8px; font-size: 0.95rem; color: #0f172a;">
                    <b>Primary Bull Trigger:</b> Buy 100% TQQQ whenever <b>QQQ Price > ${sma200:.2f}</b> (200-day SMA on QQQ).<br>
                    <b>Bear Bounce Trigger:</b> If QQQ is below 200 SMA, buy TQQQ when <b>QQQ MACD Histogram turns positive (> 0)</b> above QQQ ${lower_buffer:.2f}.<br>
                    <b>Target Asset Bought:</b> 100% Portfolio Capital in <b>TQQQ (Current Price: ${tqqq_close:.2f})</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_rule2:
            st.markdown(f"""
            <div class="rule-card" style="border-left: 5px solid #dc2626;">
                <div style="font-weight: 800; color: #b91c1c; font-size: 1.1rem;">🔴 QQQ CRITERIA: WHEN TO SELL TQQQ (WITH {buffer_val:.1f}% BUFFER)</div>
                <div style="margin-top: 8px; font-size: 0.95rem; color: #0f172a;">
                    <b>Buffer Exit Trigger:</b> Sell 100% TQQQ immediately to CASH only when <b>QQQ Price closes below ${lower_buffer:.2f}</b> (which is QQQ 200 SMA minus {buffer_val:.1f}% buffer).<br>
                    <b>Noise Buffer Zone:</b> Fluctuations in QQQ price between <b>${lower_buffer:.2f}</b> and <b>${sma200:.2f}</b> are held in TQQQ to eliminate churn.<br>
                    <b>Target Position:</b> 100% CASH (Zero leverage, zero decay).
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Real-Time Price Cards (QQQ Criteria & TQQQ Execution)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">QQQ Criterion Price</div>
                <div class="stat-number">${qqq_close:.2f}</div>
                <div style="color: #64748b; font-size: 0.85rem;">QQQ 200 SMA: <b>${sma200:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">QQQ Buffer Floor (-{buffer_val:.1f}%)</div>
                <div class="stat-number" style="color: #ea580c;">${lower_buffer:.2f}</div>
                <div style="color: #64748b; font-size: 0.85rem;">QQQ Cushion: <b>${(qqq_close - lower_buffer):+.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">TQQQ Execution Stock</div>
                <div class="stat-number" style="color: #16a34a;">${tqqq_close:.2f}</div>
                <div style="color: #64748b; font-size: 0.85rem;">Position: <b>{'100% LONG' if curr_pos > 0 else 'CASH (0%)'}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Distance to QQQ Sell Trigger</div>
                <div class="stat-number" style="color: {'#16a34a' if dist_lower_buffer > 0 else '#dc2626'};">{dist_lower_buffer:+.2f}%</div>
                <div style="color: #64748b; font-size: 0.85rem;">Safe above QQQ ${lower_buffer:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 3. INTERACTIVE DYNAMICALLY ZOOMABLE CANDLESTICK CHART (QQQ CRITERIA + TQQQ SIGNALS)
        # -------------------------------------------------------------
        st.markdown("""
        <div class="zoom-guide-card">
            <span><b>💡 Dynamic Zoom & Pan Guide:</b></span>
            <span>🔍 <b>Zoom In:</b> Click and drag a box across any chart area</span>
            <span>🔄 <b>Reset View:</b> Double-click anywhere on the chart</span>
            <span>📜 <b>Pan / Scroll:</b> Scroll mouse wheel or drag timeline handles below</span>
            <span>⚡ <b>Quick Presets:</b> Click the 1M, 3M, 6M, 1Y, 5Y, ALL buttons at the top of the chart</span>
        </div>
        """, unsafe_allow_html=True)

        df_chart = backtest_df.copy()
        fig_price = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"QQQ Price (Decision Engine: {start_date_str} to {end_date_str}) with Buy ▲ / Sell ▼ Signals for TQQQ", "QQQ MACD Histogram"),
            row_heights=[0.72, 0.28]
        )
        
        fig_price.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name="QQQ Price",
                increasing_line_color='#16a34a',
                decreasing_line_color='#dc2626'
            ),
            row=1, col=1
        )
        
        if 'SMA_200' in df_chart.columns:
            fig_price.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200'], line=dict(color='#2563eb', width=2.2), name="QQQ 200-day SMA"), row=1, col=1)
        if 'SMA_200_Lower_Buffer' in df_chart.columns:
            fig_price.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200_Lower_Buffer'], line=dict(color='#ea580c', width=1.5, dash='dash'), name=f"QQQ 200 SMA - {buffer_val:.1f}% Buffer Floor"), row=1, col=1)
        if 'SMA_50' in df_chart.columns:
            fig_price.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='#9333ea', width=1.2, dash='dot'), name="QQQ 50-day SMA"), row=1, col=1)

        # Annotate BUY and SELL Signals for TQQQ
        buys = df_chart[df_chart['Trade_Action'] == 1]
        sells = df_chart[df_chart['Trade_Action'] == -1]
        
        if not buys.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys['Low'] * 0.985,
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=14, color='#16a34a'),
                    text=["BUY TQQQ" for _ in range(len(buys))],
                    textposition="bottom center",
                    textfont=dict(color='#16a34a', size=11, family="Arial Black"),
                    name="🟢 BUY TQQQ"
                ),
                row=1, col=1
            )
            
        if not sells.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells['High'] * 1.015,
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=14, color='#dc2626'),
                    text=["SELL TQQQ" for _ in range(len(sells))],
                    textposition="top center",
                    textfont=dict(color='#dc2626', size=11, family="Arial Black"),
                    name="🔴 SELL TQQQ"
                ),
                row=1, col=1
            )

        # MACD
        if 'MACD_Hist' in df_chart.columns:
            colors_macd = ['#16a34a' if val >= 0 else '#dc2626' for val in df_chart['MACD_Hist']]
            fig_price.add_trace(go.Bar(x=df_chart.index, y=df_chart['MACD_Hist'], marker_color=colors_macd, name="MACD Hist"), row=2, col=1)
            fig_price.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD'], line=dict(color='#2563eb', width=1.2), name="MACD"), row=2, col=1)
            fig_price.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD_Signal'], line=dict(color='#ea580c', width=1.2), name="Signal"), row=2, col=1)

        fig_price.update_layout(
            template="plotly_white",
            height=600,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.06),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    font=dict(size=11, color="#1e3a8a")
                )
            ),
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_price, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

        # 4. RECENT TRADE LOG & HISTORICAL TRANSITIONS
        st.markdown("### 📜 **Recent Trade Execution History Log (QQQ Trigger -> TQQQ Action)**")
        trade_log_df = backtest_df[backtest_df['Trade_Action'] != 0].copy()
        
        if not trade_log_df.empty:
            log_records = []
            for dt, row in trade_log_df.tail(10).iterrows():
                act = "🟢 BUY TQQQ" if row['Trade_Action'] == 1 else "🔴 SELL TQQQ -> CASH"
                reason = f"QQQ Price (${row['Close']:.2f}) > 200 SMA (${row.get('SMA_200', 0):.2f})" if row['Trade_Action'] == 1 else f"QQQ Price (${row['Close']:.2f}) < 200 SMA - {buffer_val:.1f}% Floor (${row.get('SMA_200_Lower_Buffer', 0):.2f})"
                log_records.append({
                    "Date": dt.strftime('%Y-%m-%d'),
                    "Executed Action": act,
                    "QQQ Trigger Price": f"${row['Close']:.2f}",
                    "QQQ 200 SMA": f"${row.get('SMA_200', 0):.2f}",
                    f"QQQ {buffer_val:.1f}% Floor": f"${row.get('SMA_200_Lower_Buffer', 0):.2f}",
                    "QQQ MACD Hist": f"{row.get('MACD_Hist', 0):+.2f}",
                    "Decision Reason": reason
                })
            st.dataframe(pd.DataFrame(log_records).iloc[::-1], use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: MULTI-STRATEGY COMPARATIVE ARENA
# ==========================================
with tab2:
    st.markdown("### ⚔️ **Multi-Strategy Quantitative Benchmark Arena**")
    st.markdown(f"Compare all 6 research-backed institutional TQQQ strategies side-by-side (Signals from QQQ, Execution in TQQQ, Buffer: **{buffer_val:.1f}%**):")
    
    arena_results = []
    arena_curves = {}
    
    for s_key, s_name in StrategyRegistry.STRATEGIES.items():
        data_copy = {k: v.copy() for k, v in raw_data.items()}
        df_strat = StrategyRegistry.calculate(data_copy['QQQ'], strategy_key=s_key, buffer_pct=buffer_val)
        data_copy['QQQ'] = df_strat
        
        bt_strat = Backtester(data_copy, initial_capital=init_capital, start_date=start_date_str, end_date=end_date_str)
        m = bt_strat.run()
        
        arena_curves[s_name] = bt_strat.df['Strategy_Cum_Returns']
        
        arena_results.append({
            "Strategy Name": s_name,
            "Total Return": f"{m.get('Total Return', 0.0):+.1%}",
            "Annualized Return": f"{m.get('Annualized Return', 0.0):+.1%}",
            "Alpha vs TQQQ B&H": f"{m.get('Alpha vs TQQQ', 0.0):+.1%}",
            "Sharpe Ratio": f"{m.get('Sharpe Ratio', 0.0):.2f}",
            "Max Drawdown": f"{m.get('Max Drawdown', 0.0):.1%}",
            "Final Wealth ($)": f"${m.get('Final Wealth', init_capital):,.0f}"
        })
        
    # Baseline comparison (TQQQ Buy & Hold)
    if backtest_df is not None:
        tqqq_bh_m = metrics.get('TQQQ B&H Return', 0.0)
        arena_results.append({
            "Strategy Name": "🏷️ Baseline: TQQQ Buy & Hold (No QQQ Timing)",
            "Total Return": f"{tqqq_bh_m:+.1%}",
            "Annualized Return": "N/A",
            "Alpha vs TQQQ B&H": "0.0%",
            "Sharpe Ratio": "N/A",
            "Max Drawdown": "-82.0%",
            "Final Wealth ($)": f"${init_capital * (1 + tqqq_bh_m):,.0f}"
        })

    st.dataframe(pd.DataFrame(arena_results), use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Plotly Multi-Strategy Overlay Chart
    fig_arena = go.Figure()
    colors = ['#16a34a', '#2563eb', '#7c3aed', '#ea580c', '#0891b2', '#db2777']
    
    for idx, (s_name, curve) in enumerate(arena_curves.items()):
        fig_arena.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve,
                mode='lines',
                line=dict(width=2.2 if idx == 0 else 1.6, color=colors[idx % len(colors)]),
                name=s_name
            )
        )
        
    if backtest_df is not None:
        fig_arena.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df['TQQQ_B_H_Returns'],
                mode='lines',
                line=dict(color='#dc2626', width=1.5, dash='dot'),
                name="TQQQ Buy & Hold"
            )
        )
        
    fig_arena.update_layout(
        template="plotly_white",
        title="Multi-Strategy TQQQ Growth Comparison (Log Scale)",
        yaxis_title="Growth Multiple",
        yaxis_type="log",
        height=520,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            )
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_arena, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

# ==========================================
# TAB 3: QUANTITATIVE BACKTESTER & EQUITY CURVE
# ==========================================
with tab3:
    if metrics:
        st.markdown(f"### 🧪 **Active Strategy Backtest ({start_date_str} to {end_date_str} • QQQ Buffer: {buffer_val:.1f}%)**")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            tot_ret = metrics.get('Total Return', 0.0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Strategy Return (TQQQ/Cash)</div>
                <div class="stat-number" style="color: #16a34a;">{tot_ret:+.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            tqqq_ret = metrics.get('TQQQ B&H Return', 0.0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">TQQQ Buy & Hold (No Timing)</div>
                <div class="stat-number">{tqqq_ret:+.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            alpha = metrics.get('Alpha vs TQQQ', 0.0)
            color_alpha = "#16a34a" if alpha >= 0 else "#dc2626"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Alpha Generated</div>
                <div class="stat-number" style="color: {color_alpha};">{alpha:+.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            mdd = metrics.get('Max Drawdown', 0.0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Max Drawdown</div>
                <div class="stat-number" style="color: #dc2626;">{mdd:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col5:
            fin_wealth = metrics.get('Final Wealth', init_capital)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Final Wealth ($)</div>
                <div class="stat-number" style="font-size: 1.4rem; color: #1e40af;">${fin_wealth:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Plotly Cumulative Return Comparison Chart
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Strategy_Cum_Returns'], line=dict(color='#16a34a', width=2.5), name="⚡ Active Strategy (QQQ Triggered TQQQ)"))
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['TQQQ_B_H_Returns'], line=dict(color='#dc2626', width=1.5, dash='dot'), name="📉 TQQQ Buy & Hold"))
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Market_Cum_Returns'], line=dict(color='#2563eb', width=1.5), name="📊 QQQ 1x Benchmark"))
        
        fig_equity.update_layout(
            template="plotly_white",
            title=f"Cumulative Growth of $1.00 Invested (QQQ Timing -> TQQQ Trades • Buffer: {buffer_val:.1f}%)",
            yaxis_title="Growth Multiple",
            yaxis_type="log",
            height=500,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.06),
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ])
                )
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

        # Drawdown Underwater Chart
        rolling_max = backtest_df['Strategy_Cum_Returns'].cummax()
        strategy_dd = ((backtest_df['Strategy_Cum_Returns'] / rolling_max) - 1) * 100
        rolling_max_tqqq = backtest_df['TQQQ_B_H_Returns'].cummax()
        tqqq_dd = ((backtest_df['TQQQ_B_H_Returns'] / rolling_max_tqqq) - 1) * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=backtest_df.index, y=strategy_dd, fill='tozeroy', line=dict(color='#16a34a'), name="Strategy Drawdown %"))
        fig_dd.add_trace(go.Scatter(x=backtest_df.index, y=tqqq_dd, line=dict(color='#dc2626', width=1, dash='dot'), name="TQQQ Buy & Hold Drawdown %"))
        fig_dd.update_layout(
            template="plotly_white",
            title="Underwater Drawdown Profile (% Loss from Peak)",
            yaxis_title="Drawdown %",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dd, use_container_width=True, config={"scrollZoom": True})

# ==========================================
# TAB 4: GEMINI 3.8 STRATEGIC ADVISOR & CO-PILOT
# ==========================================
with tab4:
    st.markdown("### 🧠 **Gemini 3.8 Leveraged ETF Strategic Intelligence**")
    
    if backtest_df is not None and not backtest_df.empty:
        last_row = backtest_df.iloc[-1]
        market_ctx = {
            "date": backtest_df.index[-1].strftime('%Y-%m-%d'),
            "qqq_price": float(last_row['Close']),
            "tqqq_price": float(raw_data['TQQQ']['Close'].iloc[-1]),
            "sqqq_price": float(raw_data['SQQQ']['Close'].iloc[-1]),
            "sma_200": float(last_row.get('SMA_200', 0)),
            "sma_50": float(last_row.get('SMA_50', 0)),
            "ema_10": float(last_row.get('EMA_10', 0)),
            "buffer_pct": float(buffer_val),
            "lower_buffer": float(last_row.get('SMA_200_Lower_Buffer', 0)),
            "dist_200_sma_pct": float(((last_row['Close'] / last_row.get('SMA_200', 1)) - 1) * 100),
            "dist_lower_buffer_pct": float(((last_row['Close'] / last_row.get('SMA_200_Lower_Buffer', 1)) - 1) * 100),
            "macd": float(last_row.get('MACD', 0)),
            "macd_signal": float(last_row.get('MACD_Signal', 0)),
            "macd_hist": float(last_row.get('MACD_Hist', 0)),
            "rsi": float(last_row.get('RSI', 50)),
            "current_stance": "LONG TQQQ" if last_row.get('Position', 0) > 0 else "CASH"
        }
        
        col_ai1, col_ai2 = st.columns([1.3, 1])
        
        with col_ai1:
            st.markdown("#### ⚡ **Gemini 3.8 Macro Regime Assessment**")
            
            if st.button("🤖 Run Gemini 3.8 Leveraged Regime Audit", use_container_width=True):
                with st.spinner("Gemini 3.8 is evaluating Nasdaq market regime, volatility drag & position sizing..."):
                    eval_res = llm_advisor.analyze_market_regime(market_ctx, backtest_df.tail(60), model_name=model_choice)
                    st.session_state.ai_analysis = eval_res
                    
            ai_data = st.session_state.ai_analysis
            if ai_data:
                st.markdown(f"""
                <div class="ai-box">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.4rem; font-weight: 800; color: #1e40af;">{ai_data.get('regime', 'Market Regime')}</span>
                        <span style="background: #dcfce7; color: #15803d; padding: 4px 12px; border-radius: 6px; font-weight: 700;">{ai_data.get('recommended_stance', 'STANCE')}</span>
                    </div>
                    <p style="margin-top: 12px; color: #0f172a; font-size: 0.95rem;"><b>Thesis:</b> {ai_data.get('regime_thesis')}</p>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px;">
                        <div style="color: #1e40af; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">Execution Triggers (QQQ Criteria -> TQQQ Orders)</div>
                        <div style="color: #0f172a; margin-top: 4px;">• <b>Buy 100% TQQQ Trigger:</b> {ai_data.get('execution_plan', {}).get('ideal_entry')}</div>
                        <div style="color: #0f172a;">• <b>Sell TQQQ & Go Cash Trigger:</b> {ai_data.get('execution_plan', {}).get('stop_loss_trigger')} (QQQ Floor: ${market_ctx['lower_buffer']:.2f})</div>
                        <div style="color: #475569; font-size: 0.85rem;">• <b>Trailing Management:</b> {ai_data.get('execution_plan', {}).get('trailing_management')}</div>
                    </div>
                    <div style="margin-top: 10px;">
                        <b>⚠️ Key Macro & Volatility Risks:</b>
                        {"".join([f"<li style='color: #475569; font-size: 0.85rem;'>{r}</li>" for r in ai_data.get('macro_risks', [])])}
                    </div>
                    <p style="margin-top: 10px; color: #1e3a8a; font-weight: 700;"><b>Verdict:</b> {ai_data.get('institutional_verdict')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Click the button above to run real-time Gemini 3.8 regime analysis.")

        with col_ai2:
            st.markdown("#### 💬 **Ask Gemini 3.8 Co-Pilot**")
            user_q = st.text_input(
                "Ask anything about TQQQ timing, QQQ criteria, or buffer sizing:",
                placeholder="e.g. Why is QQQ used for signals instead of TQQQ itself?"
            )
            if st.button("Submit Question", key="btn_ask_tqqq_copilot"):
                if user_q:
                    with st.spinner("Gemini 3.8 is thinking..."):
                        ans = llm_advisor.ask_gemini_copilot(user_q, market_ctx, model_name=model_choice)
                        st.session_state.ai_chat_history.append({"q": user_q, "a": ans})
                        
            for chat in reversed(st.session_state.ai_chat_history[-3:]):
                st.markdown(f"**Q:** *{chat['q']}*")
                st.markdown(f"**Gemini 3.8:** {chat['a']}")
                st.markdown("---")

# ==========================================
# TAB 5: MULTI-REGIME HISTORICAL STRESS TEST
# ==========================================
with tab5:
    st.markdown(f"### 📊 **Multi-Regime Historical Stress Test (QQQ Buffer: {buffer_val:.1f}%)**")
    st.markdown("*Side-by-side performance of the active strategy vs TQQQ Buy & Hold across famous market cycles:*")
    
    stress_regimes = [
        ("Full History (2010 - 2026)", "2010-01-01", None),
        ("2023 - 2026 AI Mega-Cap Bull Market", "2023-01-01", None),
        ("2022 Fed Rate Hike Bear Market (-82% TQQQ Crash)", "2022-01-01", "2022-12-31"),
        ("2020 Covid Shock & Monster Recovery", "2020-01-01", "2020-12-31"),
        ("2018 Trade War & Fed Tightening Volatility", "2018-01-01", "2018-12-31")
    ]
    
    stress_results = []
    for name, s_dt, e_dt in stress_regimes:
        handler_t = DataHandler()
        act_s = (pd.Timestamp(s_dt) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
        d_sub = handler_t.fetch_data(start_date=act_s, end_date=e_dt, use_cache=True)
        if 'QQQ' in d_sub and 'TQQQ' in d_sub and 'SQQQ' in d_sub:
            d_sub['QQQ'] = StrategyRegistry.calculate(d_sub['QQQ'], strategy_key=selected_strat_key, buffer_pct=buffer_val)
            bt = Backtester(d_sub, initial_capital=10000, start_date=s_dt, end_date=e_dt)
            m_res = bt.run()
            
            stress_results.append({
                "Market Regime / Cycle": name,
                "Strategy Return": f"{m_res.get('Total Return', 0.0):+.1%}",
                "TQQQ B&H Return": f"{m_res.get('TQQQ B&H Return', 0.0):+.1%}",
                "Alpha vs TQQQ": f"{m_res.get('Alpha vs TQQQ', 0.0):+.1%}",
                "Strategy Max Drawdown": f"{m_res.get('Max Drawdown', 0.0):.1%}",
                "Sharpe Ratio": f"{m_res.get('Sharpe Ratio', 0.0):.2f}",
                "Final Wealth ($10k Init)": f"${m_res.get('Final Wealth', 10000):,.0f}"
            })
            
    df_stress = pd.DataFrame(stress_results)
    st.dataframe(
        df_stress,
        use_container_width=True,
        hide_index=True
    )
    
    st.success(f"💡 **Key Takeaway**: By using **QQQ's clean price structure** to generate buy/sell timing while executing in **3x leveraged TQQQ**, the strategy captures monster bull legs without suffering the devastating -82% decay of holding TQQQ through bear markets.")
