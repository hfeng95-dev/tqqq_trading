import html
from typing import Any, Dict, Optional

import pandas as pd

from engine import format_position_label
from strategy import StrategyRegistry

CUSTOM_CSS = """
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

    .signal-banner-partial {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
        color: #ffffff;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.28);
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
"""


def _esc(val: Any) -> str:
    return html.escape(str(val if val is not None else ""))


def render_header_html(
    model_choice: str,
    ai_analysis: Optional[Dict[str, Any]] = None,
    api_configured: bool = False,
) -> str:
    """Honest AI engine status header (never fakes a green Connected badge)."""
    if ai_analysis is not None:
        if ai_analysis.get("is_fallback"):
            badge_color = "#b45309"
            badge_text = "⚠️ Offline / Heuristic Fallback"
        else:
            badge_color = "#15803d"
            badge_text = f"● Live ({_esc(ai_analysis.get('model_used', model_choice))})"
    elif api_configured:
        badge_color = "#1d4ed8"
        badge_text = f"● Ready ({_esc(model_choice)} — On-Demand)"
    else:
        badge_color = "#b45309"
        badge_text = "⚠️ Offline Mode (No GEMINI_API_KEY)"

    return f"""
    <div class="header-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="app-title">⚡ TQQQ Alpha-Locked Trading Terminal</h1>
                <div class="app-subtitle">Institutional 3x Leveraged Trading Engine • QQQ-Derived Signals • Active Model: {_esc(model_choice)}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.82rem; color: #64748b; font-weight: 700;">AI Engine Status</div>
                <div style="color: {badge_color}; font-weight: 800; font-size: 1.02rem;">{badge_text}</div>
            </div>
        </div>
    </div>
    """


def render_architecture_banner_html(buffer_val: float, strategy_key: str) -> str:
    meta = StrategyRegistry.get_metadata(strategy_key)
    if meta.get("supports_short"):
        shield_title = "🐻 100% SQQQ Inverse Hedge"
        shield_desc = (
            f"When QQQ breaches the {buffer_val:.1f}% buffer floor, capital flips into SQQQ "
            "to capture downside bear trends until QQQ reclaims 200 SMA."
        )
    elif meta.get("supports_fractional"):
        shield_title = "🛡️ Volatility-Scaled Cash Shield"
        shield_desc = (
            "Dynamically scales between 100% TQQQ, 50% TQQQ / 50% Cash, and 100% Cash "
            "based on 20-day realized QQQ volatility to suppress 3x compounding drag."
        )
    else:
        shield_title = "🛡️ 100% Cash / Flat"
        shield_desc = (
            f"When QQQ breaches below the {buffer_val:.1f}% buffer floor, capital exits "
            "to Cash, sidestepping devastating -80% leveraged bear crashes."
        )

    return f"""
    <div class="architecture-box">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
            <div style="border-right: 1px solid #e2e8f0; padding-right: 14px;">
                <div style="font-size: 0.76rem; font-weight: 800; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em;">1. Signal Generator (Timing Master)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 3px;">📊 QQQ (1x Nasdaq Index)</div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 3px;">
                    All technical indicators (200 SMA, MACD, Wilder RSI, and <b>{buffer_val:.1f}% Price Buffer</b>) are calculated on QQQ to eliminate 3x decay noise.
                </div>
            </div>
            <div style="border-right: 1px solid #e2e8f0; padding-right: 14px;">
                <div style="font-size: 0.76rem; font-weight: 800; color: #16a34a; text-transform: uppercase; letter-spacing: 0.05em;">2. Execution Stock (Growth Engine)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #16a34a; margin-top: 3px;">🚀 TQQQ (3x UltraPro QQQ)</div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 3px;">
                    <b>Primary bull execution vehicle</b>. Derived 3x daily leverage on QQQ delivers high compounding during confirmed bull trends.
                </div>
            </div>
            <div>
                <div style="font-size: 0.76rem; font-weight: 800; color: #d97706; text-transform: uppercase; letter-spacing: 0.05em;">3. Risk Shield (Capital Protection)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #d97706; margin-top: 3px;">{_esc(shield_title)}</div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 3px;">{_esc(shield_desc)}</div>
            </div>
        </div>
    </div>
    """


def render_signal_banner_html(
    curr_pos: float,
    strat_display_name: str,
    qqq_close: float,
    sma200: float,
    lower_buffer: float,
    dist_200: float,
    dist_lower_buffer: float,
    buffer_val: float,
    tqqq_close: float,
    sqqq_close: float,
    as_of_date: str,
) -> str:
    safe_strat = _esc(strat_display_name)
    safe_date = _esc(as_of_date)

    if curr_pos >= 0.99:
        banner_cls = "signal-banner-bull"
        headline = f"🟢 100% BUY & HOLD TQQQ (${tqqq_close:.2f})"
        detail = (
            f"QQQ (${qqq_close:.2f}) is <b>{dist_200:+.2f}%</b> vs 200 SMA (${sma200:.2f}) and "
            f"<b>{dist_lower_buffer:+.2f}%</b> vs the {buffer_val:.1f}% Buffer Floor (${lower_buffer:.2f})."
        )
        alloc_label = "TQQQ ALLOCATION"
        alloc_val = "100%"
    elif curr_pos > 0.01:
        long_pct = int(round(curr_pos * 100))
        cash_pct = max(0, 100 - long_pct)
        banner_cls = "signal-banner-partial"
        headline = f"🔵 PARTIAL LONG: {long_pct}% TQQQ (${tqqq_close:.2f}) / {cash_pct}% CASH"
        detail = (
            f"Fractional volatility-controlled stance as of {safe_date}. "
            f"QQQ (${qqq_close:.2f}) is <b>{dist_lower_buffer:+.2f}%</b> vs {buffer_val:.1f}% Floor (${lower_buffer:.2f})."
        )
        alloc_label = "TQQQ / CASH"
        alloc_val = f"{long_pct}% / {cash_pct}%"
    elif curr_pos < -0.01:
        short_pct = int(round(abs(curr_pos) * 100))
        banner_cls = "signal-banner-bear"
        headline = f"🐻 TACTICAL SHORT / {short_pct}% LONG SQQQ (${sqqq_close:.2f})"
        detail = (
            f"QQQ (${qqq_close:.2f}) is <b>{dist_lower_buffer:+.2f}%</b> vs the {buffer_val:.1f}% Buffer Floor "
            f"(${lower_buffer:.2f}). Active inverse hedge deployed."
        )
        alloc_label = "SQQQ ALLOCATION"
        alloc_val = f"{short_pct}%"
    else:
        banner_cls = "signal-banner-cash"
        headline = "🔴 DEFENSIVE: 100% CASH / FLAT"
        detail = (
            f"Strategy criteria favor capital preservation as of {safe_date}. "
            f"QQQ (${qqq_close:.2f}) vs 200 SMA (${sma200:.2f}) / {buffer_val:.1f}% Floor (${lower_buffer:.2f})."
        )
        alloc_label = "CASH ALLOCATION"
        alloc_val = "100%"

    return f"""
    <div class="{banner_cls}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">CURRENT ACTIONABLE SIGNAL ({safe_strat}) • AS OF {safe_date}</div>
                <div style="font-size: 2.05rem; font-weight: 800; margin-top: 4px;">{headline}</div>
                <div style="font-size: 1.0rem; opacity: 0.95; margin-top: 4px;">{detail}</div>
            </div>
            <div style="text-align: right; background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px; min-width: 140px;">
                <div style="font-size: 0.78rem; font-weight: 700;">{alloc_label}</div>
                <div style="font-size: 1.65rem; font-weight: 800;">{alloc_val}</div>
            </div>
        </div>
    </div>
    """


def render_timeframe_scorecard_html(metrics: Dict[str, Any], init_capital: float) -> str:
    tot_ret = float(metrics.get("Total Return", 0.0))
    fin_wealth = float(metrics.get("Final Wealth", init_capital))
    net_gain = fin_wealth - init_capital
    cagr = float(metrics.get("Annualized Return", 0.0))
    cagr_alpha = float(metrics.get("CAGR Alpha vs TQQQ", 0.0))
    tqqq_bh_ret = float(metrics.get("TQQQ B&H Return", 0.0))
    tqqq_bh_cagr = float(metrics.get("TQQQ B&H CAGR", 0.0))
    mdd = float(metrics.get("Max Drawdown", 0.0))
    tqqq_mdd = float(metrics.get("TQQQ B&H Max Drawdown", 0.0))
    sharpe = float(metrics.get("Sharpe Ratio", 0.0))

    eff_start = _esc(metrics.get("Effective Start Date", ""))
    eff_end = _esc(metrics.get("Effective End Date", ""))
    trading_days = int(metrics.get("Trading Days", 0))
    years_count = max(trading_days / 252.0, 0.01)

    color_gain = "#16a34a" if net_gain >= 0 else "#dc2626"
    color_alpha = "#16a34a" if cagr_alpha >= 0 else "#dc2626"

    return f"""
    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid #16a34a; border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 12px;">
            <span style="font-weight: 800; font-size: 1.1rem; color: #1e3a8a;">
                📊 EFFECTIVE WINDOW PERFORMANCE ({eff_start} to {eff_end} • {years_count:.1f} Yrs / {trading_days} Trading Days)
            </span>
            <span style="font-weight: 700; font-size: 0.86rem; color: #64748b;">
                Starting Capital: <b>${init_capital:,.0f}</b>
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Total Strategy Return</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: {color_gain}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{tot_ret:+.1%}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">CAGR: <b>{cagr:+.1%}/yr</b></div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Net Profit / Gain ($)</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: {color_gain}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{net_gain:+,.0f} $</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Net realized gain</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Final Portfolio Value</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #1e40af; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">${fin_wealth:,.0f}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">From ${init_capital:,.0f} start</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">CAGR Alpha vs TQQQ</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: {color_alpha}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{cagr_alpha:+.1%}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">TQQQ B&H: <b>{tqqq_bh_cagr:+.1%}/yr</b> ({tqqq_bh_ret:+.1%})</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                <div style="font-size: 0.74rem; font-weight: 800; color: #64748b; text-transform: uppercase;">Max Drawdown</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #dc2626; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{mdd:.1%}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Sharpe: <b>{sharpe:.2f}</b> • TQQQ MDD: <b>{tqqq_mdd:.1%}</b></div>
            </div>
        </div>
    </div>
    """


def render_strategy_playbook_html(
    strategy_key: str,
    sma200: float,
    lower_buffer: float,
    buffer_val: float,
    tqqq_close: float,
    sqqq_close: float,
    ema10: float,
    ema9: float,
    ema30: float,
    rsi: float,
    rvol20: float,
) -> tuple[str, str]:
    """Returns (entry_card_html, exit_card_html) dynamically tailored to the active strategy."""
    meta = StrategyRegistry.get_metadata(strategy_key)
    fmt_ctx = {
        "sma200": sma200,
        "lower_buffer": lower_buffer,
        "buffer_pct": buffer_val,
        "tqqq_close": tqqq_close,
        "sqqq_close": sqqq_close,
        "ema10": ema10,
        "ema9": ema9,
        "ema30": ema30,
        "rsi": rsi,
        "rvol20": rvol20,
    }

    entry_title = _esc(meta["entry_title"].format(**fmt_ctx))
    exit_title = _esc(meta["exit_title"].format(**fmt_ctx))

    entry_items = "".join(
        f"<li style='margin-bottom: 4px;'>{_esc(rule.format(**fmt_ctx))}</li>"
        for rule in meta["entry_rules"]
    )
    exit_items = "".join(
        f"<li style='margin-bottom: 4px;'>{_esc(rule.format(**fmt_ctx))}</li>"
        for rule in meta["exit_rules"]
    )

    entry_html = f"""
    <div class="rule-card" style="border-left: 5px solid #16a34a;">
        <div style="font-weight: 800; color: #15803d; font-size: 1.05rem;">🟢 {entry_title}</div>
        <ul style="margin-top: 8px; padding-left: 18px; font-size: 0.92rem; color: #0f172a;">
            {entry_items}
        </ul>
    </div>
    """
    exit_html = f"""
    <div class="rule-card" style="border-left: 5px solid #dc2626;">
        <div style="font-weight: 800; color: #b91c1c; font-size: 1.05rem;">🔴 {exit_title}</div>
        <ul style="margin-top: 8px; padding-left: 18px; font-size: 0.92rem; color: #0f172a;">
            {exit_items}
        </ul>
    </div>
    """
    return entry_html, exit_html


def render_spot_price_cards_html(
    qqq_close: float,
    sma200: float,
    lower_buffer: float,
    buffer_val: float,
    tqqq_close: float,
    sqqq_close: float,
    curr_pos: float,
    dist_lower_buffer: float,
    as_of_date: str,
) -> list[str]:
    stance_label = _esc(format_position_label(curr_pos))
    safe_date = _esc(as_of_date)

    if curr_pos < -0.01:
        exec_label = f"SQQQ Execution Stock ({safe_date})"
        exec_price = f"${sqqq_close:.2f}"
        exec_color = "#dc2626"
    else:
        exec_label = f"TQQQ Execution Stock ({safe_date})"
        exec_price = f"${tqqq_close:.2f}"
        exec_color = "#16a34a" if curr_pos > 0.01 else "#64748b"

    floor_status = (
        f"Above QQQ ${lower_buffer:.2f} floor"
        if dist_lower_buffer >= 0
        else f"Below QQQ ${lower_buffer:.2f} floor"
    )

    return [
        f"""
        <div class="stat-card">
            <div class="stat-label">QQQ Criterion Price ({safe_date})</div>
            <div class="stat-number">${qqq_close:.2f}</div>
            <div style="color: #64748b; font-size: 0.85rem;">QQQ 200 SMA: <b>${sma200:.2f}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">QQQ Buffer Floor (-{buffer_val:.1f}%)</div>
            <div class="stat-number" style="color: #ea580c;">${lower_buffer:.2f}</div>
            <div style="color: #64748b; font-size: 0.85rem;">QQQ Cushion: <b>${(qqq_close - lower_buffer):+.2f}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">{_esc(exec_label)}</div>
            <div class="stat-number" style="color: {exec_color};">{exec_price}</div>
            <div style="color: #64748b; font-size: 0.85rem;">Stance: <b>{stance_label}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">Distance to QQQ Buffer Floor</div>
            <div class="stat-number" style="color: {'#16a34a' if dist_lower_buffer >= 0 else '#dc2626'};">{dist_lower_buffer:+.2f}%</div>
            <div style="color: #64748b; font-size: 0.85rem;">{_esc(floor_status)}</div>
        </div>
        """,
    ]


def render_backtest_kpi_cards_html(metrics: Dict[str, Any], init_capital: float) -> list[str]:
    tot_ret = float(metrics.get("Total Return", 0.0))
    tqqq_ret = float(metrics.get("TQQQ B&H Return", 0.0))
    cagr_alpha = float(metrics.get("CAGR Alpha vs TQQQ", 0.0))
    mdd = float(metrics.get("Max Drawdown", 0.0))
    tqqq_mdd = float(metrics.get("TQQQ B&H Max Drawdown", 0.0))
    fin_wealth = float(metrics.get("Final Wealth", init_capital))
    color_alpha = "#16a34a" if cagr_alpha >= 0 else "#dc2626"

    return [
        f"""
        <div class="stat-card">
            <div class="stat-label">Strategy Return</div>
            <div class="stat-number" style="color: {'#16a34a' if tot_ret >= 0 else '#dc2626'};">{tot_ret:+.1%}</div>
            <div style="color: #64748b; font-size: 0.82rem;">CAGR: <b>{float(metrics.get('Annualized Return', 0.0)):+.1%}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">TQQQ Buy &amp; Hold</div>
            <div class="stat-number">{tqqq_ret:+.1%}</div>
            <div style="color: #64748b; font-size: 0.82rem;">CAGR: <b>{float(metrics.get('TQQQ B&H CAGR', 0.0)):+.1%}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">CAGR Alpha vs TQQQ</div>
            <div class="stat-number" style="color: {color_alpha};">{cagr_alpha:+.1%}</div>
            <div style="color: #64748b; font-size: 0.82rem;">Total Spread: <b>{float(metrics.get('Alpha vs TQQQ', 0.0)):+.1%}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">Max Drawdown</div>
            <div class="stat-number" style="color: #dc2626;">{mdd:.1%}</div>
            <div style="color: #64748b; font-size: 0.82rem;">TQQQ B&amp;H: <b>{tqqq_mdd:.1%}</b></div>
        </div>
        """,
        f"""
        <div class="stat-card">
            <div class="stat-label">Final Wealth ($)</div>
            <div class="stat-number" style="font-size: 1.45rem; color: #1e40af;">${fin_wealth:,.0f}</div>
            <div style="color: #64748b; font-size: 0.82rem;">Sharpe: <b>{float(metrics.get('Sharpe Ratio', 0.0)):.2f}</b></div>
        </div>
        """,
    ]


def render_gemini_analysis_card_html(
    ai_data: Dict[str, Any], lower_buffer: float
) -> str:
    """HTML-escaped Gemini assessment card with explicit Live vs Offline Fallback badge."""
    is_fallback = bool(ai_data.get("is_fallback", False))
    model_used = _esc(ai_data.get("model_used", "unknown"))
    error_detail = _esc(ai_data.get("error_detail", ""))

    if is_fallback:
        status_pill = (
            "<span style='background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; "
            "padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;'>"
            "⚠️ Offline / Heuristic Fallback</span>"
        )
        error_note = (
            f"<div style='margin-top: 8px; padding: 8px 10px; background: #fffbeb; border: 1px solid #fde68a; "
            f"border-radius: 6px; color: #92400e; font-size: 0.82rem;'>"
            f"<b>Fallback Reason:</b> {error_detail or 'Gemini API unavailable'}</div>"
        )
    else:
        status_pill = (
            "<span style='background: #dcfce7; color: #15803d; border: 1px solid #86efac; "
            "padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;'>"
            f"● Live Gemini ({model_used})</span>"
        )
        error_note = ""

    regime = _esc(ai_data.get("regime", "Market Regime"))
    stance = _esc(ai_data.get("recommended_stance", "STANCE"))
    thesis = _esc(ai_data.get("regime_thesis", ""))
    verdict = _esc(ai_data.get("institutional_verdict", ""))
    vol_risk = _esc(ai_data.get("volatility_decay_risk", "N/A"))

    raw_exec_plan = ai_data.get("execution_plan")
    exec_plan = raw_exec_plan if isinstance(raw_exec_plan, dict) else {}
    ideal_entry = _esc(exec_plan.get("ideal_entry", ""))
    stop_trigger = _esc(exec_plan.get("stop_loss_trigger", ""))
    trailing = _esc(exec_plan.get("trailing_management", ""))

    raw_alloc = ai_data.get("allocation")
    alloc = raw_alloc if isinstance(raw_alloc, dict) else {}
    tqqq_pct = _esc(alloc.get("TQQQ_pct", 0))
    sqqq_pct = _esc(alloc.get("SQQQ_pct", 0))
    cash_pct = _esc(alloc.get("Cash_pct", 0))

    raw_risks = ai_data.get("macro_risks")
    risks = raw_risks if isinstance(raw_risks, list) else ([raw_risks] if raw_risks else [])
    risks_html = "".join(
        f"<li style='color: #475569; font-size: 0.85rem;'>{_esc(r)}</li>" for r in risks
    )

    return f"""
    <div class="ai-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <span style="font-size: 1.35rem; font-weight: 800; color: #1e40af;">{regime}</span>
            <div style="display: flex; gap: 8px; align-items: center;">
                {status_pill}
                <span style="background: #eff6ff; color: #1d4ed8; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">{stance}</span>
            </div>
        </div>
        {error_note}
        <div style="margin-top: 10px; font-size: 0.86rem; color: #475569;">
            <b>Allocation:</b> TQQQ {tqqq_pct}% • SQQQ {sqqq_pct}% • Cash {cash_pct}% &nbsp;|&nbsp;
            <b>3x Decay Risk:</b> {vol_risk}
        </div>
        <p style="margin-top: 10px; color: #0f172a; font-size: 0.95rem;"><b>Thesis:</b> {thesis}</p>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px;">
            <div style="color: #1e40af; font-weight: 700; font-size: 0.84rem; text-transform: uppercase;">Execution Triggers (QQQ Criteria -&gt; Portfolio Orders)</div>
            <div style="color: #0f172a; margin-top: 4px;">• <b>Entry Trigger:</b> {ideal_entry}</div>
            <div style="color: #0f172a;">• <b>Exit / Stop Trigger:</b> {stop_trigger} (QQQ Floor: ${lower_buffer:.2f})</div>
            <div style="color: #475569; font-size: 0.85rem;">• <b>Trailing Management:</b> {trailing}</div>
        </div>
        <div style="margin-top: 10px;">
            <b>⚠️ Key Macro &amp; Volatility Risks:</b>
            <ul style="margin-top: 4px; padding-left: 18px;">{risks_html}</ul>
        </div>
        <p style="margin-top: 10px; color: #1e3a8a; font-weight: 700;"><b>Verdict:</b> {verdict}</p>
    </div>
    """
