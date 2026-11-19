from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_signal_candlestick_chart(
    df_chart: pd.DataFrame,
    start_date_str: str,
    end_date_str: str,
    buffer_pct: float,
    strategy_key: str = "alpha_locked",
) -> go.Figure:
    """QQQ candlestick + SMA/Buffer bands + BUY/SELL/SHORT markers + MACD subplot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"QQQ Price (Decision Engine: {start_date_str} to {end_date_str}) with Signal Markers",
            "QQQ MACD Histogram",
        ),
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="QQQ Price",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    if "SMA_200" in df_chart.columns:
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["SMA_200"],
                line=dict(color="#2563eb", width=2.2),
                name="QQQ 200-day SMA",
            ),
            row=1,
            col=1,
        )
    if "SMA_200_Lower_Buffer" in df_chart.columns:
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["SMA_200_Lower_Buffer"],
                line=dict(color="#ea580c", width=1.5, dash="dash"),
                name=f"QQQ 200 SMA - {buffer_pct:.1f}% Floor",
            ),
            row=1,
            col=1,
        )
    if "SMA_50" in df_chart.columns:
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["SMA_50"],
                line=dict(color="#9333ea", width=1.2, dash="dot"),
                name="QQQ 50-day SMA",
            ),
            row=1,
            col=1,
        )

    buys = df_chart[df_chart["Trade_Action"] == 1]
    sells = df_chart[df_chart["Trade_Action"] == -1]

    if not buys.empty:
        buy_labels = [
            f"TQQQ {int(round(p * 100))}%" if 0.01 < p < 0.99 else "BUY TQQQ"
            for p in buys["Position"]
        ]
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Low"] * 0.985,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=13, color="#16a34a"),
                text=buy_labels,
                textposition="bottom center",
                textfont=dict(color="#16a34a", size=10, family="Arial Black"),
                name="BUY / SCALE UP",
            ),
            row=1,
            col=1,
        )

    if not sells.empty:
        sell_labels = []
        for p in sells["Position"]:
            if p <= -0.01:
                sell_labels.append("SHORT SQQQ")
            elif p > 0.01:
                sell_labels.append(f"REDUCE {int(round(p * 100))}%")
            else:
                sell_labels.append("SELL -> CASH")
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["High"] * 1.015,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=13, color="#dc2626"),
                text=sell_labels,
                textposition="top center",
                textfont=dict(color="#dc2626", size=10, family="Arial Black"),
                name="SELL / HEDGE" if strategy_key == "long_short_hedged" else "SELL / REDUCE",
            ),
            row=1,
            col=1,
        )

    if "MACD_Hist" in df_chart.columns:
        colors_macd = [
            "#16a34a" if (pd.notna(val) and val >= 0) else "#dc2626"
            for val in df_chart["MACD_Hist"]
        ]
        fig.add_trace(
            go.Bar(
                x=df_chart.index,
                y=df_chart["MACD_Hist"],
                marker_color=colors_macd,
                name="MACD Hist",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["MACD"],
                line=dict(color="#2563eb", width=1.2),
                name="MACD",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_chart.index,
                y=df_chart["MACD_Signal"],
                line=dict(color="#ea580c", width=1.2),
                name="Signal",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ],
                font=dict(size=11, color="#1e3a8a"),
            ),
        ),
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_arena_comparison_chart(
    arena_curves: Dict[str, pd.Series], backtest_df: pd.DataFrame
) -> go.Figure:
    """Multi-strategy log-scale equity growth comparison."""
    fig = go.Figure()
    colors = ["#16a34a", "#2563eb", "#7c3aed", "#ea580c", "#0891b2", "#db2777"]

    for idx, (s_name, curve) in enumerate(arena_curves.items()):
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve,
                mode="lines",
                line=dict(width=2.2 if idx == 0 else 1.6, color=colors[idx % len(colors)]),
                name=s_name,
            )
        )

    if backtest_df is not None and not backtest_df.empty:
        fig.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["TQQQ_B_H_Returns"],
                mode="lines",
                line=dict(color="#dc2626", width=1.5, dash="dot"),
                name="TQQQ Buy & Hold",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=backtest_df.index,
                y=backtest_df["Market_Cum_Returns"],
                mode="lines",
                line=dict(color="#475569", width=1.2, dash="dash"),
                name="QQQ 1x Buy & Hold",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Multi-Strategy Growth Comparison (Log Scale, Rebased to 1.00x)",
        yaxis_title="Growth Multiple",
        yaxis_type="log",
        height=520,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_equity_curve_chart(backtest_df: pd.DataFrame, buffer_pct: float) -> go.Figure:
    """Cumulative growth of $1.00 invested (log scale)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Strategy_Cum_Returns"],
            line=dict(color="#16a34a", width=2.5),
            name="Active Strategy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["TQQQ_B_H_Returns"],
            line=dict(color="#dc2626", width=1.5, dash="dot"),
            name="TQQQ Buy & Hold",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Market_Cum_Returns"],
            line=dict(color="#2563eb", width=1.5),
            name="QQQ 1x Benchmark",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Cumulative Growth of $1.00 Invested (Buffer: {buffer_pct:.1f}%)",
        yaxis_title="Growth Multiple",
        yaxis_type="log",
        height=500,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_drawdown_chart(backtest_df: pd.DataFrame) -> go.Figure:
    """Underwater drawdown profile comparing active strategy vs TQQQ Buy & Hold."""
    rolling_max = backtest_df["Strategy_Cum_Returns"].cummax()
    strategy_dd = ((backtest_df["Strategy_Cum_Returns"] / rolling_max) - 1.0) * 100.0
    rolling_max_tqqq = backtest_df["TQQQ_B_H_Returns"].cummax()
    tqqq_dd = ((backtest_df["TQQQ_B_H_Returns"] / rolling_max_tqqq) - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=strategy_dd,
            fill="tozeroy",
            line=dict(color="#16a34a"),
            name="Strategy Drawdown %",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=tqqq_dd,
            line=dict(color="#dc2626", width=1, dash="dot"),
            name="TQQQ Buy & Hold Drawdown %",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Underwater Drawdown Profile (% Loss from Peak)",
        yaxis_title="Drawdown %",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
