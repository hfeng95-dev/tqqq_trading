import numpy as np
import pandas as pd
from typing import Dict, Any
from indicators import Indicators


class StrategyRegistry:
    """
    Quantitative Strategy Engine for TQQQ / SQQQ / QQQ Leveraged Trading.
    Implements multiple institutional strategies for side-by-side alpha comparison
    with configurable switch-over buffer (hysteresis band) to eliminate churn.
    """

    STRATEGIES: Dict[str, str] = {
        "alpha_locked": "1. Alpha-Locked Master (200 SMA + Bear MACD Bounce)",
        "dual_momentum": "2. Dual Momentum (10 EMA + 200 SMA + RSI Filter)",
        "vol_managed": "3. Volatility-Targeted Momentum (Eliminates 3x Drag)",
        "triple_ma": "4. Triple MA Golden Cross (9 EMA / 30 EMA / 200 SMA)",
        "rsi_mean_reversion": "5. Bull Pullback Dip Buyer (200 SMA + RSI < 40)",
        "long_short_hedged": "6. Long/Short Active Switcher (TQQQ Bull / SQQQ Bear)",
    }

    STRATEGY_METADATA: Dict[str, Dict[str, Any]] = {
        "alpha_locked": {
            "short_name": "Alpha-Locked Master",
            "entry_title": "QQQ CRITERIA: WHEN TO BUY TQQQ",
            "exit_title": "QQQ CRITERIA: WHEN TO SELL TQQQ (WITH {buffer_pct:.1f}% BUFFER)",
            "entry_rules": [
                "Primary Bull Trigger: Buy 100% TQQQ when QQQ closes above its 200-day SMA (${sma200:.2f}).",
                "Bear Bounce Trigger: If in cash below 200 SMA, buy 100% TQQQ when QQQ is above the buffer floor (${lower_buffer:.2f}) and MACD Histogram > 0.",
                "Execution Asset: 100% TQQQ (${tqqq_close:.2f}).",
            ],
            "exit_rules": [
                "Buffer Exit Trigger: Exit 100% to CASH only when QQQ closes below ${lower_buffer:.2f} (200 SMA minus {buffer_pct:.1f}%) AND MACD Histogram <= 0.",
                "Noise Buffer Zone: Hold 100% TQQQ while QQQ fluctuates between ${lower_buffer:.2f} and ${sma200:.2f} to avoid whipsaw churn.",
                "Defensive Asset: 100% CASH (zero leverage, zero volatility drag).",
            ],
            "supports_short": False,
            "supports_fractional": False,
        },
        "dual_momentum": {
            "short_name": "Dual Momentum",
            "entry_title": "QQQ CRITERIA: DUAL MOMENTUM ENTRY",
            "exit_title": "QQQ CRITERIA: DUAL MOMENTUM EXIT ({buffer_pct:.1f}% BUFFER)",
            "entry_rules": [
                "Trend + Momentum Trigger: Buy 100% TQQQ when QQQ > 200 SMA (${sma200:.2f}), QQQ > 10 EMA (${ema10:.2f}), and RSI > 45.",
                "Normalized MACD Recovery: Below 200 SMA, enter 100% TQQQ if MACD Histogram / Close > 0.20% (price-normalized impulse).",
                "Execution Asset: 100% TQQQ (${tqqq_close:.2f}).",
            ],
            "exit_rules": [
                "Hard Floor Exit: Sell to 100% CASH if QQQ closes below ${lower_buffer:.2f} (200 SMA minus {buffer_pct:.1f}%).",
                "Short-Term Breakdown: Sell to 100% CASH if QQQ closes below 10 EMA (${ema10:.2f}) AND RSI drops below 40.",
                "Defensive Asset: 100% CASH.",
            ],
            "supports_short": False,
            "supports_fractional": False,
        },
        "vol_managed": {
            "short_name": "Volatility-Targeted Momentum",
            "entry_title": "QQQ CRITERIA: VOLATILITY-SCALED SIZING",
            "exit_title": "QQQ CRITERIA: HIGH-VOLATILITY CASH SHIELD",
            "entry_rules": [
                "Low Volatility Bull (100% TQQQ): QQQ > ${lower_buffer:.2f} AND 20-day realized vol < 22% (current: {rvol20:.1f}%).",
                "Moderate Volatility Bull (50% TQQQ / 50% Cash): QQQ > ${lower_buffer:.2f} AND 20-day realized vol between 22% and 32%.",
                "Controlled Bear Bounce (50% TQQQ / 50% Cash): Below ${lower_buffer:.2f} only if MACD Histogram > 0 AND 20-day vol < 28%.",
            ],
            "exit_rules": [
                "Vol Spike Shield: Cut exposure to 100% CASH whenever 20-day realized volatility reaches >= 32% above floor.",
                "Bear Regime Shield: Stay 100% CASH below ${lower_buffer:.2f} whenever MACD <= 0 or realized vol >= 28%.",
                "Defensive Asset: Fractional 50% or 100% CASH to suppress 3x volatility decay.",
            ],
            "supports_short": False,
            "supports_fractional": True,
        },
        "triple_ma": {
            "short_name": "Triple MA Golden Cross",
            "entry_title": "QQQ CRITERIA: TRIPLE MA ALIGNMENT",
            "exit_title": "QQQ CRITERIA: TRIPLE MA BREAKDOWN",
            "entry_rules": [
                "Golden Cross Filter: Hold 100% TQQQ when QQQ > ${lower_buffer:.2f} (200 SMA - {buffer_pct:.1f}%) AND 9 EMA (${ema9:.2f}) > 30 EMA (${ema30:.2f}).",
                "Trend Confirmation: Requires both macro trend support and fast-over-medium moving average alignment.",
                "Execution Asset: 100% TQQQ (${tqqq_close:.2f}).",
            ],
            "exit_rules": [
                "Death Cross or Floor Breach: Exit immediately to 100% CASH if 9 EMA (${ema9:.2f}) <= 30 EMA (${ema30:.2f}) OR QQQ <= ${lower_buffer:.2f}.",
                "Zero Lag Tolerance: No holding through negative EMA cross.",
                "Defensive Asset: 100% CASH.",
            ],
            "supports_short": False,
            "supports_fractional": False,
        },
        "rsi_mean_reversion": {
            "short_name": "Bull Pullback Dip Buyer",
            "entry_title": "QQQ CRITERIA: OVERSOLD DIP ENTRY",
            "exit_title": "QQQ CRITERIA: OVERBOUGHT / FLOOR EXIT",
            "entry_rules": [
                "Bull Pullback Entry: In cash above buffer floor (${lower_buffer:.2f}), buy 100% TQQQ when Wilder RSI(14) drops below 40 (current: {rsi:.1f}).",
                "Deep Bear Oversold Bounce: Below ${lower_buffer:.2f}, take tactical 100% TQQQ bounce only when RSI(14) < 30.",
                "Hold Rule: Once long above ${lower_buffer:.2f}, hold until RSI exceeds 70 or QQQ breaks the floor.",
            ],
            "exit_rules": [
                "Overbought Profit-Take: Exit 100% TQQQ to CASH when RSI(14) > 70 while above ${lower_buffer:.2f}.",
                "Macro Floor Exit: Exit to 100% CASH if QQQ drops to/below ${lower_buffer:.2f} unless deeply oversold (RSI < 30).",
                "Defensive Asset: 100% CASH waiting for next RSI pullback.",
            ],
            "supports_short": False,
            "supports_fractional": False,
        },
        "long_short_hedged": {
            "short_name": "Long/Short Active Switcher",
            "entry_title": "QQQ CRITERIA: LONG TQQQ BULL REGIME",
            "exit_title": "QQQ CRITERIA: FLIP TO SHORT / LONG SQQQ",
            "entry_rules": [
                "Bull Reclaim: Flip from SQQQ to 100% TQQQ (${tqqq_close:.2f}) when QQQ closes above 200 SMA (${sma200:.2f}).",
                "Bull Hold Band: Stay 100% LONG TQQQ while QQQ remains at or above ${lower_buffer:.2f} (200 SMA minus {buffer_pct:.1f}%).",
                "Long Execution Asset: 100% TQQQ.",
            ],
            "exit_rules": [
                "Bear Regime Flip: Sell TQQQ and rotate 100% into SQQQ (${sqqq_close:.2f}) when QQQ closes below ${lower_buffer:.2f}.",
                "Bear Hold Rule: Remain 100% LONG SQQQ (-1.0 exposure) until QQQ reclaims 200 SMA (${sma200:.2f}).",
                "Short Execution Asset: 100% SQQQ (3x Inverse Nasdaq-100).",
            ],
            "supports_short": True,
            "supports_fractional": False,
        },
    }

    @classmethod
    def get_metadata(cls, strategy_key: str) -> Dict[str, Any]:
        return cls.STRATEGY_METADATA.get(strategy_key, cls.STRATEGY_METADATA["alpha_locked"])

    @classmethod
    def calculate(
        cls,
        df_in: pd.DataFrame,
        strategy_key: str = "alpha_locked",
        buffer_pct: float = 5.0,
    ) -> pd.DataFrame:
        df = df_in.copy()
        df = Indicators.apply_all(df)

        # Additional indicators
        if "EMA_9" not in df.columns:
            df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        if "EMA_30" not in df.columns:
            df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()
        if "Realized_Vol_20" not in df.columns:
            df["Realized_Vol_20"] = (
                df["Close"].pct_change().rolling(20).std() * np.sqrt(252.0) * 100.0
            )

        # Buffer bands
        buf_factor = buffer_pct / 100.0
        df["SMA_200_Upper_Buffer"] = df["SMA_200"] * (1.0 + buf_factor)
        df["SMA_200_Lower_Buffer"] = df["SMA_200"] * (1.0 - buf_factor)

        n = len(df)
        signals = np.zeros(n, dtype=np.float64)

        closes = df["Close"].to_numpy(dtype=np.float64)
        sma200s = df["SMA_200"].to_numpy(dtype=np.float64)
        ema10s = df["EMA_10"].to_numpy(dtype=np.float64)
        ema9s = df["EMA_9"].to_numpy(dtype=np.float64)
        ema30s = df["EMA_30"].to_numpy(dtype=np.float64)
        macd_hists = df["MACD_Hist"].to_numpy(dtype=np.float64)
        rsis = df["RSI"].to_numpy(dtype=np.float64)
        rvol20s = df["Realized_Vol_20"].to_numpy(dtype=np.float64)
        lower_bands = df["SMA_200_Lower_Buffer"].to_numpy(dtype=np.float64)

        for i in range(1, n):
            sma200 = sma200s[i]
            if np.isnan(sma200):
                continue

            close = closes[i]
            ema10 = ema10s[i]
            ema9 = ema9s[i]
            ema30 = ema30s[i]
            macd_hist = macd_hists[i]
            rsi = 50.0 if np.isnan(rsis[i]) else rsis[i]
            rvol20 = 20.0 if np.isnan(rvol20s[i]) else rvol20s[i]
            lower_band = lower_bands[i]
            prev_sig = signals[i - 1]

            if strategy_key == "alpha_locked":
                if prev_sig > 0.0:
                    if close < lower_band and macd_hist <= 0.0:
                        signals[i] = 0.0
                    else:
                        signals[i] = 1.0
                else:
                    if close > sma200:
                        signals[i] = 1.0
                    elif close > lower_band and macd_hist > 0.0:
                        signals[i] = 1.0
                    else:
                        signals[i] = 0.0

            elif strategy_key == "dual_momentum":
                macd_norm = (macd_hist / close) if close > 0.0 else 0.0
                if prev_sig > 0.0:
                    if close < lower_band or (close < ema10 and rsi < 40.0):
                        signals[i] = 0.0
                    else:
                        signals[i] = 1.0
                else:
                    if close > sma200 and close > ema10 and rsi > 45.0:
                        signals[i] = 1.0
                    elif close < sma200 and macd_norm > 0.002:
                        signals[i] = 1.0
                    else:
                        signals[i] = 0.0

            elif strategy_key == "vol_managed":
                if close > lower_band:
                    if rvol20 < 22.0:
                        signals[i] = 1.0
                    elif rvol20 < 32.0:
                        signals[i] = 0.5
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.5 if (macd_hist > 0.0 and rvol20 < 28.0) else 0.0

            elif strategy_key == "triple_ma":
                if close > lower_band and ema9 > ema30:
                    signals[i] = 1.0
                else:
                    signals[i] = 0.0

            elif strategy_key == "rsi_mean_reversion":
                if close > lower_band:
                    if prev_sig > 0.0:
                        signals[i] = 0.0 if rsi > 70.0 else 1.0
                    else:
                        signals[i] = 1.0 if rsi < 40.0 else 0.0
                else:
                    signals[i] = 1.0 if rsi < 30.0 else 0.0

            elif strategy_key == "long_short_hedged":
                if prev_sig > 0.0:
                    signals[i] = -1.0 if close < lower_band else 1.0
                else:
                    signals[i] = 1.0 if close > sma200 else -1.0

            else:
                signals[i] = 1.0 if close > lower_band else 0.0

        df["Position"] = signals

        # Calculate Trade Signals
        trade_action = np.zeros(n, dtype=np.int64)
        pos_diff = np.diff(signals, prepend=signals[0] if n > 0 else 0.0)
        trade_action[pos_diff > 0.0] = 1
        trade_action[pos_diff < 0.0] = -1
        df["Trade_Action"] = trade_action

        return df


class TQQQStrategy:
    """Backwards-compatible wrapper around StrategyRegistry."""

    @staticmethod
    def calculate_positions(
        df: pd.DataFrame, strategy_key: str = "alpha_locked", buffer_pct: float = 5.0
    ) -> pd.DataFrame:
        return StrategyRegistry.calculate(
            df, strategy_key=strategy_key, buffer_pct=buffer_pct
        )
