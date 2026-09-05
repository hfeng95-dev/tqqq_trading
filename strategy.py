import pandas as pd
import numpy as np
from indicators import Indicators

class StrategyRegistry:
    """
    Quantitative Strategy Engine for TQQQ / SQQQ / QQQ Leveraged Trading.
    Implements multiple leading institutional strategies for side-by-side alpha comparison
    with configurable switch-over buffer (hysteresis band) to eliminate churn.
    """
    
    STRATEGIES = {
        "alpha_locked": "1. Alpha-Locked Master (200 SMA + Bear MACD Bounce)",
        "dual_momentum": "2. Dual Momentum (10 EMA + 200 SMA + RSI Filter)",
        "vol_managed": "3. Volatility-Targeted Momentum (Eliminates 3x Drag)",
        "triple_ma": "4. Triple MA Golden Cross (9 EMA / 30 EMA / 200 SMA)",
        "rsi_mean_reversion": "5. Bull Pullback Dip Buyer (200 SMA + RSI < 40)",
        "long_short_hedged": "6. Long/Short Active Switcher (TQQQ Bull / SQQQ Bear)"
    }
    
    @classmethod
    def calculate(cls, df_in: pd.DataFrame, strategy_key: str = "alpha_locked", buffer_pct: float = 5.0) -> pd.DataFrame:
        df = df_in.copy()
        df = Indicators.apply_all(df)
        
        # Additional indicators
        if 'EMA_9' not in df.columns:
            df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        if 'EMA_30' not in df.columns:
            df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
        if 'Realized_Vol_20' not in df.columns:
            df['Realized_Vol_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100

        # Buffer bands
        buf_factor = buffer_pct / 100.0
        df['SMA_200_Upper_Buffer'] = df['SMA_200'] * (1.0 + buf_factor)
        df['SMA_200_Lower_Buffer'] = df['SMA_200'] * (1.0 - buf_factor)
        
        signals = np.zeros(len(df))
        
        for i in range(1, len(df)):
            close = df['Close'].iloc[i]
            sma200 = df['SMA_200'].iloc[i]
            sma50 = df['SMA_50'].iloc[i]
            ema10 = df['EMA_10'].iloc[i]
            ema9 = df['EMA_9'].iloc[i]
            ema30 = df['EMA_30'].iloc[i]
            macd_hist = df['MACD_Hist'].iloc[i]
            rsi = df['RSI'].iloc[i]
            rvol20 = df['Realized_Vol_20'].iloc[i] if not pd.isna(df['Realized_Vol_20'].iloc[i]) else 20.0
            
            lower_band = df['SMA_200_Lower_Buffer'].iloc[i]
            upper_band = df['SMA_200_Upper_Buffer'].iloc[i]
            prev_sig = signals[i-1]

            if pd.isna(sma200):
                continue

            # ----------------------------------------------------
            # 1. ALPHA-LOCKED MASTER (With Configurable Buffer)
            # ----------------------------------------------------
            if strategy_key == "alpha_locked":
                if prev_sig > 0:
                    # In Long position: Only exit if price drops below lower buffer band and MACD <= 0
                    if close < lower_band and macd_hist <= 0:
                        signals[i] = 0.0 # Exit to Cash
                    else:
                        signals[i] = 1.0 # Hold through buffer noise
                else:
                    # In Cash: Enter Long when close reclaims above 200 SMA or MACD recovery bounce
                    if close > sma200:
                        signals[i] = 1.0
                    elif close > lower_band and macd_hist > 0:
                        signals[i] = 1.0
                    else:
                        signals[i] = 0.0

            # ----------------------------------------------------
            # 2. DUAL MOMENTUM (10 EMA + 200 SMA + RSI)
            # ----------------------------------------------------
            elif strategy_key == "dual_momentum":
                if prev_sig > 0:
                    if close < lower_band or (close < ema10 and rsi < 40):
                        signals[i] = 0.0
                    else:
                        signals[i] = 1.0
                else:
                    if close > sma200 and close > ema10 and rsi > 45:
                        signals[i] = 1.0
                    elif close < sma200 and macd_hist > 0.5:
                        signals[i] = 1.0
                    else:
                        signals[i] = 0.0

            # ----------------------------------------------------
            # 3. VOLATILITY-TARGETED MOMENTUM
            # ----------------------------------------------------
            elif strategy_key == "vol_managed":
                if close > lower_band:
                    if rvol20 < 22:
                        signals[i] = 1.0  # Full 3x allocation
                    elif rvol20 < 32:
                        signals[i] = 0.5  # Half exposure
                    else:
                        signals[i] = 0.0  # High volatility cash shield
                else:
                    signals[i] = 0.5 if (macd_hist > 0 and rvol20 < 28) else 0.0

            # ----------------------------------------------------
            # 4. TRIPLE MA GOLDEN CROSS (9 EMA / 30 EMA / 200 SMA)
            # ----------------------------------------------------
            elif strategy_key == "triple_ma":
                if close > lower_band and ema9 > ema30:
                    signals[i] = 1.0
                else:
                    signals[i] = 0.0

            # ----------------------------------------------------
            # 5. BULL PULLBACK DIP BUYER (200 SMA + RSI < 40)
            # ----------------------------------------------------
            elif strategy_key == "rsi_mean_reversion":
                if close > lower_band:
                    if rsi < 38 or prev_sig == 1.0:
                        if close < sma50 and rsi > 70:
                            signals[i] = 0.0
                        else:
                            signals[i] = 1.0
                    else:
                        signals[i] = 1.0
                else:
                    signals[i] = 1.0 if rsi < 30 else 0.0

            # ----------------------------------------------------
            # 6. LONG / SHORT ACTIVE SWITCHER (TQQQ / SQQQ)
            # ----------------------------------------------------
            elif strategy_key == "long_short_hedged":
                if prev_sig > 0:
                    if close < lower_band:
                        signals[i] = -1.0 # Flip to Short
                    else:
                        signals[i] = 1.0
                else:
                    if close > sma200:
                        signals[i] = 1.0
                    else:
                        signals[i] = -1.0

            else:
                signals[i] = 1.0 if close > lower_band else 0.0

        df['Position'] = signals
        
        # Calculate Trade Signals
        df['Trade_Action'] = 0
        pos_diff = df['Position'].diff()
        df.loc[pos_diff > 0, 'Trade_Action'] = 1   # BUY
        df.loc[pos_diff < 0, 'Trade_Action'] = -1 # SELL
        
        return df

class TQQQStrategy:
    """Backwards-compatible wrapper around StrategyRegistry."""
    @staticmethod
    def calculate_positions(df, strategy_key="alpha_locked", buffer_pct=5.0):
        return StrategyRegistry.calculate(df, strategy_key=strategy_key, buffer_pct=buffer_pct)
