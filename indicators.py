import numpy as np
import pandas as pd


class Indicators:
    @staticmethod
    def add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
        df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
        df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Wilder's smoothed EMA RSI with explicit zero-loss and flat-series guards."""
        delta = df["Close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi_vals = np.where(
                avg_gain.isna() | avg_loss.isna(),
                np.nan,
                np.where(
                    avg_loss == 0.0,
                    np.where(avg_gain > 0.0, 100.0, 50.0),
                    100.0 - (100.0 / (1.0 + rs)),
                ),
            )

        df["RSI"] = pd.Series(rsi_vals, index=df.index, dtype=float)
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        return df

    @classmethod
    def apply_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls.add_sma(df, 200)
        df = cls.add_ema(df, 200)
        df = cls.add_sma(df, 50)
        df = cls.add_ema(df, 50)
        df = cls.add_ema(df, 10)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        return df
