import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def add_sma(df, period):
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df, period):
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        # Calculate fast and slow EMAs
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df

    @classmethod
    def apply_all(cls, df):
        df = cls.add_sma(df, 200)
        df = cls.add_ema(df, 200)
        df = cls.add_sma(df, 50)
        df = cls.add_ema(df, 50)
        df = cls.add_ema(df, 10)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        return df
