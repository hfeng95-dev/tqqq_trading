import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, df_map, initial_capital=10000, start_date=None, end_date=None):
        self.df_map = df_map
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        """
        Runs the backtest. Handles warm-up data by slicing returns at start_date and end_date.
        """
        # 1. Calculate daily returns on FULL data to avoid edge effects
        tqqq_rets = self.df_map['TQQQ']['Close'].pct_change()
        sqqq_rets = self.df_map['SQQQ']['Close'].pct_change()
        qqq_rets = self.df_map['QQQ']['Close'].pct_change()
        
        # Create a combined df for strategy calc
        df = self.df_map['QQQ'].copy()
        pos = df['Position'].shift(1) # Trade signal at T, realized at T+1
        
        df['Strategy_Returns'] = 0.0
        mask_tqqq = (pos == 1)
        mask_sqqq = (pos == -1)
        if mask_tqqq.any():
            df.loc[mask_tqqq, 'Strategy_Returns'] = tqqq_rets.loc[mask_tqqq]
        if mask_sqqq.any():
            df.loc[mask_sqqq, 'Strategy_Returns'] = sqqq_rets.loc[mask_sqqq]
        
        # 2. SLICE to the requested start_date and end_date
        if self.start_date:
            start_ts = pd.Timestamp(self.start_date)
            df = df[df.index >= start_ts]
            qqq_rets = qqq_rets[qqq_rets.index >= start_ts]
            tqqq_rets = tqqq_rets[tqqq_rets.index >= start_ts]
            
        if self.end_date:
            end_ts = pd.Timestamp(self.end_date)
            df = df[df.index <= end_ts]
            qqq_rets = qqq_rets[qqq_rets.index <= end_ts]
            tqqq_rets = tqqq_rets[tqqq_rets.index <= end_ts]
            
        # 3. Calculate Cumulative Metrics on the sliced window
        df['Market_Cum_Returns'] = (1 + qqq_rets).cumprod()
        df['TQQQ_B_H_Returns'] = (1 + tqqq_rets).cumprod()
        df['Strategy_Cum_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        
        df['Portfolio_Value'] = self.initial_capital * df['Strategy_Cum_Returns']
        
        self.df = df
        return self.calculate_metrics()

    def calculate_metrics(self):
        valid_df = self.df.dropna(subset=['Strategy_Returns'])
        if valid_df.empty:
            return {'Error': 'No data in sliced period'}
            
        days = len(valid_df)
        years = max(days / 252.0, 0.01)
        
        total_return = valid_df['Strategy_Cum_Returns'].iloc[-1] - 1
        ann_return = (1 + total_return) ** (1 / years) - 1
        
        std = valid_df['Strategy_Returns'].std()
        sharpe = (valid_df['Strategy_Returns'].mean() / std) * np.sqrt(252) if std > 0 else 0
        
        rolling_max = valid_df['Strategy_Cum_Returns'].cummax()
        drawdown = (valid_df['Strategy_Cum_Returns'] / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        tqqq_bh_return = valid_df['TQQQ_B_H_Returns'].iloc[-1] - 1
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Final Wealth': valid_df['Portfolio_Value'].iloc[-1],
            'TQQQ B&H Return': tqqq_bh_return,
            'Alpha vs TQQQ': total_return - tqqq_bh_return
        }
        
        return metrics
