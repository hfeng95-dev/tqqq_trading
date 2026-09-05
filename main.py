import argparse
import pandas as pd
from data_handler import DataHandler
from strategy import TQQQStrategy
from backtester import Backtester
import os

def run_backtest(start_date, end_date, name="Custom"):
    handler = DataHandler()
    
    # FETCH WARM-UP DATA: Go back 300 days to ensure SMA_200 is ready
    actual_start = pd.Timestamp(start_date) - pd.Timedelta(days=350)
    data = handler.fetch_data(start_date=actual_start.strftime('%Y-%m-%d'), end_date=end_date)
    
    if 'QQQ' not in data or 'TQQQ' not in data or 'SQQQ' not in data:
        print(f"Error: Missing required ticker data for {name}.")
        return None
        
    df_qqq = data['QQQ']
    df_qqq = TQQQStrategy.calculate_positions(df_qqq)
    
    data['QQQ'] = df_qqq # Update with positions
    
    # Backtester now takes the requested start_date to slice metrics correctly
    backtester = Backtester(data, start_date=start_date)
    metrics = backtester.run()
    
    print(f"\n--- Result: {name} ---")
    if 'Error' in metrics:
        print(f"Error running backtest: {metrics['Error']}")
        return None

    # Print formatted metrics
    print(f"{'Metric':<20} | {'Strategy':<12} | {'TQQQ B&H':<12} | {'Alpha':<12}")
    print("-" * 65)
    
    total_ret = metrics['Total Return']
    tqqq_ret = metrics['TQQQ B&H Return']
    alpha = metrics['Alpha vs TQQQ']
    
    print(f"{'Total Return':<20} | {total_ret:>11.2%} | {tqqq_ret:>11.2%} | {alpha:>11.2%}")
    print(f"{'Max Drawdown':<20} | {metrics['Max Drawdown']:>11.2%} | {'N/A':<12} | {'N/A':<12}")
    print(f"{'Sharpe Ratio':<20} | {metrics['Sharpe Ratio']:>11.2f} | {'N/A':<12} | {'N/A':<12}")
    print(f"{'Final Wealth':<20} | ${metrics['Final Wealth']:>10.2f} | {'N/A':<12} | {'N/A':<12}")
    
    if alpha > 0:
        print(f"\nSUCCESS: Strategy beat TQQQ Buy & Hold by {alpha:.2%}")
    else:
        print(f"\nFAILURE: Strategy lagged TQQQ Buy & Hold by {abs(alpha):.2%}")

    # Save results
    if not os.path.exists('results'):
        os.makedirs('results')
    df_qqq.to_csv(f'results/backtest_{name}_{start_date}.csv')
    return metrics

def generate_signals():
    print("\n--- Generating Current Signals ---")
    handler = DataHandler()
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=2*365)).strftime('%Y-%m-%d')
    data = handler.fetch_data(start_date=start_date)
    
    if 'QQQ' not in data:
        print("Error: QQQ data not available.")
        return
        
    df = data['QQQ']
    df = TQQQStrategy.calculate_positions(df)
    
    last_row = df.iloc[-1]
    
    print(f"\nLast Date: {df.index[-1].date()}")
    print(f"QQQ Price: {last_row['Close']:.2f}")
    print(f"RSI: {last_row['RSI']:.2f}")
    
    pos = last_row['Position']
    if pos == 1:
        print("Recommendation: LONG TQQQ")
    elif pos == -1:
        print("Recommendation: LONG SQQQ (Bearish)")
    else:
        print("Recommendation: FLAT / CASH")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TQQQ Swing Trading Framework - Optimized')
    parser.add_argument('--mode', type=str, choices=['backtest', 'signals', 'regression'], default='signals', help='Mode to run')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date for backtest')
    parser.add_argument('--end', type=str, default=None, help='End date for backtest')
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        run_backtest(args.start, args.end)
    elif args.mode == 'signals':
        generate_signals()
    elif args.mode == 'regression':
        regimes = [
            ('Full_History', '2010-01-01', None),
            ('2022_Bear_Market', '2022-01-01', '2022-12-31'),
            ('2023_2024_Bull', '2023-01-01', None),
        ]
        summaries = []
        for name, start, end in regimes:
            m = run_backtest(start, end, name=name)
            if m: summaries.append((name, m))
            
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)
        for name, m in summaries:
            print(f"{name:<20}: Alpha {m['Alpha vs TQQQ']:>10.2%}, MDD {m['Max Drawdown']:>10.2%}")
