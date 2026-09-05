import yfinance as yf
import pandas as pd
import os

class DataHandler:
    def __init__(self, tickers=['TQQQ', 'SQQQ', 'QQQ'], cache_dir='data_cache'):
        self.tickers = tickers
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch_data(self, start_date='2010-01-01', end_date=None, interval='1d', use_cache=True):
        """
        Fetches historical data for the defined tickers.
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        data_frames = {}
        for ticker in self.tickers:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}_{interval}.csv")
            
            if use_cache and os.path.exists(cache_file):
                print(f"Loading {ticker} from cache...")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                data_frames[ticker] = df
            else:
                print(f"Fetching {ticker} from yfinance...")
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=True)
                    if not df.empty:
                        # Flatten multi-index columns if they exist (yfinance sometimes returns them)
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        
                        df.to_csv(cache_file)
                        data_frames[ticker] = df
                    else:
                        print(f"Warning: No data found for {ticker}")
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    
        return data_frames

if __name__ == "__main__":
    handler = DataHandler()
    data = handler.fetch_data(start_date='2023-01-01')
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows")
