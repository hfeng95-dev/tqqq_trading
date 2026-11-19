import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKERS: List[str] = ["TQQQ", "SQQQ", "QQQ"]
DEFAULT_HISTORY_START = "2008-01-01"


class DataHandler:
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        cache_dir: str = "data_cache",
    ):
        self.tickers: List[str] = (
            list(tickers) if tickers is not None else list(DEFAULT_TICKERS)
        )
        self.cache_dir = cache_dir
        self.last_errors: Dict[str, str] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out.index = pd.to_datetime(out.index)
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
        out.index.name = "Date"
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        if "Close" in out.columns:
            out = out.dropna(subset=["Close"])
        return out

    def _canonical_path(self, ticker: str, interval: str = "1d") -> str:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{safe_ticker}_{interval}.csv")

    @staticmethod
    def _atomic_save_csv(df: pd.DataFrame, path: str) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        df.to_csv(tmp_path)
        os.replace(tmp_path, path)

    @staticmethod
    def _download_range(
        ticker: str, start_str: str, end_inclusive: pd.Timestamp, interval: str
    ) -> pd.DataFrame:
        # yfinance `end` is exclusive; add +1 day to include end_date bar
        dl_end = (end_inclusive + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(
            ticker,
            start=start_str,
            end=dl_end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        return DataHandler._normalize_df(df)

    def fetch_data(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical OHLCV data using one canonical CSV per ticker
        ({ticker}_{interval}.csv) with incremental refresh and in-memory slicing.
        """
        req_start = pd.Timestamp(start_date)
        if req_start.tzinfo is not None:
            req_start = req_start.tz_localize(None)

        if end_date is None:
            req_end = pd.Timestamp.now().normalize()
        else:
            req_end = pd.Timestamp(end_date)
            if req_end.tzinfo is not None:
                req_end = req_end.tz_localize(None)

        self.last_errors = {}
        data_frames: Dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            cache_file = self._canonical_path(ticker, interval=interval)
            full_df: pd.DataFrame = pd.DataFrame()

            if use_cache and not force_refresh and os.path.exists(cache_file):
                try:
                    full_df = self._normalize_df(
                        pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    )
                    logger.debug(
                        "Loaded %s from canonical cache (%d rows)", ticker, len(full_df)
                    )
                except Exception as exc:
                    logger.warning("Failed reading cache for %s: %s", ticker, exc)
                    full_df = pd.DataFrame()

            need_full_fetch = full_df.empty or force_refresh or not use_cache
            if not need_full_fetch:
                cached_max = full_df.index.max()
                # Incrementally fetch only if requested end_date is beyond cached_max
                if req_end > cached_max:
                    inc_start = (cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(
                        "Incrementally refreshing %s from %s to %s",
                        ticker,
                        inc_start,
                        req_end.strftime("%Y-%m-%d"),
                    )
                    try:
                        inc_df = self._download_range(
                            ticker, inc_start, req_end, interval
                        )
                        if not inc_df.empty:
                            full_df = self._normalize_df(
                                pd.concat([full_df, inc_df], axis=0)
                            )
                            if use_cache:
                                self._atomic_save_csv(full_df, cache_file)
                    except Exception as exc:
                        logger.warning(
                            "Incremental refresh failed for %s (%s); using cached data",
                            ticker,
                            exc,
                        )
                        self.last_errors[ticker] = str(exc)
            else:
                fetch_start = min(
                    req_start, pd.Timestamp(DEFAULT_HISTORY_START)
                ).strftime("%Y-%m-%d")
                logger.info(
                    "Fetching full canonical history for %s (%s to %s)",
                    ticker,
                    fetch_start,
                    req_end.strftime("%Y-%m-%d"),
                )
                try:
                    fetched = self._download_range(
                        ticker, fetch_start, req_end, interval
                    )
                    if not fetched.empty:
                        full_df = fetched
                        if use_cache or force_refresh:
                            self._atomic_save_csv(full_df, cache_file)
                    else:
                        msg = f"No data returned by yfinance for {ticker}"
                        logger.warning(msg)
                        self.last_errors[ticker] = msg
                except Exception as exc:
                    msg = f"Error fetching {ticker}: {exc}"
                    logger.error(msg)
                    self.last_errors[ticker] = msg

            if not full_df.empty:
                sliced = full_df.loc[
                    (full_df.index >= req_start) & (full_df.index <= req_end)
                ].copy()
                if not sliced.empty:
                    data_frames[ticker] = sliced
                else:
                    msg = f"No {ticker} bars within {req_start.date()}..{req_end.date()}"
                    logger.warning(msg)
                    self.last_errors[ticker] = msg

        return data_frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = DataHandler()
    data = handler.fetch_data(start_date="2023-01-01")
    for ticker, df in data.items():
        logger.info("%s: %d rows (%s to %s)", ticker, len(df), df.index[0].date(), df.index[-1].date())
