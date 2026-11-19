import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class Backtester:
    def __init__(
        self,
        df_map: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        slippage_bps: float = 0.0,
        risk_free_rate: float = 0.0,
    ):
        self.df_map = df_map
        self.initial_capital = float(initial_capital)
        self.start_date = start_date
        self.end_date = end_date
        self.slippage_bps = float(slippage_bps)
        self.risk_free_rate = float(risk_free_rate)
        self.df: pd.DataFrame = pd.DataFrame()

    def run(self) -> Dict[str, Any]:
        """
        Runs the backtest across common QQQ/TQQQ/SQQQ trading dates (>= inception),
        models fractional position weights, cash yield, and slippage, and rebases
        Day-0 equity of the evaluation window cleanly to 1.0 (initial_capital).
        """
        for req_key in ("QQQ", "TQQQ", "SQQQ"):
            if req_key not in self.df_map or self.df_map[req_key] is None or self.df_map[req_key].empty:
                self.df = pd.DataFrame()
                return {"Error": f"Missing required ticker data: {req_key}"}

        qqq_df = self.df_map["QQQ"].sort_index()
        tqqq_df = self.df_map["TQQQ"].sort_index()
        sqqq_df = self.df_map["SQQQ"].sort_index()

        # Inner-join trading dates so pre-inception dates never inject NaNs
        common_idx = (
            qqq_df.index.intersection(tqqq_df.index)
            .intersection(sqqq_df.index)
            .sort_values()
        )
        if common_idx.empty:
            self.df = pd.DataFrame()
            return {"Error": "No overlapping trading dates across QQQ, TQQQ, and SQQQ"}

        df = qqq_df.loc[common_idx].copy()
        if "Position" not in df.columns:
            self.df = pd.DataFrame()
            return {"Error": "QQQ DataFrame is missing 'Position' column"}

        # Compute daily returns on aligned full history before slicing
        tqqq_rets = tqqq_df.loc[common_idx, "Close"].pct_change().fillna(0.0)
        sqqq_rets = sqqq_df.loc[common_idx, "Close"].pct_change().fillna(0.0)
        qqq_rets = df["Close"].pct_change().fillna(0.0)

        # Signal decided at close of T-1, realized on day T
        pos = df["Position"].shift(1).fillna(0.0).astype(float)
        long_w = pos.clip(lower=0.0)
        short_w = (-pos).clip(lower=0.0)
        cash_w = (1.0 - long_w - short_w).clip(lower=0.0)

        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / 252.0) - 1.0
        turnover = pos.diff().abs().fillna(0.0)
        cost = turnover * (self.slippage_bps / 10000.0)

        df["Strategy_Returns"] = (
            long_w * tqqq_rets + short_w * sqqq_rets + cash_w * daily_rf
        ) - cost

        # Slice to requested evaluation window
        if self.start_date is not None:
            start_ts = pd.Timestamp(self.start_date)
            df = df[df.index >= start_ts]
        if self.end_date is not None:
            end_ts = pd.Timestamp(self.end_date)
            df = df[df.index <= end_ts]

        if df.empty:
            self.df = df
            return {"Error": "No data in sliced period"}

        # Rebase Day-0 equity cleanly to 1.0 (eliminate prior-day return leak)
        qqq_rets_slice = qqq_rets.reindex(df.index).fillna(0.0).copy()
        tqqq_rets_slice = tqqq_rets.reindex(df.index).fillna(0.0).copy()
        strat_rets_slice = df["Strategy_Returns"].fillna(0.0).copy()

        qqq_rets_slice.iloc[0] = 0.0
        tqqq_rets_slice.iloc[0] = 0.0
        strat_rets_slice.iloc[0] = 0.0

        df["QQQ_Returns"] = qqq_rets_slice
        df["TQQQ_Returns"] = tqqq_rets_slice
        df["Strategy_Returns"] = strat_rets_slice

        df["Market_Cum_Returns"] = (1.0 + qqq_rets_slice).cumprod()
        df["TQQQ_B_H_Returns"] = (1.0 + tqqq_rets_slice).cumprod()
        df["Strategy_Cum_Returns"] = (1.0 + strat_rets_slice).cumprod()
        df["Portfolio_Value"] = self.initial_capital * df["Strategy_Cum_Returns"]

        self.df = df
        return self.calculate_metrics()

    @staticmethod
    def _compute_series_stats(
        cum_series: pd.Series, ret_series: pd.Series, daily_rf: float
    ) -> Dict[str, float]:
        n_obs = len(cum_series)
        n_periods = max(n_obs - 1, 0)
        total_return = float(cum_series.iloc[-1] - 1.0)

        if n_periods == 0:
            cagr = 0.0
        elif total_return <= -1.0:
            cagr = -1.0
        else:
            years = n_periods / 252.0
            if years < (1.0 / 252.0):
                cagr = 0.0
            else:
                log_growth = np.log1p(total_return) / years
                cagr = float(np.expm1(np.clip(log_growth, -700.0, 50.0)))

        active_rets = ret_series.iloc[1:] if n_obs > 1 else ret_series
        excess_rets = active_rets - daily_rf
        std = float(excess_rets.std(ddof=1)) if len(excess_rets) > 1 else 0.0
        sharpe = (
            float((excess_rets.mean() / std) * np.sqrt(252.0))
            if std > 1e-12
            else 0.0
        )

        rolling_max = cum_series.cummax()
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(rolling_max > 0.0, (cum_series / rolling_max) - 1.0, 0.0)
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }

    def calculate_metrics(self) -> Dict[str, Any]:
        if self.df is None or self.df.empty:
            return {"Error": "No data in sliced period"}

        valid_df = self.df.dropna(subset=["Strategy_Returns"])
        if valid_df.empty:
            return {"Error": "No data in sliced period"}

        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / 252.0) - 1.0

        strat_stats = self._compute_series_stats(
            valid_df["Strategy_Cum_Returns"], valid_df["Strategy_Returns"], daily_rf
        )
        tqqq_stats = self._compute_series_stats(
            valid_df["TQQQ_B_H_Returns"], valid_df["TQQQ_Returns"], daily_rf
        )
        qqq_stats = self._compute_series_stats(
            valid_df["Market_Cum_Returns"], valid_df["QQQ_Returns"], daily_rf
        )

        final_wealth = float(valid_df["Portfolio_Value"].iloc[-1])
        tqqq_final_wealth = float(
            self.initial_capital * valid_df["TQQQ_B_H_Returns"].iloc[-1]
        )
        qqq_final_wealth = float(
            self.initial_capital * valid_df["Market_Cum_Returns"].iloc[-1]
        )

        return {
            "Total Return": strat_stats["total_return"],
            "Annualized Return": strat_stats["cagr"],
            "Sharpe Ratio": strat_stats["sharpe"],
            "Max Drawdown": strat_stats["max_drawdown"],
            "Final Wealth": final_wealth,
            "TQQQ B&H Return": tqqq_stats["total_return"],
            "TQQQ B&H CAGR": tqqq_stats["cagr"],
            "TQQQ B&H Sharpe": tqqq_stats["sharpe"],
            "TQQQ B&H Max Drawdown": tqqq_stats["max_drawdown"],
            "TQQQ B&H Final Wealth": tqqq_final_wealth,
            "QQQ B&H Return": qqq_stats["total_return"],
            "QQQ B&H CAGR": qqq_stats["cagr"],
            "QQQ B&H Sharpe": qqq_stats["sharpe"],
            "QQQ B&H Max Drawdown": qqq_stats["max_drawdown"],
            "QQQ B&H Final Wealth": qqq_final_wealth,
            "Alpha vs TQQQ": strat_stats["total_return"] - tqqq_stats["total_return"],
            "CAGR Alpha vs TQQQ": strat_stats["cagr"] - tqqq_stats["cagr"],
            "Effective Start Date": valid_df.index[0].strftime("%Y-%m-%d"),
            "Effective End Date": valid_df.index[-1].strftime("%Y-%m-%d"),
            "Trading Days": int(len(valid_df)),
        }
