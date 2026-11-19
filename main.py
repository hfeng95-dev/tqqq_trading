import argparse
import logging
import os
from typing import Optional

import pandas as pd

import engine
from strategy import StrategyRegistry

logger = logging.getLogger(__name__)


def run_backtest(
    start_date: str,
    end_date: Optional[str],
    name: str = "Custom",
    strategy_key: str = "alpha_locked",
    buffer_pct: float = 5.0,
    initial_capital: float = 10000.0,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
):
    raw_data, errors = engine.load_canonical_data(
        start_date=start_date, end_date=end_date
    )
    if errors:
        logger.warning("Data warnings for %s: %s", name, errors)

    metrics, bt_df = engine.run_backtest_pipeline(
        raw_data,
        strategy_key=strategy_key,
        buffer_pct=buffer_pct,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )

    print(f"\n--- Result: {name} ({strategy_key}, buffer={buffer_pct:.1f}%) ---")
    if not metrics or "Error" in metrics or bt_df is None or bt_df.empty:
        err = metrics.get("Error") if isinstance(metrics, dict) else "Unknown error"
        print(f"Error running backtest: {err}")
        return None

    print(
        f"Effective Window: {metrics['Effective Start Date']} -> {metrics['Effective End Date']} "
        f"({metrics['Trading Days']} trading days)"
    )
    print(f"{'Metric':<20} | {'Strategy':>12} | {'TQQQ B&H':>12} | {'Diff / Alpha':>12}")
    print("-" * 66)

    total_ret = metrics["Total Return"]
    tqqq_ret = metrics["TQQQ B&H Return"]
    alpha = metrics["Alpha vs TQQQ"]
    cagr = metrics["Annualized Return"]
    tqqq_cagr = metrics["TQQQ B&H CAGR"]
    cagr_alpha = metrics["CAGR Alpha vs TQQQ"]
    mdd = metrics["Max Drawdown"]
    tqqq_mdd = metrics["TQQQ B&H Max Drawdown"]
    sharpe = metrics["Sharpe Ratio"]
    tqqq_sharpe = metrics["TQQQ B&H Sharpe"]
    wealth = metrics["Final Wealth"]
    tqqq_wealth = metrics["TQQQ B&H Final Wealth"]

    print(f"{'Total Return':<20} | {total_ret:>11.2%} | {tqqq_ret:>11.2%} | {alpha:>+11.2%}")
    print(f"{'CAGR':<20} | {cagr:>11.2%} | {tqqq_cagr:>11.2%} | {cagr_alpha:>+11.2%}")
    print(f"{'Max Drawdown':<20} | {mdd:>11.2%} | {tqqq_mdd:>11.2%} | {(mdd - tqqq_mdd):>+11.2%}")
    print(f"{'Sharpe Ratio':<20} | {sharpe:>12.2f} | {tqqq_sharpe:>12.2f} | {(sharpe - tqqq_sharpe):>+12.2f}")
    print(f"{'Final Wealth':<20} | ${wealth:>10,.2f} | ${tqqq_wealth:>10,.2f} | ${(wealth - tqqq_wealth):>+10,.2f}")

    if cagr_alpha >= 0:
        print(f"\nOUTPERFORMED TQQQ Buy & Hold by {cagr_alpha:+.2%}/yr CAGR ({alpha:+.2%} total spread)")
    else:
        print(f"\nLAGGED TQQQ Buy & Hold by {abs(cagr_alpha):.2%}/yr CAGR ({alpha:+.2%} total spread)")

    os.makedirs("results", exist_ok=True)
    end_tag = end_date or metrics["Effective End Date"]
    out_path = f"results/backtest_{name}_{strategy_key}_{start_date}_{end_tag}.csv"
    bt_df.to_csv(out_path)
    print(f"Exported sliced backtest results to {out_path}")
    return metrics


def generate_signals(strategy_key: str = "alpha_locked", buffer_pct: float = 5.0):
    print(f"\n--- Generating Current Signals ({strategy_key}, buffer={buffer_pct:.1f}%) ---")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    raw_data, _ = engine.load_canonical_data(start_date=start_date)

    if "QQQ" not in raw_data or raw_data["QQQ"].empty:
        print("Error: QQQ data not available.")
        return

    df = StrategyRegistry.calculate(
        raw_data["QQQ"], strategy_key=strategy_key, buffer_pct=buffer_pct
    )
    last_row = df.iloc[-1]
    pos = float(last_row["Position"])

    print(f"\nLast Date: {df.index[-1].date()}")
    print(f"QQQ Price: ${float(last_row['Close']):.2f}")
    print(f"QQQ 200 SMA: ${float(last_row.get('SMA_200', 0.0)):.2f}")
    print(f"QQQ {buffer_pct:.1f}% Floor: ${float(last_row.get('SMA_200_Lower_Buffer', 0.0)):.2f}")
    print(f"Wilder RSI (14): {float(last_row.get('RSI', 50.0)):.2f}")
    print(f"Recommendation: {engine.format_position_label(pos)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="TQQQ Swing Trading Framework")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "signals", "regression"],
        default="signals",
        help="Execution mode",
    )
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(StrategyRegistry.STRATEGIES.keys()),
        default="alpha_locked",
        help="Quantitative strategy key",
    )
    parser.add_argument("--buffer", type=float, default=5.0, help="QQQ 200-SMA switch buffer (%)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital ($)")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Turnover slippage in bps")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual cash risk-free rate (e.g. 0.04)")

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(
            args.start,
            args.end,
            strategy_key=args.strategy,
            buffer_pct=args.buffer,
            initial_capital=args.capital,
            slippage_bps=args.slippage_bps,
            risk_free_rate=args.rf,
        )
    elif args.mode == "signals":
        generate_signals(strategy_key=args.strategy, buffer_pct=args.buffer)
    elif args.mode == "regression":
        regimes = [
            ("Full_History", "2010-01-01", None),
            ("2022_Bear_Market", "2022-01-01", "2022-12-31"),
            ("2023_Present_Bull", "2023-01-01", None),
        ]
        summaries = []
        for name, start, end in regimes:
            m = run_backtest(
                start,
                end,
                name=name,
                strategy_key=args.strategy,
                buffer_pct=args.buffer,
                initial_capital=args.capital,
                slippage_bps=args.slippage_bps,
                risk_free_rate=args.rf,
            )
            if m:
                summaries.append((name, m))

        print("\n" + "=" * 72)
        print("OVERALL REGRESSION SUMMARY")
        print("=" * 72)
        for name, m in summaries:
            print(
                f"{name:<20}: CAGR Alpha {m['CAGR Alpha vs TQQQ']:>+8.2%} | "
                f"MDD {m['Max Drawdown']:>7.2%} (TQQQ {m['TQQQ B&H Max Drawdown']:>7.2%}) | "
                f"Sharpe {m['Sharpe Ratio']:>5.2f}"
            )


if __name__ == "__main__":
    main()
