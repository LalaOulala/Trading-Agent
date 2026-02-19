#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from agent_trade_sdk.tools.market_data import (  # noqa: E402
    yfinance_market_snapshot_raw,
    yfinance_price_history_raw,
    yfinance_quote_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Yahoo Finance market tools.")
    parser.add_argument(
        "--action",
        required=True,
        choices=["quote", "history", "snapshot"],
        help="Market data action.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol for quote/history actions.")
    parser.add_argument(
        "--symbols-csv",
        default="SPY,QQQ,DIA,IWM",
        help="Comma-separated symbols for snapshot action.",
    )
    parser.add_argument(
        "--period",
        default="1mo",
        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        help="History period.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"],
        help="History interval.",
    )
    parser.add_argument("--max-points", type=int, default=60, help="Max returned bars.")
    args = parser.parse_args()

    if args.action == "quote":
        payload = yfinance_quote_raw(symbol=args.symbol)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.action == "history":
        payload = yfinance_price_history_raw(
            symbol=args.symbol,
            period=args.period,  # type: ignore[arg-type]
            interval=args.interval,  # type: ignore[arg-type]
            max_points=args.max_points,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    payload = yfinance_market_snapshot_raw(symbols_csv=args.symbols_csv)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
