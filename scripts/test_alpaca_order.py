#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from agent_trade_sdk.config import Settings  # noqa: E402
from agent_trade_sdk.tools.trading import AlpacaPaperBroker  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Alpaca paper trading actions.")
    parser.add_argument(
        "--action",
        required=True,
        choices=["account", "positions", "buy", "sell", "short", "close"],
        help="Action to execute.",
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol.")
    parser.add_argument("--qty", type=float, default=None, help="Quantity for sell/short or buy.")
    parser.add_argument("--notional", type=float, default=None, help="Notional USD for buy orders.")
    parser.add_argument(
        "--time-in-force",
        choices=["day", "gtc", "opg", "cls", "ioc", "fok"],
        default="day",
        help="Order time in force.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the intended payload without sending an order.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Allow running when ALPACA_PAPER is false (not recommended).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings.from_env(require_openrouter=False)

    if not settings.alpaca_paper and not args.allow_live:
        raise RuntimeError(
            "ALPACA_PAPER is false. Refusing execution. Use --allow-live only if intentional."
        )

    if args.action == "account":
        broker = AlpacaPaperBroker()
        print(json.dumps(broker.get_account(), indent=2, ensure_ascii=False))
        return

    if args.action == "positions":
        broker = AlpacaPaperBroker()
        print(json.dumps({"positions": broker.get_positions()}, indent=2, ensure_ascii=False))
        return

    if args.action == "close":
        if args.dry_run:
            payload = {"action": "close", "symbol": args.symbol.upper()}
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        broker = AlpacaPaperBroker()
        print(json.dumps(broker.close_position(args.symbol), indent=2, ensure_ascii=False))
        return

    side = "buy"
    qty = args.qty
    notional = args.notional

    if args.action == "sell":
        side = "sell"
        notional = None
        if qty is None:
            raise ValueError("--qty is required for sell action.")
    elif args.action == "short":
        side = "sell"
        notional = None
        if qty is None:
            raise ValueError("--qty is required for short action.")
    else:
        side = "buy"
        if qty is None and notional is None:
            notional = 10.0

    if args.dry_run:
        payload = {
            "action": args.action,
            "symbol": args.symbol.upper(),
            "side": side,
            "qty": qty,
            "notional": notional,
            "time_in_force": args.time_in_force,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    broker = AlpacaPaperBroker()
    order = broker.submit_market_order(
        symbol=args.symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        notional=notional,
        time_in_force=args.time_in_force,  # type: ignore[arg-type]
    )
    print(json.dumps(order, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
