#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from agent_trade_sdk.tools.search import tavily_search_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Tavily search integration.")
    parser.add_argument("--query", required=True, help="Query string.")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results.")
    parser.add_argument(
        "--topic",
        choices=["news", "general"],
        default="news",
        help="Tavily topic mode.",
    )
    parser.add_argument(
        "--time-range",
        choices=["day", "week", "month", "year"],
        default="day",
        help="Tavily recency filter.",
    )
    args = parser.parse_args()

    payload = tavily_search_raw(
        query=args.query,
        max_results=args.max_results,
        topic=args.topic,
        time_range=args.time_range,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
