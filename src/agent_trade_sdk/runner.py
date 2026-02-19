from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import RunConfig, Runner

from agent_trade_sdk.agent import build_trading_agent
from agent_trade_sdk.config import Settings
from agent_trade_sdk.session_log import SessionMarkdownLogger
from agent_trade_sdk.tools.market_data import yfinance_market_snapshot_raw
from agent_trade_sdk.tools.search import tavily_search_raw
from agent_trade_sdk.tools.trading import AlpacaPaperBroker


def _collect_input_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio": {},
        "market": {},
        "news": {},
        "errors": [],
    }

    portfolio_symbols: list[str] = []
    try:
        broker = AlpacaPaperBroker()
        account = broker.get_account()
        positions = broker.get_positions()
        portfolio_symbols = [str(item.get("symbol")) for item in positions if item.get("symbol")]
        snapshot["portfolio"] = {
            "account": account,
            "positions": positions,
            "positions_count": len(positions),
        }
    except Exception as exc:
        snapshot["errors"].append({"source": "alpaca_portfolio", "error": str(exc)})

    try:
        symbols = ["SPY", "QQQ", "IWM", "DIA"]
        for symbol in portfolio_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
        snapshot["market"] = yfinance_market_snapshot_raw(symbols_csv=",".join(symbols))
    except Exception as exc:
        snapshot["errors"].append({"source": "yfinance_market_snapshot", "error": str(exc)})

    try:
        settings = Settings.from_env(require_openrouter=False)
        if settings.tavily_api_key:
            news_payload = tavily_search_raw(
                query="US stock market today key drivers and catalysts",
                max_results=6,
                topic="news",
                time_range="day",
            )
            compact_news: list[dict[str, Any]] = []
            for item in (news_payload.get("results") or [])[:6]:
                compact_news.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "published_date": item.get("published_date"),
                        "content": (item.get("content") or "")[:350],
                        "score": item.get("score"),
                    }
                )
            snapshot["news"] = {
                "query": news_payload.get("query"),
                "answer": news_payload.get("answer"),
                "results": compact_news,
            }
        else:
            snapshot["errors"].append(
                {"source": "tavily_market_news", "error": "Missing TAVILY_API_KEY"}
            )
    except Exception as exc:
        snapshot["errors"].append({"source": "tavily_market_news", "error": str(exc)})

    return snapshot


async def run_once(
    prompt: str,
    model: str | None = None,
    log_dir: str = "logs",
    enable_tracing: bool = False,
) -> tuple[str, Path]:
    agent = build_trading_agent(model_name=model)
    session_id = uuid.uuid4().hex[:8]
    model_name = model or Settings.from_env(require_openrouter=False).openrouter_model
    logger = SessionMarkdownLogger(
        log_dir=Path(log_dir),
        prompt=prompt,
        model_name=model_name,
        tracing_enabled=enable_tracing,
        session_id=session_id,
    )

    snapshot = _collect_input_snapshot()
    logger.log_input_snapshot(snapshot)

    run_config = RunConfig(tracing_disabled=not enable_tracing)

    try:
        result = Runner.run_streamed(agent, prompt, run_config=run_config)
        async for event in result.stream_events():
            logger.log_stream_event(event)
    except Exception as exc:
        logger.log_error(exc)
        raise

    final_output = str(result.final_output)
    logger.log_final_output(final_output)
    return final_output, logger.file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prompt against the trading agent.")
    parser.add_argument("--prompt", required=True, help="User prompt for the agent run.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where session markdown logs are written.",
    )
    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable SDK tracing (disabled by default in this V1 workflow).",
    )
    args = parser.parse_args()

    output, log_path = asyncio.run(
        run_once(
            prompt=args.prompt,
            model=args.model,
            log_dir=args.log_dir,
            enable_tracing=args.enable_tracing,
        )
    )
    print(output)
    print(f"\nSession log: {log_path}")


if __name__ == "__main__":
    main()
