from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agents import RunConfig, Runner

from agent_trade_sdk.agent import build_trading_agent
from agent_trade_sdk.config import Settings
from agent_trade_sdk.post_run_memory import run_post_run_memory_cycle
from agent_trade_sdk.session_log import SessionMarkdownLogger
from agent_trade_sdk.tools.market_data import yfinance_market_snapshot_raw
from agent_trade_sdk.tools.perplexity_snapshot import (
    compact_perplexity_snapshot_for_prompt,
    perplexity_market_snapshot_raw,
)
from agent_trade_sdk.tools.search import tavily_search_raw
from agent_trade_sdk.tools.trading import AlpacaPaperBroker


NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class RunCycleResult:
    final_output: str
    log_path: Path
    session_id: str
    model_name: str
    input_snapshot: dict[str, Any]
    bootstrap_context: dict[str, Any]
    runtime_summary: dict[str, Any]


def _json_dumps_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _logs_root_path(log_dir: str | Path) -> Path:
    return Path(log_dir)


def _sessions_log_dir(log_dir: str | Path) -> Path:
    return _logs_root_path(log_dir) / "sessions"


def _collect_input_snapshot() -> dict[str, Any]:
    settings = Settings.from_env(require_openrouter=False)
    snapshot: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio": {},
        "market": {},
        "news": {},
        "perplexity_market_research": {},
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

    try:
        if settings.perplexity_api_key:
            snapshot["perplexity_market_research"] = perplexity_market_snapshot_raw()
        else:
            snapshot["errors"].append(
                {"source": "perplexity_market_snapshot", "error": "Missing PERPLEXITY_API_KEY"}
            )
    except Exception as exc:
        snapshot["errors"].append({"source": "perplexity_market_snapshot", "error": str(exc)})

    return snapshot


def _compact_market_for_prompt(market: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(market, dict):
        return {}
    compact_snapshots: list[dict[str, Any]] = []
    for item in (market.get("snapshots") or [])[:10]:
        if not isinstance(item, dict):
            continue
        quote = item.get("quote") or {}
        compact_snapshots.append(
            {
                "symbol": item.get("symbol"),
                "price": quote.get("regular_market_price"),
                "prev_close": quote.get("regular_market_previous_close"),
                "change": quote.get("regular_market_change"),
                "change_pct": quote.get("regular_market_change_percent"),
                "volume": quote.get("regular_market_volume"),
                "sector": quote.get("sector"),
                "industry": quote.get("industry"),
            }
        )
    return {
        "requested_symbols": market.get("requested_symbols"),
        "quotes": compact_snapshots,
        "errors": market.get("errors"),
    }


def _compact_news_for_prompt(news: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(news, dict):
        return {}
    compact_results: list[dict[str, Any]] = []
    for item in (news.get("results") or [])[:6]:
        if not isinstance(item, dict):
            continue
        compact_results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "published_date": item.get("published_date"),
                "score": item.get("score"),
            }
        )
    return {
        "query": news.get("query"),
        "results": compact_results,
    }


def _compact_portfolio_for_prompt(portfolio: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(portfolio, dict):
        return {}
    positions = portfolio.get("positions") or []
    compact_positions: list[dict[str, Any]] = []
    for item in positions[:15]:
        if not isinstance(item, dict):
            continue
        compact_positions.append(
            {
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "qty": item.get("qty"),
                "market_value": item.get("market_value"),
                "avg_entry_price": item.get("avg_entry_price"),
                "unrealized_pl": item.get("unrealized_pl"),
            }
        )

    account = portfolio.get("account") or {}
    compact_account = (
        {
            "buying_power": account.get("buying_power"),
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "portfolio_value": account.get("portfolio_value"),
            "pattern_day_trader": account.get("pattern_day_trader"),
            "status": account.get("status"),
        }
        if isinstance(account, dict)
        else {}
    )
    return {
        "account": compact_account,
        "positions_count": portfolio.get("positions_count"),
        "positions": compact_positions,
    }


def _build_agent_bootstrap_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "captured_at_utc": snapshot.get("captured_at_utc"),
        "portfolio": _compact_portfolio_for_prompt(snapshot.get("portfolio") or {}),
        "market": _compact_market_for_prompt(snapshot.get("market") or {}),
        "news": _compact_news_for_prompt(snapshot.get("news") or {}),
        "errors": snapshot.get("errors") or [],
    }
    perplexity_block = snapshot.get("perplexity_market_research")
    if isinstance(perplexity_block, dict) and perplexity_block:
        context["perplexity_market_research"] = compact_perplexity_snapshot_for_prompt(perplexity_block)
    return context


def _build_effective_prompt(user_prompt: str, snapshot: dict[str, Any]) -> str:
    bootstrap_context = _build_agent_bootstrap_context(snapshot)
    return (
        "Contexte de pré-run (snapshot initial déjà collecté avant ton exécution):\n"
        "- Utilise ce snapshot comme point de départ.\n"
        "- Vérifie et approfondis avec les tools si les données sont ambiguës, incomplètes ou possiblement "
        "stales.\n"
        "- Tu peux enchaîner plusieurs recherches complémentaires (web, social, market data) avant de "
        "décider.\n"
        "- Le snapshot n'est pas une vérité absolue: recoupe les signaux avant de trader.\n\n"
        "### Snapshot initial (JSON)\n"
        f"```json\n{_json_dumps_compact(bootstrap_context)}\n```\n\n"
        "### Tâche utilisateur\n"
        f"{user_prompt}"
    )


def _local_us_market_clock_estimate() -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    weekday = now_ny.weekday()  # 0=Mon
    open_dt = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_dt = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)

    is_weekday = weekday < 5
    is_open = bool(is_weekday and open_dt <= now_ny < close_dt)

    next_open = None
    next_close = None
    if is_open:
        next_close = close_dt.astimezone(timezone.utc).isoformat()
    else:
        candidate_day = now_ny.date()
        if is_weekday and now_ny < open_dt:
            candidate_open = datetime.combine(candidate_day, time(9, 30), tzinfo=NY_TZ)
        else:
            next_day = candidate_day + timedelta(days=1)
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            candidate_open = datetime.combine(next_day, time(9, 30), tzinfo=NY_TZ)
        next_open = candidate_open.astimezone(timezone.utc).isoformat()

    return {
        "is_open": is_open,
        "timestamp": now_utc.isoformat(),
        "next_open": next_open,
        "next_close": next_close,
        "source": "local_estimate",
    }


def _get_us_market_clock() -> dict[str, Any]:
    try:
        broker = AlpacaPaperBroker()
        payload = broker.get_market_clock()
        payload["source"] = "alpaca"
        return payload
    except Exception as exc:
        payload = _local_us_market_clock_estimate()
        payload["fallback_error"] = str(exc)
        return payload


def _seconds_until_next_interval(interval_minutes: int) -> float:
    now = datetime.now(timezone.utc)
    step = max(1, interval_minutes) * 60
    now_ts = now.timestamp()
    next_ts = ((int(now_ts) // step) + 1) * step
    return max(1.0, next_ts - now_ts)


async def run_once(
    prompt: str,
    model: str | None = None,
    log_dir: str = "logs",
    enable_tracing: bool = False,
) -> RunCycleResult:
    agent = build_trading_agent(model_name=model)
    session_id = uuid.uuid4().hex[:8]
    model_name = model or Settings.from_env(require_openrouter=False).openrouter_model
    logger = SessionMarkdownLogger(
        log_dir=_sessions_log_dir(log_dir),
        prompt=prompt,
        model_name=model_name,
        tracing_enabled=enable_tracing,
        session_id=session_id,
    )

    snapshot = _collect_input_snapshot()
    logger.log_input_snapshot(snapshot)
    bootstrap_context = _build_agent_bootstrap_context(snapshot)
    effective_prompt = _build_effective_prompt(prompt, snapshot)

    run_config = RunConfig(tracing_disabled=not enable_tracing)

    try:
        result = Runner.run_streamed(agent, effective_prompt, run_config=run_config)
        async for event in result.stream_events():
            logger.log_stream_event(event)
    except Exception as exc:
        logger.log_error(exc)
        raise

    final_output = str(result.final_output)
    logger.log_final_output(final_output)
    runtime_summary = logger.build_runtime_summary()
    return RunCycleResult(
        final_output=final_output,
        log_path=logger.file_path,
        session_id=session_id,
        model_name=model_name,
        input_snapshot=snapshot,
        bootstrap_context=bootstrap_context,
        runtime_summary=runtime_summary,
    )


async def run_loop(
    prompt: str,
    model: str | None = None,
    log_dir: str = "logs",
    enable_tracing: bool = False,
    interval_minutes: int = 15,
    market_hours_only: bool = True,
    closed_poll_seconds: int = 60,
) -> None:
    while True:
        if market_hours_only:
            clock = _get_us_market_clock()
            if not bool(clock.get("is_open")):
                print(
                    "[market-closed] "
                    f"source={clock.get('source')} "
                    f"timestamp={clock.get('timestamp')} "
                    f"next_open={clock.get('next_open')} "
                    f"(poll in {closed_poll_seconds}s)"
                )
                await asyncio.sleep(max(5, closed_poll_seconds))
                continue

        try:
            cycle = await run_once(
                prompt=prompt,
                model=model,
                log_dir=log_dir,
                enable_tracing=enable_tracing,
            )
            print(cycle.final_output)
            print(f"\nSession log: {cycle.log_path}")

            try:
                memory_result = await run_post_run_memory_cycle(
                    session_id=cycle.session_id,
                    user_prompt=prompt,
                    model_name=cycle.model_name,
                    final_output=cycle.final_output,
                    bootstrap_context=cycle.bootstrap_context,
                    runtime_summary=cycle.runtime_summary,
                    session_log_path=cycle.log_path,
                    logs_root=_logs_root_path(log_dir),
                    enable_tracing=enable_tracing,
                )
                print(
                    "[post-run] "
                    f"journal={memory_result.journal_file_path} "
                    f"soul_diff={memory_result.soul_diff_path}"
                )
            except Exception as exc:
                print(f"[post-run-error] {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"[run-error] {type(exc).__name__}: {exc}")

        sleep_s = _seconds_until_next_interval(interval_minutes)
        print(f"[scheduler] sleeping {sleep_s:.0f}s until next {interval_minutes}m cycle")
        await asyncio.sleep(sleep_s)


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
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously on a fixed interval (default 15 minutes).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=15,
        help="Loop cadence in minutes when --loop is enabled.",
    )
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="If set with --loop, run even when US market is closed.",
    )
    parser.add_argument(
        "--closed-poll-seconds",
        type=int,
        default=60,
        help="Polling interval while waiting for US market open (loop mode).",
    )
    args = parser.parse_args()

    if args.loop:
        asyncio.run(
            run_loop(
                prompt=args.prompt,
                model=args.model,
                log_dir=args.log_dir,
                enable_tracing=args.enable_tracing,
                interval_minutes=args.interval_minutes,
                market_hours_only=not args.ignore_market_hours,
                closed_poll_seconds=args.closed_poll_seconds,
            )
        )
        return

    cycle = asyncio.run(
        run_once(
            prompt=args.prompt,
            model=args.model,
            log_dir=args.log_dir,
            enable_tracing=args.enable_tracing,
        )
    )
    print(cycle.final_output)
    print(f"\nSession log: {cycle.log_path}")


if __name__ == "__main__":
    main()
