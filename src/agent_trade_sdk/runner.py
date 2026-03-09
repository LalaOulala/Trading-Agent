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

from agents import Runner
from agents.tracing import custom_span, get_current_trace, trace as sdk_trace

from agent_trade_sdk.agent import build_trading_agent
from agent_trade_sdk.config import Settings
from agent_trade_sdk.memory_architecture import (
    BEHAVIOR_PATH,
    MemoryApplyResult,
    apply_memory_outputs,
    load_behavior_text,
    load_short_memory_latest,
)
from agent_trade_sdk.post_run_memory import PostRunMemoryResult, run_post_run_memory_cycle
from agent_trade_sdk.reflection_memory import (
    build_fallback_reflection_from_short_memory,
    compact_reflection_for_trace,
    load_reflection_latest,
    reflection_prompt_block,
)
from agent_trade_sdk.session_log import SessionMarkdownLogger
from agent_trade_sdk.source_quality import evaluate_snapshot_source_quality
from agent_trade_sdk.strategy_guardrails import compact_stall_for_trace, compute_stall_guardrails
from agent_trade_sdk.tracing_support import build_agents_run_config, build_trace_export_config
from agent_trade_sdk.tools.market_data import yfinance_market_snapshot_raw
from agent_trade_sdk.tools.perplexity_snapshot import (
    compact_perplexity_snapshot_for_prompt,
    perplexity_market_snapshot_raw,
)
from agent_trade_sdk.tools.search import tavily_search_raw
from agent_trade_sdk.tools.trading import AlpacaPaperBroker


NY_TZ = ZoneInfo("America/New_York")
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_FILE = ROOT_DIR / "prompts" / "default_loop_prompt.txt"


@dataclass(frozen=True)
class RunCycleResult:
    final_output: str
    log_path: Path
    session_id: str
    model_name: str
    input_snapshot: dict[str, Any]
    bootstrap_context: dict[str, Any]
    runtime_summary: dict[str, Any]
    short_memory_input: dict[str, Any] | None
    reflection_input: dict[str, Any] | None
    memory_apply_result: MemoryApplyResult
    post_run_memory_result: PostRunMemoryResult | None
    post_run_memory_error: str | None


def _json_dumps_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _logs_root_path(log_dir: str | Path) -> Path:
    return Path(log_dir)


def _sessions_log_dir(log_dir: str | Path) -> Path:
    return _logs_root_path(log_dir) / "sessions"


def _trace_custom_span_if_active(
    name: str,
    *,
    data: dict[str, Any],
    enable_tracing: bool,
) -> None:
    if not enable_tracing:
        return
    if get_current_trace() is None:
        return
    with custom_span(name, data=data, disabled=False):
        pass


def _compact_account_for_trace(account: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(account, dict):
        return {}
    return {
        "buying_power": account.get("buying_power"),
        "equity": account.get("equity"),
        "cash": account.get("cash"),
        "portfolio_value": account.get("portfolio_value"),
        "status": account.get("status"),
    }


def _compact_source_quality_for_trace(source_quality: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source_quality, dict):
        return {}
    compact: dict[str, Any] = {}
    for provider in ("tavily", "perplexity"):
        payload = source_quality.get(provider)
        if not isinstance(payload, dict):
            continue
        compact[provider] = {
            "usable_for_decision": payload.get("usable_for_decision"),
            "freshness_hours_median": payload.get("freshness_hours_median"),
            "duplicate_ratio": payload.get("duplicate_ratio"),
            "trusted_domain_ratio": payload.get("trusted_domain_ratio"),
            "finance_relevance_score": payload.get("finance_relevance_score"),
            "diagnostics": payload.get("diagnostics"),
        }
    return compact


def _build_decision_source_attribution(
    *,
    runtime_summary: dict[str, Any],
    snapshot: dict[str, Any],
    final_output: str,
) -> dict[str, Any]:
    tool_calls = runtime_summary.get("tool_calls") if isinstance(runtime_summary, dict) else []
    tavily_runtime_calls = 0
    if isinstance(tool_calls, list):
        tavily_runtime_calls = sum(
            1
            for item in tool_calls
            if isinstance(item, dict) and str(item.get("tool_name")) == "web_search_tavily"
        )
    lower_output = final_output.lower()
    news_block = snapshot.get("news") if isinstance(snapshot, dict) else {}
    if not isinstance(news_block, dict):
        news_block = {}
    perplexity_block = (
        snapshot.get("perplexity_market_research")
        if isinstance(snapshot, dict)
        else {}
    )
    return {
        "pre_run_tavily_available": bool(news_block.get("results")),
        "pre_run_perplexity_available": bool(
            isinstance(perplexity_block, dict) and perplexity_block
        ),
        "runtime_tavily_tool_calls": tavily_runtime_calls,
        "decision_mentions_tavily": "tavily" in lower_output or "web_search" in lower_output,
        "decision_mentions_perplexity": "perplexity" in lower_output,
        "source_quality": _compact_source_quality_for_trace(snapshot.get("source_quality")),
    }


def _collect_input_snapshot(enable_tracing: bool = False) -> dict[str, Any]:
    settings = Settings.from_env(require_openrouter=False)
    snapshot: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_clock": {},
        "portfolio": {},
        "market": {},
        "news": {},
        "perplexity_market_research": {},
        "source_quality": {},
        "strategy_guardrails": {},
        "errors": [],
    }

    portfolio_symbols: list[str] = []
    _trace_custom_span_if_active(
        "pre_run.snapshot_collection_started",
        data={"captured_at_utc": snapshot["captured_at_utc"]},
        enable_tracing=enable_tracing,
    )
    try:
        clock_payload = _get_us_market_clock()
        snapshot["market_clock"] = clock_payload
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.market_clock",
            data={
                "status": "ok",
                "market_clock": clock_payload,
            },
            enable_tracing=enable_tracing,
        )
    except Exception as exc:
        snapshot["errors"].append({"source": "market_clock", "error": str(exc)})
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.market_clock",
            data={
                "status": "error",
                "error": str(exc),
            },
            enable_tracing=enable_tracing,
        )
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
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.alpaca_portfolio",
            data={
                "status": "ok",
                "positions_count": len(positions),
                "symbols": portfolio_symbols[:20],
                "account": _compact_account_for_trace(account),
            },
            enable_tracing=enable_tracing,
        )
    except Exception as exc:
        snapshot["errors"].append({"source": "alpaca_portfolio", "error": str(exc)})
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.alpaca_portfolio",
            data={
                "status": "error",
                "error": str(exc),
            },
            enable_tracing=enable_tracing,
        )

    try:
        symbols = ["SPY", "QQQ", "IWM", "DIA"]
        for symbol in portfolio_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
        snapshot["market"] = yfinance_market_snapshot_raw(symbols_csv=",".join(symbols))
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.yfinance_market",
            data={
                "status": "ok",
                "requested_symbols": symbols,
                "market": _compact_market_for_prompt(snapshot["market"]),
            },
            enable_tracing=enable_tracing,
        )
    except Exception as exc:
        snapshot["errors"].append({"source": "yfinance_market_snapshot", "error": str(exc)})
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.yfinance_market",
            data={
                "status": "error",
                "error": str(exc),
            },
            enable_tracing=enable_tracing,
        )

    try:
        if settings.tavily_api_key:
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.tavily_news.call",
                data={
                    "query": "US stock market today key drivers and catalysts",
                    "max_results": 6,
                    "topic": "news",
                    "time_range": "day",
                },
                enable_tracing=enable_tracing,
            )
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
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.tavily_news.output",
                data={
                    "status": "ok",
                    "results_count": len(compact_news),
                    "answer_excerpt": (news_payload.get("answer") or "")[:400],
                    "top_titles": [item.get("title") for item in compact_news[:3]],
                },
                enable_tracing=enable_tracing,
            )
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.tavily_news",
                data={
                    "status": "ok",
                    "news": _compact_news_for_prompt(snapshot["news"]),
                    "answer_excerpt": (news_payload.get("answer") or "")[:400],
                },
                enable_tracing=enable_tracing,
            )
        else:
            snapshot["errors"].append(
                {"source": "tavily_market_news", "error": "Missing TAVILY_API_KEY"}
            )
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.tavily_news",
                data={
                    "status": "skipped",
                    "reason": "Missing TAVILY_API_KEY",
                },
                enable_tracing=enable_tracing,
            )
    except Exception as exc:
        snapshot["errors"].append({"source": "tavily_market_news", "error": str(exc)})
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.tavily_news",
            data={
                "status": "error",
                "error": str(exc),
            },
            enable_tracing=enable_tracing,
        )

    try:
        if settings.perplexity_api_key:
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.perplexity_market_research.call",
                data={
                    "provider": "perplexity",
                    "model": settings.perplexity_model,
                    "search_recency_filter": settings.perplexity_snapshot_search_recency,
                },
                enable_tracing=enable_tracing,
            )
            snapshot["perplexity_market_research"] = perplexity_market_snapshot_raw()
            perplexity_block = snapshot.get("perplexity_market_research") or {}
            compact_perplexity = (
                compact_perplexity_snapshot_for_prompt(perplexity_block)
                if isinstance(perplexity_block, dict)
                else {}
            )
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.perplexity_market_research.output",
                data={
                    "status": "ok",
                    "provider": "perplexity",
                    "citations_count": len(perplexity_block.get("citations") or [])
                    if isinstance(perplexity_block, dict)
                    else 0,
                    "search_results_count": len(perplexity_block.get("search_results") or [])
                    if isinstance(perplexity_block, dict)
                    else 0,
                    "snapshot": compact_perplexity,
                },
                enable_tracing=enable_tracing,
            )
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.perplexity_market_research",
                data={
                    "status": "ok",
                    "perplexity_market_research": compact_perplexity,
                },
                enable_tracing=enable_tracing,
            )
        else:
            snapshot["errors"].append(
                {"source": "perplexity_market_snapshot", "error": "Missing PERPLEXITY_API_KEY"}
            )
            _trace_custom_span_if_active(
                "pre_run.snapshot_source.perplexity_market_research",
                data={
                    "status": "skipped",
                    "reason": "Missing PERPLEXITY_API_KEY",
                },
                enable_tracing=enable_tracing,
            )
    except Exception as exc:
        snapshot["errors"].append({"source": "perplexity_market_snapshot", "error": str(exc)})
        _trace_custom_span_if_active(
            "pre_run.snapshot_source.perplexity_market_research",
            data={
                "status": "error",
                "error": str(exc),
            },
            enable_tracing=enable_tracing,
        )

    snapshot["source_quality"] = evaluate_snapshot_source_quality(snapshot)
    _trace_custom_span_if_active(
        "pre_run.snapshot_source_quality",
        data={"source_quality": _compact_source_quality_for_trace(snapshot.get("source_quality"))},
        enable_tracing=enable_tracing,
    )

    _trace_custom_span_if_active(
        "pre_run.snapshot_collection_summary",
        data={
            "captured_at_utc": snapshot.get("captured_at_utc"),
            "bootstrap_context": _build_agent_bootstrap_context(snapshot),
            "errors": snapshot.get("errors") or [],
        },
        enable_tracing=enable_tracing,
    )

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
        "market_clock": snapshot.get("market_clock") or {},
        "portfolio": _compact_portfolio_for_prompt(snapshot.get("portfolio") or {}),
        "market": _compact_market_for_prompt(snapshot.get("market") or {}),
        "news": _compact_news_for_prompt(snapshot.get("news") or {}),
        "source_quality": _compact_source_quality_for_trace(snapshot.get("source_quality")),
        "strategy_guardrails": snapshot.get("strategy_guardrails") or {},
        "errors": snapshot.get("errors") or [],
    }
    perplexity_block = snapshot.get("perplexity_market_research")
    if isinstance(perplexity_block, dict) and perplexity_block:
        context["perplexity_market_research"] = compact_perplexity_snapshot_for_prompt(perplexity_block)
    return context


def _short_memory_fallback_prompt(short_memory: dict[str, Any] | None) -> str:
    if not isinstance(short_memory, dict):
        return ""
    summary = str(short_memory.get("session_summary") or "").strip()
    report = str(short_memory.get("decision_report") or "").strip()
    directives = short_memory.get("next_session_directives")
    pitfalls = short_memory.get("pitfalls_to_avoid_next_run")

    lines: list[str] = ["### Fallback mémoire courte (compacte)"]
    if summary:
        lines.append(f"- session_summary: {summary[:420]}")
    if report:
        lines.append(f"- decision_report: {report[:420]}")
    if isinstance(pitfalls, list) and pitfalls:
        lines.append("- pitfalls_to_avoid_next_run:")
        for item in pitfalls[:3]:
            lines.append(f"  - {str(item)[:180]}")
    if isinstance(directives, list) and directives:
        lines.append("- next_session_directives:")
        for item in directives[:3]:
            lines.append(f"  - {str(item)[:180]}")
    return "\n".join(lines)[:1200] + "\n"


def _build_effective_prompt(
    user_prompt: str,
    snapshot: dict[str, Any],
    reflection_memory: dict[str, Any] | None = None,
    short_memory_fallback: dict[str, Any] | None = None,
) -> str:
    bootstrap_context = _build_agent_bootstrap_context(snapshot)
    return (
        f"{reflection_prompt_block(reflection_memory, max_total_chars=3000)}\n"
        f"{_short_memory_fallback_prompt(short_memory_fallback)}\n"
        "Contexte de pré-run (snapshot initial déjà collecté avant ton exécution):\n"
        "- Utilise ce snapshot comme point de départ.\n"
        "- Vérifie et approfondis avec les tools si les données sont ambiguës, incomplètes ou possiblement "
        "stales.\n"
        "- Tu peux enchaîner plusieurs recherches complémentaires (web, market data) avant de décider.\n"
        "- Le snapshot n'est pas une vérité absolue: recoupe les signaux avant de trader.\n\n"
        "Exigence d'horloge marché:\n"
        "- Avant toute conclusion pre-market/post-close, vérifie market_clock du snapshot ou appelle "
        "get_market_clock_snapshot.\n\n"
        "### Snapshot initial (JSON)\n"
        f"```json\n{_json_dumps_compact(bootstrap_context)}\n```\n\n"
        "### Tâche utilisateur\n"
        f"{user_prompt}\n\n"
        "### Exigences de continuité (mémoire)\n"
        "- En fin de session, produis un résumé de session, une autocritique, des pièges à éviter, des "
        "directives pour le prochain run et des questions ouvertes.\n"
        "- Si tu veux ajuster ta stratégie ou ton style de trading durable, propose une mise à jour complète de "
        "behavior.md dans long_memory_update_intent.updated_behavior_markdown.\n"
        "- Si tu ne veux pas modifier behavior.md, mets should_update_behavior=false et updated_behavior_markdown=null.\n"
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


def _resolve_prompt(
    *,
    prompt: str | None,
    prompt_file: str | None,
) -> tuple[str, Path | None]:
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip(), None

    candidate = Path(prompt_file).expanduser() if prompt_file else DEFAULT_PROMPT_FILE
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    if not candidate.exists():
        raise ValueError(
            "No --prompt provided and prompt file was not found: "
            f"{candidate}. Provide --prompt or create this file."
        )

    content = candidate.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(
            "Prompt file is empty: "
            f"{candidate}. Add prompt content or provide --prompt."
        )

    return content, candidate


def _build_cycle_trace(
    *,
    session_id: str,
    model: str | None,
    enable_tracing: bool,
    loop: bool,
    interval_minutes: int | None = None,
    market_hours_only: bool | None = None,
):
    tracing_settings = Settings.from_env(require_openrouter=False) if enable_tracing else None
    raw_metadata: dict[str, Any] = {
        "component": "cycle",
        "session_id": session_id,
        "provider": "openrouter",
        "model_override": model,
        "loop": loop,
    }
    if interval_minutes is not None:
        raw_metadata["interval_minutes"] = interval_minutes
    if market_hours_only is not None:
        raw_metadata["market_hours_only"] = market_hours_only
    metadata: dict[str, str] = {
        key: str(value)
        for key, value in raw_metadata.items()
        if value is not None
    }
    return sdk_trace(
        "agent_trade_sdk.trading_cycle",
        group_id=session_id,
        metadata=metadata,
        tracing=build_trace_export_config(
            enable_tracing=enable_tracing,
            settings=tracing_settings,
        ),
        disabled=not enable_tracing,
    )


async def run_once(
    prompt: str,
    model: str | None = None,
    log_dir: str = "logs",
    enable_tracing: bool = False,
    session_id: str | None = None,
) -> RunCycleResult:
    settings = Settings.from_env(require_openrouter=False)
    agent = build_trading_agent(model_name=model)
    session_id = session_id or uuid.uuid4().hex[:8]
    model_name = model or settings.openrouter_model
    logger = SessionMarkdownLogger(
        log_dir=_sessions_log_dir(log_dir),
        prompt=prompt,
        model_name=model_name,
        tracing_enabled=enable_tracing,
        session_id=session_id,
    )

    short_memory = load_short_memory_latest()
    reflection_memory = load_reflection_latest()
    behavior_text = load_behavior_text()
    _trace_custom_span_if_active(
        "pre_run.memory_inputs_loaded",
        data={
            "short_memory_present": short_memory is not None,
            "short_memory_keys": sorted(list((short_memory or {}).keys()))[:20],
            "reflection_memory_present": reflection_memory is not None,
            "reflection_memory_keys": sorted(list((reflection_memory or {}).keys()))[:20],
            "behavior_path": str(BEHAVIOR_PATH),
            "behavior_length_chars": len(behavior_text),
        },
        enable_tracing=enable_tracing,
    )
    snapshot = _collect_input_snapshot(enable_tracing=enable_tracing)
    guardrails = compute_stall_guardrails(snapshot, lookback_runs=6).model_dump()
    snapshot["strategy_guardrails"] = guardrails

    fallback_reflection = (
        reflection_memory
        if isinstance(reflection_memory, dict)
        else build_fallback_reflection_from_short_memory(short_memory, session_id=session_id)
    )
    if isinstance(fallback_reflection, dict) and isinstance(guardrails, dict):
        existing_flags = fallback_reflection.get("stall_flags") or []
        if isinstance(existing_flags, list):
            merged_flags = [str(item) for item in existing_flags if str(item).strip()]
        else:
            merged_flags = []
        for item in guardrails.get("stall_flags") or []:
            text = str(item).strip()
            if text and text not in merged_flags:
                merged_flags.append(text)
        fallback_reflection["stall_flags"] = merged_flags

        existing_rules = fallback_reflection.get("hard_rules_next_run") or []
        if isinstance(existing_rules, list):
            merged_rules = [str(item) for item in existing_rules if str(item).strip()]
        else:
            merged_rules = []
        for item in guardrails.get("hard_rules_next_run") or []:
            text = str(item).strip()
            if text and text not in merged_rules:
                merged_rules.append(text)
        fallback_reflection["hard_rules_next_run"] = merged_rules[:8]

    _trace_custom_span_if_active(
        "pre_run.strategy_guardrails",
        data={"strategy_guardrails": compact_stall_for_trace(guardrails)},
        enable_tracing=enable_tracing,
    )

    logger.log_input_snapshot(snapshot)
    logger.log_short_memory_input(short_memory)
    logger.log_reflection_input(fallback_reflection)
    logger.log_source_quality_assessment(snapshot.get("source_quality"))
    logger.log_behavior_input(BEHAVIOR_PATH, behavior_text)
    bootstrap_context = _build_agent_bootstrap_context(snapshot)
    effective_prompt = _build_effective_prompt(
        prompt,
        snapshot,
        reflection_memory=fallback_reflection,
        short_memory_fallback=None if reflection_memory else short_memory,
    )
    _trace_custom_span_if_active(
        "pre_run.prompt_built",
        data={
            "session_id": session_id,
            "bootstrap_context": bootstrap_context,
            "effective_prompt_length_chars": len(effective_prompt),
            "reflection_context": compact_reflection_for_trace(fallback_reflection),
            "strategy_guardrails": compact_stall_for_trace(guardrails),
        },
        enable_tracing=enable_tracing,
    )

    run_config = build_agents_run_config(
        enable_tracing=enable_tracing,
        workflow_name="agent_trade_sdk.main_run",
        group_id=session_id,
        trace_metadata={
            "component": "main_run",
            "session_id": session_id,
            "provider": "openrouter",
            "model": model_name,
            "streaming": True,
        },
        settings=settings,
    )

    try:
        result = Runner.run_streamed(agent, effective_prompt, run_config=run_config)
        async for event in result.stream_events():
            logger.log_stream_event(event)
    except Exception as exc:
        logger.log_error(exc)
        _trace_custom_span_if_active(
            "run.main_run_error",
            data={
                "session_id": session_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            enable_tracing=enable_tracing,
        )
        raise

    final_output = str(result.final_output)
    logger.log_final_output(final_output)
    runtime_summary = logger.build_runtime_summary()
    _trace_custom_span_if_active(
        "run.runtime_summary",
        data={
            "session_id": session_id,
            "event_count": runtime_summary.get("event_count"),
            "tool_error_count": runtime_summary.get("tool_error_count"),
            "tool_calls": runtime_summary.get("tool_calls"),
            "tool_outputs": runtime_summary.get("tool_outputs"),
            "reasoning_summaries": runtime_summary.get("reasoning_summaries"),
            "message_outputs": runtime_summary.get("message_outputs"),
        },
        enable_tracing=enable_tracing,
    )
    source_attribution = _build_decision_source_attribution(
        runtime_summary=runtime_summary,
        snapshot=snapshot,
        final_output=final_output,
    )
    _trace_custom_span_if_active(
        "run.decision.source_attribution",
        data={
            "session_id": session_id,
            "source_attribution": source_attribution,
        },
        enable_tracing=enable_tracing,
    )
    memory_apply_result = apply_memory_outputs(
        raw_final_output=final_output,
        session_id=session_id,
        model_name=model_name,
        logs_root=_logs_root_path(log_dir),
        runtime_summary=runtime_summary,
    )
    logger.log_memory_apply_result(memory_apply_result.summary_for_log())
    _trace_custom_span_if_active(
        "run.memory_apply_outputs",
        data={
            "session_id": session_id,
            "memory_apply_result": memory_apply_result.summary_for_log(),
        },
        enable_tracing=enable_tracing,
    )
    post_run_result: PostRunMemoryResult | None = None
    post_run_error: str | None = None
    try:
        post_run_result = await run_post_run_memory_cycle(
            session_id=session_id,
            user_prompt=prompt,
            model_name=model_name,
            final_output=final_output,
            bootstrap_context=bootstrap_context,
            runtime_summary=runtime_summary,
            session_log_path=logger.file_path,
            logs_root=_logs_root_path(log_dir),
            enable_tracing=enable_tracing,
        )
        logger.log_post_run_status(
            {
                "status": "ok",
                "journal_file_path": str(post_run_result.journal_file_path),
                "reflection_latest_path": str(post_run_result.reflection_latest_path),
                "reflection_archive_path": str(post_run_result.reflection_archive_path),
                "behavior_updated": post_run_result.behavior_updated,
                "behavior_diff_path": str(post_run_result.behavior_diff_path)
                if post_run_result.behavior_diff_path
                else None,
            }
        )
        _trace_custom_span_if_active(
            "run.post_run_reflection",
            data={
                "session_id": session_id,
                "journal_file_path": str(post_run_result.journal_file_path),
                "reflection_latest_path": str(post_run_result.reflection_latest_path),
                "reflection_archive_path": str(post_run_result.reflection_archive_path),
                "behavior_updated": post_run_result.behavior_updated,
                "behavior_diff_path": str(post_run_result.behavior_diff_path)
                if post_run_result.behavior_diff_path
                else None,
            },
            enable_tracing=enable_tracing,
        )
    except Exception as exc:
        post_run_error = f"{type(exc).__name__}: {exc}"
        logger.log_post_run_status({"status": "error", "error": post_run_error})
        _trace_custom_span_if_active(
            "run.post_run_reflection_error",
            data={
                "session_id": session_id,
                "error": post_run_error,
            },
            enable_tracing=enable_tracing,
        )

    return RunCycleResult(
        final_output=final_output,
        log_path=logger.file_path,
        session_id=session_id,
        model_name=model_name,
        input_snapshot=snapshot,
        bootstrap_context=bootstrap_context,
        runtime_summary=runtime_summary,
        short_memory_input=short_memory,
        reflection_input=fallback_reflection,
        memory_apply_result=memory_apply_result,
        post_run_memory_result=post_run_result,
        post_run_memory_error=post_run_error,
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
            cycle_session_id = uuid.uuid4().hex[:8]
            cycle_trace = _build_cycle_trace(
                session_id=cycle_session_id,
                model=model,
                enable_tracing=enable_tracing,
                loop=True,
                interval_minutes=interval_minutes,
                market_hours_only=market_hours_only,
            )

            with cycle_trace:
                with custom_span(
                    "cycle.main_run",
                    data={
                        "session_id": cycle_session_id,
                        "streaming": True,
                    },
                    disabled=not enable_tracing,
                ):
                    cycle = await run_once(
                        prompt=prompt,
                        model=model,
                        log_dir=log_dir,
                        enable_tracing=enable_tracing,
                        session_id=cycle_session_id,
                    )

                print(cycle.final_output)
                print(f"\nSession log: {cycle.log_path}")
                memory_info = cycle.memory_apply_result.summary_for_log()
                if memory_info.get("parse_error"):
                    print(f"[memory-error] {memory_info['parse_error']}")
                else:
                    print(
                        "[memory] "
                        f"short={memory_info.get('short_memory_latest_path')} "
                        f"behavior_updated={memory_info.get('behavior_updated')}"
                    )
                    for warning in memory_info.get("system_warnings") or []:
                        print(f"[memory-warning] {warning}")
                    if memory_info.get("behavior_diff_path"):
                        print(f"[memory] behavior_diff={memory_info['behavior_diff_path']}")
                if cycle.post_run_memory_error:
                    print(f"[post-run-reflection-error] {cycle.post_run_memory_error}")
                elif cycle.post_run_memory_result:
                    print(
                        "[post-run-reflection] "
                        f"journal={cycle.post_run_memory_result.journal_file_path} "
                        f"reflection={cycle.post_run_memory_result.reflection_latest_path} "
                        f"behavior_updated={cycle.post_run_memory_result.behavior_updated}"
                    )
        except Exception as exc:
            print(f"[run-error] {type(exc).__name__}: {exc}")

        sleep_s = _seconds_until_next_interval(interval_minutes)
        print(f"[scheduler] sleeping {sleep_s:.0f}s until next {interval_minutes}m cycle")
        await asyncio.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prompt against the trading agent.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="User prompt for the agent run. Optional if --prompt-file (or default prompt file) is available.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "Path to a text file containing the default prompt. "
            "Used when --prompt is omitted. Defaults to prompts/default_loop_prompt.txt."
        ),
    )
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

    try:
        effective_prompt, loaded_prompt_path = _resolve_prompt(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return

    if loaded_prompt_path is not None:
        print(f"[prompt] loaded from {loaded_prompt_path}")

    if args.loop:
        asyncio.run(
            run_loop(
                prompt=effective_prompt,
                model=args.model,
                log_dir=args.log_dir,
                enable_tracing=args.enable_tracing,
                interval_minutes=args.interval_minutes,
                market_hours_only=not args.ignore_market_hours,
                closed_poll_seconds=args.closed_poll_seconds,
            )
        )
        return

    cycle_session_id = uuid.uuid4().hex[:8]
    cycle_trace = _build_cycle_trace(
        session_id=cycle_session_id,
        model=args.model,
        enable_tracing=args.enable_tracing,
        loop=False,
    )
    with cycle_trace:
        if args.enable_tracing:
            with custom_span(
                "cycle.main_run",
                data={"session_id": cycle_session_id, "streaming": True},
                disabled=False,
            ):
                cycle = asyncio.run(
                    run_once(
                        prompt=effective_prompt,
                        model=args.model,
                        log_dir=args.log_dir,
                        enable_tracing=args.enable_tracing,
                        session_id=cycle_session_id,
                    )
                )
        else:
            cycle = asyncio.run(
                run_once(
                    prompt=effective_prompt,
                    model=args.model,
                    log_dir=args.log_dir,
                    enable_tracing=args.enable_tracing,
                    session_id=cycle_session_id,
                )
            )
    print(cycle.final_output)
    print(f"\nSession log: {cycle.log_path}")
    memory_info = cycle.memory_apply_result.summary_for_log()
    if memory_info.get("parse_error"):
        print(f"[memory-error] {memory_info['parse_error']}")
    else:
        print(
            "[memory] "
            f"short={memory_info.get('short_memory_latest_path')} "
            f"behavior_updated={memory_info.get('behavior_updated')}"
        )
        for warning in memory_info.get("system_warnings") or []:
            print(f"[memory-warning] {warning}")
        if memory_info.get("behavior_diff_path"):
            print(f"[memory] behavior_diff={memory_info['behavior_diff_path']}")
    if cycle.post_run_memory_error:
        print(f"[post-run-reflection-error] {cycle.post_run_memory_error}")
    elif cycle.post_run_memory_result:
        print(
            "[post-run-reflection] "
            f"journal={cycle.post_run_memory_result.journal_file_path} "
            f"reflection={cycle.post_run_memory_result.reflection_latest_path} "
            f"behavior_updated={cycle.post_run_memory_result.behavior_updated}"
        )


if __name__ == "__main__":
    main()
