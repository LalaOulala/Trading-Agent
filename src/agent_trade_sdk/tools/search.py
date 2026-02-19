from __future__ import annotations

import json
from typing import Any, Literal

from agents import function_tool
from tavily import TavilyClient

from agent_trade_sdk.config import Settings


SearchTopic = Literal["news", "general"]
SearchTimeRange = Literal["day", "week", "month", "year"]


def _build_tavily_client() -> TavilyClient:
    settings = Settings.from_env(require_openrouter=False)
    if not settings.tavily_api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in environment.")
    return TavilyClient(api_key=settings.tavily_api_key)


def _compact_search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact_results: list[dict[str, Any]] = []
    for item in payload.get("results", []) or []:
        compact_results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "score": item.get("score"),
                "published_date": item.get("published_date"),
                "content": (item.get("content") or "")[:500],
            }
        )
    return {
        "answer": payload.get("answer"),
        "query": payload.get("query"),
        "results": compact_results,
    }


def tavily_search_raw(
    query: str,
    max_results: int = 5,
    topic: SearchTopic = "news",
    time_range: SearchTimeRange = "day",
) -> dict[str, Any]:
    client = _build_tavily_client()
    payload = client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        time_range=time_range,
    )
    if isinstance(payload, dict):
        return payload
    return {"raw_response": payload}


@function_tool
def web_search_tavily(
    query: str,
    max_results: int = 5,
    topic: SearchTopic = "news",
    time_range: SearchTimeRange = "day",
) -> str:
    """Search the web with Tavily and return a compact JSON payload with sources."""

    payload = tavily_search_raw(
        query=query,
        max_results=max_results,
        topic=topic,
        time_range=time_range,
    )
    return json.dumps(_compact_search_payload(payload), ensure_ascii=False)


@function_tool
def social_signal_search(
    keyword: str,
    max_results: int = 5,
    time_range: SearchTimeRange = "day",
) -> str:
    """Search social chatter with Tavily by filtering to social domains from env config."""

    settings = Settings.from_env(require_openrouter=False)
    sites = settings.social_search_sites or ("x.com", "reddit.com", "stocktwits.com")
    site_filter = " OR ".join(f"site:{site}" for site in sites)
    query = f"{keyword} ({site_filter})"

    payload = tavily_search_raw(
        query=query,
        max_results=max_results,
        topic="news",
        time_range=time_range,
    )
    compact = _compact_search_payload(payload)
    compact["social_sites"] = list(sites)
    return json.dumps(compact, ensure_ascii=False)
