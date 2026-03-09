from __future__ import annotations

import re
from datetime import datetime, timezone
from statistics import median
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field


ProviderLiteral = Literal["tavily", "perplexity"]

TRUSTED_FINANCE_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "wsj.com",
    "ft.com",
    "marketwatch.com",
    "seekingalpha.com",
    "investing.com",
    "finance.yahoo.com",
    "federalreserve.gov",
    "sec.gov",
    "nytimes.com",
    "apnews.com",
}

FINANCE_KEYWORDS = {
    "earnings",
    "guidance",
    "revenue",
    "eps",
    "fed",
    "fomc",
    "inflation",
    "cpi",
    "pce",
    "yields",
    "treasury",
    "dow",
    "s&p",
    "nasdaq",
    "sector",
    "stocks",
    "equities",
    "options",
    "volatility",
    "vix",
    "buyback",
    "ipo",
    "merger",
    "acquisition",
    "tariff",
    "oil",
    "brent",
}


class SourceQualityReport(BaseModel):
    provider: ProviderLiteral
    freshness_hours_median: float | None = None
    duplicate_ratio: float = Field(0.0, ge=0.0, le=1.0)
    trusted_domain_ratio: float = Field(0.0, ge=0.0, le=1.0)
    finance_relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    usable_for_decision: bool = False
    diagnostics: list[str] = Field(default_factory=list)


def _safe_float_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _normalize_domain(url: str | None) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_trusted_domain(url: str | None) -> bool:
    domain = _normalize_domain(url)
    if not domain:
        return False
    return any(domain == ref or domain.endswith(f".{ref}") for ref in TRUSTED_FINANCE_DOMAINS)


def _normalize_title(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _coerce_dt(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    if " " in text and "T" not in text:
        candidates.append(text.replace(" ", "T"))
        if not text.endswith("Z"):
            candidates.append(text.replace(" ", "T") + "Z")

    for item in candidates:
        try:
            parsed = datetime.fromisoformat(item)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _finance_relevance_score_from_text(texts: list[str]) -> float:
    if not texts:
        return 0.0
    tokens = " ".join(texts).lower()
    keyword_hits = sum(1 for kw in FINANCE_KEYWORDS if kw in tokens)
    ticker_hits = len(set(re.findall(r"\b[A-Z]{2,5}\b", " ".join(texts))))
    base = min(1.0, (keyword_hits / 10.0) + min(0.4, ticker_hits / 25.0))
    return round(base, 3)


def _compute_quality(
    *,
    provider: ProviderLiteral,
    items: list[dict[str, Any]],
) -> SourceQualityReport:
    diagnostics: list[str] = []
    total = len(items)
    if total == 0:
        return SourceQualityReport(
            provider=provider,
            diagnostics=["no_results"],
            usable_for_decision=False,
        )

    seen_keys: set[tuple[str, str]] = set()
    unique_count = 0
    trusted_count = 0
    freshness_hours: list[float] = []
    relevance_texts: list[str] = []

    now = datetime.now(timezone.utc)
    for item in items:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip() or None
        title_key = _normalize_title(title)
        domain_key = _normalize_domain(url)
        dedupe_key = (title_key, domain_key)
        if dedupe_key not in seen_keys:
            unique_count += 1
            seen_keys.add(dedupe_key)

        if _is_trusted_domain(url):
            trusted_count += 1

        timestamp_value = (
            item.get("published_date")
            or item.get("date")
            or item.get("last_updated")
            or item.get("timestamp")
        )
        parsed_dt = _coerce_dt(str(timestamp_value) if timestamp_value is not None else None)
        if parsed_dt is not None:
            freshness_hours.append(max(0.0, (now - parsed_dt).total_seconds() / 3600.0))

        snippet = str(item.get("snippet") or "")
        content = str(item.get("content") or "")
        relevance_texts.append(" ".join(part for part in (title, snippet, content) if part).strip())

    duplicate_ratio = round(1.0 - _safe_float_ratio(unique_count, total), 3)
    trusted_ratio = round(_safe_float_ratio(trusted_count, total), 3)
    freshness_median = round(median(freshness_hours), 2) if freshness_hours else None
    relevance = _finance_relevance_score_from_text(relevance_texts)

    if freshness_median is None:
        diagnostics.append("missing_publish_dates")
    elif freshness_median > 48:
        diagnostics.append("stale_results")

    if duplicate_ratio > 0.4:
        diagnostics.append("high_duplicates")
    if trusted_ratio < 0.3:
        diagnostics.append("low_trusted_sources")
    if relevance < 0.45:
        diagnostics.append("low_finance_relevance")

    usable = total >= 2 and duplicate_ratio <= 0.6 and relevance >= 0.45
    if freshness_median is not None and freshness_median > 72:
        usable = False
    if not usable and "not_usable_for_decision" not in diagnostics:
        diagnostics.append("not_usable_for_decision")

    return SourceQualityReport(
        provider=provider,
        freshness_hours_median=freshness_median,
        duplicate_ratio=duplicate_ratio,
        trusted_domain_ratio=trusted_ratio,
        finance_relevance_score=relevance,
        usable_for_decision=usable,
        diagnostics=diagnostics,
    )


def evaluate_tavily_quality(news_payload: dict[str, Any] | None) -> SourceQualityReport:
    payload = news_payload if isinstance(news_payload, dict) else {}
    results_raw = payload.get("results") or []
    items: list[dict[str, Any]] = []
    if isinstance(results_raw, list):
        for item in results_raw:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_date": item.get("published_date"),
                    "content": item.get("content"),
                }
            )
    return _compute_quality(provider="tavily", items=items)


def evaluate_perplexity_quality(perplexity_payload: dict[str, Any] | None) -> SourceQualityReport:
    payload = perplexity_payload if isinstance(perplexity_payload, dict) else {}
    results_raw = payload.get("search_results") or []
    items: list[dict[str, Any]] = []
    if isinstance(results_raw, list):
        for item in results_raw:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "date": item.get("date"),
                    "last_updated": item.get("last_updated"),
                    "snippet": item.get("snippet"),
                }
            )
    report = _compute_quality(provider="perplexity", items=items)
    payload_diagnostics = payload.get("quality_diagnostics")
    if isinstance(payload_diagnostics, list):
        existing = set(report.diagnostics)
        for item in payload_diagnostics:
            text = str(item).strip()
            if text and text not in existing:
                report.diagnostics.append(text)
                existing.add(text)
    return report


def evaluate_snapshot_source_quality(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    news_payload = snapshot.get("news") if isinstance(snapshot, dict) else {}
    perplexity_payload = (
        snapshot.get("perplexity_market_research")
        if isinstance(snapshot, dict)
        else {}
    )
    tavily = evaluate_tavily_quality(news_payload if isinstance(news_payload, dict) else {})
    perplexity = evaluate_perplexity_quality(
        perplexity_payload if isinstance(perplexity_payload, dict) else {}
    )
    return {
        "tavily": tavily.model_dump(),
        "perplexity": perplexity.model_dump(),
    }
