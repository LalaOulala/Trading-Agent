from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agent_trade_sdk.config import Settings


AssetClass = Literal["equities", "rates", "fx", "commodities", "crypto", "credit", "vol"]
Impact = Literal["low", "medium", "high"]
Regime = Literal["risk_on", "risk_off", "mixed"]


class MarketMove(BaseModel):
    asset_class: AssetClass
    asset: str = Field(
        ...,
        description="Ex: S&P 500, Nasdaq 100, US10Y, DXY, WTI, Gold, BTC, VIX",
    )
    direction: Literal["up", "down", "flat"]
    magnitude: str | None = Field(None, description="Ex: +1.2%, -8 bps")
    timeframe: Literal["last_6h", "last_24h"]
    driver: str = Field(..., description="Cause principale, spécifique (pas généraliste)")
    why_it_matters_for_us_equities: str
    evidence_ids: list[int] = Field(
        default_factory=list,
        description="Indices vers search_results (0..n-1)",
    )


class BreakingNewsItem(BaseModel):
    timestamp_hint: str | None = Field(
        None,
        description="Heure/date si présente dans la source (ET ou UTC). Sinon laisse null.",
    )
    category: Literal[
        "earnings",
        "macro",
        "fed",
        "geopolitics",
        "policy_regulation",
        "mna",
        "company",
        "markets",
        "other",
    ]
    headline: str
    tickers_or_assets: list[str] = Field(
        default_factory=list,
        description="Tickers US si possible, sinon actifs (US10Y, DXY...)",
    )
    why_it_matters: str
    evidence_ids: list[int] = Field(default_factory=list)


class ThemeItem(BaseModel):
    theme: str = Field(..., description="Ex: 'Regional banks stress', 'AI capex', 'Oil supply shock'")
    sectors_involved: list[str] = Field(
        default_factory=list,
        description="Ex: Financials, Healthcare, Industrials...",
    )
    representative_tickers: list[str] = Field(
        default_factory=list,
        description="Tickers US diversifiés",
    )
    net_bias: Literal["bullish", "bearish", "mixed", "unclear"]
    reasoning: str
    evidence_ids: list[int] = Field(default_factory=list)


class WatchNextEvent(BaseModel):
    time_window_et: str = Field(..., description="Ex: 'Today 08:30 ET', 'Tue after close'")
    event_type: Literal[
        "macro_data",
        "earnings",
        "fed_speaker",
        "treasury_auction",
        "sec_filing",
        "company_event",
        "geopolitics",
        "other",
    ]
    event: str = Field(..., description="Nom précis (ex: CPI, FOMC minutes, AAPL earnings...)")
    tickers_or_assets: list[str] = Field(default_factory=list)
    what_to_watch: str = Field(
        ...,
        description="Le signal clef à surveiller (surprise vs consensus, guidance, wording...)",
    )
    expected_volatility: Impact
    evidence_ids: list[int] = Field(default_factory=list)


class NaturalLanguageBlock(BaseModel):
    headline: str = Field(..., description="1 phrase type desk headline")
    narrative: str = Field(..., description="8-12 lignes, orienté trading US, pas généraliste")
    bullet_takeaways: list[str] = Field(
        default_factory=list,
        description="6-10 bullets actionnables (sans dire 'achète/vends')",
    )


class MarketSnapshot(BaseModel):
    as_of_utc: str
    window: Literal["last_24h"]
    regime: Regime
    natural_language: NaturalLanguageBlock
    top_drivers: list[str] = Field(..., description="3 à 7 drivers principaux, spécifiques")
    breaking_news_last_6h: list[BreakingNewsItem] = Field(..., description="5 à 10 items récents")
    key_moves: list[MarketMove] = Field(..., description="8 à 14 mouvements cross-asset")
    themes_diversified: list[ThemeItem] = Field(..., description="6 à 10 thèmes, secteurs variés")
    watch_next_24h: list[WatchNextEvent] = Field(..., description="8 à 15 événements à surveiller")
    risk_flags: list[str]
    confidence: float = Field(..., ge=0, le=1)


def build_perplexity_snapshot_prompt() -> str:
    return (
        "Rôle: analyste de desk 'US equities + macro' orienté trading (intraday/swing court terme).\n"
        "Objectif: fournir un snapshot ULTRA-FRAIS (dernières 24h, avec emphase sur dernières 6h) utile à des agents.\n\n"
        "Exigences de fraîcheur:\n"
        "- Ne couvre QUE les dernières 24h.\n"
        "- Dans 'breaking_news_last_6h', mets des infos de dernière minute (idéalement <6h) et indique une "
        "'timestamp_hint' si trouvable.\n\n"
        "Exigences de diversité (IMPORTANT):\n"
        "- Equities US: au moins 6 tickers/événements venant de secteurs différents (pas plus de 2 items Tech/AI).\n"
        "- Inclure aussi: rates (UST), FX (USD/DXY), commodities (oil ou gold), volatility (VIX) — au minimum 1 "
        "item chacun dans key_moves.\n"
        "- Varier les thèmes: earnings/guidance, macro data, Fed, geopolitique, régulation, M&A, credit/liquidity.\n\n"
        "Exigences de qualité (anti-généralités):\n"
        "- Pas de phrases vagues du style 'markets rose on optimism'. Donne des drivers précis + ce qui a changé.\n"
        "- Mets 'why_it_matters_for_us_equities' pour chaque key move.\n\n"
        "Sources:\n"
        "- N'inclus AUCUNE URL dans le JSON.\n"
        "- Utilise evidence_ids (0..n-1) qui pointent vers les search_results renvoyés par l'API.\n"
        "- Priorise des médias/organismes financiers reconnus et des sources spécialisées.\n\n"
        "Sortie:\n"
        "- Retourne STRICTEMENT un JSON valide conforme au schéma.\n"
    )


def _extract_message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return ""


def _extract_search_results(completion: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in getattr(completion, "search_results", None) or []:
        results.append(
            {
                "title": getattr(item, "title", None),
                "url": getattr(item, "url", None),
                "date": getattr(item, "date", None),
                "last_updated": getattr(item, "last_updated", None),
                "snippet": getattr(item, "snippet", None),
                "source": getattr(item, "source", None),
            }
        )
    return results


def _extract_citations(completion: Any) -> list[str]:
    citations = getattr(completion, "citations", None)
    if not citations:
        return []
    return [str(c) for c in citations]


def perplexity_market_snapshot_raw() -> dict[str, Any]:
    settings = Settings.from_env(require_openrouter=False)
    if not settings.perplexity_api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY (or PPLX_API_KEY) in environment.")

    # Local import keeps the project importable even if dependency is temporarily absent.
    from perplexity import Perplexity

    client = Perplexity(api_key=settings.perplexity_api_key)
    requested_at_utc = datetime.now(timezone.utc).isoformat()
    domain_denylist = [
        "-reddit.com",
        "-pinterest.com",
        "-quora.com",
        "-tiktok.com",
        "-instagram.com",
        "-facebook.com",
        "-wikipedia.org",
    ]

    completion = client.chat.completions.create(
        model=settings.perplexity_model,
        messages=[
            {"role": "system", "content": settings.perplexity_snapshot_system_prompt},
            {"role": "user", "content": build_perplexity_snapshot_prompt()},
        ],
        search_recency_filter=settings.perplexity_snapshot_search_recency,
        search_domain_filter=domain_denylist,
        web_search_options={"search_context_size": settings.perplexity_snapshot_search_context_size},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "market_snapshot_us_equities",
                "strict": True,
                "schema": MarketSnapshot.model_json_schema(),
            },
        },
        max_tokens=settings.perplexity_snapshot_max_tokens,
    )

    raw_content = _extract_message_content_text(completion.choices[0].message.content)
    if not raw_content:
        raise RuntimeError("Perplexity returned empty message content.")

    try:
        parsed = json.loads(raw_content)
        if not parsed.get("as_of_utc"):
            parsed["as_of_utc"] = requested_at_utc
        if not parsed.get("window"):
            parsed["window"] = "last_24h"
        snapshot = MarketSnapshot.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Perplexity snapshot response validation failed: {exc}") from exc

    return {
        "provider": "perplexity",
        "requested_at_utc": requested_at_utc,
        "model": settings.perplexity_model,
        "search_recency_filter": settings.perplexity_snapshot_search_recency,
        "search_domain_filter": domain_denylist,
        "snapshot": snapshot.model_dump(),
        "search_results": _extract_search_results(completion),
        "citations": _extract_citations(completion),
    }


def compact_perplexity_snapshot_for_prompt(payload: dict[str, Any], max_sources: int = 5) -> dict[str, Any]:
    snapshot = payload.get("snapshot") or {}
    search_results = payload.get("search_results") or []

    compact_sources: list[dict[str, Any]] = []
    for item in search_results[:max_sources]:
        if not isinstance(item, dict):
            continue
        compact_sources.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "date": item.get("date") or item.get("last_updated"),
                "source": item.get("source"),
            }
        )

    return {
        "provider": payload.get("provider"),
        "requested_at_utc": payload.get("requested_at_utc"),
        "model": payload.get("model"),
        "search_recency_filter": payload.get("search_recency_filter"),
        "snapshot": snapshot,
        "sources": compact_sources,
    }
