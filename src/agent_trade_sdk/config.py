from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an int, got: {raw}") from exc


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(items)


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    openai_tracing_api_key: str | None
    perplexity_api_key: str | None
    perplexity_model: str
    perplexity_snapshot_search_recency: str
    perplexity_snapshot_search_context_size: str
    perplexity_snapshot_max_tokens: int
    perplexity_snapshot_system_prompt: str
    tavily_api_key: str | None
    social_search_mode: str
    social_search_sites: tuple[str, ...]
    alpaca_api_key: str | None
    alpaca_secret_key: str | None
    alpaca_paper: bool
    alpaca_base_url: str
    trading_allowed_symbols: tuple[str, ...]
    trading_max_notional_usd: float | None

    @classmethod
    def from_env(cls, require_openrouter: bool = True) -> "Settings":
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if require_openrouter and not openrouter_api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment.")

        return cls(
            openrouter_api_key=openrouter_api_key,
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_tracing_api_key=os.getenv("OPENAI_TRACING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY"),
            perplexity_model=os.getenv("PERPLEXITY_MODEL", "sonar"),
            perplexity_snapshot_search_recency=os.getenv(
                "PERPLEXITY_SNAPSHOT_SEARCH_RECENCY", "day"
            ),
            perplexity_snapshot_search_context_size=os.getenv(
                "PERPLEXITY_SNAPSHOT_SEARCH_CONTEXT_SIZE", "medium"
            ),
            perplexity_snapshot_max_tokens=_env_int("PERPLEXITY_SNAPSHOT_MAX_TOKENS", 1200),
            perplexity_snapshot_system_prompt=os.getenv(
                "PERPLEXITY_SNAPSHOT_SYSTEM_PROMPT",
                "Réponds en français, concis, style desk.",
            ),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            social_search_mode=os.getenv("SOCIAL_SEARCH_MODE", "tavily_sites"),
            social_search_sites=_env_csv("SOCIAL_SEARCH_SITES", "x.com,reddit.com,stocktwits.com"),
            alpaca_api_key=os.getenv("ALPACA_API_KEY"),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
            alpaca_paper=_env_bool("ALPACA_PAPER", True),
            alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            trading_allowed_symbols=tuple(
                symbol.upper()
                for symbol in _env_csv("TRADING_ALLOWED_SYMBOLS", "")
                if symbol.strip()
            ),
            trading_max_notional_usd=_env_float("TRADING_MAX_NOTIONAL_USD"),
        )
