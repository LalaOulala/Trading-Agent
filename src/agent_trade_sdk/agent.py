from __future__ import annotations

from pathlib import Path

from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

from agent_trade_sdk.config import Settings
from agent_trade_sdk.tools.search import social_signal_search, web_search_tavily
from agent_trade_sdk.tools.trading import (
    close_open_position,
    get_account_snapshot,
    list_open_positions,
    open_short_position,
    place_market_order,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
SOUL_PATH = ROOT_DIR / "SOUL.md"


def _load_soul_text(path: Path = SOUL_PATH) -> str:
    if not path.exists():
        return "You are a risk-aware paper trading agent."
    return path.read_text(encoding="utf-8").strip()


def build_trading_agent(model_name: str | None = None) -> Agent:
    settings = Settings.from_env(require_openrouter=True)
    model = LitellmModel(
        model=model_name or settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )

    instructions = (
        f"{_load_soul_text()}\n\n"
        "Execution protocol:\n"
        "1) Gather evidence with search tools before placing an order.\n"
        "2) Explain rationale, risk, and invalidation level.\n"
        "3) Use trading tools only on paper account.\n"
        "4) If confidence is low, do not trade."
    )

    return Agent(
        name="SimulatedTradingAgent",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(include_usage=True),
        tools=[
            web_search_tavily,
            social_signal_search,
            get_account_snapshot,
            list_open_positions,
            place_market_order,
            open_short_position,
            close_open_position,
        ],
    )
