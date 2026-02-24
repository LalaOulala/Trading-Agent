from __future__ import annotations

from pathlib import Path

from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

from agent_trade_sdk.config import Settings
from agent_trade_sdk.tools.market_data import (
    get_market_quote,
    get_market_snapshot,
    get_price_history,
)
from agent_trade_sdk.tools.search import web_search_tavily
from agent_trade_sdk.tools.trading import (
    close_open_position,
    get_account_snapshot,
    list_open_positions,
    open_short_position,
    place_market_order,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
SOUL_PATH = ROOT_DIR / "SOUL.md"
BEHAVIOR_PATH = ROOT_DIR / "behavior.md"


def _load_soul_text(path: Path = SOUL_PATH) -> str:
    if not path.exists():
        return "You are a risk-aware paper trading agent."
    return path.read_text(encoding="utf-8").strip()


def _load_behavior_text(path: Path = BEHAVIOR_PATH) -> str:
    if not path.exists():
        return (
            "# Behavior - Adaptive Trading Profile\n\n"
            "## Current Style\n\n"
            "- Not initialized yet.\n"
        )
    return path.read_text(encoding="utf-8").strip()


def build_trading_agent(model_name: str | None = None) -> Agent:
    settings = Settings.from_env(require_openrouter=True)
    model = LitellmModel(
        model=model_name or settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )

    instructions = (
        "Tu es un agent de paper trading autonome sur actions US.\n\n"
        "Tu reçois 2 mémoires de long terme distinctes:\n"
        "1) SOUL.md (stable, non modifiable): identité, principes immuables, garde-fous de caractère.\n"
        "2) behavior.md (modifiable): style de trading actuel, objectifs, leçons, hypothèses, questions "
        "ouvertes.\n\n"
        "Tu ne dois jamais proposer de modifier SOUL.md. Tu peux proposer de réécrire behavior.md en fin de "
        "session si tu as appris quelque chose d'utile et durable.\n\n"
        "=== SOUL.md (stable) ===\n"
        f"{_load_soul_text()}\n\n"
        "=== behavior.md (modifiable, mémoire longue) ===\n"
        f"{_load_behavior_text()}\n\n"
        "Workflow de session (obligatoire):\n"
        "1) Lire la mémoire courte précédente (si fournie) et identifier les pièges à éviter.\n"
        "2) Lire le snapshot de pré-run et formuler des lignes directrices pour ce cycle.\n"
        "3) Vérifier/rafraîchir avec les tools les faits critiques avant toute décision.\n"
        "4) Donner une intention claire avant chaque tool call (ce que tu cherches à vérifier/invalider).\n"
        "5) Décider une seule action de trading: BUY, SELL, SHORT, CLOSE ou NO_TRADE.\n"
        "6) Exécuter via tools Alpaca seulement si la conviction est suffisante et si les garde-fous sont "
        "respectés.\n"
        "7) Produire en sortie un JSON strict conforme au contrat ci-dessous, incluant résumé de session, "
        "autocritique et directives pour la prochaine session.\n\n"
        "Contrat de sortie JSON (obligatoire, EXACT en top-level):\n"
        "{\n"
        '  "trading_decision": {\n'
        '    "action": "BUY|SELL|SHORT|CLOSE|NO_TRADE",\n'
        '    "symbol": "string|null",\n'
        '    "confidence": "string|null",\n'
        '    "rationale": "string",\n'
        '    "signal_principal": "string|null",\n'
        '    "risk_identified": "string|null",\n'
        '    "invalidation_condition": "string|null",\n'
        '    "executed_order": "object|null"\n'
        "  },\n"
        '  "session_summary": "string",\n'
        '  "decision_report": "string",\n'
        '  "self_critique": "string",\n'
        '  "pitfalls_to_avoid_next_run": ["..."],\n'
        '  "next_session_directives": ["..."],\n'
        '  "open_questions": ["..."],\n'
        '  "long_memory_update_intent": {\n'
        '    "should_update_behavior": true|false,\n'
        '    "why": "string",\n'
        '    "update_summary": ["..."],\n'
        '    "updated_behavior_markdown": "full markdown string or null"\n'
        "  }\n"
        "}\n\n"
        "Si should_update_behavior=false, updated_behavior_markdown doit être null.\n"
        "Si should_update_behavior=true, updated_behavior_markdown doit contenir le contenu COMPLET de "
        "behavior.md (pas un patch)."
    )

    return Agent(
        name="SimulatedTradingAgent",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(include_usage=True),
        tools=[
            get_market_snapshot,
            get_market_quote,
            get_price_history,
            web_search_tavily,
            get_account_snapshot,
            list_open_positions,
            place_market_order,
            open_short_position,
            close_open_position,
        ],
    )
