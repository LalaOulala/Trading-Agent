#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from perplexity import Perplexity
from pydantic import BaseModel, Field, ValidationError


# Charge la config locale (.env à la racine du projet)
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(ENV_FILE)


def get_perplexity_api_key() -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Perplexity API key introuvable. Ajoute PERPLEXITY_API_KEY=... "
            "(ou PPLX_API_KEY=...) dans .env"
        )
    return api_key


class MarketMove(BaseModel):
    asset: str = Field(
        ...,
        description="Nom de l'actif ou indice (ex: S&P 500, EUR/USD, US10Y, Brent, BTC)",
    )
    direction: Literal["up", "down", "flat"]
    magnitude: Optional[str] = Field(
        None,
        description="Amplitude si connue (ex: +1.2%, -8 bps)",
    )
    driver: str = Field(..., description="Cause principale du mouvement")
    evidence_ids: List[int] = Field(
        default_factory=list,
        description="Indices vers search_results (0..n-1)",
    )


class MarketSnapshot(BaseModel):
    as_of_utc: str
    regime: Literal["risk_on", "risk_off", "mixed"]
    summary: str = Field(..., description="Resume 5-8 lignes lisible humain")
    top_drivers: List[str]
    key_moves: List[MarketMove]
    catalysts_next_24h: List[str]
    risk_flags: List[str]
    confidence: float = Field(..., ge=0, le=1)


def build_prompt() -> str:
    return (
        "Tu es un analyste macro. Donne un snapshot des marchés financiers "
        "(actions, taux, FX, commodities, crypto).\n"
        "Objectif: un état des lieux actionnable pour des agents.\n\n"
        "Contraintes:\n"
        "- Retourne STRICTEMENT un JSON valide qui respecte le schéma fourni.\n"
        "- Pas d'URLs dans le JSON.\n"
        "- Pour chaque 'key_moves[*].evidence_ids', mets les indices (0..n-1) "
        "des résultats web pertinents.\n\n"
        "Contenu attendu:\n"
        "- regime: risk_on / risk_off / mixed\n"
        "- top_drivers: 3 à 7 drivers principaux\n"
        "- key_moves: 6 à 12 mouvements marquants cross-asset + driver\n"
        "- catalysts_next_24h: macro, résultats, banques centrales, géopolitique\n"
        "- risk_flags: ce qui peut invalider un scénario\n"
        "- confidence: 0..1\n"
    )


def _extract_text_content(content: object) -> str:
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


def _print_search_results(completion: object) -> None:
    search_results = getattr(completion, "search_results", None)
    if not search_results:
        return

    print("\n--- SEARCH RESULTS (sources) ---")
    for i, result in enumerate(search_results):
        # Compatible objets Pydantic du SDK Perplexity
        title = getattr(result, "title", None)
        url = getattr(result, "url", None)
        date = getattr(result, "date", None) or getattr(result, "last_updated", None)
        if date:
            print(f"[{i}] {title} — {url} ({date})")
        else:
            print(f"[{i}] {title} — {url}")


def _print_citations(completion: object) -> None:
    citations = getattr(completion, "citations", None)
    if not citations:
        return

    print("\n--- CITATIONS ---")
    for citation in citations:
        print(citation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Perplexity pour générer un snapshot de marché (hors logique projet)."
    )
    parser.add_argument("--model", default="sonar", help="Ex: sonar, sonar-pro")
    parser.add_argument(
        "--search-recency",
        default="hour",
        choices=["hour", "day", "week", "month", "year"],
        help="Freshness des résultats de recherche Perplexity.",
    )
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Affiche la réponse brute du modèle (avant validation).",
    )
    parser.add_argument(
        "--system-prompt",
        default="Réponds en français, style concis, orienté marchés.",
    )
    parser.add_argument(
        "--user-prompt",
        default=build_prompt(),
        help="Prompt utilisateur. Par défaut: snapshot cross-asset.",
    )
    args = parser.parse_args()

    client = Perplexity(api_key=get_perplexity_api_key())

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.user_prompt},
        ],
        search_recency_filter=args.search_recency,
        web_search_options={"search_context_size": "low"},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "market_snapshot",
                "strict": True,
                "schema": MarketSnapshot.model_json_schema(),
            },
        },
        max_tokens=args.max_tokens,
    )

    raw_content = _extract_text_content(completion.choices[0].message.content)
    as_of = datetime.now(timezone.utc).isoformat()

    if args.show_raw:
        print("--- RAW MODEL CONTENT ---")
        print(raw_content)
        print()

    try:
        data = json.loads(raw_content)
        data["as_of_utc"] = data.get("as_of_utc") or as_of
        snapshot = MarketSnapshot.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        print("Reponse non conforme / JSON invalide.")
        print(raw_content)
        raise SystemExit(1) from exc

    print(snapshot.model_dump_json(indent=2, ensure_ascii=False))
    _print_search_results(completion)
    _print_citations(completion)


if __name__ == "__main__":
    main()
