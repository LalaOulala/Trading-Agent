# Architecture

## Objectif

Construire un socle propre pour un agent autonome de trading simulé:
- recherche d'information web/social
- décision pilotée par LLM
- exécution d'ordres en paper trading
- boucle agentique extensible via un profil comportemental (`SOUL.md` + `behavior.md`)
- introspection inter-session exploitable par le run suivant

## Stack retenue

1. **Orchestration agentique**: `openai-agents-python` (primitives simples: `Agent`, `Runner`, `@function_tool`, loop intégré)
2. **Modèle LLM**: `LitellmModel` vers OpenRouter
3. **Données de marché**: Yahoo Finance via `yfinance` (quote, historique, snapshot)
4. **Recherche web/social**: Tavily (phase 1), requêtes sociales via filtres de domaines
5. **Execution trading**: `alpaca-py` (paper trading API)
6. **Config**: `.env` + `python-dotenv`
7. **Tracing**: OpenAI tracing spans (pré-run, tools, réflexion post-run)
8. **Réflexion**: second agent post-run avec sortie structurée (`ReflectionConclusion`)

## Philosophie agentique moderne

- Boucle cible: `Observe -> Orient -> Decide -> Act -> Reflect`.
- Séparer exécution et introspection pour éviter l’auto-ancrage.
- Garder les garde-fous critiques en code déterministe.
- Contraindre les tools avec schémas stricts et sorties fidèles.
- Conserver une traçabilité de bout en bout: source -> interprétation -> décision.

## Boucle d’exécution

1. Chargement mémoire:
- `behavior.md` (long terme adaptatif)
- `memory/reflection/latest.json` (conclusion inter-session prioritaire)
- fallback compact depuis `memory/short/latest.json`

2. Snapshot pré-run:
- horloge marché (`market_clock`)
- portfolio Alpaca
- market snapshot yfinance
- news Tavily
- recherche Perplexity

3. Évaluation de qualité des sources:
- `SourceQualityReport` pour Tavily et Perplexity
- métriques: fraîcheur, duplicats, domaines fiables, pertinence finance

4. Guardrails anti-inaction:
- détection deterministic (`stall_score`) sur historique de runs
- règles correctrices injectées avant décision

5. Exécution agent principal:
- tools de marché/recherche/trading
- sortie JSON stricte (décision + mémoire forensic)

6. Réflexion post-run:
- journal markdown inter-session (`logs/journals/`)
- conclusion structurée (`memory/reflection/latest.json`)
- mise à jour optionnelle de `behavior.md` uniquement
- `SOUL.md` reste immuable

## Modules clés

- `/Users/lala/CascadeProjects/agent_trade_sdk/src/agent_trade_sdk/runner.py`
- `/Users/lala/CascadeProjects/agent_trade_sdk/src/agent_trade_sdk/post_run_memory.py`
- `/Users/lala/CascadeProjects/agent_trade_sdk/src/agent_trade_sdk/reflection_memory.py`
- `/Users/lala/CascadeProjects/agent_trade_sdk/src/agent_trade_sdk/source_quality.py`
- `/Users/lala/CascadeProjects/agent_trade_sdk/src/agent_trade_sdk/strategy_guardrails.py`
- `/Users/lala/CascadeProjects/agent_trade_sdk/src/agent_trade_sdk/session_log.py`

## Format des tools (OpenAI Agents SDK)

Le format recommandé en Python:
- déclarer une fonction typée
- ajouter docstring claire
- décorer avec `@function_tool`

Le SDK génère automatiquement le schéma tool (args + validation) à partir des type hints/docstring.

Exemple:

```python
@function_tool
def place_market_order(
    symbol: str,
    side: Literal["buy", "sell"],
    qty: float | None = None,
    notional: float | None = None,
) -> str:
    """Submit an Alpaca paper market order."""
```

## Tracing

- Les spans couvrent:
- pré-run (sources, qualité, guardrails)
- exécution runtime (tool calls, outputs, reasoning/message summaries)
- attribution source dans la décision finale
- réflexion post-run (journal, conclusion, update behavior)
