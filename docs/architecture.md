# Architecture - Phase 1

## Objectif

Construire un socle propre pour un agent autonome de trading simulé:
- recherche d'information web/social
- décision pilotée par LLM
- exécution d'ordres en paper trading
- boucle agentique extensible via un profil comportemental (`SOUL.md`)

## Stack retenue

1. **Orchestration agentique**: `openai-agents-python` (primitives simples: `Agent`, `Runner`, `@function_tool`, loop intégré)
2. **Modèle LLM**: `LitellmModel` vers OpenRouter
3. **Données de marché**: Yahoo Finance via `yfinance` (quote, historique, snapshot)
4. **Recherche web/social**: Tavily (phase 1), requêtes sociales via filtres de domaines
5. **Execution trading**: `alpaca-py` (paper trading API)
6. **Config**: `.env` + `python-dotenv`
7. **Journal V1**: logs Markdown par session (`logs/session_*.md`)

## Pourquoi Tavily en phase 1

- API simple et rapide à intégrer en tool.
- Résultats déjà orientés "agent workflows" (résumés, URLs).
- Permet de prototyper vite.

## Alternatives recherche (à évaluer en phase 2)

1. **Exa**: bon pour recherche sémantique profonde.
2. **SerpAPI / Brave Search API**: meilleur coverage moteur de recherche.
3. **Providers sociaux dédiés**:
   - X API (officielle)
   - Reddit API
   - StockTwits API

Stratégie recommandée: Tavily en fallback + connecteurs natifs par réseau social pour fiabilité.

## Structure de code

```text
agent_trade_sdk/
├── .env.example
├── SOUL.md
├── docs/
│   ├── architecture.md
│   └── diagrams/
│       └── trading_agent_class_diagram.puml
├── scripts/
│   ├── test_alpaca_order.py
│   ├── test_tavily_search.py
│   └── test_yfinance_market.py
└── src/
    └── agent_trade_sdk/
        ├── agent.py
        ├── config.py
        ├── runner.py
        └── tools/
            ├── market_data.py
            ├── search.py
            └── trading.py
```

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

## Flux cible (phase 1)

1. Prompt utilisateur
2. Agent planifie l'enquête
3. Tools Yahoo Finance (snapshot/quote/historique)
4. Tool Tavily web/social
5. Synthèse + décision
6. Tool Alpaca (paper order)
7. Logging de la décision et de l'exécution
