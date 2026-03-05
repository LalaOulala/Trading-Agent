# Agent Trade SDK

Agent de trading paper avec boucle agentique moderne:
- `Observe -> Orient -> Decide -> Act -> Reflect`
- Orchestration OpenAI Agents SDK
- Réflexion inter-session injectée au prochain run
- Guardrails anti-inaction + traçage des sources

Principaux composants:
- Orchestration agentique avec `openai-agents-python`
- Modèle via OpenRouter en passant par `LitellmModel`
- Données de marché via `yfinance`
- Tools de recherche web/social via Tavily
- Snapshot macro/news enrichi via Perplexity (si clé disponible)
- Tools de paper trading via Alpaca

## Boucle agentique

1. Snapshot pré-run (portfolio, market data, news, Perplexity, horloge marché).
2. Scoring qualité des sources (Tavily/Perplexity).
3. Guardrails déterministes anti-inaction.
4. Run agent d'exécution (tools + décision JSON).
5. Persistance mémoire courte forensic.
6. Run agent de réflexion inter-session.
7. Injection de `memory/reflection/latest.json` au run suivant.

## Setup rapide

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Remplis ensuite `.env` avec tes clés.

## Tester Tavily

```bash
python scripts/test_tavily_search.py --query "NVIDIA earnings market reaction today"
```

## Tester Yahoo Finance (snapshot marché)

```bash
python scripts/test_yfinance_market.py --action snapshot --symbols-csv "SPY,QQQ,AAPL,TSLA,NVDA"
python scripts/test_yfinance_market.py --action quote --symbol SPY
python scripts/test_yfinance_market.py --action history --symbol SPY --period 1mo --interval 1d
```

## Tester Alpaca (paper trading)

Recommandé: commencer par `--dry-run`.

```bash
python scripts/test_alpaca_order.py --action account
python scripts/test_alpaca_order.py --action buy --symbol SPY --notional 10 --dry-run
python scripts/test_alpaca_order.py --action buy --symbol SPY --notional 10
python scripts/test_alpaca_order.py --action positions
python scripts/test_alpaca_order.py --action close --symbol SPY
```

## Lancer un run agent unique

```bash
python -m agent_trade_sdk.runner --prompt "Analyse AAPL et propose une action avec justification."
```

Le runner crée:
- un log Markdown de session dans `logs/sessions/`
- un journal inter-session dans `logs/journals/`
- une mémoire réflexion injectée au run suivant dans `memory/reflection/latest.json`

Le tracing SDK reste désactivé par défaut.
Pour changer le dossier:

```bash
python -m agent_trade_sdk.runner --prompt "..." --log-dir logs
```

Pour réactiver explicitement le tracing SDK:

```bash
python -m agent_trade_sdk.runner --prompt "..." --enable-tracing
```

## Tracing OpenAI

```bash
python -m agent_trade_sdk.runner --prompt "..." --enable-tracing
```

Les spans incluent notamment:
- appels/sorties Tavily et Perplexity au pré-run
- qualité des sources
- attribution source -> décision
- statut de la réflexion post-run

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Notes de sécurité

- Utilise uniquement des clés paper trading Alpaca.
- Ce projet est éducatif. Aucune recommandation financière.
- Mets des limites strictes dans `SOUL.md` + variables d'environnement (`TRADING_MAX_NOTIONAL_USD`, `TRADING_ALLOWED_SYMBOLS`).

## Fichiers importants

- Architecture: `docs/architecture.md`
- Diagramme classes: `docs/diagrams/trading_agent_class_diagram.puml`
- Identité stable: `SOUL.md`
- Mémoire longue adaptive: `behavior.md`
- Réflexion injectée: `memory/reflection/latest.json`
