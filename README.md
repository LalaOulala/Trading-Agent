# Agent Trade SDK (Phase 1)

Prototype de base pour un agent de trading simulé:
- Orchestration agentique avec `openai-agents-python`
- Modèle via OpenRouter en passant par `LitellmModel`
- Tools de recherche web/social via Tavily
- Tools de paper trading via Alpaca

Ce dépôt couvre la **phase 1**: choix techno, structure, diagramme de classes, scripts de test.

## 1) Setup rapide

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Remplis ensuite `.env` avec tes clés.

## 2) Tester Tavily

```bash
python scripts/test_tavily_search.py --query "NVIDIA earnings market reaction today"
```

## 3) Tester Alpaca (paper trading)

Recommandé: commencer par `--dry-run`.

```bash
python scripts/test_alpaca_order.py --action account
python scripts/test_alpaca_order.py --action buy --symbol SPY --notional 10 --dry-run
python scripts/test_alpaca_order.py --action buy --symbol SPY --notional 10
python scripts/test_alpaca_order.py --action positions
python scripts/test_alpaca_order.py --action close --symbol SPY
```

## 4) Lancer un run agent unique

```bash
python -m agent_trade_sdk.runner --prompt "Analyse AAPL et propose une action avec justification."
```

## Notes de sécurité

- Utilise uniquement des clés paper trading Alpaca.
- Ce projet est éducatif. Aucune recommandation financière.
- Mets des limites strictes dans `SOUL.md` + variables d'environnement (`TRADING_MAX_NOTIONAL_USD`, `TRADING_ALLOWED_SYMBOLS`).

## Fichiers importants

- Architecture: `docs/architecture.md`
- Diagramme classes: `docs/diagrams/trading_agent_class_diagram.puml`
- Profil comportemental agent: `SOUL.md`
- Exemple variables: `.env.example`
