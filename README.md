# Agent Trade SDK (Phase 1)

Prototype de base pour un agent de trading simulé:
- Orchestration agentique avec `openai-agents-python`
- Modèle via OpenRouter en passant par `LitellmModel`
- Données de marché via `yfinance`
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

## 3) Tester Yahoo Finance (snapshot marché)

```bash
python scripts/test_yfinance_market.py --action snapshot --symbols-csv "SPY,QQQ,AAPL,TSLA,NVDA"
python scripts/test_yfinance_market.py --action quote --symbol SPY
python scripts/test_yfinance_market.py --action history --symbol SPY --period 1mo --interval 1d
```

## 4) Tester Alpaca (paper trading)

Recommandé: commencer par `--dry-run`.

```bash
python scripts/test_alpaca_order.py --action account
python scripts/test_alpaca_order.py --action buy --symbol SPY --notional 10 --dry-run
python scripts/test_alpaca_order.py --action buy --symbol SPY --notional 10
python scripts/test_alpaca_order.py --action positions
python scripts/test_alpaca_order.py --action close --symbol SPY
```

## 5) Lancer un run agent unique

```bash
python -m agent_trade_sdk.runner --prompt "Analyse AAPL et propose une action avec justification."
```

Le runner V1 crée un log Markdown de session (tracing désactivé par défaut) dans `logs/`.
Pour changer le dossier:

```bash
python -m agent_trade_sdk.runner --prompt "..." --log-dir logs
```

Pour réactiver explicitement le tracing SDK:

```bash
python -m agent_trade_sdk.runner --prompt "..." --enable-tracing
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
