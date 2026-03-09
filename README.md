# Agent Trade SDK

Agent de trading paper avec boucle agentique moderne:
- `Observe -> Orient -> Decide -> Act -> Reflect`
- Orchestration OpenAI Agents SDK
- Réflexion inter-session injectée au prochain run
- Guardrails anti-inaction + traçage des sources
- Résilience inter-run (fallback mémoire si sortie non-JSON)

Principaux composants:
- Orchestration agentique avec `openai-agents-python`
- Modèle via OpenRouter en passant par `LitellmModel`
- Données de marché via `yfinance`
- Tools de recherche web/social via Tavily
- Snapshot macro/news enrichi via Perplexity (si clé disponible)
- Perplexity en langage naturel + résumé 5 lignes persisté entre runs
- Tools de paper trading via Alpaca

## Boucle agentique

1. Snapshot pré-run (portfolio, market data, news, Perplexity, horloge marché).
2. Scoring qualité des sources (Tavily/Perplexity).
3. Guardrails déterministes anti-inaction.
4. Run agent d'exécution (tools + décision JSON).
5. Persistance mémoire courte forensic.
6. Run agent de réflexion inter-session.
7. Injection de `memory/reflection/latest.json` au run suivant.

## Variables d'environnement importantes

- `REFLECTION_ALLOW_BEHAVIOR_AUTOWRITE` (défaut `false`): autorise l'écriture automatique de `behavior.md` en post-run.
- `LITELLM_LOG` (défaut `ERROR`): niveau de logs LiteLLM (surchargable).
- `LITELLM_SUPPRESS_DEBUG_INFO` (défaut `true`): réduit le bruit runtime.

## Setup rapide

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Remplis ensuite `.env` avec tes clés.

## Prompt par défaut (`prompts/`)

Le runner peut lire un prompt depuis un fichier, pour éviter de le repasser à chaque lancement.

- Fichier par défaut: `prompts/default_loop_prompt.txt`
- Utilisé automatiquement si `--prompt` est omis
- Surchargable via `--prompt-file`

Exemple d'override:

```bash
python -m agent_trade_sdk.runner --prompt-file prompts/default_loop_prompt.txt
```

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
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner --prompt "Analyse AAPL et propose une action avec justification."
```

Si `--prompt` est omis, le runner charge automatiquement:

- `prompts/default_loop_prompt.txt`

Exemple:

```bash
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner
```

Le runner crée:
- un log Markdown de session dans `logs/sessions/`
- un journal inter-session dans `logs/journals/`
- une mémoire réflexion injectée au run suivant dans `memory/reflection/latest.json`

Le tracing SDK reste désactivé par défaut.
Pour changer le dossier:

```bash
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner --prompt "..." --log-dir logs
```

Pour réactiver explicitement le tracing SDK:

```bash
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner --prompt "..." --enable-tracing
```

## Lancer en boucle sans repasser le prompt

Le mode loop peut démarrer sans `--prompt` si le fichier `prompts/default_loop_prompt.txt` existe.

```bash
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner --loop --interval-minutes 15 --enable-tracing
```

On peut aussi forcer un autre fichier:

```bash
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner --loop --prompt-file /chemin/vers/mon_prompt.txt
```

## Tracing OpenAI (à terminer)

```bash
PYTHONPATH=src .venv/bin/python -m agent_trade_sdk.runner --prompt "..." --enable-tracing
```

Les spans incluent notamment:
- appels/sorties Tavily et Perplexity au pré-run
- qualité des sources
- attribution source -> décision
- statut de la réflexion post-run
- payloads de spans compactés/clampés (<10KB) pour éviter les erreurs tracing

## Contrat Perplexity

- Sortie demandée en langage naturel (pas de JSON strict).
- Le prompt impose:
  - continuité depuis le run précédent,
  - nouveautés fraîches (<6h),
  - thèmes nouveaux (ouverture obligatoire),
  - résumé final 5 lignes.
- Le résumé est persisté dans `memory/perplexity/latest_summary.json` et réinjecté au run suivant.
- Diagnostic `low_novelty` si recouvrement thématique trop élevé avec le résumé précédent.

## Résilience mémoire

- Le parse de sortie agent est robuste (bloc `json` + extraction objet équilibré).
- Si parse invalide:
  - mémoire courte fallback écrite (pas de crash loop),
  - réflexion lancée sur contexte sanitizé,
  - auto-update `behavior.md` bloqué.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Notes de sécurité

- Utilise uniquement des clés paper trading Alpaca.
- Ce projet est éducatif. Aucune recommandation financière.
- Limites strictes dans `SOUL.md`.

## Fichiers importants

- Architecture: `docs/architecture.md`
- Diagramme classes: `docs/diagrams/trading_agent_class_diagram.puml`
- Identité stable: `SOUL.md`
- Mémoire longue adaptive: `behavior.md`
- Réflexion injectée: `memory/reflection/latest.json`
- Continuité Perplexity: `memory/perplexity/latest_summary.json`
