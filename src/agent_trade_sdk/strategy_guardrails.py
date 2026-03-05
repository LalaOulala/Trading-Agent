from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_trade_sdk.memory_architecture import SHORT_MEMORY_ARCHIVE_DIR


SECTOR_KEYWORDS: dict[str, set[str]] = {
    "semiconductors": {
        "nvda",
        "amd",
        "smh",
        "asml",
        "tsm",
        "intel",
        "intc",
        "avgo",
        "semis",
        "semiconducteurs",
    },
    "financials": {"xlf", "bac", "jpm", "gs", "ms", "wfc", "financial", "banques", "banks"},
    "energy": {"xle", "oil", "wti", "brent", "energy", "énergie"},
    "industrials": {"vrt", "industrial", "industrials", "ita", "boeing", "cat", "de"},
    "small_caps": {"iwm", "small caps", "small-cap", "mid caps"},
    "consumer": {"hd", "cost", "wmt", "xly", "xlp", "consumer"},
    "broad_index": {"spy", "qqq", "dia", "s&p", "nasdaq", "dow"},
}


class StallGuardrailResult(BaseModel):
    lookback_runs: int
    no_trade_streak: int = 0
    no_trade_ratio: float = Field(0.0, ge=0.0, le=1.0)
    total_unrealized_pl: float = 0.0
    concentration_ratio: float = Field(0.0, ge=0.0, le=1.0)
    explored_sector_breadth: int = 0
    stall_score: float = Field(0.0, ge=0.0, le=1.0)
    stall_flags: list[str] = Field(default_factory=list)
    hard_rules_next_run: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


def _to_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return 0.0


def _load_recent_short_memory_records(lookback_runs: int) -> list[dict[str, Any]]:
    if not SHORT_MEMORY_ARCHIVE_DIR.exists():
        return []
    files = sorted(
        [path for path in SHORT_MEMORY_ARCHIVE_DIR.glob("short_memory_*.json") if path.is_file()],
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for path in files[: max(lookback_runs * 2, lookback_runs)]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        records.append(payload)
        if len(records) >= lookback_runs:
            break
    return records


def _compute_no_trade_metrics(records: list[dict[str, Any]]) -> tuple[int, float]:
    if not records:
        return 0, 0.0

    actions: list[str] = []
    for record in records:
        decision = record.get("trading_decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        actions.append(str(action or "").upper())

    no_trade_count = sum(1 for action in actions if action == "NO_TRADE")
    streak = 0
    for action in actions:
        if action != "NO_TRADE":
            break
        streak += 1
    ratio = no_trade_count / len(actions) if actions else 0.0
    return streak, round(max(0.0, min(1.0, ratio)), 3)


def _extract_sector_mentions(text: str) -> set[str]:
    lowered = text.lower()
    found: set[str] = set()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            found.add(sector)
    tokens = set(re.findall(r"\b[A-Z]{2,5}\b", text))
    if tokens:
        upper_map = {token.lower() for token in tokens}
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(keyword in upper_map for keyword in keywords):
                found.add(sector)
    return found


def _compute_exploration_breadth(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    found: set[str] = set()
    for record in records:
        chunks = [
            str(record.get("session_summary") or ""),
            str(record.get("decision_report") or ""),
            str(record.get("self_critique") or ""),
        ]
        decision = record.get("trading_decision")
        if isinstance(decision, dict):
            chunks.append(str(decision.get("rationale") or ""))
        for chunk in chunks:
            found.update(_extract_sector_mentions(chunk))
    return len(found)


def _compute_portfolio_risk(snapshot: dict[str, Any]) -> tuple[float, float, list[str]]:
    diagnostics: list[str] = []
    portfolio = snapshot.get("portfolio") if isinstance(snapshot, dict) else {}
    market = snapshot.get("market") if isinstance(snapshot, dict) else {}
    positions = portfolio.get("positions") if isinstance(portfolio, dict) else []
    snapshots = market.get("snapshots") if isinstance(market, dict) else []
    if not isinstance(positions, list):
        positions = []
    if not isinstance(snapshots, list):
        snapshots = []

    sector_by_symbol: dict[str, str] = {}
    for item in snapshots:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper().strip()
        quote = item.get("quote") if isinstance(item.get("quote"), dict) else {}
        sector = str(quote.get("sector") or "").strip() or "unknown"
        if symbol:
            sector_by_symbol[symbol] = sector

    total_value = 0.0
    total_unrealized = 0.0
    sector_totals: dict[str, float] = {}
    for position in positions:
        if not isinstance(position, dict):
            continue
        symbol = str(position.get("symbol") or "").upper().strip()
        market_value = abs(_to_float(position.get("market_value")))
        unrealized = _to_float(position.get("unrealized_pl"))
        sector = sector_by_symbol.get(symbol, "unknown")

        total_value += market_value
        total_unrealized += unrealized
        sector_totals[sector] = sector_totals.get(sector, 0.0) + market_value

    concentration = 0.0
    if total_value > 0 and sector_totals:
        concentration = max(sector_totals.values()) / total_value
    else:
        diagnostics.append("portfolio_notional_missing_or_empty")

    return round(total_unrealized, 2), round(max(0.0, min(1.0, concentration)), 3), diagnostics


def compute_stall_guardrails(
    snapshot: dict[str, Any],
    *,
    lookback_runs: int = 6,
) -> StallGuardrailResult:
    records = _load_recent_short_memory_records(lookback_runs=lookback_runs)
    no_trade_streak, no_trade_ratio = _compute_no_trade_metrics(records)
    explored_breadth = _compute_exploration_breadth(records)
    total_unrealized_pl, concentration_ratio, diagnostics = _compute_portfolio_risk(snapshot)

    score = 0.0
    flags: list[str] = []
    hard_rules: list[str] = []

    if no_trade_streak >= 3:
        score += 0.4
        flags.append("no_trade_streak>=3")
    if no_trade_ratio >= 0.67:
        score += 0.2
        flags.append("no_trade_ratio_high")
    if total_unrealized_pl < 0:
        score += 0.15
        flags.append("portfolio_unrealized_pl_negative")
    if concentration_ratio >= 0.6:
        score += 0.15
        flags.append("portfolio_sector_concentration_high")
    if explored_breadth < 2 and records:
        score += 0.2
        flags.append("sector_exploration_low")

    score = round(max(0.0, min(1.0, score)), 3)

    if score >= 0.55:
        hard_rules.append(
            "Avant NO_TRADE, exécuter au moins 2 investigations hors semiconducteurs (finance/énergie/industrie/small caps)."
        )
        hard_rules.append(
            "Comparer explicitement au moins 3 secteurs avec métriques de volume, catalyseur, et momentum relatif."
        )
        hard_rules.append(
            "Si les positions existantes se dégradent, définir un plan d'action clair (réduction, couverture, ou rotation)."
        )
    if no_trade_streak >= 4:
        hard_rules.append(
            "NO_TRADE répété: justifier par invalidation factuelle, sinon proposer un pivot de stratégie concret dès ce run."
        )
    if concentration_ratio >= 0.6:
        hard_rules.append(
            "Éviter d'augmenter la concentration sectorielle dominante; privilégier un setup qui améliore la diversification."
        )

    if not hard_rules:
        diagnostics.append("no_enforced_hard_rules")

    return StallGuardrailResult(
        lookback_runs=lookback_runs,
        no_trade_streak=no_trade_streak,
        no_trade_ratio=no_trade_ratio,
        total_unrealized_pl=total_unrealized_pl,
        concentration_ratio=concentration_ratio,
        explored_sector_breadth=explored_breadth,
        stall_score=score,
        stall_flags=flags,
        hard_rules_next_run=hard_rules,
        diagnostics=diagnostics,
    )


def compact_stall_for_trace(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "stall_score": payload.get("stall_score"),
        "no_trade_streak": payload.get("no_trade_streak"),
        "no_trade_ratio": payload.get("no_trade_ratio"),
        "concentration_ratio": payload.get("concentration_ratio"),
        "total_unrealized_pl": payload.get("total_unrealized_pl"),
        "stall_flags": payload.get("stall_flags"),
        "hard_rules_next_run": payload.get("hard_rules_next_run"),
    }

