from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]
SHORT_MEMORY_DIR = ROOT_DIR / "memory" / "short"
SHORT_MEMORY_LATEST_PATH = SHORT_MEMORY_DIR / "latest.json"
SHORT_MEMORY_ARCHIVE_DIR = SHORT_MEMORY_DIR / "archive"
BEHAVIOR_PATH = ROOT_DIR / "behavior.md"

TradingAction = Literal["BUY", "SELL", "SHORT", "CLOSE", "NO_TRADE"]
TRADING_TOOL_NAMES = {"place_market_order", "open_short_position", "close_open_position"}
TRADING_ACTIONS = {"BUY", "SELL", "SHORT", "CLOSE"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _safe_json(data: Any, *, max_chars: int = 20000) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(data)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _extract_fenced_blocks(raw_text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    pattern = re.compile(r"```([a-zA-Z0-9_-]+)?\s*\n(.*?)\n```", flags=re.DOTALL)
    for match in pattern.finditer(raw_text):
        language = match.group(1).strip().lower() if match.group(1) else None
        body = (match.group(2) or "").strip()
        if body:
            blocks.append((language, body))
    return blocks


def _extract_balanced_json_objects(raw_text: str, *, max_chars: int = 250_000) -> list[str]:
    candidates: list[str] = []
    depth = 0
    start_idx: int | None = None
    in_string = False
    escaping = False

    for idx, char in enumerate(raw_text):
        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidate = raw_text[start_idx : idx + 1].strip()
                if candidate and len(candidate) <= max_chars:
                    candidates.append(candidate)
                start_idx = None

    return candidates


def _try_parse_json_dict(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    parsed_candidates: list[dict[str, Any]] = []

    direct = _try_parse_json_dict(text)
    if direct is not None:
        parsed_candidates.append(direct)

    fenced_blocks = _extract_fenced_blocks(text)
    for language, block in fenced_blocks:
        if language and language not in {"json", "javascript", "js"}:
            continue
        parsed = _try_parse_json_dict(block)
        if parsed is not None:
            parsed_candidates.append(parsed)

    for candidate in _extract_balanced_json_objects(text):
        parsed = _try_parse_json_dict(candidate)
        if parsed is not None:
            parsed_candidates.append(parsed)

    if not parsed_candidates:
        raise ValueError("Unable to extract any valid JSON object from final output.")

    for payload in reversed(parsed_candidates):
        if "trading_decision" in payload:
            return payload

    for payload in reversed(parsed_candidates):
        if isinstance(payload.get("action"), str) or isinstance(payload.get("decision"), str):
            return payload

    return parsed_candidates[-1]


def _stringify_value(value: Any, *, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate(value, max_chars=max_chars)
    try:
        return _truncate(json.dumps(value, ensure_ascii=False), max_chars=max_chars)
    except Exception:
        return _truncate(str(value), max_chars=max_chars)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _truncate(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + " ... (truncated)"


class TradingDecisionBlock(BaseModel):
    action: TradingAction
    symbol: str | None = None
    confidence: str | None = None
    rationale: str = Field(
        ...,
        description="Rationale synthétique de la décision (texte, pas un objet).",
    )
    signal_principal: str | None = None
    risk_identified: str | None = None
    invalidation_condition: str | None = None
    executed_order: dict[str, Any] | None = None


class LongMemoryUpdateIntent(BaseModel):
    should_update_behavior: bool
    why: str = Field(..., description="Pourquoi mettre à jour (ou non) behavior.md.")
    update_summary: list[str] = Field(
        default_factory=list,
        description="3 à 8 points résumant les changements de behavior.md si applicable.",
    )
    updated_behavior_markdown: str | None = Field(
        default=None,
        description="Contenu COMPLET de behavior.md si should_update_behavior=true, sinon null.",
    )


class TradingSessionLLMOutput(BaseModel):
    trading_decision: TradingDecisionBlock
    session_summary: str
    decision_report: str
    self_critique: str
    pitfalls_to_avoid_next_run: list[str] = Field(default_factory=list)
    next_session_directives: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    long_memory_update_intent: LongMemoryUpdateIntent


class ShortMemoryRecord(BaseModel):
    generated_at_utc: str
    source_session_id: str
    model_name: str
    session_summary: str
    decision_report: str
    self_critique: str
    pitfalls_to_avoid_next_run: list[str] = Field(default_factory=list)
    next_session_directives: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    system_warnings: list[str] = Field(default_factory=list)
    execution_validation: dict[str, Any] | None = None
    long_memory_update_intent: dict[str, Any]
    trading_decision: dict[str, Any]


@dataclass(frozen=True)
class BehaviorHistoryResult:
    updated: bool
    before_path: Path | None
    after_path: Path | None
    diff_path: Path | None
    reason: str
    summary: list[str]


@dataclass(frozen=True)
class MemoryApplyResult:
    parsed_output: TradingSessionLLMOutput | None
    short_memory_latest_path: Path | None
    short_memory_archive_path: Path | None
    behavior_history: BehaviorHistoryResult | None
    system_warnings: list[str]
    execution_validation: dict[str, Any] | None
    parse_error: str | None

    def summary_for_log(self) -> dict[str, Any]:
        behavior = self.behavior_history
        return {
            "parse_error": self.parse_error,
            "short_memory_latest_path": str(self.short_memory_latest_path)
            if self.short_memory_latest_path
            else None,
            "short_memory_archive_path": str(self.short_memory_archive_path)
            if self.short_memory_archive_path
            else None,
            "behavior_updated": bool(behavior.updated) if behavior else False,
            "behavior_before_path": str(behavior.before_path) if behavior and behavior.before_path else None,
            "behavior_after_path": str(behavior.after_path) if behavior and behavior.after_path else None,
            "behavior_diff_path": str(behavior.diff_path) if behavior and behavior.diff_path else None,
            "behavior_reason": behavior.reason if behavior else None,
            "behavior_update_summary": behavior.summary if behavior else [],
            "system_warnings": self.system_warnings,
            "execution_validation": self.execution_validation,
        }


def load_short_memory_latest() -> dict[str, Any] | None:
    if not SHORT_MEMORY_LATEST_PATH.exists():
        return None
    try:
        payload = json.loads(SHORT_MEMORY_LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "warning": "latest short memory file could not be parsed",
            "path": str(SHORT_MEMORY_LATEST_PATH),
            "raw_excerpt": _truncate(SHORT_MEMORY_LATEST_PATH.read_text(encoding="utf-8"), max_chars=2000),
        }
    try:
        validated = ShortMemoryRecord.model_validate(payload)
        return validated.model_dump()
    except ValidationError:
        # Keep raw payload if schema evolved manually; the agent can still inspect it.
        return payload if isinstance(payload, dict) else {"raw": payload}


def load_behavior_text() -> str:
    if not BEHAVIOR_PATH.exists():
        return (
            "# Behavior - Adaptive Trading Profile\n\n"
            "## Current Style\n\n"
            "- Style not initialized.\n"
        )
    return BEHAVIOR_PATH.read_text(encoding="utf-8").strip()


def parse_llm_trading_session_output(raw_text: str) -> TradingSessionLLMOutput:
    payload = _extract_json_payload(raw_text)
    if "trading_decision" not in payload:
        legacy_action = payload.get("action") or payload.get("decision")
        if isinstance(legacy_action, str):
            legacy_rationale = payload.get("rationale")
            legacy_signal = payload.get("signal_principal") or payload.get("principal_signal")
            legacy_risk = payload.get("risk_identified") or payload.get("risque_identifie") or payload.get(
                "identified_risk"
            )
            legacy_invalidation = payload.get("invalidation_condition") or payload.get(
                "condition_invalidation"
            )
            payload = {
                "trading_decision": {
                    "action": legacy_action.upper(),
                    "symbol": payload.get("symbol"),
                    "confidence": payload.get("confidence"),
                    "rationale": _stringify_value(legacy_rationale) or "Legacy final output migrated.",
                    "signal_principal": _stringify_value(legacy_signal) or None,
                    "risk_identified": _stringify_value(legacy_risk) or None,
                    "invalidation_condition": _stringify_value(legacy_invalidation) or None,
                    "executed_order": payload.get("executed_order"),
                },
                "session_summary": (
                    "Legacy output format detected. Decision migrated to new memory schema for continuity."
                ),
                "decision_report": _stringify_value(legacy_rationale) or "No decision report provided.",
                "self_critique": (
                    "Le run a utilisé un ancien format de sortie. La mémoire a été reconstruite de façon "
                    "minimale; améliorer le respect du contrat JSON au prochain run."
                ),
                "pitfalls_to_avoid_next_run": [
                    "Ne pas retourner l'ancien format JSON (action/decision top-level)."
                ],
                "next_session_directives": [
                    "Respecter exactement le contrat JSON avec trading_decision + mémoire courte."
                ],
                "open_questions": [],
                "long_memory_update_intent": {
                    "should_update_behavior": False,
                    "why": "Legacy output format: skip behavior update this run.",
                    "update_summary": [],
                    "updated_behavior_markdown": None,
                },
            }

    parsed = TradingSessionLLMOutput.model_validate(payload)
    if parsed.long_memory_update_intent.should_update_behavior and not (
        parsed.long_memory_update_intent.updated_behavior_markdown
        and parsed.long_memory_update_intent.updated_behavior_markdown.strip()
    ):
        raise ValueError(
            "should_update_behavior=true but updated_behavior_markdown is empty or missing."
        )
    return parsed


def _build_parse_failure_fallback_output(
    *,
    parse_error: str,
    runtime_summary: dict[str, Any] | None,
) -> TradingSessionLLMOutput:
    audit = _runtime_trading_execution_audit(runtime_summary)
    report_lines = [
        "Sortie agent non parseable: fallback mémoire appliqué.",
        f"parse_error: {parse_error}",
        f"runtime_event_count: {(runtime_summary or {}).get('event_count') if isinstance(runtime_summary, dict) else None}",
        f"trading_tool_calls_count: {audit.get('trading_tool_calls_count')}",
        f"trading_tool_success_count: {audit.get('trading_tool_success_count')}",
    ]
    return TradingSessionLLMOutput(
        trading_decision=TradingDecisionBlock(
            action="NO_TRADE",
            symbol=None,
            confidence="low",
            rationale=(
                "Sortie finale invalide techniquement; décision neutralisée automatiquement pour préserver "
                "la cohérence inter-run."
            ),
            signal_principal="Parse failure on main output",
            risk_identified="Memory contamination risk from malformed output",
            invalidation_condition="Rétablir un JSON de sortie valide contenant trading_decision",
            executed_order=None,
        ),
        session_summary=(
            "Fallback mémoire: run conservé malgré une sortie finale non parseable. "
            "Le prochain cycle doit produire un JSON strict valide."
        ),
        decision_report=" | ".join(report_lines),
        self_critique=(
            "La sortie du run principal était invalide et n'a pas été propagée en mémoire brute. "
            "Fallback minimal appliqué."
        ),
        pitfalls_to_avoid_next_run=[
            "Ne pas mixer prose et JSON non borné sans bloc JSON valide contenant trading_decision.",
            "Toujours vérifier la validité JSON finale avant clôture du run.",
        ],
        next_session_directives=[
            "Retourner un JSON strict avec la clé trading_decision.",
            "Conserver executed_order=null si aucun tool de trading n'a réussi.",
        ],
        open_questions=["Pourquoi la sortie finale a-t-elle perdu sa structure JSON ?"],
        long_memory_update_intent=LongMemoryUpdateIntent(
            should_update_behavior=False,
            why="Parse failure fallback: behavior update disabled.",
            update_summary=[],
            updated_behavior_markdown=None,
        ),
    )


def _write_short_memory(
    *,
    parsed_output: TradingSessionLLMOutput,
    session_id: str,
    model_name: str,
    system_warnings: list[str] | None = None,
    execution_validation: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    record = ShortMemoryRecord(
        generated_at_utc=_utc_now().isoformat(),
        source_session_id=session_id,
        model_name=model_name,
        session_summary=parsed_output.session_summary,
        decision_report=parsed_output.decision_report,
        self_critique=parsed_output.self_critique,
        pitfalls_to_avoid_next_run=parsed_output.pitfalls_to_avoid_next_run,
        next_session_directives=parsed_output.next_session_directives,
        open_questions=parsed_output.open_questions,
        system_warnings=system_warnings or [],
        execution_validation=execution_validation,
        long_memory_update_intent=parsed_output.long_memory_update_intent.model_dump(
            exclude={"updated_behavior_markdown"}
        ),
        trading_decision=parsed_output.trading_decision.model_dump(),
    )
    SHORT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    SHORT_MEMORY_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_stamp()
    archive_path = SHORT_MEMORY_ARCHIVE_DIR / f"short_memory_{timestamp}_{session_id}.json"
    content = json.dumps(record.model_dump(), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(archive_path, content)
    _atomic_write_text(SHORT_MEMORY_LATEST_PATH, content)
    return SHORT_MEMORY_LATEST_PATH, archive_path


def _archive_behavior_update(
    *,
    new_behavior_text: str,
    logs_root: Path,
    session_id: str,
) -> BehaviorHistoryResult:
    current_text = BEHAVIOR_PATH.read_text(encoding="utf-8") if BEHAVIOR_PATH.exists() else ""
    normalized_new = new_behavior_text.rstrip() + "\n"

    if current_text == normalized_new:
        return BehaviorHistoryResult(
            updated=False,
            before_path=None,
            after_path=None,
            diff_path=None,
            reason="No change in behavior.md content.",
            summary=[],
        )

    timestamp = _utc_stamp()
    behaviors_dir = logs_root / "behaviors"
    behaviors_dir.mkdir(parents=True, exist_ok=True)
    before_path = behaviors_dir / f"behavior_{timestamp}_{session_id}_before.md"
    after_path = behaviors_dir / f"behavior_{timestamp}_{session_id}_after.md"
    diff_path = behaviors_dir / f"behavior_{timestamp}_{session_id}.diff"

    before_path.write_text(current_text, encoding="utf-8")
    _atomic_write_text(BEHAVIOR_PATH, normalized_new)
    after_text = BEHAVIOR_PATH.read_text(encoding="utf-8")
    after_path.write_text(after_text, encoding="utf-8")
    diff_text = "".join(
        difflib.unified_diff(
            current_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="behavior_before.md",
            tofile="behavior_after.md",
        )
    )
    diff_path.write_text(diff_text, encoding="utf-8")
    return BehaviorHistoryResult(
        updated=True,
        before_path=before_path,
        after_path=after_path,
        diff_path=diff_path,
        reason="Applied LLM-proposed behavior.md update.",
        summary=[],
    )


def _runtime_trading_execution_audit(runtime_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runtime_summary, dict):
        return {
            "trading_tool_calls_count": 0,
            "trading_tool_outputs_count": 0,
            "trading_tool_success_count": 0,
            "trading_tool_error_count": 0,
            "trading_tools_called": [],
        }

    tool_calls = runtime_summary.get("tool_calls") or []
    tool_outputs = runtime_summary.get("tool_outputs") or []
    trading_calls = [
        item
        for item in tool_calls
        if isinstance(item, dict) and str(item.get("tool_name")) in TRADING_TOOL_NAMES
    ]
    trading_outputs = [
        item
        for item in tool_outputs
        if isinstance(item, dict) and str(item.get("tool_name")) in TRADING_TOOL_NAMES
    ]
    trading_success = [item for item in trading_outputs if not bool(item.get("is_error"))]
    trading_errors = [item for item in trading_outputs if bool(item.get("is_error"))]
    return {
        "trading_tool_calls_count": len(trading_calls),
        "trading_tool_outputs_count": len(trading_outputs),
        "trading_tool_success_count": len(trading_success),
        "trading_tool_error_count": len(trading_errors),
        "trading_tools_called": sorted(
            {str(item.get("tool_name")) for item in trading_calls if item.get("tool_name")}
        ),
        "last_trading_tool_output_excerpt": trading_outputs[-1].get("output_excerpt") if trading_outputs else None,
    }


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _enforce_trading_execution_consistency(
    parsed: TradingSessionLLMOutput,
    *,
    runtime_summary: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any] | None]:
    audit = _runtime_trading_execution_audit(runtime_summary)
    warnings: list[str] = []
    decision = parsed.trading_decision
    action = decision.action
    success_count = int(audit.get("trading_tool_success_count") or 0)
    call_count = int(audit.get("trading_tool_calls_count") or 0)

    if action in TRADING_ACTIONS and success_count == 0:
        if call_count > 0:
            warning = (
                "Validation d'exécution: action de trade déclarée "
                f"({action}) mais aucun tool de trading n'a réussi pendant ce run "
                f"(calls={call_count}, successes=0). executed_order a été invalidé (null)."
            )
        else:
            warning = (
                "Validation d'exécution: action de trade déclarée "
                f"({action}) sans aucun appel à un tool de trading pendant ce run. "
                "executed_order a été invalidé (null) et aucun ordre Alpaca n'a été soumis."
            )
        warnings.append(warning)
        decision.executed_order = None
        _append_unique(parsed.pitfalls_to_avoid_next_run, "Ne jamais déclarer un ordre exécuté sans tool Alpaca réussi.")
        _append_unique(
            parsed.next_session_directives,
            "Si tu choisis BUY/SELL/SHORT/CLOSE, appelle un tool de trading puis recopie fidèlement son JSON dans executed_order.",
        )
        parsed.self_critique = (
            parsed.self_critique.rstrip()
            + " [WARNING SYSTEME: décision de trade sans exécution Alpaca confirmée; executed_order invalidé.]"
        )
    elif action in TRADING_ACTIONS and success_count > 0 and decision.executed_order is None:
        warnings.append(
            "Validation d'exécution: un tool de trading a réussi pendant ce run mais executed_order est null dans la sortie finale. "
            "La mémoire conserve l'action, mais l'output LLM n'a pas recopié fidèlement l'exécution."
        )
        _append_unique(
            parsed.next_session_directives,
            "Quand un tool de trading réussit, recopie son JSON réel dans executed_order (sans reformater ni inventer).",
        )
    elif action == "NO_TRADE" and success_count > 0:
        warnings.append(
            "Validation d'exécution: NO_TRADE déclaré alors qu'un tool de trading a réussi pendant ce run. "
            "La sortie finale est incohérente avec les appels de tools."
        )

    if action == "NO_TRADE" and decision.executed_order is not None:
        warnings.append(
            "Validation d'exécution: executed_order non null alors que l'action est NO_TRADE. executed_order a été invalidé (null)."
        )
        decision.executed_order = None

    if not warnings:
        return [], {
            **audit,
            "status": "ok",
            "decision_action": action,
            "llm_executed_order_present": decision.executed_order is not None,
        }

    return warnings, {
        **audit,
        "status": "warning",
        "decision_action": action,
        "llm_executed_order_present": decision.executed_order is not None,
        "warnings": warnings,
    }


def apply_memory_outputs(
    *,
    raw_final_output: str,
    session_id: str,
    model_name: str,
    logs_root: Path,
    runtime_summary: dict[str, Any] | None = None,
) -> MemoryApplyResult:
    try:
        parsed = parse_llm_trading_session_output(raw_final_output)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
        fallback_output = _build_parse_failure_fallback_output(
            parse_error=parse_error,
            runtime_summary=runtime_summary,
        )
        warnings = [
            "Final output parse failed; fallback short memory record generated.",
            parse_error,
        ]
        audit = _runtime_trading_execution_audit(runtime_summary)
        short_latest, short_archive = _write_short_memory(
            parsed_output=fallback_output,
            session_id=session_id,
            model_name=model_name,
            system_warnings=warnings,
            execution_validation={**audit, "status": "parse_failed"},
        )
        return MemoryApplyResult(
            parsed_output=None,
            short_memory_latest_path=short_latest,
            short_memory_archive_path=short_archive,
            behavior_history=None,
            system_warnings=warnings,
            execution_validation={**audit, "status": "parse_failed"},
            parse_error=parse_error,
        )

    system_warnings, execution_validation = _enforce_trading_execution_consistency(
        parsed,
        runtime_summary=runtime_summary,
    )

    short_latest, short_archive = _write_short_memory(
        parsed_output=parsed,
        session_id=session_id,
        model_name=model_name,
        system_warnings=system_warnings,
        execution_validation=execution_validation,
    )

    behavior_intent = parsed.long_memory_update_intent
    behavior_history = BehaviorHistoryResult(
        updated=False,
        before_path=None,
        after_path=None,
        diff_path=None,
        reason=(
            "Behavior update handled by post-run reflection cycle."
            if behavior_intent.should_update_behavior
            else behavior_intent.why
        ),
        summary=behavior_intent.update_summary,
    )
    if behavior_intent.should_update_behavior:
        _append_unique(
            system_warnings,
            "behavior.md update requested in main output but deferred to post-run reflection writer.",
        )

    return MemoryApplyResult(
        parsed_output=parsed,
        short_memory_latest_path=short_latest,
        short_memory_archive_path=short_archive,
        behavior_history=behavior_history,
        system_warnings=system_warnings,
        execution_validation=execution_validation,
        parse_error=None,
    )


def short_memory_prompt_block(short_memory: dict[str, Any] | None) -> str:
    if not short_memory:
        return (
            "### Mémoire courte (session précédente)\n"
            "Aucune mémoire courte disponible (premier run ou mémoire indisponible).\n"
        )
    warnings = short_memory.get("system_warnings") if isinstance(short_memory, dict) else None
    warning_block = ""
    if isinstance(warnings, list) and warnings:
        warning_lines = "\n".join(f"- {str(item)}" for item in warnings[:5])
        warning_block = (
            "### Avertissements système (session précédente)\n"
            "Ces avertissements proviennent du code (validation d'exécution / cohérence) et priment sur tes suppositions.\n"
            f"{warning_lines}\n\n"
        )
    return (
        "### Mémoire courte (session précédente)\n"
        "Utilise cette mémoire pour garder un fil conducteur et éviter de répéter les mêmes erreurs.\n"
        f"{warning_block}"
        f"```json\n{_safe_json(short_memory, max_chars=50000)}\n```\n"
    )
