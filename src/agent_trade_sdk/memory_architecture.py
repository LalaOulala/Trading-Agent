from __future__ import annotations

import difflib
import json
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


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            # Strip first and last code fences when present.
            if lines[0].startswith("```") and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1]).strip()
    return text


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
    json_text = _extract_json_text(raw_text)
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("Final output JSON must be an object.")
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


def _write_short_memory(
    *,
    parsed_output: TradingSessionLLMOutput,
    session_id: str,
    model_name: str,
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


def apply_memory_outputs(
    *,
    raw_final_output: str,
    session_id: str,
    model_name: str,
    logs_root: Path,
) -> MemoryApplyResult:
    try:
        parsed = parse_llm_trading_session_output(raw_final_output)
    except Exception as exc:
        return MemoryApplyResult(
            parsed_output=None,
            short_memory_latest_path=None,
            short_memory_archive_path=None,
            behavior_history=None,
            parse_error=f"{type(exc).__name__}: {exc}",
        )

    short_latest, short_archive = _write_short_memory(
        parsed_output=parsed,
        session_id=session_id,
        model_name=model_name,
    )

    behavior_intent = parsed.long_memory_update_intent
    behavior_history: BehaviorHistoryResult
    if behavior_intent.should_update_behavior and behavior_intent.updated_behavior_markdown:
        behavior_history = _archive_behavior_update(
            new_behavior_text=behavior_intent.updated_behavior_markdown,
            logs_root=logs_root,
            session_id=session_id,
        )
        # Preserve the LLM's summary in the result.
        behavior_history = BehaviorHistoryResult(
            updated=behavior_history.updated,
            before_path=behavior_history.before_path,
            after_path=behavior_history.after_path,
            diff_path=behavior_history.diff_path,
            reason=behavior_history.reason,
            summary=behavior_intent.update_summary,
        )
    else:
        behavior_history = BehaviorHistoryResult(
            updated=False,
            before_path=None,
            after_path=None,
            diff_path=None,
            reason=behavior_intent.why,
            summary=behavior_intent.update_summary,
        )

    return MemoryApplyResult(
        parsed_output=parsed,
        short_memory_latest_path=short_latest,
        short_memory_archive_path=short_archive,
        behavior_history=behavior_history,
        parse_error=None,
    )


def short_memory_prompt_block(short_memory: dict[str, Any] | None) -> str:
    if not short_memory:
        return (
            "### Mémoire courte (session précédente)\n"
            "Aucune mémoire courte disponible (premier run ou mémoire indisponible).\n"
        )
    return (
        "### Mémoire courte (session précédente)\n"
        "Utilise cette mémoire pour garder un fil conducteur et éviter de répéter les mêmes erreurs.\n"
        f"```json\n{_safe_json(short_memory, max_chars=50000)}\n```\n"
    )
