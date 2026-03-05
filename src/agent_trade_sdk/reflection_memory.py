from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


ROOT_DIR = Path(__file__).resolve().parents[2]
REFLECTION_MEMORY_DIR = ROOT_DIR / "memory" / "reflection"
REFLECTION_MEMORY_LATEST_PATH = REFLECTION_MEMORY_DIR / "latest.json"
REFLECTION_MEMORY_ARCHIVE_DIR = REFLECTION_MEMORY_DIR / "archive"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _truncate(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ... (truncated)"


def _safe_json(data: Any, *, max_chars: int = 4000) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(data)
    return _truncate(text, max_chars=max_chars)


def _normalize_list_of_str(
    value: Any,
    *,
    max_items: int = 8,
    max_item_chars: int = 220,
) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:max_items]:
        text = str(item).strip()
        if not text:
            continue
        out.append(_truncate(text, max_chars=max_item_chars))
    return out


class ReflectionConclusion(BaseModel):
    generated_at_utc: str
    source_session_id: str
    strategy_health_score: float = Field(..., ge=0.0, le=1.0)
    stall_flags: list[str] = Field(default_factory=list)
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    next_run_focus: list[str] = Field(default_factory=list)
    hard_rules_next_run: list[str] = Field(default_factory=list)
    conclusion_for_prompt: str = Field(
        ...,
        min_length=1,
        max_length=1500,
    )


def load_reflection_latest() -> dict[str, Any] | None:
    if not REFLECTION_MEMORY_LATEST_PATH.exists():
        return None
    try:
        payload = json.loads(REFLECTION_MEMORY_LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "warning": "latest reflection memory file could not be parsed",
            "path": str(REFLECTION_MEMORY_LATEST_PATH),
            "raw_excerpt": _truncate(
                REFLECTION_MEMORY_LATEST_PATH.read_text(encoding="utf-8"),
                max_chars=2000,
            ),
        }
    try:
        validated = ReflectionConclusion.model_validate(payload)
        return validated.model_dump()
    except ValidationError:
        return payload if isinstance(payload, dict) else {"raw": payload}


def write_reflection_conclusion(
    *,
    conclusion: ReflectionConclusion,
    session_id: str,
) -> tuple[Path, Path]:
    REFLECTION_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    REFLECTION_MEMORY_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_stamp()
    archive_path = REFLECTION_MEMORY_ARCHIVE_DIR / f"reflection_{timestamp}_{session_id}.json"
    content = json.dumps(conclusion.model_dump(), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(archive_path, content)
    _atomic_write_text(REFLECTION_MEMORY_LATEST_PATH, content)
    return REFLECTION_MEMORY_LATEST_PATH, archive_path


def build_fallback_reflection_from_short_memory(
    short_memory: dict[str, Any] | None,
    *,
    session_id: str,
) -> dict[str, Any] | None:
    if not isinstance(short_memory, dict):
        return None

    session_summary = str(short_memory.get("session_summary") or "").strip()
    decision_report = str(short_memory.get("decision_report") or "").strip()
    self_critique = str(short_memory.get("self_critique") or "").strip()
    if not any((session_summary, decision_report, self_critique)):
        return None

    pitfalls = _normalize_list_of_str(short_memory.get("pitfalls_to_avoid_next_run"), max_items=4)
    directives = _normalize_list_of_str(short_memory.get("next_session_directives"), max_items=4)
    open_questions = _normalize_list_of_str(short_memory.get("open_questions"), max_items=3)
    action = (
        str((short_memory.get("trading_decision") or {}).get("action") or "")
        if isinstance(short_memory.get("trading_decision"), dict)
        else ""
    )

    parts: list[str] = []
    if session_summary:
        parts.append(f"Résumé: {_truncate(session_summary, max_chars=500)}")
    if decision_report:
        parts.append(f"Décision: {_truncate(decision_report, max_chars=500)}")
    if self_critique:
        parts.append(f"Autocritique: {_truncate(self_critique, max_chars=500)}")
    conclusion_text = _truncate("\n".join(parts), max_chars=1500)

    fallback = ReflectionConclusion(
        generated_at_utc=_utc_now().isoformat(),
        source_session_id=session_id,
        strategy_health_score=0.5,
        stall_flags=["fallback_from_short_memory"] if action == "NO_TRADE" else [],
        what_worked=pitfalls[:2],
        what_failed=pitfalls[2:4] if len(pitfalls) > 2 else [],
        next_run_focus=directives[:4],
        hard_rules_next_run=directives[:2],
        conclusion_for_prompt=conclusion_text or "Fallback reflection generated from short memory.",
    )
    payload = fallback.model_dump()
    if open_questions:
        payload["open_questions"] = open_questions
    return payload


def reflection_prompt_block(
    reflection: dict[str, Any] | None,
    *,
    max_total_chars: int = 3000,
) -> str:
    if not isinstance(reflection, dict):
        return (
            "### Réflexion inter-session\n"
            "Aucune réflexion disponible (fallback minimal). Reste discipliné, évite les répétitions, "
            "et améliore la méthode d'enquête si NO_TRADE se répète.\n"
        )

    conclusion = str(reflection.get("conclusion_for_prompt") or "").strip()
    hard_rules = _normalize_list_of_str(reflection.get("hard_rules_next_run"), max_items=6, max_item_chars=220)
    stall_flags = _normalize_list_of_str(reflection.get("stall_flags"), max_items=6, max_item_chars=160)
    next_focus = _normalize_list_of_str(reflection.get("next_run_focus"), max_items=6, max_item_chars=200)
    strategy_health = reflection.get("strategy_health_score")
    generated_at = reflection.get("generated_at_utc")
    source_session = reflection.get("source_session_id")

    lines: list[str] = [
        "### Réflexion inter-session (prioritaire)",
        f"- generated_at_utc: {generated_at}",
        f"- source_session_id: {source_session}",
        f"- strategy_health_score: {strategy_health}",
    ]
    if stall_flags:
        lines.append("- stall_flags:")
        lines.extend([f"  - {item}" for item in stall_flags])
    if hard_rules:
        lines.append("- hard_rules_next_run:")
        lines.extend([f"  - {item}" for item in hard_rules])
    if next_focus:
        lines.append("- next_run_focus:")
        lines.extend([f"  - {item}" for item in next_focus])
    if conclusion:
        lines.append("")
        lines.append("Conclusion synthétique:")
        lines.append(_truncate(conclusion, max_chars=1500))
    else:
        lines.append("Conclusion synthétique: indisponible.")

    text = "\n".join(lines) + "\n"
    return _truncate(text, max_chars=max_total_chars)


def compact_reflection_for_trace(reflection: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(reflection, dict):
        return {}
    return {
        "source_session_id": reflection.get("source_session_id"),
        "generated_at_utc": reflection.get("generated_at_utc"),
        "strategy_health_score": reflection.get("strategy_health_score"),
        "stall_flags": _normalize_list_of_str(reflection.get("stall_flags"), max_items=4, max_item_chars=100),
        "hard_rules_next_run": _normalize_list_of_str(
            reflection.get("hard_rules_next_run"), max_items=4, max_item_chars=140
        ),
        "conclusion_excerpt": _truncate(str(reflection.get("conclusion_for_prompt") or ""), max_chars=450),
    }

