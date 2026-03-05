from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field

from agent_trade_sdk.config import Settings
from agent_trade_sdk.memory_architecture import BEHAVIOR_PATH
from agent_trade_sdk.reflection_memory import ReflectionConclusion, write_reflection_conclusion
from agent_trade_sdk.tracing_support import build_agents_run_config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(data: Any, max_chars: int = 24000) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(data)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


def _truncate(value: str, max_chars: int = 12000) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n... (truncated)"


class BehaviorUpdateIntent(BaseModel):
    should_update_behavior: bool
    why: str = Field(..., description="Pourquoi mettre à jour (ou non) behavior.md.")
    update_summary: list[str] = Field(default_factory=list)
    updated_behavior_markdown: str | None = Field(
        default=None,
        description="Contenu complet de behavior.md si should_update_behavior=true, sinon null.",
    )


class PostRunMemoryOutput(BaseModel):
    journal_entry_markdown: str = Field(
        ...,
        description="Entrée markdown du journal personnel pour ce cycle (1 entrée complète).",
    )
    reflection_conclusion: ReflectionConclusion
    behavior_update_intent: BehaviorUpdateIntent


@dataclass(frozen=True)
class PostRunMemoryResult:
    journal_file_path: Path
    reflection_latest_path: Path
    reflection_archive_path: Path
    behavior_file_path: Path
    behavior_history_before_path: Path | None
    behavior_history_after_path: Path | None
    behavior_diff_path: Path | None
    behavior_updated: bool
    behavior_update_summary: list[str]


def _build_reflection_agent(model_name: str | None = None) -> Agent:
    settings = Settings.from_env(require_openrouter=True)
    model = LitellmModel(
        model=model_name or settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )

    instructions = (
        "Tu es le module de réflexion inter-session d'un agent de trading paper autonome.\n"
        "Mission: analyser le cycle terminé, produire une introspection utile et exploitable pour le cycle suivant.\n"
        "Contraintes non négociables:\n"
        "- SOUL.md est immuable et ne doit jamais être modifié.\n"
        "- Tu peux proposer une mise à jour de behavior.md uniquement si elle est utile et durable.\n"
        "- Reste concret: faits, erreurs, blocages, actions correctrices.\n"
        "- Si NO_TRADE se répète, explicite le risque d'inaction chronique et des règles correctrices.\n"
        "- Retourne strictement le format structuré demandé."
    )

    return Agent(
        name="TradingAgentReflectionWriter",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(include_usage=False),
        output_type=PostRunMemoryOutput,
    )


def _build_post_run_prompt(
    *,
    session_id: str,
    user_prompt: str,
    final_output: str,
    bootstrap_context: dict[str, Any],
    runtime_summary: dict[str, Any],
    session_log_path: Path,
    current_behavior: str,
) -> str:
    return (
        "Tu dois produire la réflexion inter-session d'un run de trading.\n\n"
        "Sortie obligatoire:\n"
        "1) journal_entry_markdown: journal intime de trader (debuggable humainement).\n"
        "2) reflection_conclusion: synthèse structurée pour injection dans le prochain prompt.\n"
        "3) behavior_update_intent: proposition optionnelle de mise à jour de behavior.md.\n\n"
        "Règles de qualité:\n"
        "- Identifier ce qui a marché / échoué / bloqué.\n"
        "- Évaluer le risque d'inaction chronique (NO_TRADE répété) sans forcer un trade aveugle.\n"
        "- Définir hard_rules_next_run quand des blocages sont visibles.\n"
        "- conclusion_for_prompt doit rester concise (<=1500 chars).\n"
        "- should_update_behavior=true seulement si amélioration durable explicite.\n"
        "- Si should_update_behavior=false, updated_behavior_markdown doit être null.\n\n"
        f"session_id: {session_id}\n"
        f"session_log_path: {session_log_path}\n\n"
        "=== USER PROMPT DU CYCLE ===\n"
        f"{_truncate(user_prompt, 8000)}\n\n"
        "=== FINAL OUTPUT DU CYCLE ===\n"
        f"{_truncate(final_output, 12000)}\n\n"
        "=== SNAPSHOT INITIAL COMPACT (JSON) ===\n"
        f"{_safe_json(bootstrap_context, max_chars=24000)}\n\n"
        "=== RUNTIME SUMMARY (tools/reasoning/messages) ===\n"
        f"{_safe_json(runtime_summary, max_chars=28000)}\n\n"
        "=== behavior.md ACTUEL ===\n"
        f"{_truncate(current_behavior, 22000)}\n"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _append_session_log_post_run(
    *,
    session_log_path: Path,
    journal_path: Path,
    reflection_latest_path: Path,
    reflection_archive_path: Path,
    behavior_updated: bool,
    behavior_diff_path: Path | None,
    behavior_update_summary: list[str],
) -> None:
    if not session_log_path.exists():
        return
    lines = [
        "## Post-Run Reflection\n",
        "\n",
        f"- time_utc: `{_utc_now_iso()}`\n",
        f"- journal_entry_file: `{journal_path}`\n",
        f"- reflection_latest_file: `{reflection_latest_path}`\n",
        f"- reflection_archive_file: `{reflection_archive_path}`\n",
        f"- behavior_updated: `{str(behavior_updated).lower()}`\n",
    ]
    if behavior_diff_path:
        lines.append(f"- behavior_diff_file: `{behavior_diff_path}`\n")
    lines.append("\n")
    if behavior_update_summary:
        lines.append("### behavior.md Update Summary\n\n")
        for item in behavior_update_summary:
            lines.append(f"- {item}\n")
        lines.append("\n")
    with session_log_path.open("a", encoding="utf-8") as fh:
        fh.write("".join(lines))


def _append_journal_entry(
    *,
    journal_dir: Path,
    session_id: str,
    journal_entry_markdown: str,
    session_log_path: Path,
) -> Path:
    now = datetime.now(timezone.utc)
    day_file = journal_dir / f"journal_{now.strftime('%Y-%m-%d')}.md"
    journal_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "\n\n---\n\n"
        f"## Run {now.strftime('%H:%M:%S')} UTC - session `{session_id}`\n\n"
        f"- session_log: `{session_log_path}`\n\n"
    )
    with day_file.open("a", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(journal_entry_markdown.strip() + "\n")
    return day_file


def _write_behavior_with_history(
    *,
    new_behavior_text: str,
    behavior_history_dir: Path,
    session_id: str,
) -> tuple[bool, Path | None, Path | None, Path | None]:
    before_text = BEHAVIOR_PATH.read_text(encoding="utf-8") if BEHAVIOR_PATH.exists() else ""
    normalized_new = new_behavior_text.rstrip() + "\n"
    if before_text == normalized_new:
        return False, None, None, None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    behavior_history_dir.mkdir(parents=True, exist_ok=True)

    before_path = behavior_history_dir / f"behavior_before_{timestamp}_{session_id}.md"
    after_path = behavior_history_dir / f"behavior_after_{timestamp}_{session_id}.md"
    diff_path = behavior_history_dir / f"behavior_diff_{timestamp}_{session_id}.diff"

    before_path.write_text(before_text, encoding="utf-8")
    _atomic_write_text(BEHAVIOR_PATH, normalized_new)
    after_text = BEHAVIOR_PATH.read_text(encoding="utf-8")
    after_path.write_text(after_text, encoding="utf-8")

    diff_text = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="behavior_before.md",
            tofile="behavior_after.md",
        )
    )
    diff_path.write_text(diff_text, encoding="utf-8")
    return True, before_path, after_path, diff_path


async def run_post_run_memory_cycle(
    *,
    session_id: str,
    user_prompt: str,
    model_name: str | None,
    final_output: str,
    bootstrap_context: dict[str, Any],
    runtime_summary: dict[str, Any],
    session_log_path: Path,
    logs_root: Path,
    enable_tracing: bool,
) -> PostRunMemoryResult:
    current_behavior = BEHAVIOR_PATH.read_text(encoding="utf-8") if BEHAVIOR_PATH.exists() else ""
    reflection_agent = _build_reflection_agent(model_name=model_name)
    prompt = _build_post_run_prompt(
        session_id=session_id,
        user_prompt=user_prompt,
        final_output=final_output,
        bootstrap_context=bootstrap_context,
        runtime_summary=runtime_summary,
        session_log_path=session_log_path,
        current_behavior=current_behavior,
    )

    result = await Runner.run(
        reflection_agent,
        prompt,
        run_config=build_agents_run_config(
            enable_tracing=enable_tracing,
            workflow_name="agent_trade_sdk.post_run_memory",
            group_id=session_id,
            trace_metadata={
                "component": "post_run_memory",
                "session_id": session_id,
                "provider": "openrouter",
                "model": model_name,
                "max_turns": 2,
            },
        ),
        max_turns=2,
    )
    reflection_output = result.final_output
    if not isinstance(reflection_output, PostRunMemoryOutput):
        if isinstance(reflection_output, str):
            reflection_output = PostRunMemoryOutput.model_validate(json.loads(reflection_output))
        elif hasattr(reflection_output, "model_dump"):
            reflection_output = PostRunMemoryOutput.model_validate(reflection_output.model_dump())
        else:
            reflection_output = PostRunMemoryOutput.model_validate(reflection_output)

    journal_path = _append_journal_entry(
        journal_dir=logs_root / "journals",
        session_id=session_id,
        journal_entry_markdown=reflection_output.journal_entry_markdown,
        session_log_path=session_log_path,
    )
    reflection_latest_path, reflection_archive_path = write_reflection_conclusion(
        conclusion=reflection_output.reflection_conclusion,
        session_id=session_id,
    )

    behavior_intent = reflection_output.behavior_update_intent
    behavior_updated = False
    behavior_before_path: Path | None = None
    behavior_after_path: Path | None = None
    behavior_diff_path: Path | None = None
    if behavior_intent.should_update_behavior and behavior_intent.updated_behavior_markdown:
        (
            behavior_updated,
            behavior_before_path,
            behavior_after_path,
            behavior_diff_path,
        ) = _write_behavior_with_history(
            new_behavior_text=behavior_intent.updated_behavior_markdown,
            behavior_history_dir=logs_root / "behaviors",
            session_id=session_id,
        )

    _append_session_log_post_run(
        session_log_path=session_log_path,
        journal_path=journal_path,
        reflection_latest_path=reflection_latest_path,
        reflection_archive_path=reflection_archive_path,
        behavior_updated=behavior_updated,
        behavior_diff_path=behavior_diff_path,
        behavior_update_summary=behavior_intent.update_summary,
    )

    return PostRunMemoryResult(
        journal_file_path=journal_path,
        reflection_latest_path=reflection_latest_path,
        reflection_archive_path=reflection_archive_path,
        behavior_file_path=BEHAVIOR_PATH,
        behavior_history_before_path=behavior_before_path,
        behavior_history_after_path=behavior_after_path,
        behavior_diff_path=behavior_diff_path,
        behavior_updated=behavior_updated,
        behavior_update_summary=behavior_intent.update_summary,
    )
