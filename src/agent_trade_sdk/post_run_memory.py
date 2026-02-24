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

from agent_trade_sdk.agent import SOUL_PATH
from agent_trade_sdk.config import Settings
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


class PostRunMemoryOutput(BaseModel):
    journal_entry_markdown: str = Field(
        ...,
        description="Entrée markdown du journal personnel pour ce cycle (1 entrée complète).",
    )
    updated_soul_markdown: str = Field(
        ...,
        description="Contenu COMPLET de SOUL.md après mise à jour (markdown complet).",
    )
    soul_update_summary: list[str] = Field(
        default_factory=list,
        description="Résumé court des changements apportés au SOUL.",
    )


@dataclass(frozen=True)
class PostRunMemoryResult:
    journal_file_path: Path
    soul_file_path: Path
    soul_history_before_path: Path
    soul_history_after_path: Path
    soul_diff_path: Path
    soul_update_summary: list[str]


def _build_reflection_agent(model_name: str | None = None) -> Agent:
    settings = Settings.from_env(require_openrouter=True)
    model = LitellmModel(
        model=model_name or settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )

    instructions = (
        "Tu es le module de mémoire/réflexion d'un agent de trading paper autonome.\n"
        "Ta mission: analyser un cycle de run terminé, écrire une entrée de journal utile pour debug "
        "humain, puis réécrire complètement SOUL.md avec les apprentissages de process et de trading "
        "observés sur ce cycle.\n"
        "Tu peux modifier directement SOUL.md sans validation humaine. Garde un document cohérent et lisible.\n"
        "Conserve un style concret, opérationnel, sans fluff. Retourne la sortie structurée demandée."
    )

    return Agent(
        name="TradingAgentMemoryWriter",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(include_usage=True),
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
    current_soul: str,
) -> str:
    return (
        "Tu dois faire la sortie de boucle d'un run de trading.\n\n"
        "Tâches obligatoires:\n"
        "1) Écrire une entrée de journal personnel (Markdown) résumant ce cycle.\n"
        "2) Réécrire ENTIEREMENT SOUL.md en intégrant ce que l'agent a appris pendant ce cycle.\n\n"
        "Directives:\n"
        "- Le journal doit être utile pour debug humain (faits, outils, erreurs, décision, exécution, leçons).\n"
        "- Le SOUL mis à jour doit rester un contrat de comportement exploitable pour le prochain cycle.\n"
        "- Tu peux changer librement le SOUL (expérience volontairement autonome), mais garde de la cohérence.\n"
        "- N'inclus pas de backticks triples dans les champs texte si possible.\n\n"
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
        "=== SOUL.md ACTUEL ===\n"
        f"{_truncate(current_soul, 20000)}\n"
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
    soul_update_summary: list[str],
    soul_diff_path: Path,
) -> None:
    if not session_log_path.exists():
        return
    lines = [
        "## Post-Run Memory\n",
        "\n",
        f"- time_utc: `{_utc_now_iso()}`\n",
        f"- journal_entry_file: `{journal_path}`\n",
        f"- soul_diff_file: `{soul_diff_path}`\n",
        "\n",
    ]
    if soul_update_summary:
        lines.append("### SOUL Update Summary\n\n")
        for item in soul_update_summary:
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
        f"\n\n---\n\n"
        f"## Run {now.strftime('%H:%M:%S')} UTC - session `{session_id}`\n\n"
        f"- session_log: `{session_log_path}`\n\n"
    )
    with day_file.open("a", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(journal_entry_markdown.strip() + "\n")
    return day_file


def _write_soul_with_history(
    *,
    soul_path: Path,
    new_soul_text: str,
    soul_history_dir: Path,
    session_id: str,
) -> tuple[Path, Path, Path]:
    before_text = soul_path.read_text(encoding="utf-8") if soul_path.exists() else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    soul_history_dir.mkdir(parents=True, exist_ok=True)

    before_path = soul_history_dir / f"SOUL_before_{timestamp}_{session_id}.md"
    after_path = soul_history_dir / f"SOUL_after_{timestamp}_{session_id}.md"
    diff_path = soul_history_dir / f"SOUL_diff_{timestamp}_{session_id}.diff"

    before_path.write_text(before_text, encoding="utf-8")
    _atomic_write_text(soul_path, new_soul_text.rstrip() + "\n")
    after_text = soul_path.read_text(encoding="utf-8")
    after_path.write_text(after_text, encoding="utf-8")

    diff_text = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="SOUL_before.md",
            tofile="SOUL_after.md",
        )
    )
    diff_path.write_text(diff_text, encoding="utf-8")
    return before_path, after_path, diff_path


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
    current_soul = SOUL_PATH.read_text(encoding="utf-8") if SOUL_PATH.exists() else ""
    reflection_agent = _build_reflection_agent(model_name=model_name)
    prompt = _build_post_run_prompt(
        session_id=session_id,
        user_prompt=user_prompt,
        final_output=final_output,
        bootstrap_context=bootstrap_context,
        runtime_summary=runtime_summary,
        session_log_path=session_log_path,
        current_soul=current_soul,
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
        # Defensive fallbacks for providers/models with different structured-output behavior.
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
    before_path, after_path, diff_path = _write_soul_with_history(
        soul_path=SOUL_PATH,
        new_soul_text=reflection_output.updated_soul_markdown,
        soul_history_dir=logs_root / "souls",
        session_id=session_id,
    )
    _append_session_log_post_run(
        session_log_path=session_log_path,
        journal_path=journal_path,
        soul_update_summary=reflection_output.soul_update_summary,
        soul_diff_path=diff_path,
    )

    return PostRunMemoryResult(
        journal_file_path=journal_path,
        soul_file_path=SOUL_PATH,
        soul_history_before_path=before_path,
        soul_history_after_path=after_path,
        soul_diff_path=diff_path,
        soul_update_summary=reflection_output.soul_update_summary,
    )
