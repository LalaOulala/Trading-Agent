from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent


PARIS_TZ = ZoneInfo("Europe/Paris")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _paris_now() -> datetime:
    return datetime.now(PARIS_TZ)


def _safe_dump(data: Any, max_chars: int = 6000) -> str:
    def default_serializer(value: Any) -> str:
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=default_serializer)
    except Exception:
        text = str(data)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


def _truncate_text(value: str, max_chars: int = 800) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + " ... (truncated)"


def _to_mapping(raw_item: Any) -> dict[str, Any]:
    if isinstance(raw_item, dict):
        return raw_item
    if hasattr(raw_item, "model_dump"):
        try:
            data = raw_item.model_dump(exclude_none=True)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"value": str(raw_item)}


def _extract_message_text(raw: dict[str, Any]) -> str:
    content = raw.get("content")
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text_value = item.get("text")
        if isinstance(text_value, str):
            chunks.append(text_value)
            continue
        if isinstance(text_value, dict):
            maybe = text_value.get("value")
            if isinstance(maybe, str):
                chunks.append(maybe)
    return "\n".join(chunks).strip()


def _extract_reasoning_summary(raw: dict[str, Any]) -> str:
    summary = raw.get("summary")
    if isinstance(summary, str):
        return summary
    if isinstance(summary, list):
        parts: list[str] = []
        for block in summary:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, dict):
                    value = text.get("value")
                    if isinstance(value, str):
                        parts.append(value)
        return "\n".join([p for p in parts if p]).strip()
    return ""


@dataclass
class SessionMarkdownLogger:
    log_dir: Path
    prompt: str
    model_name: str
    tracing_enabled: bool
    session_id: str
    file_path: Path = field(init=False)
    event_index: int = field(default=0, init=False)
    _call_id_to_tool: dict[str, str] = field(default_factory=dict, init=False)
    _tool_calls_summary: list[dict[str, Any]] = field(default_factory=list, init=False)
    _tool_outputs_summary: list[dict[str, Any]] = field(default_factory=list, init=False)
    _reasoning_summaries: list[str] = field(default_factory=list, init=False)
    _message_outputs: list[str] = field(default_factory=list, init=False)

    _alpaca_order_tools: set[str] = field(
        default_factory=lambda: {"place_market_order", "open_short_position", "close_open_position"},
        init=False,
    )
    _snapshot_tools: set[str] = field(default_factory=lambda: {"get_market_snapshot"}, init=False)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = _paris_now().strftime("%d-%m-%Y_%Hh%M_%S")
        self.file_path = self.log_dir / f"session_{stamp}_{self.session_id}.md"
        self._write_header()

    def _append(self, text: str) -> None:
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(text)

    def _write_header(self) -> None:
        utc_ts = _utc_now().isoformat()
        paris_ts = _paris_now().isoformat()
        self._append(
            "# Trading Agent Session Log\n\n"
            f"- session_id: `{self.session_id}`\n"
            f"- started_at_utc: `{utc_ts}`\n"
            f"- started_at_paris: `{paris_ts}`\n"
            f"- model: `{self.model_name}`\n"
            f"- tracing_enabled: `{str(self.tracing_enabled).lower()}`\n\n"
            "## Prompt\n\n"
            f"```text\n{self.prompt}\n```\n\n"
        )

    def log_input_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._append("## Input Snapshot\n\n")
        self._append(f"```json\n{_safe_dump(snapshot)}\n```\n\n")

    def log_separator_new_snapshot(self, source_tool: str) -> None:
        now_utc = _utc_now().isoformat()
        self._append(
            "---\n\n"
            f"## New Snapshot Received\n\n- time_utc: `{now_utc}`\n- source_tool: `{source_tool}`\n\n"
        )

    def log_stream_event(self, event: Any) -> None:
        self.event_index += 1
        now_utc = _utc_now().isoformat()

        if isinstance(event, RawResponsesStreamEvent):
            return

        if isinstance(event, AgentUpdatedStreamEvent):
            self._append(
                f"### Event {self.event_index} - Agent Updated\n\n"
                f"- time_utc: `{now_utc}`\n"
                f"- new_agent: `{event.new_agent.name}`\n\n"
            )
            return

        if not isinstance(event, RunItemStreamEvent):
            self._append(
                f"### Event {self.event_index} - Unknown Event\n\n"
                f"- time_utc: `{now_utc}`\n"
                f"- type: `{type(event).__name__}`\n\n"
            )
            return

        raw = _to_mapping(getattr(event.item, "raw_item", {}))
        item_type = getattr(event.item, "type", "unknown")
        self._append(
            f"### Event {self.event_index} - `{event.name}`\n\n"
            f"- time_utc: `{now_utc}`\n"
            f"- item_type: `{item_type}`\n"
        )

        if event.name == "tool_called":
            tool_name = str(raw.get("name", "unknown_tool"))
            call_id = str(raw.get("call_id", raw.get("id", "")))
            if call_id:
                self._call_id_to_tool[call_id] = tool_name
            self._append(f"- tool_name: `{tool_name}`\n")
            if call_id:
                self._append(f"- call_id: `{call_id}`\n")
            if tool_name in self._alpaca_order_tools:
                self._append("- category: `alpaca_order`\n")
            arguments = raw.get("arguments")
            self._tool_calls_summary.append(
                {
                    "event_index": self.event_index,
                    "tool_name": tool_name,
                    "call_id": call_id or None,
                    "arguments_excerpt": _truncate_text(_safe_dump(arguments, max_chars=1200))
                    if arguments is not None
                    else None,
                }
            )
            if arguments:
                self._append("\n**Tool arguments**\n\n")
                self._append(f"```json\n{_safe_dump(arguments)}\n```\n")

        elif event.name == "tool_output":
            output_raw = _to_mapping(getattr(event.item, "raw_item", {}))
            call_id = str(output_raw.get("call_id", output_raw.get("id", "")))
            tool_name = self._call_id_to_tool.get(call_id, "unknown_tool")
            self._append(f"- tool_name: `{tool_name}`\n")
            if call_id:
                self._append(f"- call_id: `{call_id}`\n")
            if tool_name in self._snapshot_tools:
                self._append("- category: `snapshot_update`\n")
                self._append("\n")
                self.log_separator_new_snapshot(tool_name)
            if tool_name in self._alpaca_order_tools:
                self._append("- category: `alpaca_order_execution`\n")
            output_value = getattr(event.item, "output", None)
            output_excerpt = _truncate_text(_safe_dump(output_value, max_chars=1500))
            is_error = "error" in output_excerpt.lower()
            self._tool_outputs_summary.append(
                {
                    "event_index": self.event_index,
                    "tool_name": tool_name,
                    "call_id": call_id or None,
                    "is_error": is_error,
                    "category": (
                        "snapshot_update"
                        if tool_name in self._snapshot_tools
                        else "alpaca_order_execution"
                        if tool_name in self._alpaca_order_tools
                        else None
                    ),
                    "output_excerpt": output_excerpt,
                }
            )
            self._append("\n**Tool output**\n\n")
            self._append(f"```json\n{_safe_dump(output_value)}\n```\n")

        elif event.name == "reasoning_item_created":
            summary = _extract_reasoning_summary(raw)
            if summary:
                self._reasoning_summaries.append(_truncate_text(summary, max_chars=2000))
                self._append(f"\n**Reasoning summary**\n\n{summary}\n")
            else:
                self._append("\n**Reasoning raw item**\n\n")
                self._append(f"```json\n{_safe_dump(raw)}\n```\n")

        elif event.name == "message_output_created":
            text = _extract_message_text(raw)
            if text:
                self._message_outputs.append(_truncate_text(text, max_chars=3000))
                self._append("\n**Message output**\n\n")
                self._append(f"```text\n{text}\n```\n")
            else:
                self._append("\n**Message raw item**\n\n")
                self._append(f"```json\n{_safe_dump(raw)}\n```\n")
        else:
            self._append("\n**Event raw item**\n\n")
            self._append(f"```json\n{_safe_dump(raw)}\n```\n")

        self._append("\n")

    def build_runtime_summary(self) -> dict[str, Any]:
        tool_error_count = sum(1 for item in self._tool_outputs_summary if item.get("is_error"))
        return {
            "event_count": self.event_index,
            "tool_calls": self._tool_calls_summary,
            "tool_outputs": self._tool_outputs_summary,
            "tool_error_count": tool_error_count,
            "reasoning_summaries": self._reasoning_summaries[-10:],
            "message_outputs": self._message_outputs[-10:],
        }

    def log_final_output(self, output: str) -> None:
        finished_utc = _utc_now().isoformat()
        self._append(
            "## Final Output\n\n"
            f"- finished_at_utc: `{finished_utc}`\n\n"
            f"```text\n{output}\n```\n\n"
        )

    def log_error(self, error: Exception) -> None:
        now_utc = _utc_now().isoformat()
        self._append(
            "## Error\n\n"
            f"- time_utc: `{now_utc}`\n"
            f"- type: `{type(error).__name__}`\n\n"
            f"```text\n{str(error)}\n```\n\n"
        )
