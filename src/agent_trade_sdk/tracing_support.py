from __future__ import annotations

import json
from typing import Any

from agents import RunConfig, set_tracing_export_api_key

from agent_trade_sdk.config import Settings


def _patch_sdk_tracing_usage_sanitizer() -> None:
    """Defensive patch for SDK/API schema drift on trace generation usage fields.

    Some SDK versions include usage keys rejected by the tracing ingest API.
    We keep only conservative keys to avoid non-fatal 400 spam.
    """
    try:
        from agents.tracing.processors import BackendSpanExporter
    except Exception:
        return

    safe_usage_keys = frozenset({"input_tokens", "output_tokens"})
    current = getattr(BackendSpanExporter, "_OPENAI_TRACING_ALLOWED_USAGE_KEYS", None)
    if current != safe_usage_keys:
        BackendSpanExporter._OPENAI_TRACING_ALLOWED_USAGE_KEYS = safe_usage_keys


def _metadata_value_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _compact_trace_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    compact = {
        key: _metadata_value_to_string(value)
        for key, value in metadata.items()
        if value is not None
    }
    return compact or None


def ensure_tracing_export_ready(
    *,
    enable_tracing: bool,
    settings: Settings | None = None,
) -> str | None:
    if not enable_tracing:
        return None

    resolved_settings = settings or Settings.from_env(require_openrouter=False)
    api_key = resolved_settings.openai_tracing_api_key
    if not api_key:
        raise RuntimeError(
            "Tracing is enabled but no tracing export key was found. "
            "Set OPENAI_TRACING_API_KEY (preferred) or OPENAI_API_KEY."
        )

    _patch_sdk_tracing_usage_sanitizer()
    set_tracing_export_api_key(api_key)
    return api_key


def build_trace_export_config(
    *,
    enable_tracing: bool,
    settings: Settings | None = None,
) -> dict[str, str] | None:
    api_key = ensure_tracing_export_ready(enable_tracing=enable_tracing, settings=settings)
    if not api_key:
        return None
    return {"api_key": api_key}


def build_agents_run_config(
    *,
    enable_tracing: bool,
    workflow_name: str,
    group_id: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> RunConfig:
    return RunConfig(
        tracing_disabled=not enable_tracing,
        tracing=build_trace_export_config(enable_tracing=enable_tracing, settings=settings),
        workflow_name=workflow_name,
        group_id=group_id,
        trace_metadata=_compact_trace_metadata(trace_metadata),
    )
