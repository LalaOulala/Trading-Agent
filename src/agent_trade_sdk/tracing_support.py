from __future__ import annotations

from typing import Any

from agents import RunConfig, set_tracing_export_api_key

from agent_trade_sdk.config import Settings


def _compact_trace_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    compact = {key: value for key, value in metadata.items() if value is not None}
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
