from __future__ import annotations

from agent_trade_sdk.runner import _clamp_trace_payload, _compact_runtime_summary_for_trace


def test_clamp_trace_payload_truncates_large_data() -> None:
    payload = {"blob": "x" * 20_000}
    clamped = _clamp_trace_payload(payload, max_chars=1_000)
    assert clamped.get("truncated") is True
    assert clamped.get("original_size_chars", 0) > 1_000


def test_compact_runtime_summary_limits_output_size() -> None:
    runtime_summary = {
        "event_count": 3000,
        "tool_error_count": 2,
        "tool_calls": [
            {"event_index": idx, "tool_name": "web_search_tavily", "arguments_excerpt": "a" * 800}
            for idx in range(20)
        ],
        "tool_outputs": [
            {"event_index": idx, "tool_name": "get_market_snapshot", "output_excerpt": "b" * 1200}
            for idx in range(20)
        ],
        "reasoning_summaries": ["c" * 1400] * 10,
        "message_outputs": ["d" * 1400] * 10,
    }
    compact = _compact_runtime_summary_for_trace(runtime_summary)
    assert compact["tool_calls_count"] == 20
    assert compact["tool_outputs_count"] == 20
    assert len(compact["tool_calls_tail"]) <= 8
    assert len(compact["tool_outputs_tail"]) <= 8
