from __future__ import annotations

import json

import pytest

from agent_trade_sdk import memory_architecture as ma


def test_parse_llm_output_with_text_and_fenced_json() -> None:
    raw_output = """
Analyse desk avant JSON.

```json
{
  "trading_decision": {
    "action": "NO_TRADE",
    "symbol": null,
    "confidence": "medium",
    "rationale": "Attente de confirmation.",
    "signal_principal": "Market risk-off",
    "risk_identified": "Volatilité élevée",
    "invalidation_condition": "Rebond confirmé",
    "executed_order": null
  },
  "session_summary": "Résumé",
  "decision_report": "Rapport",
  "self_critique": "Critique",
  "pitfalls_to_avoid_next_run": [],
  "next_session_directives": [],
  "open_questions": [],
  "long_memory_update_intent": {
    "should_update_behavior": false,
    "why": "No durable change",
    "update_summary": [],
    "updated_behavior_markdown": null
  }
}
```
"""
    parsed = ma.parse_llm_trading_session_output(raw_output)
    assert parsed.trading_decision.action == "NO_TRADE"
    assert parsed.session_summary == "Résumé"


def test_parse_llm_output_prefers_payload_with_trading_decision() -> None:
    raw_output = """
```json
{"foo": "bar"}
```

```json
{
  "trading_decision": {
    "action": "BUY",
    "symbol": "SPY",
    "confidence": "high",
    "rationale": "Signal valide",
    "signal_principal": "RS",
    "risk_identified": "Risque normal",
    "invalidation_condition": "Cassure",
    "executed_order": null
  },
  "session_summary": "ok",
  "decision_report": "ok",
  "self_critique": "ok",
  "pitfalls_to_avoid_next_run": [],
  "next_session_directives": [],
  "open_questions": [],
  "long_memory_update_intent": {
    "should_update_behavior": false,
    "why": "No change",
    "update_summary": [],
    "updated_behavior_markdown": null
  }
}
```
"""
    parsed = ma.parse_llm_trading_session_output(raw_output)
    assert parsed.trading_decision.action == "BUY"
    assert parsed.trading_decision.symbol == "SPY"


def test_apply_memory_outputs_writes_fallback_short_memory_on_parse_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_dir = tmp_path / "memory" / "short"
    short_latest = short_dir / "latest.json"
    short_archive = short_dir / "archive"
    monkeypatch.setattr(ma, "SHORT_MEMORY_DIR", short_dir)
    monkeypatch.setattr(ma, "SHORT_MEMORY_LATEST_PATH", short_latest)
    monkeypatch.setattr(ma, "SHORT_MEMORY_ARCHIVE_DIR", short_archive)

    result = ma.apply_memory_outputs(
        raw_final_output="this is not json at all",
        session_id="sess1234",
        model_name="test-model",
        logs_root=tmp_path / "logs",
        runtime_summary={"event_count": 12, "tool_calls": [], "tool_outputs": []},
    )
    assert result.parse_error is not None
    assert result.short_memory_latest_path is not None
    assert result.short_memory_latest_path.exists()

    payload = json.loads(result.short_memory_latest_path.read_text(encoding="utf-8"))
    assert payload["trading_decision"]["action"] == "NO_TRADE"
    assert payload["long_memory_update_intent"]["should_update_behavior"] is False
