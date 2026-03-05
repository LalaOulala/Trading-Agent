from __future__ import annotations

import json
from pathlib import Path

import agent_trade_sdk.strategy_guardrails as sg


def _write_short_memory(path: Path, idx: int, action: str, rationale: str) -> None:
    payload = {
        "generated_at_utc": f"2026-03-05T1{idx}:00:00+00:00",
        "source_session_id": f"sess{idx}",
        "trading_decision": {"action": action, "rationale": rationale},
        "session_summary": rationale,
        "decision_report": rationale,
        "self_critique": rationale,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compute_stall_guardrails_triggers_on_no_trade_streak(tmp_path: Path, monkeypatch) -> None:
    archive_dir = tmp_path / "memory" / "short" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sg, "SHORT_MEMORY_ARCHIVE_DIR", archive_dir)

    _write_short_memory(
        archive_dir / "short_memory_20260305T140000Z_s1.json",
        idx=1,
        action="NO_TRADE",
        rationale="Focus semis NVDA AMD only.",
    )
    _write_short_memory(
        archive_dir / "short_memory_20260305T141000Z_s2.json",
        idx=2,
        action="NO_TRADE",
        rationale="Still semis, no exploration.",
    )
    _write_short_memory(
        archive_dir / "short_memory_20260305T142000Z_s3.json",
        idx=3,
        action="NO_TRADE",
        rationale="Semiconducteurs uniquement.",
    )

    snapshot = {
        "portfolio": {
            "positions": [
                {"symbol": "AMD", "market_value": "1200", "unrealized_pl": "-40"},
                {"symbol": "NVDA", "market_value": "800", "unrealized_pl": "-20"},
            ]
        },
        "market": {
            "snapshots": [
                {"symbol": "AMD", "quote": {"sector": "Technology"}},
                {"symbol": "NVDA", "quote": {"sector": "Technology"}},
            ]
        },
    }

    result = sg.compute_stall_guardrails(snapshot, lookback_runs=6)
    assert result.no_trade_streak >= 3
    assert result.stall_score >= 0.55
    assert result.hard_rules_next_run
    assert "no_trade_streak>=3" in result.stall_flags

