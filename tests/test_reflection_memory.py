from __future__ import annotations

from pydantic import ValidationError

from agent_trade_sdk.reflection_memory import (
    ReflectionConclusion,
    build_fallback_reflection_from_short_memory,
)


def test_reflection_conclusion_schema_limits() -> None:
    valid = ReflectionConclusion(
        generated_at_utc="2026-03-05T15:00:00+00:00",
        source_session_id="abc12345",
        strategy_health_score=0.72,
        stall_flags=["no_trade_streak>=3"],
        what_worked=["Bonne discipline d'exécution"],
        what_failed=["Exploration sectorielle faible"],
        next_run_focus=["Comparer finance vs énergie vs industrie"],
        hard_rules_next_run=["Exécuter 2 scans hors semiconducteurs"],
        conclusion_for_prompt="Synthèse courte exploitable pour le prochain run.",
    )
    assert valid.strategy_health_score == 0.72

    too_long = "x" * 1501
    try:
        ReflectionConclusion(
            generated_at_utc="2026-03-05T15:00:00+00:00",
            source_session_id="abc12345",
            strategy_health_score=0.5,
            conclusion_for_prompt=too_long,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for conclusion_for_prompt > 1500 chars.")


def test_fallback_reflection_from_short_memory() -> None:
    short_memory = {
        "session_summary": "Session de marché stable, peu d'opportunités claires.",
        "decision_report": "NO_TRADE répété et focalisation trop semis.",
        "self_critique": "Exploration insuffisante hors semiconducteurs.",
        "pitfalls_to_avoid_next_run": ["Rester bloqué sur NVDA"],
        "next_session_directives": ["Scanner finance et énergie en priorité"],
        "trading_decision": {"action": "NO_TRADE"},
    }
    reflection = build_fallback_reflection_from_short_memory(
        short_memory,
        session_id="sess1234",
    )
    assert reflection is not None
    assert reflection["source_session_id"] == "sess1234"
    assert "fallback_from_short_memory" in reflection["stall_flags"]
    assert len(reflection["conclusion_for_prompt"]) <= 1500

