from __future__ import annotations

from agent_trade_sdk.tools.perplexity_snapshot import (
    SUMMARY_END_MARKER,
    SUMMARY_START_MARKER,
    _extract_summary_5_lines,
    _overlap_ratio,
    build_perplexity_snapshot_prompt,
)


def test_extract_summary_5_lines_with_markers() -> None:
    text = (
        "Section A\n"
        f"{SUMMARY_START_MARKER}\n"
        "- ligne 1\n"
        "- ligne 2\n"
        "- ligne 3\n"
        "- ligne 4\n"
        "- ligne 5\n"
        "- ligne 6\n"
        f"{SUMMARY_END_MARKER}\n"
    )
    summary = _extract_summary_5_lines(text)
    lines = summary.splitlines()
    assert len(lines) == 5
    assert lines[0] == "ligne 1"
    assert lines[-1] == "ligne 5"


def test_extract_summary_5_lines_fallback_without_markers() -> None:
    text = "\n".join(
        [
            "Analyse desk...",
            "Résumé 5 lignes:",
            "1) point A",
            "2) point B",
            "3) point C",
            "4) point D",
            "5) point E",
        ]
    )
    summary = _extract_summary_5_lines(text)
    assert len(summary.splitlines()) == 5
    assert "point A" in summary


def test_overlap_ratio_detects_low_novelty() -> None:
    previous = "oil spike defense up risk-off small caps weak"
    current = "oil spike defense up risk-off small caps weak and gold bid"
    assert _overlap_ratio(previous, current) >= 0.65


def test_prompt_contains_continuity_and_summary_markers() -> None:
    prompt = build_perplexity_snapshot_prompt("line1\nline2")
    assert "Continuité depuis run précédent" in prompt
    assert SUMMARY_START_MARKER in prompt
    assert SUMMARY_END_MARKER in prompt
