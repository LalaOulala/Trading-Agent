from __future__ import annotations

from agent_trade_sdk.source_quality import (
    evaluate_perplexity_quality,
    evaluate_tavily_quality,
)


def test_tavily_quality_report_stable_payload() -> None:
    payload = {
        "results": [
            {
                "title": "NVIDIA earnings beat expectations; guidance raised",
                "url": "https://www.reuters.com/markets/us/nvidia-earnings-2026-03-05/",
                "published_date": "2026-03-05T14:10:00+00:00",
                "content": "US equities, earnings, guidance, options flow and volatility.",
            },
            {
                "title": "Banks lead as yields move higher",
                "url": "https://www.cnbc.com/2026/03/05/us-markets-live-updates.html",
                "published_date": "2026-03-05T13:40:00+00:00",
                "content": "Financials outperform as treasury yields climb.",
            },
        ]
    }
    report = evaluate_tavily_quality(payload)
    assert report.provider == "tavily"
    assert report.duplicate_ratio <= 0.5
    assert report.finance_relevance_score >= 0.45
    assert report.usable_for_decision is True


def test_perplexity_quality_report_handles_missing_dates() -> None:
    payload = {
        "search_results": [
            {
                "title": "US market recap: sector rotation",
                "url": "https://www.bloomberg.com/news/articles/2026-03-05/rotation",
                "snippet": "Financials and industrials led the move.",
            },
            {
                "title": "US market recap: sector rotation",
                "url": "https://www.bloomberg.com/news/articles/2026-03-05/rotation",
                "snippet": "Duplicate entry to test duplicate ratio.",
            },
        ]
    }
    report = evaluate_perplexity_quality(payload)
    assert report.provider == "perplexity"
    assert report.freshness_hours_median is None
    assert report.duplicate_ratio >= 0.4
    assert "missing_publish_dates" in report.diagnostics

