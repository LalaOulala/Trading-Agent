from __future__ import annotations

import asyncio
import json

import agent_trade_sdk.tools.trading as trading


def test_get_market_clock_snapshot_tool_output_schema(monkeypatch) -> None:
    class DummyBroker:
        def get_market_clock(self) -> dict[str, object]:
            return {
                "is_open": True,
                "timestamp": "2026-03-05T14:30:00+00:00",
                "next_open": None,
                "next_close": "2026-03-05T21:00:00+00:00",
            }

    monkeypatch.setattr(trading, "AlpacaPaperBroker", DummyBroker)

    raw = asyncio.run(trading.get_market_clock_snapshot.on_invoke_tool(None, "{}"))
    payload = json.loads(raw)

    assert set(payload.keys()) == {"is_open", "timestamp", "next_open", "next_close", "source"}
    assert payload["is_open"] is True
    assert payload["source"] == "alpaca_tool"

