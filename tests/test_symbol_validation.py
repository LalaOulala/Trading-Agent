from __future__ import annotations

import pytest

from agent_trade_sdk.tools import market_data
from agent_trade_sdk.tools.symbol_validation import SymbolValidationError, normalize_symbol
from agent_trade_sdk.tools.trading import AlpacaPaperBroker, AlpacaConnectionConfig


def test_normalize_symbol_valid() -> None:
    assert normalize_symbol("iyh") == "IYH"
    assert normalize_symbol("BRK-B") == "BRK-B"
    assert normalize_symbol("^VIX") == "^VIX"


@pytest.mark.parametrize(
    "value",
    [":IYH", "", " ", "}GET_MARKET_QUOTE:8 <|TOOL_CALL_ARGUMENT_BEGIN|> {", "SP:Y"],
)
def test_normalize_symbol_invalid(value: str) -> None:
    with pytest.raises(SymbolValidationError):
        normalize_symbol(value)


def test_yfinance_quote_raw_rejects_invalid_symbol_before_external_call() -> None:
    with pytest.raises(SymbolValidationError):
        market_data.yfinance_quote_raw(":IYH")


def test_market_snapshot_filters_invalid_symbols_without_external_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fake_quote(symbol: str) -> dict[str, str]:
        calls.append(symbol)
        return {"symbol": symbol}

    monkeypatch.setattr(market_data, "yfinance_quote_raw", _fake_quote)
    payload = market_data.yfinance_market_snapshot_raw(symbols_csv=":IYH,SPY")
    assert calls == ["SPY"]
    assert payload["requested_symbols"] == ["SPY"]
    assert payload["errors"]
    assert payload["errors"][0]["symbol"] == ":IYH"


def test_trading_symbol_policy_rejects_corrupted_symbol() -> None:
    config = AlpacaConnectionConfig(
        api_key="k",
        secret_key="s",
        paper=True,
        base_url="https://paper-api.alpaca.markets",
        allowed_symbols=(),
        max_notional_usd=None,
    )
    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)
    broker.config = config
    with pytest.raises(SymbolValidationError):
        broker._enforce_symbol_policy(":IYH")
