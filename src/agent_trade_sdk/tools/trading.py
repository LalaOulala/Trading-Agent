from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from agents import function_tool
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from agent_trade_sdk.config import Settings


OrderSideLiteral = Literal["buy", "sell"]
TimeInForceLiteral = Literal["day", "gtc", "opg", "cls", "ioc", "fok"]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _to_float_or_none(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _map_time_in_force(value: TimeInForceLiteral) -> TimeInForce:
    mapping: dict[str, TimeInForce] = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "opg": TimeInForce.OPG,
        "cls": TimeInForce.CLS,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }
    return mapping[value]


@dataclass(frozen=True)
class AlpacaConnectionConfig:
    api_key: str
    secret_key: str
    paper: bool
    base_url: str
    allowed_symbols: tuple[str, ...]
    max_notional_usd: float | None

    @classmethod
    def from_env(cls) -> "AlpacaConnectionConfig":
        settings = Settings.from_env(require_openrouter=False)
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise RuntimeError("Missing ALPACA_API_KEY and/or ALPACA_SECRET_KEY in environment.")
        return cls(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper,
            base_url=settings.alpaca_base_url,
            allowed_symbols=settings.trading_allowed_symbols,
            max_notional_usd=settings.trading_max_notional_usd,
        )


class AlpacaPaperBroker:
    def __init__(self, config: AlpacaConnectionConfig | None = None) -> None:
        self.config = config or AlpacaConnectionConfig.from_env()
        self.client = TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key,
            paper=self.config.paper,
            url_override=self.config.base_url,
        )
        self._ensure_paper_mode()

    def _ensure_paper_mode(self) -> None:
        if not self.config.paper:
            raise RuntimeError("Refusing to run: ALPACA_PAPER must be true for this project.")

    def _enforce_symbol_policy(self, symbol: str) -> str:
        normalized = symbol.upper().strip()
        allowed = self.config.allowed_symbols
        if allowed and normalized not in allowed:
            raise ValueError(f"Symbol {normalized} is not allowed. Allowed symbols: {allowed}")
        return normalized

    def _enforce_notional_policy(self, notional: float | None) -> None:
        if notional is None or self.config.max_notional_usd is None:
            return
        if notional > self.config.max_notional_usd:
            raise ValueError(
                f"Notional {notional} exceeds TRADING_MAX_NOTIONAL_USD={self.config.max_notional_usd}"
            )

    @staticmethod
    def _order_to_dict(order: Any) -> dict[str, Any]:
        return {
            "id": str(getattr(order, "id", None)),
            "symbol": getattr(order, "symbol", None),
            "side": _enum_value(getattr(order, "side", None)),
            "type": _enum_value(getattr(order, "type", None)),
            "status": _enum_value(getattr(order, "status", None)),
            "qty": str(getattr(order, "qty", None)),
            "notional": str(getattr(order, "notional", None)),
            "filled_qty": str(getattr(order, "filled_qty", None)),
            "filled_avg_price": str(getattr(order, "filled_avg_price", None)),
            "created_at": str(getattr(order, "created_at", None)),
        }

    def get_account(self) -> dict[str, Any]:
        account = self.client.get_account()
        return {
            "id": str(account.id),
            "status": str(account.status),
            "currency": str(account.currency),
            "buying_power": str(account.buying_power),
            "equity": str(account.equity),
            "cash": str(account.cash),
            "portfolio_value": str(account.portfolio_value),
            "pattern_day_trader": bool(account.pattern_day_trader),
        }

    def get_positions(self) -> list[dict[str, Any]]:
        positions = self.client.get_all_positions()
        return [
            {
                "symbol": position.symbol,
                "side": _enum_value(position.side),
                "qty": str(position.qty),
                "market_value": str(position.market_value),
                "avg_entry_price": str(position.avg_entry_price),
                "unrealized_pl": str(position.unrealized_pl),
            }
            for position in positions
        ]

    def submit_market_order(
        self,
        symbol: str,
        side: OrderSideLiteral,
        qty: float | None = None,
        notional: float | None = None,
        time_in_force: TimeInForceLiteral = "day",
    ) -> dict[str, Any]:
        normalized_symbol = self._enforce_symbol_policy(symbol)
        qty = _to_float_or_none(qty)
        notional = _to_float_or_none(notional)

        if qty is None and notional is None:
            raise ValueError("Provide qty or notional.")
        if qty is not None and notional is not None:
            raise ValueError("Provide only one of qty or notional.")

        self._enforce_notional_policy(notional)

        order_data = MarketOrderRequest(
            symbol=normalized_symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            qty=qty,
            notional=notional,
            time_in_force=_map_time_in_force(time_in_force),
        )
        order = self.client.submit_order(order_data=order_data)
        return self._order_to_dict(order)

    def close_position(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = self._enforce_symbol_policy(symbol)
        order = self.client.close_position(symbol_or_asset_id=normalized_symbol)
        return self._order_to_dict(order)


@function_tool
def get_account_snapshot() -> str:
    """Return Alpaca paper account metrics as compact JSON."""

    broker = AlpacaPaperBroker()
    return json.dumps(broker.get_account(), ensure_ascii=False)


@function_tool
def list_open_positions() -> str:
    """Return currently open Alpaca positions as compact JSON."""

    broker = AlpacaPaperBroker()
    return json.dumps({"positions": broker.get_positions()}, ensure_ascii=False)


@function_tool
def place_market_order(
    symbol: str,
    side: OrderSideLiteral,
    qty: float | None = None,
    notional: float | None = None,
    time_in_force: TimeInForceLiteral = "day",
) -> str:
    """Submit a market order on Alpaca paper trading and return execution metadata as JSON."""

    broker = AlpacaPaperBroker()
    order = broker.submit_market_order(
        symbol=symbol,
        side=side,
        qty=qty,
        notional=notional,
        time_in_force=time_in_force,
    )
    return json.dumps(order, ensure_ascii=False)


@function_tool
def open_short_position(
    symbol: str,
    qty: float,
    time_in_force: TimeInForceLiteral = "day",
) -> str:
    """Open a short position using a SELL market order (paper account must support shorting)."""

    broker = AlpacaPaperBroker()
    order = broker.submit_market_order(
        symbol=symbol,
        side="sell",
        qty=qty,
        notional=None,
        time_in_force=time_in_force,
    )
    return json.dumps(order, ensure_ascii=False)


@function_tool
def close_open_position(symbol: str) -> str:
    """Close an open position for the given symbol."""

    broker = AlpacaPaperBroker()
    order = broker.close_position(symbol=symbol)
    return json.dumps(order, ensure_ascii=False)
