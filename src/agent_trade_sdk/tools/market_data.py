from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Literal

import yfinance as yf
from agents import function_tool


HistoryPeriod = Literal[
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]
HistoryInterval = Literal[
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _safe_fast_info(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        fast_info = ticker.fast_info
        keys = (
            "last_price",
            "previous_close",
            "open",
            "day_high",
            "day_low",
            "last_volume",
            "market_cap",
            "currency",
            "exchange",
            "timezone",
            "year_high",
            "year_low",
        )
        payload: dict[str, Any] = {}
        for key in keys:
            try:
                payload[key] = _to_jsonable(fast_info.get(key))
            except Exception:
                payload[key] = None
        return payload
    except Exception:
        return {}


def yfinance_quote_raw(symbol: str) -> dict[str, Any]:
    normalized_symbol = symbol.strip().upper()
    ticker = yf.Ticker(normalized_symbol)

    info: dict[str, Any] = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    quote = {
        "symbol": normalized_symbol,
        "short_name": _to_jsonable(info.get("shortName")),
        "long_name": _to_jsonable(info.get("longName")),
        "exchange": _to_jsonable(info.get("exchange")),
        "currency": _to_jsonable(info.get("currency")),
        "regular_market_price": _to_jsonable(info.get("regularMarketPrice")),
        "regular_market_previous_close": _to_jsonable(info.get("regularMarketPreviousClose")),
        "regular_market_change": _to_jsonable(info.get("regularMarketChange")),
        "regular_market_change_percent": _to_jsonable(info.get("regularMarketChangePercent")),
        "regular_market_volume": _to_jsonable(info.get("regularMarketVolume")),
        "market_cap": _to_jsonable(info.get("marketCap")),
        "sector": _to_jsonable(info.get("sector")),
        "industry": _to_jsonable(info.get("industry")),
    }

    recent_bars: list[dict[str, Any]] = []
    try:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty:
            for idx, row in hist.tail(5).iterrows():
                recent_bars.append(
                    {
                        "timestamp": _to_jsonable(idx),
                        "open": _to_jsonable(row.get("Open")),
                        "high": _to_jsonable(row.get("High")),
                        "low": _to_jsonable(row.get("Low")),
                        "close": _to_jsonable(row.get("Close")),
                        "volume": _to_jsonable(row.get("Volume")),
                    }
                )
    except Exception:
        recent_bars = []

    return {
        "symbol": normalized_symbol,
        "quote": quote,
        "fast_info": _safe_fast_info(ticker),
        "recent_bars_1d": recent_bars,
    }


@function_tool
def get_market_quote(symbol: str) -> str:
    """Get a compact quote snapshot for one symbol from Yahoo Finance."""

    payload = yfinance_quote_raw(symbol=symbol)
    return json.dumps(payload, ensure_ascii=False)


def yfinance_price_history_raw(
    symbol: str,
    period: HistoryPeriod = "1mo",
    interval: HistoryInterval = "1d",
    max_points: int = 60,
) -> dict[str, Any]:
    """Get OHLCV history for a symbol from Yahoo Finance as a dict."""

    normalized_symbol = symbol.strip().upper()
    ticker = yf.Ticker(normalized_symbol)
    history = ticker.history(period=period, interval=interval, auto_adjust=False)

    rows: list[dict[str, Any]] = []
    if not history.empty:
        tail = history.tail(max_points)
        for idx, row in tail.iterrows():
            rows.append(
                {
                    "timestamp": _to_jsonable(idx),
                    "open": _to_jsonable(row.get("Open")),
                    "high": _to_jsonable(row.get("High")),
                    "low": _to_jsonable(row.get("Low")),
                    "close": _to_jsonable(row.get("Close")),
                    "volume": _to_jsonable(row.get("Volume")),
                }
            )

    payload = {
        "symbol": normalized_symbol,
        "period": period,
        "interval": interval,
        "count": len(rows),
        "bars": rows,
    }
    return payload


@function_tool
def get_price_history(
    symbol: str,
    period: HistoryPeriod = "1mo",
    interval: HistoryInterval = "1d",
    max_points: int = 60,
) -> str:
    """Get OHLCV history for a symbol from Yahoo Finance as compact JSON."""

    payload = yfinance_price_history_raw(
        symbol=symbol,
        period=period,
        interval=interval,
        max_points=max_points,
    )
    return json.dumps(payload, ensure_ascii=False)


def yfinance_market_snapshot_raw(
    symbols_csv: str = "SPY,QQQ,DIA,IWM",
) -> dict[str, Any]:
    """Get a multi-symbol market snapshot from Yahoo Finance as a dict."""

    symbols = [token.strip().upper() for token in symbols_csv.split(",") if token.strip()]
    snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            snapshots.append(yfinance_quote_raw(symbol=symbol))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    payload = {
        "requested_symbols": symbols,
        "snapshots": snapshots,
        "errors": errors,
    }
    return payload


@function_tool
def get_market_snapshot(
    symbols_csv: str = "SPY,QQQ,DIA,IWM",
) -> str:
    """Get a multi-symbol market snapshot from Yahoo Finance."""

    payload = yfinance_market_snapshot_raw(symbols_csv=symbols_csv)
    return json.dumps(payload, ensure_ascii=False)
