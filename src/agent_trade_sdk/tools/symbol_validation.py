from __future__ import annotations

import re
from dataclasses import dataclass


_TOOL_FRAGMENT_PATTERNS = (
    "<|TOOL_CALL",
    "TOOL_CALL_ARGUMENT",
    "GET_MARKET_QUOTE",
    "GET_PRICE_HISTORY",
    "PLACE_MARKET_ORDER",
    "OPEN_SHORT_POSITION",
    "CLOSE_OPEN_POSITION",
)

_INVALID_CHARS = set("{}[]()<>|:/\\'\"` \t\n\r")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=]{0,11}$")
_FUTURE_LIKE_PATTERN = re.compile(r"^[A-Z]{1,6}=F$")


@dataclass(frozen=True)
class SymbolValidationError(ValueError):
    symbol: str
    reason: str

    def __str__(self) -> str:
        return f"Invalid symbol '{self.symbol}': {self.reason}"


def normalize_symbol(symbol: str) -> str:
    raw = str(symbol or "")
    cleaned = raw.strip().upper()
    if not cleaned:
        raise SymbolValidationError(raw, "empty symbol")
    if cleaned.startswith(":"):
        raise SymbolValidationError(cleaned, "unexpected ':' prefix")
    if any(fragment in cleaned for fragment in _TOOL_FRAGMENT_PATTERNS):
        raise SymbolValidationError(cleaned, "tool-call fragment detected in symbol")
    if any(char in _INVALID_CHARS for char in cleaned):
        raise SymbolValidationError(cleaned, "contains invalid characters")
    if _FUTURE_LIKE_PATTERN.fullmatch(cleaned):
        return cleaned
    if not _SYMBOL_PATTERN.fullmatch(cleaned):
        raise SymbolValidationError(cleaned, "does not match allowed symbol pattern")
    return cleaned


def normalize_symbols_csv(symbols_csv: str) -> tuple[list[str], list[dict[str, str]]]:
    normalized: list[str] = []
    errors: list[dict[str, str]] = []
    for token in str(symbols_csv or "").split(","):
        candidate = token.strip()
        if not candidate:
            continue
        try:
            normalized.append(normalize_symbol(candidate))
        except SymbolValidationError as exc:
            errors.append({"symbol": candidate, "error": str(exc)})
    return normalized, errors
