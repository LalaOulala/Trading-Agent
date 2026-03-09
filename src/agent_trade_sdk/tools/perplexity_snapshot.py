from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_trade_sdk.config import Settings


ROOT_DIR = Path(__file__).resolve().parents[3]
PERPLEXITY_MEMORY_DIR = ROOT_DIR / "memory" / "perplexity"
PERPLEXITY_LATEST_SUMMARY_PATH = PERPLEXITY_MEMORY_DIR / "latest_summary.json"
PERPLEXITY_ARCHIVE_DIR = PERPLEXITY_MEMORY_DIR / "archive"

SUMMARY_START_MARKER = "=== SUMMARY_5_LINES ==="
SUMMARY_END_MARKER = "=== END_SUMMARY_5_LINES ==="


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _extract_message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return ""


def _extract_search_results(completion: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in getattr(completion, "search_results", None) or []:
        results.append(
            {
                "title": getattr(item, "title", None),
                "url": getattr(item, "url", None),
                "date": getattr(item, "date", None),
                "last_updated": getattr(item, "last_updated", None),
                "snippet": getattr(item, "snippet", None),
                "source": getattr(item, "source", None),
            }
        )
    return results


def _extract_citations(completion: Any) -> list[str]:
    citations = getattr(completion, "citations", None)
    if not citations:
        return []
    return [str(c) for c in citations]


def _normalize_summary_lines(text: str, *, max_lines: int = 5) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[\-*•\d\)\.\s]+", "", line).strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _extract_summary_5_lines(snapshot_text: str) -> str:
    if not snapshot_text.strip():
        return ""
    block_pattern = re.compile(
        rf"{re.escape(SUMMARY_START_MARKER)}\s*(.*?)\s*{re.escape(SUMMARY_END_MARKER)}",
        flags=re.DOTALL | re.IGNORECASE,
    )
    block_match = block_pattern.search(snapshot_text)
    if block_match:
        return _normalize_summary_lines(block_match.group(1), max_lines=5)

    lowered = snapshot_text.lower()
    fallback_markers = ("résumé 5 lignes", "resume 5 lines", "summary 5 lines")
    marker_index = min(
        (lowered.find(marker) for marker in fallback_markers if marker in lowered),
        default=-1,
    )
    if marker_index >= 0:
        tail = snapshot_text[marker_index:]
        return _normalize_summary_lines(tail, max_lines=5)

    return _normalize_summary_lines("\n".join(snapshot_text.splitlines()[-12:]), max_lines=5)


def _tokenize_for_novelty(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in {"the", "and", "for", "with", "sur", "dans", "les", "des", "une", "que"}
    }


def _overlap_ratio(previous: str, current: str) -> float:
    prev_tokens = _tokenize_for_novelty(previous)
    curr_tokens = _tokenize_for_novelty(current)
    if not prev_tokens or not curr_tokens:
        return 0.0
    common = prev_tokens.intersection(curr_tokens)
    return round(len(common) / max(len(prev_tokens), 1), 3)


def _load_latest_summary_payload() -> dict[str, Any] | None:
    if not PERPLEXITY_LATEST_SUMMARY_PATH.exists():
        return None
    try:
        raw = json.loads(PERPLEXITY_LATEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _load_latest_summary_text() -> str | None:
    payload = _load_latest_summary_payload()
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary_5_lines")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def _persist_latest_summary(
    *,
    summary_5_lines: str,
    requested_at_utc: str,
    model: str,
    overlap_ratio: float,
    diagnostics: list[str],
) -> tuple[Path, Path] | None:
    if not summary_5_lines.strip():
        return None
    payload = {
        "generated_at_utc": _utc_now().isoformat(),
        "requested_at_utc": requested_at_utc,
        "model": model,
        "summary_5_lines": summary_5_lines,
        "novelty_overlap_ratio": overlap_ratio,
        "quality_diagnostics": diagnostics,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    archive_path = PERPLEXITY_ARCHIVE_DIR / f"summary_{stamp}.json"
    _atomic_write_text(archive_path, content)
    _atomic_write_text(PERPLEXITY_LATEST_SUMMARY_PATH, content)
    return PERPLEXITY_LATEST_SUMMARY_PATH, archive_path


def build_perplexity_snapshot_prompt(previous_summary_5_lines: str | None = None) -> str:
    continuity_block = (
        f"Résumé précédent (pour continuité):\n{previous_summary_5_lines}\n\n"
        if previous_summary_5_lines
        else "Aucun résumé précédent disponible.\n\n"
    )
    return (
        "Rôle: analyste de desk actions US. Réponds UNIQUEMENT en langage naturel (pas de JSON).\n"
        "Objectif: snapshot marché ultra-frais orienté exécution, avec équilibre 50% continuité / 50% nouveauté.\n\n"
        f"{continuity_block}"
        "Contraintes éditoriales obligatoires:\n"
        "1) Section 'Continuité depuis run précédent': confirme ce qui reste valide vs invalide.\n"
        "2) Section 'Nouveautés fraîches (<6h)': événements et signaux très récents avec timestamp approximatif.\n"
        "3) Section 'Thèmes nouveaux': ouvre sur des angles non abordés précédemment (évite la redite).\n"
        "4) Mentionne explicitement les actifs/tickers impliqués quand possible.\n"
        "5) Termine impérativement avec le bloc exact:\n"
        f"{SUMMARY_START_MARKER}\n"
        "- ligne 1\n"
        "- ligne 2\n"
        "- ligne 3\n"
        "- ligne 4\n"
        "- ligne 5\n"
        f"{SUMMARY_END_MARKER}\n\n"
        "Qualité attendue:\n"
        "- Priorité à la fraîcheur et aux sources finance fiables.\n"
        "- Pas de contenu générique ni de recommandation d'achat/vente directe.\n"
        "- Ne pas répéter uniquement les mêmes thèmes; introduire de la nouveauté utile.\n"
    )


def perplexity_market_snapshot_raw() -> dict[str, Any]:
    settings = Settings.from_env(require_openrouter=False)
    if not settings.perplexity_api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY (or PPLX_API_KEY) in environment.")

    # Local import keeps the project importable even if dependency is temporarily absent.
    from perplexity import Perplexity

    client = Perplexity(api_key=settings.perplexity_api_key)
    requested_at_utc = _utc_now().isoformat()
    previous_summary_5_lines = _load_latest_summary_text()

    domain_denylist = [
        "-reddit.com",
        "-pinterest.com",
        "-quora.com",
        "-tiktok.com",
        "-instagram.com",
        "-facebook.com",
        "-wikipedia.org",
    ]
    completion = client.chat.completions.create(
        model=settings.perplexity_model,
        messages=[
            {"role": "system", "content": settings.perplexity_snapshot_system_prompt},
            {
                "role": "user",
                "content": build_perplexity_snapshot_prompt(previous_summary_5_lines),
            },
        ],
        search_recency_filter=settings.perplexity_snapshot_search_recency,
        search_domain_filter=domain_denylist,
        web_search_options={"search_context_size": settings.perplexity_snapshot_search_context_size},
        max_tokens=settings.perplexity_snapshot_max_tokens,
    )

    snapshot_text = _extract_message_content_text(completion.choices[0].message.content)
    if not snapshot_text:
        raise RuntimeError("Perplexity returned empty message content.")

    summary_5_lines = _extract_summary_5_lines(snapshot_text)
    diagnostics: list[str] = []
    if not summary_5_lines:
        diagnostics.append("missing_summary_5_lines")

    overlap = _overlap_ratio(previous_summary_5_lines or "", summary_5_lines or "")
    if previous_summary_5_lines and overlap >= 0.65:
        diagnostics.append("low_novelty")

    persisted_paths = _persist_latest_summary(
        summary_5_lines=summary_5_lines,
        requested_at_utc=requested_at_utc,
        model=settings.perplexity_model,
        overlap_ratio=overlap,
        diagnostics=diagnostics,
    )

    return {
        "provider": "perplexity",
        "requested_at_utc": requested_at_utc,
        "model": settings.perplexity_model,
        "search_recency_filter": settings.perplexity_snapshot_search_recency,
        "search_domain_filter": domain_denylist,
        "snapshot_text": snapshot_text,
        "summary_5_lines": summary_5_lines,
        "previous_summary_5_lines": previous_summary_5_lines,
        "novelty_overlap_ratio": overlap,
        "quality_diagnostics": diagnostics,
        "search_results": _extract_search_results(completion),
        "citations": _extract_citations(completion),
        "summary_memory_latest_path": str(persisted_paths[0]) if persisted_paths else None,
        "summary_memory_archive_path": str(persisted_paths[1]) if persisted_paths else None,
    }


def compact_perplexity_snapshot_for_prompt(payload: dict[str, Any], max_sources: int = 5) -> dict[str, Any]:
    search_results = payload.get("search_results") or []
    compact_sources: list[dict[str, Any]] = []
    for item in search_results[:max_sources]:
        if not isinstance(item, dict):
            continue
        compact_sources.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "date": item.get("date") or item.get("last_updated"),
                "source": item.get("source"),
            }
        )

    snapshot_text = str(payload.get("snapshot_text") or "")
    summary_5_lines = str(payload.get("summary_5_lines") or "")
    return {
        "provider": payload.get("provider"),
        "requested_at_utc": payload.get("requested_at_utc"),
        "model": payload.get("model"),
        "summary_5_lines": summary_5_lines,
        "novelty_overlap_ratio": payload.get("novelty_overlap_ratio"),
        "quality_diagnostics": payload.get("quality_diagnostics") or [],
        "snapshot_text_excerpt": snapshot_text[:2000],
        "top_sources": compact_sources,
    }
