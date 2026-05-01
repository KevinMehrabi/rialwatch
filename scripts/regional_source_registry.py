#!/usr/bin/env python3
"""Persistent regional signal source memory for RialWatch discovery loops."""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

REGISTRY_FILENAME = "regional_signal_source_registry.json"
SCHEMA_VERSION = 1
TELEGRAM_HANDLE_RE = re.compile(r"^[a-z0-9_]{5,}$")
SOURCE_KIND_RANK = {
    "unknown": 0,
    "aggregator": 1,
    "regional_market_channel": 2,
    "regional_fx_board": 2,
    "settlement_channel": 3,
    "exchange_shop": 4,
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_registry() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "sources": [],
        "summary": {},
    }


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return empty_registry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_registry()
    if not isinstance(payload, dict):
        return empty_registry()
    sources = payload.get("sources")
    if not isinstance(sources, list):
        payload["sources"] = []
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("generated_at", now_iso())
    payload.setdefault("summary", {})
    return payload


def write_registry(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_telegram_handle(raw: str) -> Optional[str]:
    token = str(raw or "").strip().lower()
    if not token:
        return None
    if "://" in token or token.startswith("t.me/") or token.startswith("telegram.me/"):
        if token.startswith("t.me/") or token.startswith("telegram.me/"):
            token = "https://" + token
        parsed = urllib.parse.urlparse(token)
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0].lower() == "s" and len(parts) > 1:
            token = parts[1]
        elif parts:
            token = parts[0]
    token = re.sub(r"[^a-z0-9_]", "", token)
    return token if TELEGRAM_HANDLE_RE.fullmatch(token) else None


def canonical_source_key(platform: str, handle_or_url: str, public_url: str = "") -> Optional[str]:
    platform_norm = str(platform or "").strip().lower()
    if platform_norm == "telegram":
        handle = normalize_telegram_handle(handle_or_url) or normalize_telegram_handle(public_url)
        return f"telegram:{handle}" if handle else None
    if platform_norm == "provider":
        slug = re.sub(r"[^a-z0-9_:-]", "_", str(handle_or_url or public_url).strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return f"provider:{slug}" if slug else None
    parsed = urllib.parse.urlparse(str(public_url or handle_or_url or "").strip())
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        normalized = urllib.parse.urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", "", ""))
        return f"website:{normalized}"
    return None


def split_tokens(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in re.split(r"[|,]", str(value)) if part.strip()}


def parse_int(value: Any) -> int:
    if value is None:
        return 0
    raw = str(value).strip()
    if not raw:
        return 0
    try:
        return int(float(raw))
    except ValueError:
        return 0


def parse_float(value: Any) -> float:
    if value is None:
        return 0.0
    raw = str(value).strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def has_success(update: Dict[str, Any]) -> bool:
    if parse_int(update.get("usable_record_count")) > 0:
        return True
    if parse_int(update.get("board_message_count")) > 0:
        return True
    if parse_int(update.get("quote_message_count")) > 0:
        return True
    return False


def merge_metric(record: Dict[str, Any], update: Dict[str, Any], field: str) -> None:
    value = parse_int(update.get(field))
    record[f"last_{field}"] = value
    record[f"best_{field}"] = max(parse_int(record.get(f"best_{field}")), value)


def summarize_registry(sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_country: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    active = 0
    for source in sources:
        country = str(source.get("country_guess") or "unknown")
        kind = str(source.get("source_kind") or "unknown")
        by_country[country] = by_country.get(country, 0) + 1
        by_kind[kind] = by_kind.get(kind, 0) + 1
        for family in split_tokens(source.get("signal_families")):
            by_family[family] = by_family.get(family, 0) + 1
        if source.get("last_success_at") or parse_int(source.get("best_usable_record_count")) > 0:
            active += 1
    return {
        "total_sources": len(sources),
        "active_sources": active,
        "sources_by_country": dict(sorted(by_country.items())),
        "sources_by_kind": dict(sorted(by_kind.items())),
        "sources_by_signal_family": dict(sorted(by_family.items())),
        "active_iran_sources": sum(
            1
            for source in sources
            if source.get("country_guess") == "Iran"
            and (source.get("last_success_at") or parse_int(source.get("best_usable_record_count")) > 0)
        ),
    }


def upsert_sources(registry: Dict[str, Any], updates: Iterable[Dict[str, Any]], observed_at: Optional[str] = None) -> Dict[str, Any]:
    observed = observed_at or now_iso()
    existing: Dict[str, Dict[str, Any]] = {
        str(source.get("key")): dict(source)
        for source in registry.get("sources", [])
        if isinstance(source, dict) and source.get("key")
    }

    for update in updates:
        platform = str(update.get("platform") or "telegram").strip().lower()
        handle_or_url = str(update.get("handle_or_url") or update.get("handle") or "").strip()
        public_url = str(update.get("public_url") or "").strip()
        key = str(update.get("key") or "") or (canonical_source_key(platform, handle_or_url, public_url) or "")
        if not key:
            continue

        record = existing.get(key, {"key": key, "first_seen_at": observed})
        record["platform"] = platform
        if platform == "telegram":
            handle = normalize_telegram_handle(handle_or_url) or normalize_telegram_handle(public_url)
            if not handle:
                continue
            record["handle_or_url"] = handle
            record["public_url"] = public_url or f"https://t.me/s/{handle}"
        else:
            record["handle_or_url"] = handle_or_url or public_url
            record["public_url"] = public_url or handle_or_url

        for field in ("title", "status", "latest_timestamp", "top_sample"):
            value = str(update.get(field) or "").strip()
            if value:
                record[field] = value

        source_kind = str(update.get("source_kind") or "").strip()
        if source_kind:
            current_kind = str(record.get("source_kind") or "unknown")
            if SOURCE_KIND_RANK.get(source_kind, 0) >= SOURCE_KIND_RANK.get(current_kind, 0):
                record["source_kind"] = source_kind
        record.setdefault("source_kind", "unknown")

        country = str(update.get("country_guess") or "").strip()
        if country and (not record.get("country_guess") or record.get("country_guess") == "unknown") and country != "unknown":
            record["country_guess"] = country
        record.setdefault("country_guess", "unknown")

        city = str(update.get("city_guess") or "").strip()
        if city and (not record.get("city_guess") or record.get("city_guess") == "unknown") and city != "unknown":
            record["city_guess"] = city
        record.setdefault("city_guess", "unknown")

        country_hints = split_tokens(record.get("country_hints"))
        if country and country != "unknown":
            country_hints.add(country)
        if country_hints:
            record["country_hints"] = sorted(country_hints)

        origins = split_tokens(record.get("origins")) | split_tokens(update.get("origins")) | split_tokens(update.get("discovery_origins"))
        if origins:
            record["origins"] = sorted(origins)

        localities = split_tokens(record.get("locality_hints")) | split_tokens(update.get("locality_hints"))
        if localities:
            record["locality_hints"] = sorted(localities)

        families = split_tokens(record.get("signal_families")) | split_tokens(update.get("signal_families"))
        if families:
            record["signal_families"] = sorted(families)

        for field in ("quote_message_count", "board_message_count", "usable_record_count", "buy_sell_pair_count"):
            merge_metric(record, update, field)

        parseability = parse_float(update.get("parseability_score"))
        if parseability:
            record["last_parseability_score"] = parseability
            record["best_parseability_score"] = max(parse_float(record.get("best_parseability_score")), parseability)

        record["last_seen_at"] = observed
        if has_success(update):
            record["last_success_at"] = observed
        existing[key] = record

    sources = sorted(existing.values(), key=lambda source: str(source.get("key")))
    registry["schema_version"] = SCHEMA_VERSION
    registry["generated_at"] = observed
    registry["sources"] = sources
    registry["summary"] = summarize_registry(sources)
    return registry


def registry_sources(
    registry: Dict[str, Any],
    *,
    platform: Optional[str] = None,
    signal_families: Optional[Set[str]] = None,
    active_only: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    platform_norm = platform.lower() if platform else None
    for source in registry.get("sources", []):
        if not isinstance(source, dict):
            continue
        if platform_norm and str(source.get("platform", "")).lower() != platform_norm:
            continue
        if signal_families and not (split_tokens(source.get("signal_families")) & signal_families):
            continue
        if active_only and not source.get("last_success_at") and parse_int(source.get("best_usable_record_count")) <= 0:
            continue
        out.append(source)
    return out
