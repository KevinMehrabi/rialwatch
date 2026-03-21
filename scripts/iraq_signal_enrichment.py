#!/usr/bin/env python3
"""Enrich Iraq (Sulaymaniyah) locality signals from existing regional FX board channels.

Diagnostics-only research utility. This script does not change benchmark methodology.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.exchange_shop_baskets import benchmark_rate, record_weight
from scripts.telegram_quote_pilot_ingestion import (
    PilotChannel,
    extract_message_rows,
    fetch_url as fetch_public_url,
    normalize_public_url,
)

NUM_TOKEN = r"\d{2,3}(?:[\s,٬،]\d{3})+|\d{4,8}"
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{4,8})(?!\d)")
PAIR_RE = re.compile(
    rf"(?<!\d)({NUM_TOKEN})(?!\d)\s*[/\\|\-]\s*({NUM_TOKEN})(?!\d)",
    re.IGNORECASE,
)

IRAQ_ALIASES = (
    "سلیمانیه",
    "سليمانية",
    "سلیمانیە",
    "sulaimaniya",
    "sulaymaniyah",
    "sulaymaniya",
    "iraq",
    "عراق",
)
TEHRAN_ALIASES = ("تهران", "tehran")
LOCALITY_CANONICAL = "IRAQ_SULAYMANIYAH"

SOURCE_TYPE_MULTIPLIER = {
    "regional_fx_board": 1.0,
    "regional_market_channel": 0.88,
    "exchange_shop": 0.72,
    "aggregator": 0.55,
    "unknown": 0.45,
}


@dataclass
class CandidateSource:
    handle: str
    title: str
    public_url: str
    source_type: str
    quote_density_score: float


@dataclass
class IraqSignalRecord:
    source: str
    handle: str
    title: str
    source_type: str
    message_text_sample: str
    inferred_value: float
    normalized_irr_value: float
    extraction_type: str
    freshness: str
    parseability_score: int
    timestamp_iso: str
    inferred_unit: str
    tehran_reference: str
    delta_value: str
    quote_density_score: float


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def normalize_text(text: str) -> str:
    out = translit_digits(text or "")
    out = out.replace("\u200c", " ")
    out = out.replace("،", ",").replace("٬", ",")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def clip_text(text: str, limit: int = 280) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(str(value).strip())))
    except Exception:
        return default


def parse_number(token: str) -> Optional[int]:
    cleaned = re.sub(r"[^0-9]", "", translit_digits(token or ""))
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def parse_signed_number(token: str) -> Optional[int]:
    raw = translit_digits(token or "").replace(",", "").replace(" ", "")
    if not raw:
        return None
    sign = -1 if raw.startswith("-") else 1
    raw = raw[1:] if raw[:1] in "+-" else raw
    if not raw.isdigit():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return sign * value


def detect_unit(text: str, numbers: Sequence[int]) -> str:
    lowered = normalize_text(text).lower()
    if "تومان" in lowered or "toman" in lowered:
        return "toman"
    if "ریال" in lowered or "rial" in lowered or "irr" in lowered:
        return "rial"
    if numbers:
        med = statistics.median(numbers)
        return "toman" if med < 400_000 else "rial"
    return "unknown"


def to_rial(value: float, unit: str) -> float:
    if unit == "toman":
        return value * 10.0
    return value


def freshness_from_timestamp(timestamp_iso: str) -> str:
    raw = str(timestamp_iso or "").strip()
    if not raw:
        return "old"
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return "old"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    age_hours = (dt.datetime.now(dt.timezone.utc) - parsed.astimezone(dt.timezone.utc)).total_seconds() / 3600.0
    if age_hours <= 30:
        return "fresh"
    if age_hours <= 96:
        return "recent"
    if age_hours <= 720:
        return "stale"
    return "old"


def alias_regex(aliases: Sequence[str]) -> str:
    escaped = sorted({re.escape(a.lower()) for a in aliases if a}, key=len, reverse=True)
    return r"(?:%s)" % "|".join(escaped) if escaped else r"(?:)"


def detect_locality_mentions(text: str) -> int:
    lowered = normalize_text(text).lower()
    groups = [IRAQ_ALIASES, TEHRAN_ALIASES, ("هرات", "herat"), ("دبی", "dubai"), ("استانبول", "istanbul")]
    return sum(1 for aliases in groups if any(alias.lower() in lowered for alias in aliases))


def extract_pair_for_alias(text: str, aliases: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    lowered = normalize_text(text).lower()
    alias_pat = alias_regex(aliases)
    forward = re.compile(
        rf"{alias_pat}[^\d\n]{{0,32}}({NUM_TOKEN})\s*[/\\|\-]\s*({NUM_TOKEN})",
        re.IGNORECASE,
    )
    reverse = re.compile(
        rf"({NUM_TOKEN})\s*[/\\|\-]\s*({NUM_TOKEN})[^\d\n]{{0,32}}{alias_pat}",
        re.IGNORECASE,
    )
    for patt in (forward, reverse):
        match = patt.search(lowered)
        if not match:
            continue
        first = parse_number(match.group(1))
        second = parse_number(match.group(2))
        if first is None or second is None:
            continue
        if first < 10_000 or second < 10_000:
            continue
        mid = (first + second) / 2.0
        return float(first), float(second), float(mid)
    return None


def extract_single_for_alias(text: str, aliases: Sequence[str]) -> Optional[float]:
    lowered = normalize_text(text).lower()
    alias_pat = alias_regex(aliases)
    patterns = (
        re.compile(rf"{alias_pat}[^\d+\-\n]{{0,28}}({NUM_TOKEN})", re.IGNORECASE),
        re.compile(rf"{alias_pat}[^\d+\-]{{0,14}}\n\s*({NUM_TOKEN})", re.IGNORECASE),
    )
    for patt in patterns:
        match = patt.search(lowered)
        if not match:
            continue
        value = parse_number(match.group(1))
        if value is None:
            continue
        if value < 10_000:
            continue
        return float(value)
    return None


def extract_delta_for_alias(text: str, aliases: Sequence[str]) -> Optional[int]:
    lowered = normalize_text(text).lower()
    alias_pat = alias_regex(aliases)
    patterns = (
        re.compile(rf"{alias_pat}[^0-9+\-\n]{{0,24}}([+-]\s*\d{{1,6}}(?:[,\s]\d{{3}})?)", re.IGNORECASE),
        re.compile(rf"([+-]\s*\d{{1,6}}(?:[,\s]\d{{3}})?)[^0-9\n]{{0,12}}{alias_pat}", re.IGNORECASE),
    )
    for patt in patterns:
        match = patt.search(lowered)
        if not match:
            continue
        delta = parse_signed_number(match.group(1))
        if delta is None:
            continue
        if abs(delta) < 50:
            continue
        return delta
    return None


def extract_tehran_reference(text: str) -> Optional[float]:
    pair = extract_pair_for_alias(text, TEHRAN_ALIASES)
    if pair is not None:
        return pair[2]
    return extract_single_for_alias(text, TEHRAN_ALIASES)


def parse_iraq_signal(
    *,
    text: str,
    timestamp_iso: str,
    source_type: str,
    benchmark_value: float,
    quote_density_score: float,
) -> Optional[Tuple[float, float, str, str, int, str, str]]:
    normalized = normalize_text(text)
    lowered = normalized.lower()
    if not any(alias.lower() in lowered for alias in IRAQ_ALIASES):
        return None

    all_numbers = [parse_number(match.group(0)) for match in NUMBER_RE.finditer(lowered)]
    all_numbers = [n for n in all_numbers if n is not None]
    unit = detect_unit(lowered, all_numbers)

    raw_value: Optional[float] = None
    extraction_type = ""
    tehran_reference = ""
    delta_value = ""

    pair = extract_pair_for_alias(lowered, IRAQ_ALIASES)
    if pair is not None:
        raw_value = pair[2]
        extraction_type = "pair"
    else:
        single = extract_single_for_alias(lowered, IRAQ_ALIASES)
        if single is not None:
            raw_value = single
            extraction_type = "single"
        else:
            delta = extract_delta_for_alias(lowered, IRAQ_ALIASES)
            tehran_raw = extract_tehran_reference(lowered)
            if delta is not None and tehran_raw is not None:
                raw_value = tehran_raw + float(delta)
                extraction_type = "relative"
                tehran_reference = f"{tehran_raw:.2f}"
                delta_value = str(delta)

    if raw_value is None or raw_value <= 0:
        return None

    normalized_value = to_rial(raw_value, unit)
    min_rate = benchmark_value * 0.45 if benchmark_value > 0 else 500_000.0
    max_rate = benchmark_value * 1.80 if benchmark_value > 0 else 2_500_000.0
    if normalized_value < min_rate or normalized_value > max_rate:
        return None

    base = {"pair": 86, "single": 74, "relative": 68}.get(extraction_type, 60)
    locality_count = detect_locality_mentions(lowered)
    if any(word in lowered for word in ("خرید", "فروش", "buy", "sell")):
        base += 6
    base += min(8, max(0, locality_count - 1) * 2)
    if source_type == "regional_fx_board":
        base += 5
    elif source_type == "aggregator":
        base -= 8
    parseability = max(20, min(100, base))

    freshness = freshness_from_timestamp(timestamp_iso)
    return raw_value, normalized_value, extraction_type, freshness, parseability, tehran_reference, delta_value


def source_category(source_type: str) -> str:
    if source_type in {"regional_fx_board", "regional_market_channel"}:
        return "regional_market_channel"
    if source_type == "exchange_shop":
        return "direct_shop"
    if source_type == "aggregator":
        return "aggregator"
    return "unknown"


def pseudo_weight(
    *,
    parseability_score: int,
    freshness: str,
    source_type: str,
    quote_density_score: float,
) -> float:
    freshness_score = {"fresh": 90.0, "recent": 70.0, "stale": 45.0, "old": 28.0}.get(freshness, 28.0)
    directness = 70.0 if source_type in {"regional_fx_board", "regional_market_channel"} else 58.0
    structure = float(parseability_score)
    source_cat = source_category(source_type)
    pseudo = type(
        "Pseudo",
        (),
        {
            "overall_quality": float(parseability_score),
            "freshness_score": freshness_score,
            "structure_score": structure,
            "directness_score": directness,
            "channel_readiness_score": float(quote_density_score),
            "source_category": source_cat,
        },
    )
    return record_weight(pseudo) * SOURCE_TYPE_MULTIPLIER.get(source_type, 0.45)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    denom = sum(weights)
    if denom <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / denom


def dispersion_label(values: Sequence[float]) -> str:
    if len(values) <= 1:
        return "low"
    mean_val = statistics.mean(values)
    if mean_val <= 0:
        return "unknown"
    cv = statistics.pstdev(values) / mean_val
    if cv <= 0.03:
        return "low"
    if cv <= 0.08:
        return "medium"
    return "high"


def overall_freshness(records: Sequence[IraqSignalRecord]) -> str:
    labels = [record.freshness for record in records]
    if "fresh" in labels:
        return "fresh"
    if "recent" in labels:
        return "recent"
    if "stale" in labels:
        return "stale"
    return "old"


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def channel_type_to_source_type(channel_type: str) -> str:
    lowered = str(channel_type or "").strip().lower()
    if lowered in {"dealer_network_channel", "market_price_channel", "regional_market_channel"}:
        return "regional_market_channel"
    if lowered in {"individual_exchange_shop", "exchange_shop"}:
        return "exchange_shop"
    if lowered == "aggregator":
        return "aggregator"
    return "unknown"


def seed_candidates_from_quote_samples(
    quote_samples_path: Path,
    channel_rows: Sequence[Dict[str, str]],
    existing_handles: Sequence[str],
) -> List[CandidateSource]:
    if not quote_samples_path.exists():
        return []
    try:
        payload = json.loads(quote_samples_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    channel_lookup = {
        str(row.get("handle", "")).strip().lower(): row
        for row in channel_rows
        if str(row.get("handle", "")).strip()
    }
    existing = {handle.strip().lower() for handle in existing_handles if handle}
    iraq_tokens = tuple(alias.lower() for alias in IRAQ_ALIASES)
    selected: Dict[str, CandidateSource] = {}

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        handle = str(entry.get("handle", "")).strip().lower()
        if not handle:
            continue
        records = entry.get("quote_message_records", [])
        if not isinstance(records, list) or not records:
            continue

        iraq_hint_count = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            text = normalize_text(str(rec.get("message_text_sample", ""))).lower()
            cities = [str(city).strip().lower() for city in (rec.get("city_mentions") or [])]
            if any(token in text for token in iraq_tokens) or any(city in {"sulaymaniyah", "iraq"} for city in cities):
                iraq_hint_count += 1

        if iraq_hint_count < 2 and handle not in existing:
            continue

        row = channel_lookup.get(handle, {})
        channel_type = str(entry.get("channel_type_guess", "")).strip() or str(row.get("channel_type_guess", "")).strip()
        source_type = channel_type_to_source_type(channel_type)
        title = str(entry.get("title", "")).strip() or str(row.get("title", "")).strip() or handle
        public_url = normalize_public_url(
            handle,
            str(entry.get("public_url", "")).strip() or str(row.get("public_url", "")).strip() or f"https://t.me/s/{handle}",
        )
        density_floor = parse_float(row.get("parseable_score"), 40.0)
        quote_density = min(100.0, max(density_floor, 45.0 + (iraq_hint_count * 7.0)))
        selected[handle] = CandidateSource(
            handle=handle,
            title=title,
            public_url=public_url,
            source_type=source_type,
            quote_density_score=quote_density,
        )

    return sorted(selected.values(), key=lambda item: item.handle)


def merge_candidates(primary: Sequence[CandidateSource], extra: Sequence[CandidateSource]) -> List[CandidateSource]:
    merged: Dict[str, CandidateSource] = {item.handle: item for item in primary}
    for item in extra:
        incumbent = merged.get(item.handle)
        if incumbent is None:
            merged[item.handle] = item
            continue
        if incumbent.source_type in {"unknown", "aggregator"} and item.source_type not in {"unknown", "aggregator"}:
            incumbent.source_type = item.source_type
        if item.quote_density_score > incumbent.quote_density_score:
            incumbent.quote_density_score = item.quote_density_score
        if not incumbent.title or incumbent.title == incumbent.handle:
            incumbent.title = item.title
        if incumbent.public_url.endswith(f"/{incumbent.handle}") and item.public_url and item.public_url != incumbent.public_url:
            incumbent.public_url = item.public_url
    return sorted(merged.values(), key=lambda item: item.handle)


def select_candidates(candidate_rows: Sequence[Dict[str, str]], existing_iraq_handles: Sequence[str]) -> List[CandidateSource]:
    existing = {handle.strip().lower() for handle in existing_iraq_handles if handle}
    selected: Dict[str, CandidateSource] = {}
    iraq_tokens = tuple(alias.lower() for alias in IRAQ_ALIASES)

    for row in candidate_rows:
        handle = str(row.get("handle", "")).strip().lower()
        if not handle:
            continue
        title = str(row.get("title", "")).strip() or handle
        source_type = str(row.get("source_type", "")).strip() or "unknown"
        locality_mentions = str(row.get("locality_mentions", "")).strip()
        top_sample = str(row.get("top_sample", "")).strip()
        text = normalize_text(" ".join([handle, title, locality_mentions, top_sample])).lower()
        board_count = parse_int(row.get("board_message_count"), 0)
        locality_count = parse_int(row.get("localities_detected_count"), 0)
        has_iraq_hint = any(token in text for token in iraq_tokens)
        useful_channel = source_type in {"regional_fx_board", "regional_market_channel"} and (board_count > 0 or locality_count >= 2)

        if not (has_iraq_hint or useful_channel or handle in existing):
            continue

        quote_density = parse_float(row.get("quote_density_score"), 0.0)
        public_url = normalize_public_url(handle, str(row.get("public_url", "")).strip())
        selected[handle] = CandidateSource(
            handle=handle,
            title=title,
            public_url=public_url,
            source_type=source_type,
            quote_density_score=quote_density,
        )

    for handle in sorted(existing):
        if handle in selected:
            continue
        selected[handle] = CandidateSource(
            handle=handle,
            title=handle,
            public_url=f"https://t.me/s/{handle}",
            source_type="regional_market_channel",
            quote_density_score=70.0,
        )

    return sorted(selected.values(), key=lambda row: row.handle)


def enrich_from_existing_records(existing_rows: Sequence[Dict[str, str]]) -> List[IraqSignalRecord]:
    out: List[IraqSignalRecord] = []
    for row in existing_rows:
        if str(row.get("locality_name", "")).strip() != "Sulaymaniyah":
            continue
        handle = str(row.get("handle", "")).strip()
        if not handle:
            continue
        value = parse_float(row.get("normalized_rate_irr"), 0.0)
        if value <= 0:
            continue
        quote_basis = str(row.get("quote_basis", "")).strip() or "single"
        if quote_basis == "midpoint":
            extraction_type = "pair"
        elif quote_basis in {"single_value", "single"}:
            extraction_type = "single"
        elif quote_basis == "relative":
            extraction_type = "relative"
        else:
            extraction_type = "single"
        out.append(
            IraqSignalRecord(
                source=handle,
                handle=handle,
                title=str(row.get("title", "")).strip() or handle,
                source_type=str(row.get("source_type", "")).strip() or "regional_market_channel",
                message_text_sample=str(row.get("message_text_sample", "")).strip(),
                inferred_value=parse_float(row.get("sulaymaniyah_quote"), value),
                normalized_irr_value=value,
                extraction_type=extraction_type,
                freshness=str(row.get("freshness_indicator", "")).strip() or "old",
                parseability_score=parse_int(row.get("parseability_score"), 55),
                timestamp_iso=str(row.get("timestamp_iso", "")).strip(),
                inferred_unit=str(row.get("inferred_unit", "")).strip() or "unknown",
                tehran_reference="",
                delta_value="",
                quote_density_score=parse_float(row.get("quote_density_score"), 0.0),
            )
        )
    return out


def fetch_enriched_iraq_records(
    candidates: Sequence[CandidateSource],
    benchmark_value: float,
    timeout: int,
    sleep_seconds: float,
) -> Tuple[List[IraqSignalRecord], List[str]]:
    enriched: List[IraqSignalRecord] = []
    errors: List[str] = []

    for idx, candidate in enumerate(candidates):
        body, status, err = fetch_public_url(candidate.public_url, timeout=timeout)
        if body is None or (status is not None and status >= 400):
            errors.append(f"{candidate.handle}:{err or f'http_{status}'}")
            continue

        channel = PilotChannel(
            handle=candidate.handle,
            title=candidate.title,
            source_priority="iraq_enrichment",
            origin_priority="iraq_enrichment",
            priority_score=0.0,
            channel_type_guess=candidate.source_type,
            likely_individual_shop=False,
            public_url=candidate.public_url,
            selection_note="iraq_signal_enrichment",
        )
        messages, _total_seen = extract_message_rows(body, channel)
        for message in messages:
            parsed = parse_iraq_signal(
                text=message.message_text,
                timestamp_iso=message.timestamp_iso,
                source_type=candidate.source_type,
                benchmark_value=benchmark_value,
                quote_density_score=candidate.quote_density_score,
            )
            if parsed is None:
                continue
            inferred_value, normalized_value, extraction_type, freshness, parseability, tehran_ref, delta_value = parsed
            enriched.append(
                IraqSignalRecord(
                    source=candidate.handle,
                    handle=candidate.handle,
                    title=candidate.title,
                    source_type=candidate.source_type,
                    message_text_sample=clip_text(message.message_text),
                    inferred_value=round(inferred_value, 2),
                    normalized_irr_value=round(normalized_value, 2),
                    extraction_type=extraction_type,
                    freshness=freshness,
                    parseability_score=parseability,
                    timestamp_iso=message.timestamp_iso,
                    inferred_unit=detect_unit(message.message_text, [n for n in [parse_number(m.group(0)) for m in NUMBER_RE.finditer(normalize_text(message.message_text))] if n is not None]),
                    tehran_reference=tehran_ref,
                    delta_value=delta_value,
                    quote_density_score=candidate.quote_density_score,
                )
            )

        if sleep_seconds > 0 and idx < len(candidates) - 1:
            time.sleep(sleep_seconds)

    return enriched, errors


def deduplicate_records(records: Sequence[IraqSignalRecord]) -> List[IraqSignalRecord]:
    best_by_key: Dict[Tuple[str, str], IraqSignalRecord] = {}
    extraction_rank = {"pair": 3, "single": 2, "relative": 1}
    for record in records:
        ts_prefix = str(record.timestamp_iso or "").strip()[:16]
        key = (record.handle, ts_prefix)
        incumbent = best_by_key.get(key)
        if incumbent is None:
            best_by_key[key] = record
            continue
        if record.parseability_score > incumbent.parseability_score:
            best_by_key[key] = record
            continue
        if record.parseability_score == incumbent.parseability_score:
            if extraction_rank.get(record.extraction_type, 0) > extraction_rank.get(incumbent.extraction_type, 0):
                best_by_key[key] = record
    return sorted(
        best_by_key.values(),
        key=lambda row: (
            row.handle,
            row.timestamp_iso,
            row.extraction_type,
            int(round(row.normalized_irr_value)),
        ),
    )


def summarize_iraq(records: Sequence[IraqSignalRecord], benchmark_value: float) -> Dict[str, Any]:
    if not records:
        return {
            "locality_name": "Iraq",
            "signal_type_used": None,
            "usable_record_count": 0,
            "contributing_source_count": 0,
            "median_rate": None,
            "weighted_rate": None,
            "spread_vs_benchmark_pct": None,
            "freshness_status": "old",
            "dispersion_level": "unknown",
            "basket_confidence": 0.0,
            "recommended_display_state": "hide",
            "suppression_reason": "no_usable_records",
        }

    values = [record.normalized_irr_value for record in records]
    weights: List[float] = []
    source_weight: Dict[str, float] = {}
    for record in records:
        weight = pseudo_weight(
            parseability_score=record.parseability_score,
            freshness=record.freshness,
            source_type=record.source_type,
            quote_density_score=record.quote_density_score,
        )
        weights.append(weight)
        source_weight[record.source_type] = source_weight.get(record.source_type, 0.0) + weight

    median_rate = statistics.median(values)
    weighted_rate = weighted_mean(values, weights) or median_rate
    fresh = overall_freshness(records)
    dispersion = dispersion_label(values)
    contributing_sources = len({record.handle for record in records})
    avg_parse = statistics.mean(record.parseability_score for record in records)
    confidence = (
        min(28.0, len(records) * 4.0)
        + min(18.0, contributing_sources * 6.0)
        + min(20.0, avg_parse / 4.0)
        + (16.0 if fresh == "fresh" else 10.0 if fresh == "recent" else 3.0 if fresh == "stale" else 0.0)
        + (12.0 if dispersion == "low" else 6.0 if dispersion == "medium" else 0.0)
    )
    confidence = round(max(0.0, min(100.0, confidence)), 2)
    spread = ((weighted_rate - benchmark_value) / benchmark_value) * 100.0 if benchmark_value > 0 else None
    top_type = max(source_weight, key=source_weight.get) if source_weight else "unknown"

    if fresh in {"fresh", "recent"} and len(records) >= 4 and contributing_sources >= 2 and confidence >= 58:
        display = "publish"
        reason = ""
    elif len(records) >= 2 and confidence >= 36:
        display = "monitor"
        reason = "needs_more_fresh_sources" if fresh in {"stale", "old"} else "limited_coverage"
    else:
        display = "hide"
        reason = "insufficient_signal"

    return {
        "locality_name": "Iraq",
        "signal_type_used": top_type,
        "usable_record_count": len(records),
        "contributing_source_count": contributing_sources,
        "median_rate": round(median_rate, 2),
        "weighted_rate": round(weighted_rate, 2),
        "spread_vs_benchmark_pct": round(spread, 4) if spread is not None else None,
        "freshness_status": fresh,
        "dispersion_level": dispersion,
        "basket_confidence": confidence,
        "recommended_display_state": display,
        "suppression_reason": reason,
    }


def write_records_csv(path: Path, rows: Sequence[IraqSignalRecord]) -> None:
    fieldnames = [
        "source",
        "handle",
        "title",
        "source_type",
        "message_text_sample",
        "inferred_value",
        "normalized_irr_value",
        "extraction_type",
        "freshness",
        "parseability_score",
        "timestamp_iso",
        "inferred_unit",
        "tehran_reference",
        "delta_value",
        "quote_density_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.handle, r.timestamp_iso, -r.parseability_score)):
            writer.writerow(
                {
                    "source": row.source,
                    "handle": row.handle,
                    "title": row.title,
                    "source_type": row.source_type,
                    "message_text_sample": row.message_text_sample,
                    "inferred_value": f"{row.inferred_value:.2f}",
                    "normalized_irr_value": f"{row.normalized_irr_value:.2f}",
                    "extraction_type": row.extraction_type,
                    "freshness": row.freshness,
                    "parseability_score": row.parseability_score,
                    "timestamp_iso": row.timestamp_iso,
                    "inferred_unit": row.inferred_unit,
                    "tehran_reference": row.tehran_reference,
                    "delta_value": row.delta_value,
                    "quote_density_score": f"{row.quote_density_score:.2f}",
                }
            )


def update_basket_review(path: Path, iraq_row: Dict[str, Any], benchmark_value: float) -> str:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    localities = payload.get("localities")
    if not isinstance(localities, list):
        localities = []

    old_state = "hide"
    replaced = False
    for idx, row in enumerate(localities):
        if not isinstance(row, dict):
            continue
        if str(row.get("locality_name", "")).strip() == "Iraq":
            old_state = str(row.get("recommended_display_state", "hide"))
            localities[idx] = iraq_row
            replaced = True
            break
    if not replaced:
        localities.append(iraq_row)

    payload["generated_at"] = now_iso()
    payload["diagnostics_only"] = True
    payload["benchmark_weighted_rate"] = benchmark_value
    payload["localities"] = localities
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return old_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich Iraq Sulaymaniyah signals from regional FX board candidates.")
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=Path("survey_outputs/regional_fx_board_records.csv"),
    )
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path("survey_outputs/regional_fx_board_candidates.csv"),
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path("site/api/benchmark.json"),
    )
    parser.add_argument(
        "--basket-review-json",
        type=Path,
        default=Path("site/api/regional_fx_board_basket_review.json"),
    )
    parser.add_argument(
        "--out-records-csv",
        type=Path,
        default=Path("survey_outputs/iraq_signal_enriched_records.csv"),
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("survey_outputs/iraq_signal_summary.json"),
    )
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def main() -> int:
    args = parse_args()

    records_csv = resolve_path(args.records_csv)
    candidates_csv = resolve_path(args.candidates_csv)
    benchmark_json = resolve_path(args.benchmark_json)
    basket_review_json = resolve_path(args.basket_review_json)
    out_records_csv = resolve_path(args.out_records_csv)
    out_summary_json = resolve_path(args.out_summary_json)

    for parent in (out_records_csv.parent, out_summary_json.parent, basket_review_json.parent):
        parent.mkdir(parents=True, exist_ok=True)

    if not records_csv.exists():
        raise FileNotFoundError(f"Missing input records CSV: {records_csv}")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Missing input candidates CSV: {candidates_csv}")
    if not benchmark_json.exists():
        raise FileNotFoundError(f"Missing benchmark JSON: {benchmark_json}")

    benchmark_value = benchmark_rate(benchmark_json.parent)
    existing_rows = load_csv(records_csv)
    candidate_rows = load_csv(candidates_csv)
    survey_dir = records_csv.parent
    channel_rows: List[Dict[str, str]] = []
    channel_survey_csv = survey_dir / "channel_survey.csv"
    if channel_survey_csv.exists():
        channel_rows = load_csv(channel_survey_csv)

    existing_enriched = enrich_from_existing_records(existing_rows)
    existing_iraq_handles = sorted({row.handle for row in existing_enriched})
    candidates = select_candidates(candidate_rows, existing_iraq_handles)
    sample_seeded_candidates = seed_candidates_from_quote_samples(
        quote_samples_path=survey_dir / "quote_message_samples.json",
        channel_rows=channel_rows,
        existing_handles=existing_iraq_handles,
    )
    candidates = merge_candidates(candidates, sample_seeded_candidates)

    fetched_records, fetch_errors = fetch_enriched_iraq_records(
        candidates=candidates,
        benchmark_value=benchmark_value,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )

    merged_records = deduplicate_records(existing_enriched + fetched_records)
    write_records_csv(out_records_csv, merged_records)

    iraq_row = summarize_iraq(merged_records, benchmark_value=benchmark_value)
    previous_state = update_basket_review(basket_review_json, iraq_row, benchmark_value)

    summary = {
        "generated_at": now_iso(),
        "benchmark_weighted_rate": benchmark_value,
        "total_iraq_signals_extracted": len(merged_records),
        "sources_contributing": sorted({row.handle for row in merged_records}),
        "number_of_sources_contributing": len({row.handle for row in merged_records}),
        "new_usable_record_count": iraq_row["usable_record_count"],
        "median_rate": iraq_row["median_rate"],
        "weighted_rate": iraq_row["weighted_rate"],
        "dispersion_level": iraq_row["dispersion_level"],
        "basket_confidence": iraq_row["basket_confidence"],
        "recommended_display_state": iraq_row["recommended_display_state"],
        "previous_display_state": previous_state,
        "moved_from_monitor_to_publish": previous_state == "monitor" and iraq_row["recommended_display_state"] == "publish",
        "extraction_type_counts": {
            "pair": sum(1 for row in merged_records if row.extraction_type == "pair"),
            "single": sum(1 for row in merged_records if row.extraction_type == "single"),
            "relative": sum(1 for row in merged_records if row.extraction_type == "relative"),
        },
        "candidate_channels_considered": [source.handle for source in candidates],
        "fetch_errors": fetch_errors,
        "locality_canonical": LOCALITY_CANONICAL,
    }
    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"total Iraq signals extracted: {summary['total_iraq_signals_extracted']}")
    print(f"number of sources contributing: {summary['number_of_sources_contributing']}")
    print(f"new usable record count: {summary['new_usable_record_count']}")
    print(
        "whether Iraq moves from monitor to publish: "
        f"{'yes' if summary['moved_from_monitor_to_publish'] else 'no'}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
