#!/usr/bin/env python3
"""Build diagnostics-only exchange shop baskets for the RialWatch homepage.

This script keeps exchange shop and dealer-style locality baskets separate from
the core benchmark. It reuses existing research-mode Telegram quote parsing,
adds newly ranked P1 candidates, and writes site-facing diagnostics artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.exchange_shop_candidate_ranking import TARGET_LOCALITIES, derive_locality_bucket
from scripts.telegram_quote_pilot_ingestion import (
    PilotChannel,
    apply_in_channel_dedup,
    extract_message_rows,
    fetch_url,
    normalize_public_url,
    parse_quote_records_from_message,
)

SOURCE_CATEGORY_BASE_WEIGHT = {
    "direct_shop": 1.00,
    "settlement_exchange": 0.88,
    "regional_market_channel": 0.82,
    "market_channel": 0.68,
    "aggregator": 0.55,
    "unknown": 0.50,
}

LOCALITY_ORDER = ("Iran", "Turkey", "UAE", "Iraq", "Afghanistan", "UK", "Germany", "unknown")
VALID_STATUSES = {"ready_for_research_ingestion", "monitor_only"}
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")
SIMPLE_NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[,٬،]\d{3})+|\d{5,8})(?!\d)")
USD_ALIASES = ("دلار آمریکا", "دلار", "USD", "usd")


@dataclass
class BasketRecord:
    handle: str
    title: str
    locality: str
    source_category: str
    source_priority: str
    likely_individual_shop: bool
    channel_type_guess: str
    normalized_rate_rial: float
    quote_basis: str
    overall_quality: float
    freshness_score: float
    structure_score: float
    directness_score: float
    timestamp_iso: str
    dedup_keep: bool
    duplication_flag: str
    from_new_p1: bool
    channel_readiness_score: float


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return default


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_locality_internal_status(survey_dir: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    uae_summary_path = survey_dir / "uae_basket_review_summary.json"
    if uae_summary_path.exists():
        try:
            payload = json.loads(uae_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        support = str(payload.get("uae_card_support", "")).strip()
        if support == "still_hidden":
            statuses["UAE"] = "watchlist_only"
        elif support == "monitor_card":
            statuses["UAE"] = "monitor_card"
        elif support == "full_card":
            statuses["UAE"] = "active"
    return statuses


def build_lookup(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            lookup[value] = row
    return lookup


def rate_from_quote_row(row: Dict[str, str]) -> Tuple[Optional[float], str]:
    midpoint = safe_float(row.get("midpoint_rial"), default=0.0)
    if midpoint > 0:
        return midpoint, "midpoint"
    sell = safe_float(row.get("sell_quote_rial"), default=0.0)
    if sell > 0:
        return sell, "sell"
    buy = safe_float(row.get("buy_quote_rial"), default=0.0)
    if buy > 0:
        return buy, "buy"
    midpoint_native = safe_float(row.get("midpoint"), default=0.0)
    if midpoint_native > 0:
        unit = str(row.get("value_unit_guess") or "unknown").strip().lower()
        factor = 10.0 if unit == "toman" else 1.0
        return midpoint_native * factor, "inferred"
    return None, "unusable"


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def parse_int_token(token: str) -> Optional[int]:
    cleaned = re.sub(r"[^0-9]", "", translit_digits(token or ""))
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def message_unit_guess(text: str, fallback: str = "unknown") -> str:
    lowered = str(text or "").lower()
    if "تومان" in lowered or "toman" in lowered or "tmn" in lowered:
        return "toman"
    if "ریال" in lowered or "rial" in lowered or "irr" in lowered:
        return "rial"
    return fallback


def fallback_usd_rate_from_text(text: str, unit_guess: str, min_rate: float, max_rate: float) -> Tuple[Optional[float], str]:
    normalized_text = translit_digits(str(text or ""))
    actual_unit = message_unit_guess(normalized_text, fallback=unit_guess)
    factor = 10.0 if actual_unit == "toman" else 1.0
    for alias in USD_ALIASES:
        alias_pattern = re.compile(rf"(?is){re.escape(alias)}")
        for match in alias_pattern.finditer(normalized_text):
            snippet = normalized_text[match.end() : match.end() + 80]
            number_tokens = [m.group(0) for m in SIMPLE_NUMBER_RE.finditer(snippet)][:2]
            first = parse_int_token(number_tokens[0]) if len(number_tokens) >= 1 else None
            second = parse_int_token(number_tokens[1]) if len(number_tokens) >= 2 else None
            if first and second:
                first_rial = first * factor
                second_rial = second * factor
                midpoint = (first_rial + second_rial) / 2.0
                spread_ratio = abs(first_rial - second_rial) / max(first_rial, second_rial)
                if min_rate <= midpoint <= max_rate and spread_ratio <= 0.18:
                    return midpoint, "midpoint"
            if first:
                first_rial = first * factor
                if min_rate <= first_rial <= max_rate:
                    return float(first_rial), "inferred"
    return None, "unusable"


def source_category_for_row(
    channel_type_guess: str,
    likely_individual_shop: bool,
    ranked_row: Optional[Dict[str, str]],
) -> str:
    if ranked_row:
        source_type = str(ranked_row.get("source_type", "")).strip().lower()
        if source_type == "exchange_shop":
            return "direct_shop"
        if source_type == "settlement_exchange":
            return "settlement_exchange"
        if source_type == "regional_market_channel":
            return "regional_market_channel"
        if source_type == "aggregator":
            return "aggregator"
    lowered = str(channel_type_guess or "").strip().lower()
    if likely_individual_shop or lowered == "individual_exchange_shop":
        return "direct_shop"
    if "market" in lowered:
        return "market_channel"
    if "dealer" in lowered:
        return "regional_market_channel"
    if "aggregator" in lowered:
        return "aggregator"
    return "unknown"


def derive_locality(
    handle: str,
    title: str,
    country_guess: str,
    city_guess: str,
    ranked_row: Optional[Dict[str, str]],
) -> str:
    if ranked_row:
        ranked_locality = str(ranked_row.get("locality_bucket", "")).strip()
        if ranked_locality in TARGET_LOCALITIES:
            return ranked_locality
    return derive_locality_bucket(title=title, handle_or_url=handle, country_guess=country_guess, city_guess=city_guess)


def parse_timestamp(timestamp_iso: str) -> Optional[dt.datetime]:
    text = str(timestamp_iso or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(w for w in weights if w > 0)
    if total_weight <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights) if w > 0) / total_weight


def trimmed_mean(values: Sequence[float], trim_ratio: float = 0.10) -> Optional[float]:
    cleaned = sorted(float(v) for v in values if v > 0)
    if not cleaned:
        return None
    if len(cleaned) < 4:
        return statistics.mean(cleaned)
    trim_n = min(int(math.floor(len(cleaned) * trim_ratio)), max(0, (len(cleaned) - 2) // 2))
    trimmed = cleaned[trim_n : len(cleaned) - trim_n] if trim_n > 0 else cleaned
    return statistics.mean(trimmed) if trimmed else statistics.mean(cleaned)


def remove_mad_outliers(records: Sequence[BasketRecord]) -> Tuple[List[BasketRecord], int]:
    if len(records) < 5:
        return list(records), 0
    values = [rec.normalized_rate_rial for rec in records]
    center = statistics.median(values)
    deviations = [abs(v - center) for v in values]
    mad = statistics.median(deviations)
    if mad <= 0:
        return list(records), 0
    cleaned: List[BasketRecord] = []
    removed = 0
    for rec in records:
        score = abs(rec.normalized_rate_rial - center) / mad
        if score > 4.0:
            removed += 1
            continue
        cleaned.append(rec)
    return cleaned, removed


def record_weight(rec: BasketRecord) -> float:
    base = SOURCE_CATEGORY_BASE_WEIGHT.get(rec.source_category, SOURCE_CATEGORY_BASE_WEIGHT["unknown"])
    quality_factor = max(0.45, min(1.15, rec.overall_quality / 100.0))
    freshness_factor = max(0.35, min(1.10, rec.freshness_score / 100.0))
    directness_factor = max(0.55, min(1.10, 0.55 + (rec.directness_score / 200.0)))
    readiness_factor = max(0.60, min(1.10, 0.60 + (rec.channel_readiness_score / 200.0)))
    return base * quality_factor * freshness_factor * directness_factor * readiness_factor


def benchmark_rate(site_api_dir: Path) -> float:
    benchmark_path = site_api_dir / "benchmark.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return safe_float(payload.get("weighted_rate"))


def normalize_existing_records(
    quote_rows: Sequence[Dict[str, str]],
    metrics_by_handle: Dict[str, Dict[str, str]],
    survey_by_handle: Dict[str, Dict[str, str]],
    ranked_by_handle: Dict[str, Dict[str, str]],
    benchmark_value: float,
) -> List[BasketRecord]:
    records: List[BasketRecord] = []
    min_rate = benchmark_value * 0.55 if benchmark_value > 0 else 500_000.0
    max_rate = benchmark_value * 1.65 if benchmark_value > 0 else 2_500_000.0

    for row in quote_rows:
        handle = str(row.get("handle", "")).strip()
        metric_row = metrics_by_handle.get(handle)
        if metric_row and metric_row.get("recommended_status") not in VALID_STATUSES:
            continue
        if str(row.get("dedup_keep", "")).lower() != "true":
            continue
        if "USD" not in str(row.get("currency", "")):
            continue
        if safe_float(row.get("overall_record_quality_score")) < 60.0:
            continue
        if safe_float(row.get("freshness_score")) < 45.0:
            continue

        normalized_rate, quote_basis = rate_from_quote_row(row)
        if normalized_rate is None or normalized_rate <= 0 or normalized_rate < min_rate or normalized_rate > max_rate:
            normalized_rate, quote_basis = fallback_usd_rate_from_text(
                text=str(row.get("message_text_sample", "")),
                unit_guess=str(row.get("value_unit_guess", "unknown")),
                min_rate=min_rate,
                max_rate=max_rate,
            )
        if normalized_rate is None or normalized_rate <= 0:
            continue

        survey_row = survey_by_handle.get(handle, {})
        ranked_row = ranked_by_handle.get(handle)
        likely_shop = safe_bool(row.get("likely_individual_shop")) or safe_bool(survey_row.get("likely_individual_shop"))
        locality = derive_locality(
            handle=handle,
            title=str(row.get("title", "")) or str(survey_row.get("title", "")),
            country_guess=str(survey_row.get("country_guess", "")),
            city_guess=str(row.get("city_guess", "")) or str(survey_row.get("city_guess", "")),
            ranked_row=ranked_row,
        )
        records.append(
            BasketRecord(
                handle=handle,
                title=str(row.get("title", "")) or str(survey_row.get("title", "")) or handle,
                locality=locality,
                source_category=source_category_for_row(
                    channel_type_guess=str(row.get("channel_type_guess", "")) or str(survey_row.get("channel_type_guess", "")),
                    likely_individual_shop=likely_shop,
                    ranked_row=ranked_row,
                ),
                source_priority=str(row.get("source_priority", "")) or str(metric_row.get("source_priority", "")) or "existing",
                likely_individual_shop=likely_shop,
                channel_type_guess=str(row.get("channel_type_guess", "")) or str(survey_row.get("channel_type_guess", "")),
                normalized_rate_rial=normalized_rate,
                quote_basis=quote_basis,
                overall_quality=safe_float(row.get("overall_record_quality_score")),
                freshness_score=safe_float(row.get("freshness_score")),
                structure_score=safe_float(row.get("structure_score")),
                directness_score=safe_float(row.get("directness_score")),
                timestamp_iso=str(row.get("timestamp_iso", "")),
                dedup_keep=True,
                duplication_flag=str(row.get("duplication_flag", "none")),
                from_new_p1=False,
                channel_readiness_score=safe_float(metric_row.get("ingestion_readiness_score")) if metric_row else safe_float(row.get("overall_record_quality_score")),
            )
        )
    return records


def ingest_ranked_p1_channels(
    ranked_rows: Sequence[Dict[str, str]],
    request_timeout: int,
    sleep_seconds: float,
    benchmark_value: float,
) -> Tuple[List[BasketRecord], List[Dict[str, Any]]]:
    p1_rows = [row for row in ranked_rows if row.get("operational_bucket") == "P1" and row.get("platform") == "telegram"]
    now = utc_now()
    min_rate = benchmark_value * 0.55 if benchmark_value > 0 else 500_000.0
    max_rate = benchmark_value * 1.65 if benchmark_value > 0 else 2_500_000.0
    out: List[BasketRecord] = []
    summaries: List[Dict[str, Any]] = []

    for idx, row in enumerate(p1_rows):
        handle = str(row.get("handle_or_url", "")).strip()
        channel = PilotChannel(
            handle=handle,
            title=str(row.get("title", "")) or handle,
            source_priority="P1",
            origin_priority="exchange_shop_candidate_ranking",
            priority_score=safe_float(row.get("operational_score")),
            channel_type_guess="individual_exchange_shop" if str(row.get("source_type", "")) == "exchange_shop" else str(row.get("source_type", "")),
            likely_individual_shop=safe_bool(row.get("likely_individual_shop")),
            public_url=normalize_public_url(handle, f"https://t.me/s/{handle}"),
            selection_note=str(row.get("notes", "")),
        )
        body, status_code, error = fetch_url(channel.public_url, timeout=request_timeout)
        summary_row: Dict[str, Any] = {
            "handle": handle,
            "title": channel.title,
            "public_url": channel.public_url,
            "status_code": status_code,
            "error": error,
            "messages_seen": 0,
            "records_extracted": 0,
            "usable_records": 0,
        }
        if not body or status_code != 200:
            summaries.append(summary_row)
            if sleep_seconds > 0 and idx < len(p1_rows) - 1:
                time.sleep(sleep_seconds)
            continue

        messages, total_seen = extract_message_rows(body, channel)
        parsed_records = []
        for msg in messages:
            parsed_records.extend(parse_quote_records_from_message(msg, now_dt=now))
        dedup_stats = apply_in_channel_dedup(parsed_records)

        usable_here = 0
        for rec in parsed_records:
            if not rec.dedup_keep:
                continue
            if "USD" not in rec.currency:
                continue
            if rec.overall_record_quality_score < 60:
                continue
            if rec.freshness_score < 45:
                continue
            normalized_rate = rec.midpoint_rial or rec.sell_quote_rial or rec.buy_quote_rial
            quote_basis = "midpoint" if rec.midpoint_rial else ("sell" if rec.sell_quote_rial else ("buy" if rec.buy_quote_rial else "inferred"))
            if not normalized_rate or normalized_rate < min_rate or normalized_rate > max_rate:
                normalized_rate, quote_basis = fallback_usd_rate_from_text(
                    text=rec.message_text_sample,
                    unit_guess=rec.value_unit_guess,
                    min_rate=min_rate,
                    max_rate=max_rate,
                )
            if not normalized_rate or normalized_rate < min_rate or normalized_rate > max_rate:
                continue
            usable_here += 1
            out.append(
                BasketRecord(
                    handle=rec.handle,
                    title=rec.title,
                    locality=str(row.get("locality_bucket", "unknown")),
                    source_category=source_category_for_row(
                        channel_type_guess=channel.channel_type_guess,
                        likely_individual_shop=channel.likely_individual_shop,
                        ranked_row=row,
                    ),
                    source_priority="P1",
                    likely_individual_shop=channel.likely_individual_shop,
                    channel_type_guess=channel.channel_type_guess,
                    normalized_rate_rial=float(normalized_rate),
                    quote_basis=quote_basis,
                    overall_quality=float(rec.overall_record_quality_score),
                    freshness_score=float(rec.freshness_score),
                    structure_score=float(rec.structure_score),
                    directness_score=float(rec.directness_score),
                    timestamp_iso=rec.timestamp_iso,
                    dedup_keep=True,
                    duplication_flag=rec.duplication_flag,
                    from_new_p1=True,
                    channel_readiness_score=float(rec.overall_record_quality_score),
                )
            )

        summary_row.update(
            {
                "messages_seen": total_seen,
                "records_extracted": len(parsed_records),
                "usable_records": usable_here,
                "dedup_stats": dedup_stats,
            }
        )
        summaries.append(summary_row)
        if sleep_seconds > 0 and idx < len(p1_rows) - 1:
            time.sleep(sleep_seconds)

    return out, summaries


def summarize_basket(
    basket_name: str,
    records: Sequence[BasketRecord],
    benchmark_value: float,
) -> Dict[str, Any]:
    base_row = {
        "basket_name": basket_name,
        "weighted_rate": None,
        "median_rate": None,
        "spread_vs_benchmark_pct": None,
        "usable_record_count": 0,
        "contributing_channel_count": 0,
        "basket_confidence": 0.0,
        "publishable": False,
        "suppression_reason": "no_usable_records",
    }
    if not records:
        return base_row

    cleaned, outliers_removed = remove_mad_outliers(records)
    if not cleaned:
        return base_row

    values = [rec.normalized_rate_rial for rec in cleaned]
    weights = [record_weight(rec) for rec in cleaned]
    median_rate = statistics.median(values)
    weighted_rate = weighted_mean(values, weights) or median_rate
    trimmed = trimmed_mean(values) or median_rate
    mean_rate = statistics.mean(values)
    dispersion_cv = statistics.pstdev(values) / mean_rate if len(values) > 1 and mean_rate > 0 else 0.0
    avg_freshness = statistics.mean(rec.freshness_score for rec in cleaned)
    direct_weight = sum(w for rec, w in zip(cleaned, weights) if rec.source_category == "direct_shop")
    total_weight = sum(weights)
    direct_share = (direct_weight / total_weight) if total_weight > 0 else 0.0

    channel_weights: Dict[str, float] = {}
    for rec, weight in zip(cleaned, weights):
        channel_weights[rec.handle] = channel_weights.get(rec.handle, 0.0) + weight
    contributing_channel_count = len(channel_weights)
    top_channel_share = (max(channel_weights.values()) / total_weight) if channel_weights and total_weight > 0 else 1.0

    count_component = min(30.0, len(cleaned) * 3.0)
    channel_component = min(25.0, contributing_channel_count * 7.0)
    dispersion_component = max(0.0, 25.0 - (dispersion_cv * 120.0))
    freshness_component = max(0.0, min(10.0, avg_freshness / 10.0))
    direct_component = max(0.0, min(10.0, direct_share * 10.0))
    confidence = round(count_component + channel_component + dispersion_component + freshness_component + direct_component, 2)

    publishable = True
    suppression_reason = ""
    if len(cleaned) < 4:
        publishable = False
        suppression_reason = "insufficient_usable_records"
    elif contributing_channel_count < 2:
        publishable = False
        suppression_reason = "single_channel_concentration"
    elif dispersion_cv > 0.18:
        publishable = False
        suppression_reason = "high_dispersion"
    elif top_channel_share > 0.78:
        publishable = False
        suppression_reason = "concentrated_single_source"
    elif confidence < 55.0:
        publishable = False
        suppression_reason = "low_confidence"

    spread_pct = ((weighted_rate - benchmark_value) / benchmark_value) * 100.0 if benchmark_value > 0 else None
    return {
        "basket_name": basket_name,
        "weighted_rate": round(weighted_rate, 2),
        "median_rate": round(median_rate, 2),
        "trimmed_mean_rate": round(trimmed, 2),
        "spread_vs_benchmark_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "usable_record_count": len(cleaned),
        "contributing_channel_count": contributing_channel_count,
        "basket_confidence": confidence,
        "publishable": publishable,
        "suppression_reason": suppression_reason,
        "dispersion_cv": round(dispersion_cv, 6),
        "top_channel_share": round(top_channel_share, 6),
        "outliers_removed": outliers_removed,
        "source_category_mix": source_category_mix(cleaned),
        "channels": sorted(channel_weights.keys()),
    }


def source_category_mix(records: Sequence[BasketRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        counts[rec.source_category] = counts.get(rec.source_category, 0) + 1
    return dict(sorted(counts.items()))


def contributing_channel_rows(records: Sequence[BasketRecord]) -> List[Dict[str, Any]]:
    by_handle: Dict[str, List[BasketRecord]] = {}
    for rec in records:
        by_handle.setdefault(rec.handle, []).append(rec)

    rows: List[Dict[str, Any]] = []
    for handle, items in by_handle.items():
        first = items[0]
        rows.append(
            {
                "handle": handle,
                "title": first.title,
                "locality": first.locality,
                "source_category": first.source_category,
                "source_priority": first.source_priority,
                "usable_record_count": len(items),
                "average_quality": round(statistics.mean(item.overall_quality for item in items), 2),
                "average_freshness": round(statistics.mean(item.freshness_score for item in items), 2),
                "from_new_p1": first.from_new_p1,
            }
        )
    rows.sort(key=lambda row: (-row["usable_record_count"], row["handle"]))
    return rows


def build_card_payload(
    basket_rows: Sequence[Dict[str, Any]],
    network_summary: Dict[str, Any],
    locality_internal_status: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    locality_internal_status = locality_internal_status or {}
    cards = []
    for row in basket_rows:
        if row["basket_name"] == "unknown":
            continue
        internal_status = locality_internal_status.get(row["basket_name"], "")
        render_on_homepage = internal_status != "watchlist_only"
        if not render_on_homepage:
            continue
        weighted_rate = row.get("weighted_rate")
        spread = row.get("spread_vs_benchmark_pct")
        cards.append(
            {
                "basket_name": row["basket_name"],
                "weighted_rate": weighted_rate,
                "median_rate": row.get("median_rate"),
                "spread_vs_benchmark_pct": spread,
                "usable_record_count": row.get("usable_record_count", 0),
                "contributing_channel_count": row.get("contributing_channel_count", 0),
                "basket_confidence": row.get("basket_confidence", 0.0),
                "publishable": bool(row.get("publishable")),
                "suppression_reason": row.get("suppression_reason", ""),
                "internal_status": internal_status,
                "render_on_homepage": render_on_homepage,
                "status_text": (
                    "Diagnostics basket available"
                    if row.get("publishable")
                    else f"Suppressed: {row.get('suppression_reason', 'unavailable').replace('_', ' ')}"
                ),
                "rate_text": f"{weighted_rate:,.0f} IRR" if isinstance(weighted_rate, (int, float)) else "Unavailable",
                "spread_text": f"{spread:+.2f}%" if isinstance(spread, (int, float)) else "N/A",
            }
        )
    return {
        "generated_at": network_summary["generated_at"],
        "diagnostics_only": True,
        "benchmark_weighted_rate": network_summary["benchmark_weighted_rate"],
        "locality_internal_status": locality_internal_status,
        "cards": cards,
    }


def build_network_summary(
    basket_rows: Sequence[Dict[str, Any]],
    all_records: Sequence[BasketRecord],
    p1_ingestion: Sequence[Dict[str, Any]],
    benchmark_value: float,
    locality_internal_status: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    locality_internal_status = locality_internal_status or {}
    publishable_count = sum(1 for row in basket_rows if row.get("publishable") and row.get("basket_name") != "unknown")
    rows_by_locality = {row["basket_name"]: row for row in basket_rows}
    locality_summary = []
    for locality in LOCALITY_ORDER:
        row = rows_by_locality.get(locality)
        locality_summary.append(
            {
                "basket_name": locality,
                "publishable": bool(row and row.get("publishable")),
                "usable_record_count": row.get("usable_record_count", 0) if row else 0,
                "contributing_channel_count": row.get("contributing_channel_count", 0) if row else 0,
                "basket_confidence": row.get("basket_confidence", 0.0) if row else 0.0,
                "suppression_reason": row.get("suppression_reason", "no_usable_records") if row else "no_usable_records",
                "internal_status": locality_internal_status.get(locality, ""),
            }
        )

    return {
        "generated_at": iso_now(),
        "diagnostics_only": True,
        "benchmark_weighted_rate": benchmark_value,
        "total_usable_records": len(all_records),
        "total_contributing_channels": len({rec.handle for rec in all_records}),
        "publishable_basket_count": publishable_count,
        "added_p1_candidates": list(p1_ingestion),
        "source_category_mix": source_category_mix(all_records),
        "locality_internal_status": locality_internal_status,
        "locality_summary": locality_summary,
        "channels": contributing_channel_rows(all_records),
    }


def build_exchange_shop_baskets(
    survey_dir: Path,
    site_api_dir: Path,
    request_timeout: int,
    sleep_seconds: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ranking_rows = load_csv(survey_dir / "exchange_shop_candidate_ranking.csv")
    channel_rows = load_csv(survey_dir / "channel_survey.csv")
    metric_rows = load_csv(survey_dir / "pilot_channel_metrics.csv")
    quote_rows = load_csv(survey_dir / "pilot_quote_records.csv")

    metrics_by_handle = build_lookup(metric_rows, "handle")
    survey_by_handle = build_lookup(channel_rows, "handle")
    ranked_by_handle = build_lookup(ranking_rows, "handle_or_url")
    benchmark_value = benchmark_rate(site_api_dir)
    locality_internal_status = load_locality_internal_status(survey_dir)

    existing_records = normalize_existing_records(
        quote_rows=quote_rows,
        metrics_by_handle=metrics_by_handle,
        survey_by_handle=survey_by_handle,
        ranked_by_handle=ranked_by_handle,
        benchmark_value=benchmark_value,
    )
    new_p1_records, p1_ingestion = ingest_ranked_p1_channels(
        ranked_rows=ranking_rows,
        request_timeout=request_timeout,
        sleep_seconds=sleep_seconds,
        benchmark_value=benchmark_value,
    )
    all_records = existing_records + new_p1_records

    basket_rows = []
    for locality in LOCALITY_ORDER:
        locality_records = [rec for rec in all_records if rec.locality == locality]
        basket_rows.append(summarize_basket(locality, locality_records, benchmark_value))

    baskets_payload = {
        "generated_at": iso_now(),
        "diagnostics_only": True,
        "benchmark_weighted_rate": benchmark_value,
        "baskets": basket_rows,
    }
    network_summary = build_network_summary(
        basket_rows=basket_rows,
        all_records=all_records,
        p1_ingestion=p1_ingestion,
        benchmark_value=benchmark_value,
        locality_internal_status=locality_internal_status,
    )
    card_payload = build_card_payload(
        basket_rows=basket_rows,
        network_summary=network_summary,
        locality_internal_status=locality_internal_status,
    )
    return baskets_payload, card_payload, network_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exchange shop basket diagnostics artifacts.")
    parser.add_argument("--survey-dir", type=Path, default=Path("survey_outputs"))
    parser.add_argument("--site-api-dir", type=Path, default=Path("site/api"))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baskets_payload, card_payload, network_summary = build_exchange_shop_baskets(
        survey_dir=args.survey_dir,
        site_api_dir=args.site_api_dir,
        request_timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    write_json(args.site_api_dir / "exchange_shop_baskets.json", baskets_payload)
    write_json(args.site_api_dir / "exchange_shop_baskets_card.json", card_payload)
    write_json(args.site_api_dir / "exchange_shop_network_summary.json", network_summary)

    print(f"exchange_shop_baskets_json={args.site_api_dir / 'exchange_shop_baskets.json'}")
    print(f"exchange_shop_baskets_card_json={args.site_api_dir / 'exchange_shop_baskets_card.json'}")
    print(f"exchange_shop_network_summary_json={args.site_api_dir / 'exchange_shop_network_summary.json'}")
    print(f"publishable_basket_count={network_summary['publishable_basket_count']}")
    print(f"total_usable_records={network_summary['total_usable_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
