#!/usr/bin/env python3
"""Review UAE exchange/remittance candidates for locality-basket usability.

This script stays in diagnostics mode. It evaluates the UAE candidate set across
public websites, linked rate pages, Instagram pages, and Telegram pages, then
decides whether each source emits a usable public pricing signal for the UAE
locality basket.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import statistics
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.exchange_shop_baskets import fallback_usd_rate_from_text
from scripts.telegram_quote_pilot_ingestion import (
    MessageRow,
    PilotChannel,
    apply_in_channel_dedup,
    extract_message_rows,
    normalize_public_url,
    parse_quote_records_from_message,
)
from scripts.uae_exchange_discovery import (
    AED_WORDS,
    IRAN_TRANSFER_KEYWORDS,
    RATE_KEYWORDS,
    USD_WORDS,
    clean_text,
    detect_last_seen,
    domain_for,
    extract_blocks,
    extract_links,
    extract_meta_content,
    extract_numbers,
    extract_page_title,
    fetch_url,
    normalize_instagram_url,
    normalize_telegram_url,
    normalize_website_url,
    translit_digits,
)

ROOT_RATE_HINTS = ("rate", "rates", "price", "prices", "quote", "quotes", "currency", "exchange", "usd", "aed", "irr", "dirham", "dollar", "rate", "قیمت", "نرخ", "ارز", "دلار", "درهم", "حواله")
REMITTANCE_HINTS = ("حواله", "remittance", "money transfer", "transfer", "settlement", "send money")
IMAGE_RATE_HINTS = ("rate", "rates", "price", "prices", "quote", "board", "currency", "نرخ", "قیمت", "ارز", "دلار", "درهم")
USD_RATE_MIN_MULT = 0.45
USD_RATE_MAX_MULT = 1.80
AED_RATE_MIN = 40_000.0
AED_RATE_MAX = 600_000.0


@dataclass
class UAECandidate:
    business_name: str
    website: str
    instagram: str
    telegram: str
    whatsapp_link: str
    city_or_district: str
    country: str
    source_type_guess: str
    rate_page_detected: bool
    rate_post_detected: bool
    iran_transfer_hint: bool
    usd_irr_quote_detected: bool
    aed_irr_quote_detected: bool
    quote_post_count: int
    parseability_score: int
    last_seen: str
    candidate_score: int
    discovery_origins: str
    status_guess: str


@dataclass
class Surface:
    candidate_name: str
    surface_type: str
    url: str
    label: str


@dataclass
class CandidateRecord:
    business_name: str
    primary_surface_used: str
    usd_irr_quote_detected: bool
    aed_irr_quote_detected: bool
    remittance_quote_detected: bool
    repeated_quote_signal: bool
    freshness_status: str
    parseability_score: int
    reliability_score: int
    basket_use_status: str
    usable_record_count: int
    numeric_quote_count: int
    top_signal_sample: str


@dataclass
class BasketCandidateRecord:
    business_name: str
    surface_type: str
    surface_url: str
    currency: str
    quote_basis: str
    buy_quote: str
    sell_quote: str
    midpoint: str
    normalized_irr_value: str
    inferred_unit: str
    timestamp_iso: str
    freshness_status: str
    parseability_score: int
    quote_type_guess: str
    remittance_quote_detected: bool
    message_text_sample: str


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return default


def clip_text(text: str, limit: int = 240) -> str:
    text = clean_text(text or "")
    return text if len(text) <= limit else text[:limit] + "..."


def parse_timestamp(text: str) -> Optional[dt.datetime]:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = translit_digits(raw)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def freshness_status(timestamp_text: str, now_dt: dt.datetime) -> str:
    parsed = parse_timestamp(timestamp_text)
    if parsed is None:
        return "unknown"
    age_days = (now_dt - parsed).total_seconds() / 86400.0
    if age_days <= 14:
        return "fresh"
    if age_days <= 60:
        return "recent"
    return "stale"


def image_rate_board_hint(page: str) -> bool:
    unescaped = html.unescape(page or "")
    for alt in re.findall(r'alt=["\']([^"\']+)["\']', unescaped, flags=re.IGNORECASE):
        lowered = translit_digits(alt).lower()
        if any(tok in lowered for tok in IMAGE_RATE_HINTS):
            return True
    for src in re.findall(r'src=["\']([^"\']+)["\']', unescaped, flags=re.IGNORECASE):
        lowered = urllib.parse.unquote(src).lower()
        if any(tok in lowered for tok in IMAGE_RATE_HINTS):
            return True
    return False


def same_domain(url_a: str, url_b: str) -> bool:
    if not url_a or not url_b:
        return False
    return domain_for(url_a) == domain_for(url_b)


def internal_rate_pages(base_url: str, page: str, limit: int = 3) -> List[str]:
    if not base_url or not page:
        return []
    base_parsed = urllib.parse.urlparse(base_url)
    base_domain = domain_for(base_url)
    found: List[str] = []
    seen: set[str] = set()
    for token in extract_links(page):
        candidate = normalize_website_url(urllib.parse.urljoin(base_url, token))
        if not candidate or domain_for(candidate) != base_domain:
            continue
        parsed = urllib.parse.urlparse(candidate)
        combined = urllib.parse.unquote((parsed.path or "") + "?" + (parsed.query or "")).lower()
        if not any(hint in combined for hint in ROOT_RATE_HINTS):
            continue
        if candidate in seen or candidate == base_url:
            continue
        seen.add(candidate)
        found.append(candidate)
        if len(found) >= limit:
            break
    root_url = f"{base_parsed.scheme}://{base_domain}/"
    if root_url != base_url and root_url not in seen:
        found.insert(0, root_url)
    return found[:limit]


def make_pilot_channel(candidate: UAECandidate) -> PilotChannel:
    handle = candidate.telegram.rstrip("/").split("/")[-1] if candidate.telegram else normalize_handle_from_name(candidate.business_name)
    return PilotChannel(
        handle=handle,
        title=candidate.business_name,
        source_priority="uae_review",
        origin_priority="uae_review",
        priority_score=float(candidate.candidate_score),
        channel_type_guess=candidate.source_type_guess,
        likely_individual_shop=candidate.source_type_guess == "exchange_shop",
        public_url=normalize_public_url(handle, candidate.telegram),
        selection_note="uae_basket_review",
    )


def normalize_handle_from_name(name: str) -> str:
    lowered = translit_digits(name).lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered[:40] or "uae_candidate"


def review_surface_records(
    candidate: UAECandidate,
    surface: Surface,
    benchmark_value: float,
    now_dt: dt.datetime,
    timeout: int,
) -> Tuple[List[BasketCandidateRecord], Dict[str, Any]]:
    page, status, err = fetch_url(surface.url, timeout=timeout)
    if page is None or (status is not None and status >= 400):
        return [], {
            "surface_type": surface.surface_type,
            "url": surface.url,
            "status": err or f"http_{status}" if status else "fetch_failed",
            "usable_record_count": 0,
            "top_score": 0,
            "latest_timestamp": "",
            "image_rate_hint": False,
        }

    extracted: List[BasketCandidateRecord] = []
    latest_timestamp = detect_last_seen(page)
    image_hint = image_rate_board_hint(page)

    if surface.surface_type == "telegram":
        channel = make_pilot_channel(candidate)
        rows, _total = extract_message_rows(page, channel)
        parsed = []
        for row in rows:
            parsed.extend(parse_quote_records_from_message(row, now_dt=now_dt))
        if parsed:
            apply_in_channel_dedup(parsed)
        for rec in parsed:
            if not rec.dedup_keep:
                continue
            normalized, currency = normalized_value_for_record(rec, benchmark_value)
            if normalized is None:
                continue
            extracted.append(
                BasketCandidateRecord(
                    business_name=candidate.business_name,
                    surface_type=surface.surface_type,
                    surface_url=surface.url,
                    currency=currency,
                    quote_basis=quote_basis_for_record(rec),
                    buy_quote=str(rec.buy_quote or ""),
                    sell_quote=str(rec.sell_quote or ""),
                    midpoint=f"{rec.midpoint:.2f}" if rec.midpoint is not None else "",
                    normalized_irr_value=f"{normalized:.2f}",
                    inferred_unit=rec.value_unit_guess,
                    timestamp_iso=rec.timestamp_iso,
                    freshness_status=freshness_status(rec.timestamp_iso, now_dt),
                    parseability_score=int(rec.overall_record_quality_score),
                    quote_type_guess=rec.quote_type_guess,
                    remittance_quote_detected=record_has_remittance_hint(rec.message_text_sample),
                    message_text_sample=clip_text(rec.message_text_sample),
                )
            )
        top_score = max((r.parseability_score for r in extracted), default=0)
        latest_timestamp = max([latest_timestamp] + [r.timestamp_iso for r in extracted])
        return extracted, {
            "surface_type": surface.surface_type,
            "url": surface.url,
            "status": "ok",
            "usable_record_count": len(extracted),
            "top_score": top_score,
            "latest_timestamp": latest_timestamp,
            "image_rate_hint": image_hint,
        }

    title = extract_meta_content(page, "og:title") or extract_page_title(page)
    meta_desc = extract_meta_content(page, "description") or extract_meta_content(page, "og:description")
    blocks = extract_blocks(page)
    text_blocks = [title, meta_desc] + blocks[:30]
    pseudo_records = []
    for idx, block in enumerate(text_blocks):
        block = clean_text(block)
        if not block:
            continue
        msg = MessageRow(
            handle=normalize_handle_from_name(candidate.business_name),
            title=candidate.business_name,
            source_priority="uae_review",
            channel_type_guess=candidate.source_type_guess,
            likely_individual_shop=candidate.source_type_guess == "exchange_shop",
            msg_index=idx,
            timestamp_iso=latest_timestamp or now_iso(),
            timestamp_text=latest_timestamp or "",
            message_text=block,
        )
        pseudo_records.extend(parse_quote_records_from_message(msg, now_dt=now_dt))
    if pseudo_records:
        apply_in_channel_dedup(pseudo_records)
    min_rate = benchmark_value * USD_RATE_MIN_MULT if benchmark_value > 0 else 500_000.0
    max_rate = benchmark_value * USD_RATE_MAX_MULT if benchmark_value > 0 else 2_500_000.0
    for rec in pseudo_records:
        if not rec.dedup_keep:
            continue
        normalized, currency = normalized_value_for_record(rec, benchmark_value)
        if normalized is None:
            usd_fallback, _basis = fallback_usd_rate_from_text(rec.message_text_sample, rec.value_unit_guess, min_rate=min_rate, max_rate=max_rate)
            if usd_fallback is not None:
                normalized, currency = usd_fallback, "USD"
        if normalized is None:
            continue
        extracted.append(
            BasketCandidateRecord(
                business_name=candidate.business_name,
                surface_type=surface.surface_type,
                surface_url=surface.url,
                currency=currency,
                quote_basis=quote_basis_for_record(rec),
                buy_quote=str(rec.buy_quote or ""),
                sell_quote=str(rec.sell_quote or ""),
                midpoint=f"{rec.midpoint:.2f}" if rec.midpoint is not None else "",
                normalized_irr_value=f"{normalized:.2f}",
                inferred_unit=rec.value_unit_guess,
                timestamp_iso=rec.timestamp_iso,
                freshness_status=freshness_status(rec.timestamp_iso, now_dt),
                parseability_score=int(rec.overall_record_quality_score),
                quote_type_guess=rec.quote_type_guess,
                remittance_quote_detected=record_has_remittance_hint(rec.message_text_sample),
                message_text_sample=clip_text(rec.message_text_sample),
            )
        )
    if image_hint and not extracted:
        alt_blocks = extract_image_alt_blocks(page)
        for idx, block in enumerate(alt_blocks):
            msg = MessageRow(
                handle=normalize_handle_from_name(candidate.business_name),
                title=candidate.business_name,
                source_priority="uae_review",
                channel_type_guess=candidate.source_type_guess,
                likely_individual_shop=candidate.source_type_guess == "exchange_shop",
                msg_index=100 + idx,
                timestamp_iso=latest_timestamp or now_iso(),
                timestamp_text=latest_timestamp or "",
                message_text=block,
            )
            for rec in parse_quote_records_from_message(msg, now_dt=now_dt):
                normalized, currency = normalized_value_for_record(rec, benchmark_value)
                if normalized is None:
                    continue
                extracted.append(
                    BasketCandidateRecord(
                        business_name=candidate.business_name,
                        surface_type=surface.surface_type,
                        surface_url=surface.url,
                        currency=currency,
                        quote_basis=quote_basis_for_record(rec),
                        buy_quote=str(rec.buy_quote or ""),
                        sell_quote=str(rec.sell_quote or ""),
                        midpoint=f"{rec.midpoint:.2f}" if rec.midpoint is not None else "",
                        normalized_irr_value=f"{normalized:.2f}",
                        inferred_unit=rec.value_unit_guess,
                        timestamp_iso=rec.timestamp_iso,
                        freshness_status=freshness_status(rec.timestamp_iso, now_dt),
                        parseability_score=int(rec.overall_record_quality_score),
                        quote_type_guess=rec.quote_type_guess,
                        remittance_quote_detected=record_has_remittance_hint(rec.message_text_sample),
                        message_text_sample=clip_text(rec.message_text_sample),
                    )
                )
    top_score = max((r.parseability_score for r in extracted), default=0)
    return extracted, {
        "surface_type": surface.surface_type,
        "url": surface.url,
        "status": "ok",
        "usable_record_count": len(extracted),
        "top_score": top_score,
        "latest_timestamp": latest_timestamp,
        "image_rate_hint": image_hint,
    }


def extract_image_alt_blocks(page: str) -> List[str]:
    blocks = []
    unescaped = html.unescape(page or "")
    for alt in re.findall(r'alt=["\']([^"\']+)["\']', unescaped, flags=re.IGNORECASE):
        cleaned = clean_text(alt)
        if len(cleaned) >= 8:
            blocks.append(cleaned)
    return blocks[:10]


def normalized_value_for_record(rec: Any, benchmark_value: float) -> Tuple[Optional[float], str]:
    currencies = [token for token in str(rec.currency or "").split("|") if token and token != "UNKNOWN"]
    preferred = currencies[0] if currencies else "UNKNOWN"
    if "USD" in currencies:
        normalized = rec.midpoint_rial or rec.sell_quote_rial or rec.buy_quote_rial
        if normalized and benchmark_value > 0:
            min_rate = benchmark_value * USD_RATE_MIN_MULT
            max_rate = benchmark_value * USD_RATE_MAX_MULT
            if min_rate <= float(normalized) <= max_rate:
                return float(normalized), "USD"
    if "AED" in currencies:
        normalized = rec.midpoint_rial or rec.sell_quote_rial or rec.buy_quote_rial
        if normalized and AED_RATE_MIN <= float(normalized) <= AED_RATE_MAX:
            return float(normalized), "AED"
    return None, preferred


def quote_basis_for_record(rec: Any) -> str:
    if rec.midpoint is not None:
        return "midpoint"
    if rec.sell_quote is not None:
        return "sell"
    if rec.buy_quote is not None:
        return "buy"
    return "inferred"


def record_has_remittance_hint(text: str) -> bool:
    lowered = translit_digits(text or "").lower()
    return any(tok in lowered for tok in REMITTANCE_HINTS + IRAN_TRANSFER_KEYWORDS)


def build_surfaces(candidate: UAECandidate, timeout: int) -> List[Surface]:
    surfaces: List[Surface] = []
    seen: set[str] = set()

    def add(surface_type: str, url: str, label: str) -> None:
        normalized = normalize_surface_url(surface_type, url)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        surfaces.append(Surface(candidate_name=candidate.business_name, surface_type=surface_type, url=normalized, label=label))

    add("website", candidate.website, "website")
    if candidate.website:
        page, status, _err = fetch_url(candidate.website, timeout=timeout)
        if page and (status is None or status < 400):
            for linked in internal_rate_pages(candidate.website, page):
                add("rate_page", linked, "linked_rate_page")
    add("instagram", candidate.instagram, "instagram")
    add("telegram", candidate.telegram, "telegram")
    return surfaces


def normalize_surface_url(surface_type: str, url: str) -> Optional[str]:
    if not url:
        return None
    if surface_type == "telegram":
        return normalize_telegram_url(url)
    if surface_type == "instagram":
        return normalize_instagram_url(url)
    website = normalize_website_url(url)
    if not website:
        return None
    parsed = urllib.parse.urlsplit(website)
    path = urllib.parse.quote(parsed.path or "/", safe="/%:@+~!$&'()*;,=")
    query = urllib.parse.quote_plus(parsed.query, safe="=&:%@+~!$'()*;,/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, parsed.fragment))


def load_candidates(path: Path) -> List[UAECandidate]:
    rows: List[UAECandidate] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                UAECandidate(
                    business_name=str(row.get("business_name", "")).strip(),
                    website=str(row.get("website", "")).strip(),
                    instagram=str(row.get("instagram", "")).strip(),
                    telegram=str(row.get("telegram", "")).strip(),
                    whatsapp_link=str(row.get("whatsapp_link", "")).strip(),
                    city_or_district=str(row.get("city_or_district", "")).strip() or "Dubai",
                    country=str(row.get("country", "")).strip() or "UAE",
                    source_type_guess=str(row.get("source_type_guess", "")).strip() or "unknown",
                    rate_page_detected=safe_bool(row.get("rate_page_detected")),
                    rate_post_detected=safe_bool(row.get("rate_post_detected")),
                    iran_transfer_hint=safe_bool(row.get("iran_transfer_hint")),
                    usd_irr_quote_detected=safe_bool(row.get("usd_irr_quote_detected")),
                    aed_irr_quote_detected=safe_bool(row.get("aed_irr_quote_detected")),
                    quote_post_count=safe_int(row.get("quote_post_count")),
                    parseability_score=safe_int(row.get("parseability_score")),
                    last_seen=str(row.get("last_seen", "")).strip(),
                    candidate_score=safe_int(row.get("candidate_score")),
                    discovery_origins=str(row.get("discovery_origins", "")).strip(),
                    status_guess=str(row.get("status_guess", "")).strip() or "unknown",
                )
            )
    return rows


def load_benchmark_value(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return safe_float(payload.get("weighted_rate") or payload.get("median_rate"))


def candidate_primary_surface(surface_summaries: Sequence[Dict[str, Any]]) -> str:
    best = sorted(
        surface_summaries,
        key=lambda item: (int(item.get("usable_record_count", 0)), int(item.get("top_score", 0)), str(item.get("surface_type", ""))),
        reverse=True,
    )
    if not best:
        return "none"
    item = best[0]
    return f"{item['surface_type']}:{item['url']}"


def candidate_freshness(records: Sequence[BasketCandidateRecord], fallback_last_seen: str, now_dt: dt.datetime) -> str:
    best = "unknown"
    order = {"fresh": 3, "recent": 2, "stale": 1, "unknown": 0}
    for record in records:
        status = record.freshness_status
        if order.get(status, 0) > order.get(best, 0):
            best = status
    if best != "unknown":
        return best
    return freshness_status(fallback_last_seen, now_dt)


def candidate_parseability(records: Sequence[BasketCandidateRecord], fallback_score: int) -> int:
    if not records:
        return fallback_score
    return int(round(statistics.median([record.parseability_score for record in records])))


def reliability_score(
    parseability: int,
    freshness: str,
    repeated: bool,
    numeric_count: int,
    has_usd: bool,
    has_aed: bool,
    has_remittance: bool,
    source_type: str,
) -> int:
    score = 0.45 * parseability
    score += min(numeric_count * 6, 18)
    if repeated:
        score += 12
    if has_usd:
        score += 10
    if has_aed:
        score += 6
    if has_remittance:
        score += 8
    if freshness == "fresh":
        score += 16
    elif freshness == "recent":
        score += 10
    elif freshness == "stale":
        score -= 8
    if source_type in {"exchange_shop", "remittance_exchange", "settlement_exchange"}:
        score += 8
    return max(0, min(int(round(score)), 100))


def basket_use_status(
    numeric_count: int,
    repeated: bool,
    freshness: str,
    reliability: int,
    has_signal: bool,
) -> str:
    if not has_signal or numeric_count == 0:
        return "no_usable_signal"
    if freshness == "stale":
        return "stale"
    if numeric_count >= 3 and repeated and reliability >= 68:
        return "publishable"
    if numeric_count >= 1 and reliability >= 42:
        return "monitor_only"
    return "no_usable_signal"


def summarize_support(rows: Sequence[CandidateRecord], candidate_records: Sequence[BasketCandidateRecord]) -> str:
    publishable = sum(1 for row in rows if row.basket_use_status == "publishable")
    monitor = sum(1 for row in rows if row.basket_use_status == "monitor_only")
    estimated_records = len(candidate_records)
    if publishable >= 2 and estimated_records >= 6:
        return "full_card"
    if publishable >= 1 or (monitor >= 2 and estimated_records >= 3):
        return "monitor_card"
    return "still_hidden"


def write_review_csv(path: Path, rows: Sequence[CandidateRecord]) -> None:
    fieldnames = [
        "business_name",
        "primary_surface_used",
        "usd_irr_quote_detected",
        "aed_irr_quote_detected",
        "remittance_quote_detected",
        "repeated_quote_signal",
        "freshness_status",
        "parseability_score",
        "reliability_score",
        "basket_use_status",
        "usable_record_count",
        "numeric_quote_count",
        "top_signal_sample",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: ({"publishable": 0, "monitor_only": 1, "stale": 2, "no_usable_signal": 3}[item.basket_use_status], -item.reliability_score, item.business_name.lower())):
            writer.writerow(
                {
                    "business_name": row.business_name,
                    "primary_surface_used": row.primary_surface_used,
                    "usd_irr_quote_detected": row.usd_irr_quote_detected,
                    "aed_irr_quote_detected": row.aed_irr_quote_detected,
                    "remittance_quote_detected": row.remittance_quote_detected,
                    "repeated_quote_signal": row.repeated_quote_signal,
                    "freshness_status": row.freshness_status,
                    "parseability_score": row.parseability_score,
                    "reliability_score": row.reliability_score,
                    "basket_use_status": row.basket_use_status,
                    "usable_record_count": row.usable_record_count,
                    "numeric_quote_count": row.numeric_quote_count,
                    "top_signal_sample": row.top_signal_sample,
                }
            )


def write_candidate_records_csv(path: Path, rows: Sequence[BasketCandidateRecord]) -> None:
    fieldnames = [
        "business_name",
        "surface_type",
        "surface_url",
        "currency",
        "quote_basis",
        "buy_quote",
        "sell_quote",
        "midpoint",
        "normalized_irr_value",
        "inferred_unit",
        "timestamp_iso",
        "freshness_status",
        "parseability_score",
        "quote_type_guess",
        "remittance_quote_detected",
        "message_text_sample",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (-item.parseability_score, item.business_name.lower(), item.surface_type)):
            writer.writerow(
                {
                    "business_name": row.business_name,
                    "surface_type": row.surface_type,
                    "surface_url": row.surface_url,
                    "currency": row.currency,
                    "quote_basis": row.quote_basis,
                    "buy_quote": row.buy_quote,
                    "sell_quote": row.sell_quote,
                    "midpoint": row.midpoint,
                    "normalized_irr_value": row.normalized_irr_value,
                    "inferred_unit": row.inferred_unit,
                    "timestamp_iso": row.timestamp_iso,
                    "freshness_status": row.freshness_status,
                    "parseability_score": row.parseability_score,
                    "quote_type_guess": row.quote_type_guess,
                    "remittance_quote_detected": row.remittance_quote_detected,
                    "message_text_sample": row.message_text_sample,
                }
            )


def build_summary(rows: Sequence[CandidateRecord], candidate_records: Sequence[BasketCandidateRecord]) -> Dict[str, Any]:
    publishable = [row.business_name for row in rows if row.basket_use_status == "publishable"]
    monitor_only = [row.business_name for row in rows if row.basket_use_status == "monitor_only"]
    stale = [row.business_name for row in rows if row.basket_use_status == "stale"]
    no_signal = [row.business_name for row in rows if row.basket_use_status == "no_usable_signal"]
    return {
        "generated_at": now_iso(),
        "publishable_candidates": publishable,
        "monitor_only_candidates": monitor_only,
        "stale_candidates": stale,
        "no_usable_signal_candidates": no_signal,
        "candidates_with_numeric_quotes": sorted({row.business_name for row in rows if row.numeric_quote_count > 0}),
        "candidates_with_remittance_signals": sorted({row.business_name for row in rows if row.remittance_quote_detected}),
        "estimated_publishable_uae_record_count": len([record for record in candidate_records if record.business_name in publishable]),
        "uae_card_support": summarize_support(rows, candidate_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review UAE exchange candidates for basket readiness")
    parser.add_argument("--candidates-csv", default="survey_outputs/uae_exchange_discovery_candidates.csv")
    parser.add_argument("--benchmark-json", default="site/api/benchmark.json")
    parser.add_argument("--survey-dir", default="survey_outputs")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--sleep", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir = Path(args.survey_dir)
    if not survey_dir.is_absolute():
        survey_dir = ROOT_DIR / survey_dir
    review_csv = survey_dir / "uae_basket_review.csv"
    summary_json = survey_dir / "uae_basket_review_summary.json"
    records_csv = survey_dir / "uae_basket_candidate_records.csv"

    candidates_csv = Path(args.candidates_csv)
    if not candidates_csv.is_absolute():
        candidates_csv = ROOT_DIR / candidates_csv
    benchmark_json = Path(args.benchmark_json)
    if not benchmark_json.is_absolute():
        benchmark_json = ROOT_DIR / args.benchmark_json

    candidates = load_candidates(candidates_csv)
    benchmark_value = load_benchmark_value(benchmark_json)
    now_dt = utc_now()

    candidate_rows: List[CandidateRecord] = []
    candidate_records: List[BasketCandidateRecord] = []

    for candidate in candidates:
        surfaces = build_surfaces(candidate, timeout=args.timeout)
        per_surface_summaries: List[Dict[str, Any]] = []
        per_candidate_records: List[BasketCandidateRecord] = []
        for surface in surfaces:
            records, surface_summary = review_surface_records(
                candidate=candidate,
                surface=surface,
                benchmark_value=benchmark_value,
                now_dt=now_dt,
                timeout=args.timeout,
            )
            per_candidate_records.extend(records)
            per_surface_summaries.append(surface_summary)
            if args.sleep > 0:
                time.sleep(args.sleep)

        usd_detected = any(record.currency == "USD" for record in per_candidate_records)
        aed_detected = any(record.currency == "AED" for record in per_candidate_records)
        remittance_detected = any(record.remittance_quote_detected for record in per_candidate_records) or candidate.iran_transfer_hint
        repeated = len(per_candidate_records) >= 2
        fresh_status = candidate_freshness(per_candidate_records, candidate.last_seen, now_dt)
        parseability = candidate_parseability(per_candidate_records, candidate.parseability_score)
        reliability = reliability_score(
            parseability=parseability,
            freshness=fresh_status,
            repeated=repeated,
            numeric_count=len(per_candidate_records),
            has_usd=usd_detected,
            has_aed=aed_detected,
            has_remittance=remittance_detected,
            source_type=candidate.source_type_guess,
        )
        status = basket_use_status(
            numeric_count=len(per_candidate_records),
            repeated=repeated,
            freshness=fresh_status,
            reliability=reliability,
            has_signal=bool(per_candidate_records),
        )

        candidate_rows.append(
            CandidateRecord(
                business_name=candidate.business_name,
                primary_surface_used=candidate_primary_surface(per_surface_summaries),
                usd_irr_quote_detected=usd_detected,
                aed_irr_quote_detected=aed_detected,
                remittance_quote_detected=remittance_detected,
                repeated_quote_signal=repeated,
                freshness_status=fresh_status,
                parseability_score=parseability,
                reliability_score=reliability,
                basket_use_status=status,
                usable_record_count=len(per_candidate_records),
                numeric_quote_count=len(per_candidate_records),
                top_signal_sample=per_candidate_records[0].message_text_sample if per_candidate_records else "",
            )
        )
        candidate_records.extend(per_candidate_records)

    write_review_csv(review_csv, candidate_rows)
    write_candidate_records_csv(records_csv, candidate_records)
    summary = build_summary(candidate_rows, candidate_records)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
