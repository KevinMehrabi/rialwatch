#!/usr/bin/env python3
"""Discover and extract regional Telegram/locality market signals for RialWatch.

This script is diagnostics-only. It does not alter the benchmark. It builds a
targeted locality-signal layer from public Telegram pages and public websites,
with a focus on Dubai, Sulaymaniyah, Herat, and Germany diaspora channels.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import random
import re
import socket
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.exchange_shop_baskets import (
    BasketRecord,
    benchmark_rate,
    build_lookup,
    build_network_summary,
    fallback_usd_rate_from_text,
    ingest_ranked_p1_channels,
    load_csv,
    normalize_existing_records,
    record_weight,
    remove_mad_outliers,
    safe_bool,
    safe_float,
    source_category_mix,
    weighted_mean,
)
from scripts.telegram_quote_pilot_ingestion import (
    MessageRow,
    PilotChannel,
    apply_in_channel_dedup,
    extract_message_rows,
    fetch_url as fetch_public_url,
    normalize_public_url,
    parse_quote_records_from_message,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

TARGET_BASKETS = ("Iran", "UAE", "Turkey", "Iraq", "Afghanistan", "UK", "Germany")
CITY_TO_BASKET = {
    "Tehran": "Iran",
    "Dubai": "UAE",
    "Istanbul": "Turkey",
    "Sulaymaniyah": "Iraq",
    "Herat": "Afghanistan",
    "London": "UK",
    "Frankfurt": "Germany",
    "Hamburg": "Germany",
}

QUERY_GROUPS: Dict[str, List[str]] = {
    "persian": [
        "site:t.me دلار هرات",
        "site:t.me/s دلار هرات",
        "site:t.me دلار سلیمانیه",
        "site:t.me/s دلار سلیمانیه",
        "site:t.me دلار دبی",
        "site:t.me/s دلار دبی",
        "site:t.me نرخ دلار هرات",
        "site:t.me نرخ دلار سلیمانیه",
        "site:t.me نرخ دلار دبی",
        "site:t.me بازار ارز هرات",
        "site:t.me بازار ارز سلیمانیه",
        "site:t.me بازار ارز دبی",
        "site:t.me هرات دلار",
        "site:t.me سلیمانیه دلار",
        "site:t.me دبی دلار",
        "site:t.me فرانکفورت صرافی",
        "site:t.me هامبورگ صرافی",
    ],
    "english": [
        "herat dollar telegram",
        "sulaymaniyah dollar telegram",
        "dubai dollar telegram",
        "iranian exchange frankfurt telegram",
        "iranian exchange hamburg telegram",
        "dubai settlement rate iran telegram",
        "herat market dollar iran telegram",
        "sulaymaniyah market dollar iran telegram",
    ],
    "arabic": [
        "دولار دبي ايران",
        "سوق دبي دولار ايران",
    ],
    "german": [
        "iran wechselstube frankfurt telegram",
        "iran geldwechsel hamburg telegram",
    ],
}

RAW_TME_RE = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")

REGION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Herat": ("herat", "هرات"),
    "Sulaymaniyah": ("sulaymaniyah", "sulaimaniyah", "سلیمانیه", "سليمانية"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
    "Hamburg": ("hamburg", "هامبورگ"),
    "London": ("london", "لندن"),
    "Istanbul": ("istanbul", "استانبول"),
}

EXCLUDED_DOMAINS = {
    "r.jina.ai",
    "duckduckgo.com",
    "lite.duckduckgo.com",
    "search.brave.com",
    "google.com",
    "www.google.com",
    "maps.google.com",
    "youtube.com",
    "www.youtube.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "x.com",
    "twitter.com",
    "www.twitter.com",
}

SOURCE_TYPE_WEIGHTS = {
    "exchange_shop": 1.00,
    "regional_market_channel": 0.88,
    "settlement_channel": 0.82,
    "aggregator": 0.68,
    "unknown": 0.55,
}


@dataclass
class DiscoverySource:
    key: str
    platform: str
    url: str
    handle_or_url: str
    query_hits: Set[str] = field(default_factory=set)
    origins: Set[str] = field(default_factory=set)
    seed_title: str = ""
    country_guess: str = "unknown"
    city_guess: str = "unknown"
    source_type_guess: str = "unknown"


@dataclass
class QuoteSignalRecord:
    source_handle_or_url: str
    source_title: str
    locality_bucket: str
    source_type: str
    message_text_sample: str
    buy_quote: Optional[float]
    sell_quote: Optional[float]
    midpoint: Optional[float]
    inferred_unit: str
    normalized_irr_value: Optional[float]
    freshness_indicator: str
    freshness_score: int
    parseability_score: int
    timestamp_iso: str


@dataclass
class SourceSummary:
    handle_or_url: str
    title: str
    platform: str
    country_guess: str
    city_guess: str
    source_type: str
    locality_hints: str
    quote_message_count: int
    usable_record_count: int
    buy_sell_pair_count: int
    median_parseability_score: float
    latest_timestamp: str
    status: str
    top_sample: str
    discovery_origins: str


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def clean_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clip_text(text: str, limit: int = 240) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def normalize_telegram_url(raw: str) -> Optional[Tuple[str, str]]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    if token.startswith("t.me/") or token.startswith("telegram.me/"):
        token = "https://" + token
    parsed = urllib.parse.urlparse(token)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc.lower() not in ("t.me", "www.t.me", "telegram.me", "www.telegram.me"):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    handle = parts[1] if parts[0].lower() == "s" and len(parts) > 1 else parts[0]
    handle = re.sub(r"[^A-Za-z0-9_]", "", handle.lower())
    if len(handle) < 5:
        return None
    return handle, f"https://t.me/s/{handle}"


def normalize_website_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    parsed = urllib.parse.urlparse(token)
    if parsed.scheme not in ("http", "https"):
        return None
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain or domain in EXCLUDED_DOMAINS:
        return None
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"https://{domain}{path}"


def search_urls_for_query(query: str, pages: int) -> List[str]:
    encoded = urllib.parse.quote_plus(query)
    urls: List[str] = []
    for idx in range(pages):
        offset = idx * 30
        urls.append(f"https://r.jina.ai/http://lite.duckduckgo.com/lite/?q={encoded}&s={offset}")
        brave_query = urllib.parse.quote(query, safe="")
        brave_path = f"search.brave.com/search%3Fq%3D{brave_query}%26source%3Dweb%26offset%3D{offset}%26spellcheck%3D0"
        urls.append(f"https://r.jina.ai/http://{brave_path}")
    return urls


def fetch_url(url: str, timeout: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fa,en-US;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace"), int(resp.status), None
    except urllib.error.HTTPError as exc:
        return exc.read().decode("utf-8", errors="replace"), int(exc.code), f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return None, None, f"network_error:{exc.reason}"
    except socket.timeout:
        return None, None, "timeout"
    except TimeoutError:
        return None, None, "timeout"


def extract_links(page: str) -> List[str]:
    out: List[str] = []
    unescaped = html.unescape(page)
    for pattern in (RAW_TME_RE, URL_RE):
        for match in pattern.finditer(unescaped):
            token = match.group(0).strip()
            if token:
                out.append(token)
    for href in re.findall(r'href=["\']([^"\']+)["\']', unescaped, flags=re.IGNORECASE):
        token = href.strip()
        if token:
            out.append(token)
    for encoded in re.findall(r"uddg=([^&\"'<>\\s]+)", unescaped, flags=re.IGNORECASE):
        decoded = urllib.parse.unquote(encoded)
        if decoded:
            out.append(decoded)
    return out


def detect_regions(text: str) -> List[str]:
    lowered = translit_digits(text or "").lower()
    hits: List[str] = []
    for region, aliases in REGION_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            hits.append(region)
    return hits


def region_to_basket(region: str) -> str:
    return CITY_TO_BASKET.get(region, "unknown")


def search_discovery(query_plan: Sequence[Tuple[str, str]], pages_per_query: int, timeout: int, sleep_seconds: float) -> Tuple[Dict[str, DiscoverySource], Dict[str, Any]]:
    discovered: Dict[str, DiscoverySource] = {}
    debug: Dict[str, Any] = {"successful_search_requests": 0, "failed_search_requests": 0, "query_stats": {}}
    for group, query in query_plan:
        hits = 0
        for url in search_urls_for_query(query, pages_per_query):
            page, status, err = fetch_url(url, timeout=timeout)
            if page is None or (status is not None and status >= 400):
                debug["failed_search_requests"] += 1
                continue
            debug["successful_search_requests"] += 1
            for token in extract_links(page):
                tg = normalize_telegram_url(token)
                if tg:
                    handle, public_url = tg
                    key = f"telegram:{handle}"
                    source = discovered.get(key)
                    if source is None:
                        source = DiscoverySource(key=key, platform="telegram", url=public_url, handle_or_url=handle)
                        discovered[key] = source
                    source.query_hits.add(query)
                    source.origins.add(group)
                    hits += 1
                    continue
                site = normalize_website_url(token)
                if site:
                    key = f"website:{site}"
                    source = discovered.get(key)
                    if source is None:
                        source = DiscoverySource(key=key, platform="website", url=site, handle_or_url=site)
                        discovered[key] = source
                    source.query_hits.add(query)
                    source.origins.add(group)
                    hits += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds + random.random() * sleep_seconds * 0.4)
        debug["query_stats"][query] = {"group": group, "candidate_hits": hits}
    return discovered, debug


def seed_from_existing_registry(channel_rows: Sequence[Dict[str, str]]) -> Dict[str, DiscoverySource]:
    seeded: Dict[str, DiscoverySource] = {}
    for row in channel_rows:
        handle = str(row.get("handle", "")).strip()
        public_url = str(row.get("public_url", "")).strip()
        title = str(row.get("title", "")).strip()
        city_guess = str(row.get("city_guess", "")).strip()
        channel_type = str(row.get("channel_type_guess", "")).strip()
        sample = str(row.get("last_seen_text_sample", "")).strip()
        joined = " ".join(filter(None, [title, city_guess, sample]))
        has_target_locality = bool(detect_regions(joined)) or city_guess in {"Dubai", "Frankfurt", "Hamburg"}
        if not has_target_locality:
            continue
        if not handle or not public_url:
            continue
        key = f"telegram:{handle}"
        seeded[key] = DiscoverySource(
            key=key,
            platform="telegram",
            url=public_url,
            handle_or_url=handle,
            origins={"existing_registry"},
            seed_title=title,
            city_guess=city_guess or "unknown",
            source_type_guess=channel_type or "unknown",
        )
    return seeded


def extract_page_title(page: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL)
    return clean_text(match.group(1)) if match else ""


def extract_meta_content(page: str, prop: str) -> str:
    pattern = re.compile(
        rf"<meta[^>]+(?:property|name)=[\"']{re.escape(prop)}[\"'][^>]+content=[\"'](.*?)[\"']",
        re.IGNORECASE,
    )
    match = pattern.search(page)
    return html.unescape(match.group(1)).strip() if match else ""


def extract_generic_blocks(page: str) -> List[str]:
    blocks: List[Tuple[int, str]] = []
    for pattern in (
        r"<p[^>]*>(.*?)</p>",
        r"<li[^>]*>(.*?)</li>",
        r"<div[^>]*class=[\"'][^\"']*(?:rate|quote|price|ticker)[^\"']*[\"'][^>]*>(.*?)</div>",
    ):
        for match in re.finditer(pattern, page, flags=re.IGNORECASE | re.DOTALL):
            text = clean_text(match.group(1))
            if text:
                blocks.append((match.start(), text))
    blocks.sort(key=lambda item: item[0])
    seen: Set[str] = set()
    out: List[str] = []
    for _, text in blocks:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def parse_timestamp(ts_iso: str) -> Optional[dt.datetime]:
    if not ts_iso:
        return None
    text = ts_iso.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if out.tzinfo is None:
        return out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def freshness_label(score: int) -> str:
    if score >= 85:
        return "fresh"
    if score >= 60:
        return "recent"
    if score >= 40:
        return "stale"
    return "old"


def classify_source_type(source: DiscoverySource, title: str, text: str) -> str:
    lowered = translit_digits(" ".join(filter(None, [title, text, source.source_type_guess]))).lower()
    if source.source_type_guess == "individual_exchange_shop":
        return "exchange_shop"
    if source.source_type_guess in {"dealer_network_channel", "market_price_channel"}:
        return "regional_market_channel"
    if source.source_type_guess == "aggregator":
        return "aggregator"
    if any(word in lowered for word in ("صرافی", "sarafi", "exchange", "currency exchange")) and any(word in lowered for word in ("خرید", "فروش", "rate", "قیمت", "نرخ")):
        return "exchange_shop"
    if any(word in lowered for word in ("حواله", "remittance", "transfer", "settlement", "پرداخت", "para transfer")):
        return "settlement_channel"
    if sum(1 for region in ("تهران", "هرات", "سلیمانیه", "دبی", "istanbul", "herat", "dubai", "sulaymaniyah") if region in lowered) >= 2:
        return "regional_market_channel"
    if "aggregator" in lowered or "اخبار" in lowered or "analysis" in lowered:
        return "aggregator"
    return "unknown"


def build_message_rows(source: DiscoverySource, title: str, blocks: Sequence[str], page: str) -> List[MessageRow]:
    if source.platform == "telegram":
        channel = PilotChannel(
            handle=source.handle_or_url,
            title=title or source.seed_title or source.handle_or_url,
            source_priority="regional_discovery",
            origin_priority="regional_discovery",
            priority_score=0.0,
            channel_type_guess=source.source_type_guess,
            likely_individual_shop=source.source_type_guess == "individual_exchange_shop",
            public_url=source.url,
            selection_note="regional market discovery",
        )
        rows, _ = extract_message_rows(page, channel)
        return rows

    rows: List[MessageRow] = []
    for idx, block in enumerate(blocks[:30]):
        rows.append(
            MessageRow(
                handle=source.handle_or_url,
                title=title or source.seed_title or source.handle_or_url,
                source_priority="regional_discovery",
                channel_type_guess=source.source_type_guess,
                likely_individual_shop=False,
                msg_index=idx,
                timestamp_iso="",
                timestamp_text="",
                message_text=block,
            )
        )
    return rows


def extract_quote_records_for_source(
    source: DiscoverySource,
    request_timeout: int,
    benchmark_value: float,
) -> Tuple[List[QuoteSignalRecord], SourceSummary]:
    body, status, err = fetch_public_url(source.url, timeout=request_timeout)
    title = source.seed_title
    if body:
        title = extract_meta_content(body, "og:title") or extract_page_title(body) or title or source.handle_or_url
    else:
        title = title or source.handle_or_url

    if body is None:
        return [], SourceSummary(
            handle_or_url=source.handle_or_url,
            title=title,
            platform=source.platform,
            country_guess="unknown",
            city_guess=source.city_guess,
            source_type=source.source_type_guess or "unknown",
            locality_hints="",
            quote_message_count=0,
            usable_record_count=0,
            buy_sell_pair_count=0,
            median_parseability_score=0.0,
            latest_timestamp="",
            status=err or "fetch_failed",
            top_sample="",
            discovery_origins="|".join(sorted(source.origins)),
        )

    blocks = extract_generic_blocks(body)
    rows = build_message_rows(source, title, blocks, body)
    parsed_records = []
    now = now_utc()
    for row in rows:
        parsed_records.extend(parse_quote_records_from_message(row, now_dt=now))
    if source.platform == "telegram" and parsed_records:
        apply_in_channel_dedup(parsed_records)

    source_type = classify_source_type(source, title, " ".join(blocks[:20]))
    country_guess = "unknown"
    city_guess = source.city_guess or "unknown"
    locality_hints = set(detect_regions(" ".join(filter(None, [title, " ".join(blocks[:15])]))))
    if city_guess in REGION_ALIASES:
        locality_hints.add(city_guess)
    if "Dubai" in locality_hints:
        country_guess = "UAE"
    elif "Sulaymaniyah" in locality_hints:
        country_guess = "Iraq"
    elif "Herat" in locality_hints:
        country_guess = "Afghanistan"
    elif "Frankfurt" in locality_hints or "Hamburg" in locality_hints:
        country_guess = "Germany"
    elif "London" in locality_hints:
        country_guess = "UK"
    elif "Istanbul" in locality_hints:
        country_guess = "Turkey"
    elif "Tehran" in locality_hints:
        country_guess = "Iran"

    min_rate = benchmark_value * 0.45 if benchmark_value > 0 else 500_000.0
    max_rate = benchmark_value * 1.80 if benchmark_value > 0 else 2_500_000.0

    extracted: List[QuoteSignalRecord] = []
    parse_scores: List[int] = []
    buy_sell_pairs = 0
    latest_ts = ""
    top_sample = ""

    for rec in parsed_records:
        text = rec.message_text_sample
        localities = detect_regions(text)
        if not localities and city_guess in REGION_ALIASES:
            localities = [city_guess]

        normalized_irr = rec.midpoint_rial or rec.sell_quote_rial or rec.buy_quote_rial
        midpoint = rec.midpoint
        buy_quote = float(rec.buy_quote) if rec.buy_quote is not None else None
        sell_quote = float(rec.sell_quote) if rec.sell_quote is not None else None
        inferred_unit = rec.value_unit_guess

        if not normalized_irr:
            normalized_irr, _quote_basis = fallback_usd_rate_from_text(text, inferred_unit, min_rate=min_rate, max_rate=max_rate)
        if normalized_irr and not (min_rate <= float(normalized_irr) <= max_rate):
            normalized_irr = None

        if not localities or normalized_irr is None:
            continue

        if rec.buy_quote is not None and rec.sell_quote is not None:
            buy_sell_pairs += 1
        parse_scores.append(int(rec.overall_record_quality_score))
        latest_ts = max(latest_ts, rec.timestamp_iso)
        if not top_sample:
            top_sample = text

        for locality in localities:
            basket = region_to_basket(locality)
            if basket == "unknown":
                continue
            extracted.append(
                QuoteSignalRecord(
                    source_handle_or_url=source.handle_or_url,
                    source_title=title,
                    locality_bucket=basket,
                    source_type=source_type,
                    message_text_sample=text,
                    buy_quote=buy_quote,
                    sell_quote=sell_quote,
                    midpoint=midpoint,
                    inferred_unit=inferred_unit,
                    normalized_irr_value=float(normalized_irr),
                    freshness_indicator=freshness_label(int(rec.freshness_score)),
                    freshness_score=int(rec.freshness_score),
                    parseability_score=int(rec.overall_record_quality_score),
                    timestamp_iso=rec.timestamp_iso,
                )
            )

    return extracted, SourceSummary(
        handle_or_url=source.handle_or_url,
        title=title,
        platform=source.platform,
        country_guess=country_guess,
        city_guess=city_guess,
        source_type=source_type,
        locality_hints="|".join(sorted(locality_hints)),
        quote_message_count=len(parsed_records),
        usable_record_count=len(extracted),
        buy_sell_pair_count=buy_sell_pairs,
        median_parseability_score=round(statistics.median(parse_scores), 2) if parse_scores else 0.0,
        latest_timestamp=latest_ts,
        status="ok" if status == 200 else (err or f"http_{status}"),
        top_sample=clip_text(top_sample),
        discovery_origins="|".join(sorted(source.origins)),
    )


def source_category_from_signal_type(source_type: str) -> str:
    mapping = {
        "exchange_shop": "direct_shop",
        "regional_market_channel": "regional_market_channel",
        "settlement_channel": "settlement_exchange",
        "aggregator": "aggregator",
        "unknown": "unknown",
    }
    return mapping.get(source_type, "unknown")


def to_basket_record(record: QuoteSignalRecord) -> BasketRecord:
    source_category = source_category_from_signal_type(record.source_type)
    return BasketRecord(
        handle=record.source_handle_or_url,
        title=record.source_title,
        locality=record.locality_bucket,
        source_category=source_category,
        source_priority="regional_discovery",
        likely_individual_shop=record.source_type == "exchange_shop",
        channel_type_guess=record.source_type,
        normalized_rate_rial=float(record.normalized_irr_value or 0.0),
        quote_basis="midpoint" if record.midpoint is not None else ("sell" if record.sell_quote is not None else "buy"),
        overall_quality=float(record.parseability_score),
        freshness_score=float(record.freshness_score),
        structure_score=float(record.parseability_score),
        directness_score=82.0 if record.source_type == "exchange_shop" else (70.0 if record.source_type in {"regional_market_channel", "settlement_channel"} else 55.0),
        timestamp_iso=record.timestamp_iso,
        dedup_keep=True,
        duplication_flag="none",
        from_new_p1=False,
        channel_readiness_score=float(record.parseability_score),
    )


def summarize_enriched_basket(basket_name: str, records: Sequence[BasketRecord], benchmark_value: float) -> Dict[str, Any]:
    base = {
        "basket_name": basket_name,
        "signal_type_used": None,
        "weighted_rate": None,
        "median_rate": None,
        "spread_vs_benchmark_pct": None,
        "usable_record_count": 0,
        "contributing_source_count": 0,
        "basket_confidence": 0.0,
        "publishable": False,
        "suppression_reason": "no_usable_records",
        "top_sources": [],
    }
    if not records:
        return base

    cleaned, outliers_removed = remove_mad_outliers(records)
    if not cleaned:
        return base

    values = [rec.normalized_rate_rial for rec in cleaned]
    weights = [record_weight(rec) * SOURCE_TYPE_WEIGHTS.get(rec.channel_type_guess if rec.channel_type_guess in SOURCE_TYPE_WEIGHTS else rec.source_category, 1.0) for rec in cleaned]
    median_rate = statistics.median(values)
    weighted_rate = weighted_mean(values, weights) or median_rate
    mean_rate = statistics.mean(values)
    dispersion_cv = statistics.pstdev(values) / mean_rate if len(values) > 1 and mean_rate > 0 else 0.0
    freshness_avg = statistics.mean(rec.freshness_score for rec in cleaned)
    structure_avg = statistics.mean(rec.overall_quality for rec in cleaned)

    source_weights: Dict[str, float] = {}
    source_categories: Dict[str, float] = {}
    for rec, weight in zip(cleaned, weights):
        source_weights[rec.handle] = source_weights.get(rec.handle, 0.0) + weight
        source_categories[rec.source_category] = source_categories.get(rec.source_category, 0.0) + weight
    top_share = (max(source_weights.values()) / sum(source_weights.values())) if source_weights else 1.0
    dominant_category = max(source_categories, key=source_categories.get)

    confidence = (
        min(32.0, len(cleaned) * 3.5)
        + min(18.0, len(source_weights) * 6.0)
        + max(0.0, 20.0 - (dispersion_cv * 90.0))
        + min(15.0, freshness_avg / 6.5)
        + min(15.0, structure_avg / 7.0)
    )
    confidence = round(max(0.0, min(100.0, confidence)), 2)

    publishable = True
    suppression_reason = ""
    if len(cleaned) < 3:
        publishable = False
        suppression_reason = "insufficient_usable_records"
    elif confidence < 42.0:
        publishable = False
        suppression_reason = "low_confidence"
    elif freshness_avg < 35.0:
        publishable = False
        suppression_reason = "stale_signal"
    elif dispersion_cv > 0.28:
        publishable = False
        suppression_reason = "high_dispersion"
    elif top_share > 0.96 and len(cleaned) < 4:
        publishable = False
        suppression_reason = "single_source_dominance"

    spread_pct = ((weighted_rate - benchmark_value) / benchmark_value) * 100.0 if benchmark_value > 0 else None
    top_sources = sorted(source_weights.items(), key=lambda item: (-item[1], item[0]))[:3]
    signal_type_label = {
        "direct_shop": "exchange_shop",
        "regional_market_channel": "regional_market_channel",
        "settlement_exchange": "settlement_channel",
        "aggregator": "aggregator",
        "unknown": "unknown",
    }.get(dominant_category, "unknown")

    return {
        "basket_name": basket_name,
        "signal_type_used": signal_type_label,
        "weighted_rate": round(weighted_rate, 2),
        "median_rate": round(median_rate, 2),
        "spread_vs_benchmark_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "usable_record_count": len(cleaned),
        "contributing_source_count": len(source_weights),
        "basket_confidence": confidence,
        "publishable": publishable,
        "suppression_reason": suppression_reason,
        "dispersion_cv": round(dispersion_cv, 6),
        "outliers_removed": outliers_removed,
        "top_sources": [handle for handle, _ in top_sources],
    }


def write_csv(path: Path, rows: Sequence[SourceSummary]) -> None:
    fieldnames = [
        "handle_or_url",
        "title",
        "platform",
        "country_guess",
        "city_guess",
        "source_type",
        "locality_hints",
        "quote_message_count",
        "usable_record_count",
        "buy_sell_pair_count",
        "median_parseability_score",
        "latest_timestamp",
        "status",
        "top_sample",
        "discovery_origins",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (-item.usable_record_count, -item.median_parseability_score, item.handle_or_url.lower())):
            writer.writerow(
                {
                    "handle_or_url": row.handle_or_url,
                    "title": row.title,
                    "platform": row.platform,
                    "country_guess": row.country_guess,
                    "city_guess": row.city_guess,
                    "source_type": row.source_type,
                    "locality_hints": row.locality_hints,
                    "quote_message_count": row.quote_message_count,
                    "usable_record_count": row.usable_record_count,
                    "buy_sell_pair_count": row.buy_sell_pair_count,
                    "median_parseability_score": row.median_parseability_score,
                    "latest_timestamp": row.latest_timestamp,
                    "status": row.status,
                    "top_sample": row.top_sample,
                    "discovery_origins": row.discovery_origins,
                }
            )


def load_base_basket_records(survey_dir: Path, site_api_dir: Path, timeout: int, sleep_seconds: float) -> List[BasketRecord]:
    ranking_rows = load_csv(survey_dir / "exchange_shop_candidate_ranking.csv")
    channel_rows = load_csv(survey_dir / "channel_survey.csv")
    metric_rows = load_csv(survey_dir / "pilot_channel_metrics.csv")
    quote_rows = load_csv(survey_dir / "pilot_quote_records.csv")

    metrics_by_handle = build_lookup(metric_rows, "handle")
    survey_by_handle = build_lookup(channel_rows, "handle")
    ranked_by_handle = build_lookup(ranking_rows, "handle_or_url")
    bench = benchmark_rate(site_api_dir)

    existing_records = normalize_existing_records(
        quote_rows=quote_rows,
        metrics_by_handle=metrics_by_handle,
        survey_by_handle=survey_by_handle,
        ranked_by_handle=ranked_by_handle,
        benchmark_value=bench,
    )
    new_p1_records, _ = ingest_ranked_p1_channels(
        ranked_rows=ranking_rows,
        request_timeout=timeout,
        sleep_seconds=sleep_seconds,
        benchmark_value=bench,
    )
    return existing_records + new_p1_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted regional/locality market signal discovery for RialWatch")
    parser.add_argument("--survey-dir", type=Path, default=Path("survey_outputs"))
    parser.add_argument("--site-api-dir", type=Path, default=Path("site/api"))
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument("--max-discovered-sources", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir = args.survey_dir if args.survey_dir.is_absolute() else ROOT_DIR / args.survey_dir
    site_api_dir = args.site_api_dir if args.site_api_dir.is_absolute() else ROOT_DIR / args.site_api_dir
    survey_dir.mkdir(parents=True, exist_ok=True)
    site_api_dir.mkdir(parents=True, exist_ok=True)

    channel_rows = load_csv(survey_dir / "channel_survey.csv")
    base_basket_payload = json.loads((site_api_dir / "exchange_shop_baskets.json").read_text(encoding="utf-8")) if (site_api_dir / "exchange_shop_baskets.json").exists() else {"baskets": []}
    previous_usable = {row["basket_name"]: int(row.get("usable_record_count", 0) or 0) for row in base_basket_payload.get("baskets", []) if isinstance(row, dict)}

    query_plan = [(group, query) for group, queries in QUERY_GROUPS.items() for query in queries]
    discovered_sources, search_debug = search_discovery(
        query_plan=query_plan,
        pages_per_query=args.pages_per_query,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    existing_seeded = seed_from_existing_registry(channel_rows)
    discovered_sources.update({k: v for k, v in existing_seeded.items() if k not in discovered_sources})

    ordered_sources = sorted(discovered_sources.values(), key=lambda item: (item.platform, item.url))
    if args.max_discovered_sources > 0:
        ordered_sources = ordered_sources[: args.max_discovered_sources]

    benchmark_value = benchmark_rate(site_api_dir)
    quote_records: List[QuoteSignalRecord] = []
    source_rows: List[SourceSummary] = []
    for idx, source in enumerate(ordered_sources):
        extracted, summary = extract_quote_records_for_source(source, request_timeout=args.timeout, benchmark_value=benchmark_value)
        quote_records.extend(extracted)
        if summary.usable_record_count > 0 or summary.quote_message_count > 0:
            source_rows.append(summary)
        if args.sleep_seconds > 0 and idx < len(ordered_sources) - 1:
            time.sleep(args.sleep_seconds + random.random() * args.sleep_seconds * 0.35)

    regional_basket_records = [to_basket_record(record) for record in quote_records]
    base_records = load_base_basket_records(survey_dir, site_api_dir, timeout=args.timeout, sleep_seconds=max(0.1, args.sleep_seconds))
    all_records = base_records + regional_basket_records

    enriched_baskets = []
    top_sources_by_locality: Dict[str, List[str]] = {}
    for basket in TARGET_BASKETS:
        records = [rec for rec in all_records if rec.locality == basket]
        row = summarize_enriched_basket(basket, records, benchmark_value=benchmark_value)
        enriched_baskets.append(row)
        top_sources_by_locality[basket] = list(row.get("top_sources", []))

    gained_localities = [
        row["basket_name"]
        for row in enriched_baskets
        if int(row.get("usable_record_count", 0) or 0) > 0 and previous_usable.get(row["basket_name"], 0) == 0
    ]
    remaining_empty = [row["basket_name"] for row in enriched_baskets if int(row.get("usable_record_count", 0) or 0) == 0]
    can_render = {
        basket: any(row["basket_name"] == basket and row.get("publishable") for row in enriched_baskets)
        for basket in ("UAE", "Iraq", "Afghanistan", "Germany")
    }

    candidates_csv = survey_dir / "regional_market_signal_candidates.csv"
    summary_json = survey_dir / "regional_market_signal_summary.json"
    enriched_json = site_api_dir / "exchange_shop_baskets_enriched.json"

    write_csv(candidates_csv, source_rows)
    enriched_payload = {
        "generated_at": now_iso(),
        "diagnostics_only": True,
        "benchmark_weighted_rate": benchmark_value,
        "baskets": enriched_baskets,
    }
    enriched_json.write_text(json.dumps(enriched_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "new_regional_signal_sources_discovered": len([row for row in source_rows if "existing_registry" not in row.discovery_origins]),
        "existing_registry_signal_sources_used": len([row for row in source_rows if "existing_registry" in row.discovery_origins]),
        "which_locality_baskets_gained_usable_data": gained_localities,
        "which_baskets_remain_empty": remaining_empty,
        "can_render_meaningful_cards": can_render,
        "top_sources_contributing_to_each_improved_basket": {basket: top_sources_by_locality.get(basket, []) for basket in gained_localities},
        "search_debug": search_debug,
        "queries_run": len(query_plan),
        "sources_crawled": len(ordered_sources),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"new_regional_signal_sources_discovered={summary['new_regional_signal_sources_discovered']}")
    print(f"which_locality_baskets_gained_usable_data={','.join(gained_localities) if gained_localities else 'none'}")
    print(f"which_baskets_remain_empty={','.join(remaining_empty) if remaining_empty else 'none'}")
    print(f"uae_can_render={can_render['UAE']}")
    print(f"iraq_can_render={can_render['Iraq']}")
    print(f"afghanistan_can_render={can_render['Afghanistan']}")
    print(f"germany_can_render={can_render['Germany']}")
    print(f"regional_market_signal_candidates_csv={candidates_csv}")
    print(f"regional_market_signal_summary_json={summary_json}")
    print(f"exchange_shop_baskets_enriched_json={enriched_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
