#!/usr/bin/env python3
"""Discover Persian-language regional FX board Telegram channels for RialWatch.

Diagnostics-only. This script targets public Telegram quote-board channels that
publish locality-level signals such as Tehran, Herat, Sulaymaniyah, and Dubai.
It does not modify the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
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

from scripts.exchange_shop_baskets import benchmark_rate, fallback_usd_rate_from_text, record_weight
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

PRIMARY_LOCALITIES = ("Tehran", "Herat", "Sulaymaniyah", "Dubai")
SECONDARY_LOCALITIES = ("Istanbul", "London", "Frankfurt")
ALL_LOCALITIES = PRIMARY_LOCALITIES + SECONDARY_LOCALITIES
LOCALITY_TO_BASKET = {
    "Tehran": "Iran",
    "Herat": "Afghanistan",
    "Sulaymaniyah": "Iraq",
    "Dubai": "UAE",
    "Istanbul": "Turkey",
    "London": "UK",
    "Frankfurt": "Germany",
}
QUERY_GROUPS: Dict[str, List[str]] = {
    "persian": [
        "site:t.me دلار دبی",
        "site:t.me/s دلار دبی",
        "site:t.me نرخ دلار دبی",
        "site:t.me بازار ارز دبی",
        "site:t.me هرات دلار",
        "site:t.me/s هرات دلار",
        "site:t.me نرخ دلار هرات",
        "site:t.me بازار ارز هرات",
        "site:t.me سلیمانیه دلار",
        "site:t.me/s سلیمانیه دلار",
        "site:t.me نرخ دلار سلیمانیه",
        "site:t.me بازار ارز سلیمانیه",
        "site:t.me دلار تهران",
        "site:t.me/s دلار تهران",
        "site:t.me نرخ دلار تهران",
        "site:t.me بازار ارز تهران",
        "site:t.me نرخ دلار لحظه ای",
        "site:t.me قیمت دلار امروز",
        "site:t.me تابلو دلار",
        "site:t.me تابلوی دلار",
        "site:t.me نرخ ارز امروز",
        "site:t.me تابلو ارز",
        "site:t.me بازار آزاد دلار",
        "site:t.me هرات سلیمانیه دبی",
        "site:t.me تهران هرات دبی",
    ],
    "support": [
        "dubai dollar telegram iran",
        "herat dollar telegram iran",
        "sulaymaniyah dollar telegram iran",
        "tehran herat dubai telegram",
        "iran fx board telegram",
    ],
}

RAW_TME_RE = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")
SLASH_PAIR_RE = re.compile(
    r"(?<!\d)(\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)\s*[/\\|\-]\s*(\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)"
)
BUY_WORDS = ("خرید", "buy", "bid")
SELL_WORDS = ("فروش", "sell", "offer", "ask")
BOARD_HINTS = ("تابلو", "board", "quote board", "نرخ", "قیمت", "بازار", "لحظه ای", "لحظه‌ای")
NEWS_HINTS = ("خبر", "اخبار", "تحلیل", "analysis", "news", "breaking")
SHOP_HINTS = ("صرافی", "تماس", "contact", "whatsapp", "آدرس", "address")
QUOTE_HINTS = ("دلار", "usd", "درهم", "aed", "یورو", "eur", "پوند", "gbp", "نرخ", "قیمت", "عرض", "طلب", "خرید", "فروش")
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

LOCALITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Herat": ("herat", "هرات"),
    "Sulaymaniyah": ("sulaymaniyah", "sulaimaniyah", "سلیمانیه", "سليمانية"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
    "London": ("london", "لندن"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
}

SOURCE_TYPE_MULTIPLIER = {
    "regional_fx_board": 1.0,
    "regional_market_channel": 0.88,
    "exchange_shop": 0.72,
    "aggregator": 0.55,
    "unknown": 0.45,
}


@dataclass
class DiscoverySource:
    handle: str
    public_url: str
    query_hits: Set[str] = field(default_factory=set)
    discovery_origins: Set[str] = field(default_factory=set)
    source_type_hint: str = "unknown"


@dataclass
class CandidateRow:
    handle: str
    title: str
    public_url: str
    source_type: str
    quote_message_count: int
    board_message_count: int
    locality_mentions: str
    localities_detected_count: int
    quote_density_score: int
    median_parseability_score: float
    latest_timestamp: str
    status: str
    top_sample: str
    discovery_origins: str


@dataclass
class BoardRecord:
    handle: str
    title: str
    message_text_sample: str
    localities_detected: str
    tehran_quote: str
    herat_quote: str
    sulaymaniyah_quote: str
    dubai_quote: str
    istanbul_quote: str
    london_quote: str
    frankfurt_quote: str
    inferred_unit: str
    normalized_irr_values: str
    buy_quote: str
    sell_quote: str
    midpoint: str
    freshness_indicator: str
    parseability_score: int
    quote_density_score: int
    source_type: str
    timestamp_iso: str
    locality_name: str
    normalized_rate_irr: float
    quote_basis: str
    quote_currency_guess: str


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    for token in re.findall(r"\]\((https?://[^)\s]+)\)", unescaped, flags=re.IGNORECASE):
        cleaned = token.strip()
        if cleaned:
            out.append(cleaned)
    for href in re.findall(r'href=["\']([^"\']+)["\']', unescaped, flags=re.IGNORECASE):
        token = href.strip()
        if token:
            out.append(token)
    for encoded in re.findall(r"uddg=([^&\"'<>\s]+)", unescaped, flags=re.IGNORECASE):
        decoded = urllib.parse.unquote(encoded).strip(")];>,")
        if decoded:
            out.append(decoded)
    return out


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


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def detect_localities(text: str) -> List[str]:
    lowered = translit_digits(text or "").lower()
    hits: List[str] = []
    for locality, aliases in LOCALITY_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            hits.append(locality)
    return hits


def parse_number_token(token: str) -> Optional[int]:
    cleaned = re.sub(r"[^0-9]", "", translit_digits(token or ""))
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def detect_unit(text: str, numbers: Sequence[int]) -> str:
    lowered = translit_digits(text or "").lower()
    if "تومان" in lowered or "toman" in lowered or "tmn" in lowered:
        return "toman"
    if "ریال" in lowered or "irr" in lowered or "rial" in lowered:
        return "rial"
    if numbers:
        median_value = statistics.median(numbers)
        return "toman" if median_value < 400_000 else "rial"
    return "unknown"


def to_rial(value: Optional[int], unit: str) -> Optional[float]:
    if value is None:
        return None
    return float(value * 10) if unit == "toman" else float(value)


def detect_buy_sell_numbers(text: str) -> Tuple[Optional[int], Optional[int]]:
    lowered = translit_digits(text or "").lower()
    nums = [parse_number_token(m.group(0)) for m in NUMBER_RE.finditer(lowered)]
    nums = [n for n in nums if n is not None]
    buy = None
    sell = None
    for word in BUY_WORDS:
        pattern = re.compile(rf"{re.escape(word)}[^0-9]{{0,24}}(\d{{2,3}}(?:[\s,٬،]\d{{3}})+|\d{{5,8}})|(\d{{2,3}}(?:[\s,٬،]\d{{3}})+|\d{{5,8}})[^0-9]{{0,24}}{re.escape(word)}")
        match = pattern.search(lowered)
        if match:
            buy = parse_number_token(match.group(1) or match.group(2))
            break
    for word in SELL_WORDS:
        pattern = re.compile(rf"{re.escape(word)}[^0-9]{{0,24}}(\d{{2,3}}(?:[\s,٬،]\d{{3}})+|\d{{5,8}})|(\d{{2,3}}(?:[\s,٬،]\d{{3}})+|\d{{5,8}})[^0-9]{{0,24}}{re.escape(word)}")
        match = pattern.search(lowered)
        if match:
            sell = parse_number_token(match.group(1) or match.group(2))
            break
    if buy is None and sell is None:
        pair = SLASH_PAIR_RE.search(lowered)
        if pair:
            buy = parse_number_token(pair.group(1))
            sell = parse_number_token(pair.group(2))
    if buy is None and sell is None and len(nums) >= 2 and any(h in lowered for h in BUY_WORDS + SELL_WORDS):
        buy, sell = nums[0], nums[1]
    return buy, sell


def infer_quote_currency(text: str, locality: str) -> str:
    lowered = translit_digits(text or "").lower()
    if "درهم" in lowered or "aed" in lowered:
        return "AED"
    if "یورو" in lowered or "eur" in lowered:
        return "EUR"
    if "پوند" in lowered or "gbp" in lowered:
        return "GBP"
    if "دلار" in lowered or "usd" in lowered:
        return "USD"
    if locality in {"Tehran", "Herat", "Sulaymaniyah"}:
        return "USD"
    if locality == "Dubai":
        return "AED"
    return "UNKNOWN"


def comparable_usd_irr(value_irr: float, currency: str) -> Optional[float]:
    if currency == "USD":
        return float(value_irr)
    if currency == "AED":
        return float(value_irr) * 3.6725
    return None


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
    age_hours = (utc_now() - parsed.astimezone(dt.timezone.utc)).total_seconds() / 3600.0
    if age_hours <= 30:
        return "fresh"
    if age_hours <= 96:
        return "recent"
    if age_hours <= 720:
        return "stale"
    return "old"


def extract_locality_quotes(text: str, benchmark_value: float) -> List[Tuple[str, Optional[int], Optional[int], Optional[float], str, str, str]]:
    lowered = translit_digits(text or "")
    numbers = [parse_number_token(m.group(0)) for m in NUMBER_RE.finditer(lowered)]
    numbers = [n for n in numbers if n is not None]
    unit = detect_unit(lowered, numbers)
    min_rate = benchmark_value * 0.45 if benchmark_value > 0 else 500_000.0
    max_rate = benchmark_value * 1.80 if benchmark_value > 0 else 2_500_000.0

    out: List[Tuple[str, Optional[int], Optional[int], Optional[float], str, str, str]] = []
    lines = [segment.strip() for segment in re.split(r"[\n\r]+|(?<=\.)\s+", lowered) if segment.strip()]
    if not lines:
        lines = [lowered]

    for locality, aliases in LOCALITY_ALIASES.items():
        best: Optional[Tuple[Optional[int], Optional[int], Optional[float], str, str, str]] = None
        for line in lines:
            if not any(alias in line.lower() for alias in aliases):
                continue
            if not any(hint in line.lower() for hint in QUOTE_HINTS + BOARD_HINTS) and not SLASH_PAIR_RE.search(line):
                continue
            line_numbers = [parse_number_token(m.group(0)) for m in NUMBER_RE.finditer(line)]
            line_numbers = [n for n in line_numbers if n is not None]
            if not line_numbers:
                continue
            local_unit = detect_unit(line, line_numbers)
            currency = infer_quote_currency(line, locality)
            buy, sell = detect_buy_sell_numbers(line)
            midpoint: Optional[float] = None
            basis = "inferred"
            if buy is not None and sell is not None:
                buy_rial = to_rial(buy, local_unit)
                sell_rial = to_rial(sell, local_unit)
                if buy_rial is not None and sell_rial is not None:
                    midpoint = (buy_rial + sell_rial) / 2.0
                    basis = "midpoint"
            else:
                pair = SLASH_PAIR_RE.search(line)
                if pair:
                    first = parse_number_token(pair.group(1))
                    second = parse_number_token(pair.group(2))
                    if first is not None and second is not None:
                        buy = first
                        sell = second
                        midpoint = (to_rial(first, local_unit) + to_rial(second, local_unit)) / 2.0
                        basis = "midpoint"
                elif line_numbers:
                    candidate = to_rial(line_numbers[0], local_unit)
                    if candidate is not None:
                        midpoint = candidate
                        basis = "single_value"
            if midpoint is None:
                continue
            comparable_value = comparable_usd_irr(midpoint, currency)
            if comparable_value is not None:
                if not (min_rate <= comparable_value <= max_rate):
                    continue
            else:
                # Keep raw AED-like values when no comparable conversion exists.
                if currency == "AED" and 40_000.0 <= midpoint <= 600_000.0:
                    pass
                else:
                    continue
            board_density = len(detect_localities(line))
            candidate_tuple = (buy, sell, comparable_value if comparable_value is not None else midpoint, local_unit, basis, currency)
            if best is None or board_density > 1:
                best = candidate_tuple
        if best is not None:
            out.append((locality, best[0], best[1], best[2], best[3], best[4], best[5]))
    return out


def extract_meta_content(page: str, prop: str) -> str:
    pattern = re.compile(
        rf"<meta[^>]+(?:property|name)=[\"']{re.escape(prop)}[\"'][^>]+content=[\"'](.*?)[\"']",
        re.IGNORECASE,
    )
    match = pattern.search(page)
    return html.unescape(match.group(1)).strip() if match else ""


def extract_page_title(page: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL)
    return clean_text(match.group(1)) if match else ""


def detect_quote_density_score(messages: Sequence[str]) -> int:
    total = 0
    for msg in messages[:20]:
        num_count = len([m.group(0) for m in NUMBER_RE.finditer(translit_digits(msg))])
        locality_count = len(detect_localities(msg))
        if num_count >= 2:
            total += 10
        if locality_count >= 2:
            total += 15
        if any(hint in translit_digits(msg).lower() for hint in BOARD_HINTS):
            total += 5
    return max(0, min(100, total))


def classify_source_type(title: str, messages: Sequence[str], likely_shop: bool = False) -> str:
    joined = translit_digits(" ".join([title] + list(messages[:20]))).lower()
    locality_hits = len(set(detect_localities(joined)))
    numeric_rich = sum(1 for msg in messages[:20] if len([m.group(0) for m in NUMBER_RE.finditer(translit_digits(msg))]) >= 2)
    if locality_hits >= 3 and numeric_rich >= 3:
        return "regional_fx_board"
    if locality_hits >= 2 and numeric_rich >= 2:
        return "regional_market_channel"
    if likely_shop or any(tok in joined for tok in SHOP_HINTS):
        return "exchange_shop"
    if any(tok in joined for tok in NEWS_HINTS):
        return "aggregator"
    if numeric_rich >= 1 and locality_hits >= 1:
        return "regional_market_channel"
    return "unknown"


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
                if not tg:
                    continue
                handle, public_url = tg
                source = discovered.get(handle)
                if source is None:
                    source = DiscoverySource(handle=handle, public_url=public_url)
                    discovered[handle] = source
                source.query_hits.add(query)
                source.discovery_origins.add(group)
                hits += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds + random.random() * sleep_seconds * 0.35)
        debug["query_stats"][query] = {"group": group, "candidate_hits": hits}
    return discovered, debug


def seed_from_channel_survey(channel_rows: Sequence[Dict[str, str]]) -> Dict[str, DiscoverySource]:
    seeded: Dict[str, DiscoverySource] = {}
    for row in channel_rows:
        ch_type = str(row.get("channel_type_guess", "")).strip()
        if ch_type not in {"market_price_channel", "dealer_network_channel", "aggregator"}:
            continue
        handle = str(row.get("handle", "")).strip().lower()
        public_url = str(row.get("public_url", "")).strip()
        title = str(row.get("title", "")).strip()
        sample = str(row.get("last_seen_text_sample", "")).strip()
        combined = " ".join(filter(None, [title, sample]))
        if not detect_localities(combined):
            continue
        if not handle or not public_url:
            continue
        seeded[handle] = DiscoverySource(
            handle=handle,
            public_url=normalize_public_url(handle, public_url),
            query_hits={"existing_registry_seed"},
            discovery_origins={"existing_registry"},
            source_type_hint=ch_type,
        )
    return seeded


def freshness_indicator(timestamp_iso: str, freshness_score: int) -> str:
    if freshness_score >= 85:
        return "fresh"
    if freshness_score >= 60:
        return "recent"
    if freshness_score >= 40:
        return "stale"
    return "old"


def process_source(source: DiscoverySource, benchmark_value: float, timeout: int) -> Tuple[List[BoardRecord], CandidateRow]:
    body, status, err = fetch_public_url(source.public_url, timeout=timeout)
    if body is None:
        return [], CandidateRow(
            handle=source.handle,
            title=source.handle,
            public_url=source.public_url,
            source_type=source.source_type_hint or "unknown",
            quote_message_count=0,
            board_message_count=0,
            locality_mentions="",
            localities_detected_count=0,
            quote_density_score=0,
            median_parseability_score=0.0,
            latest_timestamp="",
            status=err or "fetch_failed",
            top_sample="",
            discovery_origins="|".join(sorted(source.discovery_origins)),
        )

    title = extract_meta_content(body, "og:title") or extract_page_title(body) or source.handle
    channel = PilotChannel(
        handle=source.handle,
        title=title,
        source_priority="regional_fx_board",
        origin_priority="regional_fx_board",
        priority_score=0.0,
        channel_type_guess=source.source_type_hint or "unknown",
        likely_individual_shop=False,
        public_url=source.public_url,
        selection_note="regional_fx_board_discovery",
    )
    msg_rows, total_seen = extract_message_rows(body, channel)
    parsed_records = []
    now_dt = utc_now()
    for msg in msg_rows:
        parsed_records.extend(parse_quote_records_from_message(msg, now_dt=now_dt))
    if parsed_records:
        apply_in_channel_dedup(parsed_records)

    messages = [row.message_text for row in msg_rows]
    localities = sorted({loc for msg in messages for loc in detect_localities(msg)})
    quote_density = detect_quote_density_score(messages)
    source_type = classify_source_type(title, messages)

    board_records: List[BoardRecord] = []
    parse_scores: List[int] = []
    latest_timestamp = ""
    board_message_count = 0
    top_sample = ""

    for msg in msg_rows:
        per_locality = extract_locality_quotes(msg.message_text, benchmark_value=benchmark_value)
        if not per_locality:
            continue
        board_message_count += 1
        top_sample = top_sample or msg.message_text
        matching_rec = next((rec for rec in parsed_records if rec.message_index == msg.msg_index and rec.dedup_keep), None)
        parse_score = matching_rec.overall_record_quality_score if matching_rec else min(92, 40 + 10 * len(per_locality) + 8 * len(detect_localities(msg.message_text)))
        parse_scores.append(int(parse_score))
        latest_timestamp = max(latest_timestamp, msg.timestamp_iso)

        quote_map: Dict[str, float] = {}
        inferred_unit = matching_rec.value_unit_guess if matching_rec else detect_unit(msg.message_text, [parse_number_token(m.group(0)) for m in NUMBER_RE.finditer(translit_digits(msg.message_text)) if parse_number_token(m.group(0)) is not None])
        buy_quote = matching_rec.buy_quote if matching_rec else None
        sell_quote = matching_rec.sell_quote if matching_rec else None
        midpoint = matching_rec.midpoint if matching_rec else None
        fresh_score = matching_rec.freshness_score if matching_rec else 0
        fresh_label = freshness_indicator(msg.timestamp_iso, fresh_score) if matching_rec else freshness_from_timestamp(msg.timestamp_iso)
        quote_basis = "midpoint" if matching_rec and matching_rec.midpoint is not None else ("sell" if matching_rec and matching_rec.sell_quote is not None else ("buy" if matching_rec and matching_rec.buy_quote is not None else "board"))

        for locality, _buy, _sell, normalized, unit, _basis, currency_guess in per_locality:
            if normalized is None:
                continue
            quote_map[locality] = float(normalized)
            board_records.append(
                BoardRecord(
                    handle=source.handle,
                    title=title,
                    message_text_sample=clip_text(msg.message_text),
                    localities_detected="|".join(sorted([loc for loc, *_ in per_locality])),
                    tehran_quote=format_quote(quote_map.get("Tehran")),
                    herat_quote=format_quote(quote_map.get("Herat")),
                    sulaymaniyah_quote=format_quote(quote_map.get("Sulaymaniyah")),
                    dubai_quote=format_quote(quote_map.get("Dubai")),
                    istanbul_quote=format_quote(quote_map.get("Istanbul")),
                    london_quote=format_quote(quote_map.get("London")),
                    frankfurt_quote=format_quote(quote_map.get("Frankfurt")),
                    inferred_unit=unit or inferred_unit,
                    normalized_irr_values=json.dumps({k: round(v, 2) for k, v in quote_map.items()}, ensure_ascii=False, sort_keys=True),
                    buy_quote=str(_buy or buy_quote or ""),
                    sell_quote=str(_sell or sell_quote or ""),
                    midpoint=f"{midpoint:.2f}" if midpoint is not None else "",
                    freshness_indicator=fresh_label,
                    parseability_score=int(parse_score),
                    quote_density_score=quote_density,
                    source_type=source_type,
                    timestamp_iso=msg.timestamp_iso,
                    locality_name=locality,
                    normalized_rate_irr=float(normalized),
                    quote_basis=_basis,
                    quote_currency_guess=currency_guess,
                )
            )

    candidate = CandidateRow(
        handle=source.handle,
        title=title,
        public_url=source.public_url,
        source_type=source_type,
        quote_message_count=len(parsed_records),
        board_message_count=board_message_count,
        locality_mentions="|".join(localities),
        localities_detected_count=len(localities),
        quote_density_score=quote_density,
        median_parseability_score=round(statistics.median(parse_scores), 2) if parse_scores else 0.0,
        latest_timestamp=latest_timestamp,
        status="ok" if status == 200 else (err or f"http_{status}"),
        top_sample=clip_text(top_sample),
        discovery_origins="|".join(sorted(source.discovery_origins)),
    )
    return board_records, candidate


def format_quote(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else ""


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    denom = sum(weights)
    if denom <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / denom


def dispersion_level(values: Sequence[float]) -> str:
    if len(values) <= 1:
        return "low"
    mean_value = statistics.mean(values)
    if mean_value <= 0:
        return "unknown"
    cv = statistics.pstdev(values) / mean_value
    if cv <= 0.03:
        return "low"
    if cv <= 0.08:
        return "medium"
    return "high"


def locality_freshness(records: Sequence[BoardRecord]) -> str:
    if any(r.freshness_indicator == "fresh" for r in records):
        return "fresh"
    if any(r.freshness_indicator == "recent" for r in records):
        return "recent"
    if any(r.freshness_indicator == "stale" for r in records):
        return "stale"
    return "old"


def summarize_locality(locality_name: str, records: Sequence[BoardRecord], benchmark_value: float) -> Dict[str, Any]:
    basket_name = LOCALITY_TO_BASKET.get(locality_name, locality_name)
    if not records:
        return {
            "locality_name": basket_name,
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

    values = [r.normalized_rate_irr for r in records]
    weights = []
    source_types = {}
    for record in records:
        source_category = (
            "regional_market_channel"
            if record.source_type in {"regional_fx_board", "regional_market_channel"}
            else ("direct_shop" if record.source_type == "exchange_shop" else ("aggregator" if record.source_type == "aggregator" else "unknown"))
        )
        pseudo = type("Pseudo", (), {
            "overall_quality": float(record.parseability_score),
            "freshness_score": 85.0 if record.freshness_indicator == "fresh" else (65.0 if record.freshness_indicator == "recent" else (42.0 if record.freshness_indicator == "stale" else 25.0)),
            "structure_score": float(record.parseability_score),
            "directness_score": 70.0 if record.source_type in {"regional_fx_board", "regional_market_channel"} else 55.0,
            "channel_readiness_score": float(record.quote_density_score),
            "source_category": source_category,
        })
        weight = record_weight(pseudo) * SOURCE_TYPE_MULTIPLIER.get(record.source_type, 0.45)
        weights.append(weight)
        source_types[record.source_type] = source_types.get(record.source_type, 0.0) + weight

    median_rate = statistics.median(values)
    weighted_rate = weighted_mean(values, weights) or median_rate
    freshness = locality_freshness(records)
    dispersion = dispersion_level(values)
    contributing_sources = len({r.handle for r in records})
    avg_parse = statistics.mean(r.parseability_score for r in records)
    confidence = (
        min(28.0, len(records) * 4.0)
        + min(18.0, contributing_sources * 6.0)
        + min(20.0, avg_parse / 4.0)
        + (16.0 if freshness == "fresh" else 10.0 if freshness == "recent" else 3.0 if freshness == "stale" else 0.0)
        + (12.0 if dispersion == "low" else 6.0 if dispersion == "medium" else 0.0)
    )
    confidence = round(max(0.0, min(100.0, confidence)), 2)
    top_type = max(source_types, key=source_types.get) if source_types else "unknown"
    spread = ((weighted_rate - benchmark_value) / benchmark_value) * 100.0 if benchmark_value > 0 else None

    if freshness in {"fresh", "recent"} and len(records) >= 4 and contributing_sources >= 2 and confidence >= 58:
        display_state = "publish"
        suppression_reason = ""
    elif len(records) >= 2 and confidence >= 36:
        display_state = "monitor"
        suppression_reason = "needs_more_fresh_sources" if freshness in {"stale", "old"} else "limited_coverage"
    else:
        display_state = "hide"
        suppression_reason = "insufficient_signal"

    return {
        "locality_name": basket_name,
        "signal_type_used": top_type,
        "usable_record_count": len(records),
        "contributing_source_count": contributing_sources,
        "median_rate": round(median_rate, 2),
        "weighted_rate": round(weighted_rate, 2),
        "spread_vs_benchmark_pct": round(spread, 4) if spread is not None else None,
        "freshness_status": freshness,
        "dispersion_level": dispersion,
        "basket_confidence": confidence,
        "recommended_display_state": display_state,
        "suppression_reason": suppression_reason,
    }


def write_candidates_csv(path: Path, rows: Sequence[CandidateRow]) -> None:
    fieldnames = [
        "handle",
        "title",
        "public_url",
        "source_type",
        "quote_message_count",
        "board_message_count",
        "locality_mentions",
        "localities_detected_count",
        "quote_density_score",
        "median_parseability_score",
        "latest_timestamp",
        "status",
        "top_sample",
        "discovery_origins",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (-item.board_message_count, -item.quote_density_score, item.handle)):
            writer.writerow(
                {
                    "handle": row.handle,
                    "title": row.title,
                    "public_url": row.public_url,
                    "source_type": row.source_type,
                    "quote_message_count": row.quote_message_count,
                    "board_message_count": row.board_message_count,
                    "locality_mentions": row.locality_mentions,
                    "localities_detected_count": row.localities_detected_count,
                    "quote_density_score": row.quote_density_score,
                    "median_parseability_score": row.median_parseability_score,
                    "latest_timestamp": row.latest_timestamp,
                    "status": row.status,
                    "top_sample": row.top_sample,
                    "discovery_origins": row.discovery_origins,
                }
            )


def write_records_csv(path: Path, rows: Sequence[BoardRecord]) -> None:
    fieldnames = [
        "handle",
        "title",
        "message_text_sample",
        "localities_detected",
        "tehran_quote",
        "herat_quote",
        "sulaymaniyah_quote",
        "dubai_quote",
        "istanbul_quote",
        "london_quote",
        "frankfurt_quote",
        "inferred_unit",
        "normalized_irr_values",
        "buy_quote",
        "sell_quote",
        "midpoint",
        "freshness_indicator",
        "parseability_score",
        "quote_density_score",
        "source_type",
        "timestamp_iso",
        "locality_name",
        "normalized_rate_irr",
        "quote_basis",
        "quote_currency_guess",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (-item.parseability_score, item.handle, item.locality_name, item.timestamp_iso)):
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Persian-language regional FX board Telegram channels")
    parser.add_argument("--survey-dir", type=Path, default=Path("survey_outputs"))
    parser.add_argument("--site-api-dir", type=Path, default=Path("site/api"))
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-sources", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir = args.survey_dir if args.survey_dir.is_absolute() else ROOT_DIR / args.survey_dir
    site_api_dir = args.site_api_dir if args.site_api_dir.is_absolute() else ROOT_DIR / args.site_api_dir
    survey_dir.mkdir(parents=True, exist_ok=True)
    site_api_dir.mkdir(parents=True, exist_ok=True)

    benchmark_value = benchmark_rate(site_api_dir)
    channel_rows = load_csv(survey_dir / "channel_survey.csv") if (survey_dir / "channel_survey.csv").exists() else []

    query_plan = [(group, query) for group, queries in QUERY_GROUPS.items() for query in queries]
    discovered, search_debug = search_discovery(
        query_plan=query_plan,
        pages_per_query=args.pages_per_query,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    seeded = seed_from_channel_survey(channel_rows)
    for handle, source in seeded.items():
        if handle not in discovered:
            discovered[handle] = source
        else:
            discovered[handle].discovery_origins.update(source.discovery_origins)
            discovered[handle].query_hits.update(source.query_hits)

    ordered_sources = sorted(discovered.values(), key=lambda item: item.handle)
    if args.max_sources > 0:
        ordered_sources = ordered_sources[: args.max_sources]

    candidate_rows: List[CandidateRow] = []
    board_records: List[BoardRecord] = []
    for idx, source in enumerate(ordered_sources):
        records, candidate = process_source(source, benchmark_value=benchmark_value, timeout=args.timeout)
        if candidate.board_message_count > 0 or candidate.quote_message_count > 0:
            candidate_rows.append(candidate)
        board_records.extend(records)
        if args.sleep_seconds > 0 and idx < len(ordered_sources) - 1:
            time.sleep(args.sleep_seconds + random.random() * args.sleep_seconds * 0.25)

    candidates_csv = survey_dir / "regional_fx_board_candidates.csv"
    records_csv = survey_dir / "regional_fx_board_records.csv"
    summary_json = survey_dir / "regional_fx_board_summary.json"
    basket_review_json = site_api_dir / "regional_fx_board_basket_review.json"

    write_candidates_csv(candidates_csv, candidate_rows)
    write_records_csv(records_csv, board_records)

    locality_basket_rows = []
    gained_localities = []
    for locality in PRIMARY_LOCALITIES + SECONDARY_LOCALITIES:
        locality_records = [record for record in board_records if record.locality_name == locality]
        summary_row = summarize_locality(locality, locality_records, benchmark_value=benchmark_value)
        locality_basket_rows.append(summary_row)
        if summary_row["usable_record_count"] > 0:
            gained_localities.append(summary_row["locality_name"])

    basket_payload = {
        "generated_at": now_iso(),
        "diagnostics_only": True,
        "benchmark_weighted_rate": benchmark_value,
        "localities": locality_basket_rows,
    }
    basket_review_json.write_text(json.dumps(basket_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    channels_with_quote = {
        locality: sorted({record.handle for record in board_records if record.locality_name == locality})
        for locality in PRIMARY_LOCALITIES
    }
    locality_support = {row["locality_name"]: row["recommended_display_state"] for row in locality_basket_rows}

    summary = {
        "generated_at": now_iso(),
        "new_board_channels_discovered": len([row for row in candidate_rows if "existing_registry" not in row.discovery_origins]),
        "channels_with_dubai_quotes": channels_with_quote["Dubai"],
        "channels_with_herat_quotes": channels_with_quote["Herat"],
        "channels_with_sulaymaniyah_quotes": channels_with_quote["Sulaymaniyah"],
        "channels_with_tehran_quotes": channels_with_quote["Tehran"],
        "which_locality_baskets_gained_usable_records": sorted(set(gained_localities)),
        "uae_display_state": locality_support.get("UAE", "hide"),
        "iraq_display_state": locality_support.get("Iraq", "hide"),
        "afghanistan_display_state": locality_support.get("Afghanistan", "hide"),
        "search_debug": search_debug,
        "queries_run": len(query_plan),
        "sources_crawled": len(ordered_sources),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
