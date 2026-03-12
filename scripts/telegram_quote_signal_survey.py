#!/usr/bin/env python3
"""Quote-signal survey for public Telegram channels discovered by RialWatch.

This script runs in research mode only. It does not modify production ingestion.
It reloads promising channels from the first-pass survey and estimates quote-signal
richness by parsing public https://t.me/s/<channel> pages.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

QUOTE_NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,7})(?!\d)")
SLASH_PAIR_RE = re.compile(
    r"(?<!\d)(\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,7})(?!\d)\s*[/\\|\-]\s*(\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,7})(?!\d)"
)

BUY_WORDS = ("خرید", "buy", "bid")
SELL_WORDS = ("فروش", "sell", "offer", "ask")
QUOTE_WORDS = ("نرخ", "قیمت", "rate", "quote", "fx", "ارز")
SHOP_HINT_WORDS = ("صرافی", "exchange", "آدرس", "تماس", "phone", "whatsapp")
NETWORK_HINT_WORDS = ("شبکه", "همکار", "dealer", "channel", "کانال", "بازار", "market")

CURRENCY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "USD": ("usd", "us dollar", "دلار", "دلار آزاد", "دلارتهران", "دلار تهران"),
    "EUR": ("eur", "euro", "یورو"),
    "AED": ("aed", "dirham", "درهم"),
    "GBP": ("gbp", "pound", "پوند"),
}

CITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Mashhad": ("mashhad", "مشهد"),
    "Isfahan": ("isfahan", "اصفهان"),
    "Shiraz": ("shiraz", "شیراز"),
    "Tabriz": ("tabriz", "تبریز"),
    "Karaj": ("karaj", "کرج"),
    "Qom": ("qom", "قم"),
    "Ahvaz": ("ahvaz", "اهواز"),
    "Rasht": ("rasht", "رشت"),
    "Kerman": ("kerman", "کرمان"),
    "Herat": ("herat", "هرات"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
    "Hamburg": ("hamburg", "هامبورگ"),
    "London": ("london", "لندن"),
}


@dataclass
class CandidateChannel:
    handle: str
    public_url: str
    title: str
    channel_type_guess: str
    likely_individual_shop: bool
    parseable_score: int
    quote_post_count: int


@dataclass
class ChannelAnalysis:
    handle: str
    public_url: str
    title: str
    channel_type_guess: str
    likely_individual_shop: bool
    parseable_score: int
    extracted_quote_messages: int
    extracted_usd_quotes: int
    extracted_eur_quotes: int
    extracted_buy_sell_pairs: int
    city_mentions: List[str]
    signal_richness_score: int
    recommended_priority: str
    priority_score: float
    format_stability: float
    notes: str
    status: str


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def clean_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_public_url(handle: str, value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return f"https://t.me/s/{handle}"
    if candidate.startswith("t.me/"):
        candidate = "https://" + candidate
    if not candidate.startswith("http://") and not candidate.startswith("https://"):
        return f"https://t.me/s/{handle}"

    try:
        parsed = urllib.parse.urlparse(candidate)
    except Exception:
        return f"https://t.me/s/{handle}"

    if parsed.netloc.lower() not in ("t.me", "www.t.me", "telegram.me", "www.telegram.me"):
        return f"https://t.me/s/{handle}"

    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return f"https://t.me/s/{handle}"

    if parts[0].lower() == "s" and len(parts) >= 2:
        h = parts[1]
    else:
        h = parts[0]
    h = re.sub(r"[^A-Za-z0-9_]", "", h.lower())
    if not h:
        h = handle
    return f"https://t.me/s/{h}"


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
            body = resp.read().decode("utf-8", errors="replace")
            return body, int(resp.status), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return body, int(exc.code), f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return None, None, f"network_error:{exc.reason}"
    except TimeoutError:
        return None, None, "timeout"


def extract_meta_content(page: str, prop: str) -> str:
    pattern = re.compile(
        rf"<meta[^>]+(?:property|name)=[\"']{re.escape(prop)}[\"'][^>]+content=[\"'](.*?)[\"']",
        re.IGNORECASE,
    )
    match = pattern.search(page)
    return html.unescape(match.group(1)).strip() if match else ""


def extract_message_blocks(page: str) -> List[str]:
    blocks: List[Tuple[int, str]] = []
    patterns = (
        r"<div class=\"tgme_widget_message_text[^\"]*\"[^>]*>(.*?)</div>",
        r"<div class=\"tgme_widget_message_caption[^\"]*\"[^>]*>(.*?)</div>",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, page, flags=re.IGNORECASE | re.DOTALL):
            text = clean_text(match.group(1))
            if text:
                blocks.append((match.start(), text))

    blocks.sort(key=lambda x: x[0])
    return [item[1] for item in blocks]


def load_discovered_channels(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    if not isinstance(arr, list):
        return out
    for item in arr:
        if not isinstance(item, dict):
            continue
        handle = str(item.get("handle", "")).strip().lower()
        if not handle:
            continue
        out[handle] = {
            "public_url": str(item.get("public_url", "")).strip(),
            "title": str(item.get("title", "")).strip(),
        }
    return out


def build_candidate_channels(
    channel_survey_csv: Path,
    discovered_map: Dict[str, Dict[str, str]],
    max_channels: int,
) -> List[CandidateChannel]:
    candidates: List[Tuple[float, CandidateChannel]] = []

    with channel_survey_csv.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            handle = str(row.get("handle", "")).strip().lower()
            if not handle:
                continue

            parseable_score = to_int(row.get("parseable_score", "0"), 0)
            quote_post_count = to_int(row.get("quote_post_count", "0"), 0)
            likely_shop = to_bool(row.get("likely_individual_shop", "false"))
            channel_type = str(row.get("channel_type_guess", "unclear")).strip() or "unclear"
            status = str(row.get("status", "")).strip()

            is_network_or_market = channel_type in ("dealer_network_channel", "market_price_channel")
            keep = (
                quote_post_count > 0
                or parseable_score >= 50
                or likely_shop
                or (is_network_or_market and (quote_post_count >= 1 or parseable_score >= 35))
            )
            if not keep:
                continue

            if status.startswith("error"):
                continue

            public_url = str(row.get("public_url", "")).strip()
            title = str(row.get("title", "")).strip()
            if handle in discovered_map:
                if not public_url:
                    public_url = discovered_map[handle].get("public_url", "")
                if not title:
                    title = discovered_map[handle].get("title", "")

            ch = CandidateChannel(
                handle=handle,
                public_url=normalize_public_url(handle, public_url),
                title=title,
                channel_type_guess=channel_type,
                likely_individual_shop=likely_shop,
                parseable_score=parseable_score,
                quote_post_count=quote_post_count,
            )

            priority_seed = (
                parseable_score
                + quote_post_count * 4
                + (14 if likely_shop else 0)
                + (8 if is_network_or_market else 0)
            )
            candidates.append((float(priority_seed), ch))

    candidates.sort(key=lambda x: (-x[0], x[1].handle))
    selected = [x[1] for x in candidates]
    if max_channels > 0:
        selected = selected[:max_channels]
    return selected


def parse_number(token: str) -> Optional[int]:
    cleaned = translit_digits(token)
    cleaned = re.sub(r"[^0-9]", "", cleaned)
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def detect_currencies(text: str) -> List[str]:
    lowered = text.lower()
    currencies: List[str] = []
    for code, aliases in CURRENCY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                currencies.append(code)
                break
    return sorted(set(currencies))


def detect_city_mentions(text: str) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for city, aliases in CITY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                hits.append(city)
                break
    return sorted(set(hits))


def extract_numbers(text: str) -> List[int]:
    out: List[int] = []
    normalized = translit_digits(text)
    for m in QUOTE_NUMBER_RE.finditer(normalized):
        value = parse_number(m.group(0))
        if value is not None:
            out.append(value)
    return out


def find_keyword_number(text: str, words: Sequence[str]) -> Optional[int]:
    normalized = translit_digits(text).lower()
    for word in words:
        pattern1 = re.compile(rf"{re.escape(word)}[^0-9]{{0,24}}({QUOTE_NUMBER_RE.pattern})", re.IGNORECASE)
        m1 = pattern1.search(normalized)
        if m1:
            value = parse_number(m1.group(1))
            if value is not None:
                return value

        pattern2 = re.compile(rf"({QUOTE_NUMBER_RE.pattern})[^0-9]{{0,24}}{re.escape(word)}", re.IGNORECASE)
        m2 = pattern2.search(normalized)
        if m2:
            value = parse_number(m2.group(1))
            if value is not None:
                return value
    return None


def detect_slash_pair(text: str) -> Tuple[Optional[int], Optional[int]]:
    normalized = translit_digits(text)
    m = SLASH_PAIR_RE.search(normalized)
    if not m:
        return None, None
    a = parse_number(m.group(1))
    b = parse_number(m.group(2))
    return a, b


def message_fingerprint(text: str) -> str:
    normalized = translit_digits(text).lower()
    normalized = re.sub(QUOTE_NUMBER_RE.pattern, "#", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:140]


def classify_quote_origin(
    message_text: str,
    likely_individual_shop: bool,
    channel_type_guess: str,
) -> str:
    lowered = message_text.lower()
    has_shop_hint = any(word in lowered for word in SHOP_HINT_WORDS)
    has_network_hint = any(word in lowered for word in NETWORK_HINT_WORDS)

    if likely_individual_shop and not has_network_hint:
        return "direct_shop_quote"
    if channel_type_guess in ("dealer_network_channel", "market_price_channel") or has_network_hint:
        return "network_market_quote"
    if has_shop_hint:
        return "direct_shop_quote"
    return "unclear"


def parse_quote_message(
    text: str,
    handle: str,
    channel_type_guess: str,
    likely_individual_shop: bool,
) -> Optional[Dict[str, object]]:
    normalized = translit_digits(text)
    lowered = normalized.lower()

    currencies = detect_currencies(normalized)
    numbers = extract_numbers(normalized)
    buy = find_keyword_number(normalized, BUY_WORDS)
    sell = find_keyword_number(normalized, SELL_WORDS)
    slash_a, slash_b = detect_slash_pair(normalized)

    if buy is None and sell is None and slash_a is not None and slash_b is not None:
        buy, sell = slash_a, slash_b

    if buy is None and sell is not None and slash_a is not None:
        buy = slash_a
    if sell is None and buy is not None and slash_b is not None:
        sell = slash_b

    if buy is None and sell is None and len(numbers) >= 2 and any(w in lowered for w in BUY_WORDS + SELL_WORDS):
        buy, sell = numbers[0], numbers[1]

    has_quote_word = any(w in lowered for w in QUOTE_WORDS)
    has_buy_sell_word = any(w in lowered for w in BUY_WORDS + SELL_WORDS)
    has_city = bool(detect_city_mentions(normalized))

    is_quote_like = (
        (len(numbers) >= 1 and (bool(currencies) or has_quote_word or has_buy_sell_word or has_city))
        or (buy is not None and sell is not None)
        or (slash_a is not None and slash_b is not None)
    )

    if not is_quote_like:
        return None

    midpoint: Optional[float] = None
    if buy is not None and sell is not None and buy > 0 and sell > 0:
        midpoint = (buy + sell) / 2.0

    cities = detect_city_mentions(normalized)
    origin_guess = classify_quote_origin(normalized, likely_individual_shop, channel_type_guess)

    sample = normalized
    if len(sample) > 280:
        sample = sample[:280] + "..."

    return {
        "handle": handle,
        "message_text_sample": sample,
        "detected_currencies": currencies,
        "detected_buy_quote": buy,
        "detected_sell_quote": sell,
        "detected_midpoint": midpoint,
        "city_mentions": cities,
        "quote_origin_guess": origin_guess,
        "number_count": len(numbers),
    }


def analyze_channel(
    candidate: CandidateChannel,
    timeout: int,
) -> Tuple[ChannelAnalysis, Dict[str, object]]:
    page, status, err = fetch_url(candidate.public_url, timeout=timeout)
    status_label = "ok"
    if page is None:
        status_label = f"error:{err or 'fetch_failed'}"

    title = candidate.title
    quote_records: List[Dict[str, object]] = []
    total_messages_seen = 0

    if page is not None:
        page_title = extract_meta_content(page, "og:title") or extract_meta_content(page, "twitter:title")
        if page_title:
            title = page_title

        message_blocks = extract_message_blocks(page)
        total_messages_seen = len(message_blocks)
        for msg in message_blocks:
            parsed = parse_quote_message(
                text=msg,
                handle=candidate.handle,
                channel_type_guess=candidate.channel_type_guess,
                likely_individual_shop=candidate.likely_individual_shop,
            )
            if parsed:
                quote_records.append(parsed)

        if total_messages_seen == 0:
            status_label = "ok_no_text"

    extracted_quote_messages = len(quote_records)
    extracted_usd_quotes = sum(1 for x in quote_records if "USD" in x.get("detected_currencies", []))
    extracted_eur_quotes = sum(1 for x in quote_records if "EUR" in x.get("detected_currencies", []))
    extracted_buy_sell_pairs = sum(
        1 for x in quote_records if x.get("detected_buy_quote") is not None and x.get("detected_sell_quote") is not None
    )

    cities: List[str] = []
    for rec in quote_records:
        for city in rec.get("city_mentions", []):
            if city not in cities:
                cities.append(city)

    fingerprints = [message_fingerprint(str(x.get("message_text_sample", ""))) for x in quote_records]
    format_stability = 0.0
    if fingerprints:
        counts: Dict[str, int] = {}
        for fp in fingerprints:
            counts[fp] = counts.get(fp, 0) + 1
        format_stability = max(counts.values()) / max(1, len(fingerprints))

    signal_richness = (
        35.0 * min(1.0, extracted_quote_messages / 20.0)
        + 15.0 * min(1.0, extracted_usd_quotes / 15.0)
        + 10.0 * min(1.0, extracted_eur_quotes / 10.0)
        + 20.0 * min(1.0, extracted_buy_sell_pairs / 12.0)
        + 10.0 * min(1.0, format_stability)
        + 5.0 * min(1.0, len(cities) / 5.0)
    )
    if candidate.likely_individual_shop and extracted_buy_sell_pairs > 0:
        signal_richness += 5.0

    signal_richness_score = int(max(0, min(100, round(signal_richness))))

    priority_score = float(signal_richness_score)
    if candidate.likely_individual_shop:
        priority_score += 12.0
    if candidate.channel_type_guess in ("dealer_network_channel", "market_price_channel"):
        priority_score += 8.0
    if format_stability >= 0.60:
        priority_score += 8.0
    if extracted_quote_messages >= 8:
        priority_score += 6.0
    if extracted_buy_sell_pairs >= 5:
        priority_score += 5.0
    if len(cities) >= 2 and candidate.channel_type_guess in ("dealer_network_channel", "market_price_channel"):
        priority_score += 4.0

    if priority_score >= 85:
        recommended_priority = "P1"
    elif priority_score >= 70:
        recommended_priority = "P2"
    elif priority_score >= 55:
        recommended_priority = "P3"
    else:
        recommended_priority = "P4"

    note_bits: List[str] = []
    if extracted_quote_messages >= 8:
        note_bits.append("repeated quote messages")
    if extracted_buy_sell_pairs >= 3:
        note_bits.append("buy/sell pairs present")
    if format_stability >= 0.6:
        note_bits.append("stable post format")
    if candidate.likely_individual_shop:
        note_bits.append("likely direct shop")
    if candidate.channel_type_guess in ("dealer_network_channel", "market_price_channel") and len(cities) >= 2:
        note_bits.append("multi-city coverage")
    if not note_bits and status_label.startswith("error"):
        note_bits.append("crawl failed")
    if not note_bits:
        note_bits.append("limited structured signals")

    analysis = ChannelAnalysis(
        handle=candidate.handle,
        public_url=candidate.public_url,
        title=title,
        channel_type_guess=candidate.channel_type_guess,
        likely_individual_shop=candidate.likely_individual_shop,
        parseable_score=candidate.parseable_score,
        extracted_quote_messages=extracted_quote_messages,
        extracted_usd_quotes=extracted_usd_quotes,
        extracted_eur_quotes=extracted_eur_quotes,
        extracted_buy_sell_pairs=extracted_buy_sell_pairs,
        city_mentions=sorted(cities),
        signal_richness_score=signal_richness_score,
        recommended_priority=recommended_priority,
        priority_score=priority_score,
        format_stability=format_stability,
        notes="; ".join(note_bits),
        status=status_label,
    )

    sample_payload = {
        "handle": candidate.handle,
        "title": title,
        "public_url": candidate.public_url,
        "channel_type_guess": candidate.channel_type_guess,
        "likely_individual_shop": candidate.likely_individual_shop,
        "status": status_label,
        "crawled_at": now_iso(),
        "total_messages_seen": total_messages_seen,
        "extracted_quote_messages": extracted_quote_messages,
        "quote_message_records": quote_records,
    }

    return analysis, sample_payload


def write_quote_signal_survey(path: Path, rows: Sequence[ChannelAnalysis]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "title",
                "channel_type_guess",
                "likely_individual_shop",
                "parseable_score",
                "extracted_quote_messages",
                "extracted_usd_quotes",
                "extracted_eur_quotes",
                "extracted_buy_sell_pairs",
                "city_mentions",
                "signal_richness_score",
                "recommended_priority",
                "notes",
                "status",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x.handle):
            writer.writerow(
                {
                    "handle": row.handle,
                    "title": row.title,
                    "channel_type_guess": row.channel_type_guess,
                    "likely_individual_shop": row.likely_individual_shop,
                    "parseable_score": row.parseable_score,
                    "extracted_quote_messages": row.extracted_quote_messages,
                    "extracted_usd_quotes": row.extracted_usd_quotes,
                    "extracted_eur_quotes": row.extracted_eur_quotes,
                    "extracted_buy_sell_pairs": row.extracted_buy_sell_pairs,
                    "city_mentions": "|".join(row.city_mentions),
                    "signal_richness_score": row.signal_richness_score,
                    "recommended_priority": row.recommended_priority,
                    "notes": row.notes,
                    "status": row.status,
                }
            )


def write_priority_candidates(path: Path, rows: Sequence[ChannelAnalysis]) -> None:
    ordered = sorted(rows, key=lambda x: (-x.priority_score, -x.signal_richness_score, x.handle))
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "rank",
                "handle",
                "title",
                "recommended_priority",
                "priority_score",
                "signal_richness_score",
                "channel_type_guess",
                "likely_individual_shop",
                "extracted_quote_messages",
                "extracted_buy_sell_pairs",
                "city_mentions",
                "notes",
            ],
        )
        writer.writeheader()
        rank = 1
        for row in ordered:
            writer.writerow(
                {
                    "rank": rank,
                    "handle": row.handle,
                    "title": row.title,
                    "recommended_priority": row.recommended_priority,
                    "priority_score": f"{row.priority_score:.2f}",
                    "signal_richness_score": row.signal_richness_score,
                    "channel_type_guess": row.channel_type_guess,
                    "likely_individual_shop": row.likely_individual_shop,
                    "extracted_quote_messages": row.extracted_quote_messages,
                    "extracted_buy_sell_pairs": row.extracted_buy_sell_pairs,
                    "city_mentions": "|".join(row.city_mentions),
                    "notes": row.notes,
                }
            )
            rank += 1


def compute_summary(rows: Sequence[ChannelAnalysis]) -> Dict[str, object]:
    channels_analyzed = len(rows)
    channels_with_multiple_quote_messages = sum(1 for x in rows if x.extracted_quote_messages >= 2)
    total_quote_messages_detected = sum(x.extracted_quote_messages for x in rows)
    total_buy_sell_pairs_detected = sum(x.extracted_buy_sell_pairs for x in rows)

    likely_high_value_individual_shops = sum(
        1
        for x in rows
        if x.likely_individual_shop and x.recommended_priority in ("P1", "P2") and x.extracted_quote_messages >= 3
    )

    likely_high_value_network_channels = sum(
        1
        for x in rows
        if x.channel_type_guess in ("dealer_network_channel", "market_price_channel")
        and x.recommended_priority in ("P1", "P2")
        and x.extracted_quote_messages >= 3
    )

    ordered = sorted(rows, key=lambda x: (-x.priority_score, -x.signal_richness_score, x.handle))
    top_20 = []
    for row in ordered[:20]:
        top_20.append(
            {
                "handle": row.handle,
                "title": row.title,
                "recommended_priority": row.recommended_priority,
                "priority_score": round(row.priority_score, 2),
                "signal_richness_score": row.signal_richness_score,
                "extracted_quote_messages": row.extracted_quote_messages,
                "extracted_buy_sell_pairs": row.extracted_buy_sell_pairs,
                "channel_type_guess": row.channel_type_guess,
                "likely_individual_shop": row.likely_individual_shop,
            }
        )

    if total_quote_messages_detected < 80:
        universe_bucket = "few_dozen_signals"
    elif total_quote_messages_detected < 250:
        universe_bucket = "around_100_signals"
    else:
        universe_bucket = "several_hundred_signals"

    return {
        "generated_at": now_iso(),
        "channels_analyzed": channels_analyzed,
        "channels_with_multiple_quote_messages": channels_with_multiple_quote_messages,
        "total_quote_messages_detected": total_quote_messages_detected,
        "total_buy_sell_pairs_detected": total_buy_sell_pairs_detected,
        "likely_high_value_individual_shops": likely_high_value_individual_shops,
        "likely_high_value_network_channels": likely_high_value_network_channels,
        "top_20_priority_candidates": top_20,
        "estimated_quote_signal_universe": universe_bucket,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram quote-signal survey for RialWatch.")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Directory with first-pass survey outputs")
    parser.add_argument("--out-dir", default="survey_outputs", help="Output directory for quote-signal outputs")
    parser.add_argument("--max-channels", type=int, default=220, help="Max candidate channels to analyze (0=all)")
    parser.add_argument("--timeout", type=int, default=18, help="HTTP timeout in seconds")
    parser.add_argument("--sleep-seconds", type=float, default=0.55, help="Sleep between channel requests")
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_survey_csv = survey_dir / "channel_survey.csv"
    discovered_json = survey_dir / "discovered_channels.json"

    if not channel_survey_csv.exists():
        raise SystemExit(f"missing input file: {channel_survey_csv}")
    if not discovered_json.exists():
        raise SystemExit(f"missing input file: {discovered_json}")

    print("[1/4] Loading first-pass survey files...")
    discovered_map = load_discovered_channels(discovered_json)
    candidates = build_candidate_channels(
        channel_survey_csv=channel_survey_csv,
        discovered_map=discovered_map,
        max_channels=args.max_channels,
    )
    print(f"Selected candidate channels: {len(candidates)}")

    print("[2/4] Crawling candidate channels and extracting quote messages...")
    analyses: List[ChannelAnalysis] = []
    sample_payload: List[Dict[str, object]] = []

    for idx, candidate in enumerate(candidates, start=1):
        analysis, sample = analyze_channel(candidate, timeout=args.timeout)
        analyses.append(analysis)
        sample_payload.append(sample)

        if idx % 20 == 0:
            print(f"  Processed {idx}/{len(candidates)} channels")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print("[3/4] Writing output files...")
    quote_signal_survey_path = out_dir / "quote_signal_survey.csv"
    quote_message_samples_path = out_dir / "quote_message_samples.json"
    quote_signal_summary_path = out_dir / "quote_signal_summary.json"
    priority_candidates_path = out_dir / "priority_ingestion_candidates.csv"

    write_quote_signal_survey(quote_signal_survey_path, analyses)
    with quote_message_samples_path.open("w", encoding="utf-8") as fp:
        json.dump(sample_payload, fp, ensure_ascii=False, indent=2)

    write_priority_candidates(priority_candidates_path, analyses)

    summary = compute_summary(analyses)
    with quote_signal_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("[4/4] Completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
