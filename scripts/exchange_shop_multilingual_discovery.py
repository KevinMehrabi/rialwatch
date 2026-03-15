#!/usr/bin/env python3
"""Multilingual discovery of exchange-shop and dealer quote sources for RialWatch.

Discovery only. This script searches public web results, normalizes Telegram and
website candidates, classifies them, merges against the current registry, and
exports a multilingual discovery survey.
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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

QUERY_GROUPS: Dict[str, List[str]] = {
    "persian": [
        "site:t.me صرافی",
        "site:t.me/s صرافی",
        "صرافی تهران",
        "صرافی دبی",
        "صرافی استانبول",
        "صرافی لندن",
        "صرافی فرانکفورت",
        "نرخ دلار تهران",
        "نرخ دلار هرات",
        "نرخ دلار سلیمانیه",
        "نرخ دلار دبی",
        "قیمت دلار امروز",
        "بازار ارز",
    ],
    "english": [
        "iranian exchange dubai telegram",
        "iranian exchange istanbul telegram",
        "iranian exchange london telegram",
        "iranian exchange frankfurt telegram",
        "iranian currency exchange dubai",
        "iranian currency exchange turkey",
        "iranian money exchange london",
        "iranian remittance exchange dubai",
    ],
    "turkish": [
        "iran döviz bürosu istanbul",
        "iran para transferi istanbul",
        "iran para değişimi istanbul",
        "iran döviz istanbul",
    ],
    "arabic": [
        "صرافة ايرانية دبي",
        "تحويل اموال ايران دبي",
        "صرافة دبي ايران",
        "تحويل عملة ايران دبي",
    ],
    "german": [
        "iran wechselstube frankfurt",
        "iran geldwechsel hamburg",
        "iran geldtransfer frankfurt",
        "iran wechselstube deutschland",
    ],
}

EXCLUDED_HANDLES = {
    "iv",
    "share",
    "joinchat",
    "addstickers",
    "proxy",
    "s",
    "login",
    "contact",
}

EXCLUDED_DOMAINS = {
    "duckduckgo.com",
    "www.duckduckgo.com",
    "lite.duckduckgo.com",
    "r.jina.ai",
    "google.com",
    "www.google.com",
    "maps.google.com",
    "youtube.com",
    "www.youtube.com",
    "instagram.com",
    "www.instagram.com",
    "facebook.com",
    "www.facebook.com",
    "x.com",
    "twitter.com",
    "www.twitter.com",
    "linkedin.com",
    "www.linkedin.com",
}

HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{5,}$")
RAW_TME_RE = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
BARE_TME_RE = re.compile(r"(?:^|\s)(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d|09\d{9}|\+98\s?9\d{9})")
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,7})(?!\d)")

BUY_WORDS = ("خرید", "buy", "bid", "alış", "ankauf", "شراء")
SELL_WORDS = ("فروش", "sell", "offer", "ask", "satış", "verkauf", "بيع")

SHOP_KEYWORDS = (
    "صرافی",
    "exchange",
    "sarafi",
    "currency exchange",
    "wechselstube",
    "döviz bürosu",
    "geldwechsel",
    "صرافة",
)
SETTLEMENT_KEYWORDS = (
    "حواله",
    "remittance",
    "money transfer",
    "settlement",
    "para transferi",
    "geldtransfer",
    "تحويل",
    "حواله جات",
)
MARKET_KEYWORDS = (
    "نرخ",
    "قیمت",
    "market",
    "dealer",
    "بازار ارز",
    "tehran",
    "herat",
    "dubai",
    "istanbul",
    "sulaymaniyah",
    "regional",
)
AGGREGATOR_KEYWORDS = (
    "analysis",
    "news",
    "اخبار",
    "signal",
    "سیگنال",
    "تحلیل",
    "price board",
)
ADDRESS_KEYWORDS = (
    "آدرس",
    "خیابان",
    "بلوار",
    "کوچه",
    "پلاک",
    "واحد",
    "address",
    "street",
    "road",
    "suite",
    "adres",
    "cadde",
    "sokak",
    "adresse",
)

CITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
    "Sulaymaniyah": ("sulaymaniyah", "sulaimaniyah", "سلیمانیه", "سليمانية"),
    "Herat": ("herat", "هرات"),
    "London": ("london", "لندن"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
    "Hamburg": ("hamburg", "هامبورگ"),
}

COUNTRY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Iran": ("iran", "iranian", "ایران", "تهران"),
    "UAE": ("uae", "dubai", "امارات", "دبی", "دوبی"),
    "Turkey": ("turkey", "turkish", "istanbul", "ترکیه", "استانبول"),
    "Iraq": ("iraq", "sulaymaniyah", "sulaimaniyah", "عراق", "سلیمانیه", "سليمانية"),
    "Afghanistan": ("afghanistan", "herat", "افغانستان", "هرات"),
    "UK": ("uk", "united kingdom", "britain", "london", "انگلیس", "بریتانیا", "لندن"),
    "Germany": ("germany", "deutschland", "frankfurt", "hamburg", "آلمان", "فرانکفورت", "هامبورگ"),
}

CITY_TO_COUNTRY = {
    "Tehran": "Iran",
    "Dubai": "UAE",
    "Istanbul": "Turkey",
    "Sulaymaniyah": "Iraq",
    "Herat": "Afghanistan",
    "London": "UK",
    "Frankfurt": "Germany",
    "Hamburg": "Germany",
}

SHOP_NAME_STOPWORDS = {
    "telegram",
    "contact",
    "channel",
    "exchange",
    "sarafi",
    "صرافی",
    "currency",
    "remittance",
    "حواله",
    "ltd",
    "limited",
    "company",
}


@dataclass
class DiscoveryCandidate:
    key: str
    platform: str
    telegram_handle: Optional[str] = None
    telegram_url: Optional[str] = None
    website_url: Optional[str] = None
    raw_urls: Set[str] = field(default_factory=set)
    query_hits: Set[str] = field(default_factory=set)
    query_languages: Set[str] = field(default_factory=set)


@dataclass
class CandidateRecord:
    handle_or_url: str
    title: str
    platform: str
    country_guess: str
    city_guess: str
    language_guess: str
    source_type: str
    quote_post_count: int
    buy_sell_pair_count: int
    parseability_score: int
    likely_individual_shop: bool
    has_phone: bool
    has_address: bool
    last_seen: str
    status_guess: str


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def clean_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number_token(token: str) -> Optional[int]:
    cleaned = re.sub(r"[^0-9]", "", translit_digits(token))
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


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
        body = exc.read().decode("utf-8", errors="replace")
        return body, int(exc.code), f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return None, None, f"network_error:{exc.reason}"
    except socket.timeout:
        return None, None, "timeout"
    except TimeoutError:
        return None, None, "timeout"
    except OSError as exc:
        return None, None, f"os_error:{exc}"


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
    handle = parts[1] if parts[0].lower() == "s" and len(parts) >= 2 else parts[0]
    handle = handle.strip().lower()
    if not handle or handle.startswith("+") or handle in EXCLUDED_HANDLES:
        return None
    if not HANDLE_RE.match(handle):
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
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return f"https://{domain}{path}"


def extract_links_from_search_page(page: str) -> List[str]:
    found: List[str] = []
    unescaped = html.unescape(page)
    for blob in (unescaped, urllib.parse.unquote(unescaped)):
        for pattern in (RAW_TME_RE, BARE_TME_RE, URL_RE):
            for match in pattern.finditer(blob):
                token = match.group(0).strip()
                if token:
                    found.append(token)
    for href in re.findall(r'href=["\']([^"\']+)["\']', unescaped, flags=re.IGNORECASE):
        token = href.strip()
        if token:
            found.append(token)
    for encoded in re.findall(r"uddg=([^&\"'<>\\s]+)", unescaped, flags=re.IGNORECASE):
        decoded = urllib.parse.unquote(encoded)
        if decoded:
            found.append(decoded)
    return found


def search_urls_for_query(query: str, pages: int) -> List[str]:
    encoded = urllib.parse.quote_plus(query)
    urls: List[str] = []
    for page_idx in range(pages):
        offset = page_idx * 30
        urls.append(f"https://r.jina.ai/http://lite.duckduckgo.com/lite/?q={encoded}&s={offset}")
    return urls


def build_query_plan(max_base_queries_per_language: int = 0) -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    for language, queries in QUERY_GROUPS.items():
        active_queries = list(queries)
        if max_base_queries_per_language > 0:
            active_queries = active_queries[:max_base_queries_per_language]
        for query in active_queries:
            plan.append((language, query))
            lower = query.lower()
            if "site:t.me" not in lower:
                plan.append((language, f"site:t.me {query}"))
            if "site:t.me/s" not in lower:
                plan.append((language, f"site:t.me/s {query}"))
    # Preserve order but deduplicate exact query strings.
    deduped: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for language, query in plan:
        key = f"{language}\t{query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append((language, query))
    return deduped


def run_search_discovery(
    query_plan: Sequence[Tuple[str, str]],
    pages_per_query: int,
    timeout: int,
    request_sleep: float,
) -> Tuple[Dict[str, DiscoveryCandidate], Dict[str, object]]:
    out: Dict[str, DiscoveryCandidate] = {}
    debug: Dict[str, object] = {
        "successful_search_requests": 0,
        "failed_search_requests": 0,
        "query_stats": {},
    }

    for language, query in query_plan:
        hits = 0
        for url in search_urls_for_query(query, pages_per_query):
            page, status, err = fetch_url(url, timeout=timeout)
            if page is None or (status is not None and status >= 400):
                debug["failed_search_requests"] = int(debug["failed_search_requests"]) + 1
                continue

            debug["successful_search_requests"] = int(debug["successful_search_requests"]) + 1
            for token in extract_links_from_search_page(page):
                norm_tg = normalize_telegram_url(token)
                if norm_tg:
                    handle, public_url = norm_tg
                    key = f"telegram:{handle}"
                    candidate = out.get(key)
                    if candidate is None:
                        candidate = DiscoveryCandidate(key=key, platform="telegram", telegram_handle=handle, telegram_url=public_url)
                        out[key] = candidate
                    candidate.raw_urls.add(token)
                    candidate.query_hits.add(query)
                    candidate.query_languages.add(language)
                    hits += 1
                    continue

                norm_site = normalize_website_url(token)
                if not norm_site:
                    continue
                key = f"website:{norm_site}"
                candidate = out.get(key)
                if candidate is None:
                    candidate = DiscoveryCandidate(key=key, platform="website", website_url=norm_site)
                    out[key] = candidate
                candidate.raw_urls.add(token)
                candidate.query_hits.add(query)
                candidate.query_languages.add(language)
                hits += 1

            if request_sleep > 0:
                time.sleep(request_sleep + random.random() * request_sleep * 0.5)

        debug["query_stats"][query] = {"candidate_hits": hits, "language": language}

    return out, debug


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


def extract_message_blocks(page: str) -> List[str]:
    blocks: List[Tuple[int, str]] = []
    for pattern in (
        r'<div class="tgme_widget_message_text[^\"]*"[^>]*>(.*?)</div>',
        r'<div class="tgme_widget_message_caption[^\"]*"[^>]*>(.*?)</div>',
        r"<p[^>]*>(.*?)</p>",
        r"<li[^>]*>(.*?)</li>",
    ):
        for match in re.finditer(pattern, page, flags=re.IGNORECASE | re.DOTALL):
            text = clean_text(match.group(1))
            if text:
                blocks.append((match.start(), text))
    blocks.sort(key=lambda item: item[0])
    ordered = [text for _, text in blocks]
    seen: Set[str] = set()
    deduped: List[str] = []
    for text in ordered:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def detect_language(text: str, fallback_languages: Sequence[str]) -> str:
    lowered = text.lower()
    persian_hits = sum(lowered.count(token) for token in ("صرافی", "دلار", "بازار", "حواله", "تهران", "ایران"))
    arabic_hits = sum(lowered.count(token) for token in ("صرافة", "تحويل", "دبي", "ايران", "عملة"))
    turkish_hits = sum(lowered.count(token) for token in ("döviz", "istanbul", "para", "transferi", "değişimi"))
    german_hits = sum(lowered.count(token) for token in ("wechselstube", "geldtransfer", "geldwechsel", "deutschland"))
    english_hits = sum(lowered.count(token) for token in ("exchange", "remittance", "currency", "dealer", "market"))
    scores = {
        "Persian": persian_hits,
        "Arabic": arabic_hits,
        "Turkish": turkish_hits,
        "German": german_hits,
        "English": english_hits,
    }
    best_language = max(scores, key=scores.get)
    if scores[best_language] > 0:
        return best_language
    if len(set(fallback_languages)) == 1:
        return fallback_languages[0].capitalize()
    if fallback_languages:
        return "mixed"
    return "unknown"


def guess_city(text: str) -> str:
    lowered = text.lower()
    best_city = "unknown"
    best_score = 0
    for city, aliases in CITY_ALIASES.items():
        score = sum(lowered.count(alias.lower()) for alias in aliases)
        if score > best_score:
            best_city = city
            best_score = score
    return best_city


def guess_country(text: str, city_guess: str) -> str:
    if city_guess in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[city_guess]
    lowered = text.lower()
    best_country = "unknown"
    best_score = 0
    for country, aliases in COUNTRY_KEYWORDS.items():
        score = sum(lowered.count(alias.lower()) for alias in aliases)
        if score > best_score:
            best_country = country
            best_score = score
    return best_country


def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    lowered = translit_digits(text).lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def extract_numbers(text: str) -> List[int]:
    values: List[int] = []
    normalized = translit_digits(text)
    for match in NUMBER_RE.finditer(normalized):
        parsed = parse_number_token(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def find_keyword_number(text: str, words: Sequence[str]) -> Optional[int]:
    normalized = translit_digits(text).lower()
    pattern = NUMBER_RE.pattern
    for word in words:
        matcher = re.compile(
            rf"{re.escape(word.lower())}[^0-9]{{0,24}}({pattern})|({pattern})[^0-9]{{0,24}}{re.escape(word.lower())}",
            re.IGNORECASE,
        )
        match = matcher.search(normalized)
        if not match:
            continue
        token = match.group(1) or match.group(2)
        if token:
            parsed = parse_number_token(token)
            if parsed is not None:
                return parsed
    return None


def count_quote_messages(blocks: Sequence[str]) -> Tuple[int, int]:
    quote_count = 0
    pair_count = 0
    for block in blocks:
        lowered = translit_digits(block).lower()
        numbers = extract_numbers(block)
        has_market_word = contains_keyword(lowered, MARKET_KEYWORDS) or contains_keyword(lowered, ("usd", "eur", "دلار", "یورو"))
        buy_val = find_keyword_number(block, BUY_WORDS)
        sell_val = find_keyword_number(block, SELL_WORDS)
        if numbers and (has_market_word or (buy_val is not None and sell_val is not None)):
            quote_count += 1
        if buy_val is not None and sell_val is not None:
            pair_count += 1
    return quote_count, pair_count


def detect_source_type(
    text: str,
    quote_post_count: int,
    buy_sell_pair_count: int,
    country_guess: str,
    has_phone: bool,
    has_address: bool,
) -> Tuple[str, bool]:
    lowered = translit_digits(text).lower()
    has_shop_words = contains_keyword(lowered, SHOP_KEYWORDS)
    has_settlement_words = contains_keyword(lowered, SETTLEMENT_KEYWORDS)
    has_market_words = contains_keyword(lowered, MARKET_KEYWORDS)
    has_aggregator_words = contains_keyword(lowered, AGGREGATOR_KEYWORDS)
    multiple_regions = sum(1 for city in ("tehran", "herat", "dubai", "istanbul", "sulaymaniyah") if city in lowered) >= 2

    likely_individual_shop = bool(has_shop_words and (has_phone or has_address))
    if likely_individual_shop:
        return "exchange_shop", True
    if has_settlement_words and country_guess in {"UAE", "Turkey", "Iraq", "Afghanistan", "UK", "Germany"}:
        return "settlement_exchange", False
    if quote_post_count >= 3 and (multiple_regions or contains_keyword(lowered, ("tehran", "herat", "dubai", "istanbul", "sulaymaniyah"))):
        return "regional_market_channel", False
    if has_aggregator_words or (quote_post_count >= 3 and has_market_words and not has_phone and not has_address):
        return "aggregator", False
    if has_shop_words:
        return "exchange_shop", False
    return "unknown", False


def normalize_shop_name(title: str) -> str:
    lowered = translit_digits(title).lower()
    lowered = re.sub(r"[^a-z0-9\u0600-\u06ff]+", " ", lowered)
    parts = [part for part in lowered.split() if part and part not in SHOP_NAME_STOPWORDS]
    return " ".join(parts).strip()


def parseability_score(
    quote_post_count: int,
    buy_sell_pair_count: int,
    has_phone: bool,
    has_address: bool,
    platform: str,
    source_type: str,
) -> int:
    score = 0
    score += min(quote_post_count * 8, 40)
    score += min(buy_sell_pair_count * 10, 30)
    if has_phone:
        score += 8
    if has_address:
        score += 8
    if platform == "telegram":
        score += 6
    if source_type in {"exchange_shop", "regional_market_channel", "settlement_exchange"}:
        score += 8
    return max(0, min(score, 100))


def extract_last_seen(page: str, platform: str) -> str:
    if platform == "telegram":
        matches = re.findall(r'datetime="([^"]+)"', page)
        if matches:
            return max(matches)
    meta = extract_meta_content(page, "article:published_time") or extract_meta_content(page, "og:updated_time")
    return meta or ""


def crawl_candidate(candidate: DiscoveryCandidate, timeout: int) -> CandidateRecord:
    target_url = candidate.telegram_url if candidate.platform == "telegram" else candidate.website_url
    page, status, err = fetch_url(target_url or "", timeout=timeout)
    status_guess = "ok"
    if page is None:
        status_guess = err or "fetch_failed"
        return CandidateRecord(
            handle_or_url=candidate.telegram_handle or candidate.website_url or candidate.key,
            title="",
            platform=candidate.platform,
            country_guess="unknown",
            city_guess="unknown",
            language_guess=detect_language("", sorted(candidate.query_languages)),
            source_type="unknown",
            quote_post_count=0,
            buy_sell_pair_count=0,
            parseability_score=0,
            likely_individual_shop=False,
            has_phone=False,
            has_address=False,
            last_seen="",
            status_guess=status_guess,
        )
    if status is not None and status >= 400:
        status_guess = err or f"http_{status}"

    title = ""
    if candidate.platform == "telegram":
        title = extract_meta_content(page, "og:title") or extract_page_title(page)
    else:
        title = extract_page_title(page) or extract_meta_content(page, "og:title")

    meta_desc = extract_meta_content(page, "description") or extract_meta_content(page, "og:description")
    blocks = extract_message_blocks(page)
    joined_text = " ".join(filter(None, [title, meta_desc] + blocks[:25]))
    quote_post_count, buy_sell_pair_count = count_quote_messages(blocks[:25])
    has_phone = bool(PHONE_RE.search(joined_text))
    has_address = contains_keyword(joined_text, ADDRESS_KEYWORDS)
    city_guess = guess_city(joined_text)
    country_guess = guess_country(joined_text, city_guess)
    source_type, likely_individual_shop = detect_source_type(
        joined_text,
        quote_post_count=quote_post_count,
        buy_sell_pair_count=buy_sell_pair_count,
        country_guess=country_guess,
        has_phone=has_phone,
        has_address=has_address,
    )
    language_guess = detect_language(joined_text, sorted(candidate.query_languages))
    score = parseability_score(
        quote_post_count=quote_post_count,
        buy_sell_pair_count=buy_sell_pair_count,
        has_phone=has_phone,
        has_address=has_address,
        platform=candidate.platform,
        source_type=source_type,
    )
    if not blocks and status_guess == "ok":
        status_guess = "ok_no_text"

    return CandidateRecord(
        handle_or_url=candidate.telegram_handle or candidate.website_url or candidate.key,
        title=title,
        platform=candidate.platform,
        country_guess=country_guess,
        city_guess=city_guess,
        language_guess=language_guess,
        source_type=source_type,
        quote_post_count=quote_post_count,
        buy_sell_pair_count=buy_sell_pair_count,
        parseability_score=score,
        likely_individual_shop=likely_individual_shop,
        has_phone=has_phone,
        has_address=has_address,
        last_seen=extract_last_seen(page, candidate.platform),
        status_guess=status_guess,
    )


def load_existing_registry(path: Path) -> Tuple[Set[str], Set[str], Set[str], int]:
    handles: Set[str] = set()
    urls: Set[str] = set()
    names: Set[str] = set()
    total = 0
    if not path.exists():
        return handles, urls, names, total
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            handle = str(row.get("handle", "")).strip().lower()
            if handle:
                handles.add(handle)
            public_url = str(row.get("public_url", "")).strip().lower()
            if public_url:
                urls.add(public_url)
            normalized_name = normalize_shop_name(str(row.get("title", "")))
            if normalized_name:
                names.add(normalized_name)
    return handles, urls, names, total


def filter_new_records(records: Sequence[CandidateRecord], existing_registry: Tuple[Set[str], Set[str], Set[str], int]) -> List[CandidateRecord]:
    existing_handles, existing_urls, existing_names, _ = existing_registry
    new_records: List[CandidateRecord] = []
    seen_keys: Set[str] = set()
    seen_names: Set[str] = set()
    for record in sorted(records, key=lambda item: (item.platform, item.handle_or_url)):
        key = f"{record.platform}:{record.handle_or_url.lower()}"
        if key in seen_keys:
            continue
        if record.platform == "telegram" and record.handle_or_url.lower() in existing_handles:
            continue
        if record.platform == "website":
            canonical_url = normalize_website_url(record.handle_or_url) or record.handle_or_url.lower()
            if canonical_url in existing_urls:
                continue
        normalized_name = normalize_shop_name(record.title)
        if normalized_name and (normalized_name in existing_names or normalized_name in seen_names):
            continue
        seen_keys.add(key)
        if normalized_name:
            seen_names.add(normalized_name)
        new_records.append(record)
    return new_records


def summarize_records(records: Sequence[CandidateRecord], existing_total: int) -> Dict[str, object]:
    def count_if(source_type: str) -> int:
        return sum(1 for record in records if record.source_type == source_type)

    by_country: Dict[str, int] = {}
    by_language: Dict[str, int] = {}
    for record in records:
        by_country[record.country_guess] = by_country.get(record.country_guess, 0) + 1
        by_language[record.language_guess] = by_language.get(record.language_guess, 0) + 1

    potential_publishable_sources = sum(
        1 for record in records if record.parseability_score >= 55 and record.source_type in {"exchange_shop", "settlement_exchange", "regional_market_channel"}
    )
    potential_additional_records = sum(
        max(1, min(record.quote_post_count, 5) + record.buy_sell_pair_count)
        for record in records
        if record.parseability_score >= 55
    )

    return {
        "generated_at": now_iso(),
        "new_sources_discovered": len(records),
        "total_sources_in_registry": existing_total + len(records),
        "likely_exchange_shops": count_if("exchange_shop"),
        "regional_market_channels": count_if("regional_market_channel"),
        "settlement_exchanges": count_if("settlement_exchange"),
        "sources_by_country": dict(sorted(by_country.items())),
        "sources_by_language": dict(sorted(by_language.items())),
        "potential_publishable_sources": potential_publishable_sources,
        "potential_additional_records": potential_additional_records,
    }


def write_candidates_csv(path: Path, records: Sequence[CandidateRecord]) -> None:
    fieldnames = [
        "handle_or_url",
        "title",
        "platform",
        "country_guess",
        "city_guess",
        "language_guess",
        "source_type",
        "quote_post_count",
        "buy_sell_pair_count",
        "parseability_score",
        "likely_individual_shop",
        "has_phone",
        "has_address",
        "last_seen",
        "status_guess",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.source_type, item.country_guess, item.platform, item.handle_or_url.lower())):
            writer.writerow(
                {
                    "handle_or_url": record.handle_or_url,
                    "title": record.title,
                    "platform": record.platform,
                    "country_guess": record.country_guess,
                    "city_guess": record.city_guess,
                    "language_guess": record.language_guess,
                    "source_type": record.source_type,
                    "quote_post_count": record.quote_post_count,
                    "buy_sell_pair_count": record.buy_sell_pair_count,
                    "parseability_score": record.parseability_score,
                    "likely_individual_shop": record.likely_individual_shop,
                    "has_phone": record.has_phone,
                    "has_address": record.has_address,
                    "last_seen": record.last_seen,
                    "status_guess": record.status_guess,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multilingual exchange-shop and dealer discovery for RialWatch")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Output directory for survey artifacts")
    parser.add_argument(
        "--max-base-queries-per-language",
        type=int,
        default=0,
        help="Limit the number of base queries used per language before site:t.me expansion (0 = all)",
    )
    parser.add_argument("--pages-per-query", type=int, default=1, help="DuckDuckGo lite pages per query")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.45, help="Base sleep between network requests")
    parser.add_argument("--max-candidates", type=int, default=350, help="Maximum candidates to crawl after search dedupe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    survey_dir = Path(args.survey_dir)
    if not survey_dir.is_absolute():
        survey_dir = root / survey_dir
    survey_dir.mkdir(parents=True, exist_ok=True)

    existing_registry_path = survey_dir / "channel_survey.csv"
    candidates_csv = survey_dir / "exchange_shop_multilingual_candidates.csv"
    summary_json = survey_dir / "exchange_shop_multilingual_summary.json"

    query_plan = build_query_plan(max_base_queries_per_language=args.max_base_queries_per_language)
    discovered, debug = run_search_discovery(
        query_plan=query_plan,
        pages_per_query=args.pages_per_query,
        timeout=args.timeout,
        request_sleep=args.sleep,
    )

    ordered_candidates = sorted(discovered.values(), key=lambda item: (item.platform, item.key))
    if args.max_candidates > 0:
        ordered_candidates = ordered_candidates[: args.max_candidates]

    records: List[CandidateRecord] = []
    for candidate in ordered_candidates:
        records.append(crawl_candidate(candidate, timeout=args.timeout))
        if args.sleep > 0:
            time.sleep(args.sleep + random.random() * args.sleep * 0.5)

    existing_registry = load_existing_registry(existing_registry_path)
    new_records = filter_new_records(records, existing_registry)
    summary = summarize_records(new_records, existing_total=existing_registry[3])
    summary["search_debug"] = debug
    summary["queries_run"] = len(query_plan)
    summary["candidates_crawled"] = len(ordered_candidates)

    write_candidates_csv(candidates_csv, new_records)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"queries_run={len(query_plan)}")
    print(f"candidates_crawled={len(ordered_candidates)}")
    print(f"new_sources_discovered={summary['new_sources_discovered']}")
    print(f"total_sources_in_registry={summary['total_sources_in_registry']}")
    print(f"likely_exchange_shops={summary['likely_exchange_shops']}")
    print(f"regional_market_channels={summary['regional_market_channels']}")
    print(f"settlement_exchanges={summary['settlement_exchanges']}")
    print(f"potential_publishable_sources={summary['potential_publishable_sources']}")
    print(f"potential_additional_records={summary['potential_additional_records']}")
    print(f"candidates_csv={candidates_csv}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
