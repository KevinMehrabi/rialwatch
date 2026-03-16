#!/usr/bin/env python3
"""Business-oriented exchange-shop discovery for RialWatch.

This discovery pass looks beyond Telegram-first queries and searches public web
results for exchange businesses, remittance shops, directory listings, and
social profiles across the Iranian FX ecosystem. It remains research-only.
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

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

TARGET_LOCALITIES: Dict[str, Tuple[str, str]] = {
    "Iran": ("Iran", ""),
    "UAE": ("UAE", "Dubai"),
    "Turkey": ("Turkey", "Istanbul"),
    "UK": ("UK", "London"),
    "Germany_Frankfurt": ("Germany", "Frankfurt"),
    "Germany_Hamburg": ("Germany", "Hamburg"),
    "Iraq": ("Iraq", "Sulaymaniyah"),
    "Afghanistan": ("Afghanistan", "Herat"),
}

QUERY_GROUPS: Dict[str, List[str]] = {
    "maps": [
        "iranian exchange dubai google maps",
        "iranian exchange istanbul google maps",
        "iranian exchange london google maps",
        "iranian exchange frankfurt google maps",
        "iranian exchange hamburg google maps",
        "iranian exchange sulaymaniyah google maps",
        "iranian exchange herat google maps",
        "صرافی دبی گوگل مپ",
        "صرافی استانبول گوگل مپ",
        "صرافی لندن گوگل مپ",
    ],
    "instagram": [
        "site:instagram.com صرافی دبی",
        "site:instagram.com صرافی استانبول",
        "site:instagram.com صرافی لندن",
        "site:instagram.com sarafi dubai",
        "site:instagram.com iran exchange istanbul",
        "site:instagram.com currency exchange iran dubai",
    ],
    "directories": [
        "iranian exchange dubai directory",
        "iranian exchange istanbul directory",
        "iranian exchange london directory",
        "iranian exchange frankfurt directory",
        "iranian exchange hamburg directory",
        "iranian remittance dubai directory",
        "currency exchange herat directory",
        "currency exchange sulaymaniyah directory",
    ],
    "general": [
        "iranian exchange dubai",
        "iranian exchange istanbul",
        "iranian exchange london",
        "iranian exchange frankfurt",
        "iranian exchange hamburg",
        "iranian remittance dubai",
        "صرافی دبی",
        "صرافی استانبول",
        "صرافی لندن",
        "صرافی فرانکفورت",
        "صرافی هامبورگ",
        "صرافی سلیمانیه",
        "صرافی هرات",
    ],
}

EXCLUDED_DOMAINS = {
    "r.jina.ai",
    "duckduckgo.com",
    "lite.duckduckgo.com",
    "search.brave.com",
    "google.com",
    "www.google.com",
    "maps.google.com",
    "wa.me",
    "api.whatsapp.com",
    "facebook.com",
    "www.facebook.com",
    "youtube.com",
    "www.youtube.com",
    "linkedin.com",
    "www.linkedin.com",
    "x.com",
    "twitter.com",
    "www.twitter.com",
}

HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{3,64}$")
RAW_TME_RE = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
RAW_IG_RE = re.compile(r"https?://(?:www\.)?instagram\.com/[^\s\"'<>]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d|09\d{9}|\+98\s?9\d{9}|\+44\s?\d[\d\s]{7,}|\+90\s?\d[\d\s]{7,})")
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")

SHOP_KEYWORDS = (
    "صرافی",
    "exchange",
    "sarafi",
    "currency exchange",
    "wechselstube",
    "doviz",
    "döviz",
    "money exchange",
)
REMITTANCE_KEYWORDS = (
    "حواله",
    "remittance",
    "money transfer",
    "cash transfer",
    "settlement",
    "hawala",
    "geldtransfer",
    "para transferi",
    "تحويل",
)
RATE_KEYWORDS = (
    "نرخ",
    "قیمت",
    "rate",
    "quote",
    "fx",
    "usd",
    "eur",
    "دلار",
    "یورو",
    "buy",
    "sell",
    "خرید",
    "فروش",
)
REGIONAL_MARKET_KEYWORDS = (
    "tehran",
    "herat",
    "dubai",
    "istanbul",
    "sulaymaniyah",
    "تهران",
    "هرات",
    "دبی",
    "استانبول",
    "سلیمانیه",
)
ADDRESS_KEYWORDS = (
    "address",
    "street",
    "road",
    "suite",
    "office",
    "آدرس",
    "خیابان",
    "بلوار",
    "کوچه",
    "پلاک",
    "واحد",
    "cadde",
    "sokak",
)
SOCIAL_CTA_KEYWORDS = ("instagram", "telegram", "whatsapp", "تماس", "contact", "call")
SHOP_NAME_STOPWORDS = {
    "exchange",
    "currency",
    "iranian",
    "iran",
    "money",
    "remittance",
    "telegram",
    "instagram",
    "channel",
    "page",
    "llc",
    "ltd",
    "limited",
    "group",
    "صرافی",
    "حواله",
    "ایران",
}

CITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
    "London": ("london", "لندن"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
    "Hamburg": ("hamburg", "هامبورگ"),
    "Sulaymaniyah": ("sulaymaniyah", "sulaimaniyah", "سلیمانیه", "سليمانية"),
    "Herat": ("herat", "هرات"),
}

CITY_TO_COUNTRY = {
    "Tehran": "Iran",
    "Dubai": "UAE",
    "Istanbul": "Turkey",
    "London": "UK",
    "Frankfurt": "Germany",
    "Hamburg": "Germany",
    "Sulaymaniyah": "Iraq",
    "Herat": "Afghanistan",
}

COUNTRY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Iran": ("iran", "iranian", "ایران", "تهران"),
    "UAE": ("uae", "dubai", "امارات", "دبی", "دوبی"),
    "Turkey": ("turkey", "turkish", "istanbul", "ترکیه", "استانبول"),
    "UK": ("uk", "united kingdom", "london", "england", "britain", "لندن", "انگلیس"),
    "Germany": ("germany", "deutschland", "frankfurt", "hamburg", "آلمان", "فرانکفورت", "هامبورگ"),
    "Iraq": ("iraq", "sulaymaniyah", "sulaimaniyah", "عراق", "سلیمانیه", "سليمانية"),
    "Afghanistan": ("afghanistan", "herat", "افغانستان", "هرات"),
}


@dataclass
class DiscoverySeed:
    key: str
    kind: str
    url: str
    query_hits: Set[str] = field(default_factory=set)
    query_groups: Set[str] = field(default_factory=set)


@dataclass
class BusinessRecord:
    business_name: str
    website: str
    telegram: str
    instagram: str
    country: str
    city: str
    likely_exchange_shop: bool
    rate_page_detected: bool
    last_seen_rate: str
    candidate_score: int
    source_type: str
    sources_with_social_feeds: bool
    discovery_origins: str
    status_guess: str


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def clean_text(raw_html: str) -> str:
    text = re.sub(r"<br\\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    except OSError as exc:
        return None, None, f"os_error:{exc}"


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


def normalize_telegram_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    if token.startswith("t.me/") or token.startswith("telegram.me/"):
        token = "https://" + token
    parsed = urllib.parse.urlparse(token)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    handle = parts[1] if parts[0].lower() == "s" and len(parts) > 1 else parts[0]
    handle = handle.strip().lower()
    if not HANDLE_RE.match(handle):
        return None
    return f"https://t.me/s/{handle}"


def normalize_instagram_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    if token.startswith("instagram.com/"):
        token = "https://" + token
    parsed = urllib.parse.urlparse(token)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in {"instagram.com", "www.instagram.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    handle = parts[0].strip().lower()
    if handle in {"p", "reel", "stories", "explore"} or not HANDLE_RE.match(handle):
        return None
    return f"https://www.instagram.com/{handle}/"


def normalize_website_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    parsed = urllib.parse.urlparse(token)
    if parsed.scheme not in {"http", "https"}:
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


def domain_for(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    return domain[4:] if domain.startswith("www.") else domain


def normalize_business_name(text: str) -> str:
    lowered = translit_digits(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\u0600-\u06ff]+", " ", lowered)
    parts = [part for part in lowered.split() if part and part not in SHOP_NAME_STOPWORDS]
    return " ".join(parts).strip()


def extract_links(page: str) -> List[str]:
    out: List[str] = []
    unescaped = html.unescape(page)
    for pattern in (RAW_TME_RE, RAW_IG_RE, URL_RE):
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


def guess_city(text: str) -> str:
    lowered = translit_digits(text).lower()
    best_city = "unknown"
    best_score = 0
    for city, aliases in CITY_ALIASES.items():
        score = sum(lowered.count(alias.lower()) for alias in aliases)
        if score > best_score:
            best_city = city
            best_score = score
    return best_city


def guess_country(text: str, city: str) -> str:
    if city in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[city]
    lowered = translit_digits(text).lower()
    best_country = "unknown"
    best_score = 0
    for country, aliases in COUNTRY_KEYWORDS.items():
        score = sum(lowered.count(alias.lower()) for alias in aliases)
        if score > best_score:
            best_country = country
            best_score = score
    return best_country


def detect_language(text: str) -> str:
    lowered = text.lower()
    scores = {
        "Persian": sum(lowered.count(tok) for tok in ("صرافی", "دلار", "حواله", "ارز", "تهران", "دبی")),
        "English": sum(lowered.count(tok) for tok in ("exchange", "remittance", "currency", "dealer", "rate")),
        "Turkish": sum(lowered.count(tok) for tok in ("döviz", "istanbul", "transferi", "değişimi")),
        "German": sum(lowered.count(tok) for tok in ("wechselstube", "geldtransfer", "geldwechsel")),
        "Arabic": sum(lowered.count(tok) for tok in ("صرافة", "تحويل", "عملة", "دبي")),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


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
    deduped: List[str] = []
    seen: Set[str] = set()
    for _, text in blocks:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    lowered = translit_digits(text).lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def parse_number_token(token: str) -> Optional[int]:
    cleaned = re.sub(r"[^0-9]", "", translit_digits(token or ""))
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def extract_numbers(text: str) -> List[int]:
    values: List[int] = []
    for match in NUMBER_RE.finditer(translit_digits(text)):
        parsed = parse_number_token(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def count_quote_posts(blocks: Sequence[str]) -> Tuple[int, int]:
    quote_posts = 0
    buy_sell_pairs = 0
    for block in blocks[:30]:
        numbers = extract_numbers(block)
        if not numbers:
            continue
        has_rate_words = contains_keyword(block, RATE_KEYWORDS)
        if has_rate_words:
            quote_posts += 1
        lowered = translit_digits(block).lower()
        if has_rate_words and any(word in lowered for word in ("خرید", "فروش", "buy", "sell")) and len(numbers) >= 2:
            buy_sell_pairs += 1
    return quote_posts, buy_sell_pairs


def detect_source_type(text: str, country: str, quote_posts: int, has_phone: bool, has_address: bool) -> Tuple[str, bool]:
    lowered = translit_digits(text).lower()
    has_shop_words = contains_keyword(lowered, SHOP_KEYWORDS)
    has_remittance_words = contains_keyword(lowered, REMITTANCE_KEYWORDS)
    has_rate_words = contains_keyword(lowered, RATE_KEYWORDS)
    multiple_regions = sum(1 for word in REGIONAL_MARKET_KEYWORDS if word in lowered) >= 2

    likely_shop = bool(has_shop_words and (has_phone or has_address or has_rate_words))
    if likely_shop:
        return "exchange_shop", True
    if has_remittance_words and country in {"UAE", "Turkey", "UK", "Germany", "Iraq", "Afghanistan"}:
        return "remittance_exchange", False
    if quote_posts >= 3 and multiple_regions:
        return "regional_market_channel", False
    return "unknown", False


def detect_rate_page(text: str, quote_posts: int) -> bool:
    if quote_posts >= 2:
        return True
    lowered = translit_digits(text).lower()
    if contains_keyword(lowered, RATE_KEYWORDS) and len(extract_numbers(text)) >= 2:
        return True
    return False


def extract_last_seen_rate(page: str) -> str:
    tg_matches = re.findall(r'datetime="([^"]+)"', page)
    if tg_matches:
        return max(tg_matches)
    return extract_meta_content(page, "article:published_time") or extract_meta_content(page, "og:updated_time")


def compute_candidate_score(
    likely_exchange_shop: bool,
    rate_page_detected: bool,
    quote_posts: int,
    buy_sell_pairs: int,
    has_phone: bool,
    has_address: bool,
    has_social: bool,
    source_type: str,
    country: str,
) -> int:
    score = 0
    if likely_exchange_shop:
        score += 25
    if rate_page_detected:
        score += 20
    score += min(quote_posts * 4, 20)
    score += min(buy_sell_pairs * 5, 15)
    if has_phone:
        score += 8
    if has_address:
        score += 8
    if has_social:
        score += 7
    if source_type == "remittance_exchange":
        score += 6
    if country in {"Iran", "UAE", "Turkey", "UK", "Germany"}:
        score += 6
    return max(0, min(score, 100))


def build_query_plan() -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    for group, queries in QUERY_GROUPS.items():
        for query in queries:
            plan.append((group, query))
    seen: Set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for group, query in plan:
        key = f"{group}\t{query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append((group, query))
    return deduped


def run_search_discovery(query_plan: Sequence[Tuple[str, str]], pages_per_query: int, timeout: int, sleep_seconds: float) -> Tuple[Dict[str, DiscoverySeed], Dict[str, Any]]:
    discovered: Dict[str, DiscoverySeed] = {}
    debug: Dict[str, Any] = {
        "successful_search_requests": 0,
        "failed_search_requests": 0,
        "query_stats": {},
    }
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
                    key = f"telegram:{tg}"
                    seed = discovered.get(key)
                    if seed is None:
                        seed = DiscoverySeed(key=key, kind="telegram", url=tg)
                        discovered[key] = seed
                    seed.query_hits.add(query)
                    seed.query_groups.add(group)
                    hits += 1
                    continue
                ig = normalize_instagram_url(token)
                if ig:
                    key = f"instagram:{ig}"
                    seed = discovered.get(key)
                    if seed is None:
                        seed = DiscoverySeed(key=key, kind="instagram", url=ig)
                        discovered[key] = seed
                    seed.query_hits.add(query)
                    seed.query_groups.add(group)
                    hits += 1
                    continue
                site = normalize_website_url(token)
                if site:
                    key = f"website:{domain_for(site)}"
                    seed = discovered.get(key)
                    if seed is None:
                        seed = DiscoverySeed(key=key, kind="website", url=site)
                        discovered[key] = seed
                    seed.query_hits.add(query)
                    seed.query_groups.add(group)
                    hits += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds + random.random() * sleep_seconds * 0.5)
        debug["query_stats"][query] = {"group": group, "candidate_hits": hits}
    return discovered, debug


def enrich_seed(seed: DiscoverySeed, timeout: int) -> BusinessRecord:
    page, status, err = fetch_url(seed.url, timeout=timeout)
    if page is None:
        return BusinessRecord(
            business_name="",
            website=seed.url if seed.kind == "website" else "",
            telegram=seed.url if seed.kind == "telegram" else "",
            instagram=seed.url if seed.kind == "instagram" else "",
            country="unknown",
            city="unknown",
            likely_exchange_shop=False,
            rate_page_detected=False,
            last_seen_rate="",
            candidate_score=0,
            source_type="unknown",
            sources_with_social_feeds=seed.kind in {"telegram", "instagram"},
            discovery_origins="|".join(sorted(seed.query_groups)),
            status_guess=err or "fetch_failed",
        )

    title = extract_meta_content(page, "og:title") or extract_page_title(page)
    meta_desc = extract_meta_content(page, "description") or extract_meta_content(page, "og:description")
    blocks = extract_message_blocks(page)
    joined_text = " ".join(filter(None, [title, meta_desc] + blocks[:30]))
    links = extract_links(page)
    website = seed.url if seed.kind == "website" else ""
    telegram = seed.url if seed.kind == "telegram" else ""
    instagram = seed.url if seed.kind == "instagram" else ""

    for token in links:
        if not telegram:
            telegram = normalize_telegram_url(token) or telegram
        if not instagram:
            instagram = normalize_instagram_url(token) or instagram
        if not website:
            maybe_site = normalize_website_url(token)
            if maybe_site and domain_for(maybe_site) not in {"instagram.com", "t.me", "telegram.me"}:
                website = maybe_site

    business_name = title or meta_desc or seed.url
    city = guess_city(joined_text)
    country = guess_country(joined_text, city)
    quote_posts, buy_sell_pairs = count_quote_posts(blocks)
    has_phone = bool(PHONE_RE.search(joined_text))
    has_address = contains_keyword(joined_text, ADDRESS_KEYWORDS)
    source_type, likely_shop = detect_source_type(
        joined_text,
        country=country,
        quote_posts=quote_posts,
        has_phone=has_phone,
        has_address=has_address,
    )
    rate_page_detected = detect_rate_page(joined_text, quote_posts)
    has_social = bool(telegram or instagram)
    score = compute_candidate_score(
        likely_exchange_shop=likely_shop,
        rate_page_detected=rate_page_detected,
        quote_posts=quote_posts,
        buy_sell_pairs=buy_sell_pairs,
        has_phone=has_phone,
        has_address=has_address,
        has_social=has_social,
        source_type=source_type,
        country=country,
    )
    status_guess = "ok"
    if status is not None and status >= 400:
        status_guess = err or f"http_{status}"
    elif not blocks and not meta_desc:
        status_guess = "ok_no_text"

    return BusinessRecord(
        business_name=business_name,
        website=website,
        telegram=telegram,
        instagram=instagram,
        country=country,
        city=city,
        likely_exchange_shop=likely_shop,
        rate_page_detected=rate_page_detected,
        last_seen_rate=extract_last_seen_rate(page),
        candidate_score=score,
        source_type=source_type,
        sources_with_social_feeds=has_social,
        discovery_origins="|".join(sorted(seed.query_groups)),
        status_guess=status_guess,
    )


def merge_records(records: Sequence[BusinessRecord]) -> List[BusinessRecord]:
    grouped: Dict[str, BusinessRecord] = {}
    for record in records:
        name_key = normalize_business_name(record.business_name)
        domain_key = domain_for(record.website) if record.website else ""
        social_key = record.telegram or record.instagram
        key = name_key or domain_key or social_key or record.business_name or record.website or record.telegram or record.instagram
        if not key:
            continue
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = record
            continue
        grouped[key] = BusinessRecord(
            business_name=existing.business_name if len(existing.business_name) >= len(record.business_name) else record.business_name,
            website=existing.website or record.website,
            telegram=existing.telegram or record.telegram,
            instagram=existing.instagram or record.instagram,
            country=existing.country if existing.country != "unknown" else record.country,
            city=existing.city if existing.city != "unknown" else record.city,
            likely_exchange_shop=existing.likely_exchange_shop or record.likely_exchange_shop,
            rate_page_detected=existing.rate_page_detected or record.rate_page_detected,
            last_seen_rate=max(existing.last_seen_rate, record.last_seen_rate),
            candidate_score=max(existing.candidate_score, record.candidate_score),
            source_type=existing.source_type if existing.source_type != "unknown" else record.source_type,
            sources_with_social_feeds=existing.sources_with_social_feeds or record.sources_with_social_feeds,
            discovery_origins="|".join(sorted(set(filter(None, (existing.discovery_origins + "|" + record.discovery_origins).split("|"))))),
            status_guess=existing.status_guess if existing.status_guess == "ok" else record.status_guess,
        )
    return list(grouped.values())


def load_existing_registry(multilingual_csv: Path, channel_survey_csv: Path) -> Tuple[Set[str], Set[str], Set[str], int]:
    existing_names: Set[str] = set()
    existing_domains: Set[str] = set()
    existing_handles: Set[str] = set()
    total = 0

    for path in (multilingual_csv,):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total += 1
                normalized_name = normalize_business_name(str(row.get("title", "")))
                if normalized_name:
                    existing_names.add(normalized_name)
                handle = str(row.get("handle_or_url", "")).strip().lower()
                if handle:
                    existing_handles.add(handle)
                maybe_url = normalize_website_url(str(row.get("handle_or_url", "")) or "")
                if maybe_url:
                    existing_domains.add(domain_for(maybe_url))

    if channel_survey_csv.exists():
        with channel_survey_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total += 1
                normalized_name = normalize_business_name(str(row.get("title", "")))
                if normalized_name:
                    existing_names.add(normalized_name)
                handle = str(row.get("handle", "")).strip().lower()
                if handle:
                    existing_handles.add(handle)
    return existing_names, existing_domains, existing_handles, total


def filter_new_records(records: Sequence[BusinessRecord], registry: Tuple[Set[str], Set[str], Set[str], int]) -> List[BusinessRecord]:
    existing_names, existing_domains, existing_handles, _ = registry
    seen_names: Set[str] = set()
    seen_domains: Set[str] = set()
    seen_handles: Set[str] = set()
    out: List[BusinessRecord] = []
    for record in sorted(records, key=lambda item: (-item.candidate_score, item.business_name.lower())):
        name_key = normalize_business_name(record.business_name)
        domain_key = domain_for(record.website) if record.website else ""
        tg_handle = record.telegram.rstrip("/").split("/")[-1].lower() if record.telegram else ""
        ig_handle = record.instagram.rstrip("/").split("/")[-1].lower() if record.instagram else ""
        if name_key and (name_key in existing_names or name_key in seen_names):
            continue
        if domain_key and (domain_key in existing_domains or domain_key in seen_domains):
            continue
        if tg_handle and (tg_handle in existing_handles or tg_handle in seen_handles):
            continue
        if ig_handle and ig_handle in seen_handles:
            continue
        if name_key:
            seen_names.add(name_key)
        if domain_key:
            seen_domains.add(domain_key)
        if tg_handle:
            seen_handles.add(tg_handle)
        if ig_handle:
            seen_handles.add(ig_handle)
        out.append(record)
    return out


def summarize_records(records: Sequence[BusinessRecord], existing_total: int, search_debug: Dict[str, Any], query_count: int, crawled: int) -> Dict[str, Any]:
    by_country: Dict[str, int] = {}
    for record in records:
        by_country[record.country] = by_country.get(record.country, 0) + 1
    return {
        "generated_at": now_iso(),
        "new_businesses_discovered": len(records),
        "new_potential_exchange_shops": sum(1 for record in records if record.likely_exchange_shop),
        "sources_by_country": dict(sorted(by_country.items())),
        "sources_with_rate_pages": sum(1 for record in records if record.rate_page_detected),
        "sources_with_social_feeds": sum(1 for record in records if record.sources_with_social_feeds),
        "total_sources_in_registry": existing_total + len(records),
        "search_debug": search_debug,
        "queries_run": query_count,
        "candidates_crawled": crawled,
    }


def post_filter_records(records: Sequence[BusinessRecord]) -> List[BusinessRecord]:
    filtered: List[BusinessRecord] = []
    for record in records:
        if record.website:
            domain = domain_for(record.website)
            if domain in EXCLUDED_DOMAINS:
                continue
        if record.source_type == "unknown" and not record.likely_exchange_shop:
            continue
        if (
            record.candidate_score < 18
            and not record.likely_exchange_shop
            and not record.rate_page_detected
            and record.source_type == "unknown"
        ):
            continue
        filtered.append(record)
    return filtered


def write_candidates_csv(path: Path, records: Sequence[BusinessRecord]) -> None:
    fieldnames = [
        "business_name",
        "website",
        "telegram",
        "instagram",
        "country",
        "city",
        "likely_exchange_shop",
        "rate_page_detected",
        "last_seen_rate",
        "candidate_score",
        "source_type",
        "sources_with_social_feeds",
        "discovery_origins",
        "status_guess",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (-item.candidate_score, item.country, item.business_name.lower())):
            writer.writerow(
                {
                    "business_name": record.business_name,
                    "website": record.website,
                    "telegram": record.telegram,
                    "instagram": record.instagram,
                    "country": record.country,
                    "city": record.city,
                    "likely_exchange_shop": record.likely_exchange_shop,
                    "rate_page_detected": record.rate_page_detected,
                    "last_seen_rate": record.last_seen_rate,
                    "candidate_score": record.candidate_score,
                    "source_type": record.source_type,
                    "sources_with_social_feeds": record.sources_with_social_feeds,
                    "discovery_origins": record.discovery_origins,
                    "status_guess": record.status_guess,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Business directory and social discovery for RialWatch exchange shops")
    parser.add_argument("--survey-dir", default="survey_outputs")
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.45)
    parser.add_argument("--max-seeds", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir = Path(args.survey_dir)
    if not survey_dir.is_absolute():
        survey_dir = ROOT_DIR / survey_dir
    survey_dir.mkdir(parents=True, exist_ok=True)

    business_csv = survey_dir / "exchange_shop_business_candidates.csv"
    summary_json = survey_dir / "exchange_shop_business_summary.json"
    multilingual_csv = survey_dir / "exchange_shop_multilingual_candidates.csv"
    channel_survey_csv = survey_dir / "channel_survey.csv"

    query_plan = build_query_plan()
    discovered, search_debug = run_search_discovery(
        query_plan=query_plan,
        pages_per_query=args.pages_per_query,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )

    ordered_seeds = sorted(discovered.values(), key=lambda item: (item.kind, item.url))
    if args.max_seeds > 0:
        ordered_seeds = ordered_seeds[: args.max_seeds]

    enriched: List[BusinessRecord] = []
    for seed in ordered_seeds:
        enriched.append(enrich_seed(seed, timeout=args.timeout))
        if args.sleep > 0:
            time.sleep(args.sleep + random.random() * args.sleep * 0.5)

    merged = merge_records(enriched)
    registry = load_existing_registry(multilingual_csv=multilingual_csv, channel_survey_csv=channel_survey_csv)
    new_records = post_filter_records(filter_new_records(merged, registry))
    summary = summarize_records(
        records=new_records,
        existing_total=registry[3],
        search_debug=search_debug,
        query_count=len(query_plan),
        crawled=len(ordered_seeds),
    )

    write_candidates_csv(business_csv, new_records)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"queries_run={len(query_plan)}")
    print(f"candidates_crawled={len(ordered_seeds)}")
    print(f"new_businesses_discovered={summary['new_businesses_discovered']}")
    print(f"new_potential_exchange_shops={summary['new_potential_exchange_shops']}")
    print(f"sources_with_rate_pages={summary['sources_with_rate_pages']}")
    print(f"sources_with_social_feeds={summary['sources_with_social_feeds']}")
    print(f"business_csv={business_csv}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
