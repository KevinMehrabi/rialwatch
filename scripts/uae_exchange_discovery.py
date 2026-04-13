#!/usr/bin/env python3
"""UAE-focused exchange and remittance discovery for RialWatch.

This discovery pass is diagnostics-only. It looks for Iranian-linked exchange,
remittance, and settlement businesses in Dubai/UAE using public search results,
business websites, Instagram pages, WhatsApp links, and Telegram as a
secondary enrichment surface.
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

QUERY_GROUPS: Dict[str, List[str]] = {
    "persian": [
        "صرافی دبی",
        "صرافی ایرانی دبی",
        "حواله ایران دبی",
        "تبدیل ارز ایرانی دبی",
        "نرخ دلار دبی",
        "دلار دبی ایران",
    ],
    "english": [
        "Iranian exchange Dubai",
        "Iranian money exchange Dubai",
        "Iran remittance Dubai",
        "Persian exchange Dubai",
        "Dubai exchange Iran transfer",
        "Dubai settlement Iran exchange",
    ],
    "arabic": [
        "صرافة ايرانية دبي",
        "تحويل ايران دبي",
        "مكاتب صرافة ايران دبي",
        "صرافة دبي ايران",
    ],
    "districts": [
        "iranian exchange deira",
        "iranian exchange bur dubai",
        "iranian exchange al ras",
        "iranian exchange karama",
        "صرافی دیره دبی",
        "صرافی بر دبی",
        "حواله ایرانی دیره",
    ],
    "instagram": [
        "site:instagram.com صرافی دبی",
        "site:instagram.com صرافة دبي ايران",
        "site:instagram.com iran exchange dubai",
        "site:instagram.com iran remittance dubai",
        "site:instagram.com persian exchange dubai",
    ],
    "business": [
        "iranian exchange dubai map",
        "iranian remittance dubai directory",
        "dubai iran transfer business",
        "deira money exchange iran",
        "bur dubai remittance iran",
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
RAW_WA_RE = re.compile(r"https?://(?:wa\.me|api\.whatsapp\.com)/[^\s\"'<>]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d|05\d{8}|\+971\s?\d[\d\s]{7,}|\+98\s?\d[\d\s]{7,})")
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")

SHOP_KEYWORDS = (
    "صرافی",
    "صرافة",
    "exchange",
    "sarafi",
    "money exchange",
    "currency exchange",
)
REMITTANCE_KEYWORDS = (
    "حواله",
    "remittance",
    "money transfer",
    "transfer",
    "settlement",
    "hawala",
    "تحويل",
)
IRAN_TRANSFER_KEYWORDS = (
    "ایران",
    "iran",
    "iranian",
    "to iran",
    "iran transfer",
    "تحويل ايران",
    "حواله ایران",
)
RATE_KEYWORDS = (
    "نرخ",
    "قیمت",
    "rate",
    "quote",
    "usd",
    "eur",
    "aed",
    "دلار",
    "یورو",
    "درهم",
    "buy",
    "sell",
    "خرید",
    "فروش",
)
USD_WORDS = ("usd", "دلار", "dollar")
AED_WORDS = ("aed", "درهم", "dirham")
DISTRICT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Deira": ("deira", "دیره"),
    "Bur Dubai": ("bur dubai", "بردبی", "بر دبی"),
    "Al Ras": ("al ras", "الراس", "راس"),
    "Karama": ("karama", "کرامه", "الکرامه"),
}
CITY_PRIORITY = ("Deira", "Bur Dubai", "Al Ras", "Karama", "Dubai")

STOPWORDS = {
    "exchange",
    "money",
    "currency",
    "telegram",
    "instagram",
    "whatsapp",
    "iranian",
    "iran",
    "dubai",
    "uae",
    "صرافی",
    "حواله",
    "دبی",
    "امارات",
    "group",
    "llc",
    "limited",
}

NOISE_DOMAINS = {
    "alanchand.com",
    "bonbast.com",
    "navasan.net",
    "moneytransfers.com",
    "wise.com",
    "tgju.org",
    "themoneyconverter.com",
    "exchangerates247.com",
    "independent.co.uk",
    "news24online.com",
    "turkiyetoday.com",
    "2gis.ae",
}
NOISE_TITLE_HINTS = (
    "best exchange",
    "بهترین صرافی",
    "معرفی",
    "راهنما",
    "guide",
    "watch:",
    "drone",
    "strike",
    "convert",
    "exchange rate",
    "currency converter",
    "on the map",
)
BUSINESS_HINTS = (
    "exchange",
    "صرافی",
    "حواله",
    "remittance",
    "money transfer",
    "settlement",
    "ارسال حواله",
    "انتقال پول",
)
BUSINESS_DOMAIN_HINTS = (
    "exchange",
    "sarafi",
    "saraf",
    "pardakht",
    "card",
    "ansari",
    "razouki",
    "dinar",
    "emarat",
    "hafez",
)
UAE_LOCALITY_HINTS = (
    "dubai",
    "uae",
    "deira",
    "bur dubai",
    "al ras",
    "karama",
    "دبی",
    "امارات",
    "درهم",
    "دیره",
)


@dataclass
class DiscoverySeed:
    key: str
    kind: str
    url: str
    query_hits: Set[str] = field(default_factory=set)
    query_groups: Set[str] = field(default_factory=set)


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


def clip_text(text: str, limit: int = 220) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def fetch_url(url: str, timeout: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fa,en-US;q=0.9,en;q=0.8,ar;q=0.7",
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
    if token.startswith("t.me/") or token.startswith("telegram.me/"):
        token = "https://" + token
    try:
        parsed = urllib.parse.urlparse(token)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    handle = parts[1] if parts[0].lower() == "s" and len(parts) > 1 else parts[0]
    handle = re.sub(r"[^A-Za-z0-9_.]", "", handle.strip().lower())
    if handle in {"s", "share", "joinchat", "login", "contact"} or not HANDLE_RE.match(handle):
        return None
    return f"https://t.me/s/{handle}"


def normalize_instagram_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if token.startswith("instagram.com/"):
        token = "https://" + token
    try:
        parsed = urllib.parse.urlparse(token)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in {"instagram.com", "www.instagram.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    handle = parts[0].strip().lower()
    if handle in {"p", "reel", "explore", "stories"} or not HANDLE_RE.match(handle):
        return None
    return f"https://www.instagram.com/{handle}/"


def normalize_whatsapp_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    try:
        parsed = urllib.parse.urlparse(token)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    domain = parsed.netloc.lower()
    if domain not in {"wa.me", "api.whatsapp.com", "www.whatsapp.com"}:
        return None
    return token


def normalize_website_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    try:
        parsed = urllib.parse.urlparse(token)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain or domain in EXCLUDED_DOMAINS:
        return None
    if domain in {"t.me", "telegram.me", "instagram.com", "wa.me", "api.whatsapp.com"}:
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
    parts = [part for part in lowered.split() if part and part not in STOPWORDS]
    return " ".join(parts).strip()


def extract_links(page: str) -> List[str]:
    out: List[str] = []
    unescaped = html.unescape(page)
    for pattern in (RAW_TME_RE, RAW_IG_RE, RAW_WA_RE, URL_RE):
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


def extract_blocks(page: str) -> List[str]:
    blocks: List[Tuple[int, str]] = []
    for pattern in (
        r'<div class="tgme_widget_message_text[^\"]*"[^>]*>(.*?)</div>',
        r'<div class="tgme_widget_message_caption[^\"]*"[^>]*>(.*?)</div>',
        r"<p[^>]*>(.*?)</p>",
        r"<li[^>]*>(.*?)</li>",
        r"<article[^>]*>(.*?)</article>",
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


def detect_city_or_district(text: str) -> str:
    lowered = translit_digits(text).lower()
    best = "Dubai"
    best_score = 0
    for city in CITY_PRIORITY:
        aliases = DISTRICT_ALIASES[city]
        score = sum(lowered.count(alias.lower()) for alias in aliases)
        if score > best_score:
            best = city
            best_score = score
    return best if best_score > 0 else "Dubai"


def detect_last_seen(page: str) -> str:
    tg_matches = re.findall(r'datetime="([^"]+)"', page)
    if tg_matches:
        return max(tg_matches)
    return (
        extract_meta_content(page, "article:published_time")
        or extract_meta_content(page, "og:updated_time")
        or extract_meta_content(page, "article:modified_time")
    )


def classify_source_type(text: str, has_whatsapp: bool, has_telegram: bool, has_instagram: bool, rate_posts: int, iran_transfer_hint: bool) -> str:
    lowered = translit_digits(text).lower()
    has_shop_words = contains_keyword(lowered, SHOP_KEYWORDS)
    has_remittance_words = contains_keyword(lowered, REMITTANCE_KEYWORDS)
    has_rate_words = contains_keyword(lowered, RATE_KEYWORDS)
    if has_shop_words and (has_rate_words or has_whatsapp or has_telegram or has_instagram):
        return "exchange_shop"
    if iran_transfer_hint and (has_remittance_words or has_whatsapp):
        return "remittance_exchange"
    if has_remittance_words:
        return "settlement_exchange"
    if rate_posts >= 3:
        return "regional_market_signal"
    return "unknown"


def detect_quote_signals(blocks: Sequence[str]) -> Tuple[int, int, bool, bool]:
    quote_post_count = 0
    usd_detected = False
    aed_detected = False
    for block in blocks[:40]:
        numbers = extract_numbers(block)
        lowered = translit_digits(block).lower()
        if not numbers:
            continue
        if contains_keyword(lowered, RATE_KEYWORDS):
            quote_post_count += 1
        if any(word in lowered for word in USD_WORDS) and any(value >= 100000 for value in numbers):
            usd_detected = True
        if any(word in lowered for word in AED_WORDS) and any(value >= 10000 for value in numbers):
            aed_detected = True
    return quote_post_count, sum(1 for block in blocks[:40] if contains_keyword(block, RATE_KEYWORDS) and len(extract_numbers(block)) >= 2), usd_detected, aed_detected


def compute_parseability_score(
    quote_post_count: int,
    usd_detected: bool,
    aed_detected: bool,
    has_telegram: bool,
    has_instagram: bool,
    has_whatsapp: bool,
    rate_page_detected: bool,
) -> int:
    score = 0
    score += min(quote_post_count * 8, 40)
    if usd_detected:
        score += 18
    if aed_detected:
        score += 12
    if rate_page_detected:
        score += 12
    if has_telegram:
        score += 8
    if has_instagram:
        score += 6
    if has_whatsapp:
        score += 4
    return max(0, min(score, 100))


def compute_candidate_score(
    source_type: str,
    has_website: bool,
    has_instagram: bool,
    has_telegram: bool,
    has_whatsapp: bool,
    iran_transfer_hint: bool,
    rate_page_detected: bool,
    rate_post_detected: bool,
    usd_quote_detected: bool,
    aed_quote_detected: bool,
    parseability_score: int,
) -> int:
    score = parseability_score
    if has_website:
        score += 10
    if has_instagram:
        score += 8
    if has_telegram:
        score += 6
    if has_whatsapp:
        score += 8
    if iran_transfer_hint:
        score += 10
    if rate_page_detected:
        score += 10
    if rate_post_detected:
        score += 8
    if usd_quote_detected:
        score += 10
    if aed_quote_detected:
        score += 6
    if source_type in {"exchange_shop", "remittance_exchange", "settlement_exchange"}:
        score += 10
    return max(0, min(score, 100))


def is_business_like_candidate(record: UAECandidate) -> bool:
    domain = domain_for(record.website) if record.website else ""
    title = translit_digits(record.business_name).lower()
    has_direct_contact = bool(record.instagram or record.telegram or record.whatsapp_link)
    has_business_hint = contains_keyword(title, BUSINESS_HINTS)
    domain_has_business_hint = any(token in domain for token in BUSINESS_DOMAIN_HINTS)
    title_has_uae_hint = any(token in title for token in UAE_LOCALITY_HINTS)
    if domain in NOISE_DOMAINS:
        return False
    if any(hint in title for hint in NOISE_TITLE_HINTS) and not domain_has_business_hint:
        return False
    if any(hint in title for hint in ("watch:", "drone", "strike", "explosion")):
        return False
    if record.source_type_guess == "unknown" and not has_direct_contact:
        return False
    if not (has_direct_contact or domain_has_business_hint):
        return False
    if not (title_has_uae_hint or domain_has_business_hint):
        return False
    if not (record.rate_page_detected or record.rate_post_detected or record.iran_transfer_hint or has_direct_contact):
        return False
    if record.candidate_score < 40 and not has_direct_contact:
        return False
    if record.source_type_guess in {"exchange_shop", "remittance_exchange", "settlement_exchange"}:
        return True
    return has_business_hint and has_direct_contact


def build_query_plan() -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    for group, queries in QUERY_GROUPS.items():
        for query in queries:
            plan.append((group, query))
    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for group, query in plan:
        key = f"{group}\t{query}"
        if key in seen:
            continue
        seen.add(key)
        out.append((group, query))
    return out


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
                wa = normalize_whatsapp_url(token)
                if wa:
                    key = f"whatsapp:{wa}"
                    seed = discovered.get(key)
                    if seed is None:
                        seed = DiscoverySeed(key=key, kind="whatsapp", url=wa)
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
                time.sleep(sleep_seconds + random.random() * sleep_seconds * 0.4)
        debug["query_stats"][query] = {"group": group, "candidate_hits": hits}
    return discovered, debug


def derive_business_name(title: str, meta_desc: str, seed: DiscoverySeed, links: Sequence[str]) -> str:
    for value in (title, meta_desc):
        cleaned = re.sub(r"\s+\|\s+.*$", "", clean_text(value))
        if cleaned and len(cleaned) >= 4:
            return cleaned
    domain = domain_for(seed.url) if seed.kind == "website" else ""
    if domain:
        label = domain.split(".")[0].replace("-", " ").replace("_", " ").strip()
        if label:
            return label.title()
    for link in links:
        ig = normalize_instagram_url(link)
        if ig:
            handle = ig.rstrip("/").split("/")[-1]
            return handle.replace("_", " ").replace(".", " ").strip()
    return seed.url


def enrich_seed(seed: DiscoverySeed, timeout: int) -> UAECandidate:
    page, status, err = fetch_url(seed.url, timeout=timeout)
    if page is None:
        website = seed.url if seed.kind == "website" else ""
        instagram = seed.url if seed.kind == "instagram" else ""
        telegram = seed.url if seed.kind == "telegram" else ""
        whatsapp_link = seed.url if seed.kind == "whatsapp" else ""
        return UAECandidate(
            business_name=seed.url,
            website=website,
            instagram=instagram,
            telegram=telegram,
            whatsapp_link=whatsapp_link,
            city_or_district="Dubai",
            country="UAE",
            source_type_guess="unknown",
            rate_page_detected=False,
            rate_post_detected=False,
            iran_transfer_hint=False,
            usd_irr_quote_detected=False,
            aed_irr_quote_detected=False,
            quote_post_count=0,
            parseability_score=0,
            last_seen="",
            candidate_score=0,
            discovery_origins="|".join(sorted(seed.query_groups)),
            status_guess=err or "fetch_failed",
        )

    title = extract_meta_content(page, "og:title") or extract_page_title(page)
    meta_desc = extract_meta_content(page, "description") or extract_meta_content(page, "og:description")
    blocks = extract_blocks(page)
    links = extract_links(page)
    website = seed.url if seed.kind == "website" else ""
    instagram = seed.url if seed.kind == "instagram" else ""
    telegram = seed.url if seed.kind == "telegram" else ""
    whatsapp_link = seed.url if seed.kind == "whatsapp" else ""

    for token in links:
        if not website:
            website = normalize_website_url(token) or website
        if not instagram:
            instagram = normalize_instagram_url(token) or instagram
        if not telegram:
            telegram = normalize_telegram_url(token) or telegram
        if not whatsapp_link:
            whatsapp_link = normalize_whatsapp_url(token) or whatsapp_link

    joined_text = " ".join(filter(None, [title, meta_desc] + blocks[:40]))
    business_name = derive_business_name(title, meta_desc, seed, links)
    city_or_district = detect_city_or_district(joined_text)
    iran_transfer_hint = contains_keyword(joined_text, IRAN_TRANSFER_KEYWORDS)
    quote_post_count, pair_count, usd_irr_quote_detected, aed_irr_quote_detected = detect_quote_signals(blocks)
    rate_page_detected = (
        contains_keyword(joined_text, RATE_KEYWORDS)
        and len(extract_numbers(joined_text)) >= 2
    )
    rate_post_detected = quote_post_count > 0 or pair_count > 0
    source_type_guess = classify_source_type(
        joined_text,
        has_whatsapp=bool(whatsapp_link),
        has_telegram=bool(telegram),
        has_instagram=bool(instagram),
        rate_posts=quote_post_count,
        iran_transfer_hint=iran_transfer_hint,
    )
    parseability_score = compute_parseability_score(
        quote_post_count=quote_post_count,
        usd_detected=usd_irr_quote_detected,
        aed_detected=aed_irr_quote_detected,
        has_telegram=bool(telegram),
        has_instagram=bool(instagram),
        has_whatsapp=bool(whatsapp_link),
        rate_page_detected=rate_page_detected,
    )
    candidate_score = compute_candidate_score(
        source_type=source_type_guess,
        has_website=bool(website),
        has_instagram=bool(instagram),
        has_telegram=bool(telegram),
        has_whatsapp=bool(whatsapp_link),
        iran_transfer_hint=iran_transfer_hint,
        rate_page_detected=rate_page_detected,
        rate_post_detected=rate_post_detected,
        usd_quote_detected=usd_irr_quote_detected,
        aed_quote_detected=aed_irr_quote_detected,
        parseability_score=parseability_score,
    )
    status_guess = "ok"
    if status is not None and status >= 400:
        status_guess = err or f"http_{status}"
    elif not blocks and not meta_desc and not title:
        status_guess = "ok_no_text"

    return UAECandidate(
        business_name=business_name,
        website=website,
        instagram=instagram,
        telegram=telegram,
        whatsapp_link=whatsapp_link,
        city_or_district=city_or_district,
        country="UAE",
        source_type_guess=source_type_guess,
        rate_page_detected=rate_page_detected,
        rate_post_detected=rate_post_detected,
        iran_transfer_hint=iran_transfer_hint,
        usd_irr_quote_detected=usd_irr_quote_detected,
        aed_irr_quote_detected=aed_irr_quote_detected,
        quote_post_count=quote_post_count,
        parseability_score=parseability_score,
        last_seen=detect_last_seen(page),
        candidate_score=candidate_score,
        discovery_origins="|".join(sorted(seed.query_groups)),
        status_guess=status_guess,
    )


def merge_records(records: Sequence[UAECandidate]) -> List[UAECandidate]:
    merged: Dict[str, UAECandidate] = {}
    for record in records:
        name_key = normalize_business_name(record.business_name)
        domain_key = domain_for(record.website) if record.website else ""
        social_key = record.telegram or record.instagram or record.whatsapp_link
        key = name_key or domain_key or social_key or record.business_name
        if not key:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            continue
        merged[key] = UAECandidate(
            business_name=existing.business_name if len(existing.business_name) >= len(record.business_name) else record.business_name,
            website=existing.website or record.website,
            instagram=existing.instagram or record.instagram,
            telegram=existing.telegram or record.telegram,
            whatsapp_link=existing.whatsapp_link or record.whatsapp_link,
            city_or_district=existing.city_or_district if existing.city_or_district != "Dubai" else record.city_or_district,
            country="UAE",
            source_type_guess=existing.source_type_guess if existing.source_type_guess != "unknown" else record.source_type_guess,
            rate_page_detected=existing.rate_page_detected or record.rate_page_detected,
            rate_post_detected=existing.rate_post_detected or record.rate_post_detected,
            iran_transfer_hint=existing.iran_transfer_hint or record.iran_transfer_hint,
            usd_irr_quote_detected=existing.usd_irr_quote_detected or record.usd_irr_quote_detected,
            aed_irr_quote_detected=existing.aed_irr_quote_detected or record.aed_irr_quote_detected,
            quote_post_count=max(existing.quote_post_count, record.quote_post_count),
            parseability_score=max(existing.parseability_score, record.parseability_score),
            last_seen=max(existing.last_seen, record.last_seen),
            candidate_score=max(existing.candidate_score, record.candidate_score),
            discovery_origins="|".join(sorted(set(filter(None, (existing.discovery_origins + "|" + record.discovery_origins).split("|"))))),
            status_guess=existing.status_guess if existing.status_guess == "ok" else record.status_guess,
        )
    return list(merged.values())


def load_existing_registry(multilingual_csv: Path, business_csv: Path, ranking_csv: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    existing_names: Set[str] = set()
    existing_domains: Set[str] = set()
    existing_handles: Set[str] = set()

    def register_name(value: str) -> None:
        norm = normalize_business_name(value)
        if norm:
            existing_names.add(norm)

    def register_url(value: str) -> None:
        if not value:
            return
        tg = normalize_telegram_url(value)
        if tg:
            existing_handles.add(tg.rstrip("/").split("/")[-1].lower())
            return
        ig = normalize_instagram_url(value)
        if ig:
            existing_handles.add(ig.rstrip("/").split("/")[-1].lower())
            return
        website = normalize_website_url(value)
        if website:
            existing_domains.add(domain_for(website))

    for path in (multilingual_csv, business_csv, ranking_csv):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                register_name(str(row.get("business_name", "")) or str(row.get("title", "")) or str(row.get("handle_or_url", "")))
                for key in ("website", "telegram", "instagram", "handle_or_url"):
                    register_url(str(row.get(key, "")))
    return existing_names, existing_domains, existing_handles


def filter_new_records(records: Sequence[UAECandidate], registry: Tuple[Set[str], Set[str], Set[str]]) -> List[UAECandidate]:
    existing_names, existing_domains, existing_handles = registry
    seen_names: Set[str] = set()
    seen_domains: Set[str] = set()
    seen_handles: Set[str] = set()
    out: List[UAECandidate] = []
    for record in sorted(records, key=lambda item: (-item.candidate_score, -item.parseability_score, item.business_name.lower())):
        name_key = normalize_business_name(record.business_name)
        domain_key = domain_for(record.website) if record.website else ""
        handles = []
        for url in (record.telegram, record.instagram):
            if url:
                handles.append(url.rstrip("/").split("/")[-1].lower())
        if name_key and (name_key in existing_names or name_key in seen_names):
            continue
        if domain_key and (domain_key in existing_domains or domain_key in seen_domains):
            continue
        if any(handle in existing_handles or handle in seen_handles for handle in handles):
            continue
        if not is_business_like_candidate(record):
            continue
        if record.candidate_score < 20 and not record.rate_post_detected and not record.rate_page_detected:
            continue
        if name_key:
            seen_names.add(name_key)
        if domain_key:
            seen_domains.add(domain_key)
        for handle in handles:
            seen_handles.add(handle)
        out.append(record)
    return out


def write_candidates_csv(path: Path, records: Sequence[UAECandidate]) -> None:
    fieldnames = [
        "business_name",
        "website",
        "instagram",
        "telegram",
        "whatsapp_link",
        "city_or_district",
        "country",
        "source_type_guess",
        "rate_page_detected",
        "rate_post_detected",
        "iran_transfer_hint",
        "usd_irr_quote_detected",
        "aed_irr_quote_detected",
        "quote_post_count",
        "parseability_score",
        "last_seen",
        "candidate_score",
        "discovery_origins",
        "status_guess",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (-item.candidate_score, -item.parseability_score, item.business_name.lower())):
            writer.writerow(
                {
                    "business_name": record.business_name,
                    "website": record.website,
                    "instagram": record.instagram,
                    "telegram": record.telegram,
                    "whatsapp_link": record.whatsapp_link,
                    "city_or_district": record.city_or_district,
                    "country": record.country,
                    "source_type_guess": record.source_type_guess,
                    "rate_page_detected": record.rate_page_detected,
                    "rate_post_detected": record.rate_post_detected,
                    "iran_transfer_hint": record.iran_transfer_hint,
                    "usd_irr_quote_detected": record.usd_irr_quote_detected,
                    "aed_irr_quote_detected": record.aed_irr_quote_detected,
                    "quote_post_count": record.quote_post_count,
                    "parseability_score": record.parseability_score,
                    "last_seen": record.last_seen,
                    "candidate_score": record.candidate_score,
                    "discovery_origins": record.discovery_origins,
                    "status_guess": record.status_guess,
                }
            )


def summarize_records(records: Sequence[UAECandidate], search_debug: Dict[str, Any], query_count: int, crawled: int) -> Dict[str, Any]:
    strong_candidates = [
        record.business_name
        for record in records
        if record.candidate_score >= 65 and (record.rate_post_detected or record.rate_page_detected or record.iran_transfer_hint)
    ]
    return {
        "generated_at": now_iso(),
        "new_uae_candidates_discovered": len(records),
        "likely_exchange_or_remittance_businesses": sum(
            1
            for record in records
            if record.source_type_guess in {"exchange_shop", "remittance_exchange", "settlement_exchange"}
        ),
        "sources_with_websites": sum(1 for record in records if bool(record.website)),
        "sources_with_instagram": sum(1 for record in records if bool(record.instagram)),
        "sources_with_telegram": sum(1 for record in records if bool(record.telegram)),
        "sources_with_whatsapp": sum(1 for record in records if bool(record.whatsapp_link)),
        "sources_with_rate_signals": sum(1 for record in records if record.rate_page_detected or record.rate_post_detected),
        "candidates_strong_enough_for_uae_basket_review": strong_candidates,
        "search_debug": search_debug,
        "queries_run": query_count,
        "candidates_crawled": crawled,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAE exchange/remittance discovery for RialWatch")
    parser.add_argument("--survey-dir", default="survey_outputs")
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--max-seeds", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir = Path(args.survey_dir)
    if not survey_dir.is_absolute():
        survey_dir = ROOT_DIR / survey_dir
    survey_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = survey_dir / "uae_exchange_discovery_candidates.csv"
    summary_json = survey_dir / "uae_exchange_discovery_summary.json"
    multilingual_csv = survey_dir / "exchange_shop_multilingual_candidates.csv"
    business_csv = survey_dir / "exchange_shop_business_candidates.csv"
    ranking_csv = survey_dir / "exchange_shop_candidate_ranking.csv"

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

    enriched: List[UAECandidate] = []
    for seed in ordered_seeds:
        enriched.append(enrich_seed(seed, timeout=args.timeout))
        if args.sleep > 0:
            time.sleep(args.sleep + random.random() * args.sleep * 0.4)

    merged = merge_records(enriched)
    registry = load_existing_registry(multilingual_csv=multilingual_csv, business_csv=business_csv, ranking_csv=ranking_csv)
    filtered = filter_new_records(merged, registry)
    summary = summarize_records(
        records=filtered,
        search_debug=search_debug,
        query_count=len(query_plan),
        crawled=len(ordered_seeds),
    )

    write_candidates_csv(candidates_csv, filtered)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
