#!/usr/bin/env python3
"""Discover and classify public Telegram FX quote channels for RialWatch.

This script performs a discovery survey only (no ingestion):
1) Search web queries for candidate Telegram channels
2) Normalize and deduplicate discovered handles
3) Crawl public https://t.me/s/<channel> pages without login
4) Classify channel type and parseability
5) Export survey artifacts
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
    "Chrome/122.0.0.0 Safari/537.36"
)

NUMBER_PATTERN = re.compile(r"[0-9۰-۹٠-٩]{2,3}(?:[\s,٬،][0-9۰-۹٠-٩]{3})+")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d|09\d{9})")
WEBSITE_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
RAW_TME_PATTERN = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
BARE_TME_PATTERN = re.compile(r"(?:^|\s)(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_]{5,})")
HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{5,}$")

QUOTE_KEYWORDS = (
    "usd",
    "دلار",
    "eur",
    "یورو",
    "aed",
    "درهم",
    "نرخ",
    "قیمت",
    "buy",
    "sell",
    "خرید",
    "فروش",
)

NETWORK_KEYWORDS = (
    "شبکه",
    "همکار",
    "dealer",
    "عمده",
    "تلگرامی صرافان",
    "معامله گر",
)

AGGREGATOR_KEYWORDS = (
    "اخبار ارز",
    "price channel",
    "quotes",
    "نرخ لحظه ای",
    "سیگنال",
    "تحلیل",
    "news",
)

ADDRESS_KEYWORDS = (
    "آدرس",
    "خیابان",
    "بلوار",
    "کوچه",
    "میدان",
    "پلاک",
    "واحد",
    "طبقه",
    "address",
    "street",
    "ave",
)

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
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
    "Hamburg": ("hamburg", "هامبورگ"),
    "London": ("london", "لندن"),
}

DEFAULT_SEARCH_QUERIES = [
    "صرافی",
    "صرافی تهران",
    "صرافی مشهد",
    "نرخ دلار صرافی",
    "قیمت دلار صرافی",
    "بازار ارز تهران",
    "site:t.me صرافی",
    "site:t.me/s صرافی",
    "صرافی استانبول",
    "صرافی دبی",
    "صرافی فرانکفورت",
    "صرافی لندن",
    "کانال نرخ ارز",
    "کانال قیمت دلار",
    "site:t.me نرخ دلار",
    "site:t.me/s نرخ ارز",
    "صرافی اصفهان",
    "صرافی شیراز",
    "صرافی تبریز",
    "صرافی کرج",
    "صرافی قم",
    "صرافی اهواز",
    "صرافی رشت",
    "صرافی کرمان",
    "کانال صرافی تهران",
    "کانال صرافی مشهد",
    "کانال صرافی اصفهان",
    "کانال صرافی شیراز",
    "کانال صرافی تبریز",
    "کانال صرافی کرج",
    "کانال دلار تهران",
    "کانال دلار مشهد",
    "قیمت دلار تهران صرافی",
    "قیمت دلار مشهد صرافی",
    "نرخ ارز صرافی تهران",
    "نرخ ارز صرافی مشهد",
    "نرخ ارز صرافی اصفهان",
    "نرخ ارز صرافی شیراز",
    "صرافی ایرانی دبی",
    "صرافی ایرانی استانبول",
    "صرافی ایرانی فرانکفورت",
    "صرافی ایرانی لندن",
    "صرافی هامبورگ",
    "صرافی ایرانی هامبورگ",
    "market rate telegram iran",
    "iran sarafi telegram",
    "usd toman telegram iran",
    "site:t.me کانال صرافی",
    "site:t.me/s کانال صرافی",
    "site:t.me/s دلار تهران",
    "site:t.me/s قیمت دلار",
    "site:t.me/s نرخ ارز",
]

DEFAULT_EXCLUDE_HANDLES = {
    "iv",
    "share",
    "joinchat",
    "addstickers",
    "proxy",
    "s",
    "login",
    "contact",
}


@dataclass
class SearchHit:
    handle: str
    normalized_url: str
    query: str
    source: str
    raw_url: str


@dataclass
class ChannelSnapshot:
    handle: str
    public_url: str
    title: str = ""
    status: str = "not_crawled"
    http_status: Optional[int] = None
    crawl_error: Optional[str] = None
    city_guess: str = "unknown"
    channel_type_guess: str = "unclear"
    likely_individual_shop: bool = False
    parseable_score: int = 0
    quote_post_count: int = 0
    total_posts_visible: int = 0
    text_message_count: int = 0
    has_phone: bool = False
    has_address: bool = False
    has_website: bool = False
    mentioned_handles: List[str] = field(default_factory=list)
    linked_channels: List[str] = field(default_factory=list)
    website_links: List[str] = field(default_factory=list)
    sample_messages: List[str] = field(default_factory=list)
    last_seen_text_sample: str = ""
    last_seen_datetime: Optional[str] = None
    crawled_at: str = ""


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


def extract_meta_content(page: str, prop: str) -> str:
    pattern = re.compile(
        rf"<meta[^>]+(?:property|name)=[\"']{re.escape(prop)}[\"'][^>]+content=[\"'](.*?)[\"']",
        re.IGNORECASE,
    )
    match = pattern.search(page)
    return html.unescape(match.group(1)).strip() if match else ""


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
    except socket.timeout:
        return None, None, "timeout"
    except urllib.error.URLError as exc:
        return None, None, f"network_error:{exc.reason}"
    except TimeoutError:
        return None, None, "timeout"


def normalize_channel_url(raw: str, excluded: Set[str]) -> Optional[Tuple[str, str]]:
    candidate = raw.strip().strip("'\"<>()[]{}.,;")
    if not candidate:
        return None
    if candidate.startswith("t.me/") or candidate.startswith("telegram.me/"):
        candidate = "https://" + candidate
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc.lower() not in ("t.me", "www.t.me", "telegram.me", "www.telegram.me"):
        return None

    path_parts = [p for p in parsed.path.split("/") if p]
    if not path_parts:
        return None

    handle = ""
    if path_parts[0].lower() == "s" and len(path_parts) >= 2:
        handle = path_parts[1]
    else:
        handle = path_parts[0]

    handle = handle.strip()
    if not handle:
        return None
    if handle.startswith("+"):
        return None
    if handle.lower() in excluded:
        return None
    if not HANDLE_RE.match(handle):
        return None

    canonical = handle.lower()
    return canonical, f"https://t.me/s/{canonical}"


def extract_candidate_links(page: str) -> List[str]:
    found: List[str] = []
    page_unescaped = html.unescape(page)

    # Direct links present in raw page text.
    for text_blob in (page_unescaped, urllib.parse.unquote(page_unescaped)):
        for pattern in (RAW_TME_PATTERN, BARE_TME_PATTERN):
            for match in pattern.finditer(text_blob):
                token = match.group(0).strip()
                if token:
                    found.append(token)

    # Follow common search-engine wrappers like duckduckgo's uddg=encoded_target.
    for href in re.findall(r'href=["\']([^"\']+)["\']', page_unescaped, flags=re.IGNORECASE):
        token = href.strip()
        if not token:
            continue
        if token.startswith("//"):
            token = "https:" + token

        parsed = urllib.parse.urlparse(token)
        if parsed.netloc.lower() in ("duckduckgo.com", "www.duckduckgo.com"):
            q = urllib.parse.parse_qs(parsed.query)
            for key in ("uddg", "u"):
                for item in q.get(key, []):
                    decoded = urllib.parse.unquote(item)
                    if decoded:
                        found.append(decoded)

    # Also capture uddg occurrences outside href attributes.
    for encoded in re.findall(r"uddg=([^&\"'<>\\s]+)", page_unescaped, flags=re.IGNORECASE):
        decoded = urllib.parse.unquote(encoded)
        if decoded:
            found.append(decoded)

    return found


def search_urls_for_query(query: str, pages_per_engine: int) -> List[Tuple[str, str]]:
    encoded = urllib.parse.quote_plus(query)
    urls: List[Tuple[str, str]] = []

    for page_idx in range(pages_per_engine):
        ddg_offset = page_idx * 30
        urls.append(
            (
                "jina_duckduckgo_lite",
                f"https://r.jina.ai/http://lite.duckduckgo.com/lite/?q={encoded}&s={ddg_offset}",
            )
        )

    return urls


def run_discovery_search(
    queries: Sequence[str],
    pages_per_engine: int,
    timeout: int,
    request_sleep: float,
    excluded: Set[str],
) -> Tuple[List[SearchHit], Dict[str, object]]:
    hits: List[SearchHit] = []
    debug = {
        "query_stats": {},
        "source_stats": {},
        "failed_search_requests": 0,
        "successful_search_requests": 0,
    }

    for query in queries:
        query_key = query.strip()
        discovered_for_query = 0
        query_variants = [query_key]
        if "site:t.me" not in query_key.lower():
            query_variants.append(f"site:t.me {query_key}")

        for variant in query_variants:
            for source, url in search_urls_for_query(variant, pages_per_engine):
                source_stats = debug["source_stats"].setdefault(  # type: ignore[union-attr]
                    source, {"ok_requests": 0, "failed_requests": 0, "candidate_hits": 0}
                )

                page, status, err = fetch_url(url, timeout=timeout)
                if page is None and err:
                    debug["failed_search_requests"] = int(debug["failed_search_requests"]) + 1
                    source_stats["failed_requests"] += 1
                    continue
                if status and status >= 400:
                    debug["failed_search_requests"] = int(debug["failed_search_requests"]) + 1
                    source_stats["failed_requests"] += 1
                    continue

                debug["successful_search_requests"] = int(debug["successful_search_requests"]) + 1
                source_stats["ok_requests"] += 1
                for token in extract_candidate_links(page or ""):
                    normalized = normalize_channel_url(token, excluded=excluded)
                    if not normalized:
                        continue
                    handle, normalized_url = normalized
                    hits.append(
                        SearchHit(
                            handle=handle,
                            normalized_url=normalized_url,
                            query=query_key,
                            source=source,
                            raw_url=token,
                        )
                    )
                    discovered_for_query += 1
                    source_stats["candidate_hits"] += 1

                sleep_for = request_sleep + random.random() * (request_sleep * 0.8)
                time.sleep(sleep_for)

        debug["query_stats"][query_key] = {
            "candidate_hits": discovered_for_query,
        }

    return hits, debug


def extract_message_blocks(page: str) -> List[str]:
    blocks: List[str] = []
    for pattern in (
        r"<div class=\"tgme_widget_message_text[^\"]*\"[^>]*>(.*?)</div>",
        r"<div class=\"tgme_widget_message_caption[^\"]*\"[^>]*>(.*?)</div>",
    ):
        for match in re.finditer(pattern, page, flags=re.IGNORECASE | re.DOTALL):
            text = clean_text(match.group(1))
            if text:
                blocks.append(text)
    return blocks


def extract_last_datetime(page: str) -> Optional[str]:
    datetimes = re.findall(r"datetime=\"([^\"]+)\"", page)
    if not datetimes:
        return None
    try:
        return max(datetimes)
    except Exception:
        return datetimes[-1]


def guess_city(text: str) -> str:
    lowered = text.lower()
    best_city = "unknown"
    best_score = 0
    for city, aliases in CITY_ALIASES.items():
        score = 0
        for alias in aliases:
            alias_l = alias.lower()
            score += lowered.count(alias_l)
        if score > best_score:
            best_city = city
            best_score = score
    return best_city


def parse_number_tokens(text: str) -> List[int]:
    out: List[int] = []
    normalized = translit_digits(text)
    for match in NUMBER_PATTERN.finditer(normalized):
        token = re.sub(r"[^0-9]", "", match.group(0))
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def is_quote_post(text: str) -> bool:
    lowered = translit_digits(text).lower()
    has_keyword = any(keyword in lowered for keyword in QUOTE_KEYWORDS)
    nums = parse_number_tokens(text)
    if not nums:
        return False

    has_buy_sell = any(k in lowered for k in ("buy", "sell", "خرید", "فروش"))
    has_rate_word = any(k in lowered for k in ("نرخ", "قیمت", "rate", "quote"))
    return has_keyword and (has_buy_sell or has_rate_word or len(nums) >= 2)


def classify_channel(
    all_text: str,
    quote_posts: int,
    has_phone: bool,
    has_address: bool,
    has_website: bool,
    city_guess: str,
    linked_channel_count: int,
) -> str:
    lowered = all_text.lower()

    network_signal = any(word in lowered for word in NETWORK_KEYWORDS)
    aggregator_signal = any(word in lowered for word in AGGREGATOR_KEYWORDS)

    if network_signal and quote_posts >= 4:
        return "dealer_network_channel"

    if quote_posts >= 4 and (has_phone or has_address or has_website) and city_guess != "unknown":
        return "individual_exchange_shop"

    if linked_channel_count >= 8 and quote_posts <= 3:
        return "aggregator"

    if aggregator_signal and quote_posts >= 4:
        return "aggregator"

    if quote_posts >= 4:
        return "market_price_channel"

    return "unclear"


def compute_parseable_score(
    quote_posts: int,
    text_message_count: int,
    total_posts_visible: int,
    numeric_message_count: int,
) -> int:
    if total_posts_visible <= 0:
        return 0

    structure_ratio = min(1.0, quote_posts / max(1, text_message_count))
    numeric_ratio = min(1.0, numeric_message_count / max(1, text_message_count))
    text_ratio = min(1.0, text_message_count / max(1, total_posts_visible))
    frequency_ratio = min(1.0, total_posts_visible / 30.0)

    score = 100.0 * (
        0.40 * structure_ratio
        + 0.25 * numeric_ratio
        + 0.20 * text_ratio
        + 0.15 * frequency_ratio
    )

    if text_ratio < 0.30:
        score *= 0.75
    if quote_posts == 0:
        score *= 0.50

    return max(0, min(100, int(round(score))))


def crawl_channel(handle: str, timeout: int, excluded: Set[str]) -> ChannelSnapshot:
    public_url = f"https://t.me/s/{handle}"
    snapshot = ChannelSnapshot(handle=handle, public_url=public_url, crawled_at=now_iso())

    page, status, err = fetch_url(public_url, timeout=timeout)
    snapshot.http_status = status

    if page is None:
        snapshot.status = "error"
        snapshot.crawl_error = err or "fetch_failed"
        return snapshot

    if status and status >= 400:
        snapshot.status = "error"
        snapshot.crawl_error = err or f"http_{status}"
        snapshot.title = extract_meta_content(page, "og:title")
        return snapshot

    snapshot.title = extract_meta_content(page, "og:title") or extract_meta_content(page, "twitter:title")
    snapshot.status = "ok"

    text_messages = extract_message_blocks(page)
    snapshot.sample_messages = text_messages[:8]
    snapshot.last_seen_text_sample = text_messages[0][:240] if text_messages else ""
    snapshot.last_seen_datetime = extract_last_datetime(page)
    snapshot.text_message_count = len(text_messages)

    visible_posts = len(re.findall(r"tgme_widget_message_wrap", page))
    if visible_posts == 0:
        visible_posts = len(re.findall(r"tgme_widget_message\b", page))
    snapshot.total_posts_visible = visible_posts

    page_text = clean_text(page)
    combined_text = "\n".join(text_messages)
    all_text = f"{snapshot.title}\n{combined_text}\n{page_text[:12000]}"

    quote_posts = 0
    numeric_posts = 0
    for msg in text_messages:
        if parse_number_tokens(msg):
            numeric_posts += 1
        if is_quote_post(msg):
            quote_posts += 1

    snapshot.quote_post_count = quote_posts

    phones = PHONE_PATTERN.findall(all_text)
    snapshot.has_phone = bool(phones)

    lowered_all = all_text.lower()
    snapshot.has_address = any(word in lowered_all for word in ADDRESS_KEYWORDS)

    websites: List[str] = []
    for link in WEBSITE_PATTERN.findall(page):
        if "t.me/" in link.lower() or "telegram.me/" in link.lower():
            continue
        websites.append(link.strip())
    snapshot.website_links = sorted(set(websites))[:20]
    snapshot.has_website = bool(snapshot.website_links)

    linked_channels: Set[str] = set()
    for token in extract_candidate_links(page):
        normalized = normalize_channel_url(token, excluded=excluded)
        if not normalized:
            continue
        linked_channels.add(normalized[0])
    linked_channels.discard(handle)

    mentioned_handles: Set[str] = set()
    for m in MENTION_PATTERN.findall(all_text):
        if m.lower() == handle:
            continue
        if HANDLE_RE.match(m):
            mentioned_handles.add(m.lower())

    snapshot.linked_channels = sorted(linked_channels)
    snapshot.mentioned_handles = sorted(mentioned_handles)

    snapshot.city_guess = guess_city(all_text)
    snapshot.channel_type_guess = classify_channel(
        all_text=all_text,
        quote_posts=quote_posts,
        has_phone=snapshot.has_phone,
        has_address=snapshot.has_address,
        has_website=snapshot.has_website,
        city_guess=snapshot.city_guess,
        linked_channel_count=len(linked_channels) + len(mentioned_handles),
    )
    snapshot.likely_individual_shop = snapshot.channel_type_guess == "individual_exchange_shop"
    snapshot.parseable_score = compute_parseable_score(
        quote_posts=quote_posts,
        text_message_count=snapshot.text_message_count,
        total_posts_visible=snapshot.total_posts_visible,
        numeric_message_count=numeric_posts,
    )

    if snapshot.text_message_count == 0 and snapshot.status == "ok":
        snapshot.status = "ok_no_text"

    return snapshot


def dedupe_search_hits(hits: Iterable[SearchHit]) -> Dict[str, Dict[str, object]]:
    by_handle: Dict[str, Dict[str, object]] = {}
    for hit in hits:
        entry = by_handle.setdefault(
            hit.handle,
            {
                "handle": hit.handle,
                "public_url": hit.normalized_url,
                "discovered_queries": set(),
                "discovery_sources": set(),
                "raw_urls": set(),
            },
        )
        entry["discovered_queries"].add(hit.query)
        entry["discovery_sources"].add(hit.source)
        entry["raw_urls"].add(hit.raw_url)

    for item in by_handle.values():
        item["discovered_queries"] = sorted(item["discovered_queries"])
        item["discovery_sources"] = sorted(item["discovery_sources"])
        item["raw_urls"] = sorted(item["raw_urls"])

    return by_handle


def load_seed_handles(path: Path, excluded: Set[str]) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    entries: List[object]
    if isinstance(payload, list):
        entries = payload
    else:
        return {}

    discovered: Dict[str, Dict[str, object]] = {}
    for item in entries:
        handle: Optional[str] = None
        if isinstance(item, str):
            handle = item
        elif isinstance(item, dict):
            if isinstance(item.get("handle"), str):
                handle = item["handle"]
            elif isinstance(item.get("public_url"), str):
                normalized = normalize_channel_url(item["public_url"], excluded=excluded)
                if normalized:
                    handle = normalized[0]

        if not handle:
            continue
        handle = handle.strip().lower()
        if not HANDLE_RE.match(handle):
            continue
        if handle in excluded:
            continue
        discovered[handle] = {
            "handle": handle,
            "public_url": f"https://t.me/s/{handle}",
            "discovered_queries": [],
            "discovery_sources": ["seed_file"],
            "raw_urls": [f"https://t.me/{handle}"],
        }
    return discovered


def load_queries(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        q = line.strip()
        if not q or q.startswith("#"):
            continue
        out.append(q)
    return out


def write_outputs(
    out_dir: Path,
    discovered: Dict[str, Dict[str, object]],
    snapshots: Dict[str, ChannelSnapshot],
    crawl_errors: Dict[str, str],
    summary: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered_rows = []
    for handle in sorted(discovered.keys()):
        entry = dict(discovered[handle])
        snap = snapshots.get(handle)
        entry["title"] = snap.title if snap else ""
        entry["status"] = snap.status if snap else "not_crawled"
        entry["parseable_score"] = snap.parseable_score if snap else 0
        discovered_rows.append(entry)

    with (out_dir / "discovered_channels.json").open("w", encoding="utf-8") as fp:
        json.dump(discovered_rows, fp, ensure_ascii=False, indent=2)

    csv_path = out_dir / "channel_survey.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "public_url",
                "title",
                "city_guess",
                "channel_type_guess",
                "likely_individual_shop",
                "parseable_score",
                "quote_post_count",
                "has_phone",
                "has_address",
                "has_website",
                "last_seen_text_sample",
                "status",
            ],
        )
        writer.writeheader()
        for handle in sorted(discovered.keys()):
            snap = snapshots.get(handle)
            if not snap:
                writer.writerow(
                    {
                        "handle": handle,
                        "public_url": discovered[handle]["public_url"],
                        "title": "",
                        "city_guess": "unknown",
                        "channel_type_guess": "unclear",
                        "likely_individual_shop": False,
                        "parseable_score": 0,
                        "quote_post_count": 0,
                        "has_phone": False,
                        "has_address": False,
                        "has_website": False,
                        "last_seen_text_sample": "",
                        "status": "not_crawled",
                    }
                )
                continue

            writer.writerow(
                {
                    "handle": snap.handle,
                    "public_url": snap.public_url,
                    "title": snap.title,
                    "city_guess": snap.city_guess,
                    "channel_type_guess": snap.channel_type_guess,
                    "likely_individual_shop": snap.likely_individual_shop,
                    "parseable_score": snap.parseable_score,
                    "quote_post_count": snap.quote_post_count,
                    "has_phone": snap.has_phone,
                    "has_address": snap.has_address,
                    "has_website": snap.has_website,
                    "last_seen_text_sample": snap.last_seen_text_sample,
                    "status": snap.status,
                }
            )

    snapshot_rows = []
    for handle in sorted(snapshots.keys()):
        snap = snapshots[handle]
        snapshot_rows.append(
            {
                "handle": snap.handle,
                "public_url": snap.public_url,
                "title": snap.title,
                "status": snap.status,
                "http_status": snap.http_status,
                "crawl_error": snap.crawl_error,
                "city_guess": snap.city_guess,
                "channel_type_guess": snap.channel_type_guess,
                "likely_individual_shop": snap.likely_individual_shop,
                "parseable_score": snap.parseable_score,
                "quote_post_count": snap.quote_post_count,
                "total_posts_visible": snap.total_posts_visible,
                "text_message_count": snap.text_message_count,
                "has_phone": snap.has_phone,
                "has_address": snap.has_address,
                "has_website": snap.has_website,
                "website_links": snap.website_links,
                "linked_channels": snap.linked_channels,
                "mentioned_handles": snap.mentioned_handles,
                "sample_messages": snap.sample_messages,
                "last_seen_text_sample": snap.last_seen_text_sample,
                "last_seen_datetime": snap.last_seen_datetime,
                "crawled_at": snap.crawled_at,
            }
        )

    with (out_dir / "channel_snapshots.json").open("w", encoding="utf-8") as fp:
        json.dump(snapshot_rows, fp, ensure_ascii=False, indent=2)

    payload = dict(summary)
    payload["crawl_errors_by_handle"] = crawl_errors
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def compute_summary(
    discovered: Dict[str, Dict[str, object]],
    snapshots: Dict[str, ChannelSnapshot],
) -> Dict[str, object]:
    channels_with_quote_posts = 0
    likely_individual_exchange_shops = 0
    market_channels = 0
    parseable_feeds_above_60_score = 0

    by_type: Dict[str, int] = {}
    by_city: Dict[str, int] = {}

    for handle in discovered:
        snap = snapshots.get(handle)
        if not snap:
            continue

        if snap.quote_post_count > 0:
            channels_with_quote_posts += 1
        if snap.likely_individual_shop:
            likely_individual_exchange_shops += 1
        if snap.channel_type_guess == "market_price_channel":
            market_channels += 1
        if snap.parseable_score >= 60:
            parseable_feeds_above_60_score += 1

        by_type[snap.channel_type_guess] = by_type.get(snap.channel_type_guess, 0) + 1
        by_city[snap.city_guess] = by_city.get(snap.city_guess, 0) + 1

    return {
        "generated_at": now_iso(),
        "total_channels_discovered": len(discovered),
        "channels_with_quote_posts": channels_with_quote_posts,
        "likely_individual_exchange_shops": likely_individual_exchange_shops,
        "market_channels": market_channels,
        "parseable_feeds_above_60_score": parseable_feeds_above_60_score,
        "channel_type_distribution": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "city_distribution": dict(sorted(by_city.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey public Telegram FX quote channels.")
    parser.add_argument("--out-dir", default="survey_outputs", help="Output directory")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout (seconds)")
    parser.add_argument("--pages-per-engine", type=int, default=2, help="Search pages per query/engine")
    parser.add_argument("--request-sleep", type=float, default=1.1, help="Base seconds sleep between requests")
    parser.add_argument("--max-channels", type=int, default=400, help="Maximum channels to crawl")
    parser.add_argument(
        "--seed-handles-file",
        default="",
        help="Optional JSON file of seed handles or discovered channel objects",
    )
    parser.add_argument(
        "--query-file",
        default="",
        help="Optional text file with one search query per line",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip search engine discovery and start from seed handles",
    )
    args = parser.parse_args()

    excluded = set(DEFAULT_EXCLUDE_HANDLES)
    search_debug: Dict[str, object] = {
        "query_stats": {},
        "source_stats": {},
        "failed_search_requests": 0,
        "successful_search_requests": 0,
    }

    discovered: Dict[str, Dict[str, object]] = {}
    hits: List[SearchHit] = []

    if args.seed_handles_file:
        seed_discovered = load_seed_handles(Path(args.seed_handles_file), excluded=excluded)
        discovered.update(seed_discovered)
        print(f"Loaded {len(seed_discovered)} seed handles from {args.seed_handles_file}")

    if not args.skip_search:
        query_list = DEFAULT_SEARCH_QUERIES
        if args.query_file:
            loaded_queries = load_queries(Path(args.query_file))
            if loaded_queries:
                query_list = loaded_queries
                print(f"Loaded {len(query_list)} queries from {args.query_file}")

        print("[1/4] Running search discovery...")
        hits, search_debug = run_discovery_search(
            queries=query_list,
            pages_per_engine=args.pages_per_engine,
            timeout=args.timeout,
            request_sleep=args.request_sleep,
            excluded=excluded,
        )

        discovered_from_search = dedupe_search_hits(hits)
        for handle, entry in discovered_from_search.items():
            if handle not in discovered:
                discovered[handle] = entry
                continue
            existing = discovered[handle]
            for key in ("discovered_queries", "discovery_sources", "raw_urls"):
                merged = sorted(set(existing.get(key, [])) | set(entry.get(key, [])))
                existing[key] = merged
        print(f"Discovered {len(discovered_from_search)} unique channels from search hits={len(hits)}")
    else:
        print("[1/4] Search discovery skipped (--skip-search)")

    # Crawl expansion queue starts with discovered handles and grows from channel cross-links.
    queue: List[str] = sorted(discovered.keys())
    snapshots: Dict[str, ChannelSnapshot] = {}
    crawl_errors: Dict[str, str] = {}

    print("[2/4] Crawling channel pages and expanding graph...")
    idx = 0
    while idx < len(queue) and len(snapshots) < args.max_channels:
        handle = queue[idx]
        idx += 1

        if handle in snapshots:
            continue

        snap = crawl_channel(handle, timeout=args.timeout, excluded=excluded)
        snapshots[handle] = snap

        if snap.status.startswith("error"):
            crawl_errors[handle] = snap.crawl_error or "error"

        # Expand candidate universe from links and @mentions.
        discovered.setdefault(
            handle,
            {
                "handle": handle,
                "public_url": f"https://t.me/s/{handle}",
                "discovered_queries": [],
                "discovery_sources": ["channel_expansion"],
                "raw_urls": [f"https://t.me/{handle}"],
            },
        )

        for related in snap.linked_channels + snap.mentioned_handles:
            if related in excluded or not HANDLE_RE.match(related):
                continue
            if related not in discovered:
                discovered[related] = {
                    "handle": related,
                    "public_url": f"https://t.me/s/{related}",
                    "discovered_queries": [],
                    "discovery_sources": ["channel_expansion"],
                    "raw_urls": [f"https://t.me/{related}"],
                }
                queue.append(related)
            else:
                srcs = discovered[related].setdefault("discovery_sources", [])
                if "channel_expansion" not in srcs:
                    srcs.append("channel_expansion")

        if idx % 25 == 0:
            print(f"  Crawled {idx} / queue {len(queue)} (snapshots={len(snapshots)})")

        sleep_for = args.request_sleep + random.random() * (args.request_sleep * 0.8)
        time.sleep(sleep_for)

    print("[3/4] Computing summary and writing outputs...")
    summary = compute_summary(discovered=discovered, snapshots=snapshots)
    summary["search_debug"] = search_debug
    summary["channels_crawled"] = len(snapshots)
    summary["crawl_error_count"] = len(crawl_errors)

    out_dir = Path(args.out_dir)
    write_outputs(out_dir=out_dir, discovered=discovered, snapshots=snapshots, crawl_errors=crawl_errors, summary=summary)

    print("[4/4] Completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
