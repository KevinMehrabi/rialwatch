#!/usr/bin/env python3
"""Expand likely inside-Iran direct-shop Telegram channels (research mode).

This script discovers and scores additional Telegram channels likely operated by
individual exchange shops (sarafi), then merges against existing survey outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

DEFAULT_QUERIES = [
    "site:t.me صرافی",
    "site:t.me/s صرافی",
    "صرافی تهران",
    "صرافی مشهد",
    "صرافی اصفهان",
    "صرافی شیراز",
    "صرافی تبریز",
    "صرافی کرج",
    "صرافی قم",
    "صرافی رشت",
    "صرافی بندرعباس",
    "نرخ دلار صرافی",
    "قیمت دلار صرافی",
    "sarafi telegram",
    "exchange telegram iran",
    "currency exchange iran telegram",
    "حواله صرافی",
    "حواله دلار صرافی",
]

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

HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{5,}$")
RAW_TME_RE = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
BARE_TME_RE = re.compile(r"(?:^|\s)(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d|09\d{9}|\+98\s?9\d{9})")
MAP_RE = re.compile(r"https?://(?:maps\.google|goo\.gl/maps|nshn\.ir|balad\.ir|map\.ir)[^\s\"'<>]*", re.IGNORECASE)
ADDR_WORDS = (
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
SHOP_WORDS = ("صرافی", "حواله", "exchange", "sarafi", "currency", "خرید", "فروش")
COMMENTARY_WORDS = (
    "تحلیل",
    "analysis",
    "news",
    "اخبار",
    "سیگنال",
    "signal",
    "forex",
    "crypto",
    "bitcoin",
)
QUOTE_WORDS = ("دلار", "usd", "یورو", "eur", "نرخ", "قیمت", "buy", "sell", "خرید", "فروش")

NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,7})(?!\d)")
BUY_WORDS = ("خرید", "buy", "bid")
SELL_WORDS = ("فروش", "sell", "offer", "ask")

CITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Mashhad": ("mashhad", "مشهد"),
    "Isfahan": ("isfahan", "اصفهان"),
    "Shiraz": ("shiraz", "شیراز"),
    "Tabriz": ("tabriz", "تبریز"),
    "Karaj": ("karaj", "کرج"),
    "Qom": ("qom", "قم"),
    "Rasht": ("rasht", "رشت"),
    "BandarAbbas": ("bandar abbas", "bandarabbas", "بندرعباس", "بندر عباس"),
    "Kerman": ("kerman", "کرمان"),
    "Ahvaz": ("ahvaz", "اهواز"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
    "Frankfurt": ("frankfurt", "فرانکفورت"),
    "Hamburg": ("hamburg", "هامبورگ"),
    "London": ("london", "لندن"),
}

INSIDE_IRAN_CITIES = {
    "Tehran",
    "Mashhad",
    "Isfahan",
    "Shiraz",
    "Tabriz",
    "Karaj",
    "Qom",
    "Rasht",
    "BandarAbbas",
    "Kerman",
    "Ahvaz",
}

DIASPORA_CITIES = {"Dubai", "Istanbul", "Frankfurt", "Hamburg", "London"}


@dataclass
class Candidate:
    handle: str
    public_url: str
    query_hits: Set[str]
    raw_urls: Set[str]


@dataclass
class ChannelScore:
    handle: str
    public_url: str
    title: str
    city_guess: str
    quote_post_count: int
    buy_sell_pair_count: int
    parseability_score: int
    likely_individual_shop: bool
    has_phone: bool
    has_address: bool
    has_map_link: bool
    has_shop_name: bool
    status: str
    commentary_heavy: bool
    last_seen_text_sample: str
    discovery_queries: List[str]


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


def normalize_channel_url(raw: str) -> Optional[Tuple[str, str]]:
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

    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return None

    if parts[0].lower() == "s" and len(parts) >= 2:
        handle = parts[1]
    else:
        handle = parts[0]

    handle = handle.strip().lower()
    if not handle or handle.startswith("+"):
        return None
    if handle in EXCLUDED_HANDLES:
        return None
    if not HANDLE_RE.match(handle):
        return None

    return handle, f"https://t.me/s/{handle}"


def extract_candidate_links(page: str) -> List[str]:
    found: List[str] = []
    unescaped = html.unescape(page)

    for blob in (unescaped, urllib.parse.unquote(unescaped)):
        for pat in (RAW_TME_RE, BARE_TME_RE):
            for m in pat.finditer(blob):
                token = m.group(0).strip()
                if token:
                    found.append(token)

    for encoded in re.findall(r"uddg=([^&\"'<>\s]+)", unescaped, flags=re.IGNORECASE):
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


def run_discovery(
    queries: Sequence[str],
    pages_per_query: int,
    timeout: int,
    request_sleep: float,
) -> Tuple[Dict[str, Candidate], Dict[str, object]]:
    out: Dict[str, Candidate] = {}
    debug = {
        "query_stats": {},
        "successful_requests": 0,
        "failed_requests": 0,
    }

    for query in queries:
        hits = 0
        for url in search_urls_for_query(query, pages_per_query):
            page, status, err = fetch_url(url, timeout=timeout)
            if page is None or (status is not None and status >= 400):
                debug["failed_requests"] += 1
                continue

            debug["successful_requests"] += 1
            for token in extract_candidate_links(page):
                norm = normalize_channel_url(token)
                if not norm:
                    continue
                handle, public_url = norm
                cand = out.get(handle)
                if cand is None:
                    cand = Candidate(handle=handle, public_url=public_url, query_hits=set(), raw_urls=set())
                    out[handle] = cand
                cand.query_hits.add(query)
                cand.raw_urls.add(token)
                hits += 1

            if request_sleep > 0:
                time.sleep(request_sleep + random.random() * request_sleep * 0.5)

        debug["query_stats"][query] = {"candidate_hits": hits}

    return out, debug


def extract_message_blocks(page: str) -> List[str]:
    blocks: List[Tuple[int, str]] = []
    for patt in (
        r'<div class="tgme_widget_message_text[^\"]*"[^>]*>(.*?)</div>',
        r'<div class="tgme_widget_message_caption[^\"]*"[^>]*>(.*?)</div>',
    ):
        for m in re.finditer(patt, page, flags=re.IGNORECASE | re.DOTALL):
            txt = clean_text(m.group(1))
            if txt:
                blocks.append((m.start(), txt))
    blocks.sort(key=lambda x: x[0])
    return [x[1] for x in blocks]


def extract_meta_content(page: str, prop: str) -> str:
    patt = re.compile(
        rf"<meta[^>]+(?:property|name)=[\"']{re.escape(prop)}[\"'][^>]+content=[\"'](.*?)[\"']",
        re.IGNORECASE,
    )
    m = patt.search(page)
    return html.unescape(m.group(1)).strip() if m else ""


def detect_city(text: str) -> str:
    lowered = text.lower()
    best = "unknown"
    score = 0
    for city, aliases in CITY_ALIASES.items():
        c = sum(lowered.count(alias.lower()) for alias in aliases)
        if c > score:
            score = c
            best = city
    return best


def parse_int_token(token: str) -> Optional[int]:
    cleaned = translit_digits(token)
    cleaned = re.sub(r"[^0-9]", "", cleaned)
    if not cleaned:
        return None
    try:
        n = int(cleaned)
    except ValueError:
        return None
    return n if n > 0 else None


def extract_numbers(text: str) -> List[int]:
    vals: List[int] = []
    for m in NUMBER_RE.finditer(translit_digits(text)):
        n = parse_int_token(m.group(0))
        if n is not None:
            vals.append(n)
    return vals


def find_keyword_number(text: str, words: Sequence[str]) -> Optional[int]:
    normalized = translit_digits(text).lower()
    num_pat = NUMBER_RE.pattern
    for word in words:
        patt = re.compile(rf"{re.escape(word)}[^0-9]{{0,24}}({num_pat})|({num_pat})[^0-9]{{0,24}}{re.escape(word)}", re.IGNORECASE)
        m = patt.search(normalized)
        if not m:
            continue
        tok = m.group(1) or m.group(2)
        if tok:
            n = parse_int_token(tok)
            if n is not None:
                return n
    return None


def quote_like(msg: str) -> Tuple[bool, bool]:
    lowered = translit_digits(msg).lower()
    nums = extract_numbers(msg)
    has_quote_word = any(w in lowered for w in QUOTE_WORDS)
    buy = find_keyword_number(msg, BUY_WORDS)
    sell = find_keyword_number(msg, SELL_WORDS)
    has_pair = bool(buy is not None and sell is not None)
    is_quote = bool(nums and (has_quote_word or has_pair))
    return is_quote, has_pair


def score_channel(candidate: Candidate, timeout: int) -> ChannelScore:
    page, status, err = fetch_url(candidate.public_url, timeout=timeout)
    title = ""
    city_guess = "unknown"
    quote_posts = 0
    pair_posts = 0
    parseability = 0
    likely_shop = False
    has_phone = False
    has_address = False
    has_map = False
    has_shop_name = False
    commentary_heavy = False
    sample = ""
    status_label = "ok"

    if page is None:
        return ChannelScore(
            handle=candidate.handle,
            public_url=candidate.public_url,
            title="",
            city_guess="unknown",
            quote_post_count=0,
            buy_sell_pair_count=0,
            parseability_score=0,
            likely_individual_shop=False,
            has_phone=False,
            has_address=False,
            has_map_link=False,
            has_shop_name=False,
            status=f"error:{err or 'fetch_failed'}",
            commentary_heavy=False,
            last_seen_text_sample="",
            discovery_queries=sorted(candidate.query_hits),
        )

    if status is not None and status >= 400:
        status_label = f"error:http_{status}"

    title = extract_meta_content(page, "og:title") or extract_meta_content(page, "twitter:title")
    blocks = extract_message_blocks(page)
    if blocks:
        sample = blocks[0][:260]

    all_text = "\n".join(blocks)
    text_with_title = f"{title}\n{all_text}".strip()

    has_phone = bool(PHONE_RE.search(text_with_title))
    has_map = bool(MAP_RE.search(page))
    has_address = any(w in text_with_title.lower() for w in [x.lower() for x in ADDR_WORDS])
    has_shop_name = "صرافی" in text_with_title or "sarafi" in text_with_title.lower() or "exchange" in text_with_title.lower()

    commentary_hits = sum(text_with_title.lower().count(w.lower()) for w in COMMENTARY_WORDS)
    shop_hits = sum(text_with_title.lower().count(w.lower()) for w in SHOP_WORDS)
    commentary_heavy = commentary_hits > max(4, shop_hits * 1.3)

    for msg in blocks:
        is_quote, has_pair = quote_like(msg)
        if is_quote:
            quote_posts += 1
        if has_pair:
            pair_posts += 1

    city_guess = detect_city(text_with_title)

    total_msgs = len(blocks)
    if total_msgs > 0:
        quote_ratio = quote_posts / total_msgs
        pair_ratio = pair_posts / total_msgs
        parseability = int(round(100 * (0.45 * min(1.0, quote_ratio * 2.4) + 0.35 * min(1.0, pair_ratio * 3.2) + 0.20 * min(1.0, total_msgs / 20))))
    else:
        parseability = 0

    direct_signals = sum(
        [
            1 if has_phone else 0,
            1 if has_address else 0,
            1 if has_map else 0,
            1 if has_shop_name else 0,
            1 if city_guess in INSIDE_IRAN_CITIES else 0,
            1 if pair_posts >= 2 else 0,
            1 if quote_posts >= 4 else 0,
        ]
    )

    likely_shop = bool(direct_signals >= 4 and not commentary_heavy)

    return ChannelScore(
        handle=candidate.handle,
        public_url=candidate.public_url,
        title=title,
        city_guess=city_guess,
        quote_post_count=quote_posts,
        buy_sell_pair_count=pair_posts,
        parseability_score=parseability,
        likely_individual_shop=likely_shop,
        has_phone=has_phone,
        has_address=has_address,
        has_map_link=has_map,
        has_shop_name=has_shop_name,
        status=status_label,
        commentary_heavy=commentary_heavy,
        last_seen_text_sample=sample,
        discovery_queries=sorted(candidate.query_hits),
    )


def load_existing_registry(path: Path) -> Tuple[Set[str], int]:
    handles: Set[str] = set()
    inside_shop_count = 0

    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            handle = str(row.get("handle", "")).strip().lower()
            if handle:
                handles.add(handle)

            likely_shop = str(row.get("likely_individual_shop", "")).strip().lower() == "true"
            city = str(row.get("city_guess", "unknown")).strip()
            if likely_shop and city in INSIDE_IRAN_CITIES:
                inside_shop_count += 1

    return handles, inside_shop_count


def estimate_additional_usable_records(rows: Sequence[ChannelScore]) -> int:
    total = 0
    for r in rows:
        if not r.likely_individual_shop:
            continue
        if r.city_guess not in INSIDE_IRAN_CITIES:
            continue

        score = 0
        if r.parseability_score >= 75:
            score += 8
        elif r.parseability_score >= 60:
            score += 6
        elif r.parseability_score >= 45:
            score += 3

        if r.buy_sell_pair_count >= 6:
            score += 5
        elif r.buy_sell_pair_count >= 3:
            score += 3
        elif r.buy_sell_pair_count >= 1:
            score += 1

        if r.quote_post_count >= 12:
            score += 4
        elif r.quote_post_count >= 6:
            score += 2
        elif r.quote_post_count >= 3:
            score += 1

        total += max(0, min(15, score))

    return total


def write_candidates(path: Path, rows: Sequence[ChannelScore]) -> None:
    ordered = sorted(rows, key=lambda r: (-int(r.likely_individual_shop), -r.parseability_score, -r.buy_sell_pair_count, r.handle))
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "public_url",
                "title",
                "city_guess",
                "quote_post_count",
                "buy_sell_pair_count",
                "parseability_score",
                "likely_individual_shop",
                "has_phone",
                "has_address",
                "has_map_link",
                "has_shop_name",
                "commentary_heavy",
                "status",
                "discovery_queries",
                "last_seen_text_sample",
            ],
        )
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    "handle": r.handle,
                    "public_url": r.public_url,
                    "title": r.title,
                    "city_guess": r.city_guess,
                    "quote_post_count": r.quote_post_count,
                    "buy_sell_pair_count": r.buy_sell_pair_count,
                    "parseability_score": r.parseability_score,
                    "likely_individual_shop": r.likely_individual_shop,
                    "has_phone": r.has_phone,
                    "has_address": r.has_address,
                    "has_map_link": r.has_map_link,
                    "has_shop_name": r.has_shop_name,
                    "commentary_heavy": r.commentary_heavy,
                    "status": r.status,
                    "discovery_queries": "|".join(r.discovery_queries),
                    "last_seen_text_sample": r.last_seen_text_sample,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand inside-Iran direct-shop Telegram channels")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Survey output directory")
    parser.add_argument("--pages-per-query", type=int, default=1, help="Search pages per query")
    parser.add_argument("--max-crawl", type=int, default=220, help="Max new channels to crawl")
    parser.add_argument("--timeout", type=int, default=18, help="HTTP timeout")
    parser.add_argument("--sleep", type=float, default=0.45, help="Sleep between requests")
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    existing_csv = survey_dir / "channel_survey.csv"
    if not existing_csv.exists():
        raise SystemExit(f"missing required file: {existing_csv}")

    candidates_csv = survey_dir / "direct_shop_expansion_candidates.csv"
    summary_json = survey_dir / "direct_shop_expansion_summary.json"

    existing_handles, existing_inside_shop_count = load_existing_registry(existing_csv)

    print("[1/4] Running discovery expansion...")
    discovered, search_debug = run_discovery(
        queries=DEFAULT_QUERIES,
        pages_per_query=args.pages_per_query,
        timeout=args.timeout,
        request_sleep=args.sleep,
    )

    # Keep only handles not already present in existing registry.
    new_candidates = [c for h, c in discovered.items() if h not in existing_handles]
    new_candidates.sort(key=lambda c: (len(c.query_hits), c.handle), reverse=True)
    if args.max_crawl > 0:
        new_candidates = new_candidates[: args.max_crawl]

    print(f"Discovered candidate handles (new vs existing): {len(new_candidates)}")

    print("[2/4] Crawling and scoring candidate channels...")
    scored: List[ChannelScore] = []
    for idx, cand in enumerate(new_candidates, start=1):
        scored.append(score_channel(cand, timeout=args.timeout))
        if idx % 20 == 0:
            print(f"  Scored {idx}/{len(new_candidates)}")
        if args.sleep > 0:
            time.sleep(args.sleep + random.random() * args.sleep * 0.4)

    # Keep channels that look like exchange-shop candidates or are quote-active.
    filtered = [
        r
        for r in scored
        if (
            r.likely_individual_shop
            or (r.quote_post_count >= 3 and r.parseability_score >= 45 and not r.commentary_heavy)
            or (r.buy_sell_pair_count >= 2 and not r.commentary_heavy)
        )
    ]

    print("[3/4] Writing candidate dataset...")
    write_candidates(candidates_csv, filtered)

    likely_direct_shops = [r for r in filtered if r.likely_individual_shop]
    with_pairs = [r for r in filtered if r.buy_sell_pair_count > 0]
    inside_iran_likely = [r for r in likely_direct_shops if r.city_guess in INSIDE_IRAN_CITIES]

    additional_usable_est = estimate_additional_usable_records(filtered)
    total_inside_now = existing_inside_shop_count + len(inside_iran_likely)

    summary = {
        "generated_at": now_iso(),
        "search_debug": search_debug,
        "existing_registry_handles": len(existing_handles),
        "existing_inside_iran_likely_shop_channels": existing_inside_shop_count,
        "new_channels_discovered": len(new_candidates),
        "new_channels_retained_after_scoring": len(filtered),
        "likely_direct_shop_channels": len(likely_direct_shops),
        "inside_iran_likely_direct_shop_channels": len(inside_iran_likely),
        "channels_with_buy_sell_quotes": len(with_pairs),
        "total_inside_iran_shop_channels_now_known": total_inside_now,
        "estimated_additional_usable_records_if_ingested": additional_usable_est,
        "target_progress_note": "aiming toward 80-150 usable inside-Iran dealer records",
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] Completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
