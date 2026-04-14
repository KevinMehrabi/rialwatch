#!/usr/bin/env python3
"""Iraq multi-surface source discovery for RialWatch diagnostics.

Expands beyond Telegram-first search to discover Iraq/Erbil/Sulaymaniyah/Baghdad
exchange and remittance sources across web and social surfaces, then emits
Telegram seed handles for deeper crawl validation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
SURVEY_DIR = ROOT_DIR / "survey_outputs"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
RAW_TME_RE = re.compile(r"https?://(?:www\.)?(?:t\.me|telegram\.me)/[^\s\"'<>]+", re.IGNORECASE)
RAW_IG_RE = re.compile(r"https?://(?:www\.)?instagram\.com/[^\s\"'<>]+", re.IGNORECASE)
RAW_FB_RE = re.compile(r"https?://(?:www\.)?(?:facebook\.com|fb\.com)/[^\s\"'<>]+", re.IGNORECASE)
RAW_WA_RE = re.compile(r"https?://(?:wa\.me|api\.whatsapp\.com)/[^\s\"'<>]+", re.IGNORECASE)
HANDLE_RE = re.compile(r"^[A-Za-z0-9_\.]{4,64}$")
NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{4,8})(?!\d)")

CITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Sulaymaniyah": ("سلیمانیه", "سليمانية", "سلێمانی", "sulaymaniyah", "sulaimaniyah"),
    "Erbil": ("اربیل", "اربيل", "هولیر", "هولير", "هەولێر", "erbil", "hewler", "hawler"),
    "Baghdad": ("بغداد", "baghdad"),
}

EXCHANGE_WORDS = (
    "صرافی",
    "صرافة",
    "exchange",
    "currency exchange",
    "bureau de change",
    "دراو",
)
REMITTANCE_WORDS = (
    "حواله",
    "تحويل",
    "remittance",
    "money transfer",
    "hawala",
)
RATE_WORDS = (
    "دلار",
    "usd",
    "سعر",
    "نرخ",
    "price",
    "rate",
    "borse",
    "بورس",
)

DEFAULT_QUERIES: Tuple[str, ...] = (
    "صرافی سلیمانیه",
    "صرافی اربیل",
    "صرافی بغداد",
    "نرخ دلار سلیمانیه",
    "نرخ دلار اربیل",
    "نرخ دلار بغداد",
    "حواله عراق",
    "site:t.me دلار سلیمانیه",
    "site:t.me/s دلار سلیمانیه",
    "site:t.me دلار اربیل",
    "site:t.me/s دلار اربیل",
    "site:t.me بغداد دلار",
    "site:t.me/s بغداد دلار",
    "صرافة السليمانية",
    "صرافة اربيل",
    "صرافة بغداد",
    "سعر الدولار السليمانية",
    "سعر الدولار اربيل",
    "سعر الدولار بغداد",
    "تحويل اموال العراق",
    "telegram sulaymaniyah dollar",
    "telegram erbil dollar",
    "telegram baghdad dollar",
    "iraq exchange remittance",
    "iraqi dinar exchange",
    "site:instagram.com صرافی اربیل",
    "site:instagram.com صرافة بغداد",
    "site:facebook.com صرافة بغداد",
    "site:facebook.com erbil exchange",
)


@dataclass
class Candidate:
    url: str
    source_queries: Set[str]
    source_engines: Set[str]


@dataclass
class CandidateRow:
    source_url: str
    title: str
    city_guess: str
    source_type_guess: str
    has_rate_signal: bool
    has_exchange_terms: bool
    has_remittance_terms: bool
    telegram_handles: List[str]
    instagram_urls: List[str]
    facebook_urls: List[str]
    whatsapp_urls: List[str]
    candidate_score: int
    status: str
    last_seen: str


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(text: str) -> str:
    out = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    out = out.replace("\u200c", " ").replace("،", ",").replace("٬", ",")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


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
            body = resp.read().decode("utf-8", errors="replace")
            return body, int(resp.status), None
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = None
        return body, int(exc.code), f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return None, None, f"network_error:{exc.reason}"
    except socket.timeout:
        return None, None, "timeout"
    except TimeoutError:
        return None, None, "timeout"
    except OSError as exc:
        return None, None, f"os_error:{exc}"


def search_urls_for_query(query: str, pages: int) -> List[Tuple[str, str]]:
    encoded = urllib.parse.quote_plus(query)
    out: List[Tuple[str, str]] = []
    for idx in range(max(1, pages)):
        offset = idx * 30
        out.append(("duckduckgo", f"https://r.jina.ai/http://lite.duckduckgo.com/lite/?q={encoded}&s={offset}"))
        brave_query = urllib.parse.quote(query, safe="")
        brave_path = f"search.brave.com/search%3Fq%3D{brave_query}%26source%3Dweb%26offset%3D{offset}%26spellcheck%3D0"
        out.append(("brave", f"https://r.jina.ai/http://{brave_path}"))
    return out


def normalize_url(raw: str) -> Optional[str]:
    token = raw.strip().strip("'\"<>()[]{}.,;")
    if not token:
        return None
    try:
        parsed = urllib.parse.urlparse(token)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"https://{host}{path}"


def normalize_telegram_handle(raw: str) -> Optional[str]:
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
    handle = re.sub(r"[^A-Za-z0-9_\.]", "", handle.strip().lower())
    if not handle or not HANDLE_RE.match(handle):
        return None
    if handle in {"joinchat", "contact", "share", "login", "s"}:
        return None
    return handle


def extract_links(html_text: str) -> List[str]:
    out: List[str] = []
    unescaped = html.unescape(html_text)
    decoded = urllib.parse.unquote(unescaped)
    for blob in (unescaped, decoded):
        for patt in (RAW_TME_RE, RAW_IG_RE, RAW_FB_RE, RAW_WA_RE, URL_RE):
            for match in patt.finditer(blob):
                token = match.group(0).strip()
                if token:
                    out.append(token)
        for href in re.findall(r'href=["\']([^"\']+)["\']', blob, flags=re.IGNORECASE):
            if href:
                out.append(href.strip())
        for encoded in re.findall(r"uddg=([^&\"'<>\\s]+)", blob, flags=re.IGNORECASE):
            out.append(urllib.parse.unquote(encoded))
    return out


def text_from_html(page: str) -> str:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL)
    title = html.unescape(title_match.group(1)).strip() if title_match else ""
    text = re.sub(r"<br\\s*/?>", "\n", page, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = normalize_text(text)
    return normalize_text(f"{title} {text}")


def guess_city(text: str) -> str:
    lowered = normalize_text(text).lower()
    best = "unknown"
    best_score = 0
    for city, aliases in CITY_ALIASES.items():
        score = sum(lowered.count(alias.lower()) for alias in aliases)
        if score > best_score:
            best = city
            best_score = score
    return best


def source_type_guess(*, has_exchange: bool, has_remit: bool, has_rate: bool, city_guess: str) -> str:
    if has_exchange and has_rate:
        return "exchange_shop"
    if has_remit and city_guess in {"Erbil", "Sulaymaniyah", "Baghdad"}:
        return "remittance_exchange"
    if has_rate:
        return "regional_market_channel"
    return "unknown"


def parse_candidate(url: str, page: Optional[str], status: Optional[int], err: Optional[str]) -> CandidateRow:
    if page is None:
        return CandidateRow(
            source_url=url,
            title="",
            city_guess="unknown",
            source_type_guess="unknown",
            has_rate_signal=False,
            has_exchange_terms=False,
            has_remittance_terms=False,
            telegram_handles=[],
            instagram_urls=[],
            facebook_urls=[],
            whatsapp_urls=[],
            candidate_score=0,
            status=err or f"http_{status}" if status else "fetch_failed",
            last_seen="",
        )

    text = text_from_html(page)
    lowered = text.lower()
    has_exchange = any(word in lowered for word in EXCHANGE_WORDS)
    has_remit = any(word in lowered for word in REMITTANCE_WORDS)
    has_rate = any(word in lowered for word in RATE_WORDS) and bool(NUMBER_RE.search(lowered))
    city = guess_city(lowered)
    links = extract_links(page)

    tg_handles: Set[str] = set()
    ig_urls: Set[str] = set()
    fb_urls: Set[str] = set()
    wa_urls: Set[str] = set()
    for token in links:
        tg = normalize_telegram_handle(token)
        if tg:
            tg_handles.add(tg)
        norm = normalize_url(token)
        if not norm:
            continue
        if "instagram.com/" in norm:
            ig_urls.add(norm)
        elif "facebook.com/" in norm or "fb.com/" in norm:
            fb_urls.add(norm)
        elif "wa.me/" in norm or "api.whatsapp.com/" in norm:
            wa_urls.add(norm)

    source_type = source_type_guess(
        has_exchange=has_exchange,
        has_remit=has_remit,
        has_rate=has_rate,
        city_guess=city,
    )

    score = 0
    if has_rate:
        score += 34
    if has_exchange:
        score += 20
    if has_remit:
        score += 14
    score += min(10, len(tg_handles) * 5)
    score += min(8, len(wa_urls) * 4)
    score += min(6, len(ig_urls) * 3)
    score += min(6, len(fb_urls) * 2)
    if city != "unknown":
        score += 7
    score = max(0, min(100, score))

    last_seen = ""
    for tag in ("article:published_time", "og:updated_time"):
        m = re.search(
            rf"<meta[^>]+(?:property|name)=['\"]{re.escape(tag)}['\"][^>]+content=['\"](.*?)['\"]",
            page,
            flags=re.IGNORECASE,
        )
        if m:
            last_seen = m.group(1).strip()
            break

    return CandidateRow(
        source_url=url,
        title=(re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL).group(1).strip() if re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL) else "")[:160],
        city_guess=city,
        source_type_guess=source_type,
        has_rate_signal=has_rate,
        has_exchange_terms=has_exchange,
        has_remittance_terms=has_remit,
        telegram_handles=sorted(tg_handles),
        instagram_urls=sorted(ig_urls),
        facebook_urls=sorted(fb_urls),
        whatsapp_urls=sorted(wa_urls),
        candidate_score=score,
        status="ok" if (status is None or status < 400) else f"http_{status}",
        last_seen=last_seen,
    )


def write_csv(path: Path, rows: Sequence[CandidateRow]) -> None:
    fields = [
        "source_url",
        "title",
        "city_guess",
        "source_type_guess",
        "has_rate_signal",
        "has_exchange_terms",
        "has_remittance_terms",
        "telegram_handles",
        "instagram_urls",
        "facebook_urls",
        "whatsapp_urls",
        "candidate_score",
        "status",
        "last_seen",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (-r.candidate_score, r.city_guess, r.source_url)):
            writer.writerow(
                {
                    "source_url": row.source_url,
                    "title": row.title,
                    "city_guess": row.city_guess,
                    "source_type_guess": row.source_type_guess,
                    "has_rate_signal": row.has_rate_signal,
                    "has_exchange_terms": row.has_exchange_terms,
                    "has_remittance_terms": row.has_remittance_terms,
                    "telegram_handles": "|".join(row.telegram_handles),
                    "instagram_urls": "|".join(row.instagram_urls),
                    "facebook_urls": "|".join(row.facebook_urls),
                    "whatsapp_urls": "|".join(row.whatsapp_urls),
                    "candidate_score": row.candidate_score,
                    "status": row.status,
                    "last_seen": row.last_seen,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iraq multi-surface discovery (web/social/telegram seeds).")
    parser.add_argument("--out-dir", type=Path, default=SURVEY_DIR)
    parser.add_argument("--pages-per-query", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: Dict[str, Candidate] = {}
    search_debug: Dict[str, Any] = {
        "queries_run": len(DEFAULT_QUERIES),
        "successful_search_requests": 0,
        "failed_search_requests": 0,
        "query_hits": {},
    }

    for query in DEFAULT_QUERIES:
        hits = 0
        for engine, url in search_urls_for_query(query, pages=args.pages_per_query):
            page, status, err = fetch_url(url, timeout=args.timeout)
            if page is None or (status is not None and status >= 400):
                search_debug["failed_search_requests"] += 1
                continue
            search_debug["successful_search_requests"] += 1
            for token in extract_links(page):
                normalized = normalize_url(token)
                if not normalized:
                    continue
                candidate = candidates.get(normalized)
                if candidate is None:
                    candidate = Candidate(url=normalized, source_queries=set(), source_engines=set())
                    candidates[normalized] = candidate
                candidate.source_queries.add(query)
                candidate.source_engines.add(engine)
                hits += 1
        search_debug["query_hits"][query] = hits

    ordered_urls = sorted(candidates.keys())[: max(1, args.max_candidates)]
    rows: List[CandidateRow] = []
    seed_handles: Set[str] = set()

    for url in ordered_urls:
        page, status, err = fetch_url(url, timeout=args.timeout)
        parsed = parse_candidate(url, page, status, err)
        rows.append(parsed)
        for handle in parsed.telegram_handles:
            seed_handles.add(handle)

    candidates_csv = out_dir / "iraq_multisurface_candidates.csv"
    summary_json = out_dir / "iraq_multisurface_summary.json"
    seeds_json = out_dir / "iraq_multisurface_seed_handles.json"

    write_csv(candidates_csv, rows)
    seeds_payload = sorted(seed_handles)
    seeds_json.write_text(json.dumps(seeds_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    city_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for row in rows:
        city_counts[row.city_guess] = city_counts.get(row.city_guess, 0) + 1
        type_counts[row.source_type_guess] = type_counts.get(row.source_type_guess, 0) + 1

    summary = {
        "generated_at": now_iso(),
        "queries_run": len(DEFAULT_QUERIES),
        "search_debug": search_debug,
        "candidate_urls_discovered": len(candidates),
        "candidates_crawled": len(rows),
        "seed_telegram_handles_discovered": len(seed_handles),
        "sources_with_rate_signals": sum(1 for row in rows if row.has_rate_signal),
        "exchange_like_sources": sum(1 for row in rows if row.source_type_guess == "exchange_shop"),
        "remittance_like_sources": sum(1 for row in rows if row.source_type_guess == "remittance_exchange"),
        "regional_market_like_sources": sum(1 for row in rows if row.source_type_guess == "regional_market_channel"),
        "city_distribution": dict(sorted(city_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "source_type_distribution": dict(sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_scored_sources": [
            {
                "source_url": row.source_url,
                "city_guess": row.city_guess,
                "source_type_guess": row.source_type_guess,
                "candidate_score": row.candidate_score,
                "status": row.status,
                "telegram_handles": row.telegram_handles,
            }
            for row in sorted(rows, key=lambda r: (-r.candidate_score, r.source_url))[:20]
        ],
        "artifacts": {
            "candidates_csv": str(candidates_csv),
            "summary_json": str(summary_json),
            "seed_handles_json": str(seeds_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

