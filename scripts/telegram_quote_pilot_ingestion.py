#!/usr/bin/env python3
"""Pilot Telegram quote ingestion and normalization for RialWatch (research mode).

This script is intentionally separate from production pipeline code.
It ingests only public Telegram pages and produces pilot survey artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
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

NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")
BUY_PATTERN_TEMPLATE = r"{word}[^0-9]{{0,28}}({num})|({num})[^0-9]{{0,28}}{word}"
SLASH_PAIR_RE = re.compile(
    r"(?<!\d)(\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)\s*[/\\|\-]\s*(\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)"
)

BUY_WORDS = ("خرید", "buy", "bid")
SELL_WORDS = ("فروش", "sell", "offer", "ask")
QUOTE_WORDS = ("نرخ", "قیمت", "rate", "quote", "fx", "ارز")
SHOP_HINTS = ("صرافی", "exchange", "تماس", "address", "آدرس", "whatsapp", "تلگرام")
REPOST_HINTS = ("forwarded from", "فوروارد", "بازنشر", "via @", "منبع", "source:")
AGG_HINTS = ("شبکه", "همکار", "dealer", "market", "بازار", "کانال", "quote channel")

CURRENCY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "USD": ("usd", "us dollar", "دلار", "دلار آزاد", "دلارتهران", "دلار تهران"),
    "EUR": ("eur", "euro", "یورو"),
    "AED": ("aed", "dirham", "درهم"),
    "GBP": ("gbp", "pound", "پوند"),
    "TRY": ("try", "lira", "لیر"),
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
class PilotChannel:
    handle: str
    title: str
    source_priority: str
    origin_priority: str
    priority_score: float
    channel_type_guess: str
    likely_individual_shop: bool
    public_url: str
    selection_note: str


@dataclass
class MessageRow:
    handle: str
    title: str
    source_priority: str
    channel_type_guess: str
    likely_individual_shop: bool
    msg_index: int
    timestamp_iso: str
    timestamp_text: str
    message_text: str


@dataclass
class RecordRow:
    handle: str
    title: str
    source_priority: str
    channel_type_guess: str
    likely_individual_shop: bool
    message_index: int
    message_text_sample: str
    currency: str
    buy_quote: Optional[int]
    sell_quote: Optional[int]
    midpoint: Optional[float]
    buy_quote_rial: Optional[int]
    sell_quote_rial: Optional[int]
    midpoint_rial: Optional[float]
    raw_numeric_values: str
    value_unit_guess: str
    city_guess: str
    quote_type_guess: str
    timestamp_iso: str
    timestamp_text: str
    freshness_score: int
    structure_score: int
    duplication_flag: str
    directness_score: int
    overall_record_quality_score: int
    dedup_keep: bool
    exact_key: str
    near_key: str
    num_key: str
    cross_key: str


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return now_utc().replace(microsecond=0).isoformat()


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def normalize_handle(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", (value or "").strip().lower())


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def clean_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clip_text(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def normalize_public_url(handle: str, raw_url: str) -> str:
    handle = normalize_handle(handle)
    candidate = (raw_url or "").strip()
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

    if parts[0].lower() == "s" and len(parts) > 1:
        h = normalize_handle(parts[1])
    else:
        h = normalize_handle(parts[0])
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
    m = pattern.search(page)
    return html.unescape(m.group(1)).strip() if m else ""


def extract_message_rows(page: str, channel: PilotChannel) -> Tuple[List[MessageRow], int]:
    starts = [m.start() for m in re.finditer(r'<div class="tgme_widget_message_wrap', page)]
    if not starts:
        return [], 0

    rows: List[MessageRow] = []
    total_messages_seen = len(starts)

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(page)
        chunk = page[start:end]

        text_parts: List[str] = []
        for patt in (
            r'<div class="tgme_widget_message_text[^\"]*"[^>]*>(.*?)</div>',
            r'<div class="tgme_widget_message_caption[^\"]*"[^>]*>(.*?)</div>',
        ):
            for m in re.finditer(patt, chunk, flags=re.IGNORECASE | re.DOTALL):
                txt = clean_text(m.group(1))
                if txt:
                    text_parts.append(txt)

        combined = "\n".join(text_parts).strip()
        if not combined:
            continue

        dt_match = re.search(r'datetime="([^"]+)"', chunk)
        ts_iso = dt_match.group(1).strip() if dt_match else ""
        text_match = re.search(r'<time[^>]*>(.*?)</time>', chunk, flags=re.IGNORECASE | re.DOTALL)
        ts_text = clean_text(text_match.group(1)) if text_match else ""

        rows.append(
            MessageRow(
                handle=channel.handle,
                title=channel.title,
                source_priority=channel.source_priority,
                channel_type_guess=channel.channel_type_guess,
                likely_individual_shop=channel.likely_individual_shop,
                msg_index=idx,
                timestamp_iso=ts_iso,
                timestamp_text=ts_text,
                message_text=combined,
            )
        )

    return rows, total_messages_seen


def parse_int_token(token: str) -> Optional[int]:
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


def extract_numbers(text: str) -> List[int]:
    out: List[int] = []
    normalized = translit_digits(text)
    for m in NUMBER_RE.finditer(normalized):
        v = parse_int_token(m.group(0))
        if v is not None:
            out.append(v)
    return out


def find_keyword_number(text: str, words: Sequence[str]) -> Optional[int]:
    normalized = translit_digits(text).lower()
    num_pat = NUMBER_RE.pattern
    for word in words:
        patt = re.compile(BUY_PATTERN_TEMPLATE.format(word=re.escape(word), num=num_pat), re.IGNORECASE)
        m = patt.search(normalized)
        if not m:
            continue
        g1 = m.group(1)
        g2 = m.group(2)
        token = g1 if g1 else g2
        if token:
            v = parse_int_token(token)
            if v is not None:
                return v
    return None


def detect_slash_pair(text: str) -> Tuple[Optional[int], Optional[int]]:
    normalized = translit_digits(text)
    m = SLASH_PAIR_RE.search(normalized)
    if not m:
        return None, None
    return parse_int_token(m.group(1)), parse_int_token(m.group(2))


def detect_currencies(text: str) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for code, aliases in CURRENCY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                hits.append(code)
                break
    return sorted(set(hits))


def detect_cities(text: str) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for city, aliases in CITY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                hits.append(city)
                break
    return sorted(set(hits))


def detect_unit_guess(text: str, numbers: Sequence[int]) -> str:
    lowered = text.lower()
    if "تومان" in text or "tmn" in lowered or "toman" in lowered:
        return "toman"
    if "ریال" in text or "irr" in lowered or "rial" in lowered:
        return "rial"
    if not numbers:
        return "unknown"

    ordered = sorted(numbers)
    median = ordered[len(ordered) // 2]
    if median >= 900000:
        return "rial"
    if 50000 <= median <= 350000:
        return "toman"
    return "unknown"


def to_rial(value: Optional[int], unit_guess: str) -> Optional[int]:
    if value is None:
        return None
    if unit_guess == "toman":
        return value * 10
    return value


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


def freshness_score(ts_iso: str, now_dt: dt.datetime) -> int:
    ts = parse_timestamp(ts_iso)
    if ts is None:
        return 35
    delta_h = max(0.0, (now_dt - ts).total_seconds() / 3600.0)
    if delta_h <= 6:
        return 100
    if delta_h <= 24:
        return 88
    if delta_h <= 48:
        return 76
    if delta_h <= 72:
        return 66
    if delta_h <= 168:
        return 45
    return 25


def infer_quote_type(
    text: str,
    channel_type_guess: str,
    likely_individual_shop: bool,
) -> str:
    lowered = text.lower()
    if any(tok in lowered for tok in REPOST_HINTS):
        return "reposted"
    if channel_type_guess in ("dealer_network_channel", "market_price_channel", "aggregator"):
        return "aggregated"
    if likely_individual_shop:
        return "direct"
    if any(tok in lowered for tok in SHOP_HINTS):
        return "direct"
    if any(tok in lowered for tok in AGG_HINTS):
        return "aggregated"
    return "unclear"


def structure_score(
    text: str,
    currencies: Sequence[str],
    numbers: Sequence[int],
    buy: Optional[int],
    sell: Optional[int],
    cities: Sequence[str],
    unit_guess: str,
) -> int:
    lowered = text.lower()
    score = 0
    if currencies:
        score += 20
    if numbers:
        score += 10
    if buy is not None and sell is not None:
        score += 30
    elif buy is not None or sell is not None:
        score += 14
    if len(numbers) >= 2:
        score += 12
    if len(numbers) >= 4:
        score += 6
    if cities:
        score += 8
    if unit_guess != "unknown":
        score += 8
    if any(tok in lowered for tok in QUOTE_WORDS):
        score += 10
    return max(0, min(100, score))


def directness_score(quote_type: str, likely_shop: bool) -> int:
    mapping = {
        "direct": 82,
        "aggregated": 58,
        "reposted": 34,
        "unclear": 48,
    }
    score = mapping.get(quote_type, 48)
    if likely_shop and quote_type == "direct":
        score += 8
    return max(0, min(100, score))


def text_exact_key(text: str) -> str:
    lowered = translit_digits(text).lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def text_near_key(text: str) -> str:
    lowered = translit_digits(text).lower()
    lowered = re.sub(NUMBER_RE.pattern, "#", lowered)
    lowered = re.sub(r"[@#]?[a-z0-9_]{5,}", "@user", lowered)
    lowered = re.sub(r"[^\w\s\u0600-\u06FF#]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered[:180]


def make_num_key(numbers: Sequence[int], buy: Optional[int], sell: Optional[int]) -> str:
    vals = list(numbers)
    if buy is not None:
        vals.append(buy)
    if sell is not None:
        vals.append(sell)
    dedup = []
    seen = set()
    for v in vals:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    dedup = dedup[:6]
    return "|".join(str(v) for v in dedup)


def parse_quote_records_from_message(msg: MessageRow, now_dt: dt.datetime) -> List[RecordRow]:
    text = translit_digits(msg.message_text)
    lowered = text.lower()
    numbers = extract_numbers(text)
    buy = find_keyword_number(text, BUY_WORDS)
    sell = find_keyword_number(text, SELL_WORDS)
    slash_a, slash_b = detect_slash_pair(text)

    if buy is None and sell is None and slash_a is not None and slash_b is not None:
        buy, sell = slash_a, slash_b

    if buy is None and sell is not None and slash_a is not None:
        buy = slash_a
    if sell is None and buy is not None and slash_b is not None:
        sell = slash_b

    if buy is None and sell is None and len(numbers) >= 2 and any(w in lowered for w in BUY_WORDS + SELL_WORDS):
        buy, sell = numbers[0], numbers[1]

    currencies = detect_currencies(text)
    cities = detect_cities(text)
    unit_guess = detect_unit_guess(text, numbers)

    has_quote_word = any(tok in lowered for tok in QUOTE_WORDS)
    is_quote_like = (
        (numbers and (currencies or has_quote_word or cities or buy is not None or sell is not None))
        or (buy is not None and sell is not None)
    )
    if not is_quote_like:
        return []

    quote_type = infer_quote_type(text, msg.channel_type_guess, msg.likely_individual_shop)
    fresh = freshness_score(msg.timestamp_iso, now_dt)
    struct = structure_score(text, currencies, numbers, buy, sell, cities, unit_guess)
    direct = directness_score(quote_type, msg.likely_individual_shop)

    midpoint = None
    if buy is not None and sell is not None and buy > 0 and sell > 0:
        midpoint = (buy + sell) / 2.0

    buy_rial = to_rial(buy, unit_guess)
    sell_rial = to_rial(sell, unit_guess)
    midpoint_rial = None
    if buy_rial is not None and sell_rial is not None:
        midpoint_rial = (buy_rial + sell_rial) / 2.0

    overall = int(round(0.40 * struct + 0.35 * fresh + 0.25 * direct))

    exact_key = text_exact_key(text)
    near_key = text_near_key(text)
    num_key = make_num_key(numbers, buy, sell)

    currency_label = "|".join(currencies) if currencies else "UNKNOWN"
    cross_key = f"{near_key}|{num_key}|{currency_label}"

    return [
        RecordRow(
            handle=msg.handle,
            title=msg.title,
            source_priority=msg.source_priority,
            channel_type_guess=msg.channel_type_guess,
            likely_individual_shop=msg.likely_individual_shop,
            message_index=msg.msg_index,
            message_text_sample=clip_text(text),
            currency=currency_label,
            buy_quote=buy,
            sell_quote=sell,
            midpoint=midpoint,
            buy_quote_rial=buy_rial,
            sell_quote_rial=sell_rial,
            midpoint_rial=midpoint_rial,
            raw_numeric_values="|".join(str(n) for n in numbers[:12]),
            value_unit_guess=unit_guess,
            city_guess=cities[0] if cities else "unknown",
            quote_type_guess=quote_type,
            timestamp_iso=msg.timestamp_iso,
            timestamp_text=msg.timestamp_text,
            freshness_score=fresh,
            structure_score=struct,
            duplication_flag="none",
            directness_score=direct,
            overall_record_quality_score=overall,
            dedup_keep=True,
            exact_key=exact_key,
            near_key=near_key,
            num_key=num_key,
            cross_key=cross_key,
        )
    ]


def apply_in_channel_dedup(records: List[RecordRow]) -> Dict[str, int]:
    exact_removed = 0
    near_removed = 0

    by_channel: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        by_channel.setdefault(rec.handle, []).append(idx)

    for handle, idxs in by_channel.items():
        seen_exact: Dict[str, int] = {}
        seen_near: Dict[str, int] = {}
        for idx in idxs:
            rec = records[idx]
            if rec.exact_key in seen_exact:
                rec.duplication_flag = "exact_duplicate_in_channel"
                rec.dedup_keep = False
                rec.overall_record_quality_score = max(0, rec.overall_record_quality_score - 22)
                rec.quote_type_guess = "reposted"
                exact_removed += 1
                continue

            near_sig = f"{rec.near_key}|{rec.num_key}|{rec.currency}"
            if near_sig in seen_near:
                rec.duplication_flag = "near_duplicate_in_channel"
                rec.dedup_keep = False
                rec.overall_record_quality_score = max(0, rec.overall_record_quality_score - 16)
                rec.quote_type_guess = "reposted"
                near_removed += 1
                continue

            seen_exact[rec.exact_key] = idx
            seen_near[near_sig] = idx

    return {
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
    }


def apply_cross_channel_repost_flags(records: List[RecordRow]) -> Dict[str, int]:
    groups: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        if not rec.dedup_keep:
            continue
        groups.setdefault(rec.cross_key, []).append(idx)

    repost_flagged = 0
    cross_groups = 0

    for _, idxs in groups.items():
        handles = sorted({records[i].handle for i in idxs})
        if len(handles) <= 1:
            continue
        cross_groups += 1

        def rank_key(i: int) -> Tuple[int, int, str, int]:
            rec = records[i]
            source_rank = 0 if rec.source_priority == "P1" else 1
            return (-rec.overall_record_quality_score, source_rank, rec.handle, rec.message_index)

        ranked = sorted(idxs, key=rank_key)
        keep_idx = ranked[0]

        for i in ranked[1:]:
            rec = records[i]
            if rec.duplication_flag == "none":
                rec.duplication_flag = "reposted_cross_channel"
            elif "reposted_cross_channel" not in rec.duplication_flag:
                rec.duplication_flag += "|reposted_cross_channel"
            rec.dedup_keep = False
            rec.quote_type_guess = "reposted"
            rec.overall_record_quality_score = max(0, rec.overall_record_quality_score - 18)
            repost_flagged += 1

        keep_rec = records[keep_idx]
        if keep_rec.quote_type_guess == "unclear":
            keep_rec.quote_type_guess = "aggregated"

    return {
        "cross_channel_groups": cross_groups,
        "cross_channel_reposts_flagged": repost_flagged,
    }


def load_quote_signal_map(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            handle = normalize_handle(row.get("handle", ""))
            if not handle:
                continue
            out[handle] = row
    return out


def select_pilot_channels(
    priority_csv: Path,
    quote_signal_map: Dict[str, Dict[str, str]],
) -> List[PilotChannel]:
    rows: List[Dict[str, str]] = []
    with priority_csv.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)

    p1_rows = [r for r in rows if (r.get("recommended_priority") or "").strip() == "P1"]
    p2_rows = [r for r in rows if (r.get("recommended_priority") or "").strip() == "P2"]

    selected_handles: set[str] = set()
    selected: List[PilotChannel] = []

    def append_from_row(row: Dict[str, str], source_priority: str, selection_note: str) -> None:
        handle = normalize_handle(row.get("handle", ""))
        if not handle or handle in selected_handles:
            return

        q = quote_signal_map.get(handle, {})
        ch_type = (row.get("channel_type_guess") or q.get("channel_type_guess") or "unclear").strip() or "unclear"
        likely_shop = to_bool(row.get("likely_individual_shop") or q.get("likely_individual_shop") or "false")
        title = (row.get("title") or q.get("title") or "").strip()

        selected.append(
            PilotChannel(
                handle=handle,
                title=title,
                source_priority=source_priority,
                origin_priority=(row.get("recommended_priority") or "").strip() or source_priority,
                priority_score=to_float(row.get("priority_score", "0"), 0.0),
                channel_type_guess=ch_type,
                likely_individual_shop=likely_shop,
                public_url=f"https://t.me/s/{handle}",
                selection_note=selection_note,
            )
        )
        selected_handles.add(handle)

    # Top 10 P1 candidates.
    for row in p1_rows[:10]:
        append_from_row(row, source_priority="P1", selection_note="top_p1")

    # If fewer than 10 P1 exist, promote best remaining P2 deterministically.
    if len([x for x in selected if x.source_priority == "P1"]) < 10:
        needed = 10 - len([x for x in selected if x.source_priority == "P1"])
        for row in p2_rows:
            if needed <= 0:
                break
            handle = normalize_handle(row.get("handle", ""))
            if handle in selected_handles:
                continue
            append_from_row(row, source_priority="P1", selection_note="promoted_from_p2_due_to_p1_shortage")
            needed -= 1

    # Top 10 P2 candidates (excluding already selected).
    p2_added = 0
    for row in p2_rows:
        if p2_added >= 10:
            break
        handle = normalize_handle(row.get("handle", ""))
        if handle in selected_handles:
            continue
        append_from_row(row, source_priority="P2", selection_note="top_p2")
        p2_added += 1

    # Deterministic ordering: P1 then P2, descending priority_score then handle.
    selected.sort(key=lambda x: (0 if x.source_priority == "P1" else 1, -x.priority_score, x.handle))
    return selected


def write_pilot_channels(path: Path, channels: Sequence[PilotChannel]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "title",
                "source_priority",
                "origin_priority",
                "priority_score",
                "channel_type_guess",
                "likely_individual_shop",
                "public_url",
                "selection_note",
            ],
        )
        writer.writeheader()
        for ch in channels:
            writer.writerow(
                {
                    "handle": ch.handle,
                    "title": ch.title,
                    "source_priority": ch.source_priority,
                    "origin_priority": ch.origin_priority,
                    "priority_score": f"{ch.priority_score:.2f}",
                    "channel_type_guess": ch.channel_type_guess,
                    "likely_individual_shop": ch.likely_individual_shop,
                    "public_url": ch.public_url,
                    "selection_note": ch.selection_note,
                }
            )


def infer_recommended_status(
    quote_records_extracted: int,
    deduplicated_records: int,
    likely_direct_records: int,
    likely_reposted_records: int,
    average_record_quality_score: float,
    ingestion_readiness_score: float,
    buy_sell_pair_records: int,
) -> str:
    if quote_records_extracted <= 1:
        return "low_value"

    duplicate_ratio = 1.0 - (deduplicated_records / max(1, quote_records_extracted))
    if duplicate_ratio >= 0.60 and quote_records_extracted >= 5:
        return "duplicate_heavy"

    if average_record_quality_score < 40 and quote_records_extracted >= 3:
        return "unstable_format"

    if ingestion_readiness_score >= 72 and deduplicated_records >= 6 and average_record_quality_score >= 58:
        return "ready_for_research_ingestion"

    if ingestion_readiness_score >= 45 and deduplicated_records >= 3:
        return "monitor_only"

    if buy_sell_pair_records == 0 and likely_direct_records == 0:
        return "unstable_format"

    return "low_value"


def compute_channel_metrics(
    channels: Sequence[PilotChannel],
    records: Sequence[RecordRow],
    total_messages_seen_map: Dict[str, int],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_handle: Dict[str, List[RecordRow]] = {}
    for rec in records:
        by_handle.setdefault(rec.handle, []).append(rec)

    rows: List[Dict[str, object]] = []

    status_counts = {
        "ready_for_research_ingestion": 0,
        "monitor_only": 0,
        "low_value": 0,
        "duplicate_heavy": 0,
        "unstable_format": 0,
    }

    for ch in channels:
        rlist = by_handle.get(ch.handle, [])
        dedup = [r for r in rlist if r.dedup_keep]

        quote_records_extracted = len(rlist)
        deduplicated_records = len(dedup)
        usd_records = sum(1 for r in rlist if "USD" in r.currency.split("|"))
        eur_records = sum(1 for r in rlist if "EUR" in r.currency.split("|"))
        buy_sell_pair_records = sum(1 for r in rlist if r.buy_quote is not None and r.sell_quote is not None)
        likely_direct_records = sum(1 for r in rlist if r.quote_type_guess == "direct")
        likely_reposted_records = sum(
            1
            for r in rlist
            if r.quote_type_guess == "reposted" or "repost" in r.duplication_flag or "duplicate" in r.duplication_flag
        )

        quality_base = dedup if dedup else rlist
        if quality_base:
            avg_quality = sum(r.overall_record_quality_score for r in quality_base) / len(quality_base)
        else:
            avg_quality = 0.0

        pair_ratio = (buy_sell_pair_records / max(1, deduplicated_records)) if deduplicated_records else 0.0
        direct_ratio = (likely_direct_records / max(1, deduplicated_records)) if deduplicated_records else 0.0
        volume_score = min(100.0, deduplicated_records * 10.0)

        readiness = (
            0.45 * avg_quality
            + 0.25 * volume_score
            + 0.15 * (pair_ratio * 100.0)
            + 0.15 * (direct_ratio * 100.0)
        )

        duplicate_ratio = 1.0 - (deduplicated_records / max(1, quote_records_extracted)) if quote_records_extracted else 0.0
        if duplicate_ratio > 0.50:
            readiness -= 10.0
        if deduplicated_records < 3:
            readiness -= 10.0

        readiness = max(0.0, min(100.0, readiness))

        status = infer_recommended_status(
            quote_records_extracted=quote_records_extracted,
            deduplicated_records=deduplicated_records,
            likely_direct_records=likely_direct_records,
            likely_reposted_records=likely_reposted_records,
            average_record_quality_score=avg_quality,
            ingestion_readiness_score=readiness,
            buy_sell_pair_records=buy_sell_pair_records,
        )
        status_counts[status] = status_counts.get(status, 0) + 1

        rows.append(
            {
                "handle": ch.handle,
                "title": ch.title,
                "source_priority": ch.source_priority,
                "channel_type_guess": ch.channel_type_guess,
                "likely_individual_shop": ch.likely_individual_shop,
                "total_messages_seen": total_messages_seen_map.get(ch.handle, 0),
                "quote_records_extracted": quote_records_extracted,
                "deduplicated_records": deduplicated_records,
                "usd_records": usd_records,
                "eur_records": eur_records,
                "buy_sell_pair_records": buy_sell_pair_records,
                "likely_direct_records": likely_direct_records,
                "likely_reposted_records": likely_reposted_records,
                "average_record_quality_score": round(avg_quality, 2),
                "ingestion_readiness_score": round(readiness, 2),
                "recommended_status": status,
            }
        )

    rows.sort(key=lambda x: (0 if x["source_priority"] == "P1" else 1, -float(x["ingestion_readiness_score"]), x["handle"]))
    return rows, status_counts


def write_pilot_quote_records(path: Path, records: Sequence[RecordRow]) -> None:
    ordered = sorted(
        records,
        key=lambda r: (0 if r.source_priority == "P1" else 1, r.handle, r.message_index, r.currency),
    )

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "title",
                "source_priority",
                "message_text_sample",
                "currency",
                "buy_quote",
                "sell_quote",
                "midpoint",
                "raw_numeric_values",
                "city_guess",
                "quote_type_guess",
                "timestamp_text",
                "freshness_score",
                "structure_score",
                "duplication_flag",
                "directness_score",
                "overall_record_quality_score",
                "value_unit_guess",
                "buy_quote_rial",
                "sell_quote_rial",
                "midpoint_rial",
                "dedup_keep",
                "channel_type_guess",
                "likely_individual_shop",
                "timestamp_iso",
            ],
        )
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    "handle": r.handle,
                    "title": r.title,
                    "source_priority": r.source_priority,
                    "message_text_sample": r.message_text_sample,
                    "currency": r.currency,
                    "buy_quote": r.buy_quote if r.buy_quote is not None else "",
                    "sell_quote": r.sell_quote if r.sell_quote is not None else "",
                    "midpoint": f"{r.midpoint:.2f}" if r.midpoint is not None else "",
                    "raw_numeric_values": r.raw_numeric_values,
                    "city_guess": r.city_guess,
                    "quote_type_guess": r.quote_type_guess,
                    "timestamp_text": r.timestamp_text,
                    "freshness_score": r.freshness_score,
                    "structure_score": r.structure_score,
                    "duplication_flag": r.duplication_flag,
                    "directness_score": r.directness_score,
                    "overall_record_quality_score": r.overall_record_quality_score,
                    "value_unit_guess": r.value_unit_guess,
                    "buy_quote_rial": r.buy_quote_rial if r.buy_quote_rial is not None else "",
                    "sell_quote_rial": r.sell_quote_rial if r.sell_quote_rial is not None else "",
                    "midpoint_rial": f"{r.midpoint_rial:.2f}" if r.midpoint_rial is not None else "",
                    "dedup_keep": r.dedup_keep,
                    "channel_type_guess": r.channel_type_guess,
                    "likely_individual_shop": r.likely_individual_shop,
                    "timestamp_iso": r.timestamp_iso,
                }
            )


def write_channel_metrics(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "title",
                "source_priority",
                "channel_type_guess",
                "likely_individual_shop",
                "total_messages_seen",
                "quote_records_extracted",
                "deduplicated_records",
                "usd_records",
                "eur_records",
                "buy_sell_pair_records",
                "likely_direct_records",
                "likely_reposted_records",
                "average_record_quality_score",
                "ingestion_readiness_score",
                "recommended_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def estimate_unique_daily_signal_potential(
    channels: Sequence[PilotChannel],
    records: Sequence[RecordRow],
) -> int:
    by_handle: Dict[str, List[RecordRow]] = {}
    for rec in records:
        if rec.dedup_keep:
            by_handle.setdefault(rec.handle, []).append(rec)

    total = 0.0
    for ch in channels:
        rlist = by_handle.get(ch.handle, [])
        if not rlist:
            continue

        ts_vals = [parse_timestamp(r.timestamp_iso) for r in rlist]
        ts_vals = [x for x in ts_vals if x is not None]
        count = len(rlist)

        if len(ts_vals) >= 2:
            newest = max(ts_vals)
            oldest = min(ts_vals)
            span_h = max(6.0, (newest - oldest).total_seconds() / 3600.0)
            daily = min(60.0, (count / span_h) * 24.0)
        else:
            daily = min(40.0, count * 2.0)

        total += daily

    return int(round(total))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot Telegram quote ingestion (research mode)")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Path to survey outputs")
    parser.add_argument("--timeout", type=int, default=18, help="HTTP timeout in seconds")
    parser.add_argument("--sleep-seconds", type=float, default=0.6, help="Rate-limit sleep between requests")
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    priority_csv = survey_dir / "priority_ingestion_candidates.csv"
    quote_signal_csv = survey_dir / "quote_signal_survey.csv"

    required = [priority_csv, quote_signal_csv]
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    pilot_channels_csv = survey_dir / "pilot_ingestion_channels.csv"
    pilot_quote_records_csv = survey_dir / "pilot_quote_records.csv"
    pilot_channel_metrics_csv = survey_dir / "pilot_channel_metrics.csv"
    pilot_dedup_summary_json = survey_dir / "pilot_dedup_summary.json"
    pilot_ingestion_summary_json = survey_dir / "pilot_ingestion_summary.json"

    print("[1/5] Selecting pilot channels from priority files...")
    quote_signal_map = load_quote_signal_map(quote_signal_csv)
    channels = select_pilot_channels(priority_csv, quote_signal_map)
    write_pilot_channels(pilot_channels_csv, channels)
    print(f"Selected pilot channels: {len(channels)}")

    print("[2/5] Crawling pilot channels and extracting quote records...")
    all_records: List[RecordRow] = []
    total_messages_seen_map: Dict[str, int] = {}
    channel_errors: Dict[str, str] = {}

    now_dt = now_utc()

    for idx, ch in enumerate(channels, start=1):
        page, status, err = fetch_url(ch.public_url, timeout=args.timeout)
        if page is None or (status is not None and status >= 400):
            channel_errors[ch.handle] = err or f"http_{status}"
            total_messages_seen_map[ch.handle] = 0
            if idx % 5 == 0 or idx == len(channels):
                print(f"  Crawled {idx}/{len(channels)} pilot channels")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            continue

        # Refresh title when available from page metadata.
        meta_title = extract_meta_content(page, "og:title") or extract_meta_content(page, "twitter:title")
        if meta_title:
            ch.title = meta_title

        messages, total_seen = extract_message_rows(page, ch)
        total_messages_seen_map[ch.handle] = total_seen

        for msg in messages:
            recs = parse_quote_records_from_message(msg, now_dt)
            if recs:
                all_records.extend(recs)

        if idx % 5 == 0 or idx == len(channels):
            print(f"  Crawled {idx}/{len(channels)} pilot channels")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print("[3/5] Applying deduplication and quality adjustments...")
    dedup_counts = apply_in_channel_dedup(all_records)
    cross_counts = apply_cross_channel_repost_flags(all_records)

    print("[4/5] Computing channel metrics and writing outputs...")
    write_pilot_quote_records(pilot_quote_records_csv, all_records)

    channel_metric_rows, status_counts = compute_channel_metrics(
        channels=channels,
        records=all_records,
        total_messages_seen_map=total_messages_seen_map,
    )
    write_channel_metrics(pilot_channel_metrics_csv, channel_metric_rows)

    total_before = len(all_records)
    total_after = sum(1 for r in all_records if r.dedup_keep)

    dedup_summary = {
        "generated_at": now_iso(),
        "pilot_channels_selected": len(channels),
        "total_records_before_dedup": total_before,
        "total_records_after_dedup": total_after,
        "exact_duplicates_removed": dedup_counts["exact_duplicates_removed"],
        "near_duplicates_removed": dedup_counts["near_duplicates_removed"],
        "cross_channel_groups": cross_counts["cross_channel_groups"],
        "cross_channel_reposts_flagged": cross_counts["cross_channel_reposts_flagged"],
        "channel_errors": channel_errors,
    }
    pilot_dedup_summary_json.write_text(json.dumps(dedup_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    total_usd = sum(1 for r in all_records if "USD" in r.currency.split("|"))
    total_buy_sell = sum(1 for r in all_records if r.buy_quote is not None and r.sell_quote is not None)

    summary = {
        "generated_at": now_iso(),
        "pilot_channels_analyzed": len(channels),
        "total_quote_records_extracted": total_before,
        "total_deduplicated_records": total_after,
        "total_usd_records": total_usd,
        "total_buy_sell_pair_records": total_buy_sell,
        "channels_ready_for_research_ingestion": status_counts.get("ready_for_research_ingestion", 0),
        "channels_to_monitor_only": status_counts.get("monitor_only", 0),
        "channels_low_value": status_counts.get("low_value", 0),
        "channels_duplicate_heavy": status_counts.get("duplicate_heavy", 0),
        "channels_unstable_format": status_counts.get("unstable_format", 0),
        "estimated_unique_daily_signal_potential": estimate_unique_daily_signal_potential(channels, all_records),
        "p1_channels_selected": sum(1 for c in channels if c.source_priority == "P1"),
        "p2_channels_selected": sum(1 for c in channels if c.source_priority == "P2"),
        "p1_promoted_from_p2": sum(1 for c in channels if c.selection_note.startswith("promoted_from_p2")),
    }
    pilot_ingestion_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] Completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
