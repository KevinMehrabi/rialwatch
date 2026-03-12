#!/usr/bin/env python3
"""Extract Telegram regional FX quote signals (research mode only)."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

NUMBER_RE = re.compile(r"(?<!\d)(?:\d{2,3}(?:[\s,٬،]\d{3})+|\d{5,8})(?!\d)")

REGION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Tehran": ("tehran", "تهران"),
    "Herat": ("herat", "هرات"),
    "Sulaymaniyah": ("sulaymaniyah", "sulaimaniyah", "sulaymania", "suleymaniye", "سلیمانیه", "سليمانيه"),
    "Dubai": ("dubai", "دبی", "دوبی"),
    "Istanbul": ("istanbul", "استانبول"),
}

CURRENCY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "USD": ("usd", "dollar", "دلار"),
    "EUR": ("eur", "euro", "یورو"),
    "AED": ("aed", "dirham", "درهم"),
    "TRY": ("try", "lira", "لیر"),
}

BUY_WORDS = ("خرید", "buy", "bid")
SELL_WORDS = ("فروش", "sell", "offer", "ask")


@dataclass
class TargetChannel:
    handle: str
    public_url: str
    title: str
    channel_type_guess: str
    parseable_score: int
    quote_post_count: int


@dataclass
class RegionalRecord:
    region: str
    quote_value: Optional[float]
    quote_value_rial: Optional[float]
    currency: str
    buy_quote: Optional[float]
    sell_quote: Optional[float]
    midpoint: Optional[float]
    midpoint_rial: Optional[float]
    inferred_unit_guess: str
    channel: str
    channel_title: str
    message_sample: str
    timestamp_iso: str


def translit_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def clean_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clip_text(text: str, limit: int = 260) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_int_token(token: str) -> Optional[int]:
    cleaned = translit_digits(token)
    cleaned = re.sub(r"[^0-9]", "", cleaned)
    if not cleaned:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


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
        patt = re.compile(rf"{re.escape(word)}[^0-9]{{0,26}}({num_pat})|({num_pat})[^0-9]{{0,26}}{re.escape(word)}", re.IGNORECASE)
        m = patt.search(normalized)
        if not m:
            continue
        tok = m.group(1) or m.group(2)
        if tok:
            n = parse_int_token(tok)
            if n is not None:
                return n
    return None


def detect_currencies(text: str) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for code, aliases in CURRENCY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                hits.append(code)
                break
    return sorted(set(hits))


def detect_regions(text: str) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for region, aliases in REGION_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                hits.append(region)
                break
    return hits


def infer_unit_guess(text: str, nums: Sequence[int], candidate_value: Optional[float]) -> str:
    lowered = text.lower()
    if "تومان" in text or "toman" in lowered or "tmn" in lowered:
        return "toman"
    if "ریال" in text or "rial" in lowered or "irr" in lowered:
        return "rial"

    probe = candidate_value
    if probe is None and nums:
        probe = float(sorted(nums)[len(nums) // 2])
    if probe is None:
        return "unknown"

    if probe >= 700000:
        return "rial"
    if 50000 <= probe <= 350000:
        return "toman"
    return "unknown"


def to_rial(value: Optional[float], unit_guess: str) -> Optional[float]:
    if value is None:
        return None
    if unit_guess == "toman":
        return value * 10.0
    return value


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, q))
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def trimmed_mean(values: Sequence[float], trim_frac: float = 0.10) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    trim_n = int(n * trim_frac)
    if trim_n * 2 >= n:
        trim_n = max(0, (n - 1) // 2)
    sliced = vals[trim_n : n - trim_n] if n - trim_n > trim_n else vals
    if not sliced:
        sliced = vals
    return statistics.mean(sliced)


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


def extract_message_chunks(page: str) -> List[Tuple[str, str]]:
    starts = [m.start() for m in re.finditer(r'<div class="tgme_widget_message_wrap', page)]
    if not starts:
        return []

    out: List[Tuple[str, str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(page)
        chunk = page[start:end]

        parts: List[str] = []
        for patt in (
            r'<div class="tgme_widget_message_text[^\"]*"[^>]*>(.*?)</div>',
            r'<div class="tgme_widget_message_caption[^\"]*"[^>]*>(.*?)</div>',
        ):
            for m in re.finditer(patt, chunk, flags=re.IGNORECASE | re.DOTALL):
                txt = clean_text(m.group(1))
                if txt:
                    parts.append(txt)

        text = "\n".join(parts).strip()
        if not text:
            continue

        ts_match = re.search(r'datetime="([^"]+)"', chunk)
        ts_iso = ts_match.group(1).strip() if ts_match else ""
        out.append((text, ts_iso))

    return out


def select_region_value(text: str, region: str, nums: Sequence[int]) -> Optional[float]:
    if not nums:
        return None

    normalized = translit_digits(text).lower()
    aliases = REGION_ALIASES[region]

    candidates: List[int] = []
    for alias in aliases:
        for m in re.finditer(re.escape(alias.lower()), normalized):
            lo = max(0, m.start() - 40)
            hi = min(len(normalized), m.end() + 40)
            window = normalized[lo:hi]
            for nm in NUMBER_RE.finditer(window):
                n = parse_int_token(nm.group(0))
                if n is not None:
                    candidates.append(n)

    if candidates:
        return float(candidates[0])

    return float(nums[0])


def parse_regional_records(message_text: str, ts_iso: str, channel: TargetChannel) -> List[RegionalRecord]:
    regions = detect_regions(message_text)
    if not regions:
        return []

    nums = extract_numbers(message_text)
    if not nums:
        return []

    buy = find_keyword_number(message_text, BUY_WORDS)
    sell = find_keyword_number(message_text, SELL_WORDS)
    midpoint = None
    if buy is not None and sell is not None and buy > 0 and sell > 0:
        midpoint = (buy + sell) / 2.0

    currencies = detect_currencies(message_text)
    currency = "USD" if "USD" in currencies else (currencies[0] if currencies else "USD")

    records: List[RegionalRecord] = []
    for region in regions:
        quote_value = select_region_value(message_text, region, nums)
        unit_guess = infer_unit_guess(message_text, nums, quote_value if quote_value is not None else midpoint)

        buy_r = to_rial(float(buy) if buy is not None else None, unit_guess)
        sell_r = to_rial(float(sell) if sell is not None else None, unit_guess)
        midpoint_r = to_rial(midpoint, unit_guess)
        quote_r = to_rial(quote_value, unit_guess)

        if midpoint_r is None:
            midpoint_r = quote_r

        records.append(
            RegionalRecord(
                region=region,
                quote_value=quote_value,
                quote_value_rial=quote_r,
                currency=currency,
                buy_quote=float(buy) if buy is not None else None,
                sell_quote=float(sell) if sell is not None else None,
                midpoint=midpoint,
                midpoint_rial=midpoint_r,
                inferred_unit_guess=unit_guess,
                channel=channel.handle,
                channel_title=channel.title,
                message_sample=clip_text(message_text),
                timestamp_iso=ts_iso,
            )
        )

    return records


def load_target_channels(path: Path, max_channels: int) -> List[TargetChannel]:
    channels: List[TargetChannel] = []
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ctype = str(row.get("channel_type_guess", "")).strip().lower()
            # Support both original and requested naming variants.
            is_market = ctype in ("market_channel", "market_price_channel")
            is_network = ctype in ("network_channel", "dealer_network_channel")
            if not (is_market or is_network):
                continue

            handle = str(row.get("handle", "")).strip().lower()
            if not handle:
                continue

            public_url = str(row.get("public_url", "")).strip() or f"https://t.me/s/{handle}"
            channels.append(
                TargetChannel(
                    handle=handle,
                    public_url=public_url,
                    title=str(row.get("title", "")).strip(),
                    channel_type_guess=str(row.get("channel_type_guess", "")).strip(),
                    parseable_score=to_int(row.get("parseable_score", "0"), 0),
                    quote_post_count=to_int(row.get("quote_post_count", "0"), 0),
                )
            )

    channels.sort(key=lambda c: (-c.parseable_score, -c.quote_post_count, c.handle))
    if max_channels > 0:
        channels = channels[:max_channels]
    return channels


def dedupe_records(records: List[RegionalRecord]) -> List[RegionalRecord]:
    seen: Set[Tuple[str, str, str, str, Optional[float]]] = set()
    out: List[RegionalRecord] = []
    for r in records:
        key = (r.channel, r.region, r.timestamp_iso, r.message_sample, r.midpoint_rial)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_records(path: Path, records: Sequence[RegionalRecord]) -> None:
    ordered = sorted(records, key=lambda r: (r.region, r.channel, r.timestamp_iso, r.message_sample))
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "region",
                "quote_value",
                "quote_value_rial",
                "currency",
                "buy_quote",
                "sell_quote",
                "midpoint",
                "midpoint_rial",
                "inferred_unit_guess",
                "channel",
                "channel_title",
                "message_sample",
                "timestamp_iso",
            ],
        )
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    "region": r.region,
                    "quote_value": "" if r.quote_value is None else f"{r.quote_value:.2f}",
                    "quote_value_rial": "" if r.quote_value_rial is None else f"{r.quote_value_rial:.2f}",
                    "currency": r.currency,
                    "buy_quote": "" if r.buy_quote is None else f"{r.buy_quote:.2f}",
                    "sell_quote": "" if r.sell_quote is None else f"{r.sell_quote:.2f}",
                    "midpoint": "" if r.midpoint is None else f"{r.midpoint:.2f}",
                    "midpoint_rial": "" if r.midpoint_rial is None else f"{r.midpoint_rial:.2f}",
                    "inferred_unit_guess": r.inferred_unit_guess,
                    "channel": r.channel,
                    "channel_title": r.channel_title,
                    "message_sample": r.message_sample,
                    "timestamp_iso": r.timestamp_iso,
                }
            )


def aggregate_summary(records: Sequence[RegionalRecord], channels_attempted: int, channels_with_quotes: int) -> Dict[str, object]:
    by_region: Dict[str, List[RegionalRecord]] = {region: [] for region in REGION_ALIASES.keys()}
    for rec in records:
        by_region.setdefault(rec.region, []).append(rec)

    region_rows: List[Dict[str, object]] = []
    for region in REGION_ALIASES.keys():
        recs = by_region.get(region, [])
        values = [r.midpoint_rial for r in recs if r.midpoint_rial is not None]

        median_v = statistics.median(values) if values else None
        trimmed_v = trimmed_mean(values, trim_frac=0.10) if values else None
        std_v = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        cv_v = (std_v / median_v) if (std_v is not None and median_v not in (None, 0)) else None

        region_rows.append(
            {
                "region": region,
                "quote_count": len(values),
                "median_price": round(median_v, 2) if median_v is not None else None,
                "trimmed_mean": round(trimmed_v, 2) if trimmed_v is not None else None,
                "dispersion_std": round(std_v, 4) if std_v is not None else None,
                "dispersion_cv": round(cv_v, 6) if cv_v is not None else None,
                "channels_contributing": sorted({r.channel for r in recs}),
                "channel_count": len({r.channel for r in recs}),
            }
        )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "channels_attempted": channels_attempted,
        "channels_with_regional_quotes": channels_with_quotes,
        "total_regional_quote_records": len(records),
        "regions": region_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract regional Telegram FX quote signals")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Survey output directory")
    parser.add_argument("--max-channels", type=int, default=160, help="Max market/network channels to crawl")
    parser.add_argument("--timeout", type=int, default=18, help="HTTP timeout")
    parser.add_argument("--sleep", type=float, default=0.55, help="Rate-limit sleep between requests")
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    channel_survey_csv = survey_dir / "channel_survey.csv"
    if not channel_survey_csv.exists():
        raise SystemExit(f"missing input: {channel_survey_csv}")

    records_csv = survey_dir / "telegram_regional_quotes.csv"
    summary_json = survey_dir / "telegram_regional_summary.json"

    targets = load_target_channels(channel_survey_csv, max_channels=args.max_channels)
    print(f"Target channels selected: {len(targets)}")

    all_records: List[RegionalRecord] = []
    channels_with_quotes = 0

    for idx, ch in enumerate(targets, start=1):
        page, status, err = fetch_url(ch.public_url, timeout=args.timeout)
        if page is None or (status is not None and status >= 400):
            if idx % 20 == 0 or idx == len(targets):
                print(f"  Crawled {idx}/{len(targets)} channels")
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        message_chunks = extract_message_chunks(page)
        ch_records: List[RegionalRecord] = []
        for msg_text, ts_iso in message_chunks:
            ch_records.extend(parse_regional_records(msg_text, ts_iso, ch))

        if ch_records:
            channels_with_quotes += 1
            all_records.extend(ch_records)

        if idx % 20 == 0 or idx == len(targets):
            print(f"  Crawled {idx}/{len(targets)} channels")

        if args.sleep > 0:
            time.sleep(args.sleep)

    deduped = dedupe_records(all_records)
    write_records(records_csv, deduped)

    summary = aggregate_summary(deduped, channels_attempted=len(targets), channels_with_quotes=channels_with_quotes)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
