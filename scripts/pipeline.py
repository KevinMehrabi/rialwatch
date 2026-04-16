#!/usr/bin/env python3
"""USD/IRR Open Market Reference pipeline.

Samples configured sources within the observation window,
computes the daily reference, and renders a static site under /site.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

UTC = dt.timezone.utc
IRAN_TZ = dt.timezone(dt.timedelta(hours=3, minutes=30))
WINDOW_START = dt.time(13, 45)
WINDOW_END = dt.time(14, 15)
PUBLISH_AT = dt.time(14, 20)
DEFAULT_INTRADAY_SAMPLE_TIMES = ("13:45", "14:00", "14:15")

REQUIRED_SECRETS = (
    "NAVASAN_API_KEY",
)

BENCHMARK_LABELS: Dict[str, str] = {
    "open_market": "Open Market / Street Rate",
    "official": "Official Commercial USD Rate",
    "regional_transfer": "Regional Transfer Rate",
    "crypto_usdt": "Crypto Dollar (USDT)",
    "emami_gold_coin": "Emami Gold Coin",
}

PRIMARY_BENCHMARK = "open_market"
# Primary street benchmark universe is intentionally conservative.
# Navasan open-market symbol quality is currently under review and excluded from primary publication.
PRIMARY_STREET_SOURCE_UNIVERSE: Tuple[str, ...] = ("bonbast", "alanchand_street")

INDICATOR_LABELS: Dict[str, str] = {
    "street_official_gap_pct": "Street-Official Gap",
    "street_transfer_gap_pct": "Street-Transfer Gap",
    "street_crypto_gap_pct": "Street-Crypto Gap",
    "street_gold_gap_pct": "Gold FX Gap",
    "official_commercial_trend_7d": "Official 7d Trend",
}

INDICATOR_FORMULAS: Dict[str, str] = {
    "street_official_gap_pct": "((street_usd_irr - official_commercial_usd_irr) / official_commercial_usd_irr) * 100",
    "street_transfer_gap_pct": "((street_usd_irr - regional_transfer_usd_irr) / regional_transfer_usd_irr) * 100",
    "street_crypto_gap_pct": "((crypto_usdt_irr - street_usd_irr) / street_usd_irr) * 100",
    "street_gold_gap_pct": "((gold_implied_usd_irr - street_usd_irr) / street_usd_irr) * 100",
    "official_commercial_trend_7d": "7-day percent change of official_commercial_usd_irr",
}

EMAMI_COIN_GOLD_OZ = 0.235
DEFAULT_GOLD_USD_PER_OZ = 3000.0

STRICT_CANONICAL_BENCHMARKS = {"official", "regional_transfer"}

BENCHMARK_SYMBOL_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "crypto_usdt": ("usdt", "tether", "usd_tether"),
    # Prefer exact "sekkeh" where exposed; keep existing aliases as fallback.
    "emami_gold_coin": ("sekkeh", "sekke", "emami", "coin_emami", "sekeh_emami"),
}

COMMERCIAL_AUX_SOURCE_NAMES: Tuple[str, ...] = (
    "commercial_aux_a",
    "commercial_aux_b",
    "commercial_aux_c",
    "commercial_aux",
)
COMMERCIAL_PROFILE_SOURCE_NAMES: Tuple[str, ...] = (
    "commercial_profile_transfer",
    "commercial_profile_sana",
)
LEGACY_SOURCE_ALIASES: Dict[str, str] = {
    "tgju_call2": "commercial_aux_a",
    "tgju_call3": "commercial_aux_b",
    "tgju_call4": "commercial_aux_c",
    "tgju_call": "commercial_aux",
    # Legacy persisted name used before auxiliary source split.
    "commercial": "commercial_aux",
}
COMMERCIAL_AUX_SOURCE_SET = frozenset(COMMERCIAL_AUX_SOURCE_NAMES)
COMMERCIAL_PROFILE_SOURCE_SET = frozenset(COMMERCIAL_PROFILE_SOURCE_NAMES)
COMMERCIAL_AUX_OFFICIAL_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    # Priority-ordered official-market channels from auxiliary commercial endpoints.
    # Keep transfer sell first, then alternate ICE-board USD sell channels.
    "official": (
        "ice_transfer_usd_sell",
        "ice_currency_usd_sell",
        "ice_average_usd_sell",
        "ice_commodity_transfer_usd_sell",
    ),
}
COMMERCIAL_PROFILE_OFFICIAL_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    # Symbol ids tied to profile-page collectors (not JSON endpoint fields).
    # Keep legacy profile first, then fresh ICE-board alternatives.
    "commercial_profile_transfer": (
        "ice_transfer_usd_sell",
        "ice_currency_usd_sell",
        "ice_average_usd_sell",
        "ice_commodity_transfer_usd_sell",
        "mex_usd_sell",
    ),
    # SANA profile is frequently retired/blocked; prioritize live managed-market symbols.
    "commercial_profile_sana": (
        "mex_usd_sell",
        "sana_sell_usd",
        "ice_currency_usd_sell",
        "ice_transfer_usd_sell",
    ),
}

# Exact source-to-symbol mappings for production-safe parsing.
# If a strict benchmark has no entry for a source, we intentionally return unavailable.
CANONICAL_SOURCE_SYMBOLS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "navasan": {
        "open_market": ("usd_sell", "usd"),
        # Commercial managed-market sell quote.
        "official": ("mex_usd_sell",),
        "regional_transfer": ("usd_shakhs", "usd_sherkat"),
        "crypto_usdt": ("usdt",),
        "emami_gold_coin": ("sekkeh",),
    },
    **{name: dict(COMMERCIAL_AUX_OFFICIAL_SYMBOLS) for name in COMMERCIAL_AUX_SOURCE_SET},
    **{name: {"official": COMMERCIAL_PROFILE_OFFICIAL_SYMBOLS.get(name, ())} for name in COMMERCIAL_PROFILE_SOURCE_SET},
    "alanchand": {
        "regional_transfer": ("usd-hav", "usd_hav"),
        "crypto_usdt": ("usdt",),
        "emami_gold_coin": ("sekkeh",),
    },
    "bonbast": {
        "crypto_usdt": ("usdt", "tether"),
        "emami_gold_coin": ("sekkeh",),
    },
}

# Navasan exposes mixed-unit channels in a single payload:
# - street/transfer/crypto rates in toman
# - exchange-center commercial sell quote (mex_usd_sell) in rial
# - coin prices are published in toman in some payloads and rial in others; keep conservative toman default.
NAVASAN_BENCHMARK_UNITS: Dict[str, str] = {
    "open_market": "toman",
    "official": "rial",
    "regional_transfer": "toman",
    "crypto_usdt": "toman",
    "emami_gold_coin": "toman",
}


def current_mapping_fingerprint() -> str:
    material = {
        "canonical_source_symbols": CANONICAL_SOURCE_SYMBOLS,
        "navasan_benchmark_units": NAVASAN_BENCHMARK_UNITS,
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]

BONBAST_SELECTOR_MAP: Dict[str, Tuple[str, ...]] = {
    "open_market": (
        "#usd1",
        "#usd1_top",
        "#usd",
        "[data-key='usd1']",
        "[data-symbol='usd1']",
        "[data-symbol='usd']",
    ),
    "open_market_buy": (
        "#usd2",
        "[data-key='usd2']",
        "[data-symbol='usd2']",
    ),
    "crypto_usdt": (
        "#usdt",
        "#tether",
        "[data-key='usdt']",
        "[data-symbol='usdt']",
    ),
    "emami_gold_coin": (
        "#sekkeh",
        "#sekke",
        "[data-key='sekkeh']",
        "[data-symbol='sekkeh']",
        "[data-symbol='emami']",
    ),
}

BONBAST_TEXT_HINTS: Dict[str, Tuple[str, ...]] = {
    "open_market": ("usd1", "usd", "dollar", "azad", "street"),
    "open_market_buy": ("usd2", "buy", "bid", "خرید"),
    "crypto_usdt": ("usdt", "tether"),
    "emami_gold_coin": ("sekkeh", "sekke", "emami", "coin"),
}

BONBAST_USD_BUY_PATTERNS: Tuple[str, ...] = (
    r"(?is)(?:usd|dollar)[^0-9]{0,80}(?:buy|bid|خرید)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
    r"(?is)(?:buy|bid|خرید)[^0-9]{0,80}(?:usd|dollar)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
)

BONBAST_USD_SELL_PATTERNS: Tuple[str, ...] = (
    r"(?is)(?:usd|dollar)[^0-9]{0,80}(?:sell|offer|فروش)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
    r"(?is)(?:sell|offer|فروش)[^0-9]{0,80}(?:usd|dollar)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
)

BONBAST_MAX_SPREAD_PCT_DEFAULT = 0.05
BONBAST_PEER_DEVIATION_PCT_DEFAULT = 0.20

COMMERCIAL_AUX_TIMEOUT_SECONDS_DEFAULT = 20.0
COMMERCIAL_AUX_RETRY_ATTEMPTS_DEFAULT = 3
COMMERCIAL_AUX_RETRY_BACKOFF_SECONDS_DEFAULT = 1.0
COMMERCIAL_AUX_RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524})
COMMERCIAL_AUX_HEDGE_WIDTH_DEFAULT = 2
COMMERCIAL_AUX_HOST_COOLDOWN_BASE_SECONDS = 60
COMMERCIAL_AUX_HOST_COOLDOWN_MAX_SECONDS = 600

API_SOURCE_RETRY_ATTEMPTS_DEFAULT = 3
API_SOURCE_RETRY_BACKOFF_SECONDS_DEFAULT = 1.0
# Do not retry HTTP 429 (rate-limited) responses to avoid quota amplification.
API_SOURCE_RETRYABLE_HTTP_CODES = frozenset({408, 425, 500, 502, 503, 504, 520, 521, 522, 523, 524})

STATUS_HISTORY_LOOKBACK_DAYS_DEFAULT = 7
STATUS_RECENT_SUCCESS_HOURS_DEFAULT = 24 * 7
OFFICIAL_MAX_QUOTE_AGE_HOURS_DEFAULT = 48
OFFICIAL_FRESHNESS_MAX_AGE_HOURS_DEFAULT = OFFICIAL_MAX_QUOTE_AGE_HOURS_DEFAULT
# Keep official companion readings available with explicit stale labeling when
# upstream official channels stop updating for short periods.
OFFICIAL_STALE_FALLBACK_MAX_AGE_HOURS_DEFAULT = 24 * 45

COMMERCIAL_AUX_HOST_HEALTH_LOCK = threading.Lock()
COMMERCIAL_AUX_HOST_HEALTH: Dict[str, Dict[str, Any]] = {}

ALANCHAND_STREET_SELL_PATTERNS: Tuple[str, ...] = (
    r"(?is)Sell\s+rate\s+for\s+USD\s+to\s+IRR.*?<span[^>]*fs-5[^>]*>\s*([0-9][0-9,٬،.]*)\s*</span>",
)
ALANCHAND_STREET_BUY_PATTERNS: Tuple[str, ...] = (
    r"(?is)Buy\s+rate\s+for\s+USD\s+to\s+IRR.*?<span[^>]*fs-5[^>]*>\s*([0-9][0-9,٬،.]*)\s*</span>",
)
ALANCHAND_STREET_LAST_UPDATE_PATTERN = r"(?is)Last\s*update:\s*<span>\s*Time:\s*(\d{1,2}):(\d{2})\s*\(UTC([+-]\d{1,2}:\d{2})\)"
ALANCHAND_STREET_MAX_SPREAD_PCT_DEFAULT = 0.08
ALANCHAND_PUBLIC_SINGLE_RATE_PATTERNS: Tuple[str, ...] = (
    r'(?is)"price"\s*:\s*"([0-9][0-9,٬،.]*)"',
    r"(?is)<span[^>]*fs-5[^>]*>\s*([0-9][0-9,٬،.]*)\s*</span>",
)
ALANCHAND_PUBLIC_REGIONAL_URL_DEFAULT = "https://alanchand.com/en/currencies-price/usd-hav"
ALANCHAND_PUBLIC_USDT_URL_DEFAULT = "https://alanchand.com/en/crypto-price/usdt"


@dataclass
class SourceConfig:
    name: str
    url: str
    auth_mode: str  # browser_playwright | query_api_key | header_api_key | public_html | public_json
    secret_fields: Tuple[str, ...]
    benchmark_families: Tuple[str, ...]
    default_unit: str = "toman"


@dataclass
class Sample:
    source: str
    sampled_at: dt.datetime
    value: Optional[float]
    benchmark_values: Dict[str, Optional[float]]
    quote_time: Optional[dt.datetime]
    ok: bool
    stale: bool
    error: Optional[str] = None
    health: Optional[Dict[str, Any]] = None
    source_unit: str = "toman"
    normalized_unit: str = "rial"


class PipelineError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def parse_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        cleaned = (
            value.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
            .strip()
            .replace(",", "")
            .replace("٬", "")
            .replace("،", "")
            .replace(" ", "")
        )
        if not cleaned:
            return None
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
            return None
        v = float(cleaned)
        return v if math.isfinite(v) else None
    return None


def parse_number(value: Any) -> Optional[float]:
    v = parse_float(value)
    if v is None:
        return None

    if v <= 0:
        return None

    return v


def resolve_gold_usd_per_oz(daily: Optional[Dict[str, Any]] = None) -> Optional[float]:
    if isinstance(daily, dict):
        normalized = daily.get("normalized_metrics")
        if isinstance(normalized, dict):
            parsed = parse_number(normalized.get("gold_usd_per_oz"))
            if parsed is not None:
                return parsed
    env_value = parse_number(os.environ.get("GOLD_USD_PER_OZ"))
    if env_value is not None:
        return env_value
    return DEFAULT_GOLD_USD_PER_OZ


def compute_gold_implied_fx(emami_coin_irr: Optional[float], gold_usd_per_oz: Optional[float]) -> Optional[float]:
    if emami_coin_irr is None or gold_usd_per_oz in (None, 0):
        return None
    denom = float(gold_usd_per_oz) * EMAMI_COIN_GOLD_OZ
    if denom <= 0:
        return None
    implied = float(emami_coin_irr) / denom
    return implied if math.isfinite(implied) and implied > 0 else None


def blank_benchmark_values() -> Dict[str, Optional[float]]:
    return {key: None for key in BENCHMARK_LABELS}


def normalize_unit(value: Optional[float], source_unit: str) -> Optional[float]:
    if value is None:
        return None
    unit = (source_unit or "").strip().lower()
    if unit == "toman":
        return value * 10.0
    return value


def normalize_benchmark_values(values: Dict[str, Optional[float]], source_unit: str) -> Dict[str, Optional[float]]:
    return {key: normalize_unit(val, source_unit) for key, val in values.items()}


def normalize_benchmark_values_with_units(
    values: Dict[str, Optional[float]],
    benchmark_units: Dict[str, str],
    default_unit: str,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key, val in values.items():
        unit = benchmark_units.get(key, default_unit)
        out[key] = normalize_unit(val, unit)
    return out


def detect_unit_from_text(text: str, default_unit: str) -> str:
    lowered = text.lower()
    if "toman" in lowered or "tmn" in lowered or "تومان" in text:
        return "toman"
    if "rial" in lowered or "irr" in lowered or "ریال" in text:
        return "rial"
    return default_unit


def detect_source_unit(payload: Any, default_unit: str) -> str:
    unit = default_unit
    for _path, raw in flatten_json(payload):
        if isinstance(raw, str):
            unit = detect_unit_from_text(raw, unit)
            if unit in {"toman", "rial"}:
                return unit
    return unit


def parse_source_payload(source_name: str, body: str) -> Tuple[Any, str]:
    try:
        return json.loads(body), "json"
    except json.JSONDecodeError as exc:
        if source_name != "navasan":
            raise exc

        # Navasan fallback for JavaScript assignment payloads such as:
        #   var lastrates = {...};
        #   var yesterday = {...};
        js_vars: Dict[str, Any] = {}
        for match in re.finditer(r"var\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{.*?\})\s*;", body, re.S):
            var_name = str(match.group(1) or "").strip().lower()
            var_body = match.group(2)
            if not var_name or not var_body:
                continue
            try:
                js_vars[var_name] = json.loads(var_body)
            except json.JSONDecodeError:
                continue

        for preferred in ("yesterday", "lastrates"):
            parsed = js_vars.get(preferred)
            if isinstance(parsed, dict):
                return parsed, f"javascript_var:{preferred}"
        raise exc


def parse_number_from_text(text: Any) -> Optional[float]:
    if not isinstance(text, str):
        return None
    normalized = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    for token in re.findall(r"[-+]?\d[\d,٬،.]*(?:\.\d+)?", normalized):
        parsed = parse_number(token)
        if parsed is not None:
            return parsed
    return None


def env_pct(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    parsed = parse_float(raw)
    if parsed is None:
        return default
    if parsed > 1 and parsed <= 100:
        parsed = parsed / 100.0
    if parsed <= 0 or parsed >= 1:
        return default
    return float(parsed)


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    parsed = parse_float(raw)
    if parsed is None:
        return default
    ivalue = int(parsed)
    if ivalue < minimum or ivalue > maximum:
        return default
    return ivalue


def env_seconds(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    parsed = parse_float(raw)
    if parsed is None:
        return default
    if parsed < minimum or parsed > maximum:
        return default
    return float(parsed)


def bounded_value(value: Optional[float], minimum: float, maximum: float) -> Optional[float]:
    if value is None:
        return None
    if minimum <= value <= maximum:
        return value
    return None


def try_parse_datetime(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv > 10_000_000_000:
            iv = iv // 1000
        try:
            return dt.datetime.fromtimestamp(iv, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            parsed = dt.datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            pass

    if re.fullmatch(r"\d{10,13}", text):
        return try_parse_datetime(int(text))

    return None


def parse_utc_offset(offset_text: str) -> Optional[dt.timedelta]:
    match = re.fullmatch(r"([+-])(\d{1,2}):(\d{2})", offset_text.strip())
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3))
    if hours > 23 or minutes > 59:
        return None
    return sign * dt.timedelta(hours=hours, minutes=minutes)


def parse_alanchand_last_update(page_html: str, sampled_at: dt.datetime) -> Optional[dt.datetime]:
    match = re.search(ALANCHAND_STREET_LAST_UPDATE_PATTERN, page_html)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    offset = parse_utc_offset(match.group(3))
    if offset is None:
        return None

    local_tz = dt.timezone(offset)
    sampled_local = sampled_at.astimezone(local_tz)
    candidate = dt.datetime.combine(sampled_local.date(), dt.time(hour, minute), tzinfo=local_tz)
    if candidate > sampled_local + dt.timedelta(hours=12):
        candidate -= dt.timedelta(days=1)
    return candidate.astimezone(UTC)


def normalize_eastern_digits(text: str) -> str:
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))


def jalali_to_gregorian_date(jy: int, jm: int, jd: int) -> Optional[dt.date]:
    if jy < 1200 or jy > 1700:
        return None
    if jm < 1 or jm > 12:
        return None
    if jd < 1 or jd > 31:
        return None

    jy2 = jy - 979
    jm2 = jm - 1
    jd2 = jd - 1

    j_day_no = 365 * jy2 + (jy2 // 33) * 8 + ((jy2 % 33) + 3) // 4
    jalali_month_days = (31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29)
    if jm2 > 0:
        j_day_no += sum(jalali_month_days[:jm2])
    j_day_no += jd2

    g_day_no = j_day_no + 79

    gy = 1600 + 400 * (g_day_no // 146097)
    g_day_no %= 146097

    leap = True
    if g_day_no >= 36525:
        g_day_no -= 1
        gy += 100 * (g_day_no // 36524)
        g_day_no %= 36524
        if g_day_no >= 365:
            g_day_no += 1
        else:
            leap = False

    gy += 4 * (g_day_no // 1461)
    g_day_no %= 1461

    if g_day_no >= 366:
        leap = False
        g_day_no -= 1
        gy += g_day_no // 365
        g_day_no %= 365

    gd = g_day_no + 1
    gregorian_month_days = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 1
    while gm <= 12 and gd > gregorian_month_days[gm]:
        gd -= gregorian_month_days[gm]
        gm += 1

    try:
        return dt.date(gy, gm, gd)
    except ValueError:
        return None


def parse_tgju_profile_quote_time(page_html: str) -> Optional[dt.datetime]:
    normalized = normalize_eastern_digits(page_html)
    matches = re.findall(r"(?is)<td>\s*(14\d{2})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})\s*</td>", normalized)
    if not matches:
        return None
    gregorian_dates: List[dt.date] = []
    for jy_text, jm_text, jd_text in matches:
        jy = int(jy_text)
        jm = int(jm_text)
        jd = int(jd_text)
        gregorian_date = jalali_to_gregorian_date(jy, jm, jd)
        if gregorian_date is not None:
            gregorian_dates.append(gregorian_date)
    if not gregorian_dates:
        return None
    # Profile pages can include multiple historical rows; use the newest date.
    gregorian_date = max(gregorian_dates)
    # TGJU profile pages publish daily bars; use local noon to avoid edge rollover artifacts.
    local_noon = dt.datetime.combine(gregorian_date, dt.time(12, 0), tzinfo=IRAN_TZ)
    return local_noon.astimezone(UTC)


def parse_tgju_profile_current_value(page_html: str) -> Optional[float]:
    normalized = normalize_eastern_digits(page_html)
    patterns = (
        r"(?is)<td[^>]*>\s*نرخ فعلی\s*</td>\s*<td[^>]*>\s*([0-9][0-9,٬،.]*)\s*</td>",
        r'(?is)data-col\s*=\s*"info\.last_trade\.PDrCotVal"\s*>\s*([0-9][0-9,٬،.]*)\s*<',
    )
    return extract_number_with_patterns(normalized, patterns, 100_000, 4_000_000)


def flatten_json(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_json(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            items.extend(flatten_json(value, path))
    else:
        items.append((prefix, obj))
    return items


def normalize_symbol_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_numeric_from_quote_obj(obj: Dict[str, Any]) -> Optional[float]:
    for field in ("value", "price", "last", "close", "sell", "buy", "rate", "amount", "p"):
        if field in obj:
            parsed = parse_number(obj.get(field))
            if parsed is not None:
                return parsed
    return None


def extract_value_by_symbol_candidates(payload: Any, candidates: Tuple[str, ...]) -> Optional[float]:
    target_tokens = {normalize_symbol_token(item) for item in candidates}

    def matches(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        token = normalize_symbol_token(text)
        if not token:
            return False
        return token in target_tokens

    def walk(node: Any) -> List[float]:
        found: List[float] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if matches(key):
                    if isinstance(value, dict):
                        parsed = extract_numeric_from_quote_obj(value)
                        if parsed is not None:
                            found.append(parsed)
                    else:
                        parsed = parse_number(value)
                        if parsed is not None:
                            found.append(parsed)

            id_fields = ("symbol", "name", "slug", "code", "title", "label", "item", "id")
            if any(matches(node.get(field)) for field in id_fields):
                parsed = extract_numeric_from_quote_obj(node)
                if parsed is not None:
                    found.append(parsed)

            for value in node.values():
                found.extend(walk(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(walk(item))
        return found

    values = walk(payload)
    if not values:
        return None
    return median(values)


def extract_value_by_symbol_priority(payload: Any, candidates: Tuple[str, ...]) -> Tuple[Optional[float], Optional[str]]:
    ranked: List[Tuple[int, dt.datetime, int, float, str]] = []
    for index, symbol in enumerate(candidates):
        parsed = extract_value_by_symbol_candidates(payload, (symbol,))
        if parsed is None:
            continue
        symbol_quote_time = extract_symbol_quote_time(payload, (symbol,))
        effective_time = symbol_quote_time if symbol_quote_time is not None else dt.datetime(1970, 1, 1, tzinfo=UTC)
        has_quote_time = 1 if symbol_quote_time is not None else 0
        # Prefer freshest symbol-level quote time; fall back to declared priority.
        ranked.append((has_quote_time, effective_time, -index, parsed, symbol))
    if not ranked:
        return None, None
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    _has_quote_time, _effective_time, _priority, value, selected_symbol = ranked[0]
    return value, selected_symbol


def extract_quote_time(payload: Any) -> Optional[dt.datetime]:
    candidates = flatten_json(payload)
    scored: List[Tuple[int, dt.datetime]] = []
    for path, raw in candidates:
        path_l = path.lower()
        if not any(tok in path_l for tok in ("time", "date", "updated", "timestamp", "ts")):
            continue
        parsed = try_parse_datetime(raw)
        if not parsed:
            continue
        score = 0
        if "updated" in path_l:
            score += 3
        if "timestamp" in path_l or path_l.endswith(".ts"):
            score += 2
        if "usd" in path_l:
            score += 1
        scored.append((score, parsed))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def extract_symbol_quote_time(payload: Any, symbols: Tuple[str, ...]) -> Optional[dt.datetime]:
    target_tokens = {normalize_symbol_token(item) for item in symbols}
    if not target_tokens:
        return None

    def matches(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        token = normalize_symbol_token(text)
        return bool(token and token in target_tokens)

    def parse_ts_from_obj(obj: Dict[str, Any]) -> Optional[dt.datetime]:
        for field in ("timestamp", "ts", "updated", "updated_at", "time", "date"):
            parsed = try_parse_datetime(obj.get(field))
            if parsed is not None:
                return parsed
        return None

    found: List[dt.datetime] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if matches(key):
                    if isinstance(value, dict):
                        parsed = parse_ts_from_obj(value)
                        if parsed is not None:
                            found.append(parsed)
                    else:
                        parsed = try_parse_datetime(value)
                        if parsed is not None:
                            found.append(parsed)

            id_fields = ("symbol", "name", "slug", "code", "title", "label", "item", "id")
            if any(matches(node.get(field)) for field in id_fields):
                parsed = parse_ts_from_obj(node)
                if parsed is not None:
                    found.append(parsed)

            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    if not found:
        return None
    return max(found)


def extract_symbol_quote_time_by_priority(
    payload: Any, candidates: Tuple[str, ...]
) -> Tuple[Optional[dt.datetime], Optional[str]]:
    for symbol in candidates:
        parsed = extract_symbol_quote_time(payload, (symbol,))
        if parsed is not None:
            return parsed, symbol
    return None, None


def extract_benchmark_value(payload: Any, benchmark: str) -> Optional[float]:
    candidates = flatten_json(payload)
    ranked: List[Tuple[int, float]] = []

    for path, raw in candidates:
        path_l = path.lower()
        num = parse_number(raw)
        if num is None:
            continue

        score = 0
        has_usd = "usd" in path_l
        has_irr = "irr" in path_l or "rial" in path_l
        has_sell = any(tok in path_l for tok in ("sell", "seller", "offer"))
        has_buy = any(tok in path_l for tok in ("buy", "buyer", "bid"))
        has_open = any(tok in path_l for tok in ("open", "market", "street", "free", "azad"))
        has_usdt = any(tok in path_l for tok in ("usdt", "tether"))
        has_emami = any(tok in path_l for tok in ("emami", "sekke", "coin"))
        has_gold = any(tok in path_l for tok in ("gold", "tala"))
        has_official = any(tok in path_l for tok in ("official", "bank", "cbi", "gov", "government"))
        has_transfer = any(tok in path_l for tok in ("transfer", "hawala", "remit", "remittance"))
        has_price = "price" in path_l or path_l.endswith(".value")

        if benchmark == "open_market":
            if has_usd:
                score += 4
            if has_irr:
                score += 4
            if has_open:
                score += 3
            if has_sell:
                score += 1
            if has_buy:
                score -= 1
            if 150_000 <= num <= 2_500_000:
                score += 2
        elif benchmark == "official":
            if not has_official:
                continue
            if has_usd:
                score += 4
            if has_irr:
                score += 4
            score += 3
            if 10_000 <= num <= 2_500_000:
                score += 2
        elif benchmark == "regional_transfer":
            if not has_transfer:
                continue
            if has_usd:
                score += 4
            if has_irr:
                score += 4
            score += 3
            if 150_000 <= num <= 2_500_000:
                score += 2
        elif benchmark == "crypto_usdt":
            if not has_usdt:
                continue
            score += 5
            if has_irr:
                score += 3
            if has_usd:
                score += 1
            if 150_000 <= num <= 2_500_000:
                score += 2
        elif benchmark == "emami_gold_coin":
            if not (has_emami or has_gold):
                continue
            if has_emami:
                score += 4
            if has_gold:
                score += 2
            if 5_000_000 <= num <= 20_000_000_000:
                score += 2
        else:
            continue

        if has_price:
            score += 1

        if score >= 4:
            ranked.append((score, num))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def extract_benchmark_values_with_metadata(
    payload: Any, source_name: Optional[str] = None
) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    source_key = (source_name or "").strip().lower()
    selected_symbol_by_benchmark: Dict[str, str] = {}

    def resolve(benchmark: str) -> Optional[float]:
        canonical_map = CANONICAL_SOURCE_SYMBOLS.get(source_key, {})
        canonical_symbols = canonical_map.get(benchmark)
        if canonical_symbols:
            by_symbol, selected_symbol = extract_value_by_symbol_priority(payload, canonical_symbols)
            if by_symbol is not None:
                if selected_symbol:
                    selected_symbol_by_benchmark[benchmark] = selected_symbol
                return by_symbol
            if benchmark in STRICT_CANONICAL_BENCHMARKS:
                return None

        if benchmark in STRICT_CANONICAL_BENCHMARKS:
            return None

        symbol_candidates = BENCHMARK_SYMBOL_CANDIDATES.get(benchmark)
        if symbol_candidates:
            by_symbol = extract_value_by_symbol_candidates(payload, symbol_candidates)
            if by_symbol is not None:
                return by_symbol
        return extract_benchmark_value(payload, benchmark)

    values = {
        "open_market": resolve("open_market"),
        "official": resolve("official"),
        "regional_transfer": resolve("regional_transfer"),
        "crypto_usdt": resolve("crypto_usdt"),
        "emami_gold_coin": resolve("emami_gold_coin"),
    }
    return values, selected_symbol_by_benchmark


def extract_benchmark_values(payload: Any, source_name: Optional[str] = None) -> Dict[str, Optional[float]]:
    values, _selected = extract_benchmark_values_with_metadata(payload, source_name)
    return values


def extract_usd_irr(payload: Any) -> Optional[float]:
    return extract_benchmark_values(payload).get(PRIMARY_BENCHMARK)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def median(values: List[float]) -> float:
    return float(statistics.median(values))


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def css_class(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-") or "unknown"


def read_template(templates_dir: Path, name: str) -> Template:
    path = templates_dir / name
    return Template(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def iso_date(d: dt.date) -> str:
    return d.isoformat()


def iso_ts(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_hhmm_utc(value: str) -> dt.time:
    text = value.strip()
    if not re.fullmatch(r"\d{2}:\d{2}", text):
        raise PipelineError(f"invalid HH:MM time: {value!r}")
    hour = int(text[:2])
    minute = int(text[3:])
    if hour > 23 or minute > 59:
        raise PipelineError(f"invalid HH:MM time: {value!r}")
    return dt.time(hour, minute)


def parse_sample_times(value: str) -> Tuple[dt.time, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise PipelineError("sample times list cannot be empty")
    times = tuple(parse_hhmm_utc(p) for p in parts)
    if len(set(times)) != len(times):
        raise PipelineError("sample times list contains duplicates")
    return tuple(sorted(times))


def should_sleep_until(target: dt.datetime, skip_waits: bool) -> None:
    if skip_waits:
        return
    now = utc_now()
    if now >= target:
        return
    time.sleep((target - now).total_seconds())


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value and value.strip():
        return value
    return default


def env_first(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value
    return default


def canonical_source_name(name: str) -> str:
    normalized = name.strip().lower()
    return LEGACY_SOURCE_ALIASES.get(normalized, normalized)


def is_commercial_aux_source(name: str) -> bool:
    return canonical_source_name(name) in COMMERCIAL_AUX_SOURCE_SET


def build_source_configs() -> List[SourceConfig]:
    return [
        SourceConfig(
            name="bonbast",
            url=env_or_default("BONBAST_SITE_URL", "https://bonbast.com"),
            auth_mode="browser_playwright",
            secret_fields=(),
            benchmark_families=("open_market", "official", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
            default_unit="toman",
        ),
        SourceConfig(
            name="alanchand_street",
            url=env_or_default("ALANCHAND_STREET_URL", "https://alanchand.com/en/currencies-price/usd"),
            auth_mode="public_html",
            secret_fields=(),
            benchmark_families=("open_market",),
            default_unit="rial",
        ),
        SourceConfig(
            name="navasan",
            url=env_or_default("NAVASAN_API_URL", "https://api.navasan.tech/latest/"),
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
            benchmark_families=("open_market", "official", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
            default_unit="toman",
        ),
        SourceConfig(
            name="commercial_aux_a",
            url=env_first(
                "COMMERCIAL_AUX_A_URL",
                "TGJU_CALL2_URL",
                default="https://call2.tgju.org/ajax.json",
            ),
            auth_mode="public_json",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        ),
        SourceConfig(
            name="commercial_aux_b",
            url=env_first(
                "COMMERCIAL_AUX_B_URL",
                "TGJU_CALL3_URL",
                default="https://call3.tgju.org/ajax.json",
            ),
            auth_mode="public_json",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        ),
        SourceConfig(
            name="commercial_aux_c",
            url=env_first(
                "COMMERCIAL_AUX_C_URL",
                "TGJU_CALL4_URL",
                default="https://call4.tgju.org/ajax.json",
            ),
            auth_mode="public_json",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        ),
        SourceConfig(
            name="commercial_aux",
            url=env_first(
                "COMMERCIAL_AUX_URL",
                "TGJU_CALL_URL",
                default="https://call.tgju.org/ajax.json",
            ),
            auth_mode="public_json",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        ),
        SourceConfig(
            name="commercial_profile_transfer",
            url=env_first(
                "COMMERCIAL_PROFILE_TRANSFER_URL",
                default="https://www.tgju.org/profile/ice_transfer_usd_sell",
            ),
            auth_mode="public_html",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        ),
        SourceConfig(
            name="commercial_profile_sana",
            url=env_first(
                "COMMERCIAL_PROFILE_SANA_URL",
                default="https://www.tgju.org/profile/mex_usd_sell",
            ),
            auth_mode="public_html",
            secret_fields=(),
            benchmark_families=("official",),
            default_unit="rial",
        ),
        SourceConfig(
            name="alanchand",
            url=env_or_default("ALANCHAND_API_URL", "https://api.alanchand.com/v1/rates"),
            auth_mode="header_api_key",
            secret_fields=("ALANCHAND_API_KEY",),
            benchmark_families=("regional_transfer", "crypto_usdt", "emami_gold_coin"),
            default_unit="toman",
        ),
    ]


def missing_secrets() -> List[str]:
    missing = []
    for key in REQUIRED_SECRETS:
        if not os.environ.get(key):
            missing.append(key)
    return missing


def build_request(config: SourceConfig) -> urllib.request.Request:
    if config.auth_mode == "browser_playwright":
        raise PipelineError("browser sources do not use HTTP request builder")

    url = config.url
    headers = {"User-Agent": "rialwatch-pipeline/0.2"}
    if config.auth_mode == "public_html":
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    else:
        headers["Accept"] = "application/json"

    if config.auth_mode == "query_api_key":
        query = {"api_key": os.environ["NAVASAN_API_KEY"]}
        url = with_query(url, query)
    elif config.auth_mode == "header_api_key":
        key = os.environ["ALANCHAND_API_KEY"]
        headers["Authorization"] = f"Bearer {key}"
        if config.name == "alanchand":
            # AlanChand documents the API at api.alanchand.com with Bearer auth.
            # Normalize stale /v1/rates paths and ensure explicit symbols for canonical parsing.
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").lower()
            if host in {"alanchand.com", "www.alanchand.com"}:
                parsed = parsed._replace(netloc="api.alanchand.com")
                host = "api.alanchand.com"
            if host in {"api.alanchand.com", "www.api.alanchand.com"}:
                parsed = parsed._replace(path="/")
                existing = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                if "type" not in existing:
                    existing["type"] = ["currency"]
                if "symbols" not in existing:
                    existing["symbols"] = ["usd-hav"]
                normalized_query = urllib.parse.urlencode(existing, doseq=True)
                parsed = parsed._replace(query=normalized_query)
            url = urllib.parse.urlunparse(parsed)
        else:
            headers["X-API-Key"] = key
            url = with_query(url, {"api_key": key})
    elif is_commercial_aux_source(config.name):
        # Auxiliary commercial endpoints are aggressively cached at edge on some hosts; force a fresh variant key.
        url = with_query(url, {"rev": str(time.time_ns())})

    return urllib.request.Request(url=url, headers=headers, method="GET")


def is_commercial_aux_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code) in COMMERCIAL_AUX_RETRYABLE_HTTP_CODES
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    return False


def is_api_source_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code) in API_SOURCE_RETRYABLE_HTTP_CODES
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    return False


def canonical_commercial_aux_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    path = parsed.path or "/"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def commercial_aux_endpoint_pool(preferred_url: str) -> List[str]:
    candidates = [
        preferred_url,
        env_first("COMMERCIAL_AUX_URL", "TGJU_CALL_URL", default="https://call.tgju.org/ajax.json"),
        env_first("COMMERCIAL_AUX_A_URL", "TGJU_CALL2_URL", default="https://call2.tgju.org/ajax.json"),
        env_first("COMMERCIAL_AUX_B_URL", "TGJU_CALL3_URL", default="https://call3.tgju.org/ajax.json"),
        env_first("COMMERCIAL_AUX_C_URL", "TGJU_CALL4_URL", default="https://call4.tgju.org/ajax.json"),
    ]
    deduped: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = canonical_commercial_aux_url(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def record_commercial_aux_host_result(url: str, success: bool) -> None:
    key = canonical_commercial_aux_url(url)
    now_epoch = time.time()
    with COMMERCIAL_AUX_HOST_HEALTH_LOCK:
        state = COMMERCIAL_AUX_HOST_HEALTH.setdefault(
            key,
            {
                "successes": 0,
                "failures": 0,
                "consecutive_failures": 0,
                "cooldown_until": 0.0,
            },
        )
        if success:
            state["successes"] = int(state.get("successes", 0)) + 1
            state["consecutive_failures"] = 0
            state["cooldown_until"] = 0.0
            return
        consecutive_failures = int(state.get("consecutive_failures", 0)) + 1
        state["failures"] = int(state.get("failures", 0)) + 1
        state["consecutive_failures"] = consecutive_failures
        cooldown_seconds = min(
            COMMERCIAL_AUX_HOST_COOLDOWN_BASE_SECONDS * consecutive_failures,
            COMMERCIAL_AUX_HOST_COOLDOWN_MAX_SECONDS,
        )
        state["cooldown_until"] = now_epoch + float(cooldown_seconds)


def ranked_commercial_aux_endpoints(preferred_url: str) -> List[str]:
    urls = commercial_aux_endpoint_pool(preferred_url)
    preferred_key = canonical_commercial_aux_url(preferred_url)
    now_epoch = time.time()
    with COMMERCIAL_AUX_HOST_HEALTH_LOCK:
        state_snapshot = {key: dict(value) for key, value in COMMERCIAL_AUX_HOST_HEALTH.items()}

    def sort_key(url: str) -> Tuple[int, int, int, int]:
        key = canonical_commercial_aux_url(url)
        state = state_snapshot.get(key, {})
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        in_cooldown = now_epoch < cooldown_until
        preferred_penalty = 0 if key == preferred_key else 1
        consecutive_failures = int(state.get("consecutive_failures", 0) or 0)
        successes = int(state.get("successes", 0) or 0)
        failures = int(state.get("failures", 0) or 0)
        reliability_penalty = max(0, failures - successes)
        return (1 if in_cooldown else 0, preferred_penalty, consecutive_failures, reliability_penalty)

    return sorted(urls, key=sort_key)


def fetch_request_body(req: urllib.request.Request, timeout_seconds: float) -> Tuple[str, str]:
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        final_url = resp.geturl()
    return body, final_url


def official_quote_is_fresh(quote_time: Optional[dt.datetime], sampled_at: dt.datetime) -> bool:
    if quote_time is None:
        return False
    max_age_hours = env_int(
        "OFFICIAL_MAX_QUOTE_AGE_HOURS",
        OFFICIAL_MAX_QUOTE_AGE_HOURS_DEFAULT,
        minimum=1,
        maximum=24 * 30,
    )
    max_age = dt.timedelta(hours=max_age_hours)
    return (sampled_at - quote_time) <= max_age


def official_quote_within_stale_fallback(quote_time: Optional[dt.datetime], sampled_at: dt.datetime) -> bool:
    if quote_time is None:
        return False
    max_age_hours = env_int(
        "OFFICIAL_STALE_FALLBACK_MAX_AGE_HOURS",
        OFFICIAL_STALE_FALLBACK_MAX_AGE_HOURS_DEFAULT,
        minimum=24,
        maximum=24 * 365,
    )
    max_age = dt.timedelta(hours=max_age_hours)
    return (sampled_at - quote_time) <= max_age


def with_query(url: str, extra_params: Dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    existing = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key, value in extra_params.items():
        existing[key] = [value]
    query = urllib.parse.urlencode(existing, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=query))


def redact_url_for_health(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    redacted: Dict[str, List[str]] = {}
    for key, values in query.items():
        key_l = key.lower()
        if any(token in key_l for token in ("key", "token", "auth", "secret")):
            redacted[key] = ["***"]
        else:
            redacted[key] = values
    redacted_query = urllib.parse.urlencode(redacted, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=redacted_query))


def tgju_profile_url_for_symbol(base_url: str, symbol: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    path = parsed.path or ""
    if "/profile/" in path:
        prefix = path.split("/profile/", 1)[0]
        new_path = f"{prefix}/profile/{symbol}"
    else:
        new_path = f"/profile/{symbol}"
    return urllib.parse.urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))


def extract_bonbast_from_selector_results(selector_results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    extracted = blank_benchmark_values()
    ranges: Dict[str, Tuple[float, float]] = {
        "open_market": (100_000, 3_000_000),
        "crypto_usdt": (100_000, 3_000_000),
        "emami_gold_coin": (1_000_000, 20_000_000_000),
    }

    for benchmark, entries_any in selector_results.items():
        entries = entries_any if isinstance(entries_any, list) else []
        value: Optional[float] = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for field in ("dataValue", "dataPrice", "value", "content", "text"):
                candidate = parse_number_from_text(str(entry.get(field) or ""))
                if candidate is None:
                    continue
                bounds = ranges.get(benchmark)
                if bounds is None:
                    value = candidate
                else:
                    value = bounded_value(candidate, bounds[0], bounds[1])
                if value is not None:
                    break
            if value is not None:
                break
        if benchmark in extracted:
            extracted[benchmark] = value
    return extracted


def extract_bonbast_value_from_text(page_text: str, hints: Tuple[str, ...], minimum: float, maximum: float) -> Optional[float]:
    normalized = page_text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    number_pattern = r"([0-9][0-9,٬،.]{2,})"
    for hint in hints:
        esc = re.escape(hint)
        around_patterns = (
            rf"(?i){esc}[^\d]{{0,40}}{number_pattern}",
            rf"(?i){number_pattern}[^\d]{{0,20}}{esc}",
        )
        for pattern in around_patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            captured = match.group(1)
            parsed = parse_number(captured)
            bounded = bounded_value(parsed, minimum, maximum)
            if bounded is not None:
                return bounded
    return None


def extract_number_with_patterns(text: str, patterns: Tuple[str, ...], minimum: float, maximum: float) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        parsed = parse_number(match.group(1))
        bounded = bounded_value(parsed, minimum, maximum)
        if bounded is not None:
            return bounded
    return None


def extract_bonbast_selector_numeric(
    selector_results: Dict[str, Any], key: str, minimum: float, maximum: float
) -> Tuple[Optional[float], Optional[str]]:
    entries_any = selector_results.get(key)
    entries = entries_any if isinstance(entries_any, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field in ("dataValue", "dataPrice", "value", "content", "text"):
            candidate = parse_number_from_text(str(entry.get(field) or ""))
            bounded = bounded_value(candidate, minimum, maximum)
            if bounded is not None:
                selector = entry.get("selector")
                return bounded, str(selector) if selector else None
    return None, None


def validate_bonbast_usd_quote(
    buy_rial: Optional[float], sell_rial: Optional[float], max_spread_pct: float
) -> Dict[str, Any]:
    if buy_rial is None or sell_rial is None:
        return {"ok": False, "reason": "missing USD buy/sell quote", "spread_pct": None}
    if buy_rial > sell_rial:
        return {"ok": False, "reason": "buy quote above sell quote", "spread_pct": None}
    if sell_rial <= 0:
        return {"ok": False, "reason": "invalid sell quote", "spread_pct": None}
    spread_pct = (sell_rial - buy_rial) / sell_rial
    if spread_pct <= 0:
        return {"ok": False, "reason": "non-positive bid/ask spread", "spread_pct": spread_pct}
    if spread_pct > max_spread_pct:
        return {
            "ok": False,
            "reason": "bid/ask spread exceeds configured maximum",
            "spread_pct": spread_pct,
            "max_spread_pct": max_spread_pct,
        }
    return {"ok": True, "reason": None, "spread_pct": spread_pct, "max_spread_pct": max_spread_pct}


def fetch_bonbast_browser(
    config: SourceConfig, sampled_at: dt.datetime, window_start_dt: dt.datetime, window_end_dt: dt.datetime
) -> Sample:
    health: Dict[str, Any] = {
        "collector": "playwright",
        "fetch_mode": "playwright",
        "scrape_timestamp": iso_ts(sampled_at),
        "page_load_ok": False,
        "selector_success": False,
        "fetch_success": False,
        "failure_reason": None,
        "selector_used": None,
        "fetch_duration_ms": None,
        "content_length": None,
        "final_url": None,
        "error_type": None,
    }
    benchmark_values = blank_benchmark_values()
    source_unit = config.default_unit
    normalized_unit = "rial"
    quote_time = sampled_at

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on runtime environment.
        health["error_type"] = "playwright_unavailable"
        health["error_detail"] = str(exc)
        health["failure_reason"] = "playwright unavailable"
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=False,
            error="bonbast browser collector unavailable",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    selector_script = """
    (selectorMap) => {
      const readNode = (el) => {
        if (!el) return null;
        return {
          text: (el.textContent || '').trim(),
          value: (el.value || '').toString(),
          dataValue: (el.getAttribute('data-value') || '').toString(),
          dataPrice: (el.getAttribute('data-price') || '').toString(),
          content: (el.getAttribute('content') || '').toString()
        };
      };

      const selectorResults = {};
      for (const [benchmark, selectors] of Object.entries(selectorMap)) {
        selectorResults[benchmark] = [];
        for (const selector of selectors) {
          const node = document.querySelector(selector);
          if (!node) continue;
          selectorResults[benchmark].push({
            selector,
            ...readNode(node)
          });
        }
      }

      return {
        selector_results: selectorResults,
        page_text: (document.body ? document.body.innerText : '')
      };
    }
    """

    started_at = time.monotonic()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            response = page.goto(config.url, wait_until="domcontentloaded", timeout=30_000)
            health["page_load_ok"] = True
            page.wait_for_timeout(1_000)
            scrape_payload = page.evaluate(selector_script, BONBAST_SELECTOR_MAP)
            rendered_html = page.content()
            final_url = page.url
            browser.close()
            health["http_status"] = response.status if response is not None else None
            health["final_url"] = final_url
            health["content_length"] = len(rendered_html.encode("utf-8"))
    except PlaywrightTimeoutError as exc:
        health["error_type"] = "page_timeout"
        health["error_detail"] = str(exc)
        health["failure_reason"] = "bonbast page load timeout"
        health["fetch_duration_ms"] = int((time.monotonic() - started_at) * 1000)
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=False,
            error="bonbast page load timeout",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except PlaywrightError as exc:
        health["error_type"] = "playwright_error"
        health["error_detail"] = str(exc)
        health["failure_reason"] = "bonbast browser scrape error"
        health["fetch_duration_ms"] = int((time.monotonic() - started_at) * 1000)
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=False,
            error="bonbast browser scrape error",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except Exception as exc:  # pragma: no cover - defensive.
        health["error_type"] = "unknown_error"
        health["error_detail"] = str(exc)
        health["failure_reason"] = "bonbast browser scrape error"
        health["fetch_duration_ms"] = int((time.monotonic() - started_at) * 1000)
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=False,
            error="bonbast browser scrape error",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    health["fetch_duration_ms"] = int((time.monotonic() - started_at) * 1000)
    selector_results = scrape_payload.get("selector_results") if isinstance(scrape_payload, dict) else {}
    page_text = scrape_payload.get("page_text") if isinstance(scrape_payload, dict) else ""
    selector_results = selector_results if isinstance(selector_results, dict) else {}
    page_text = page_text if isinstance(page_text, str) else ""
    source_unit = detect_unit_from_text(page_text, config.default_unit)

    buy_raw, buy_selector = extract_bonbast_selector_numeric(selector_results, "open_market_buy", 100_000, 3_000_000)
    sell_raw, sell_selector = extract_bonbast_selector_numeric(selector_results, "open_market", 100_000, 3_000_000)
    if buy_raw is None:
        buy_raw = extract_number_with_patterns(page_text, BONBAST_USD_BUY_PATTERNS, 100_000, 3_000_000)
        if buy_raw is not None:
            buy_selector = "text_pattern_buy"
    if sell_raw is None:
        sell_raw = extract_number_with_patterns(page_text, BONBAST_USD_SELL_PATTERNS, 100_000, 3_000_000)
        if sell_raw is not None:
            sell_selector = "text_pattern_sell"
    if sell_raw is None:
        sell_raw = extract_bonbast_value_from_text(page_text, BONBAST_TEXT_HINTS["open_market"], 100_000, 3_000_000)
        if sell_raw is not None:
            sell_selector = "text_hint_open_market"

    buy_rial = normalize_unit(buy_raw, source_unit)
    sell_rial = normalize_unit(sell_raw, source_unit)
    mid_rial = ((buy_rial + sell_rial) / 2.0) if buy_rial is not None and sell_rial is not None else None

    extracted = extract_bonbast_from_selector_results(selector_results)
    extracted["open_market"] = sell_raw if sell_raw is not None else extracted.get("open_market")
    extracted["crypto_usdt"] = extracted.get("crypto_usdt") or extract_bonbast_value_from_text(
        page_text, BONBAST_TEXT_HINTS["crypto_usdt"], 100_000, 3_000_000
    )
    extracted["emami_gold_coin"] = extracted.get("emami_gold_coin") or extract_bonbast_value_from_text(
        page_text, BONBAST_TEXT_HINTS["emami_gold_coin"], 1_000_000, 20_000_000_000
    )
    benchmark_values.update(normalize_benchmark_values(extracted, source_unit))

    health["selector_success"] = any(v is not None for v in extracted.values())
    health["extracted_values"] = {k: benchmark_values.get(k) for k in ("open_market", "crypto_usdt", "emami_gold_coin")}
    health["bonbast_usd_buy"] = buy_rial
    health["bonbast_usd_sell"] = sell_rial
    health["bonbast_usd_mid"] = mid_rial
    health["selector_used"] = {"buy": buy_selector, "sell": sell_selector}
    health["parse_result"] = {
        "buy_raw": buy_raw,
        "sell_raw": sell_raw,
        "buy_rial": buy_rial,
        "sell_rial": sell_rial,
        "mid_rial": mid_rial,
    }
    health["source_unit"] = source_unit
    health["normalized_unit"] = normalized_unit
    health["selector_result_counts"] = {
        benchmark: len(entries) if isinstance(entries, list) else 0 for benchmark, entries in selector_results.items()
    }

    validation = validate_bonbast_usd_quote(
        buy_rial=buy_rial,
        sell_rial=sell_rial,
        max_spread_pct=env_pct("BONBAST_MAX_SPREAD_PCT", BONBAST_MAX_SPREAD_PCT_DEFAULT),
    )
    health["validation_result"] = validation

    value = benchmark_values.get(PRIMARY_BENCHMARK)
    stale = not is_within_window_minute(sampled_at, window_start_dt, window_end_dt)
    if value is None:
        health["error_type"] = "selector_parse_failed"
        health["failure_reason"] = "unable to parse USD/IRR"
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=stale,
            error="unable to parse USD/IRR",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    if not validation.get("ok"):
        reason = str(validation.get("reason") or "bonbast validation failed")
        health["error_type"] = "validation_failed"
        health["failure_reason"] = reason
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=stale,
            error=reason,
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    health["fetch_success"] = True
    return Sample(
        config.name,
        sampled_at,
        value,
        benchmark_values,
        quote_time,
        ok=not stale,
        stale=stale,
        error="stale quote" if stale else None,
        health=health,
        source_unit=source_unit,
        normalized_unit=normalized_unit,
    )


def extract_alanchand_jsonld_price(page_html: str) -> Optional[float]:
    for match in re.finditer(r'(?is)<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', page_html):
        raw = (match.group(1) or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        queue: List[Any] = [payload]
        while queue:
            node = queue.pop(0)
            if isinstance(node, dict):
                node_type = str(node.get("@type") or "").strip().lower()
                if node_type == "product":
                    offers = node.get("offers")
                    if isinstance(offers, dict):
                        parsed = parse_number(offers.get("price"))
                        if parsed is not None:
                            return parsed
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        queue.append(value)
            elif isinstance(node, list):
                queue.extend(node)
    return None


def fetch_tgju_profile_official_public(
    config: SourceConfig,
    sampled_at: dt.datetime,
    window_start_dt: dt.datetime,
    window_end_dt: dt.datetime,
    profile_symbol: str,
) -> Sample:
    health: Dict[str, Any] = {
        "collector": "http_html",
        "fetch_mode": "public_html",
        "scrape_timestamp": iso_ts(sampled_at),
        "fetch_success": False,
        "failure_reason": None,
        "error_type": None,
        "selector_used": None,
        "profile_symbol": profile_symbol,
    }
    benchmark_values = blank_benchmark_values()
    normalized_unit = "rial"
    source_unit = config.default_unit

    body = ""
    final_url = ""
    request_urls: List[str] = []
    user_agents = (
        "rialwatch-pipeline/0.2",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    )
    last_error: Optional[str] = None

    for ua in user_agents:
        try:
            req = urllib.request.Request(
                url=config.url,
                method="GET",
                headers={
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "fa,en-US;q=0.9,en;q=0.8",
                },
            )
            redacted_url = redact_url_for_health(req.full_url)
            request_urls.append(redacted_url)
            health["request_url"] = redacted_url
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                final_url = resp.geturl()
            if body.strip():
                health["user_agent"] = ua
                break
            last_error = "empty response body"
        except urllib.error.HTTPError as exc:
            last_error = f"http {exc.code}"
            health["error_type"] = "http_error"
            health["failure_reason"] = last_error
        except urllib.error.URLError as exc:
            last_error = f"network: {exc.reason}"
            health["error_type"] = "network_error"
            health["failure_reason"] = last_error
        except TimeoutError:
            last_error = "timeout"
            health["error_type"] = "timeout"
            health["failure_reason"] = last_error

    health["request_urls"] = request_urls
    health["final_url"] = final_url or None
    health["content_length"] = len(body.encode("utf-8")) if body else 0
    health["page_load_ok"] = bool(body.strip())

    if not body.strip():
        reason = last_error or "empty response body"
        if health.get("error_type") is None:
            health["error_type"] = "empty_body"
        health["failure_reason"] = reason
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            None,
            ok=False,
            stale=False,
            error=reason,
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    quote_time = parse_tgju_profile_quote_time(body)
    raw_value = parse_tgju_profile_current_value(body)
    official_value = normalize_unit(raw_value, source_unit)
    benchmark_values["official"] = official_value

    if quote_time is not None:
        health["benchmark_quote_times"] = {"official": iso_ts(quote_time)}
    health["source_unit"] = source_unit
    health["normalized_unit"] = normalized_unit
    health["raw_extracted_values"] = {"official": raw_value}
    health["extracted_values"] = {"official": official_value}
    health["selector_used"] = "tgju_profile_rate_table_or_price_span"

    stale = bool(quote_time is not None and not is_within_window_minute(quote_time, window_start_dt, window_end_dt))
    if official_value is None:
        health["error_type"] = "selector_parse_failed"
        health["failure_reason"] = "unable to parse official quote from profile page"
        health["validation_result"] = {"ok": False, "reason": "unable to parse official quote from profile page"}
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=stale,
            error="unable to parse official quote from profile page",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    health["validation_result"] = {"ok": True, "reason": None}
    health["fetch_success"] = True
    return Sample(
        config.name,
        sampled_at,
        official_value,
        benchmark_values,
        quote_time,
        ok=True,
        stale=stale,
        error="stale quote" if stale else None,
        health=health,
        source_unit=source_unit,
        normalized_unit=normalized_unit,
    )


def fetch_tgju_profile_official_fallback_aux(
    config: SourceConfig,
    sampled_at: dt.datetime,
    window_start_dt: dt.datetime,
    window_end_dt: dt.datetime,
    fallback_reason: str,
    profile_attempts: List[Dict[str, Any]],
) -> Optional[Sample]:
    fallback_url = env_first(
        "COMMERCIAL_PROFILE_AUX_FALLBACK_URL",
        "COMMERCIAL_AUX_URL",
        "TGJU_CALL_URL",
        default="https://call.tgju.org/ajax.json",
    )
    aux_config = SourceConfig(
        name="commercial_aux",
        url=fallback_url,
        auth_mode="public_json",
        secret_fields=(),
        benchmark_families=("official",),
        default_unit="rial",
    )
    aux_sample = fetch_one(aux_config, sampled_at, window_start_dt, window_end_dt)
    official_value = parse_number(aux_sample.benchmark_values.get("official"))
    if official_value is None:
        return None

    quote_time = sample_benchmark_quote_time(aux_sample, "official")
    fallback_health = dict(aux_sample.health or {})
    fallback_health.update(
        {
            "collector": "http_json_fallback",
            "fetch_mode": "profile_fallback_aux",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "profile_attempts": profile_attempts,
        }
    )

    benchmark_values = blank_benchmark_values()
    benchmark_values["official"] = official_value
    return Sample(
        source=config.name,
        sampled_at=sampled_at,
        value=official_value,
        benchmark_values=benchmark_values,
        quote_time=quote_time,
        ok=aux_sample.ok,
        stale=aux_sample.stale,
        error=aux_sample.error,
        health=fallback_health,
        source_unit=aux_sample.source_unit,
        normalized_unit=aux_sample.normalized_unit,
    )


def fetch_tgju_profile_official_with_fallback(
    config: SourceConfig,
    sampled_at: dt.datetime,
    window_start_dt: dt.datetime,
    window_end_dt: dt.datetime,
    profile_symbols: Tuple[str, ...],
) -> Sample:
    deduped_symbols = tuple(dict.fromkeys(sym for sym in profile_symbols if isinstance(sym, str) and sym.strip()))
    if not deduped_symbols:
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error="no profile symbol candidates configured",
            health={
                "collector": "http_html",
                "fetch_mode": "public_html",
                "fetch_success": False,
                "failure_reason": "no profile symbol candidates configured",
                "error_type": "config_error",
            },
            source_unit=config.default_unit,
            normalized_unit="rial",
        )

    attempts: List[Dict[str, Any]] = []
    stale_candidate: Optional[Sample] = None
    failure_candidate: Optional[Sample] = None

    for symbol in deduped_symbols:
        symbol_url = tgju_profile_url_for_symbol(config.url, symbol)
        symbol_config = SourceConfig(
            name=config.name,
            url=symbol_url,
            auth_mode=config.auth_mode,
            secret_fields=config.secret_fields,
            benchmark_families=config.benchmark_families,
            default_unit=config.default_unit,
        )
        sample = fetch_tgju_profile_official_public(
            symbol_config,
            sampled_at,
            window_start_dt,
            window_end_dt,
            profile_symbol=symbol,
        )
        health = sample.health if isinstance(sample.health, dict) else {}
        fetch_success = health.get("fetch_success")
        failure_reason = health.get("failure_reason") or sample.error
        official_value = parse_number(sample.benchmark_values.get("official"))
        official_quote_time = sample_benchmark_quote_time(sample, "official")
        official_fresh = official_quote_is_fresh(official_quote_time, sampled_at)
        attempts.append(
            {
                "symbol": symbol,
                "url": health.get("request_url") or redact_url_for_health(symbol_url),
                "fetch_success": fetch_success,
                "failure_reason": failure_reason,
                "official_value": official_value,
                "official_quote_time": iso_ts(official_quote_time) if official_quote_time else None,
                "official_fresh": official_fresh,
            }
        )

        if fetch_success is True and official_value is not None and official_fresh:
            if isinstance(sample.health, dict):
                sample.health["profile_attempts"] = attempts
                sample.health["profile_symbol_selected"] = symbol
            return sample

        if fetch_success is True and official_value is not None and stale_candidate is None:
            stale_candidate = sample
        if failure_candidate is None:
            failure_candidate = sample

    fallback_reason = "profile quote unavailable"
    if stale_candidate is not None:
        fallback_reason = "profile quote stale"
    fallback_sample = fetch_tgju_profile_official_fallback_aux(
        config=config,
        sampled_at=sampled_at,
        window_start_dt=window_start_dt,
        window_end_dt=window_end_dt,
        fallback_reason=fallback_reason,
        profile_attempts=attempts,
    )
    if fallback_sample is not None:
        return fallback_sample

    if stale_candidate is not None:
        if isinstance(stale_candidate.health, dict):
            stale_candidate.health["profile_attempts"] = attempts
        return stale_candidate
    if failure_candidate is not None:
        if isinstance(failure_candidate.health, dict):
            failure_candidate.health["profile_attempts"] = attempts
        return failure_candidate
    return Sample(
        config.name,
        sampled_at,
        None,
        blank_benchmark_values(),
        None,
        ok=False,
        stale=False,
        error="profile quote unavailable",
        health={
            "collector": "http_html",
            "fetch_mode": "public_html",
            "fetch_success": False,
            "failure_reason": "profile quote unavailable",
            "error_type": "unavailable",
            "profile_attempts": attempts,
        },
        source_unit=config.default_unit,
        normalized_unit="rial",
    )


def fetch_alanchand_street_public(
    config: SourceConfig, sampled_at: dt.datetime, window_start_dt: dt.datetime, window_end_dt: dt.datetime
) -> Sample:
    health: Dict[str, Any] = {
        "collector": "http_html",
        "fetch_mode": "public_html",
        "scrape_timestamp": iso_ts(sampled_at),
        "fetch_success": False,
        "failure_reason": None,
        "error_type": None,
        "selector_used": None,
    }
    benchmark_values = blank_benchmark_values()
    normalized_unit = "rial"
    source_unit = config.default_unit

    try:
        req = build_request(config)
        health["request_url"] = redact_url_for_health(req.full_url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            final_url = resp.geturl()
        health["content_length"] = len(body.encode("utf-8"))
        health["final_url"] = final_url
        health["page_load_ok"] = True
    except urllib.error.HTTPError as exc:
        health["error_type"] = "http_error"
        health["failure_reason"] = f"http {exc.code}"
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            None,
            ok=False,
            stale=False,
            error=f"http {exc.code}",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except urllib.error.URLError as exc:
        health["error_type"] = "network_error"
        health["failure_reason"] = f"network: {exc.reason}"
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            None,
            ok=False,
            stale=False,
            error=f"network: {exc.reason}",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except TimeoutError:
        health["error_type"] = "timeout"
        health["failure_reason"] = "timeout"
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            None,
            ok=False,
            stale=False,
            error="timeout",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    # The USD public page publishes IRR values; page text includes FAQ mentions of toman
    # that can confuse generic unit detection, so keep a canonical IRR unit here.
    source_unit = config.default_unit
    health["unit_assumption"] = "page quotes interpreted as rial"
    quote_time = parse_alanchand_last_update(body, sampled_at)

    sell_raw = extract_number_with_patterns(body, ALANCHAND_STREET_SELL_PATTERNS, 100_000, 4_000_000)
    buy_raw = extract_number_with_patterns(body, ALANCHAND_STREET_BUY_PATTERNS, 100_000, 4_000_000)
    selector_used: Dict[str, Optional[str]] = {"sell": None, "buy": None}
    if sell_raw is not None:
        selector_used["sell"] = "sell_rate_pattern"
    if buy_raw is not None:
        selector_used["buy"] = "buy_rate_pattern"

    if sell_raw is None:
        sell_raw = extract_alanchand_jsonld_price(body)
        if sell_raw is not None:
            selector_used["sell"] = "jsonld_offers_price"

    sell_rial = normalize_unit(sell_raw, source_unit)
    buy_rial = normalize_unit(buy_raw, source_unit)
    mid_rial = ((buy_rial + sell_rial) / 2.0) if buy_rial is not None and sell_rial is not None else None

    health["selector_used"] = selector_used
    health["source_unit"] = source_unit
    health["normalized_unit"] = normalized_unit
    health["raw_extracted_values"] = {"open_market_sell": sell_raw, "open_market_buy": buy_raw}
    health["extracted_values"] = {"open_market_sell": sell_rial, "open_market_buy": buy_rial, "open_market_mid": mid_rial}
    health["quote_time"] = iso_ts(quote_time) if quote_time is not None else None

    validation: Dict[str, Any] = {"ok": True, "reason": None}
    if sell_rial is None:
        validation = {"ok": False, "reason": "unable to parse street USD sell quote"}
    elif buy_rial is not None:
        if buy_rial > sell_rial:
            validation = {"ok": False, "reason": "buy quote above sell quote"}
        else:
            spread_pct = (sell_rial - buy_rial) / sell_rial if sell_rial > 0 else None
            validation["spread_pct"] = spread_pct
            validation["max_spread_pct"] = env_pct(
                "ALANCHAND_STREET_MAX_SPREAD_PCT", ALANCHAND_STREET_MAX_SPREAD_PCT_DEFAULT
            )
            if spread_pct is None or spread_pct <= 0:
                validation = {"ok": False, "reason": "non-positive bid/ask spread"}
            elif spread_pct > float(validation["max_spread_pct"]):
                validation = {"ok": False, "reason": "bid/ask spread exceeds configured maximum", **validation}
    health["validation_result"] = validation

    benchmark_values["open_market"] = sell_rial
    stale = bool(
        quote_time is not None and not is_within_window_minute(quote_time, window_start_dt, window_end_dt)
    )
    if sell_rial is None or not validation.get("ok"):
        reason = str(validation.get("reason") or "unable to parse street USD quote")
        health["error_type"] = "validation_failed"
        health["failure_reason"] = reason
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=stale,
            error=reason,
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    health["fetch_success"] = True
    return Sample(
        config.name,
        sampled_at,
        sell_rial,
        benchmark_values,
        quote_time,
        ok=not stale,
        stale=stale,
        error="stale quote" if stale else None,
        health=health,
        source_unit=source_unit,
        normalized_unit=normalized_unit,
    )


def fetch_alanchand_public_single_rate(
    page_url: str,
    sampled_at: dt.datetime,
    minimum: float,
    maximum: float,
) -> Dict[str, Any]:
    health: Dict[str, Any] = {
        "collector": "http_html",
        "fetch_mode": "public_html",
        "request_url": None,
        "final_url": None,
        "content_length": None,
        "selector_used": None,
        "fetch_success": False,
        "failure_reason": None,
        "error_type": None,
    }
    source_unit = "rial"
    normalized_unit = "rial"

    body = ""
    try:
        req = urllib.request.Request(
            url=page_url,
            method="GET",
            headers={
                "User-Agent": "rialwatch-pipeline/0.2",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        health["request_url"] = redact_url_for_health(req.full_url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            health["final_url"] = resp.geturl()
        health["content_length"] = len(body.encode("utf-8"))
    except urllib.error.HTTPError as exc:
        health["error_type"] = "http_error"
        health["failure_reason"] = f"http {exc.code}"
        return {
            "value": None,
            "raw_value": None,
            "quote_time": None,
            "source_unit": source_unit,
            "normalized_unit": normalized_unit,
            "health": health,
        }
    except urllib.error.URLError as exc:
        health["error_type"] = "network_error"
        health["failure_reason"] = f"network: {exc.reason}"
        return {
            "value": None,
            "raw_value": None,
            "quote_time": None,
            "source_unit": source_unit,
            "normalized_unit": normalized_unit,
            "health": health,
        }
    except TimeoutError:
        health["error_type"] = "timeout"
        health["failure_reason"] = "timeout"
        return {
            "value": None,
            "raw_value": None,
            "quote_time": None,
            "source_unit": source_unit,
            "normalized_unit": normalized_unit,
            "health": health,
        }

    raw_value: Optional[float] = None
    selector_used: Optional[str] = None
    jsonld_value = bounded_value(extract_alanchand_jsonld_price(body), minimum, maximum)
    if jsonld_value is not None:
        raw_value = jsonld_value
        selector_used = "jsonld_offers_price"
    if raw_value is None:
        pattern_value = extract_number_with_patterns(body, ALANCHAND_PUBLIC_SINGLE_RATE_PATTERNS, minimum, maximum)
        if pattern_value is not None:
            raw_value = pattern_value
            selector_used = "single_rate_pattern"

    value = normalize_unit(raw_value, source_unit)
    quote_time = parse_alanchand_last_update(body, sampled_at)

    health["selector_used"] = selector_used
    health["source_unit"] = source_unit
    health["normalized_unit"] = normalized_unit
    health["raw_extracted_values"] = {"value": raw_value}
    health["extracted_values"] = {"value": value}
    health["quote_time"] = iso_ts(quote_time) if quote_time is not None else None
    if value is None:
        health["error_type"] = "selector_parse_failed"
        health["failure_reason"] = "unable to parse public quote"
    else:
        health["fetch_success"] = True

    return {
        "value": value,
        "raw_value": raw_value,
        "quote_time": quote_time,
        "source_unit": source_unit,
        "normalized_unit": normalized_unit,
        "health": health,
    }


def fetch_alanchand_companion_fallback(
    sampled_at: dt.datetime,
    window_start_dt: dt.datetime,
    window_end_dt: dt.datetime,
    primary_error: str,
    primary_error_type: str,
) -> Optional[Sample]:
    fallback_values = blank_benchmark_values()
    quote_times: Dict[str, dt.datetime] = {}
    raw_extracted_values: Dict[str, Optional[float]] = {}
    extracted_values: Dict[str, Optional[float]] = {}
    endpoint_health: Dict[str, Any] = {}
    fallback_sources: List[str] = []

    public_endpoints: Tuple[Tuple[str, str, float, float], ...] = (
        (
            "regional_transfer",
            env_or_default("ALANCHAND_REGIONAL_PUBLIC_URL", ALANCHAND_PUBLIC_REGIONAL_URL_DEFAULT),
            100_000,
            3_000_000,
        ),
        (
            "crypto_usdt",
            env_or_default("ALANCHAND_USDT_PUBLIC_URL", ALANCHAND_PUBLIC_USDT_URL_DEFAULT),
            100_000,
            3_000_000,
        ),
    )

    for benchmark_key, endpoint_url, minimum, maximum in public_endpoints:
        page_result = fetch_alanchand_public_single_rate(
            page_url=endpoint_url,
            sampled_at=sampled_at,
            minimum=minimum,
            maximum=maximum,
        )
        endpoint_health[benchmark_key] = page_result.get("health", {})
        value = parse_number(page_result.get("value"))
        if value is None:
            continue
        fallback_values[benchmark_key] = value
        raw_extracted_values[benchmark_key] = parse_number(page_result.get("raw_value"))
        extracted_values[benchmark_key] = value
        quote_time = page_result.get("quote_time")
        if isinstance(quote_time, dt.datetime):
            quote_times[benchmark_key] = quote_time

    if any(fallback_values.get(key) is not None for key in ("regional_transfer", "crypto_usdt")):
        fallback_sources.append("alanchand_public_pages")

    missing_benchmarks = [key for key in ("regional_transfer", "crypto_usdt") if fallback_values.get(key) is None]
    if missing_benchmarks:
        navasan_config = SourceConfig(
            name="navasan",
            url=env_or_default("NAVASAN_API_URL", "https://api.navasan.tech/latest/"),
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
            benchmark_families=tuple(missing_benchmarks),
            default_unit="toman",
        )
        navasan_sample = fetch_one(navasan_config, sampled_at, window_start_dt, window_end_dt)
        navasan_health = navasan_sample.health if isinstance(navasan_sample.health, dict) else {}
        navasan_raw = navasan_health.get("raw_extracted_values") if isinstance(navasan_health, dict) else {}
        navasan_quote_times = navasan_health.get("benchmark_quote_times") if isinstance(navasan_health, dict) else {}

        for benchmark_key in missing_benchmarks:
            value = parse_number(navasan_sample.benchmark_values.get(benchmark_key))
            if value is None:
                continue
            fallback_values[benchmark_key] = value
            extracted_values[benchmark_key] = value
            if isinstance(navasan_raw, dict):
                raw_extracted_values[benchmark_key] = parse_number(navasan_raw.get(benchmark_key))
            if isinstance(navasan_quote_times, dict):
                qt = try_parse_datetime(navasan_quote_times.get(benchmark_key))
                if qt is not None:
                    quote_times[benchmark_key] = qt
            if benchmark_key not in quote_times:
                qt = sample_benchmark_quote_time(navasan_sample, benchmark_key)
                if qt is not None:
                    quote_times[benchmark_key] = qt

        endpoint_health["navasan"] = {
            "fetch_success": navasan_health.get("fetch_success"),
            "failure_reason": navasan_health.get("failure_reason"),
            "validation_result": navasan_health.get("validation_result"),
            "request_url": navasan_health.get("request_url"),
            "final_url": navasan_health.get("final_url"),
        }
        if any(fallback_values.get(key) is not None for key in missing_benchmarks):
            fallback_sources.append("navasan")

    usable_values = {k: v for k, v in fallback_values.items() if parse_number(v) is not None}
    if not usable_values:
        return None

    latest_quote_time = max(quote_times.values()) if quote_times else None
    stale = not is_within_window_minute(sampled_at, window_start_dt, window_end_dt)
    health: Dict[str, Any] = {
        "collector": "http_json_fallback",
        "fetch_mode": "companion_fallback",
        "fetch_success": True,
        "failure_reason": None,
        "error_type": None,
        "fallback_used": True,
        "fallback_reason": primary_error,
        "fallback_error_type": primary_error_type,
        "fallback_sources": fallback_sources,
        "raw_extracted_values": raw_extracted_values,
        "extracted_values": extracted_values,
        "benchmark_quote_times": {k: iso_ts(v) for k, v in quote_times.items()},
        "endpoint_health": endpoint_health,
        "source_unit": "rial",
        "normalized_unit": "rial",
        "validation_result": {"ok": True},
    }

    return Sample(
        source="alanchand",
        sampled_at=sampled_at,
        value=None,
        benchmark_values=fallback_values,
        quote_time=latest_quote_time,
        ok=not stale,
        stale=stale,
        error="sample outside observation window" if stale else None,
        health=health,
        source_unit="rial",
        normalized_unit="rial",
    )


def fetch_navasan_companion_fallback(
    sampled_at: dt.datetime,
    window_start_dt: dt.datetime,
    window_end_dt: dt.datetime,
    primary_error: str,
    primary_error_type: str,
) -> Optional[Sample]:
    fallback_values = blank_benchmark_values()
    raw_extracted_values = blank_benchmark_values()
    quote_times: Dict[str, dt.datetime] = {}
    endpoint_health: Dict[str, Any] = {}
    fallback_sources: List[str] = []

    # Official benchmark fallback from resilient TGJU auxiliary endpoint pool.
    aux_config = SourceConfig(
        name="commercial_aux",
        url=env_first("COMMERCIAL_AUX_URL", "TGJU_CALL_URL", default="https://call.tgju.org/ajax.json"),
        auth_mode="public_json",
        secret_fields=(),
        benchmark_families=("official",),
        default_unit="rial",
    )
    aux_sample = fetch_one(aux_config, sampled_at, window_start_dt, window_end_dt)
    aux_health = aux_sample.health if isinstance(aux_sample.health, dict) else {}
    endpoint_health["commercial_aux"] = {
        "fetch_success": aux_health.get("fetch_success"),
        "failure_reason": aux_health.get("failure_reason"),
        "validation_result": aux_health.get("validation_result"),
        "request_url": aux_health.get("request_url"),
        "final_url": aux_health.get("final_url"),
    }
    official_value = parse_number(aux_sample.benchmark_values.get("official"))
    if official_value is not None:
        fallback_values["official"] = official_value
        raw_official = None
        if isinstance(aux_health.get("raw_extracted_values"), dict):
            raw_official = parse_number(aux_health.get("raw_extracted_values", {}).get("official"))
        raw_extracted_values["official"] = raw_official
        qt = sample_benchmark_quote_time(aux_sample, "official")
        if qt is not None:
            quote_times["official"] = qt
        fallback_sources.append("commercial_aux")

    for benchmark_key, page_url, minimum, maximum in (
        (
            "regional_transfer",
            env_or_default("ALANCHAND_REGIONAL_PUBLIC_URL", ALANCHAND_PUBLIC_REGIONAL_URL_DEFAULT),
            100_000,
            3_000_000,
        ),
        (
            "crypto_usdt",
            env_or_default("ALANCHAND_USDT_PUBLIC_URL", ALANCHAND_PUBLIC_USDT_URL_DEFAULT),
            100_000,
            3_000_000,
        ),
    ):
        page_result = fetch_alanchand_public_single_rate(
            page_url=page_url,
            sampled_at=sampled_at,
            minimum=minimum,
            maximum=maximum,
        )
        endpoint_health[benchmark_key] = page_result.get("health", {})
        value = parse_number(page_result.get("value"))
        raw_value = parse_number(page_result.get("raw_value"))
        if value is not None:
            fallback_values[benchmark_key] = value
            raw_extracted_values[benchmark_key] = raw_value
            qt = page_result.get("quote_time")
            if isinstance(qt, dt.datetime):
                quote_times[benchmark_key] = qt
            if "alanchand_public_pages" not in fallback_sources:
                fallback_sources.append("alanchand_public_pages")

    usable_values = {k: v for k, v in fallback_values.items() if parse_number(v) is not None}
    if not usable_values:
        return None

    latest_quote_time = max(quote_times.values()) if quote_times else None
    stale = not is_within_window_minute(sampled_at, window_start_dt, window_end_dt)
    health: Dict[str, Any] = {
        "collector": "http_json_fallback",
        "fetch_mode": "companion_fallback",
        "fetch_success": True,
        "failure_reason": None,
        "error_type": None,
        "fallback_used": True,
        "fallback_reason": primary_error,
        "fallback_error_type": primary_error_type,
        "fallback_sources": fallback_sources,
        "raw_extracted_values": raw_extracted_values,
        "extracted_values": fallback_values,
        "benchmark_quote_times": {k: iso_ts(v) for k, v in quote_times.items()},
        "endpoint_health": endpoint_health,
        "source_unit": "rial",
        "normalized_unit": "rial",
        "validation_result": {"ok": True},
    }

    return Sample(
        source="navasan",
        sampled_at=sampled_at,
        value=None,
        benchmark_values=fallback_values,
        quote_time=latest_quote_time,
        ok=not stale,
        stale=stale,
        error="sample outside observation window" if stale else None,
        health=health,
        source_unit="rial",
        normalized_unit="rial",
    )


def fetch_one(config: SourceConfig, sampled_at: dt.datetime, window_start_dt: dt.datetime, window_end_dt: dt.datetime) -> Sample:
    if config.auth_mode == "browser_playwright":
        return fetch_bonbast_browser(config, sampled_at, window_start_dt, window_end_dt)
    if config.auth_mode == "public_html":
        if config.name == "alanchand_street":
            return fetch_alanchand_street_public(config, sampled_at, window_start_dt, window_end_dt)
        profile_symbols = COMMERCIAL_PROFILE_OFFICIAL_SYMBOLS.get(config.name, ())
        if profile_symbols:
            return fetch_tgju_profile_official_with_fallback(
                config,
                sampled_at,
                window_start_dt,
                window_end_dt,
                profile_symbols=profile_symbols,
            )
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error="unsupported public_html source",
            health={
                "collector": "http_html",
                "fetch_mode": "public_html",
                "fetch_success": False,
                "failure_reason": "unsupported public_html source",
                "error_type": "unsupported_source",
            },
            source_unit=config.default_unit,
            normalized_unit="rial",
        )
    if config.name == "bonbast":
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error="bonbast requires browser_playwright mode",
            health={
                "collector": "http_json",
                "fetch_mode": "http",
                "fetch_success": False,
                "failure_reason": "bonbast requires browser_playwright mode",
                "error_type": "invalid_collector_mode",
            },
            source_unit=config.default_unit,
            normalized_unit="rial",
        )

    health: Dict[str, Any] = {
        "collector": "http_json",
        "scrape_timestamp": iso_ts(sampled_at),
        "page_load_ok": None,
        "selector_success": None,
        "error_type": None,
        "request_url": None,
    }
    source_unit = config.default_unit
    normalized_unit = "rial"
    is_aux_source = is_commercial_aux_source(config.name)
    is_api_source = config.auth_mode in {"query_api_key", "header_api_key"}
    request_timeout_seconds = (
        env_seconds(
            "COMMERCIAL_AUX_TIMEOUT_SECONDS",
            COMMERCIAL_AUX_TIMEOUT_SECONDS_DEFAULT,
            minimum=5.0,
            maximum=60.0,
        )
        if is_aux_source
        else (
            env_seconds(
                "API_SOURCE_TIMEOUT_SECONDS",
                20.0,
                minimum=5.0,
                maximum=60.0,
            )
            if is_api_source
            else 20.0
        )
    )
    max_attempts = (
        env_int(
            "COMMERCIAL_AUX_RETRY_ATTEMPTS",
            COMMERCIAL_AUX_RETRY_ATTEMPTS_DEFAULT,
            minimum=1,
            maximum=6,
        )
        if is_aux_source
        else (
            env_int(
                "API_SOURCE_RETRY_ATTEMPTS",
                API_SOURCE_RETRY_ATTEMPTS_DEFAULT,
                minimum=1,
                maximum=6,
            )
            if is_api_source
            else 1
        )
    )
    retry_backoff_seconds = (
        env_seconds(
            "COMMERCIAL_AUX_RETRY_BACKOFF_SECONDS",
            COMMERCIAL_AUX_RETRY_BACKOFF_SECONDS_DEFAULT,
            minimum=0.0,
            maximum=10.0,
        )
        if is_aux_source
        else (
            env_seconds(
                "API_SOURCE_RETRY_BACKOFF_SECONDS",
                API_SOURCE_RETRY_BACKOFF_SECONDS_DEFAULT,
                minimum=0.0,
                maximum=10.0,
            )
            if is_api_source
            else 0.0
        )
    )
    try:
        body = ""
        final_url = ""
        request_urls: List[str] = []
        health["timeout_seconds"] = request_timeout_seconds
        if is_aux_source:
            health["fetch_mode"] = "hedged_pool"
            hedge_width = env_int(
                "COMMERCIAL_AUX_HEDGE_WIDTH",
                COMMERCIAL_AUX_HEDGE_WIDTH_DEFAULT,
                minimum=1,
                maximum=4,
            )
            last_error: Optional[BaseException] = None
            for attempt_idx in range(max_attempts):
                ranked_endpoints = ranked_commercial_aux_endpoints(config.url)
                selected_endpoints = ranked_endpoints[: max(1, min(hedge_width, len(ranked_endpoints)))]
                health["attempt_count"] = attempt_idx + 1
                health["retry_count"] = attempt_idx
                health["hedge_width"] = len(selected_endpoints)
                health["hedged_endpoints"] = selected_endpoints
                health["request_urls"] = request_urls

                futures: Dict[Any, str] = {}
                round_success = False
                executor = ThreadPoolExecutor(max_workers=max(1, len(selected_endpoints)))
                try:
                    for endpoint_url in selected_endpoints:
                        endpoint_cfg = SourceConfig(
                            name=config.name,
                            url=endpoint_url,
                            auth_mode=config.auth_mode,
                            secret_fields=config.secret_fields,
                            benchmark_families=config.benchmark_families,
                            default_unit=config.default_unit,
                        )
                        req = build_request(endpoint_cfg)
                        redacted_url = redact_url_for_health(req.full_url)
                        request_urls.append(redacted_url)
                        if health.get("request_url") is None:
                            health["request_url"] = redacted_url
                        futures[executor.submit(fetch_request_body, req, request_timeout_seconds)] = endpoint_url

                    for future in as_completed(futures):
                        endpoint_url = futures[future]
                        try:
                            body_candidate, final_url_candidate = future.result()
                        except Exception as exc:
                            last_error = exc
                            record_commercial_aux_host_result(endpoint_url, success=False)
                            continue

                        body = body_candidate
                        final_url = final_url_candidate
                        health["final_endpoint"] = canonical_commercial_aux_url(endpoint_url)
                        record_commercial_aux_host_result(endpoint_url, success=True)
                        round_success = True
                        break
                finally:
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)

                if round_success:
                    break

                should_retry = (
                    (attempt_idx + 1) < max_attempts
                    and last_error is not None
                    and is_commercial_aux_retryable_error(last_error)
                )
                if not should_retry:
                    if last_error is not None:
                        raise last_error
                    raise urllib.error.URLError("no successful auxiliary endpoint response")
                sleep_seconds = min(retry_backoff_seconds * (2**attempt_idx), 8.0)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            if not body:
                if last_error is not None:
                    raise last_error
                raise urllib.error.URLError("no successful auxiliary endpoint response")
        else:
            for attempt_idx in range(max_attempts):
                req = build_request(config)
                redacted_url = redact_url_for_health(req.full_url)
                request_urls.append(redacted_url)
                if health.get("request_url") is None:
                    health["request_url"] = redacted_url
                health["request_urls"] = request_urls
                health["attempt_count"] = attempt_idx + 1
                health["retry_count"] = attempt_idx

                try:
                    body, final_url = fetch_request_body(req, request_timeout_seconds)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                    should_retry = (
                        is_api_source
                        and (attempt_idx + 1) < max_attempts
                        and is_api_source_retryable_error(exc)
                    )
                    if not should_retry:
                        raise
                    sleep_seconds = min(retry_backoff_seconds * (2**attempt_idx), 8.0)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
        health["request_urls"] = request_urls

        health["content_length"] = len(body.encode("utf-8"))
        health["final_url"] = final_url
        payload, payload_mode = parse_source_payload(config.name, body)
        health["payload_mode"] = payload_mode
        health["page_load_ok"] = True
    except KeyError as exc:
        health["error_type"] = "missing_secret"
        health["fetch_success"] = False
        health["failure_reason"] = f"missing secret: {exc}"
        if config.name == "alanchand":
            fallback = fetch_alanchand_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error=f"missing secret: {exc}",
                primary_error_type="missing_secret",
            )
            if fallback is not None:
                return fallback
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error=f"missing secret: {exc}",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except urllib.error.HTTPError as exc:
        health["error_type"] = "http_error"
        health["fetch_success"] = False
        health["failure_reason"] = f"http {exc.code}"
        if config.name == "alanchand":
            fallback = fetch_alanchand_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error=f"http {exc.code}",
                primary_error_type="http_error",
            )
            if fallback is not None:
                return fallback
        if config.name == "navasan":
            fallback = fetch_navasan_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error=f"http {exc.code}",
                primary_error_type="http_error",
            )
            if fallback is not None:
                return fallback
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error=f"http {exc.code}",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except urllib.error.URLError as exc:
        health["error_type"] = "network_error"
        health["fetch_success"] = False
        health["failure_reason"] = f"network: {exc.reason}"
        if config.name == "alanchand":
            fallback = fetch_alanchand_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error=f"network: {exc.reason}",
                primary_error_type="network_error",
            )
            if fallback is not None:
                return fallback
        if config.name == "navasan":
            fallback = fetch_navasan_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error=f"network: {exc.reason}",
                primary_error_type="network_error",
            )
            if fallback is not None:
                return fallback
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error=f"network: {exc.reason}",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except TimeoutError:
        health["error_type"] = "timeout"
        health["fetch_success"] = False
        health["failure_reason"] = "timeout"
        if config.name == "alanchand":
            fallback = fetch_alanchand_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error="timeout",
                primary_error_type="timeout",
            )
            if fallback is not None:
                return fallback
        if config.name == "navasan":
            fallback = fetch_navasan_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error="timeout",
                primary_error_type="timeout",
            )
            if fallback is not None:
                return fallback
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error="timeout",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )
    except json.JSONDecodeError:
        health["error_type"] = "invalid_json"
        health["fetch_success"] = False
        health["failure_reason"] = "invalid json"
        if config.name == "alanchand":
            fallback = fetch_alanchand_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error="invalid json",
                primary_error_type="invalid_json",
            )
            if fallback is not None:
                return fallback
        if config.name == "navasan":
            fallback = fetch_navasan_companion_fallback(
                sampled_at=sampled_at,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                primary_error="invalid json",
                primary_error_type="invalid_json",
            )
            if fallback is not None:
                return fallback
        return Sample(
            config.name,
            sampled_at,
            None,
            blank_benchmark_values(),
            None,
            ok=False,
            stale=False,
            error="invalid json",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    raw_extracted_values, selected_symbols = extract_benchmark_values_with_metadata(payload, config.name)
    benchmark_units: Dict[str, str] = {}
    canonical_map = CANONICAL_SOURCE_SYMBOLS.get(config.name, {})
    health["source_fields"] = {
        key: list(canonical_map.get(key, ()))
        for key in BENCHMARK_LABELS
        if canonical_map.get(key)
    }
    if selected_symbols:
        health["selected_symbol_by_benchmark"] = selected_symbols
    benchmark_quote_times: Dict[str, str] = {}
    for benchmark_key, symbols in canonical_map.items():
        if not symbols:
            continue
        preferred_symbol = selected_symbols.get(benchmark_key)
        if preferred_symbol:
            benchmark_quote_time = extract_symbol_quote_time(payload, (preferred_symbol,))
            if benchmark_quote_time is None:
                benchmark_quote_time, _matched = extract_symbol_quote_time_by_priority(payload, symbols)
        else:
            benchmark_quote_time, _matched = extract_symbol_quote_time_by_priority(payload, symbols)
        if benchmark_quote_time is not None:
            benchmark_quote_times[benchmark_key] = iso_ts(benchmark_quote_time)
    if benchmark_quote_times:
        health["benchmark_quote_times"] = benchmark_quote_times
    if config.name == "navasan":
        benchmark_units = {key: NAVASAN_BENCHMARK_UNITS.get(key, config.default_unit) for key in BENCHMARK_LABELS}
        benchmark_values = normalize_benchmark_values_with_units(
            raw_extracted_values, benchmark_units=benchmark_units, default_unit=config.default_unit
        )
        source_unit = "mixed"
    else:
        source_unit = detect_source_unit(payload, config.default_unit)
        benchmark_values = normalize_benchmark_values(raw_extracted_values, source_unit)
        benchmark_units = {key: source_unit for key in BENCHMARK_LABELS}

    value = benchmark_values.get(PRIMARY_BENCHMARK)
    quote_time = extract_quote_time(payload)
    if config.name == "navasan":
        open_market_symbols = CANONICAL_SOURCE_SYMBOLS.get("navasan", {}).get("open_market", ())
        open_market_quote_time = extract_symbol_quote_time(payload, open_market_symbols)
        if open_market_quote_time is not None:
            health["open_market_quote_time"] = iso_ts(open_market_quote_time)
            open_market_stale = not is_within_window_minute(open_market_quote_time, window_start_dt, window_end_dt)
            health["open_market_stale"] = open_market_stale
            if open_market_stale:
                # Keep companion market layers, but exclude stale street quote from primary benchmark.
                benchmark_values[PRIMARY_BENCHMARK] = None
                value = None
                health["open_market_exclusion_reason"] = "stale open_market quote timestamp"
        else:
            health["open_market_quote_time"] = None
            health["open_market_stale"] = None

    health["extracted_values"] = {k: benchmark_values.get(k) for k in benchmark_values}
    health["raw_extracted_values"] = {k: raw_extracted_values.get(k) for k in raw_extracted_values}
    health["benchmark_units"] = benchmark_units
    health["source_unit"] = source_unit
    health["normalized_unit"] = normalized_unit
    health["fetch_success"] = True

    stale = bool(quote_time is not None and not is_within_window_minute(quote_time, window_start_dt, window_end_dt))

    if value is None:
        return Sample(
            config.name,
            sampled_at,
            None,
            benchmark_values,
            quote_time,
            ok=False,
            stale=stale,
            error="unable to parse USD/IRR",
            health=health,
            source_unit=source_unit,
            normalized_unit=normalized_unit,
        )

    return Sample(
        config.name,
        sampled_at,
        value,
        benchmark_values,
        quote_time,
        ok=not stale,
        stale=stale,
        error="stale quote" if stale else None,
        health=health,
        source_unit=source_unit,
        normalized_unit=normalized_unit,
    )


def collect_samples(
    source_configs: List[SourceConfig],
    day: dt.date,
    sample_times: Tuple[dt.time, ...],
    skip_waits: bool,
    allow_outside_window: bool,
) -> Dict[str, List[Sample]]:
    samples: Dict[str, List[Sample]] = {cfg.name: [] for cfg in source_configs}
    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    window_end_dt = dt.datetime.combine(day, WINDOW_END, tzinfo=UTC)

    for sample_time in sample_times:
        target = dt.datetime.combine(day, sample_time, tzinfo=UTC)
        if not allow_outside_window:
            now = utc_now()
            if now > window_end_dt + dt.timedelta(minutes=5):
                break

        should_sleep_until(target, skip_waits=skip_waits)
        sampled_at = utc_now()

        for cfg in source_configs:
            sample = fetch_one(cfg, sampled_at, window_start_dt, window_end_dt)
            if not allow_outside_window and not is_within_window_minute(sampled_at, window_start_dt, window_end_dt):
                sample.ok = False
                sample.stale = True
                sample.error = "sample outside observation window"
            samples[cfg.name].append(sample)

    apply_bonbast_peer_validation(
        samples=samples,
        max_deviation_pct=env_pct("BONBAST_PEER_DEVIATION_PCT", BONBAST_PEER_DEVIATION_PCT_DEFAULT),
    )
    return samples


def apply_bonbast_peer_validation(samples: Dict[str, List[Sample]], max_deviation_pct: float) -> None:
    bonbast_entries = samples.get("bonbast")
    if not bonbast_entries:
        return

    for idx, bonbast_sample in enumerate(bonbast_entries):
        if not isinstance(bonbast_sample.health, dict):
            bonbast_sample.health = {}
        health = bonbast_sample.health
        validation = health.get("validation_result")
        if not isinstance(validation, dict):
            validation = {}

        if bonbast_sample.value is None or not bonbast_sample.ok or bonbast_sample.stale:
            validation["peer_band_result"] = "skipped_sample_not_valid"
            health["validation_result"] = validation
            continue

        peer_values: List[float] = []
        for source, entries in samples.items():
            if source == "bonbast" or idx >= len(entries):
                continue
            if source not in PRIMARY_STREET_SOURCE_UNIVERSE:
                continue
            peer_sample = entries[idx]
            if not peer_sample.ok or peer_sample.stale:
                continue
            peer_value = parse_number(peer_sample.benchmark_values.get(PRIMARY_BENCHMARK))
            if peer_value is not None:
                peer_values.append(peer_value)

        if not peer_values:
            validation["peer_band_result"] = "skipped_no_peer_values"
            health["validation_result"] = validation
            continue

        peer_median = median(peer_values)
        if peer_median <= 0:
            validation["peer_band_result"] = "skipped_invalid_peer_median"
            health["validation_result"] = validation
            continue

        deviation_pct = abs(float(bonbast_sample.value) - peer_median) / peer_median
        validation["peer_median"] = peer_median
        validation["peer_deviation_pct"] = deviation_pct
        validation["peer_max_deviation_pct"] = max_deviation_pct
        if deviation_pct > max_deviation_pct:
            validation["peer_band_result"] = "failed"
            bonbast_sample.ok = False
            bonbast_sample.error = "outside peer plausibility band"
            health["failure_reason"] = "outside peer plausibility band"
        else:
            validation["peer_band_result"] = "passed"
        health["validation_result"] = validation


def sample_benchmark_quote_time(sample: Sample, benchmark_key: str) -> Optional[dt.datetime]:
    if isinstance(sample.health, dict):
        bqt = sample.health.get("benchmark_quote_times")
        if isinstance(bqt, dict):
            parsed = try_parse_datetime(bqt.get(benchmark_key))
            if parsed is not None:
                return parsed
    if sample.quote_time is not None:
        return sample.quote_time
    return None


def latest_sample_value_for_benchmark(
    entries: List[Sample], benchmark_key: str, allow_stale_official: bool = False
) -> Optional[Tuple[float, Optional[dt.datetime], dt.datetime, str]]:
    ranked: List[Tuple[dt.datetime, dt.datetime, float, str]] = []
    for s in entries:
        value = parse_number(s.benchmark_values.get(benchmark_key))
        if value is None:
            continue
        if isinstance(s.health, dict):
            fetch_success = s.health.get("fetch_success")
            if fetch_success is False:
                continue
            validation = s.health.get("validation_result")
            if isinstance(validation, dict) and validation.get("ok") is False:
                continue
        quote_time = sample_benchmark_quote_time(s, benchmark_key)
        if benchmark_key == "official" and not allow_stale_official and not official_quote_is_fresh(quote_time, s.sampled_at):
            continue
        effective_time = quote_time if quote_time is not None else s.sampled_at
        ranked.append((effective_time, s.sampled_at, value, s.source_unit or "unknown"))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    effective_time, sampled_at, value, source_unit = ranked[0]
    quote_time = effective_time if effective_time != sampled_at else None
    return value, quote_time, sampled_at, source_unit


def benchmark_update_cadence_count(entries: List[Sample], benchmark_key: str, allow_stale_official: bool = False) -> int:
    quote_times: set[str] = set()
    has_valid_sample = False
    for s in entries:
        value = parse_number(s.benchmark_values.get(benchmark_key))
        if value is None:
            continue
        if isinstance(s.health, dict):
            fetch_success = s.health.get("fetch_success")
            if fetch_success is False:
                continue
            validation = s.health.get("validation_result")
            if isinstance(validation, dict) and validation.get("ok") is False:
                continue
        quote_time = sample_benchmark_quote_time(s, benchmark_key)
        if benchmark_key == "official" and not allow_stale_official and not official_quote_is_fresh(quote_time, s.sampled_at):
            continue
        has_valid_sample = True
        if quote_time is not None:
            quote_times.add(iso_ts(quote_time))
    if quote_times:
        return len(quote_times)
    return 1 if has_valid_sample else 0


def compute_benchmark_result(
    samples: Dict[str, List[Sample]],
    benchmark_key: str,
    benchmark_sources: Dict[str, Tuple[str, ...]],
    primary_allow_stale: bool = False,
) -> Dict[str, Any]:
    source_medians: Dict[str, float] = {}
    source_notes: Dict[str, str] = {}
    source_units: Dict[str, str] = {}
    source_latest_quote_times: Dict[str, dt.datetime] = {}
    source_update_counts: Dict[str, int] = {}
    invalid_or_stale = False
    using_stale_fallback = False
    stale_fallback_sources: List[str] = []
    for source, entries in samples.items():
        if benchmark_key == PRIMARY_BENCHMARK and source not in PRIMARY_STREET_SOURCE_UNIVERSE:
            source_notes[source] = "source excluded from street benchmark universe"
            continue
        families = benchmark_sources.get(source, ())
        if benchmark_key not in families:
            source_notes[source] = "source family not used for this benchmark"
            continue

        if benchmark_key == "official":
            latest = latest_sample_value_for_benchmark(entries, benchmark_key)
            if latest is None:
                stale_latest = latest_sample_value_for_benchmark(
                    entries,
                    benchmark_key,
                    allow_stale_official=True,
                )
                if stale_latest is not None:
                    stale_value, stale_quote_time, stale_sampled_at, stale_source_unit = stale_latest
                    stale_effective_quote_time = stale_quote_time if stale_quote_time is not None else stale_sampled_at
                    if official_quote_within_stale_fallback(stale_effective_quote_time, stale_sampled_at):
                        source_medians[source] = stale_value
                        source_units[source] = stale_source_unit
                        source_latest_quote_times[source] = stale_effective_quote_time
                        source_update_counts[source] = benchmark_update_cadence_count(
                            entries,
                            benchmark_key,
                            allow_stale_official=True,
                        )
                        source_notes[source] = (
                            f"stale fallback quote from {iso_ts(stale_effective_quote_time)} "
                            "(outside freshness window)"
                        )
                        using_stale_fallback = True
                        stale_fallback_sources.append(source)
                        continue

                stale_quote_candidates: List[dt.datetime] = []
                for s in entries:
                    candidate_value = parse_number(s.benchmark_values.get(benchmark_key))
                    if candidate_value is None:
                        continue
                    if isinstance(s.health, dict):
                        fetch_success = s.health.get("fetch_success")
                        if fetch_success is False:
                            continue
                        validation = s.health.get("validation_result")
                        if isinstance(validation, dict) and validation.get("ok") is False:
                            continue
                    candidate_quote_time = sample_benchmark_quote_time(s, benchmark_key)
                    if candidate_quote_time is not None:
                        stale_quote_candidates.append(candidate_quote_time)
                if stale_quote_candidates:
                    latest_stale_quote = max(stale_quote_candidates)
                    source_notes[source] = (
                        f"official quote stale since {iso_ts(latest_stale_quote)} "
                        "(outside stale-fallback horizon)"
                    )
                else:
                    source_notes[source] = "no valid samples"
                continue
            latest_value, latest_quote_time, latest_sampled_at, latest_source_unit = latest
            source_medians[source] = latest_value
            source_units[source] = latest_source_unit
            effective_quote_time = latest_quote_time if latest_quote_time is not None else latest_sampled_at
            source_latest_quote_times[source] = effective_quote_time
            cadence_count = benchmark_update_cadence_count(
                entries,
                benchmark_key,
                allow_stale_official=False,
            )
            source_update_counts[source] = cadence_count
            source_notes[source] = f"latest valid quote selected at {iso_ts(effective_quote_time)}"
            continue

        if benchmark_key == PRIMARY_BENCHMARK:
            # Primary benchmark publication only considers valid in-window samples.
            if primary_allow_stale:
                values = []
                for s in entries:
                    candidate = s.benchmark_values.get(benchmark_key)
                    if candidate is None:
                        continue
                    if isinstance(s.health, dict):
                        fetch_success = s.health.get("fetch_success")
                        validation = s.health.get("validation_result")
                        validation_failed = isinstance(validation, dict) and validation.get("ok") is False
                        if fetch_success is not True or validation_failed:
                            continue
                    if isinstance(s.error, str) and s.error.strip():
                        normalized_error = s.error.strip().lower()
                        if normalized_error not in {"stale quote", "sample outside observation window"}:
                            continue
                    values.append(candidate)
            else:
                values = [
                    s.benchmark_values.get(benchmark_key)
                    for s in entries
                    if s.ok and not s.stale and s.benchmark_values.get(benchmark_key) is not None
                ]
        else:
            values = [s.benchmark_values.get(benchmark_key) for s in entries if s.benchmark_values.get(benchmark_key) is not None]

        if not values:
            if benchmark_key == PRIMARY_BENCHMARK:
                source_notes[source] = "no valid in-window samples"
            else:
                source_notes[source] = "no valid samples"
            continue

        source_medians[source] = median(values)
        units_seen = {
            (s.source_unit or "unknown")
            for s in entries
            if s.benchmark_values.get(benchmark_key) is not None
        }
        if len(units_seen) == 1:
            source_units[source] = next(iter(units_seen))
        elif len(units_seen) > 1:
            source_units[source] = "mixed"
        if benchmark_key not in {PRIMARY_BENCHMARK, "official"} and any(s.stale for s in entries):
            # In current-day refresh mode, allow companion benchmark values from stale samples.
            if not primary_allow_stale:
                invalid_or_stale = True

    medians = list(source_medians.values())
    reasons: List[str] = []
    withheld = False

    if len(medians) < 1:
        withheld = True
        reasons.append("no valid sources available")

    fix_value: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    dispersion: Optional[float] = None
    selected_sources: Optional[List[str]] = None
    selected_quote_time: Optional[str] = None
    selection_method: Optional[str] = None

    if benchmark_key == "official" and source_latest_quote_times:
        freshest_time = max(source_latest_quote_times.values())
        freshest_sources = [src for src, ts in source_latest_quote_times.items() if ts == freshest_time]
        selected_candidates = list(freshest_sources)
        best_update_count: Optional[int] = None
        if freshest_sources:
            best_update_count = max(source_update_counts.get(src, 0) for src in freshest_sources)
            if best_update_count > 0:
                cadence_winners = [
                    src for src in freshest_sources if source_update_counts.get(src, 0) == best_update_count
                ]
                if cadence_winners:
                    selected_candidates = cadence_winners
            selection_method = (
                "freshest_quote_time_then_update_cadence"
                if len(selected_candidates) < len(freshest_sources)
                else "freshest_quote_time"
            )
        freshest_values = [source_medians[src] for src in selected_candidates if src in source_medians]
        if freshest_values:
            medians = freshest_values
            selected_sources = sorted(selected_candidates)
            selected_quote_time = iso_ts(freshest_time)
            for source, ts in source_latest_quote_times.items():
                if source in selected_candidates:
                    continue
                if source in freshest_sources and best_update_count is not None:
                    source_notes[source] = (
                        f"same freshest quote ({iso_ts(ts)}) but lower update cadence "
                        f"({source_update_counts.get(source, 0)} vs {best_update_count})"
                    )
                else:
                    source_notes[source] = (
                        f"older quote ({iso_ts(ts)}) than freshest source ({iso_ts(freshest_time)})"
                    )

    if benchmark_key == "official":
        selected_source_set = set(selected_sources or source_medians.keys())
        using_stale_fallback = bool(selected_source_set.intersection(stale_fallback_sources))

    if not withheld:
        fix_value = median(medians)
        p25 = percentile(medians, 0.25)
        p75 = percentile(medians, 0.75)
        dispersion = (p75 - p25) / fix_value if fix_value else None
        if dispersion is not None and dispersion > 0.05:
            withheld = True
            reasons.append("dispersion > 5%")

    if invalid_or_stale:
        withheld = True
        reasons.append("invalid/stale inputs")

    status = "WITHHOLD"
    if not withheld and dispersion is not None:
        if dispersion <= 0.015:
            status = "Green"
        elif dispersion <= 0.035:
            status = "Amber"
        elif dispersion <= 0.05:
            status = "Red"

    return {
        "label": BENCHMARK_LABELS[benchmark_key],
        "benchmark": benchmark_key,
        "fix": fix_value,
        "band": {"p25": p25, "p75": p75},
        "dispersion": dispersion,
        "status": status,
        "withheld": withheld,
        "withhold_reasons": reasons,
        "source_medians": source_medians,
        "source_units": source_units,
        "source_update_counts": source_update_counts,
        "source_notes": source_notes,
        "selected_sources": selected_sources,
        "selected_quote_time": selected_quote_time,
        "selection_method": selection_method,
        "using_stale_fallback": using_stale_fallback,
        "stale_fallback_sources": sorted(set(stale_fallback_sources)),
        "available": (fix_value is not None and not withheld),
    }


def compute_indicator_results(benchmark_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    street = benchmark_results.get("open_market", {})
    official = benchmark_results.get("official", {})
    transfer = benchmark_results.get("regional_transfer", {})
    crypto = benchmark_results.get("crypto_usdt", {})
    emami = benchmark_results.get("emami_gold_coin", {})

    street_fix = parse_number(street.get("fix"))
    if street.get("withheld") is True:
        street_fix = None
    official_fix = parse_number(official.get("fix"))
    if official.get("withheld") is True:
        official_fix = None
    transfer_fix = parse_number(transfer.get("fix"))
    if transfer.get("withheld") is True:
        transfer_fix = None
    crypto_fix = parse_number(crypto.get("fix"))
    if crypto.get("withheld") is True:
        crypto_fix = None
    emami_fix = parse_number(emami.get("fix"))
    if emami.get("withheld") is True:
        emami_fix = None

    street_official_gap_pct: Optional[float] = None
    if street_fix is not None and official_fix not in (None, 0):
        street_official_gap_pct = ((street_fix - official_fix) / official_fix) * 100

    street_transfer_gap_pct: Optional[float] = None
    if street_fix is not None and transfer_fix not in (None, 0):
        street_transfer_gap_pct = ((street_fix - transfer_fix) / transfer_fix) * 100

    street_crypto_gap_pct: Optional[float] = None
    if street_fix not in (None, 0) and crypto_fix is not None:
        street_crypto_gap_pct = ((crypto_fix - street_fix) / street_fix) * 100

    gold_usd_per_oz = resolve_gold_usd_per_oz()
    gold_implied_fx = compute_gold_implied_fx(emami_fix, gold_usd_per_oz)
    street_gold_gap_pct: Optional[float] = None
    if street_fix not in (None, 0) and gold_implied_fx is not None:
        street_gold_gap_pct = ((gold_implied_fx - street_fix) / street_fix) * 100

    return {
        "street_official_gap_pct": {
            "label": INDICATOR_LABELS["street_official_gap_pct"],
            "value": street_official_gap_pct,
            "available": street_official_gap_pct is not None,
            "formula": INDICATOR_FORMULAS["street_official_gap_pct"],
        },
        "street_transfer_gap_pct": {
            "label": INDICATOR_LABELS["street_transfer_gap_pct"],
            "value": street_transfer_gap_pct,
            "available": street_transfer_gap_pct is not None,
            "formula": INDICATOR_FORMULAS["street_transfer_gap_pct"],
        },
        "street_crypto_gap_pct": {
            "label": INDICATOR_LABELS["street_crypto_gap_pct"],
            "value": street_crypto_gap_pct,
            "available": street_crypto_gap_pct is not None,
            "formula": INDICATOR_FORMULAS["street_crypto_gap_pct"],
        },
        "street_gold_gap_pct": {
            "label": INDICATOR_LABELS["street_gold_gap_pct"],
            "value": street_gold_gap_pct,
            "available": street_gold_gap_pct is not None,
            "formula": INDICATOR_FORMULAS["street_gold_gap_pct"],
        },
        "official_commercial_trend_7d": {
            "label": INDICATOR_LABELS["official_commercial_trend_7d"],
            "value": None,
            "available": False,
            "formula": INDICATOR_FORMULAS["official_commercial_trend_7d"],
        },
    }


def parse_iso_date_text(value: Any) -> Optional[dt.date]:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def benchmark_series_path(site_dir: Path, benchmark_key: str) -> Path:
    return site_dir / "api" / "benchmarks" / f"{benchmark_key}.json"


def indicator_series_path(site_dir: Path, indicator_key: str) -> Path:
    return site_dir / "api" / "indicators" / f"{indicator_key}.json"


def load_public_value_series(path: Path) -> List[Tuple[dt.date, float]]:
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    rows: List[Tuple[dt.date, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        day = parse_iso_date_text(item.get("date"))
        value = parse_number(item.get("value"))
        if day is None or value is None:
            continue
        rows.append((day, value))

    rows.sort(key=lambda item: item[0])
    return rows


def reconstruct_benchmark_fix_history(site_dir: Path, benchmark_key: str) -> List[Tuple[dt.date, float]]:
    rows: List[Tuple[dt.date, float]] = []
    fix_dir = site_dir / "fix"
    if not fix_dir.exists():
        return rows

    for path in sorted(fix_dir.glob("*.json")):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
            continue
        try:
            daily = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        day = parse_iso_date_text(daily.get("date") or path.stem)
        if day is None:
            continue
        benchmark = daily.get("benchmarks", {}).get(benchmark_key, {})
        if not isinstance(benchmark, dict):
            continue
        fix = parse_number(benchmark.get("fix"))
        if fix is None:
            continue
        rows.append((day, fix))

    rows.sort(key=lambda item: item[0])
    return rows


def load_benchmark_fix_history(site_dir: Path, benchmark_key: str) -> List[Tuple[dt.date, float]]:
    rows = load_public_value_series(benchmark_series_path(site_dir, benchmark_key))
    if rows:
        return rows
    return reconstruct_benchmark_fix_history(site_dir, benchmark_key)


def extract_daily_gap_value(daily: Dict[str, Any], gap_key: str) -> Optional[float]:
    computed = daily.get("computed", {})
    if not isinstance(computed, dict):
        return None
    if computed.get("withheld") is True:
        return None

    indicators = daily.get("indicators", {})
    if isinstance(indicators, dict):
        entry = indicators.get(gap_key, {})
        if isinstance(entry, dict):
            stored = parse_float(entry.get("value"))
            if stored is not None:
                return stored

    street_fix = parse_number(computed.get("fix"))
    if street_fix is None or street_fix <= 0:
        return None

    benchmarks = daily.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        benchmarks = {}

    def benchmark_fix_value(key: str) -> Optional[float]:
        entry = benchmarks.get(key, {})
        if not isinstance(entry, dict):
            return None
        if entry.get("withheld") is True or entry.get("available") is False:
            return None
        return parse_number(entry.get("fix"))

    official_fix = benchmark_fix_value("official")
    transfer_fix = benchmark_fix_value("regional_transfer")
    crypto_fix = benchmark_fix_value("crypto_usdt")
    emami_fix = benchmark_fix_value("emami_gold_coin")

    if gap_key == "street_official_gap_pct":
        if official_fix in (None, 0):
            return None
        return ((street_fix - official_fix) / official_fix) * 100
    if gap_key == "street_transfer_gap_pct":
        if transfer_fix in (None, 0):
            return None
        return ((street_fix - transfer_fix) / transfer_fix) * 100
    if gap_key == "street_crypto_gap_pct":
        if crypto_fix is None:
            return None
        return ((crypto_fix - street_fix) / street_fix) * 100
    if gap_key == "street_gold_gap_pct":
        gold_implied_fx = compute_gold_implied_fx(emami_fix, resolve_gold_usd_per_oz(daily))
        if gold_implied_fx is None:
            return None
        return ((gold_implied_fx - street_fix) / street_fix) * 100
    return None


def reconstruct_indicator_gap_history(site_dir: Path, gap_key: str) -> List[Tuple[dt.date, float]]:
    rows: List[Tuple[dt.date, float]] = []
    fix_dir = site_dir / "fix"
    if not fix_dir.exists():
        return rows

    for path in sorted(fix_dir.glob("*.json")):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
            continue
        try:
            daily = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        day = parse_iso_date_text(daily.get("date") or path.stem)
        if day is None:
            continue
        value = extract_daily_gap_value(daily, gap_key)
        if value is None:
            continue
        rows.append((day, value))

    rows.sort(key=lambda item: item[0])
    return rows


def load_indicator_gap_history(site_dir: Path, gap_key: str) -> List[Tuple[dt.date, float]]:
    rows = load_public_value_series(indicator_series_path(site_dir, gap_key))
    if rows:
        return rows
    return reconstruct_indicator_gap_history(site_dir, gap_key)


def compute_official_trend_from_history_map(
    by_day: Dict[dt.date, float], latest_day: dt.date, latest_official_fix: Optional[float]
) -> Optional[float]:
    if latest_official_fix is None or latest_official_fix <= 0:
        return None
    target_day = latest_day - dt.timedelta(days=7)
    base_candidates = [day for day in by_day if day <= target_day]
    if not base_candidates:
        return None
    base_day = max(base_candidates)
    base_fix = by_day.get(base_day)
    if base_fix is None or base_fix <= 0:
        return None
    return ((latest_official_fix - base_fix) / base_fix) * 100


def compute_official_trend_7d(site_dir: Path, latest_day: dt.date, latest_official_fix: Optional[float]) -> Optional[float]:
    history = load_benchmark_fix_history(site_dir, "official")
    by_day: Dict[dt.date, float] = {day: fix for day, fix in history}
    by_day[latest_day] = latest_official_fix
    return compute_official_trend_from_history_map(by_day, latest_day, latest_official_fix)


def evaluate_official_sanity_warnings(benchmark_results: Dict[str, Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    street_fix = parse_number(benchmark_results.get("open_market", {}).get("fix"))
    official_fix = parse_number(benchmark_results.get("official", {}).get("fix"))
    if street_fix is None or official_fix is None:
        return warnings

    if official_fix > street_fix * 1.05:
        warnings.append(
            "official_commercial_usd_irr is materially above street_usd_irr; review parser and rial/toman normalization"
        )
    ratio = official_fix / street_fix if street_fix else None
    if ratio is not None and (ratio < 0.3 or ratio > 1.3):
        warnings.append("official_commercial_usd_irr appears outside plausible band relative to street_usd_irr")
    return warnings


def build_normalized_market_snapshot(daily: Dict[str, Any], site_dir: Optional[Path] = None) -> Dict[str, Any]:
    benchmarks = daily.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        benchmarks = {}
    indicators = daily.get("indicators", {})
    if not isinstance(indicators, dict):
        indicators = {}

    def benchmark_fix_value(key: str) -> Optional[float]:
        entry = benchmarks.get(key, {})
        if not isinstance(entry, dict):
            return None
        if entry.get("withheld") is True or entry.get("available") is False:
            return None
        return parse_number(entry.get("fix"))

    street = benchmark_fix_value("open_market")
    official = benchmark_fix_value("official")
    transfer = benchmark_fix_value("regional_transfer")
    crypto = benchmark_fix_value("crypto_usdt")
    emami = benchmark_fix_value("emami_gold_coin")

    def indicator_value(key: str) -> Optional[float]:
        entry = indicators.get(key, {})
        if isinstance(entry, dict):
            parsed = parse_float(entry.get("value"))
            if parsed is not None:
                return parsed
        return None

    street_official_gap = indicator_value("street_official_gap_pct")
    if street_official_gap is None and street is not None and official not in (None, 0):
        street_official_gap = ((street - official) / official) * 100

    street_transfer_gap = indicator_value("street_transfer_gap_pct")
    if street_transfer_gap is None and street is not None and transfer not in (None, 0):
        street_transfer_gap = ((street - transfer) / transfer) * 100

    street_crypto_gap = indicator_value("street_crypto_gap_pct")
    if street_crypto_gap is None and street not in (None, 0) and crypto is not None:
        street_crypto_gap = ((crypto - street) / street) * 100

    street_gold_gap = indicator_value("street_gold_gap_pct")
    gold_usd_per_oz = resolve_gold_usd_per_oz(daily)
    gold_implied_fx = compute_gold_implied_fx(emami, gold_usd_per_oz)
    if street_gold_gap is None and street not in (None, 0) and gold_implied_fx is not None:
        street_gold_gap = ((gold_implied_fx - street) / street) * 100

    official_trend = indicator_value("official_commercial_trend_7d")
    latest_day = parse_iso_date_text(daily.get("date"))
    if official_trend is None and site_dir is not None and latest_day is not None:
        official_trend = compute_official_trend_7d(site_dir, latest_day, official)

    primary = benchmarks.get("open_market", {})
    source_units = primary.get("source_units") if isinstance(primary, dict) else {}
    unit_values = set(source_units.values()) if isinstance(source_units, dict) else set()
    if len(unit_values) == 1:
        source_unit = next(iter(unit_values))
    elif len(unit_values) > 1:
        source_unit = "mixed"
    else:
        source_unit = "unknown"

    status = daily.get("computed", {}).get("status")
    return {
        "street_usd_irr": street,
        "official_commercial_usd_irr": official,
        "regional_transfer_usd_irr": transfer,
        "crypto_usdt_irr": crypto,
        "emami_coin_irr": emami,
        "street_official_gap_pct": street_official_gap,
        "street_transfer_gap_pct": street_transfer_gap,
        "street_crypto_gap_pct": street_crypto_gap,
        "street_gold_gap_pct": street_gold_gap,
        "gold_implied_usd_irr": gold_implied_fx,
        "gold_usd_per_oz": gold_usd_per_oz,
        "official_commercial_trend_7d": official_trend,
        "street_confidence_status": str(status) if status is not None else None,
        "source_unit": source_unit,
        "normalized_unit": "rial",
    }


def summarize_day(
    samples: Dict[str, List[Sample]],
    source_configs: List[SourceConfig],
    day: dt.date,
    primary_allow_stale: bool = False,
) -> Dict[str, Any]:
    benchmark_sources = {cfg.name: cfg.benchmark_families for cfg in source_configs}

    benchmark_results: Dict[str, Dict[str, Any]] = {}
    for key in BENCHMARK_LABELS:
        benchmark_results[key] = compute_benchmark_result(
            samples,
            key,
            benchmark_sources,
            primary_allow_stale=primary_allow_stale,
        )

    primary = benchmark_results[PRIMARY_BENCHMARK]

    source_benchmark_medians: Dict[str, Dict[str, float]] = {}
    for source, entries in samples.items():
        source_benchmark_medians[source] = {}
        for key in BENCHMARK_LABELS:
            values = [s.benchmark_values.get(key) for s in entries if s.benchmark_values.get(key) is not None]
            if values:
                source_benchmark_medians[source][key] = median(values)

    computed_benchmarks = {
        key: {
            "label": result.get("label", BENCHMARK_LABELS[key]),
            "value": result.get("fix"),
            "available": bool(result.get("available")),
            "is_primary": key == PRIMARY_BENCHMARK,
        }
        for key, result in benchmark_results.items()
    }
    indicator_results = compute_indicator_results(benchmark_results)
    official_warnings = evaluate_official_sanity_warnings(benchmark_results)
    if official_warnings:
        official_result = benchmark_results.get("official", {})
        if isinstance(official_result, dict):
            official_result["validation_warnings"] = official_warnings

    sample_count_per_source = max((len(entries) for entries in samples.values()), default=0)

    daily_payload = {
        "date": iso_date(day),
        "as_of": iso_ts(utc_now()),
        "window_utc": {
            "start": WINDOW_START.strftime("%H:%M"),
            "end": WINDOW_END.strftime("%H:%M"),
            "sample_count_per_source": sample_count_per_source,
        },
        "sources": {
	            source: {
	                "samples": [
	                    {
	                        **({"health": s.health} if isinstance(s.health, dict) else {"health": {}}),
	                        "sampled_at": iso_ts(s.sampled_at),
	                        "value": s.value,
	                        "benchmarks": s.benchmark_values,
	                        "quote_time": iso_ts(s.quote_time) if s.quote_time else None,
	                        "ok": s.ok,
	                        "stale": s.stale,
	                        "error": s.error,
	                        "fetch_success": (
	                            s.health.get("fetch_success")
	                            if isinstance(s.health, dict)
	                            else None
	                        ),
	                        "validation_result": (
	                            s.health.get("validation_result")
	                            if isinstance(s.health, dict)
	                            else None
	                        ),
	                        "failure_reason": (
	                            s.health.get("failure_reason")
	                            if isinstance(s.health, dict)
	                            else None
	                        ),
	                        "source_unit": s.source_unit,
	                        "normalized_unit": s.normalized_unit,
	                    }
	                    for s in entries
	                ],
                "median": source_benchmark_medians.get(source, {}).get("open_market"),
                "benchmark_medians": source_benchmark_medians.get(source, {}),
                "note": benchmark_results["open_market"].get("source_notes", {}).get(source),
            }
            for source, entries in samples.items()
        },
        "benchmarks": benchmark_results,
        "indicators": indicator_results,
        "computed": {
            "fix": primary.get("fix"),
            "band": primary.get("band"),
            "dispersion": primary.get("dispersion"),
            "status": primary.get("status"),
            "withheld": primary.get("withheld"),
            "withhold_reasons": primary.get("withhold_reasons"),
            "source_medians": primary.get("source_medians"),
            "source_units": primary.get("source_units"),
            "source_unit": (
                next(iter(primary.get("source_units", {}).values()))
                if isinstance(primary.get("source_units"), dict) and len(set(primary.get("source_units", {}).values())) == 1
                else ("mixed" if isinstance(primary.get("source_units"), dict) and primary.get("source_units") else "unknown")
            ),
            "normalized_unit": "rial",
            "validation_warnings": official_warnings,
            "benchmarks": computed_benchmarks,
            "indicators": indicator_results,
        },
    }
    daily_payload["normalized_metrics"] = build_normalized_market_snapshot(daily_payload)
    daily_payload["methodology"] = {
        "mapping_fingerprint": current_mapping_fingerprint(),
        "canonical_source_symbols": {
            source: {benchmark: list(symbols) for benchmark, symbols in mapping.items()}
            for source, mapping in CANONICAL_SOURCE_SYMBOLS.items()
        },
        "navasan_benchmark_units": dict(NAVASAN_BENCHMARK_UNITS),
    }
    return daily_payload


def load_existing_days(site_dir: Path) -> List[str]:
    fix_dir = site_dir / "fix"
    if not fix_dir.exists():
        return []

    dates: List[str] = []
    for path in sorted(fix_dir.glob("*.json")):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
            dates.append(path.stem)
    return dates


def normalize_series_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    date = row.get("date")
    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None

    fix = parse_number(row.get("fix"))
    p25 = parse_number(row.get("p25"))
    p75 = parse_number(row.get("p75"))
    status = row.get("status")
    withheld = row.get("withheld")
    if isinstance(withheld, str):
        withheld = withheld.strip().lower() in {"1", "true", "yes"}
    elif not isinstance(withheld, bool):
        withheld = None

    return {
        "date": date,
        "fix": fix,
        "p25": p25,
        "p75": p75,
        "status": str(status) if status is not None else None,
        "withheld": withheld,
    }


def load_existing_series_rows(site_dir: Path) -> List[Dict[str, Any]]:
    series_path = site_dir / "api" / "series.json"
    if not series_path.exists():
        return []

    try:
        payload = json.loads(series_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    items = payload.get("rows")
    if not isinstance(items, list):
        return []

    rows: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        normalized = normalize_series_row(raw)
        if normalized is not None:
            rows.append(normalized)
    return rows


def load_series_rows(site_dir: Path) -> List[Dict[str, Any]]:
    # Keep previously published rows and let immutable per-day files override them.
    rows_by_date: Dict[str, Dict[str, Any]] = {}
    for row in load_existing_series_rows(site_dir):
        rows_by_date[row["date"]] = row

    fix_dir = site_dir / "fix"
    if fix_dir.exists():
        for path in sorted(fix_dir.glob("*.json")):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
                continue

            try:
                daily = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            computed = daily.get("computed", {})
            normalized = normalize_series_row(
                {
                    "date": daily.get("date", path.stem),
                    "fix": computed.get("fix"),
                    "p25": computed.get("band", {}).get("p25"),
                    "p75": computed.get("band", {}).get("p75"),
                    "status": computed.get("status"),
                    "withheld": computed.get("withheld"),
                }
            )
            if normalized is None:
                continue
            rows_by_date[normalized["date"]] = normalized

    rows = sorted(rows_by_date.values(), key=lambda r: r["date"])
    return rows


def is_public_series_row(row: Dict[str, Any]) -> bool:
    status = row.get("status")
    fix = row.get("fix")
    withheld = row.get("withheld")

    if not isinstance(status, str) or status not in {"Green", "Amber", "Red"}:
        return False
    if withheld is not False:
        return False
    if not isinstance(fix, (int, float)):
        return False
    if not math.isfinite(float(fix)) or float(fix) <= 0:
        return False

    return True


def copy_static_assets(assets_dir: Path, site_dir: Path) -> None:
    if not assets_dir.exists():
        return

    target_root = site_dir / "assets"
    for path in assets_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(assets_dir)
        dest = target_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def render_page(templates_dir: Path, page_template: str, title: str, generated_at: str, **kwargs: Any) -> str:
    layout = read_template(templates_dir, "layout.html")
    content = read_template(templates_dir, page_template).safe_substitute(**kwargs)
    return layout.safe_substitute(title=title, content=content, generated_at=generated_at)


def publish_methodology(site_dir: Path, templates_dir: Path, generated_at: str) -> None:
    html = render_page(
        templates_dir,
        "methodology.html",
        title="Methodology",
        generated_at=generated_at,
    )
    write_text(site_dir / "methodology" / "index.html", html)


def publish_governance(site_dir: Path, templates_dir: Path, generated_at: str) -> None:
    html = render_page(
        templates_dir,
        "governance.html",
        title="Governance",
        generated_at=generated_at,
    )
    write_text(site_dir / "governance" / "index.html", html)


def publish_status(
    site_dir: Path,
    templates_dir: Path,
    generated_at: str,
    status_title: str,
    status_detail: str,
    missing: Optional[List[str]] = None,
    latest: Optional[Dict[str, Any]] = None,
) -> None:
    def humanize_withhold_reason(reason: Any) -> Optional[str]:
        if not isinstance(reason, str):
            return None
        text = reason.strip().lower()
        if not text:
            return None
        if "no valid source" in text:
            return "Insufficient valid sources"
        if "dispersion" in text:
            return "High dispersion across sources"
        if "stale" in text or "invalid/stale" in text or "outside observation window" in text:
            return "Stale or invalid source inputs"
        if "missing secret" in text or "missing secrets" in text or "config needed" in text:
            return "Configuration needed"
        if "no existing published data" in text:
            return "No published daily reference available"
        if "no intraday" in text:
            return "No intraday samples in publication window"
        if "no valid attempts" in text:
            return "No valid intraday samples in publication window"
        return "Publication withheld by methodology checks"

    def fmt_status_time(value: Any) -> str:
        parsed = try_parse_datetime(value)
        if parsed is None:
            return "Unknown"
        return parsed.strftime("%b %d, %Y · %H:%M UTC")

    def light_class(label: str) -> str:
        lowered = label.strip().lower()
        if lowered == "online":
            return "online"
        if lowered == "degraded":
            return "degraded"
        if lowered == "offline":
            return "offline"
        return "unknown"

    def render_status_label(label: str) -> str:
        cls = light_class(label)
        safe = html_lib.escape(label)
        return (
            f'<span class="status-label"><span class="status-light status-{cls}"></span>{safe}</span>'
        )

    def humanize_source_note(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        text = value.strip().lower()
        if not text:
            return ""
        if "no valid in-window samples" in text:
            return "No valid benchmark-window sample today."
        if "no valid samples" in text:
            return "No recent valid sample."
        if "outside observation window" in text:
            return "Out-of-window observation."
        if "source family not used" in text:
            return "Not currently used for this benchmark."
        return value.strip()

    def source_public_label(source_name: str, index: int) -> str:
        mapping = {
            "alanchand_street": "Street Market Feed",
            "bonbast": "Street Market Feed (Secondary)",
            "navasan": "Commercial Market Feed",
            "commercial_aux": "Commercial Market Feed (Auxiliary)",
            "commercial_aux_a": "Commercial Market Feed (Auxiliary A)",
            "commercial_aux_b": "Commercial Market Feed (Auxiliary B)",
            "commercial_aux_c": "Commercial Market Feed (Auxiliary C)",
            "commercial_profile_transfer": "Commercial Market Feed (Profile Transfer)",
            "commercial_profile_sana": "Commercial Market Feed (Profile Managed)",
            # Backward-compatible display for legacy persisted source names.
            "tgju_call": "Commercial Market Feed (Auxiliary)",
            "tgju_call2": "Commercial Market Feed (Auxiliary A)",
            "tgju_call3": "Commercial Market Feed (Auxiliary B)",
            "tgju_call4": "Commercial Market Feed (Auxiliary C)",
            "alanchand": "Regional Market Feed",
        }
        key = source_name.strip().lower()
        return mapping.get(key, f"Market Feed {index}")

    def parse_nonnegative_count(value: Any) -> int:
        parsed = parse_number(value)
        if parsed is None:
            return 0
        return max(0, int(parsed))

    def map_pipeline_state(value: str) -> str:
        upper = value.strip().upper()
        if upper in {"OK", "IMMUTABLE"}:
            return "Online"
        if upper == "WITHHOLD":
            return "Degraded"
        if upper == "CONFIG NEEDED":
            return "Offline"
        return "Unknown"

    def load_recent_fix_source_samples(lookback_days: int) -> Dict[str, List[Dict[str, Any]]]:
        history: Dict[str, List[Dict[str, Any]]] = {}
        fix_dir = site_dir / "fix"
        if not fix_dir.exists():
            return history

        cutoff_date = (utc_now() - dt.timedelta(days=lookback_days)).date()
        dated_paths: List[Tuple[dt.date, Path]] = []
        for path in fix_dir.glob("*.json"):
            try:
                stamp = dt.date.fromisoformat(path.stem)
            except ValueError:
                continue
            if stamp < cutoff_date:
                continue
            dated_paths.append((stamp, path))
        dated_paths.sort(key=lambda row: row[0], reverse=True)

        for _stamp, path in dated_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            sources_obj = payload.get("sources")
            if not isinstance(sources_obj, dict):
                continue
            for source_name, source_data in sources_obj.items():
                source_key = canonical_source_name(str(source_name))
                if not isinstance(source_data, dict):
                    continue
                samples_obj = source_data.get("samples")
                if not isinstance(samples_obj, list):
                    continue
                for sample in samples_obj:
                    if not isinstance(sample, dict):
                        continue
                    history.setdefault(source_key, []).append(sample)
        return history

    def sample_official_value(sample_obj: Dict[str, Any]) -> Optional[float]:
        benchmarks_obj = sample_obj.get("benchmarks")
        if isinstance(benchmarks_obj, dict):
            parsed = parse_number(benchmarks_obj.get("official"))
            if parsed is not None:
                return parsed
        health_obj = sample_obj.get("health")
        if isinstance(health_obj, dict):
            extracted_obj = health_obj.get("extracted_values")
            if isinstance(extracted_obj, dict):
                parsed = parse_number(extracted_obj.get("official"))
                if parsed is not None:
                    return parsed
        return None

    def sample_official_quote_time(sample_obj: Dict[str, Any]) -> Optional[dt.datetime]:
        health_obj = sample_obj.get("health")
        if isinstance(health_obj, dict):
            benchmark_quote_times = health_obj.get("benchmark_quote_times")
            if isinstance(benchmark_quote_times, dict):
                quote_ts = try_parse_datetime(benchmark_quote_times.get("official"))
                if quote_ts is not None:
                    return quote_ts
        return try_parse_datetime(sample_obj.get("quote_time"))

    missing_html = ""
    if missing:
        missing_html = "<ul class=\"mb-0 mt-2\">" + "".join(f"<li><code>{html_lib.escape(m)}</code></li>" for m in missing) + "</ul>"

    effective_latest: Dict[str, Any] = {}
    if isinstance(latest, dict):
        effective_latest = latest
    else:
        latest_path = site_dir / "api" / "latest.json"
        if latest_path.exists():
            try:
                loaded = json.loads(latest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    effective_latest = loaded
            except json.JSONDecodeError:
                effective_latest = {}

    computed = effective_latest.get("computed", {})
    if not isinstance(computed, dict):
        computed = {}
    benchmarks = effective_latest.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        benchmarks = {}
    raw_sources = effective_latest.get("sources", {})
    if not isinstance(raw_sources, dict):
        raw_sources = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for source_name, source_data in raw_sources.items():
        source_key = canonical_source_name(str(source_name))
        normalized_data = source_data if isinstance(source_data, dict) else {}
        existing = sources.get(source_key)
        if existing is None:
            sources[source_key] = dict(normalized_data)
            continue
        existing_samples = existing.get("samples")
        incoming_samples = normalized_data.get("samples")
        if not isinstance(existing_samples, list):
            existing_samples = []
        if isinstance(incoming_samples, list) and incoming_samples:
            existing["samples"] = existing_samples + incoming_samples
        if not isinstance(existing.get("note"), str) and isinstance(normalized_data.get("note"), str):
            existing["note"] = normalized_data.get("note")

    status_history_lookback_days = env_int(
        "STATUS_HISTORY_LOOKBACK_DAYS",
        STATUS_HISTORY_LOOKBACK_DAYS_DEFAULT,
        minimum=1,
        maximum=30,
    )
    recent_fix_samples = load_recent_fix_source_samples(status_history_lookback_days)

    # Status page should show the full configured source universe, even when
    # the latest published day is based on older artifacts with fewer source keys.
    for cfg in build_source_configs():
        sources.setdefault(cfg.name, {"samples": []})

    fix = parse_number(computed.get("fix"))
    withheld = bool(computed.get("withheld", True))
    publication_state = "Unknown"
    if fix is not None:
        publication_state = "Withheld" if withheld else "Published"

    reasons = computed.get("withhold_reasons", [])
    humanized_reasons: List[str] = []
    if isinstance(reasons, list):
        for reason in reasons:
            normalized = humanize_withhold_reason(reason)
            if normalized:
                humanized_reasons.append(normalized)
    publication_reason = "None"
    if publication_state == "Withheld":
        publication_reason = humanized_reasons[0] if humanized_reasons else "Publication withheld by methodology checks"

    publication_selection = effective_latest.get("publication_selection", {})
    if not isinstance(publication_selection, dict):
        publication_selection = {}
    valid_candidate_count_raw = publication_selection.get("valid_candidate_count")
    valid_candidate_count: Optional[int] = None
    if isinstance(valid_candidate_count_raw, (int, float)) and math.isfinite(float(valid_candidate_count_raw)):
        valid_candidate_count = int(valid_candidate_count_raw)

    street_source_count_raw = publication_selection.get("street_source_count_used")
    street_source_count_used: Optional[int] = None
    if isinstance(street_source_count_raw, (int, float)) and math.isfinite(float(street_source_count_raw)):
        street_source_count_used = int(street_source_count_raw)
    if street_source_count_used is None:
        source_medians = computed.get("source_medians", {}) if isinstance(computed.get("source_medians"), dict) else {}
        street_source_count_used = len([k for k, v in source_medians.items() if parse_number(v) is not None])

    same_as_previous_day = bool(publication_selection.get("same_as_previous_day"))
    delta_pct_vs_previous_day = parse_float(publication_selection.get("delta_pct_vs_previous_day"))

    publication_selection_parts: List[str] = []
    if valid_candidate_count is not None:
        label = "candidate" if valid_candidate_count == 1 else "candidates"
        publication_selection_parts.append(f"{valid_candidate_count} valid intraday {label}")
    if street_source_count_used is not None:
        source_label = "source" if street_source_count_used == 1 else "sources"
        publication_selection_parts.append(f"{street_source_count_used} street {source_label} used")
    if publication_state == "Published" and same_as_previous_day:
        publication_selection_parts.append("Freshly selected, unchanged vs prior day")
    elif publication_state == "Published" and delta_pct_vs_previous_day is not None:
        publication_selection_parts.append(f"Delta vs prior day {delta_pct_vs_previous_day:+.1f}%")
    publication_selection_context = (
        " · ".join(publication_selection_parts) if publication_selection_parts else "Selection context unavailable."
    )

    publication_fix = f"{fmt_rate(fix)} IRR" if fix is not None else "Unavailable"
    publication_updated = fmt_status_time(effective_latest.get("as_of") or generated_at)
    recent_success_window = dt.timedelta(
        hours=env_int(
            "STATUS_RECENT_SUCCESS_HOURS",
            STATUS_RECENT_SUCCESS_HOURS_DEFAULT,
            minimum=1,
            maximum=24 * 30,
        )
    )

    source_rows: List[Dict[str, str]] = []
    ordered_source_names = sorted(sources.keys(), key=lambda item: str(item).lower())
    for idx, source_name in enumerate(ordered_source_names, start=1):
        source_data = sources.get(source_name, {})
        if not isinstance(source_data, dict):
            source_data = {}
        samples = source_data.get("samples", [])
        if not isinstance(samples, list):
            samples = []
        historical_samples = recent_fix_samples.get(source_name, [])
        if isinstance(historical_samples, list) and historical_samples:
            samples = [*samples, *(s for s in historical_samples if isinstance(s, dict))]

        latest_sample: Optional[Dict[str, Any]] = None
        latest_sample_ts: Optional[dt.datetime] = None
        last_success_ts: Optional[dt.datetime] = None
        last_success_sample: Optional[Dict[str, Any]] = None
        any_fetch_success = False
        any_offline_failure = False
        any_degraded_failure = False
        any_parsed_data = False
        degraded_due_to_recent_success = False
        reachability_status = "Unknown"

        def sample_has_parsed_data(sample_obj: Dict[str, Any]) -> bool:
            if parse_number(sample_obj.get("value")) is not None:
                return True
            benchmarks_obj = sample_obj.get("benchmarks")
            if isinstance(benchmarks_obj, dict):
                for benchmark_value in benchmarks_obj.values():
                    if parse_number(benchmark_value) is not None:
                        return True
            health_obj = sample_obj.get("health")
            if isinstance(health_obj, dict):
                for field in ("extracted_values", "raw_extracted_values"):
                    extracted_obj = health_obj.get(field)
                    if isinstance(extracted_obj, dict):
                        for extracted_value in extracted_obj.values():
                            if parse_number(extracted_value) is not None:
                                return True
            return False

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_ts = try_parse_datetime(sample.get("sampled_at"))
            if sample_ts is None:
                sample_ts = try_parse_datetime(sample.get("quote_time"))
            if sample_ts is not None and (latest_sample_ts is None or sample_ts > latest_sample_ts):
                latest_sample_ts = sample_ts
                latest_sample = sample

            has_parsed_data = sample_has_parsed_data(sample)
            fetch_success = sample.get("fetch_success")
            failure_reason = sample.get("failure_reason")
            sample_error = sample.get("error")
            validation = sample.get("validation_result")
            validation_failed = isinstance(validation, dict) and validation.get("ok") is False

            if fetch_success is True:
                any_fetch_success = True
                if sample_ts is not None and (last_success_ts is None or sample_ts > last_success_ts):
                    last_success_ts = sample_ts
                    last_success_sample = sample
                elif last_success_sample is None:
                    last_success_sample = sample
                if has_parsed_data:
                    any_parsed_data = True
                if validation_failed or (isinstance(failure_reason, str) and failure_reason.strip()):
                    any_degraded_failure = True
                if not has_parsed_data:
                    any_degraded_failure = True
            elif fetch_success is False:
                any_offline_failure = True
            elif isinstance(sample_error, str) and sample_error.strip():
                # Errors with no successful fetch are treated as source offline/unreachable.
                any_offline_failure = True

        source_status = "Unknown"
        latest_status: Optional[str] = None
        if latest_sample:
            latest_has_parsed_data = sample_has_parsed_data(latest_sample)
            latest_fetch_success = latest_sample.get("fetch_success")
            latest_failure_reason = latest_sample.get("failure_reason")
            latest_sample_error = latest_sample.get("error")
            latest_validation = latest_sample.get("validation_result")
            latest_validation_failed = isinstance(latest_validation, dict) and latest_validation.get("ok") is False
            latest_failure_text = isinstance(latest_failure_reason, str) and latest_failure_reason.strip()
            latest_error_text = isinstance(latest_sample_error, str) and latest_sample_error.strip()

            if latest_fetch_success is True:
                reachability_status = "Online"
                if latest_has_parsed_data and not latest_validation_failed and not latest_failure_text:
                    latest_status = "Online"
                else:
                    latest_status = "Degraded"
            elif latest_fetch_success is False:
                reachability_status = "Offline"
                latest_status = "Offline"
            elif latest_error_text or latest_failure_text:
                reachability_status = "Offline"
                latest_status = "Offline"

        if latest_status:
            source_status = latest_status
        elif any_fetch_success and any_parsed_data and not any_degraded_failure:
            source_status = "Online"
        elif any_fetch_success:
            source_status = "Degraded"
        elif any_offline_failure or samples:
            source_status = "Offline"

        if reachability_status == "Unknown":
            if any_fetch_success:
                reachability_status = "Online"
            elif any_offline_failure or samples:
                reachability_status = "Offline"

        status_reference_ts = latest_sample_ts or utc_now()
        if (
            source_status == "Offline"
            and last_success_ts is not None
            and (status_reference_ts - last_success_ts) <= recent_success_window
        ):
            source_status = "Degraded"
            degraded_due_to_recent_success = True

        note = humanize_source_note(source_data.get("note"))
        latest_fallback_active = False
        if latest_sample:
            latest_health = latest_sample.get("health")
            if isinstance(latest_health, dict) and latest_health.get("fallback_used") is True:
                latest_fallback_active = True
                note = "Served via backup market feed."
        if source_status == "Offline" and latest_sample:
            latest_failure_reason = latest_sample.get("failure_reason")
            latest_sample_error = latest_sample.get("error")
            has_failure_text = isinstance(latest_failure_reason, str) and latest_failure_reason.strip()
            has_error_text = isinstance(latest_sample_error, str) and latest_sample_error.strip()
            if has_failure_text or has_error_text:
                note = "Source request failed."
        elif source_status == "Degraded" and not latest_fallback_active:
            if degraded_due_to_recent_success and last_success_ts is not None:
                note = (
                    "Intermittent source availability. "
                    f"Last successful update {last_success_ts.strftime('%b %d, %Y, %H:%M UTC')}."
                )
            else:
                note = "Source responded, but returned invalid data."

        if not note and latest_sample and latest_sample.get("stale") is True:
            note = "Out-of-window observation."
        if not note and latest_sample:
            failure_reason = latest_sample.get("failure_reason")
            sample_error = latest_sample.get("error")
            if source_status == "Degraded":
                if degraded_due_to_recent_success and last_success_ts is not None:
                    note = (
                        "Intermittent source availability. "
                        f"Last successful update {last_success_ts.strftime('%b %d, %Y, %H:%M UTC')}."
                    )
                else:
                    note = "Source responded, but returned invalid data."
            elif source_status == "Offline":
                if isinstance(failure_reason, str) and failure_reason.strip():
                    note = "Source request failed."
                elif isinstance(sample_error, str) and sample_error.strip():
                    note = "Source request failed."
        if not note and not samples:
            note = "No sample in latest published day."
        if not note:
            note = "Collecting normally." if source_status == "Online" else "No additional notes."

        official_quote_stale_note = ""
        freshness_status = "N/A"
        supports_official = bool(CANONICAL_SOURCE_SYMBOLS.get(source_name, {}).get("official"))
        if supports_official:
            freshness_status = "Unknown"
            freshness_sample: Optional[Dict[str, Any]] = None
            if latest_sample is not None and sample_official_value(latest_sample) is not None:
                freshness_sample = latest_sample
            elif last_success_sample is not None and sample_official_value(last_success_sample) is not None:
                freshness_sample = last_success_sample

            if freshness_sample is not None:
                official_quote_time = sample_official_quote_time(freshness_sample)
                freshness_reference = latest_sample_ts or last_success_ts or utc_now()
                if official_quote_time is not None:
                    if official_quote_is_fresh(official_quote_time, freshness_reference):
                        freshness_status = "Fresh"
                    else:
                        freshness_status = "Stale"
                        official_quote_stale_note = (
                            f"Official quote stale since {official_quote_time.strftime('%b %d, %Y, %H:%M UTC')}."
                        )

        if official_quote_stale_note and source_status != "Offline":
            if note:
                separator = " " if note.endswith(".") else ". "
                note = f"{note}{separator}{official_quote_stale_note}"
            else:
                note = official_quote_stale_note

        status_context_parts = [f"Reachability {reachability_status.lower()}."]
        if supports_official:
            status_context_parts.append(f"Official freshness {freshness_status.lower()}.")
        status_context = " ".join(status_context_parts)
        if note:
            note = f"{status_context} {note}"
        else:
            note = status_context

        source_rows.append(
            {
                "name": source_public_label(str(source_name), idx),
                "status": source_status,
                "last_success": (
                    last_success_ts.strftime("%b %d, %Y, %H:%M UTC") if last_success_ts is not None else "N/A"
                ),
                "note": note,
            }
        )

    if not source_rows:
        source_rows_html = (
            "<tr><td>N/A</td><td>"
            + render_status_label("Unknown")
            + "</td><td>N/A</td><td>No source sample data available yet.</td></tr>"
        )
    else:
        source_rows_html = "\n".join(
            "<tr>"
            f"<td>{html_lib.escape(row['name'])}</td>"
            f"<td>{render_status_label(row['status'])}</td>"
            f"<td>{html_lib.escape(row['last_success'])}</td>"
            f"<td>{html_lib.escape(row['note'])}</td>"
            "</tr>"
            for row in source_rows
        )

    diagnostics_source_rows_html = (
        "<tr><td>N/A</td><td>"
        + render_status_label("Unknown")
        + "</td><td>0</td><td>0</td><td>No diagnostics locality-signal artifact found.</td></tr>"
    )
    diagnostics_source_note = "No diagnostics locality-signal artifact found."
    diagnostics_cards_path = site_dir / "api" / "regional_market_signals_card.json"
    if diagnostics_cards_path.exists():
        try:
            diagnostics_payload = json.loads(diagnostics_cards_path.read_text(encoding="utf-8"))
            cards = diagnostics_payload.get("cards", []) if isinstance(diagnostics_payload, dict) else []
            if isinstance(cards, list) and cards:
                diagnostics_rows: List[Dict[str, str]] = []
                publish_count = 0
                monitor_count = 0
                active_count = 0
                for card in cards:
                    if not isinstance(card, dict):
                        continue
                    basket_name = str(card.get("basket_name", "")).strip() or "Unknown"
                    signal_label = str(card.get("signal_label", "")).strip()
                    display_state = str(card.get("display_state", "")).strip().lower()
                    suppression_reason = str(card.get("suppression_reason", "")).strip()

                    status = "Unknown"
                    if display_state == "publish":
                        status = "Online"
                        publish_count += 1
                        active_count += 1
                    elif display_state == "monitor":
                        status = "Degraded"
                        monitor_count += 1
                        active_count += 1
                    elif display_state == "hide":
                        status = "Offline"

                    if status == "Online":
                        note = "Diagnostics signal available."
                    elif status == "Degraded":
                        reason = suppression_reason.replace("_", " ") if suppression_reason else "limited coverage"
                        note = f"Monitoring: {reason}."
                    else:
                        reason = suppression_reason.replace("_", " ") if suppression_reason else "not available"
                        note = f"Hidden: {reason}."

                    diagnostics_rows.append(
                        {
                            "name": f"{basket_name} — {signal_label}" if signal_label else basket_name,
                            "status": status,
                            "sources": str(parse_nonnegative_count(card.get("contributing_source_count"))),
                            "records": str(parse_nonnegative_count(card.get("usable_record_count"))),
                            "note": note,
                        }
                    )
                if diagnostics_rows:
                    diagnostics_source_rows_html = "\n".join(
                        "<tr>"
                        f"<td>{html_lib.escape(row['name'])}</td>"
                        f"<td>{render_status_label(row['status'])}</td>"
                        f"<td>{html_lib.escape(row['sources'])}</td>"
                        f"<td>{html_lib.escape(row['records'])}</td>"
                        f"<td>{html_lib.escape(row['note'])}</td>"
                        "</tr>"
                        for row in diagnostics_rows
                    )
                    diagnostics_source_note = (
                        f"{active_count}/{len(diagnostics_rows)} locality signals active. "
                        f"{publish_count} publish, {monitor_count} monitor."
                    )
        except json.JSONDecodeError:
            diagnostics_source_note = "Diagnostics locality-signal artifact could not be parsed."

    total_sources = len(source_rows)
    online_sources = sum(1 for row in source_rows if row["status"] == "Online")
    degraded_count = sum(1 for row in source_rows if row["status"] == "Degraded")
    offline_count = sum(1 for row in source_rows if row["status"] == "Offline")

    source_collection_state = "Unknown"
    source_collection_note = "No source samples available."
    if total_sources > 0:
        if online_sources == total_sources:
            source_collection_state = "Online"
            source_collection_note = f"{online_sources}/{total_sources} sources online."
        elif online_sources > 0:
            source_collection_state = "Degraded"
            source_collection_note = f"{online_sources}/{total_sources} sources online."
        elif degraded_count > 0:
            source_collection_state = "Degraded"
            source_collection_note = "Sources are partially available but degraded."
        elif offline_count > 0:
            source_collection_state = "Offline"
            source_collection_note = "No active source collection currently online."

    publication_overview_state = "Unknown"
    publication_overview_note = "No publication record available."
    if publication_state == "Published":
        publication_overview_state = "Online"
        publication_overview_note = f"Published {effective_latest.get('date', 'latest')} benchmark."
        if valid_candidate_count is not None:
            candidate_label = "candidate" if valid_candidate_count == 1 else "candidates"
            publication_overview_note += f" {valid_candidate_count} valid intraday {candidate_label}."
    elif publication_state == "Withheld":
        publication_overview_state = "Degraded"
        publication_overview_note = publication_reason

    pipeline_state = map_pipeline_state(status_title)
    deployment_state = "Online"
    pipeline_note = {
        "Online": "Benchmark pipeline is operating normally.",
        "Degraded": "Benchmark pipeline is operating with reduced reliability.",
        "Offline": "Benchmark pipeline is currently unavailable.",
        "Unknown": "Benchmark pipeline status is currently unknown.",
    }.get(pipeline_state, "Benchmark pipeline status is currently unknown.")
    if "withhold" in status_detail.strip().lower() and pipeline_state != "Offline":
        pipeline_note = "Benchmark pipeline is running, but publication quality checks are currently limiting output."

    overview_cards = [
        ("Benchmark Pipeline", pipeline_state, pipeline_note),
        ("Latest Publication", publication_overview_state, publication_overview_note),
        ("Source Collection", source_collection_state, source_collection_note),
        ("Site Deployment", deployment_state, "Dashboard deployment is online."),
    ]
    overview_cards_html = "\n".join(
        "<div class=\"status-overview-card\">"
        f"<div class=\"text-secondary small\">{html_lib.escape(title)}</div>"
        f"<div class=\"mt-1\">{render_status_label(label)}</div>"
        f"<div class=\"status-note\">{html_lib.escape(note)}</div>"
        "</div>"
        for title, label, note in overview_cards
    )

    diagnostics: List[str] = []
    mapping_path = site_dir / "api" / "mapping_audit.json"
    if mapping_path.exists():
        try:
            mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
            if isinstance(mapping_data, dict):
                stale_days = mapping_data.get("stale_days", [])
                if isinstance(stale_days, list) and stale_days:
                    stale_count = len(stale_days)
                    stale_label = "day" if stale_count == 1 else "days"
                    verb = "is" if stale_count == 1 else "are"
                    diagnostics.append(
                        f"{stale_count} historical {stale_label} {verb} flagged for mapping-audit review."
                    )
        except json.JSONDecodeError:
            diagnostics.append("Methodology mapping audit data could not be read.")

    if degraded_count > 0:
        degraded_label = "feed" if degraded_count == 1 else "feeds"
        degraded_verb = "is" if degraded_count == 1 else "are"
        diagnostics.append(f"{degraded_count} market {degraded_label} {degraded_verb} currently degraded.")
    if offline_count > 0:
        offline_label = "feed" if offline_count == 1 else "feeds"
        offline_verb = "is" if offline_count == 1 else "are"
        diagnostics.append(f"{offline_count} market {offline_label} {offline_verb} currently offline.")

    methodology = effective_latest.get("methodology", {})
    if isinstance(methodology, dict):
        rebuild_note = methodology.get("rebuild_note")
        if isinstance(rebuild_note, str) and rebuild_note.strip():
            diagnostics.append("Historical correction note is active.")

    if missing:
        missing_count = len(missing)
        missing_label = "item" if missing_count == 1 else "items"
        missing_verb = "is" if missing_count == 1 else "are"
        diagnostics.append(f"{missing_count} required configuration {missing_label} {missing_verb} currently missing.")

    diagnostics_html = (
        "<ul class=\"mb-0\">" + "".join(f"<li>{html_lib.escape(item)}</li>" for item in diagnostics) + "</ul>"
        if diagnostics
        else "<p class=\"text-secondary mb-0\">No active diagnostics warnings.</p>"
    )

    html = render_page(
        templates_dir,
        "status.html",
        title="Status",
        generated_at=generated_at,
        status_title=html_lib.escape(status_title),
        status_class=css_class(status_title),
        status_detail=html_lib.escape(status_detail),
        overview_generated_at=fmt_status_time(generated_at),
        overview_cards=overview_cards_html,
        source_rows=source_rows_html,
        diagnostics_source_note=diagnostics_source_note,
        diagnostics_source_rows=diagnostics_source_rows_html,
        publication_fix=publication_fix,
        publication_state=publication_state,
        publication_updated=publication_updated,
        publication_reason=publication_reason,
        publication_selection_context=publication_selection_context,
        diagnostics_list=diagnostics_html,
        missing_list=missing_html,
    )
    write_text(site_dir / "status" / "index.html", html)


def publish_archive(site_dir: Path, templates_dir: Path, generated_at: str, days: List[str]) -> None:
    if days:
        rows = "\n".join(
            "<tr>"
            f"<td><a href=\"/fix/{d}/\">{d}</a></td>"
            f"<td><a href=\"/fix/{d}.json\">JSON</a></td>"
            "</tr>"
            for d in sorted(days, reverse=True)
        )
    else:
        rows = '<tr><td colspan="2" class="text-secondary">No published references yet.</td></tr>'

    html = render_page(
        templates_dir,
        "archive.html",
        title="Archive",
        generated_at=generated_at,
        archive_rows=rows,
    )
    write_text(site_dir / "archive" / "index.html", html)


def publish_home(site_dir: Path, templates_dir: Path, generated_at: str, latest: Dict[str, Any]) -> None:
    def humanize_withhold_reason(reason: Any) -> Optional[str]:
        if not isinstance(reason, str):
            return None
        text = reason.strip().lower()
        if not text:
            return None
        if "no valid source" in text:
            return "Insufficient valid sources"
        if "dispersion" in text:
            return "High dispersion across sources"
        if "stale" in text or "invalid/stale" in text or "outside observation window" in text:
            return "Stale or invalid source inputs"
        if "missing secret" in text or "missing secrets" in text or "config needed" in text:
            return "Configuration needed"
        if "no existing published data" in text:
            return "No published daily reference available"
        if "no intraday" in text:
            return "No intraday samples in publication window"
        if "no valid attempts" in text:
            return "No valid intraday samples in publication window"
        return "Publication withheld by methodology checks"

    c = latest.get("computed", {})
    fix = c.get("fix")
    p25 = c.get("band", {}).get("p25")
    p75 = c.get("band", {}).get("p75")
    status = str(c.get("status", "N/A"))
    status_upper = status.strip().upper()
    withheld = bool(c.get("withheld", True))
    reasons = c.get("withhold_reasons", [])

    reasons_html = ""
    if withheld and reasons:
        reasons_html = '<ul class="mb-0">' + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>"

    benchmark_map = latest.get("benchmarks", {})
    if not isinstance(benchmark_map, dict):
        benchmark_map = c.get("benchmarks", {})
    benchmark_map = benchmark_map if isinstance(benchmark_map, dict) else {}

    stale_since_pattern = re.compile(r"stale since ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z)", re.IGNORECASE)

    def benchmark_entry(key: str) -> Dict[str, Any]:
        entry = benchmark_map.get(key, {})
        return entry if isinstance(entry, dict) else {}

    def benchmark_unavailable_context(key: str) -> Optional[str]:
        entry = benchmark_entry(key)
        if not entry:
            return None
        if bool(entry.get("available")):
            return None

        source_notes = entry.get("source_notes", {})
        stale_since: Optional[dt.datetime] = None
        if isinstance(source_notes, dict):
            for note in source_notes.values():
                if not isinstance(note, str):
                    continue
                match = stale_since_pattern.search(note)
                if not match:
                    continue
                parsed = try_parse_datetime(match.group(1))
                if parsed is None:
                    continue
                if stale_since is None or parsed < stale_since:
                    stale_since = parsed
        if stale_since is not None:
            return f"Awaiting fresh quote (stale since {stale_since.strftime('%b %d, %Y')})"

        reasons = entry.get("withhold_reasons", [])
        if isinstance(reasons, list):
            lowered = [str(reason).strip().lower() for reason in reasons if str(reason).strip()]
            if any("no valid source" in reason for reason in lowered):
                return "No valid quotes in current publication window"
            if any("stale" in reason for reason in lowered):
                return "Awaiting fresh source quotes"
            if any("config" in reason or "secret" in reason for reason in lowered):
                return "Source configuration required"
        return None

    def benchmark_value_number(key: str) -> Optional[float]:
        entry = benchmark_map.get(key, {})
        if not isinstance(entry, dict):
            return None
        value = parse_number(entry.get("value"))
        if value is None:
            value = parse_number(entry.get("fix"))
        return value

    def benchmark_value_or_unavailable(key: str) -> str:
        value = benchmark_value_number(key)
        return f"{fmt_rate(value)} IRR" if value is not None else "Unavailable"

    def benchmark_value_main(key: str) -> str:
        value = benchmark_value_number(key)
        return fmt_rate(value) if value is not None else "Unavailable"

    indicator_map = latest.get("indicators", {})
    if not isinstance(indicator_map, dict):
        indicator_map = c.get("indicators", {})
    indicator_map = indicator_map if isinstance(indicator_map, dict) else {}

    published_street_fix = parse_number(c.get("fix"))
    if withheld:
        published_street_fix = None

    street = published_street_fix
    official = benchmark_value_number("official")
    transfer = benchmark_value_number("regional_transfer")
    crypto = benchmark_value_number("crypto_usdt")
    emami_coin = benchmark_value_number("emami_gold_coin")
    gold_usd_per_oz = resolve_gold_usd_per_oz(latest)
    gold_implied_fx = compute_gold_implied_fx(emami_coin, gold_usd_per_oz)
    fallback_indicator_values: Dict[str, Optional[float]] = {
        "street_official_gap_pct": (
            ((street - official) / official) * 100
            if street is not None and official not in (None, 0)
            else None
        ),
        "street_transfer_gap_pct": (
            ((street - transfer) / transfer) * 100
            if street is not None and transfer not in (None, 0)
            else None
        ),
        "street_crypto_gap_pct": (
            ((crypto - street) / street) * 100
            if crypto is not None and street not in (None, 0)
            else None
        ),
        "street_gold_gap_pct": (
            ((gold_implied_fx - street) / street) * 100
            if gold_implied_fx is not None and street not in (None, 0)
            else None
        ),
        "official_commercial_trend_7d": None,
    }

    latest_day = parse_iso_date_text(latest.get("date"))
    fallback_indicator_values["official_commercial_trend_7d"] = (
        compute_official_trend_7d(site_dir, latest_day, official) if latest_day is not None else None
    )
    if not indicator_map:
        indicator_map = {k: {"value": v} for k, v in fallback_indicator_values.items()}

    street_gap_keys = {"street_official_gap_pct", "street_transfer_gap_pct", "street_crypto_gap_pct", "street_gold_gap_pct"}

    def indicator_value_number(key: str) -> Optional[float]:
        if key in street_gap_keys:
            # Always derive street-based gaps from the published primary fix to keep
            # homepage indicators aligned with the primary benchmark card.
            return fallback_indicator_values.get(key)
        entry = indicator_map.get(key, {})
        value = parse_float(entry.get("value")) if isinstance(entry, dict) else None
        if value is None:
            value = fallback_indicator_values.get(key)
        return value

    def indicator_value_or_unavailable(key: str) -> str:
        value = indicator_value_number(key)
        if value is None:
            if key == "official_commercial_trend_7d":
                return "History building"
            if key == "street_official_gap_pct":
                return "Awaiting official quote"
            return "Unavailable"
        return f"{value:+.1f}%"

    sources_map = latest.get("sources", {})
    if not isinstance(sources_map, dict):
        sources_map = {}

    def benchmark_as_of_text(key: str) -> str:
        latest_sample_ts: Optional[dt.datetime] = None
        for source_data in sources_map.values():
            if not isinstance(source_data, dict):
                continue
            sample_rows = source_data.get("samples", [])
            if not isinstance(sample_rows, list):
                continue
            for sample in sample_rows:
                if not isinstance(sample, dict):
                    continue
                bench_map = sample.get("benchmarks", {})
                if not isinstance(bench_map, dict):
                    continue
                if parse_number(bench_map.get(key)) is None:
                    continue
                sample_ts = try_parse_datetime(sample.get("sampled_at"))
                if sample_ts is None:
                    sample_ts = try_parse_datetime(sample.get("quote_time"))
                if sample_ts is None:
                    continue
                if latest_sample_ts is None or sample_ts > latest_sample_ts:
                    latest_sample_ts = sample_ts
        if latest_sample_ts is None:
            return ""
        return f"Updated {latest_sample_ts.strftime('%H:%M UTC')}"

    def benchmark_card_meta_text(key: str) -> str:
        entry = benchmark_entry(key)
        value = benchmark_value_number(key)
        if value is not None:
            if key == "official" and bool(entry.get("using_stale_fallback")):
                selected_quote_time = try_parse_datetime(entry.get("selected_quote_time"))
                if selected_quote_time is not None:
                    return f"Last known quote {selected_quote_time.strftime('%b %d, %Y')}"
                return "Last known quote"
            return benchmark_as_of_text(key)
        context = benchmark_unavailable_context(key)
        if context:
            return context
        return "No valid quote in publication window"

    def benchmark_sparkline_html(key: str) -> str:
        width = 160.0
        height = 32.0
        pad = 3.0
        baseline_y = height / 2.0

        if latest_day is None:
            return (
                f'<svg class="sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
                'role="img" aria-label="mini line chart">'
                f'<polyline fill="none" stroke="rgba(148,163,184,0.45)" stroke-width="1.5" '
                f'points="{pad:.1f},{baseline_y:.1f} {width-pad:.1f},{baseline_y:.1f}"></polyline>'
                "</svg>"
            )
        current = benchmark_value_number(key)
        if current is None:
            return (
                f'<svg class="sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
                'role="img" aria-label="mini line chart">'
                f'<polyline fill="none" stroke="rgba(148,163,184,0.45)" stroke-width="1.5" '
                f'points="{pad:.1f},{baseline_y:.1f} {width-pad:.1f},{baseline_y:.1f}"></polyline>'
                "</svg>"
            )

        history = load_benchmark_fix_history(site_dir, key)
        by_day: Dict[dt.date, float] = {day: fix for day, fix in history if fix > 0}
        by_day[latest_day] = current
        ordered_values = [fix for _day, fix in sorted(by_day.items())][-7:]

        if len(ordered_values) < 2:
            return (
                f'<svg class="sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
                'role="img" aria-label="mini line chart">'
                f'<polyline fill="none" stroke="rgba(148,163,184,0.45)" stroke-width="1.5" '
                f'points="{pad:.1f},{baseline_y:.1f} {width-pad:.1f},{baseline_y:.1f}"></polyline>'
                "</svg>"
            )

        min_val = min(ordered_values)
        max_val = max(ordered_values)
        span = max_val - min_val

        points: List[str] = []
        for idx, value in enumerate(ordered_values):
            x = pad + (idx * (width - 2 * pad) / (len(ordered_values) - 1))
            if span <= 0:
                y = baseline_y
            else:
                y = pad + ((max_val - value) / span) * (height - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)
        fill_points = f"{pad:.1f},{height - pad:.1f} {polyline} {width - pad:.1f},{height - pad:.1f}"
        return (
            f'<svg class="sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
            'role="img" aria-label="mini line chart">'
            f'<polygon points="{fill_points}" fill="rgba(59,130,246,0.12)"></polygon>'
            f'<polyline fill="none" stroke="#3b82f6" stroke-width="2.2" points="{polyline}"></polyline>'
            "</svg>"
        )

    def indicator_gap_sparkline_html(key: str) -> str:
        width = 220.0
        height = 44.0
        pad = 3.0

        current = indicator_value_number(key)
        if latest_day is None or current is None:
            unavailable_text = "History building"
            if key == "street_official_gap_pct":
                unavailable_text = benchmark_unavailable_context("official") or "Awaiting official quote"
            return f'<div class="text-secondary small">{html_lib.escape(unavailable_text)}</div>'

        history = load_indicator_gap_history(site_dir, key)
        by_day: Dict[dt.date, float] = {day: value for day, value in history}
        by_day[latest_day] = current
        ordered_values = [value for _day, value in sorted(by_day.items())][-14:]
        if len(ordered_values) < 2:
            return '<div class="text-secondary small">History building</div>'

        min_val = min(ordered_values)
        max_val = max(ordered_values)
        span = max_val - min_val
        if span <= 0:
            span = 1.0

        points: List[str] = []
        for idx, value in enumerate(ordered_values):
            x = pad + (idx * (width - 2 * pad) / (len(ordered_values) - 1))
            y = pad + ((max_val - value) / span) * (height - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)
        fill_points = f"{pad:.1f},{height - pad:.1f} {polyline} {width - pad:.1f},{height - pad:.1f}"
        return (
            f'<svg class="indicator-sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
            'role="img" aria-label="gap trend sparkline">'
            f'<polygon points="{fill_points}" fill="rgba(59,130,246,0.18)"></polygon>'
            f'<polyline fill="none" stroke="#60a5fa" stroke-width="2.4" points="{polyline}"></polyline>'
            "</svg>"
        )

    def spread_value_or_unavailable(base: Optional[float], peer: Optional[float]) -> str:
        if base is None or peer is None:
            if peer is None:
                official_context = benchmark_unavailable_context("official")
                if official_context:
                    return "Awaiting official quote"
            return "Unavailable"
        return f"{base - peer:+,.0f} IRR"

    def derived_spread_sparkline_html(peer_key: str, current_spread: Optional[float]) -> str:
        width = 220.0
        height = 44.0
        pad = 3.0

        if latest_day is None or current_spread is None:
            if peer_key == "official":
                context = benchmark_unavailable_context("official") or "Awaiting official quote"
                return f'<div class="text-secondary small">{html_lib.escape(context)}</div>'
            return '<div class="text-secondary small">History building</div>'

        street_history = load_benchmark_fix_history(site_dir, "open_market")
        peer_history = load_benchmark_fix_history(site_dir, peer_key)
        if not street_history or not peer_history:
            return '<div class="text-secondary small">History building</div>'

        street_by_day = {day: value for day, value in street_history}
        peer_by_day = {day: value for day, value in peer_history}
        common_days = sorted(set(street_by_day) & set(peer_by_day))
        if not common_days:
            return '<div class="text-secondary small">History building</div>'

        by_day: Dict[dt.date, float] = {}
        for day in common_days:
            spread = street_by_day.get(day, 0.0) - peer_by_day.get(day, 0.0)
            by_day[day] = spread
        by_day[latest_day] = current_spread

        ordered_values = [value for _day, value in sorted(by_day.items())][-14:]
        if len(ordered_values) < 2:
            return '<div class="text-secondary small">History building</div>'

        min_val = min(ordered_values)
        max_val = max(ordered_values)
        span = max_val - min_val
        if span <= 0:
            span = 1.0

        points: List[str] = []
        for idx, value in enumerate(ordered_values):
            x = pad + (idx * (width - 2 * pad) / (len(ordered_values) - 1))
            y = pad + ((max_val - value) / span) * (height - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)
        fill_points = f"{pad:.1f},{height - pad:.1f} {polyline} {width - pad:.1f},{height - pad:.1f}"
        return (
            f'<svg class="indicator-sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
            'role="img" aria-label="spread trend sparkline">'
            f'<polygon points="{fill_points}" fill="rgba(59,130,246,0.18)"></polygon>'
            f'<polyline fill="none" stroke="#60a5fa" stroke-width="2.4" points="{polyline}"></polyline>'
            "</svg>"
        )

    publication_meta = ""
    publication_selection = latest.get("publication_selection")
    same_as_previous_day = False
    previous_day_fix: Optional[float] = None
    delta_vs_previous_day: Optional[float] = None
    delta_pct_vs_previous_day: Optional[float] = None
    if isinstance(publication_selection, dict):
        selected_ts = try_parse_datetime(publication_selection.get("selected_collected_at"))
        selected_sample = selected_ts.strftime("%H:%M UTC") if selected_ts else None
        used_fallback = bool(publication_selection.get("used_fallback"))
        same_as_previous_day = bool(publication_selection.get("same_as_previous_day"))
        previous_day_fix = parse_number(publication_selection.get("previous_day_fix"))
        delta_vs_previous_day = parse_number(publication_selection.get("delta_vs_previous_day"))
        delta_pct_vs_previous_day = parse_float(publication_selection.get("delta_pct_vs_previous_day"))
        if selected_sample:
            if used_fallback:
                publication_meta = f"Observed sample: {selected_sample} (fallback from intraday window)"
            else:
                publication_meta = f"Observed sample: {selected_sample} (intraday window)"
    if not publication_meta:
        as_of_ts = try_parse_datetime(latest.get("as_of"))
        if as_of_ts:
            publication_meta = f"Observed sample: {as_of_ts.strftime('%H:%M UTC')} (intraday window)"
    observation_timestamp = "N/A"
    if isinstance(publication_selection, dict):
        selected_ts = try_parse_datetime(publication_selection.get("selected_collected_at"))
        if selected_ts is not None:
            observation_timestamp = selected_ts.strftime("%H:%M UTC")
    if observation_timestamp == "N/A":
        as_of_ts = try_parse_datetime(latest.get("as_of"))
        if as_of_ts is not None:
            observation_timestamp = as_of_ts.strftime("%H:%M UTC")
    primary_as_of_display = "N/A"
    as_of_raw = latest.get("as_of")
    as_of_ts = try_parse_datetime(as_of_raw)
    if as_of_ts is not None:
        primary_as_of_display = as_of_ts.strftime("%b %d, %Y %H:%M UTC")
    elif isinstance(as_of_raw, str) and as_of_raw.strip():
        primary_as_of_display = as_of_raw.strip()

    source_medians = c.get("source_medians")
    if isinstance(source_medians, dict):
        street_source_count = len([k for k, v in source_medians.items() if parse_number(v) is not None])
    else:
        street_source_count = 0
    street_source_count_text = f"{street_source_count} source" if street_source_count == 1 else f"{street_source_count} sources"
    publication_state = "Withheld" if withheld else "Published"

    prior_day_change_text = "History building"
    if delta_vs_previous_day is None or delta_pct_vs_previous_day is None:
        delta_meta = compute_previous_day_delta_metadata(site_dir, latest_day, street)
        same_as_previous_day = bool(delta_meta.get("same_as_previous_day"))
        previous_day_fix = parse_number(delta_meta.get("previous_day_fix"))
        delta_vs_previous_day = parse_number(delta_meta.get("delta_vs_previous_day"))
        delta_pct_vs_previous_day = parse_float(delta_meta.get("delta_pct_vs_previous_day"))
    if delta_vs_previous_day is not None and delta_pct_vs_previous_day is not None:
        prior_day_change_text = f"{delta_vs_previous_day:+,.0f} IRR ({delta_pct_vs_previous_day:+.1f}%)"

    publication_change_meta = ""
    if publication_state == "Published" and same_as_previous_day:
        publication_change_meta = "Freshly selected, unchanged vs prior day."
    elif publication_state == "Published" and previous_day_fix is not None:
        publication_change_meta = "Freshly selected from today's intraday samples."
    primary_publication_basis = publication_meta if publication_meta else "N/A"
    primary_selection_note = publication_change_meta if publication_change_meta else (
        "Withheld by methodology checks." if publication_state == "Withheld" else "N/A"
    )

    def comparison_value_text(value: Optional[float], unit_suffix: str = "IRR") -> str:
        if value is None:
            return "Unavailable"
        return f"{fmt_rate(value)} {unit_suffix}"

    withhold_reason_text = ""
    if withheld:
        normalized_reason = None
        if isinstance(reasons, list):
            for reason in reasons:
                normalized_reason = humanize_withhold_reason(reason)
                if normalized_reason:
                    break
        if normalized_reason:
            withhold_reason_text = normalized_reason

    if status_upper == "WITHHOLD":
        primary_value_html = (
            '<div class="text-warning fw-bold mb-1">WITHHELD</div>'
            '<div class="h5 mb-1">Candidate value</div>'
            f'<div class="primary-value-row mb-1"><div class="primary-rate-value primary-rate-value-candidate">{fmt_rate(fix)}</div>'
            '<div class="primary-rate-unit">IRR per USD</div></div>'
        )
        primary_reason_html = (
            f'<div class="text-warning small mt-1">Reason: {withhold_reason_text}</div>'
            if withhold_reason_text
            else ""
        )
    else:
        primary_value_html = (
            f'<div class="primary-value-row mb-1"><div class="primary-rate-value">{fmt_rate(fix)}</div>'
            '<div class="primary-rate-unit">IRR per USD</div></div>'
        )
        primary_reason_html = ""

    official_gap_available = indicator_value_number("street_official_gap_pct") is not None
    official_gap_note = (
        "Premium to official market"
        if official_gap_available
        else (benchmark_unavailable_context("official") or "Gap resumes when a fresh official quote is available")
    )
    official_spread_available = street is not None and official is not None
    official_spread_note = (
        "Street minus official (absolute spread)"
        if official_spread_available
        else (benchmark_unavailable_context("official") or "Spread resumes when a fresh official quote is available")
    )

    html = render_page(
        templates_dir,
        "index.html",
        title="USD/IRR Dashboard",
        generated_at=generated_at,
        date=latest.get("date", "N/A"),
        as_of=latest.get("as_of", "N/A"),
        fix=fmt_rate(fix),
        primary_value_html=primary_value_html,
        primary_reason_html=primary_reason_html,
        p25=fmt_rate(p25),
        p75=fmt_rate(p75),
        status=status,
        status_class=css_class(status),
        withheld="Yes" if withheld else "No",
        publication_state=publication_state,
        publication_meta=publication_meta,
        publication_change_meta=publication_change_meta,
        primary_prior_day_change=prior_day_change_text,
        primary_street_sources=street_source_count_text,
        primary_observation_timestamp=observation_timestamp,
        primary_as_of_display=primary_as_of_display,
        primary_publication_basis=primary_publication_basis,
        primary_selection_note=primary_selection_note,
        withhold_reason_short=withhold_reason_text,
        reasons=reasons_html,
        official_value=benchmark_value_or_unavailable("official"),
        official_value_main=benchmark_value_main("official"),
        official_as_of=benchmark_card_meta_text("official"),
        official_sparkline=benchmark_sparkline_html("official"),
        regional_transfer_value=benchmark_value_or_unavailable("regional_transfer"),
        regional_transfer_value_main=benchmark_value_main("regional_transfer"),
        regional_transfer_as_of=benchmark_card_meta_text("regional_transfer"),
        regional_transfer_sparkline=benchmark_sparkline_html("regional_transfer"),
        crypto_usdt_value=benchmark_value_or_unavailable("crypto_usdt"),
        crypto_usdt_value_main=benchmark_value_main("crypto_usdt"),
        crypto_usdt_as_of=benchmark_card_meta_text("crypto_usdt"),
        crypto_usdt_sparkline=benchmark_sparkline_html("crypto_usdt"),
        emami_gold_coin_value=benchmark_value_or_unavailable("emami_gold_coin"),
        emami_gold_coin_value_main=benchmark_value_main("emami_gold_coin"),
        emami_gold_coin_as_of=benchmark_card_meta_text("emami_gold_coin"),
        emami_gold_coin_sparkline=benchmark_sparkline_html("emami_gold_coin"),
        street_official_gap_value=indicator_value_or_unavailable("street_official_gap_pct"),
        street_official_gap_note=official_gap_note,
        street_official_street_value=comparison_value_text(street),
        street_official_peer_value=comparison_value_text(official),
        street_official_gap_sparkline=indicator_gap_sparkline_html("street_official_gap_pct"),
        street_transfer_gap_value=indicator_value_or_unavailable("street_transfer_gap_pct"),
        street_transfer_gap_note="Premium/discount to transfer market",
        street_transfer_street_value=comparison_value_text(street),
        street_transfer_peer_value=comparison_value_text(transfer),
        street_transfer_gap_sparkline=indicator_gap_sparkline_html("street_transfer_gap_pct"),
        street_crypto_gap_value=indicator_value_or_unavailable("street_crypto_gap_pct"),
        street_crypto_gap_note="Premium/discount to crypto market",
        street_crypto_street_value=comparison_value_text(street),
        street_crypto_peer_value=comparison_value_text(crypto),
        street_crypto_gap_sparkline=indicator_gap_sparkline_html("street_crypto_gap_pct"),
        street_gold_gap_value=indicator_value_or_unavailable("street_gold_gap_pct"),
        street_gold_gap_note="Gold-derived FX signal (indicative)",
        street_gold_street_value=comparison_value_text(street),
        street_gold_peer_value=comparison_value_text(gold_implied_fx),
        street_gold_gap_sparkline=indicator_gap_sparkline_html("street_gold_gap_pct"),
        street_official_spread_value=spread_value_or_unavailable(street, official),
        street_official_spread_note=official_spread_note,
        street_official_spread_street_value=comparison_value_text(street),
        street_official_spread_peer_value=comparison_value_text(official),
        street_official_spread_sparkline=derived_spread_sparkline_html("official", parse_number(street - official) if street is not None and official is not None else None),
        street_transfer_spread_value=spread_value_or_unavailable(street, transfer),
        street_transfer_spread_note="Street minus transfer (absolute spread)",
        street_transfer_spread_street_value=comparison_value_text(street),
        street_transfer_spread_peer_value=comparison_value_text(transfer),
        street_transfer_spread_sparkline=derived_spread_sparkline_html("regional_transfer", parse_number(street - transfer) if street is not None and transfer is not None else None),
    )
    write_text(site_dir / "index.html", html)


def publish_daily_fix(site_dir: Path, templates_dir: Path, generated_at: str, daily: Dict[str, Any]) -> None:
    daily_out = json.loads(json.dumps(daily))
    enrich_publication_selection_metadata(site_dir, daily_out)
    daily_out["normalized_metrics"] = build_normalized_market_snapshot(daily_out, site_dir=site_dir)

    indicators = daily_out.get("indicators", {})
    if isinstance(indicators, dict):
        trend_entry = indicators.get("official_commercial_trend_7d", {})
        if isinstance(trend_entry, dict):
            trend_entry["value"] = daily_out["normalized_metrics"].get("official_commercial_trend_7d")
            trend_entry["available"] = trend_entry.get("value") is not None
    computed_indicators = daily_out.get("computed", {}).get("indicators", {})
    if isinstance(computed_indicators, dict):
        trend_entry = computed_indicators.get("official_commercial_trend_7d", {})
        if isinstance(trend_entry, dict):
            trend_entry["value"] = daily_out["normalized_metrics"].get("official_commercial_trend_7d")
            trend_entry["available"] = trend_entry.get("value") is not None

    day = daily_out["date"]
    c = daily_out.get("computed", {})
    reasons = c.get("withhold_reasons", [])

    reasons_html = ""
    if reasons:
        reasons_html = '<ul class="mb-0">' + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>"

    html = render_page(
        templates_dir,
        "fix.html",
        title=f"Reference {day}",
        generated_at=generated_at,
        date=day,
        as_of=daily_out.get("as_of", "N/A"),
        fix=fmt_rate(c.get("fix")),
        p25=fmt_rate(c.get("band", {}).get("p25")),
        p75=fmt_rate(c.get("band", {}).get("p75")),
        dispersion=(f"{c.get('dispersion', 0) * 100:.2f}%" if c.get("dispersion") is not None else "N/A"),
        status=c.get("status", "N/A"),
        status_class=css_class(str(c.get("status", "N/A"))),
        withheld="Yes" if c.get("withheld") else "No",
        reasons=reasons_html,
        source_rows=render_source_table(daily_out),
    )

    write_text(site_dir / "fix" / day / "index.html", html)
    write_json(site_dir / "fix" / f"{day}.json", daily_out)


def render_source_table(daily: Dict[str, Any]) -> str:
    rows: List[str] = []
    for source, data in daily.get("sources", {}).items():
        median_val = data.get("median")
        note = data.get("note") or ""
        sample_rows = data.get("samples", [])
        ok_count = sum(1 for s in sample_rows if s.get("ok"))
        total_count = len(sample_rows)
        rows.append(
            "<tr>"
            f"<td>{source}</td>"
            f"<td>{fmt_rate(median_val)}</td>"
            f"<td>{ok_count}/{total_count}</td>"
            f"<td>{note}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def publish_latest(site_dir: Path, daily: Dict[str, Any]) -> None:
    payload = json.loads(json.dumps(daily))
    enrich_publication_selection_metadata(site_dir, payload)
    benchmark_results = payload.get("benchmarks", {})
    if isinstance(benchmark_results, dict):
        new_indicators = compute_indicator_results(benchmark_results)
        payload["indicators"] = new_indicators
        if isinstance(payload.get("computed"), dict):
            payload["computed"]["indicators"] = json.loads(json.dumps(new_indicators))

    payload["normalized_metrics"] = build_normalized_market_snapshot(payload, site_dir=site_dir)

    indicators = payload.get("indicators", {})
    if isinstance(indicators, dict):
        trend_entry = indicators.get("official_commercial_trend_7d", {})
        if isinstance(trend_entry, dict):
            trend_entry["value"] = payload["normalized_metrics"].get("official_commercial_trend_7d")
            trend_entry["available"] = trend_entry.get("value") is not None
    computed_indicators = payload.get("computed", {}).get("indicators", {})
    if isinstance(computed_indicators, dict):
        trend_entry = computed_indicators.get("official_commercial_trend_7d", {})
        if isinstance(trend_entry, dict):
            trend_entry["value"] = payload["normalized_metrics"].get("official_commercial_trend_7d")
            trend_entry["available"] = trend_entry.get("value") is not None

    write_json(site_dir / "api" / "latest.json", payload)


def publish_series(site_dir: Path) -> None:
    rows = load_series_rows(site_dir)
    public_rows = [row for row in rows if is_public_series_row(row)]
    write_json(site_dir / "api" / "series.json", {"rows": public_rows})


def publish_benchmark_series(site_dir: Path) -> None:
    for benchmark_key in BENCHMARK_LABELS:
        rows = [
            {"date": day.isoformat(), "value": value}
            for day, value in reconstruct_benchmark_fix_history(site_dir, benchmark_key)
        ]
        write_json(benchmark_series_path(site_dir, benchmark_key), rows)


def publish_indicator_series(site_dir: Path) -> None:
    for indicator_key in (
        "street_official_gap_pct",
        "street_transfer_gap_pct",
        "street_crypto_gap_pct",
        "street_gold_gap_pct",
    ):
        rows = [
            {"date": day.isoformat(), "value": value}
            for day, value in reconstruct_indicator_gap_history(site_dir, indicator_key)
        ]
        write_json(indicator_series_path(site_dir, indicator_key), rows)

    official_history = reconstruct_benchmark_fix_history(site_dir, "official")
    official_by_day: Dict[dt.date, float] = {day: value for day, value in official_history}
    trend_rows: List[Dict[str, Any]] = []
    for day, value in official_history:
        trend = compute_official_trend_from_history_map(official_by_day, day, value)
        if trend is None:
            continue
        trend_rows.append({"date": day.isoformat(), "value": trend})
    write_json(indicator_series_path(site_dir, "official_commercial_trend_7d"), trend_rows)


def publish_public_series_artifacts(site_dir: Path) -> None:
    publish_series(site_dir)
    publish_benchmark_series(site_dir)
    publish_indicator_series(site_dir)


def publish_intraday_latest(site_dir: Path, day: dt.date) -> None:
    attempts = load_intraday_attempts(site_dir, day)
    latest_path = site_dir / "api" / "intraday" / "latest.json"
    if not attempts:
        return

    latest_attempt = attempts[-1]
    computed = latest_attempt.get("computed", {}) if isinstance(latest_attempt.get("computed"), dict) else {}
    computed_benchmarks = computed.get("benchmarks", {}) if isinstance(computed.get("benchmarks"), dict) else {}

    benchmarks_payload: Dict[str, Optional[float]] = {}
    for key in BENCHMARK_LABELS:
        entry = computed_benchmarks.get(key, {})
        if not isinstance(entry, dict):
            benchmarks_payload[key] = None
            continue
        benchmarks_payload[key] = parse_number(entry.get("value"))

    payload = {
        "date": latest_attempt.get("date"),
        "collected_at": latest_attempt.get("collected_at"),
        "window_utc": latest_attempt.get("window_utc"),
        "normalized_unit": "rial",
        "benchmarks": benchmarks_payload,
        "computed": {
            "fix": parse_number(computed.get("fix")),
            "status": computed.get("status"),
            "withheld": computed.get("withheld"),
            "band": computed.get("band"),
            "dispersion": computed.get("dispersion"),
        },
    }
    write_json(latest_path, payload)


def publish_mapping_audit(site_dir: Path) -> None:
    fix_dir = site_dir / "fix"
    current = current_mapping_fingerprint()
    stale_days: List[Dict[str, Any]] = []

    if fix_dir.exists():
        for path in sorted(fix_dir.glob("*.json")):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            methodology = payload.get("methodology", {})
            recorded = methodology.get("mapping_fingerprint") if isinstance(methodology, dict) else None
            if recorded != current:
                stale_days.append(
                    {
                        "date": path.stem,
                        "recorded_mapping_fingerprint": recorded,
                        "current_mapping_fingerprint": current,
                    }
                )

    write_json(
        site_dir / "api" / "mapping_audit.json",
        {
            "current_mapping_fingerprint": current,
            "stale_days": stale_days,
        },
    )


def intraday_day_dir(site_dir: Path, day: dt.date) -> Path:
    return site_dir / "intraday" / iso_date(day)


def unique_intraday_file(site_dir: Path, collected_at: dt.datetime) -> Path:
    day = collected_at.astimezone(UTC).date()
    day_dir = intraday_day_dir(site_dir, day)
    base = collected_at.astimezone(UTC).strftime("%H-%M-%S")
    candidate = day_dir / f"{base}.json"
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = day_dir / f"{base}-{suffix}.json"
        if not candidate.exists():
            return candidate
        suffix += 1


def serialize_sample(sample: Sample) -> Dict[str, Any]:
    health = sample.health if isinstance(sample.health, dict) else {}
    payload = {
        "sampled_at": iso_ts(sample.sampled_at),
        "value": sample.value,
        "benchmarks": sample.benchmark_values,
        "quote_time": iso_ts(sample.quote_time) if sample.quote_time else None,
        "ok": sample.ok,
        "stale": sample.stale,
        "error": sample.error,
        "health": health,
        "fetch_success": health.get("fetch_success"),
        "validation_result": health.get("validation_result"),
        "failure_reason": health.get("failure_reason"),
        "source_unit": sample.source_unit,
        "normalized_unit": sample.normalized_unit,
    }
    if sample.source == "bonbast":
        payload["bonbast"] = {
            "bonbast_usd_buy": health.get("bonbast_usd_buy"),
            "bonbast_usd_sell": health.get("bonbast_usd_sell"),
            "bonbast_usd_mid": health.get("bonbast_usd_mid"),
            "source_unit": health.get("source_unit"),
            "normalized_unit": health.get("normalized_unit"),
            "fetch_mode": health.get("fetch_mode"),
            "selector_used": health.get("selector_used"),
            "fetch_success": health.get("fetch_success"),
            "failure_reason": health.get("failure_reason"),
        }
    return payload


def collect_one_attempt(
    source_configs: List[SourceConfig],
    sampled_at: dt.datetime,
    day: dt.date,
    allow_outside_window: bool,
) -> Dict[str, List[Sample]]:
    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    window_end_dt = dt.datetime.combine(day, WINDOW_END, tzinfo=UTC)
    samples: Dict[str, List[Sample]] = {}
    for cfg in source_configs:
        sample = fetch_one(cfg, sampled_at, window_start_dt, window_end_dt)
        if not allow_outside_window and not is_within_window_minute(sampled_at, window_start_dt, window_end_dt):
            sample.ok = False
            sample.stale = True
            sample.error = "sample outside observation window"
        samples[cfg.name] = [sample]
    apply_bonbast_peer_validation(
        samples=samples,
        max_deviation_pct=env_pct("BONBAST_PEER_DEVIATION_PCT", BONBAST_PEER_DEVIATION_PCT_DEFAULT),
    )
    return samples


def write_intraday_attempt(site_dir: Path, attempt: Dict[str, Any]) -> Path:
    collected_at = try_parse_datetime(attempt.get("collected_at"))
    if collected_at is None:
        raise PipelineError("intraday attempt missing collected_at timestamp")
    path = unique_intraday_file(site_dir, collected_at)
    write_json(path, attempt)
    return path


def apply_legacy_navasan_unit_repair(
    source: str,
    benchmark_values: Dict[str, Any],
    source_unit: str,
    health: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    if source != "navasan" or source_unit != "unknown" or not isinstance(benchmark_values, dict):
        return benchmark_values, source_unit

    repaired = dict(benchmark_values)
    changed = False
    for key, unit in NAVASAN_BENCHMARK_UNITS.items():
        raw_num = parse_number(benchmark_values.get(key))
        if raw_num is None:
            continue
        normalized = normalize_unit(raw_num, unit)
        if normalized != raw_num:
            changed = True
        repaired[key] = normalized

    if changed:
        health["legacy_unit_repair"] = "applied_navasan_symbol_unit_map"
        health["benchmark_units"] = dict(NAVASAN_BENCHMARK_UNITS)
        return repaired, "mixed"
    return benchmark_values, source_unit


def parse_sample_record(source: str, payload: Dict[str, Any]) -> Optional[Sample]:
    sampled_at = try_parse_datetime(payload.get("sampled_at"))
    if sampled_at is None:
        return None
    quote_time_raw = payload.get("quote_time")
    quote_time = try_parse_datetime(quote_time_raw) if quote_time_raw is not None else None
    benchmark_values = payload.get("benchmarks") if isinstance(payload.get("benchmarks"), dict) else {}
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    if "fetch_success" in payload and "fetch_success" not in health:
        health["fetch_success"] = payload.get("fetch_success")
    if "validation_result" in payload and "validation_result" not in health:
        health["validation_result"] = payload.get("validation_result")
    if "failure_reason" in payload and "failure_reason" not in health:
        health["failure_reason"] = payload.get("failure_reason")
    source_unit = str(payload.get("source_unit") or "unknown")
    normalized_unit = str(payload.get("normalized_unit") or "rial")
    benchmark_values, source_unit = apply_legacy_navasan_unit_repair(source, benchmark_values, source_unit, health)
    value = parse_number(payload.get("value"))
    if source == "navasan" and source_unit == "unknown" and value is not None:
        # Old navasan payloads stored only `value` without unit metadata.
        # Treat plausible legacy street values as toman and normalize to rial.
        if 50_000 <= value <= 500_000:
            value = normalize_unit(value, "toman")
            health["legacy_unit_repair"] = "assumed_navasan_unknown_value_toman"
            source_unit = "mixed"
    if source == "navasan" and value is not None and source_unit == "mixed":
        # Legacy navasan samples without explicit unit stored `value` as toman street quote.
        # New mixed-unit payloads include benchmark_units metadata and are already normalized.
        has_benchmark_units = isinstance(health.get("benchmark_units"), dict) and bool(health.get("benchmark_units"))
        legacy_repair_mode = str(health.get("legacy_unit_repair") or "")
        if (
            legacy_repair_mode != "assumed_navasan_unknown_value_toman"
            and (not has_benchmark_units or legacy_repair_mode == "applied_navasan_symbol_unit_map")
        ):
            value = normalize_unit(value, NAVASAN_BENCHMARK_UNITS.get("open_market", "toman"))
    elif value is None:
        value = parse_number(benchmark_values.get(PRIMARY_BENCHMARK))

    return Sample(
        source=source,
        sampled_at=sampled_at,
        value=value,
        benchmark_values=benchmark_values,
        quote_time=quote_time,
        ok=bool(payload.get("ok")),
        stale=bool(payload.get("stale")),
        error=str(payload.get("error")) if payload.get("error") is not None else None,
        health=health,
        source_unit=source_unit,
        normalized_unit=normalized_unit,
    )


def load_intraday_attempts(site_dir: Path, day: dt.date) -> List[Dict[str, Any]]:
    day_dir = intraday_day_dir(site_dir, day)
    if not day_dir.exists():
        return []

    attempts: List[Tuple[dt.datetime, Dict[str, Any]]] = []
    for path in sorted(day_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        collected_at = try_parse_datetime(payload.get("collected_at"))
        if collected_at is None:
            continue

        payload["file"] = str(path.relative_to(site_dir))
        attempts.append((collected_at, payload))

    attempts.sort(key=lambda item: item[0])
    return [payload for _, payload in attempts]


def load_samples_from_daily_payload(daily: Dict[str, Any], source_configs: List[SourceConfig]) -> Dict[str, List[Sample]]:
    samples: Dict[str, List[Sample]] = {cfg.name: [] for cfg in source_configs}
    sources = daily.get("sources")
    if not isinstance(sources, dict):
        return samples

    for source, source_data in sources.items():
        source_key = canonical_source_name(str(source))
        if source_key not in samples:
            continue
        if not isinstance(source_data, dict):
            continue
        rows = source_data.get("samples")
        if not isinstance(rows, list):
            continue
        parsed_rows: List[Sample] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed = parse_sample_record(source_key, row)
            if parsed is not None:
                parsed_rows.append(parsed)
        samples[source_key].extend(parsed_rows)

    return samples


def compute_previous_day_delta_metadata(
    site_dir: Path,
    latest_day: Optional[dt.date],
    current_fix: Optional[float],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "same_as_previous_day": False,
        "previous_day_fix": None,
        "delta_vs_previous_day": None,
        "delta_pct_vs_previous_day": None,
    }
    if latest_day is None or current_fix is None:
        return metadata

    previous_rows: List[Tuple[dt.date, float]] = []
    for row in load_series_rows(site_dir):
        if not is_public_series_row(row):
            continue
        row_day = parse_iso_date_text(row.get("date"))
        row_fix = parse_number(row.get("fix"))
        if row_day is None or row_fix is None or row_day >= latest_day:
            continue
        previous_rows.append((row_day, row_fix))

    if not previous_rows:
        return metadata

    previous_rows.sort(key=lambda item: item[0])
    previous_fix = previous_rows[-1][1]
    if previous_fix <= 0:
        return metadata

    delta = float(current_fix) - float(previous_fix)
    delta_pct = (delta / float(previous_fix)) * 100.0
    metadata["same_as_previous_day"] = math.isclose(float(current_fix), float(previous_fix), abs_tol=1e-9)
    metadata["previous_day_fix"] = float(previous_fix)
    metadata["delta_vs_previous_day"] = float(delta)
    metadata["delta_pct_vs_previous_day"] = float(delta_pct)
    return metadata


def enrich_publication_selection_metadata(site_dir: Path, payload: Dict[str, Any]) -> None:
    selection = payload.get("publication_selection")
    if not isinstance(selection, dict):
        return

    computed = payload.get("computed", {}) if isinstance(payload.get("computed"), dict) else {}
    source_medians = computed.get("source_medians", {}) if isinstance(computed.get("source_medians"), dict) else {}
    if parse_number(selection.get("street_source_count_used")) is None:
        selection["street_source_count_used"] = len([k for k, v in source_medians.items() if parse_number(v) is not None])

    day = parse_iso_date_text(payload.get("date"))
    fix = parse_number(computed.get("fix"))
    delta_meta = compute_previous_day_delta_metadata(site_dir, day, fix)
    for key in ("same_as_previous_day", "previous_day_fix", "delta_vs_previous_day", "delta_pct_vs_previous_day"):
        if selection.get(key) is None:
            selection[key] = delta_meta.get(key)

    payload["publication_selection"] = selection


def refresh_existing_day_payload(
    site_dir: Path,
    templates_dir: Path,
    generated_at: str,
    day: dt.date,
    source_configs: List[SourceConfig],
    write_day_artifact: bool = True,
) -> Optional[Dict[str, Any]]:
    day_s = iso_date(day)
    day_json = site_dir / "fix" / f"{day_s}.json"
    if not day_json.exists():
        return None

    try:
        existing = json.loads(day_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    samples = load_samples_from_daily_payload(existing, source_configs)
    if not any(samples.values()):
        return None

    refreshed = summarize_day(samples, source_configs, day)
    existing_fix = parse_number(existing.get("computed", {}).get("fix") if isinstance(existing.get("computed"), dict) else None)
    refreshed_fix = parse_number(refreshed.get("computed", {}).get("fix") if isinstance(refreshed.get("computed"), dict) else None)
    has_legacy_sample_repairs = any(
        isinstance(sample.health, dict) and str(sample.health.get("legacy_unit_repair") or "").strip() != ""
        for rows in samples.values()
        for sample in rows
    )
    if refreshed_fix is None and (existing_fix is not None or has_legacy_sample_repairs):
        methodology = refreshed.get("methodology")
        if not isinstance(methodology, dict):
            methodology = {}
            refreshed["methodology"] = methodology
        methodology["rebuild_note"] = (
            "Recomputed from stored legacy samples with current mapping/validation; no valid benchmark inputs remained."
        )

    # Preserve original publication timestamp/selection metadata where available.
    as_of = existing.get("as_of")
    if isinstance(as_of, str) and as_of.strip():
        refreshed["as_of"] = as_of
    if isinstance(existing.get("publication_selection"), dict):
        refreshed["publication_selection"] = existing.get("publication_selection")

    if write_day_artifact:
        publish_daily_fix(site_dir, templates_dir, generated_at=generated_at, daily=refreshed)
    return refreshed


def attempt_to_samples(attempt: Dict[str, Any]) -> Dict[str, List[Sample]]:
    sources = attempt.get("sources")
    if not isinstance(sources, dict):
        return {}

    output: Dict[str, List[Sample]] = {}
    for source, data in sources.items():
        source_key = canonical_source_name(str(source))
        if not isinstance(data, dict):
            continue
        sample_data = data.get("sample")
        if not isinstance(sample_data, dict):
            continue
        parsed = parse_sample_record(source_key, sample_data)
        if parsed is None:
            continue
        output.setdefault(source_key, []).append(parsed)
    return output


def is_within_window_minute(ts: dt.datetime, start: dt.datetime, end: dt.datetime) -> bool:
    ts_minute = ts.replace(second=0, microsecond=0)
    return start <= ts_minute <= end


def in_publication_window(ts: dt.datetime, day: dt.date) -> bool:
    start = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    end = dt.datetime.combine(day, WINDOW_END, tzinfo=UTC)
    # Window checks are minute-based: include the full end minute (e.g. 14:15:59).
    return is_within_window_minute(ts, start, end)


def is_daily_valid(daily: Dict[str, Any]) -> bool:
    computed = daily.get("computed", {})
    status = computed.get("status")
    fix = computed.get("fix")
    withheld = computed.get("withheld")
    return (
        status in {"Green", "Amber", "Red"}
        and withheld is False
        and isinstance(fix, (int, float))
        and math.isfinite(float(fix))
        and float(fix) > 0
    )


def immutable_day_exists(site_dir: Path, day: str) -> bool:
    return (site_dir / "fix" / f"{day}.json").exists() or (site_dir / "fix" / day / "index.html").exists()


def build_placeholder_payload(day: dt.date, as_of: str, status: str, reason: str) -> Dict[str, Any]:
    placeholder_benchmarks = {
        key: {
            "label": BENCHMARK_LABELS[key],
            "benchmark": key,
            "fix": None,
            "band": {"p25": None, "p75": None},
            "dispersion": None,
            "status": "WITHHOLD",
            "withheld": True,
            "withhold_reasons": [reason],
            "source_medians": {},
            "source_notes": {},
            "available": False,
        }
        for key in BENCHMARK_LABELS
    }
    placeholder_indicators = {
        key: {
            "label": INDICATOR_LABELS[key],
            "value": None,
            "available": False,
            "formula": INDICATOR_FORMULAS[key],
        }
        for key in INDICATOR_LABELS
    }
    payload = {
        "date": iso_date(day),
        "as_of": as_of,
        "methodology": {
            "mapping_fingerprint": current_mapping_fingerprint(),
            "canonical_source_symbols": {
                source: {benchmark: list(symbols) for benchmark, symbols in mapping.items()}
                for source, mapping in CANONICAL_SOURCE_SYMBOLS.items()
            },
            "navasan_benchmark_units": dict(NAVASAN_BENCHMARK_UNITS),
        },
        "benchmarks": placeholder_benchmarks,
        "indicators": placeholder_indicators,
        "computed": {
            "fix": None,
            "band": {"p25": None, "p75": None},
            "dispersion": None,
            "status": status,
            "withheld": True,
            "withhold_reasons": [reason],
            "source_medians": {},
            "benchmarks": {
                key: {
                    "label": BENCHMARK_LABELS[key],
                    "value": None,
                    "available": False,
                    "is_primary": key == PRIMARY_BENCHMARK,
                }
                for key in BENCHMARK_LABELS
            },
            "indicators": placeholder_indicators,
        },
    }
    payload["normalized_metrics"] = build_normalized_market_snapshot(payload)
    return payload


def select_daily_from_intraday(
    site_dir: Path,
    day: dt.date,
    source_configs: List[SourceConfig],
    include_outside_window: bool = False,
    primary_allow_stale: bool = False,
) -> Optional[Dict[str, Any]]:
    attempts = load_intraday_attempts(site_dir, day)
    if not attempts:
        return None

    all_attempts: List[Tuple[dt.datetime, Dict[str, Any], Dict[str, Any], bool, bool]] = []
    for attempt in attempts:
        collected_at = try_parse_datetime(attempt.get("collected_at"))
        if collected_at is None:
            continue
        samples = attempt_to_samples(attempt)
        if not samples:
            continue
        daily = summarize_day(samples, source_configs, day, primary_allow_stale=primary_allow_stale)
        valid = is_daily_valid(daily)
        in_window = in_publication_window(collected_at, day)
        all_attempts.append((collected_at, attempt, daily, valid, in_window))

    if include_outside_window and primary_allow_stale:
        candidates: List[Tuple[dt.datetime, Dict[str, Any], Dict[str, Any], bool]] = [
            (collected_at, attempt, daily, valid)
            for collected_at, attempt, daily, valid, _in_window in all_attempts
        ]
        selection_scope = "latest_intraday_refresh"
    else:
        candidates = [
            (collected_at, attempt, daily, valid)
            for collected_at, attempt, daily, valid, in_window in all_attempts
            if in_window
        ]
        selection_scope = "publication_window"

    if not candidates and include_outside_window:
        candidates = [(collected_at, attempt, daily, valid) for collected_at, attempt, daily, valid, _ in all_attempts]
        selection_scope = "latest_intraday_fallback"

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    latest_attempt = candidates[-1]
    valid_candidates = [item for item in candidates if item[3]]

    if valid_candidates:
        selected = valid_candidates[-1]
        if selection_scope == "latest_intraday_refresh":
            selected_reason = "latest valid intraday attempt (current-day refresh)"
        elif selection_scope == "latest_intraday_fallback":
            selected_reason = "latest valid intraday attempt (outside-window fallback for current-day refresh)"
        else:
            selected_reason = "latest valid intraday attempt in publication window"
    else:
        selected = latest_attempt
        if selection_scope == "latest_intraday_refresh":
            selected_reason = "no valid attempts found; used latest intraday attempt (current-day refresh)"
        elif selection_scope == "latest_intraday_fallback":
            selected_reason = "no valid attempts found; used latest intraday attempt (outside-window fallback)"
        else:
            selected_reason = "no valid attempts found; used latest intraday attempt in publication window"

    selected_at, selected_attempt, daily, selected_valid = selected
    latest_at = latest_attempt[0]
    used_fallback = bool(selected_valid and selected_at != latest_at)
    if selected_valid and not used_fallback:
        if selection_scope == "latest_intraday_refresh":
            basis_label = "Selected from latest intraday refresh sample"
        elif selection_scope == "latest_intraday_fallback":
            basis_label = "Selected from latest intraday attempt (outside publication window)"
        else:
            basis_label = "Selected from intraday publication window"
    elif selected_valid and used_fallback:
        basis_label = "Fallback to most recent valid intraday sample"
    else:
        basis_label = "No valid intraday sample in publication window (WITHHOLD)"

    if selection_scope == "latest_intraday_refresh":
        selection_rule = "current-day refresh: latest valid intraday attempt across all available samples"
    else:
        selection_rule = "latest valid intraday attempt in publication window, else latest attempt"

    computed = daily.get("computed", {}) if isinstance(daily.get("computed"), dict) else {}
    source_medians = computed.get("source_medians", {}) if isinstance(computed.get("source_medians"), dict) else {}
    street_source_count_used = len([k for k, v in source_medians.items() if parse_number(v) is not None])
    current_fix = parse_number(computed.get("fix"))
    delta_meta = compute_previous_day_delta_metadata(site_dir, day, current_fix)

    daily["publication_selection"] = {
        "rule": selection_rule,
        "selection_scope": selection_scope,
        "window_utc": {"start": WINDOW_START.strftime("%H:%M"), "end": WINDOW_END.strftime("%H:%M")},
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "selected_collected_at": iso_ts(selected_at),
        "latest_candidate_collected_at": iso_ts(latest_at),
        "selected_attempt_file": selected_attempt.get("file"),
        "selected_valid": selected_valid,
        "used_fallback": used_fallback,
        "basis_label": basis_label,
        "selection_reason": selected_reason,
        "street_source_count_used": street_source_count_used,
        "same_as_previous_day": delta_meta.get("same_as_previous_day", False),
        "previous_day_fix": delta_meta.get("previous_day_fix"),
        "delta_vs_previous_day": delta_meta.get("delta_vs_previous_day"),
        "delta_pct_vs_previous_day": delta_meta.get("delta_pct_vs_previous_day"),
    }
    return daily


def run_build_only(site_dir: Path, templates_dir: Path, generated_at: str, day: dt.date) -> int:
    source_configs = build_source_configs()
    intraday_selected = select_daily_from_intraday(
        site_dir,
        day,
        source_configs,
        include_outside_window=True,
        primary_allow_stale=(day == utc_now().date()),
    )
    refreshed: Optional[Dict[str, Any]] = None
    if intraday_selected is not None and is_daily_valid(intraday_selected):
        publish_daily_fix(site_dir, templates_dir, generated_at=generated_at, daily=intraday_selected)
        refreshed = intraday_selected
    else:
        refreshed = refresh_existing_day_payload(site_dir, templates_dir, generated_at, day, source_configs)

    latest_path = site_dir / "api" / "latest.json"
    if refreshed is not None:
        latest = refreshed
        status_title = "OK"
        if intraday_selected is not None and is_daily_valid(intraday_selected):
            status_detail = (
                f"Build-only mode: refreshed {iso_date(day)} from current-day intraday sample selection using current benchmark logic."
            )
        else:
            status_detail = (
                f"Build-only mode: refreshed {iso_date(day)} published artifact using current benchmark logic."
            )
    elif latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        status_title = "OK"
        status_detail = "Build-only mode: reused existing published data and did not create a new daily reference."
    else:
        latest = build_placeholder_payload(day, generated_at, "CONFIG NEEDED", "no existing published data")
        status_title = "CONFIG NEEDED"
        status_detail = "No existing published data found in build-only mode."

    publish_status(
        site_dir,
        templates_dir,
        generated_at,
        status_title=status_title,
        status_detail=status_detail,
        missing=None,
        latest=latest,
    )
    publish_latest(site_dir, latest)
    publish_public_series_artifacts(site_dir)
    publish_intraday_latest(site_dir, day)
    publish_home(site_dir, templates_dir, generated_at, latest)
    publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
    publish_mapping_audit(site_dir)
    return 0


def run_collect_intraday(args: argparse.Namespace, site_dir: Path, templates_dir: Path, generated_at: str) -> int:
    day = utc_now().date()
    missing = missing_secrets()
    if missing:
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="CONFIG NEEDED",
            status_detail="Required API secrets are missing. Intraday collection skipped.",
            missing=missing,
        )
        return 0

    source_configs = build_source_configs()
    sampled_at = utc_now()
    samples = collect_one_attempt(
        source_configs=source_configs,
        sampled_at=sampled_at,
        day=day,
        allow_outside_window=args.allow_outside_window,
    )
    summary = summarize_day(samples, source_configs, day)
    attempt_payload = {
        "date": iso_date(day),
        "collected_at": iso_ts(sampled_at),
        "window_utc": {"start": WINDOW_START.strftime("%H:%M"), "end": WINDOW_END.strftime("%H:%M")},
        "sources": {
            source: {
                "sample": serialize_sample(entries[0]) if entries else None,
                "health": (entries[0].health if entries and isinstance(entries[0].health, dict) else {}),
            }
            for source, entries in samples.items()
        },
        "computed": summary.get("computed", {}),
    }
    path = write_intraday_attempt(site_dir, attempt_payload)
    publish_intraday_latest(site_dir, day)

    publish_status(
        site_dir,
        templates_dir,
        generated_at,
        status_title="OK",
        status_detail=f"Stored intraday collection attempt at {attempt_payload['collected_at']} UTC in {path.as_posix()}.",
        latest=summary,
    )
    return 0


def run_publish_daily(args: argparse.Namespace, site_dir: Path, templates_dir: Path, generated_at: str, day: dt.date) -> int:
    day_s = iso_date(day)
    source_configs = build_source_configs()
    if immutable_day_exists(site_dir, day_s):
        refreshed = refresh_existing_day_payload(
            site_dir,
            templates_dir,
            generated_at,
            day,
            source_configs,
            write_day_artifact=False,
        )
        if refreshed is not None:
            latest = refreshed
            status_detail = (
                f"Reference for {day_s} remains immutable; refreshed latest/status/home views using current logic."
            )
        else:
            latest_path = site_dir / "api" / "latest.json"
            latest = (
                json.loads(latest_path.read_text(encoding="utf-8"))
                if latest_path.exists()
                else build_placeholder_payload(day, generated_at, "CONFIG NEEDED", "no existing published data")
            )
            status_detail = f"Reference for {day_s} already exists and was not modified."
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="IMMUTABLE",
            status_detail=status_detail,
            missing=None,
            latest=latest,
        )
        publish_latest(site_dir, latest)
        publish_public_series_artifacts(site_dir)
        publish_intraday_latest(site_dir, day)
        publish_home(site_dir, templates_dir, generated_at, latest)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_mapping_audit(site_dir)
        return 0

    missing = missing_secrets()
    if missing:
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="CONFIG NEEDED",
            status_detail="Required API secrets are missing. No daily rate has been published.",
            missing=missing,
        )
        placeholder = build_placeholder_payload(day, generated_at, "CONFIG NEEDED", "missing secrets")
        publish_latest(site_dir, placeholder)
        publish_public_series_artifacts(site_dir)
        publish_intraday_latest(site_dir, day)
        publish_home(site_dir, templates_dir, generated_at, placeholder)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_mapping_audit(site_dir)
        return 0

    daily = select_daily_from_intraday(site_dir, day, source_configs)
    if daily is None:
        placeholder = build_placeholder_payload(
            day,
            generated_at,
            "WITHHOLD",
            "no intraday samples available in publication window",
        )
        publish_daily_fix(site_dir, templates_dir, generated_at=iso_ts(utc_now()), daily=placeholder)
        publish_latest(site_dir, placeholder)
        publish_public_series_artifacts(site_dir)
        publish_intraday_latest(site_dir, day)
        publish_mapping_audit(site_dir)
        publish_status(
            site_dir,
            templates_dir,
            generated_at=iso_ts(utc_now()),
            status_title="WITHHOLD",
            status_detail="No intraday collection attempts found in publication window; published WITHHOLD daily snapshot.",
            latest=placeholder,
        )
        publish_home(site_dir, templates_dir, generated_at=iso_ts(utc_now()), latest=placeholder)
        publish_archive(site_dir, templates_dir, generated_at=iso_ts(utc_now()), days=load_existing_days(site_dir))
        return 0

    publish_daily_fix(site_dir, templates_dir, generated_at=iso_ts(utc_now()), daily=daily)
    publish_latest(site_dir, daily)
    publish_public_series_artifacts(site_dir)
    publish_intraday_latest(site_dir, day)
    publish_mapping_audit(site_dir)
    publish_status(
        site_dir,
        templates_dir,
        generated_at=iso_ts(utc_now()),
        status_title="OK",
        status_detail=f"Published {day_s} daily reference from intraday selection at {PUBLISH_AT.strftime('%H:%M')} UTC.",
        latest=daily,
    )
    publish_home(site_dir, templates_dir, generated_at=iso_ts(utc_now()), latest=daily)
    publish_archive(site_dir, templates_dir, generated_at=iso_ts(utc_now()), days=load_existing_days(site_dir))
    return 0


def run(args: argparse.Namespace) -> int:
    site_dir = Path(args.site_dir)
    templates_dir = Path(args.templates_dir)
    assets_dir = Path(args.assets_dir)
    sample_times = parse_sample_times(args.sample_times_utc)

    site_dir.mkdir(parents=True, exist_ok=True)
    copy_static_assets(assets_dir, site_dir)

    now = utc_now()
    day = now.date()
    generated_at = iso_ts(now)

    publish_methodology(site_dir, templates_dir, generated_at)
    publish_governance(site_dir, templates_dir, generated_at)

    if args.rebuild_day:
        rebuild_day = parse_iso_date_text(args.rebuild_day)
        if rebuild_day is None:
            raise PipelineError(f"invalid --rebuild-day value: {args.rebuild_day}")
        return run_build_only(site_dir, templates_dir, generated_at, rebuild_day)

    if args.no_new_reference:
        return run_build_only(site_dir, templates_dir, generated_at, day)

    if args.mode == "collect-intraday":
        return run_collect_intraday(args, site_dir, templates_dir, generated_at)

    if args.mode == "publish-daily":
        publish_dt = dt.datetime.combine(day, PUBLISH_AT, tzinfo=UTC)
        should_sleep_until(publish_dt, skip_waits=args.skip_waits)
        return run_publish_daily(args, site_dir, templates_dir, generated_at, day)

    # Legacy mode: collect sample_times for today and publish at PUBLISH_AT.
    missing = missing_secrets()
    if missing:
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="CONFIG NEEDED",
            status_detail="Required API secrets are missing. No daily rate has been published.",
            missing=missing,
        )
        placeholder = build_placeholder_payload(day, generated_at, "CONFIG NEEDED", "missing secrets")
        publish_latest(site_dir, placeholder)
        publish_public_series_artifacts(site_dir)
        publish_intraday_latest(site_dir, day)
        publish_home(site_dir, templates_dir, generated_at, placeholder)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_mapping_audit(site_dir)
        return 0

    source_configs = build_source_configs()
    day_s = iso_date(day)

    if immutable_day_exists(site_dir, day_s):
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="IMMUTABLE",
            status_detail=f"Reference for {day_s} already exists and was not modified.",
            missing=None,
        )
        latest_path = site_dir / "api" / "latest.json"
        if latest_path.exists():
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            publish_public_series_artifacts(site_dir)
            publish_intraday_latest(site_dir, day)
            publish_home(site_dir, templates_dir, generated_at, latest)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_mapping_audit(site_dir)
        return 0

    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    if now < window_start_dt:
        should_sleep_until(window_start_dt, skip_waits=args.skip_waits)

    samples = collect_samples(
        source_configs,
        day,
        sample_times=sample_times,
        skip_waits=args.skip_waits,
        allow_outside_window=args.allow_outside_window,
    )

    publish_dt = dt.datetime.combine(day, PUBLISH_AT, tzinfo=UTC)
    should_sleep_until(publish_dt, skip_waits=args.skip_waits)

    daily = summarize_day(samples, source_configs, day)
    computed = daily.get("computed", {}) if isinstance(daily.get("computed"), dict) else {}
    source_medians = computed.get("source_medians", {}) if isinstance(computed.get("source_medians"), dict) else {}
    street_source_count_used = len([k for k, v in source_medians.items() if parse_number(v) is not None])
    current_fix = parse_number(computed.get("fix"))
    delta_meta = compute_previous_day_delta_metadata(site_dir, day, current_fix)
    latest_sampled_at: Optional[dt.datetime] = None
    for entries in samples.values():
        for sample in entries:
            if latest_sampled_at is None or sample.sampled_at > latest_sampled_at:
                latest_sampled_at = sample.sampled_at
    daily["publication_selection"] = {
        "rule": "legacy full-mode collection: summarize all configured sample times for the day",
        "sample_times_utc": [t.strftime("%H:%M") for t in sample_times],
        "selected_collected_at": iso_ts(latest_sampled_at) if latest_sampled_at else None,
        "basis_label": "Selected from intraday publication window",
        "used_fallback": False,
        "selected_valid": is_daily_valid(daily),
        "selection_reason": "legacy mode aggregates all configured collection times",
        "candidate_count": len(sample_times),
        "valid_candidate_count": len(sample_times) if is_daily_valid(daily) else 0,
        "street_source_count_used": street_source_count_used,
        "same_as_previous_day": delta_meta.get("same_as_previous_day", False),
        "previous_day_fix": delta_meta.get("previous_day_fix"),
        "delta_vs_previous_day": delta_meta.get("delta_vs_previous_day"),
        "delta_pct_vs_previous_day": delta_meta.get("delta_pct_vs_previous_day"),
    }
    publish_daily_fix(site_dir, templates_dir, generated_at=iso_ts(utc_now()), daily=daily)
    publish_latest(site_dir, daily)
    publish_public_series_artifacts(site_dir)
    publish_intraday_latest(site_dir, day)
    publish_mapping_audit(site_dir)

    publish_status(
        site_dir,
        templates_dir,
        generated_at=iso_ts(utc_now()),
        status_title="OK",
        status_detail=f"Published {day_s} reference at scheduled time {PUBLISH_AT.strftime('%H:%M')} UTC.",
        latest=daily,
    )
    publish_home(site_dir, templates_dir, generated_at=iso_ts(utc_now()), latest=daily)
    publish_archive(site_dir, templates_dir, generated_at=iso_ts(utc_now()), days=load_existing_days(site_dir))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USD/IRR Open Market Reference pipeline")
    parser.add_argument(
        "--mode",
        choices=("full", "collect-intraday", "publish-daily"),
        default="full",
        help="Pipeline mode: legacy full collection+publish, intraday collection, or daily publication from intraday data",
    )
    parser.add_argument("--site-dir", default="site", help="Output directory for static site")
    parser.add_argument("--templates-dir", default="templates", help="Template directory")
    parser.add_argument("--assets-dir", default="assets", help="Static assets copied to /site/assets")
    parser.add_argument(
        "--sample-times-utc",
        default=",".join(DEFAULT_INTRADAY_SAMPLE_TIMES),
        help="Comma-separated UTC HH:MM collection times used in legacy full mode (example: 13:45,14:00,14:15)",
    )
    parser.add_argument(
        "--skip-waits",
        action="store_true",
        help="Skip sleeping between sample/publish checkpoints (for local verification)",
    )
    parser.add_argument(
        "--allow-outside-window",
        action="store_true",
        help="Allow sampling outside 13:45-14:15 UTC (for local verification only)",
    )
    parser.add_argument(
        "--no-new-reference",
        action="store_true",
        help="Render/rebuild site using existing data only, without generating a new daily reference",
    )
    parser.add_argument(
        "--rebuild-day",
        default="",
        help="Rebuild a specific YYYY-MM-DD daily artifact from stored samples using current logic",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
