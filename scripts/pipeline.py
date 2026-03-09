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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

UTC = dt.timezone.utc
WINDOW_START = dt.time(13, 45)
WINDOW_END = dt.time(14, 15)
PUBLISH_AT = dt.time(14, 20)
DEFAULT_INTRADAY_SAMPLE_TIMES = ("13:45", "14:00", "14:15")

REQUIRED_SECRETS = (
    "NAVASAN_API_KEY",
    "ALANCHAND_API_KEY",
)

BENCHMARK_LABELS: Dict[str, str] = {
    "open_market": "Open Market / Street Rate",
    "official": "Official Commercial USD Rate",
    "regional_transfer": "Regional Transfer Rate",
    "crypto_usdt": "Crypto Dollar (USDT)",
    "emami_gold_coin": "Emami Gold Coin",
}

PRIMARY_BENCHMARK = "open_market"

INDICATOR_LABELS: Dict[str, str] = {
    "street_official_gap_pct": "Street-Official Gap",
    "street_transfer_gap_pct": "Street-Transfer Gap",
    "street_crypto_gap_pct": "Street-Crypto Gap",
    "official_commercial_trend_7d": "Official 7d Trend",
}

INDICATOR_FORMULAS: Dict[str, str] = {
    "street_official_gap_pct": "((street_usd_irr - official_commercial_usd_irr) / official_commercial_usd_irr) * 100",
    "street_transfer_gap_pct": "((street_usd_irr - regional_transfer_usd_irr) / regional_transfer_usd_irr) * 100",
    "street_crypto_gap_pct": "((crypto_usdt_irr - street_usd_irr) / street_usd_irr) * 100",
    "official_commercial_trend_7d": "7-day percent change of official_commercial_usd_irr",
}

STRICT_CANONICAL_BENCHMARKS = {"official", "regional_transfer"}

BENCHMARK_SYMBOL_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "crypto_usdt": ("usdt", "tether", "usd_tether"),
    # Prefer exact "sekkeh" where exposed; keep existing aliases as fallback.
    "emami_gold_coin": ("sekkeh", "sekke", "emami", "coin_emami", "sekeh_emami"),
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


@dataclass
class SourceConfig:
    name: str
    url: str
    auth_mode: str  # browser_playwright | query_api_key | header_api_key
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
    for field in ("value", "price", "last", "close", "sell", "buy", "rate", "amount"):
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


def extract_benchmark_values(payload: Any, source_name: Optional[str] = None) -> Dict[str, Optional[float]]:
    source_key = (source_name or "").strip().lower()

    def resolve(benchmark: str) -> Optional[float]:
        canonical_map = CANONICAL_SOURCE_SYMBOLS.get(source_key, {})
        canonical_symbols = canonical_map.get(benchmark)
        if canonical_symbols:
            by_symbol = extract_value_by_symbol_candidates(payload, canonical_symbols)
            if by_symbol is not None:
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

    return {
        "open_market": resolve("open_market"),
        "official": resolve("official"),
        "regional_transfer": resolve("regional_transfer"),
        "crypto_usdt": resolve("crypto_usdt"),
        "emami_gold_coin": resolve("emami_gold_coin"),
    }


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
            name="navasan",
            url=env_or_default("NAVASAN_API_URL", "https://api.navasan.tech/latest/"),
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
            benchmark_families=("open_market", "official", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
            default_unit="toman",
        ),
        SourceConfig(
            name="alanchand",
            url=env_or_default("ALANCHAND_API_URL", "https://api.alanchand.com/v1/rates"),
            auth_mode="header_api_key",
            secret_fields=("ALANCHAND_API_KEY",),
            benchmark_families=("open_market", "official", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
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
    headers = {
        "Accept": "application/json",
        "User-Agent": "rialwatch-pipeline/0.2",
    }

    if config.auth_mode == "query_api_key":
        query = {"api_key": os.environ["NAVASAN_API_KEY"]}
        url = with_query(url, query)
    elif config.auth_mode == "header_api_key":
        key = os.environ["ALANCHAND_API_KEY"]
        headers["Authorization"] = f"Bearer {key}"
        headers["X-API-Key"] = key
        url = with_query(url, {"api_key": key})

    return urllib.request.Request(url=url, headers=headers, method="GET")


def with_query(url: str, extra_params: Dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    existing = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key, value in extra_params.items():
        existing[key] = [value]
    query = urllib.parse.urlencode(existing, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=query))


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
    stale = bool(sampled_at < window_start_dt or sampled_at > window_end_dt)
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


def fetch_one(config: SourceConfig, sampled_at: dt.datetime, window_start_dt: dt.datetime, window_end_dt: dt.datetime) -> Sample:
    if config.auth_mode == "browser_playwright":
        return fetch_bonbast_browser(config, sampled_at, window_start_dt, window_end_dt)
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
    }
    source_unit = config.default_unit
    normalized_unit = "rial"
    try:
        req = build_request(config)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            final_url = resp.geturl()
        health["content_length"] = len(body.encode("utf-8"))
        health["final_url"] = final_url
        payload, payload_mode = parse_source_payload(config.name, body)
        health["payload_mode"] = payload_mode
        health["page_load_ok"] = True
    except KeyError as exc:
        health["error_type"] = "missing_secret"
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

    raw_extracted_values = extract_benchmark_values(payload, config.name)
    benchmark_units: Dict[str, str] = {}
    canonical_map = CANONICAL_SOURCE_SYMBOLS.get(config.name, {})
    health["source_fields"] = {
        key: list(canonical_map.get(key, ()))
        for key in BENCHMARK_LABELS
        if canonical_map.get(key)
    }
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
    health["extracted_values"] = {k: benchmark_values.get(k) for k in benchmark_values}
    health["raw_extracted_values"] = {k: raw_extracted_values.get(k) for k in raw_extracted_values}
    health["benchmark_units"] = benchmark_units
    health["source_unit"] = source_unit
    health["normalized_unit"] = normalized_unit
    health["fetch_success"] = True

    stale = False
    if quote_time is not None and (quote_time < window_start_dt or quote_time > window_end_dt):
        stale = True

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
            if not allow_outside_window and (sampled_at < window_start_dt or sampled_at > window_end_dt):
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


def compute_benchmark_result(
    samples: Dict[str, List[Sample]],
    benchmark_key: str,
    benchmark_sources: Dict[str, Tuple[str, ...]],
) -> Dict[str, Any]:
    source_medians: Dict[str, float] = {}
    source_notes: Dict[str, str] = {}
    source_units: Dict[str, str] = {}
    invalid_or_stale = False

    for source, entries in samples.items():
        families = benchmark_sources.get(source, ())
        if benchmark_key not in families:
            source_notes[source] = "source family not used for this benchmark"
            continue

        if benchmark_key == PRIMARY_BENCHMARK:
            # Primary benchmark publication only considers valid in-window samples.
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
        if benchmark_key != PRIMARY_BENCHMARK and any(s.stale for s in entries):
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
        "source_notes": source_notes,
        "available": (fix_value is not None and not withheld),
    }


def compute_indicator_results(benchmark_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    street = benchmark_results.get("open_market", {})
    official = benchmark_results.get("official", {})
    transfer = benchmark_results.get("regional_transfer", {})
    crypto = benchmark_results.get("crypto_usdt", {})

    street_fix = parse_number(street.get("fix"))
    official_fix = parse_number(official.get("fix"))
    transfer_fix = parse_number(transfer.get("fix"))
    crypto_fix = parse_number(crypto.get("fix"))

    street_official_gap_pct: Optional[float] = None
    if street_fix is not None and official_fix not in (None, 0):
        street_official_gap_pct = ((street_fix - official_fix) / official_fix) * 100

    street_transfer_gap_pct: Optional[float] = None
    if street_fix is not None and transfer_fix not in (None, 0):
        street_transfer_gap_pct = ((street_fix - transfer_fix) / transfer_fix) * 100

    street_crypto_gap_pct: Optional[float] = None
    if street_fix not in (None, 0) and crypto_fix is not None:
        street_crypto_gap_pct = ((crypto_fix - street_fix) / street_fix) * 100

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


def load_benchmark_fix_history(site_dir: Path, benchmark_key: str) -> List[Tuple[dt.date, float]]:
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


def compute_official_trend_7d(site_dir: Path, latest_day: dt.date, latest_official_fix: Optional[float]) -> Optional[float]:
    if latest_official_fix is None or latest_official_fix <= 0:
        return None
    history = load_benchmark_fix_history(site_dir, "official")
    by_day: Dict[dt.date, float] = {day: fix for day, fix in history}
    by_day[latest_day] = latest_official_fix

    target_day = latest_day - dt.timedelta(days=7)
    base_candidates = [day for day in by_day if day <= target_day]
    if not base_candidates:
        return None
    base_day = max(base_candidates)
    base_fix = by_day.get(base_day)
    if base_fix is None or base_fix <= 0:
        return None
    return ((latest_official_fix - base_fix) / base_fix) * 100


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

    street = parse_number(benchmarks.get("open_market", {}).get("fix") if isinstance(benchmarks.get("open_market"), dict) else None)
    official = parse_number(benchmarks.get("official", {}).get("fix") if isinstance(benchmarks.get("official"), dict) else None)
    transfer = parse_number(
        benchmarks.get("regional_transfer", {}).get("fix") if isinstance(benchmarks.get("regional_transfer"), dict) else None
    )
    crypto = parse_number(benchmarks.get("crypto_usdt", {}).get("fix") if isinstance(benchmarks.get("crypto_usdt"), dict) else None)
    emami = parse_number(benchmarks.get("emami_gold_coin", {}).get("fix") if isinstance(benchmarks.get("emami_gold_coin"), dict) else None)

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
        "official_commercial_trend_7d": official_trend,
        "street_confidence_status": str(status) if status is not None else None,
        "source_unit": source_unit,
        "normalized_unit": "rial",
    }


def summarize_day(samples: Dict[str, List[Sample]], source_configs: List[SourceConfig], day: dt.date) -> Dict[str, Any]:
    benchmark_sources = {cfg.name: cfg.benchmark_families for cfg in source_configs}

    benchmark_results: Dict[str, Dict[str, Any]] = {}
    for key in BENCHMARK_LABELS:
        benchmark_results[key] = compute_benchmark_result(samples, key, benchmark_sources)

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
            "navasan": "Street Market Feed",
            "bonbast": "Street Market Feed (Secondary)",
            "alanchand": "Regional Market Feed",
        }
        key = source_name.strip().lower()
        return mapping.get(key, f"Market Feed {index}")

    def map_pipeline_state(value: str) -> str:
        upper = value.strip().upper()
        if upper in {"OK", "IMMUTABLE"}:
            return "Online"
        if upper == "WITHHOLD":
            return "Degraded"
        if upper == "CONFIG NEEDED":
            return "Offline"
        return "Unknown"

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
    sources = effective_latest.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}

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

    publication_fix = f"{fmt_rate(fix)} IRR" if fix is not None else "Unavailable"
    publication_updated = fmt_status_time(effective_latest.get("as_of") or generated_at)

    source_rows: List[Dict[str, str]] = []
    for idx, (source_name, source_data) in enumerate(sorted(sources.items(), key=lambda item: str(item[0]).lower()), start=1):
        if not isinstance(source_data, dict):
            continue
        samples = source_data.get("samples", [])
        if not isinstance(samples, list):
            samples = []

        latest_sample: Optional[Dict[str, Any]] = None
        latest_sample_ts: Optional[dt.datetime] = None
        last_success_ts: Optional[dt.datetime] = None
        any_fetch_success = False
        any_offline_failure = False
        any_degraded_failure = False
        any_parsed_data = False

        def sample_has_parsed_data(sample_obj: Dict[str, Any]) -> bool:
            if parse_number(sample_obj.get("value")) is not None:
                return True
            benchmarks_obj = sample_obj.get("benchmarks")
            if isinstance(benchmarks_obj, dict):
                for benchmark_value in benchmarks_obj.values():
                    if parse_number(benchmark_value) is not None:
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
                if latest_has_parsed_data and not latest_validation_failed and not latest_failure_text:
                    latest_status = "Online"
                else:
                    latest_status = "Degraded"
            elif latest_fetch_success is False:
                latest_status = "Offline"
            elif latest_error_text or latest_failure_text:
                latest_status = "Offline"

        if latest_status:
            source_status = latest_status
        elif any_fetch_success and any_parsed_data and not any_degraded_failure:
            source_status = "Online"
        elif any_fetch_success:
            source_status = "Degraded"
        elif any_offline_failure or samples:
            source_status = "Offline"

        note = humanize_source_note(source_data.get("note"))
        if source_status == "Offline" and latest_sample:
            latest_failure_reason = latest_sample.get("failure_reason")
            latest_sample_error = latest_sample.get("error")
            has_failure_text = isinstance(latest_failure_reason, str) and latest_failure_reason.strip()
            has_error_text = isinstance(latest_sample_error, str) and latest_sample_error.strip()
            if has_failure_text or has_error_text:
                note = "Source request failed."
        elif source_status == "Degraded":
            note = "Source responded, but returned invalid data."

        if not note and latest_sample and latest_sample.get("stale") is True:
            note = "Out-of-window observation."
        if not note and latest_sample:
            failure_reason = latest_sample.get("failure_reason")
            sample_error = latest_sample.get("error")
            if source_status == "Degraded":
                note = "Source responded, but returned invalid data."
            elif source_status == "Offline":
                if isinstance(failure_reason, str) and failure_reason.strip():
                    note = "Source request failed."
                elif isinstance(sample_error, str) and sample_error.strip():
                    note = "Source request failed."
        if not note:
            note = "Collecting normally." if source_status == "Online" else "No additional notes."

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
                    diagnostics.append(
                        f"{len(stale_days)} historical day(s) were generated under an older mapping version and are flagged for review."
                    )
        except json.JSONDecodeError:
            diagnostics.append("Methodology mapping audit data could not be read.")

    if degraded_count > 0:
        diagnostics.append("One or more market feeds are currently degraded.")
    if offline_count > 0:
        diagnostics.append("One or more market feeds are currently offline.")

    methodology = effective_latest.get("methodology", {})
    if isinstance(methodology, dict):
        rebuild_note = methodology.get("rebuild_note")
        if isinstance(rebuild_note, str) and rebuild_note.strip():
            diagnostics.append("A historical correction note is active for one or more records.")

    fix_dir = site_dir / "fix"
    withheld_days: List[str] = []
    if fix_dir.exists():
        for path in sorted(fix_dir.glob("*.json")):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            computed_row = payload.get("computed", {})
            if isinstance(computed_row, dict) and computed_row.get("withheld") is True:
                withheld_days.append(path.stem)
    if withheld_days:
        diagnostics.append(f"{len(withheld_days)} historical day(s) remain withheld due to publication-quality rules.")

    if missing:
        diagnostics.append("One or more required configuration items are currently missing.")

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
        publication_fix=publication_fix,
        publication_state=publication_state,
        publication_updated=publication_updated,
        publication_reason=publication_reason,
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

    street = benchmark_value_number("open_market")
    official = benchmark_value_number("official")
    transfer = benchmark_value_number("regional_transfer")
    crypto = benchmark_value_number("crypto_usdt")
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
        "official_commercial_trend_7d": None,
    }

    latest_day = parse_iso_date_text(latest.get("date"))
    fallback_indicator_values["official_commercial_trend_7d"] = (
        compute_official_trend_7d(site_dir, latest_day, official) if latest_day is not None else None
    )
    if not indicator_map:
        indicator_map = {k: {"value": v} for k, v in fallback_indicator_values.items()}

    def indicator_value_number(key: str) -> Optional[float]:
        entry = indicator_map.get(key, {})
        value = parse_float(entry.get("value")) if isinstance(entry, dict) else None
        if value is None:
            value = fallback_indicator_values.get(key)
        return value

    def indicator_value_or_unavailable(key: str) -> str:
        value = indicator_value_number(key)
        if value is None:
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

    def benchmark_sparkline_html(key: str) -> str:
        width = 160.0
        height = 28.0
        pad = 2.0
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
        return (
            f'<svg class="sparkline-svg" viewBox="0 0 {int(width)} {int(height)}" preserveAspectRatio="none" '
            'role="img" aria-label="mini line chart">'
            f'<polyline fill="none" stroke="#3b82f6" stroke-width="2" points="{polyline}"></polyline>'
            "</svg>"
        )

    publication_meta = ""
    publication_selection = latest.get("publication_selection")
    if isinstance(publication_selection, dict):
        selected_ts = try_parse_datetime(publication_selection.get("selected_collected_at"))
        selected_sample = selected_ts.strftime("%H:%M UTC") if selected_ts else None
        used_fallback = bool(publication_selection.get("used_fallback"))
        if selected_sample:
            if used_fallback:
                publication_meta = f"Observed sample: {selected_sample} (fallback from intraday window)"
            else:
                publication_meta = f"Observed sample: {selected_sample} (intraday window)"
    if not publication_meta:
        as_of_ts = try_parse_datetime(latest.get("as_of"))
        if as_of_ts:
            publication_meta = f"Observed sample: {as_of_ts.strftime('%H:%M UTC')} (intraday window)"

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
        candidate = f"{fmt_rate(fix)} IRR"
        primary_value_html = (
            '<div class="text-warning fw-bold mb-1">WITHHELD</div>'
            f'<div class="h5 mb-1">Candidate value: {candidate}</div>'
        )
        primary_reason_html = (
            f'<div class="text-warning small mt-1">Reason: {withhold_reason_text}</div>'
            if withhold_reason_text
            else ""
        )
    else:
        primary_value_html = f'<div class="h1 mb-1">{fmt_rate(fix)} IRR</div>'
        primary_reason_html = ""

    official_trend_note = "Direction of the official benchmark over the past week."
    publication_state = "Withheld" if withheld else "Published"

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
        withhold_reason_short=withhold_reason_text,
        reasons=reasons_html,
        official_value=benchmark_value_or_unavailable("official"),
        official_value_main=benchmark_value_main("official"),
        official_as_of=benchmark_as_of_text("official"),
        official_sparkline=benchmark_sparkline_html("official"),
        regional_transfer_value=benchmark_value_or_unavailable("regional_transfer"),
        regional_transfer_value_main=benchmark_value_main("regional_transfer"),
        regional_transfer_as_of=benchmark_as_of_text("regional_transfer"),
        regional_transfer_sparkline=benchmark_sparkline_html("regional_transfer"),
        crypto_usdt_value=benchmark_value_or_unavailable("crypto_usdt"),
        crypto_usdt_value_main=benchmark_value_main("crypto_usdt"),
        crypto_usdt_as_of=benchmark_as_of_text("crypto_usdt"),
        crypto_usdt_sparkline=benchmark_sparkline_html("crypto_usdt"),
        emami_gold_coin_value=benchmark_value_or_unavailable("emami_gold_coin"),
        emami_gold_coin_value_main=benchmark_value_main("emami_gold_coin"),
        emami_gold_coin_as_of=benchmark_as_of_text("emami_gold_coin"),
        emami_gold_coin_sparkline=benchmark_sparkline_html("emami_gold_coin"),
        street_official_gap_value=indicator_value_or_unavailable("street_official_gap_pct"),
        street_official_gap_note="Premium to official market",
        street_transfer_gap_value=indicator_value_or_unavailable("street_transfer_gap_pct"),
        street_transfer_gap_note="Premium/discount to transfer market",
        street_crypto_gap_value=indicator_value_or_unavailable("street_crypto_gap_pct"),
        street_crypto_gap_note="Premium/discount to crypto market",
        official_trend_7d_value=indicator_value_or_unavailable("official_commercial_trend_7d"),
        official_trend_7d_note=official_trend_note,
    )
    write_text(site_dir / "index.html", html)


def publish_daily_fix(site_dir: Path, templates_dir: Path, generated_at: str, daily: Dict[str, Any]) -> None:
    daily_out = json.loads(json.dumps(daily))
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
        if not allow_outside_window and (sampled_at < window_start_dt or sampled_at > window_end_dt):
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
        if source not in samples:
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
            parsed = parse_sample_record(source, row)
            if parsed is not None:
                parsed_rows.append(parsed)
        samples[source] = parsed_rows

    return samples


def refresh_existing_day_payload(
    site_dir: Path,
    templates_dir: Path,
    generated_at: str,
    day: dt.date,
    source_configs: List[SourceConfig],
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

    publish_daily_fix(site_dir, templates_dir, generated_at=generated_at, daily=refreshed)
    return refreshed


def attempt_to_samples(attempt: Dict[str, Any]) -> Dict[str, List[Sample]]:
    sources = attempt.get("sources")
    if not isinstance(sources, dict):
        return {}

    output: Dict[str, List[Sample]] = {}
    for source, data in sources.items():
        if not isinstance(data, dict):
            continue
        sample_data = data.get("sample")
        if not isinstance(sample_data, dict):
            continue
        parsed = parse_sample_record(source, sample_data)
        if parsed is None:
            continue
        output[source] = [parsed]
    return output


def in_publication_window(ts: dt.datetime, day: dt.date) -> bool:
    start = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    end = dt.datetime.combine(day, WINDOW_END, tzinfo=UTC)
    return start <= ts <= end


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
        daily = summarize_day(samples, source_configs, day)
        valid = is_daily_valid(daily)
        in_window = in_publication_window(collected_at, day)
        all_attempts.append((collected_at, attempt, daily, valid, in_window))

    candidates: List[Tuple[dt.datetime, Dict[str, Any], Dict[str, Any], bool]] = [
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
        if selection_scope == "latest_intraday_fallback":
            selected_reason = "latest valid intraday attempt (outside-window fallback for current-day refresh)"
        else:
            selected_reason = "latest valid intraday attempt in publication window"
    else:
        selected = latest_attempt
        if selection_scope == "latest_intraday_fallback":
            selected_reason = "no valid attempts found; used latest intraday attempt (outside-window fallback)"
        else:
            selected_reason = "no valid attempts found; used latest intraday attempt in publication window"

    selected_at, selected_attempt, daily, selected_valid = selected
    latest_at = latest_attempt[0]
    used_fallback = bool(selected_valid and selected_at != latest_at)
    if selected_valid and not used_fallback:
        if selection_scope == "latest_intraday_fallback":
            basis_label = "Selected from latest intraday attempt (outside publication window)"
        else:
            basis_label = "Selected from intraday publication window"
    elif selected_valid and used_fallback:
        basis_label = "Fallback to most recent valid intraday sample"
    else:
        basis_label = "No valid intraday sample in publication window (WITHHOLD)"

    daily["publication_selection"] = {
        "rule": "latest valid intraday attempt in publication window, else latest attempt",
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
    }
    return daily


def run_build_only(site_dir: Path, templates_dir: Path, generated_at: str, day: dt.date) -> int:
    source_configs = build_source_configs()
    intraday_selected = select_daily_from_intraday(site_dir, day, source_configs, include_outside_window=True)
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
    publish_home(site_dir, templates_dir, generated_at, latest)
    publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
    publish_series(site_dir)
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
            publish_home(site_dir, templates_dir, generated_at, latest)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_series(site_dir)
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
        publish_home(site_dir, templates_dir, generated_at, placeholder)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_latest(site_dir, placeholder)
        publish_series(site_dir)
        publish_mapping_audit(site_dir)
        return 0

    source_configs = build_source_configs()
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
        publish_series(site_dir)
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
    publish_series(site_dir)
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
        publish_home(site_dir, templates_dir, generated_at, placeholder)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_latest(site_dir, placeholder)
        publish_series(site_dir)
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
            publish_home(site_dir, templates_dir, generated_at, latest)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_series(site_dir)
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
    }
    publish_daily_fix(site_dir, templates_dir, generated_at=iso_ts(utc_now()), daily=daily)
    publish_latest(site_dir, daily)
    publish_series(site_dir)
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
