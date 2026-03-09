#!/usr/bin/env python3
"""USD/IRR Open Market Reference pipeline.

Samples configured sources within the observation window,
computes the daily reference, and renders a static site under /site.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
SAMPLE_OFFSETS_MIN = (0, 15, 30)  # 13:45, 14:00, 14:15 UTC

REQUIRED_SECRETS = (
    "BONBAST_USERNAME",
    "BONBAST_HASH",
    "NAVASAN_API_KEY",
    "ALANCHAND_API_KEY",
)

BENCHMARK_LABELS: Dict[str, str] = {
    "open_market": "Open Market / Street Rate",
    "nima": "NIMA Rate",
    "official": "MOB USD Benchmark",
    "regional_transfer": "Regional Transfer Rate",
    "crypto_usdt": "Crypto Dollar (USDT)",
    "emami_gold_coin": "Emami Gold Coin",
}

PRIMARY_BENCHMARK = "open_market"

INDICATOR_LABELS: Dict[str, str] = {
    "street_nima_gap": "Street-NIMA Gap",
    "crypto_premium": "Crypto Premium",
}

STRICT_CANONICAL_BENCHMARKS = {"nima", "official", "regional_transfer"}

BENCHMARK_SYMBOL_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "crypto_usdt": ("usdt", "tether", "usd_tether"),
    # Prefer exact "sekkeh" where exposed; keep existing aliases as fallback.
    "emami_gold_coin": ("sekkeh", "sekke", "emami", "coin_emami", "sekeh_emami"),
}

# Exact source-to-symbol mappings for production-safe parsing.
# If a strict benchmark has no entry for a source, we intentionally return unavailable.
CANONICAL_SOURCE_SYMBOLS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "navasan": {
        "official": ("mob_usd",),
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


@dataclass
class SourceConfig:
    name: str
    url: str
    auth_mode: str  # query_user_hash | query_api_key | header_api_key
    secret_fields: Tuple[str, ...]
    benchmark_families: Tuple[str, ...]


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


class PipelineError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def parse_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        cleaned = cleaned.replace(" ", "")
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

    # Heuristic: some feeds use toman; normalize to rial when values are clearly too low.
    if 1_000 <= v < 100_000:
        return v * 10.0
    return v


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
        has_nima = "nima" in path_l
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
        elif benchmark == "nima":
            if not has_nima:
                continue
            if has_usd:
                score += 4
            if has_irr:
                score += 4
            score += 3
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

    open_market = resolve("open_market")
    nima = resolve("nima")
    official = resolve("official")
    regional_transfer = resolve("regional_transfer")
    crypto_usdt = resolve("crypto_usdt")
    emami_gold_coin = resolve("emami_gold_coin")

    return {
        "open_market": open_market,
        "nima": nima,
        "official": official,
        "regional_transfer": regional_transfer,
        "crypto_usdt": crypto_usdt,
        "emami_gold_coin": emami_gold_coin,
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
            url=env_or_default("BONBAST_API_URL", "https://api.bonbast.com/v1/rates"),
            auth_mode="query_user_hash",
            secret_fields=("BONBAST_USERNAME", "BONBAST_HASH"),
            benchmark_families=("open_market", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
        ),
        SourceConfig(
            name="navasan",
            url=env_or_default("NAVASAN_API_URL", "https://api.navasan.tech/latest/"),
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
            benchmark_families=("open_market", "nima", "official", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
        ),
        SourceConfig(
            name="alanchand",
            url=env_or_default("ALANCHAND_API_URL", "https://api.alanchand.com/v1/rates"),
            auth_mode="header_api_key",
            secret_fields=("ALANCHAND_API_KEY",),
            benchmark_families=("open_market", "nima", "official", "regional_transfer", "crypto_usdt", "emami_gold_coin"),
        ),
    ]


def missing_secrets() -> List[str]:
    missing = []
    for key in REQUIRED_SECRETS:
        if not os.environ.get(key):
            missing.append(key)
    return missing


def build_request(config: SourceConfig) -> urllib.request.Request:
    url = config.url
    headers = {
        "Accept": "application/json",
        "User-Agent": "rialwatch-pipeline/0.2",
    }

    if config.auth_mode == "query_user_hash":
        query = {
            "username": os.environ["BONBAST_USERNAME"],
            "hash": os.environ["BONBAST_HASH"],
        }
        url = with_query(url, query)
    elif config.auth_mode == "query_api_key":
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


def fetch_one(config: SourceConfig, sampled_at: dt.datetime, window_start_dt: dt.datetime, window_end_dt: dt.datetime) -> Sample:
    try:
        req = build_request(config)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
    except KeyError as exc:
        return Sample(config.name, sampled_at, None, {}, None, ok=False, stale=False, error=f"missing secret: {exc}")
    except urllib.error.HTTPError as exc:
        return Sample(config.name, sampled_at, None, {}, None, ok=False, stale=False, error=f"http {exc.code}")
    except urllib.error.URLError as exc:
        return Sample(config.name, sampled_at, None, {}, None, ok=False, stale=False, error=f"network: {exc.reason}")
    except TimeoutError:
        return Sample(config.name, sampled_at, None, {}, None, ok=False, stale=False, error="timeout")
    except json.JSONDecodeError:
        return Sample(config.name, sampled_at, None, {}, None, ok=False, stale=False, error="invalid json")

    benchmark_values = extract_benchmark_values(payload, config.name)
    value = benchmark_values.get(PRIMARY_BENCHMARK)
    quote_time = extract_quote_time(payload)

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
    )


def collect_samples(
    source_configs: List[SourceConfig],
    day: dt.date,
    skip_waits: bool,
    allow_outside_window: bool,
) -> Dict[str, List[Sample]]:
    samples: Dict[str, List[Sample]] = {cfg.name: [] for cfg in source_configs}
    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    window_end_dt = dt.datetime.combine(day, WINDOW_END, tzinfo=UTC)

    for offset in SAMPLE_OFFSETS_MIN:
        target = window_start_dt + dt.timedelta(minutes=offset)
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

    return samples


def compute_benchmark_result(
    samples: Dict[str, List[Sample]],
    benchmark_key: str,
    benchmark_sources: Dict[str, Tuple[str, ...]],
) -> Dict[str, Any]:
    source_medians: Dict[str, float] = {}
    source_notes: Dict[str, str] = {}
    invalid_or_stale = False

    for source, entries in samples.items():
        families = benchmark_sources.get(source, ())
        if benchmark_key not in families:
            source_notes[source] = "source family not used for this benchmark"
            continue

        values = [s.benchmark_values.get(benchmark_key) for s in entries if s.benchmark_values.get(benchmark_key) is not None]
        if not values:
            source_notes[source] = "no valid samples"
            continue

        source_medians[source] = median(values)
        if any(s.stale for s in entries):
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
        "source_notes": source_notes,
        "available": (fix_value is not None and not withheld),
    }


def compute_indicator_results(benchmark_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    street = benchmark_results.get("open_market", {})
    nima = benchmark_results.get("nima", {})
    crypto = benchmark_results.get("crypto_usdt", {})

    street_fix = parse_number(street.get("fix"))
    nima_fix = parse_number(nima.get("fix"))
    crypto_fix = parse_number(crypto.get("fix"))

    street_nima_gap: Optional[float] = None
    if street.get("available") and nima.get("available") and nima_fix not in (None, 0):
        street_nima_gap = ((street_fix - nima_fix) / nima_fix) * 100 if street_fix is not None else None

    crypto_premium: Optional[float] = None
    if street.get("available") and crypto.get("available") and street_fix not in (None, 0):
        crypto_premium = ((crypto_fix - street_fix) / street_fix) * 100 if crypto_fix is not None else None

    return {
        "street_nima_gap": {
            "label": INDICATOR_LABELS["street_nima_gap"],
            "value": street_nima_gap,
            "available": street_nima_gap is not None,
            "formula": "((street_rate - nima_rate) / nima_rate) * 100",
        },
        "crypto_premium": {
            "label": INDICATOR_LABELS["crypto_premium"],
            "value": crypto_premium,
            "available": crypto_premium is not None,
            "formula": "((crypto_usdt_rate - street_rate) / street_rate) * 100",
        },
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

    return {
        "date": iso_date(day),
        "as_of": iso_ts(utc_now()),
        "window_utc": {
            "start": WINDOW_START.strftime("%H:%M"),
            "end": WINDOW_END.strftime("%H:%M"),
            "sample_count_per_source": 3,
        },
        "sources": {
            source: {
                "samples": [
                    {
                        "sampled_at": iso_ts(s.sampled_at),
                        "value": s.value,
                        "benchmarks": s.benchmark_values,
                        "quote_time": iso_ts(s.quote_time) if s.quote_time else None,
                        "ok": s.ok,
                        "stale": s.stale,
                        "error": s.error,
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
            "benchmarks": computed_benchmarks,
            "indicators": indicator_results,
        },
    }


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
) -> None:
    missing_html = ""
    if missing:
        missing_html = "<ul>" + "".join(f"<li><code>{m}</code></li>" for m in missing) + "</ul>"

    html = render_page(
        templates_dir,
        "status.html",
        title="Status",
        generated_at=generated_at,
        status_title=status_title,
        status_class=css_class(status_title),
        status_detail=status_detail,
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
    c = latest.get("computed", {})
    fix = c.get("fix")
    p25 = c.get("band", {}).get("p25")
    p75 = c.get("band", {}).get("p75")
    status = str(c.get("status", "N/A"))
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

    indicator_map = latest.get("indicators", {})
    if not isinstance(indicator_map, dict):
        indicator_map = c.get("indicators", {})
    indicator_map = indicator_map if isinstance(indicator_map, dict) else {}

    if not indicator_map:
        street = benchmark_value_number("open_market")
        nima = benchmark_value_number("nima")
        crypto = benchmark_value_number("crypto_usdt")
        street_nima_gap = ((street - nima) / nima) * 100 if street is not None and nima not in (None, 0) else None
        crypto_premium = ((crypto - street) / street) * 100 if crypto is not None and street not in (None, 0) else None
        indicator_map = {
            "street_nima_gap": {"value": street_nima_gap},
            "crypto_premium": {"value": crypto_premium},
        }

    def indicator_value_or_unavailable(key: str) -> str:
        entry = indicator_map.get(key, {})
        if not isinstance(entry, dict):
            return "Unavailable"
        value = parse_float(entry.get("value"))
        if value is None:
            return "Unavailable"
        return f"{value:+.1f}%"

    html = render_page(
        templates_dir,
        "index.html",
        title="USD/IRR Dashboard",
        generated_at=generated_at,
        date=latest.get("date", "N/A"),
        as_of=latest.get("as_of", "N/A"),
        fix=fmt_rate(fix),
        p25=fmt_rate(p25),
        p75=fmt_rate(p75),
        status=status,
        status_class=css_class(status),
        withheld="Yes" if withheld else "No",
        reasons=reasons_html,
        nima_value=benchmark_value_or_unavailable("nima"),
        official_value=benchmark_value_or_unavailable("official"),
        regional_transfer_value=benchmark_value_or_unavailable("regional_transfer"),
        crypto_usdt_value=benchmark_value_or_unavailable("crypto_usdt"),
        emami_gold_coin_value=benchmark_value_or_unavailable("emami_gold_coin"),
        street_nima_gap_value=indicator_value_or_unavailable("street_nima_gap"),
        crypto_premium_value=indicator_value_or_unavailable("crypto_premium"),
    )
    write_text(site_dir / "index.html", html)


def publish_daily_fix(site_dir: Path, templates_dir: Path, generated_at: str, daily: Dict[str, Any]) -> None:
    day = daily["date"]
    c = daily.get("computed", {})
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
        as_of=daily.get("as_of", "N/A"),
        fix=fmt_rate(c.get("fix")),
        p25=fmt_rate(c.get("band", {}).get("p25")),
        p75=fmt_rate(c.get("band", {}).get("p75")),
        dispersion=(f"{c.get('dispersion', 0) * 100:.2f}%" if c.get("dispersion") is not None else "N/A"),
        status=c.get("status", "N/A"),
        status_class=css_class(str(c.get("status", "N/A"))),
        withheld="Yes" if c.get("withheld") else "No",
        reasons=reasons_html,
        source_rows=render_source_table(daily),
    )

    write_text(site_dir / "fix" / day / "index.html", html)
    write_json(site_dir / "fix" / f"{day}.json", daily)


def render_source_table(daily: Dict[str, Any]) -> str:
    rows: List[str] = []
    for source, data in daily.get("sources", {}).items():
        median_val = data.get("median")
        note = data.get("note") or ""
        ok_count = sum(1 for s in data.get("samples", []) if s.get("ok"))
        rows.append(
            "<tr>"
            f"<td>{source}</td>"
            f"<td>{fmt_rate(median_val)}</td>"
            f"<td>{ok_count}/3</td>"
            f"<td>{note}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def publish_latest(site_dir: Path, daily: Dict[str, Any]) -> None:
    write_json(site_dir / "api" / "latest.json", daily)


def publish_series(site_dir: Path) -> None:
    rows = load_series_rows(site_dir)
    public_rows = [row for row in rows if is_public_series_row(row)]
    write_json(site_dir / "api" / "series.json", {"rows": public_rows})


def immutable_day_exists(site_dir: Path, day: str) -> bool:
    return (site_dir / "fix" / f"{day}.json").exists() or (site_dir / "fix" / day / "index.html").exists()


def run(args: argparse.Namespace) -> int:
    site_dir = Path(args.site_dir)
    templates_dir = Path(args.templates_dir)
    assets_dir = Path(args.assets_dir)

    site_dir.mkdir(parents=True, exist_ok=True)
    copy_static_assets(assets_dir, site_dir)

    now = utc_now()
    day = now.date()
    generated_at = iso_ts(now)

    publish_methodology(site_dir, templates_dir, generated_at)
    publish_governance(site_dir, templates_dir, generated_at)

    if args.no_new_reference:
        latest_path = site_dir / "api" / "latest.json"
        if latest_path.exists():
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            status_title = "OK"
            status_detail = "Build-only mode: reused existing published data and did not create a new daily reference."
        else:
            empty_benchmarks = {
                key: {
                    "label": BENCHMARK_LABELS[key],
                    "benchmark": key,
                    "fix": None,
                    "band": {"p25": None, "p75": None},
                    "dispersion": None,
                    "status": "WITHHOLD",
                    "withheld": True,
                    "withhold_reasons": ["no existing published data"],
                    "source_medians": {},
                    "source_notes": {},
                    "available": False,
                }
                for key in BENCHMARK_LABELS
            }
            empty_indicators = {
                key: {
                    "label": INDICATOR_LABELS[key],
                    "value": None,
                    "available": False,
                    "formula": (
                        "((street_rate - nima_rate) / nima_rate) * 100"
                        if key == "street_nima_gap"
                        else "((crypto_usdt_rate - street_rate) / street_rate) * 100"
                    ),
                }
                for key in INDICATOR_LABELS
            }
            latest = {
                "date": iso_date(day),
                "as_of": generated_at,
                "benchmarks": empty_benchmarks,
                "indicators": empty_indicators,
                "computed": {
                    "fix": None,
                    "band": {"p25": None, "p75": None},
                    "dispersion": None,
                    "status": "CONFIG NEEDED",
                    "withheld": True,
                    "withhold_reasons": ["no existing published data"],
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
                    "indicators": empty_indicators,
                },
            }
            status_title = "CONFIG NEEDED"
            status_detail = "No existing published data found in build-only mode."

        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title=status_title,
            status_detail=status_detail,
            missing=None,
        )
        publish_home(site_dir, templates_dir, generated_at, latest)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_series(site_dir)
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

        placeholder_benchmarks = {
            key: {
                "label": BENCHMARK_LABELS[key],
                "benchmark": key,
                "fix": None,
                "band": {"p25": None, "p75": None},
                "dispersion": None,
                "status": "WITHHOLD",
                "withheld": True,
                "withhold_reasons": ["missing secrets"],
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
                "formula": (
                    "((street_rate - nima_rate) / nima_rate) * 100"
                    if key == "street_nima_gap"
                    else "((crypto_usdt_rate - street_rate) / street_rate) * 100"
                ),
            }
            for key in INDICATOR_LABELS
        }
        placeholder = {
            "date": iso_date(day),
            "as_of": generated_at,
            "benchmarks": placeholder_benchmarks,
            "indicators": placeholder_indicators,
            "computed": {
                "fix": None,
                "band": {"p25": None, "p75": None},
                "dispersion": None,
                "status": "CONFIG NEEDED",
                "withheld": True,
                "withhold_reasons": ["missing secrets"],
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
        publish_home(site_dir, templates_dir, generated_at, placeholder)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        publish_latest(site_dir, placeholder)
        publish_series(site_dir)
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
        return 0

    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    if now < window_start_dt:
        should_sleep_until(window_start_dt, skip_waits=args.skip_waits)

    samples = collect_samples(
        source_configs,
        day,
        skip_waits=args.skip_waits,
        allow_outside_window=args.allow_outside_window,
    )

    publish_dt = dt.datetime.combine(day, PUBLISH_AT, tzinfo=UTC)
    should_sleep_until(publish_dt, skip_waits=args.skip_waits)

    daily = summarize_day(samples, source_configs, day)
    publish_daily_fix(site_dir, templates_dir, generated_at=iso_ts(utc_now()), daily=daily)
    publish_latest(site_dir, daily)
    publish_series(site_dir)

    publish_status(
        site_dir,
        templates_dir,
        generated_at=iso_ts(utc_now()),
        status_title="OK",
        status_detail=f"Published {day_s} reference at scheduled time {PUBLISH_AT.strftime('%H:%M')} UTC.",
    )

    publish_home(site_dir, templates_dir, generated_at=iso_ts(utc_now()), latest=daily)
    publish_archive(site_dir, templates_dir, generated_at=iso_ts(utc_now()), days=load_existing_days(site_dir))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USD/IRR Open Market Reference pipeline")
    parser.add_argument("--site-dir", default="site", help="Output directory for static site")
    parser.add_argument("--templates-dir", default="templates", help="Template directory")
    parser.add_argument("--assets-dir", default="assets", help="Static assets copied to /site/assets")
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
