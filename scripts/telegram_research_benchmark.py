#!/usr/bin/env python3
"""Construct a Telegram-derived USD/IRR research benchmark.

Research mode only:
- Does not modify production pipeline
- Does not publish to site
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_WEIGHTS: Dict[str, float] = {
    "direct_shop": 1.00,
    "diaspora_shop": 0.90,
    "network_channel": 0.70,
    "market_channel": 0.50,
    "unclear": 0.30,
}

DIASPORA_CITIES = {"Dubai", "Istanbul", "Frankfurt", "Hamburg", "London"}
READY_STATUSES = {"ready_for_research_ingestion"}
READY_MONITOR_STATUSES = {"ready_for_research_ingestion", "monitor_only"}


@dataclass
class ChannelMeta:
    handle: str
    title: str
    source_priority: str
    channel_type_guess: str
    likely_individual_shop: bool
    ingestion_readiness_score: float
    recommended_status: str


@dataclass
class BenchmarkRecord:
    handle: str
    title: str
    source_priority: str
    recommended_status: str
    channel_type_guess: str
    likely_individual_shop: bool
    message_text_sample: str
    currency: str
    buy_quote: Optional[float]
    sell_quote: Optional[float]
    midpoint: Optional[float]
    buy_quote_rial: Optional[float]
    sell_quote_rial: Optional[float]
    midpoint_rial: Optional[float]
    raw_numeric_values: str
    city_guess: str
    quote_type_guess: str
    timestamp_text: str
    timestamp_iso: str
    freshness_score: float
    structure_score: float
    duplication_flag: str
    directness_score: float
    overall_record_quality_score: float
    inferred_unit_guess: str
    normalized_buy: Optional[float]
    normalized_sell: Optional[float]
    normalized_midpoint: Optional[float]
    normalized_to_rial_midpoint: Optional[float]
    source_type_weight_bucket: str
    base_weight: float
    adjusted_weight: float
    filter_pass: bool
    filter_reason: str
    outlier_flag: bool = False


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def to_float(value: str, default: float = 0.0) -> float:
    try:
        raw = str(value).strip()
        if not raw:
            return default
        return float(raw)
    except Exception:
        return default


def to_opt_float(value: str) -> Optional[float]:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def parse_timestamp(ts: str) -> Optional[dt.datetime]:
    raw = str(ts).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if out.tzinfo is None:
        return out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    items = sorted(values)
    q = max(0.0, min(1.0, q))
    pos = (len(items) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return items[low]
    w = pos - low
    return items[low] * (1.0 - w) + items[high] * w


def trimmed_mean(values: Sequence[float], trim_frac: float = 0.10) -> Optional[float]:
    if not values:
        return None
    items = sorted(values)
    if len(items) <= 2:
        return statistics.mean(items)
    trim_n = int(len(items) * trim_frac)
    if trim_n * 2 >= len(items):
        trim_n = max(0, (len(items) - 1) // 2)
    sliced = items[trim_n : len(items) - trim_n]
    if not sliced:
        sliced = items
    return statistics.mean(sliced)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    pairs = [(v, w) for v, w in zip(values, weights) if w > 0]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in pairs) / total_w


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.median(values)


def safe_round(value: Optional[float], ndigits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits)


def classify_source_bucket(rec: Dict[str, str], channel: ChannelMeta) -> str:
    city = str(rec.get("city_guess", "")).strip()

    if channel.likely_individual_shop:
        if city in DIASPORA_CITIES:
            return "diaspora_shop"
        return "direct_shop"

    if channel.channel_type_guess == "dealer_network_channel":
        return "network_channel"

    if channel.channel_type_guess in ("market_price_channel", "aggregator"):
        return "market_channel"

    if channel.channel_type_guess == "individual_exchange_shop":
        if city in DIASPORA_CITIES:
            return "diaspora_shop"
        return "direct_shop"

    return "unclear"


def infer_unit_guess(rec: Dict[str, str]) -> str:
    explicit = str(rec.get("value_unit_guess", "")).strip().lower()
    if explicit in ("toman", "rial"):
        return explicit

    midpoint = to_opt_float(rec.get("midpoint", ""))
    midpoint_rial = to_opt_float(rec.get("midpoint_rial", ""))
    if midpoint is not None and midpoint_rial is not None and midpoint > 0:
        ratio = midpoint_rial / midpoint
        if 8.0 <= ratio <= 12.0:
            return "toman"
        if 0.8 <= ratio <= 1.2:
            return "rial"

    probe = midpoint_rial if midpoint_rial is not None else midpoint
    if probe is None:
        return "unknown"

    if probe >= 900000:
        return "rial"
    if 50000 <= probe <= 400000:
        return "toman"
    return "unknown"


def normalize_values(rec: Dict[str, str], unit_guess: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    buy = to_opt_float(rec.get("buy_quote", ""))
    sell = to_opt_float(rec.get("sell_quote", ""))
    midpoint = to_opt_float(rec.get("midpoint", ""))

    buy_rial = to_opt_float(rec.get("buy_quote_rial", ""))
    sell_rial = to_opt_float(rec.get("sell_quote_rial", ""))
    midpoint_rial = to_opt_float(rec.get("midpoint_rial", ""))

    if midpoint is None and buy is not None and sell is not None:
        midpoint = (buy + sell) / 2.0

    if buy_rial is None and buy is not None:
        buy_rial = buy * 10.0 if unit_guess == "toman" else buy
    if sell_rial is None and sell is not None:
        sell_rial = sell * 10.0 if unit_guess == "toman" else sell

    if midpoint_rial is None:
        if buy_rial is not None and sell_rial is not None:
            midpoint_rial = (buy_rial + sell_rial) / 2.0
        elif midpoint is not None:
            midpoint_rial = midpoint * 10.0 if unit_guess == "toman" else midpoint

    return buy, sell, midpoint, midpoint_rial


def compute_adjusted_weight(rec: BenchmarkRecord, channel_meta: ChannelMeta) -> float:
    base = BASE_WEIGHTS.get(rec.source_type_weight_bucket, BASE_WEIGHTS["unclear"])

    freshness_factor = max(0.45, min(1.05, 0.45 + rec.freshness_score / 150.0))
    structure_factor = max(0.45, min(1.05, 0.45 + rec.structure_score / 160.0))
    quality_factor = max(0.50, min(1.10, 0.50 + rec.overall_record_quality_score / 140.0))
    readiness_factor = max(0.45, min(1.10, 0.45 + channel_meta.ingestion_readiness_score / 150.0))

    quote_type = rec.quote_type_guess
    quote_factor = {
        "direct": 1.08,
        "aggregated": 0.88,
        "reposted": 0.62,
        "unclear": 0.78,
    }.get(quote_type, 0.78)

    dup_factor = 1.0
    if rec.duplication_flag and rec.duplication_flag != "none":
        dup_factor = 0.55

    adjusted = base * freshness_factor * structure_factor * quality_factor * readiness_factor * quote_factor * dup_factor
    return round(max(0.01, adjusted), 6)


def load_channel_metrics(path: Path) -> Dict[str, ChannelMeta]:
    out: Dict[str, ChannelMeta] = {}
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            handle = str(row.get("handle", "")).strip().lower()
            if not handle:
                continue
            out[handle] = ChannelMeta(
                handle=handle,
                title=str(row.get("title", "")).strip(),
                source_priority=str(row.get("source_priority", "")).strip(),
                channel_type_guess=str(row.get("channel_type_guess", "unclear")).strip() or "unclear",
                likely_individual_shop=to_bool(row.get("likely_individual_shop", "false")),
                ingestion_readiness_score=to_float(row.get("ingestion_readiness_score", "0"), 0.0),
                recommended_status=str(row.get("recommended_status", "")).strip(),
            )
    return out


def build_records(
    records_csv: Path,
    channels: Dict[str, ChannelMeta],
    allowed_statuses: set[str],
    min_quality: int,
) -> Tuple[List[BenchmarkRecord], Dict[str, int]]:
    records: List[BenchmarkRecord] = []
    reason_counts: Dict[str, int] = {
        "status_excluded": 0,
        "non_usd": 0,
        "duplicate": 0,
        "low_quality": 0,
        "no_numeric": 0,
        "midpoint_only_weak": 0,
        "ok": 0,
    }

    with records_csv.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            handle = str(row.get("handle", "")).strip().lower()
            if not handle or handle not in channels:
                continue

            meta = channels[handle]
            status = meta.recommended_status
            if status not in allowed_statuses:
                reason_counts["status_excluded"] += 1
                continue

            currency = str(row.get("currency", "")).strip()
            if "USD" not in currency.split("|"):
                reason_counts["non_usd"] += 1
                continue

            dedup_keep = to_bool(row.get("dedup_keep", "false"))
            if not dedup_keep:
                reason_counts["duplicate"] += 1
                continue

            quality = to_float(row.get("overall_record_quality_score", "0"), 0.0)
            if quality < min_quality:
                reason_counts["low_quality"] += 1
                continue

            unit_guess = infer_unit_guess(row)
            buy, sell, midpoint, midpoint_rial = normalize_values(row, unit_guess)
            if midpoint_rial is None or midpoint_rial <= 0:
                reason_counts["no_numeric"] += 1
                continue

            structure = to_float(row.get("structure_score", "0"), 0.0)
            has_pair = buy is not None and sell is not None and buy > 0 and sell > 0
            if (not has_pair) and not (midpoint is not None and structure >= 72 and quality >= 70):
                reason_counts["midpoint_only_weak"] += 1
                continue

            bucket = classify_source_bucket(row, meta)
            base_w = BASE_WEIGHTS.get(bucket, BASE_WEIGHTS["unclear"])

            rec = BenchmarkRecord(
                handle=handle,
                title=meta.title,
                source_priority=meta.source_priority,
                recommended_status=meta.recommended_status,
                channel_type_guess=meta.channel_type_guess,
                likely_individual_shop=meta.likely_individual_shop,
                message_text_sample=str(row.get("message_text_sample", "")).strip(),
                currency=currency,
                buy_quote=buy,
                sell_quote=sell,
                midpoint=midpoint,
                buy_quote_rial=to_opt_float(row.get("buy_quote_rial", "")),
                sell_quote_rial=to_opt_float(row.get("sell_quote_rial", "")),
                midpoint_rial=to_opt_float(row.get("midpoint_rial", "")),
                raw_numeric_values=str(row.get("raw_numeric_values", "")).strip(),
                city_guess=str(row.get("city_guess", "unknown")).strip() or "unknown",
                quote_type_guess=str(row.get("quote_type_guess", "unclear")).strip() or "unclear",
                timestamp_text=str(row.get("timestamp_text", "")).strip(),
                timestamp_iso=str(row.get("timestamp_iso", "")).strip(),
                freshness_score=to_float(row.get("freshness_score", "0"), 0.0),
                structure_score=structure,
                duplication_flag=str(row.get("duplication_flag", "none")).strip() or "none",
                directness_score=to_float(row.get("directness_score", "0"), 0.0),
                overall_record_quality_score=quality,
                inferred_unit_guess=unit_guess,
                normalized_buy=buy,
                normalized_sell=sell,
                normalized_midpoint=midpoint,
                normalized_to_rial_midpoint=midpoint_rial,
                source_type_weight_bucket=bucket,
                base_weight=base_w,
                adjusted_weight=0.0,
                filter_pass=True,
                filter_reason="ok",
                outlier_flag=False,
            )
            rec.adjusted_weight = compute_adjusted_weight(rec, meta)

            reason_counts["ok"] += 1
            records.append(rec)

    return records, reason_counts


def apply_outlier_filter(records: List[BenchmarkRecord]) -> Tuple[List[BenchmarkRecord], int]:
    if not records:
        return records, 0

    vals = [r.normalized_to_rial_midpoint for r in records if r.normalized_to_rial_midpoint is not None]
    values = [float(v) for v in vals]
    if len(values) < 5:
        return records, 0

    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev)

    p05 = percentile(values, 0.05)
    p95 = percentile(values, 0.95)

    outliers = 0
    for r in records:
        v = r.normalized_to_rial_midpoint
        if v is None:
            continue
        is_outlier = False

        if mad > 0:
            robust_z = 0.6745 * (v - med) / mad
            if abs(robust_z) > 3.5:
                is_outlier = True

        if not is_outlier and (v < p05 or v > p95):
            is_outlier = True

        if is_outlier:
            r.outlier_flag = True
            outliers += 1

    cleaned = [r for r in records if not r.outlier_flag]
    return cleaned, outliers


def calculate_estimates(records: Sequence[BenchmarkRecord]) -> Dict[str, Optional[float]]:
    values = [r.normalized_to_rial_midpoint for r in records if r.normalized_to_rial_midpoint is not None]
    weights = [r.adjusted_weight for r in records if r.normalized_to_rial_midpoint is not None]

    direct_values = [r.normalized_to_rial_midpoint for r in records if r.source_type_weight_bucket == "direct_shop" and r.normalized_to_rial_midpoint is not None]
    direct_weights = [r.adjusted_weight for r in records if r.source_type_weight_bucket == "direct_shop" and r.normalized_to_rial_midpoint is not None]

    return {
        "unweighted_median": median(values),
        "trimmed_mean": trimmed_mean(values, trim_frac=0.10),
        "weighted_estimate": weighted_mean(values, weights),
        "direct_shop_estimate": weighted_mean(direct_values, direct_weights) if direct_values else None,
    }


def dispersion_stats(records: Sequence[BenchmarkRecord]) -> Dict[str, Optional[float]]:
    values = [r.normalized_to_rial_midpoint for r in records if r.normalized_to_rial_midpoint is not None]
    if not values:
        return {
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "std": None,
            "iqr": None,
            "mad": None,
            "cv": None,
        }

    med = statistics.median(values)
    p25 = percentile(values, 0.25)
    p75 = percentile(values, 0.75)
    iqr = p75 - p25
    mad = statistics.median([abs(v - med) for v in values])
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    cv = (std / med) if med else None

    return {
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": p25,
        "median": med,
        "p75": p75,
        "p90": percentile(values, 0.90),
        "max": max(values),
        "std": std,
        "iqr": iqr,
        "mad": mad,
        "cv": cv,
    }


def concentration_stats(records: Sequence[BenchmarkRecord]) -> Dict[str, object]:
    by_channel: Dict[str, float] = {}
    total_weight = 0.0
    for r in records:
        by_channel[r.handle] = by_channel.get(r.handle, 0.0) + r.adjusted_weight
        total_weight += r.adjusted_weight

    if total_weight <= 0:
        return {
            "channel_count": 0,
            "top_channel": None,
            "top_channel_weight_share": None,
            "top3_weight_share": None,
            "hhi": None,
            "effective_channel_count": None,
            "dominance_flag": False,
            "concentration_level": "unknown",
        }

    ordered = sorted(by_channel.items(), key=lambda kv: (-kv[1], kv[0]))
    shares = [(h, w / total_weight) for h, w in ordered]
    top_channel, top_share = shares[0]
    top3_share = sum(s for _, s in shares[:3])
    hhi = sum(s * s for _, s in shares)
    eff_n = (1.0 / hhi) if hhi > 0 else None

    concentration_level = "low"
    if top_share > 0.35 or (eff_n is not None and eff_n < 4.0):
        concentration_level = "high"
    elif top_share > 0.22 or (eff_n is not None and eff_n < 8.0):
        concentration_level = "medium"

    return {
        "channel_count": len(by_channel),
        "top_channel": top_channel,
        "top_channel_weight_share": top_share,
        "top3_weight_share": top3_share,
        "hhi": hhi,
        "effective_channel_count": eff_n,
        "dominance_flag": bool(top_share > 0.35),
        "concentration_level": concentration_level,
    }


def label_dispersion_level(disp: Dict[str, Optional[float]]) -> str:
    cv = disp.get("cv")
    if cv is None:
        return "unknown"
    if cv <= 0.02:
        return "low"
    if cv <= 0.05:
        return "medium"
    return "high"


def confidence_assessment(
    usable_count: int,
    dispersion_level: str,
    concentration_level: str,
    dominance_flag: bool,
) -> str:
    if usable_count >= 80 and dispersion_level in ("low", "medium") and concentration_level != "high" and not dominance_flag:
        return "high"
    if usable_count >= 35 and dispersion_level != "high" and concentration_level != "high":
        return "medium"
    return "low"


def recommended_next_step(confidence: str, delta_monitor_pct: Optional[float]) -> str:
    if confidence == "high":
        if delta_monitor_pct is not None and abs(delta_monitor_pct) <= 1.5:
            return "stand up a persistent research feed for ready channels; keep monitor channels as secondary checks"
        return "run a 7-day rolling backtest before promoting to persistent research feed"

    if confidence == "medium":
        return "continue daily research collection for 2-3 weeks and tighten channel-level weighting before pipeline integration"

    return "expand direct-shop coverage and improve record freshness/structure before considering benchmark integration"


def build_source_weight_report(records: Sequence[BenchmarkRecord]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[BenchmarkRecord]] = {}
    for r in records:
        grouped.setdefault(r.source_type_weight_bucket, []).append(r)

    rows: List[Dict[str, object]] = []
    for bucket in ("direct_shop", "diaspora_shop", "network_channel", "market_channel", "unclear"):
        recs = grouped.get(bucket, [])
        if recs:
            values = [r.normalized_to_rial_midpoint for r in recs if r.normalized_to_rial_midpoint is not None]
            weights = [r.adjusted_weight for r in recs if r.normalized_to_rial_midpoint is not None]
            estimate = weighted_mean(values, weights)
            total_w = sum(weights)
            avg_w = total_w / len(weights) if weights else 0.0
            avg_q = sum(r.overall_record_quality_score for r in recs) / len(recs)
            avg_f = sum(r.freshness_score for r in recs) / len(recs)
            avg_s = sum(r.structure_score for r in recs) / len(recs)
        else:
            estimate = None
            total_w = 0.0
            avg_w = 0.0
            avg_q = 0.0
            avg_f = 0.0
            avg_s = 0.0

        rows.append(
            {
                "source_type_weight_bucket": bucket,
                "base_weight": BASE_WEIGHTS[bucket],
                "record_count": len(recs),
                "total_adjusted_weight": round(total_w, 4),
                "average_adjusted_weight": round(avg_w, 6),
                "average_quality_score": round(avg_q, 2),
                "average_freshness_score": round(avg_f, 2),
                "average_structure_score": round(avg_s, 2),
                "bucket_weighted_estimate": safe_round(estimate, 2),
            }
        )
    return rows


def write_records(path: Path, records: Sequence[BenchmarkRecord]) -> None:
    ordered = sorted(records, key=lambda r: (0 if r.source_priority == "P1" else 1, r.handle, r.timestamp_iso, r.message_text_sample))
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "handle",
                "title",
                "source_priority",
                "recommended_status",
                "source_type_weight_bucket",
                "message_text_sample",
                "currency",
                "buy_quote",
                "sell_quote",
                "midpoint",
                "normalized_buy",
                "normalized_sell",
                "normalized_midpoint",
                "inferred_unit_guess",
                "normalized_to_rial_midpoint",
                "raw_numeric_values",
                "city_guess",
                "quote_type_guess",
                "timestamp_text",
                "timestamp_iso",
                "freshness_score",
                "structure_score",
                "duplication_flag",
                "directness_score",
                "overall_record_quality_score",
                "base_weight",
                "adjusted_weight",
                "outlier_flag",
                "channel_type_guess",
                "likely_individual_shop",
            ],
        )
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    "handle": r.handle,
                    "title": r.title,
                    "source_priority": r.source_priority,
                    "recommended_status": r.recommended_status,
                    "source_type_weight_bucket": r.source_type_weight_bucket,
                    "message_text_sample": r.message_text_sample,
                    "currency": r.currency,
                    "buy_quote": "" if r.buy_quote is None else f"{r.buy_quote:.2f}",
                    "sell_quote": "" if r.sell_quote is None else f"{r.sell_quote:.2f}",
                    "midpoint": "" if r.midpoint is None else f"{r.midpoint:.2f}",
                    "normalized_buy": "" if r.normalized_buy is None else f"{r.normalized_buy:.2f}",
                    "normalized_sell": "" if r.normalized_sell is None else f"{r.normalized_sell:.2f}",
                    "normalized_midpoint": "" if r.normalized_midpoint is None else f"{r.normalized_midpoint:.2f}",
                    "inferred_unit_guess": r.inferred_unit_guess,
                    "normalized_to_rial_midpoint": "" if r.normalized_to_rial_midpoint is None else f"{r.normalized_to_rial_midpoint:.2f}",
                    "raw_numeric_values": r.raw_numeric_values,
                    "city_guess": r.city_guess,
                    "quote_type_guess": r.quote_type_guess,
                    "timestamp_text": r.timestamp_text,
                    "timestamp_iso": r.timestamp_iso,
                    "freshness_score": f"{r.freshness_score:.2f}",
                    "structure_score": f"{r.structure_score:.2f}",
                    "duplication_flag": r.duplication_flag,
                    "directness_score": f"{r.directness_score:.2f}",
                    "overall_record_quality_score": f"{r.overall_record_quality_score:.2f}",
                    "base_weight": f"{r.base_weight:.4f}",
                    "adjusted_weight": f"{r.adjusted_weight:.6f}",
                    "outlier_flag": r.outlier_flag,
                    "channel_type_guess": r.channel_type_guess,
                    "likely_individual_shop": r.likely_individual_shop,
                }
            )


def write_diagnostics(path: Path, rows: Sequence[Tuple[str, str, object, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["section", "metric", "value", "notes"])
        writer.writeheader()
        for section, metric, value, notes in rows:
            writer.writerow({"section": section, "metric": metric, "value": value, "notes": notes})


def write_source_weights(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "source_type_weight_bucket",
                "base_weight",
                "record_count",
                "total_adjusted_weight",
                "average_adjusted_weight",
                "average_quality_score",
                "average_freshness_score",
                "average_structure_score",
                "bucket_weighted_estimate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_mode(
    records_csv: Path,
    channels: Dict[str, ChannelMeta],
    allowed_statuses: set[str],
    min_quality: int,
) -> Dict[str, object]:
    pre_records, reason_counts = build_records(records_csv, channels, allowed_statuses, min_quality)
    cleaned, outliers = apply_outlier_filter(pre_records)

    est = calculate_estimates(cleaned)
    disp = dispersion_stats(cleaned)
    conc = concentration_stats(cleaned)

    mix_counts = {
        "direct_shop": sum(1 for r in cleaned if r.source_type_weight_bucket == "direct_shop"),
        "diaspora_shop": sum(1 for r in cleaned if r.source_type_weight_bucket == "diaspora_shop"),
        "network_channel": sum(1 for r in cleaned if r.source_type_weight_bucket == "network_channel"),
        "market_channel": sum(1 for r in cleaned if r.source_type_weight_bucket == "market_channel"),
        "unclear": sum(1 for r in cleaned if r.source_type_weight_bucket == "unclear"),
    }

    return {
        "records_before_outlier": pre_records,
        "records": cleaned,
        "outliers_removed": outliers,
        "reason_counts": reason_counts,
        "estimates": est,
        "dispersion": disp,
        "concentration": conc,
        "mix_counts": mix_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram research benchmark constructor")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Survey output directory")
    parser.add_argument("--include-monitor", action="store_true", help="Primary mode includes monitor_only channels")
    parser.add_argument("--min-quality", type=int, default=60, help="Minimum overall_record_quality_score")
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    records_csv = survey_dir / "pilot_quote_records.csv"
    channel_metrics_csv = survey_dir / "pilot_channel_metrics.csv"
    pilot_summary_json = survey_dir / "pilot_ingestion_summary.json"

    for p in (records_csv, channel_metrics_csv, pilot_summary_json):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    channels = load_channel_metrics(channel_metrics_csv)
    pilot_summary = json.loads(pilot_summary_json.read_text(encoding="utf-8"))

    ready_mode = run_mode(records_csv, channels, READY_STATUSES, args.min_quality)
    ready_monitor_mode = run_mode(records_csv, channels, READY_MONITOR_STATUSES, args.min_quality)

    primary_mode_name = "ready_plus_monitor" if args.include_monitor else "ready_only"
    primary_mode = ready_monitor_mode if args.include_monitor else ready_mode

    primary_records: List[BenchmarkRecord] = primary_mode["records"]
    ready_est = ready_mode["estimates"]["weighted_estimate"]
    rpm_est = ready_monitor_mode["estimates"]["weighted_estimate"]

    delta_rpm_vs_ready = None
    if ready_est and rpm_est:
        delta_rpm_vs_ready = ((rpm_est - ready_est) / ready_est) * 100.0

    disp_level = label_dispersion_level(primary_mode["dispersion"])
    conc_level = str(primary_mode["concentration"].get("concentration_level", "unknown"))

    confidence = confidence_assessment(
        usable_count=len(primary_records),
        dispersion_level=disp_level,
        concentration_level=conc_level,
        dominance_flag=bool(primary_mode["concentration"].get("dominance_flag", False)),
    )

    next_step = recommended_next_step(confidence, delta_rpm_vs_ready)

    # Comparison impact text.
    monitor_impact = "mixed"
    ready_cv = ready_mode["dispersion"].get("cv")
    rpm_cv = ready_monitor_mode["dispersion"].get("cv")
    if delta_rpm_vs_ready is not None and ready_cv is not None and rpm_cv is not None:
        if abs(delta_rpm_vs_ready) <= 2.0 and rpm_cv <= ready_cv * 0.95:
            monitor_impact = "improves"
        elif abs(delta_rpm_vs_ready) > 3.0 or rpm_cv > ready_cv * 1.10:
            monitor_impact = "worsens"
        else:
            monitor_impact = "mixed"

    # Write record output for primary mode only.
    records_out = survey_dir / "telegram_research_benchmark_records.csv"
    summary_out = survey_dir / "telegram_research_benchmark_summary.json"
    diagnostics_out = survey_dir / "telegram_research_benchmark_diagnostics.csv"
    source_weights_out = survey_dir / "telegram_source_weight_report.csv"

    write_records(records_out, primary_records)

    source_weight_rows = build_source_weight_report(primary_records)
    write_source_weights(source_weights_out, source_weight_rows)

    summary_payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "mode": primary_mode_name,
        "input_records": pilot_summary.get("total_quote_records_extracted", 0),
        "filtered_records": len(primary_records),
        "outliers_removed": primary_mode["outliers_removed"],
        "direct_shop_records": primary_mode["mix_counts"]["direct_shop"],
        "diaspora_shop_records": primary_mode["mix_counts"]["diaspora_shop"],
        "network_channel_records": primary_mode["mix_counts"]["network_channel"],
        "market_channel_records": primary_mode["mix_counts"]["market_channel"],
        "unclear_records": primary_mode["mix_counts"]["unclear"],
        "telegram_unweighted_median": safe_round(primary_mode["estimates"]["unweighted_median"], 2),
        "telegram_trimmed_mean": safe_round(primary_mode["estimates"]["trimmed_mean"], 2),
        "telegram_weighted_estimate": safe_round(primary_mode["estimates"]["weighted_estimate"], 2),
        "direct_shop_estimate": safe_round(primary_mode["estimates"]["direct_shop_estimate"], 2),
        "ready_channels_only_estimate": safe_round(ready_est, 2),
        "ready_plus_monitor_estimate": safe_round(rpm_est, 2),
        "record_dispersion_stats": {
            k: safe_round(v, 4) if isinstance(v, float) else v for k, v in primary_mode["dispersion"].items()
        },
        "concentration_stats": {
            k: safe_round(v, 6) if isinstance(v, float) else v for k, v in primary_mode["concentration"].items()
        },
        "confidence_assessment": confidence,
        "recommended_next_step": next_step,
        "monitor_inclusion_impact": {
            "delta_ready_plus_monitor_vs_ready_pct": safe_round(delta_rpm_vs_ready, 4),
            "impact_label": monitor_impact,
        },
        "filter_reason_counts_primary_mode": primary_mode["reason_counts"],
    }
    summary_out.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    diagnostics_rows: List[Tuple[str, str, object, str]] = []
    diagnostics_rows.extend(
        [
            ("input", "pilot_input_records", pilot_summary.get("total_quote_records_extracted", 0), "from pilot_ingestion_summary.json"),
            ("input", "primary_mode", primary_mode_name, "default is ready_only unless --include-monitor"),
            ("filter", "usable_records_surviving_filters", len(primary_records), "after status, USD, quality, duplicate, numeric filters and outlier handling"),
            ("filter", "outliers_removed", primary_mode["outliers_removed"], "MAD + percentile filter"),
            ("mix", "direct_shop_records", primary_mode["mix_counts"]["direct_shop"], "source bucket count"),
            ("mix", "diaspora_shop_records", primary_mode["mix_counts"]["diaspora_shop"], "source bucket count"),
            ("mix", "network_channel_records", primary_mode["mix_counts"]["network_channel"], "source bucket count"),
            ("mix", "market_channel_records", primary_mode["mix_counts"]["market_channel"], "source bucket count"),
            ("mix", "unclear_records", primary_mode["mix_counts"]["unclear"], "source bucket count"),
            ("estimate", "telegram_unweighted_median", safe_round(primary_mode["estimates"]["unweighted_median"], 2), "rial"),
            ("estimate", "telegram_trimmed_mean", safe_round(primary_mode["estimates"]["trimmed_mean"], 2), "rial"),
            ("estimate", "telegram_weighted_estimate", safe_round(primary_mode["estimates"]["weighted_estimate"], 2), "rial"),
            ("estimate", "direct_shop_estimate", safe_round(primary_mode["estimates"]["direct_shop_estimate"], 2), "rial"),
            ("estimate", "ready_only_estimate", safe_round(ready_est, 2), "rial"),
            ("estimate", "ready_plus_monitor_estimate", safe_round(rpm_est, 2), "rial"),
            ("dispersion", "dispersion_level", disp_level, "based on CV thresholds"),
            ("dispersion", "cv", safe_round(primary_mode["dispersion"].get("cv"), 6), "std/median"),
            ("dispersion", "iqr", safe_round(primary_mode["dispersion"].get("iqr"), 2), "p75-p25"),
            ("concentration", "concentration_level", conc_level, "based on top share and effective channel count"),
            ("concentration", "top_channel", primary_mode["concentration"].get("top_channel"), "highest total adjusted weight"),
            ("concentration", "top_channel_weight_share", safe_round(primary_mode["concentration"].get("top_channel_weight_share"), 6), "share of adjusted weight"),
            ("concentration", "effective_channel_count", safe_round(primary_mode["concentration"].get("effective_channel_count"), 4), "1/HHI"),
            ("comparison", "delta_ready_plus_monitor_vs_ready_pct", safe_round(delta_rpm_vs_ready, 4), "(rpm-ready)/ready*100"),
            ("comparison", "monitor_inclusion_impact", monitor_impact, "whether adding monitor channels improves stability"),
        ]
    )

    write_diagnostics(diagnostics_out, diagnostics_rows)

    print(json.dumps(
        {
            "usable_records_surviving_filters": len(primary_records),
            "telegram_derived_research_estimate": safe_round(primary_mode["estimates"]["weighted_estimate"], 2),
            "direct_shop_estimate": safe_round(primary_mode["estimates"]["direct_shop_estimate"], 2),
            "trimmed_mean": safe_round(primary_mode["estimates"]["trimmed_mean"], 2),
            "weighted_estimate": safe_round(primary_mode["estimates"]["weighted_estimate"], 2),
            "dispersion_level": disp_level,
            "concentration_level": conc_level,
            "stable_enough_for_persistent_research_feed": confidence in ("high", "medium"),
            "adding_monitor_channels_impact": monitor_impact,
            "delta_ready_plus_monitor_vs_ready_pct": safe_round(delta_rpm_vs_ready, 4),
        },
        ensure_ascii=False,
        indent=2,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
