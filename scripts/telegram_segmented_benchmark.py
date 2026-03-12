#!/usr/bin/env python3
"""Segmented Telegram benchmark analysis (research only).

Builds basket-level diagnostics from pilot research records to identify whether a
clean Telegram sub-basket is suitable for a dealer-overlay research input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASKETS = (
    "inside_iran_direct_shop",
    "diaspora_direct_shop",
    "network_channel",
    "market_channel",
    "unclear",
)

DIASPORA_CITY_SET = {"Dubai", "Istanbul", "Frankfurt", "Hamburg", "London"}
DIASPORA_HINTS = ("london", "dubai", "istanbul", "hamburg", "frankfurt", "uk")

BASE_WEIGHTS = {
    "inside_iran_direct_shop": 1.00,
    "diaspora_direct_shop": 0.90,
    "network_channel": 0.65,
    "market_channel": 0.50,
    "unclear": 0.30,
}

# Research plausibility bounds for USD/IRR midpoint in rial.
MIN_PLAUSIBLE_USD_IRR_RIAL = 500_000.0
MAX_PLAUSIBLE_USD_IRR_RIAL = 5_000_000.0


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
class SegRecord:
    handle: str
    title: str
    source_priority: str
    channel_type_guess: str
    likely_individual_shop: bool
    ingestion_readiness_score: float
    recommended_status: str
    basket: str
    message_text_sample: str
    city_guess: str
    quote_type_guess: str
    freshness_score: float
    structure_score: float
    directness_score: float
    overall_record_quality_score: float
    midpoint_rial: float
    adjusted_weight: float
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


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(vals) - 1)
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


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    pairs = [(v, w) for v, w in zip(values, weights) if w > 0]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in pairs) / total_w


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


def infer_diaspora(city_guess: str, handle: str, title: str) -> bool:
    city = (city_guess or "").strip()
    if city in DIASPORA_CITY_SET:
        return True

    probe = f"{handle} {title}".lower()
    return any(tok in probe for tok in DIASPORA_HINTS)


def classify_basket(meta: ChannelMeta, city_guess: str) -> str:
    if meta.likely_individual_shop or meta.channel_type_guess == "individual_exchange_shop":
        return "diaspora_direct_shop" if infer_diaspora(city_guess, meta.handle, meta.title) else "inside_iran_direct_shop"

    if meta.channel_type_guess == "dealer_network_channel":
        return "network_channel"

    if meta.channel_type_guess in ("market_price_channel", "aggregator"):
        return "market_channel"

    return "unclear"


def compute_weight(rec: SegRecord) -> float:
    base = BASE_WEIGHTS.get(rec.basket, BASE_WEIGHTS["unclear"])
    freshness_factor = max(0.45, min(1.05, 0.45 + rec.freshness_score / 150.0))
    structure_factor = max(0.45, min(1.05, 0.45 + rec.structure_score / 160.0))
    quality_factor = max(0.45, min(1.10, 0.45 + rec.overall_record_quality_score / 140.0))
    readiness_factor = max(0.45, min(1.10, 0.45 + rec.ingestion_readiness_score / 150.0))

    quote_factor = {
        "direct": 1.08,
        "aggregated": 0.88,
        "reposted": 0.62,
        "unclear": 0.75,
    }.get(rec.quote_type_guess, 0.75)

    weight = base * freshness_factor * structure_factor * quality_factor * readiness_factor * quote_factor
    return round(max(0.01, weight), 6)


def load_records(
    pilot_quote_records_csv: Path,
    channels: Dict[str, ChannelMeta],
    allowed_statuses: set[str],
    min_quality: int,
) -> Tuple[List[SegRecord], Dict[str, int]]:
    out: List[SegRecord] = []
    reason_counts = {
        "status_excluded": 0,
        "non_usd": 0,
        "duplicate": 0,
        "low_quality": 0,
        "no_numeric_midpoint": 0,
        "weak_midpoint_only": 0,
        "out_of_plausible_range": 0,
        "ok": 0,
    }

    with pilot_quote_records_csv.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            handle = str(row.get("handle", "")).strip().lower()
            if not handle or handle not in channels:
                continue
            meta = channels[handle]

            if meta.recommended_status not in allowed_statuses:
                reason_counts["status_excluded"] += 1
                continue

            currency = str(row.get("currency", "")).strip()
            if "USD" not in currency.split("|"):
                reason_counts["non_usd"] += 1
                continue

            if not to_bool(row.get("dedup_keep", "false")):
                reason_counts["duplicate"] += 1
                continue

            quality = to_float(row.get("overall_record_quality_score", "0"), 0.0)
            if quality < min_quality:
                reason_counts["low_quality"] += 1
                continue

            midpoint_rial = to_opt_float(row.get("midpoint_rial", ""))
            if midpoint_rial is None or midpoint_rial <= 0:
                reason_counts["no_numeric_midpoint"] += 1
                continue

            midpoint_raw = to_opt_float(row.get("midpoint", ""))
            # Some records are emitted in toman-equivalent scale; normalize when obviously small.
            if midpoint_rial < MIN_PLAUSIBLE_USD_IRR_RIAL and midpoint_raw is not None and 50_000 <= midpoint_raw <= 500_000:
                midpoint_rial = midpoint_raw * 10.0

            if midpoint_rial < MIN_PLAUSIBLE_USD_IRR_RIAL or midpoint_rial > MAX_PLAUSIBLE_USD_IRR_RIAL:
                reason_counts["out_of_plausible_range"] += 1
                continue

            buy = to_opt_float(row.get("buy_quote", ""))
            sell = to_opt_float(row.get("sell_quote", ""))
            has_pair = buy is not None and sell is not None and buy > 0 and sell > 0
            structure = to_float(row.get("structure_score", "0"), 0.0)
            if (not has_pair) and not (structure >= 72 and quality >= 70):
                reason_counts["weak_midpoint_only"] += 1
                continue

            city = str(row.get("city_guess", "unknown")).strip() or "unknown"
            basket = classify_basket(meta, city)

            rec = SegRecord(
                handle=handle,
                title=meta.title,
                source_priority=meta.source_priority,
                channel_type_guess=meta.channel_type_guess,
                likely_individual_shop=meta.likely_individual_shop,
                ingestion_readiness_score=meta.ingestion_readiness_score,
                recommended_status=meta.recommended_status,
                basket=basket,
                message_text_sample=str(row.get("message_text_sample", "")).strip(),
                city_guess=city,
                quote_type_guess=str(row.get("quote_type_guess", "unclear")).strip() or "unclear",
                freshness_score=to_float(row.get("freshness_score", "0"), 0.0),
                structure_score=structure,
                directness_score=to_float(row.get("directness_score", "0"), 0.0),
                overall_record_quality_score=quality,
                midpoint_rial=midpoint_rial,
                adjusted_weight=0.0,
            )
            rec.adjusted_weight = compute_weight(rec)
            out.append(rec)
            reason_counts["ok"] += 1

    return out, reason_counts


def apply_outlier_filter_by_basket(records: List[SegRecord]) -> Tuple[List[SegRecord], int, Dict[str, int]]:
    outliers_total = 0
    outliers_by_basket: Dict[str, int] = {b: 0 for b in BASKETS}

    by_basket: Dict[str, List[SegRecord]] = {b: [] for b in BASKETS}
    for rec in records:
        by_basket[rec.basket].append(rec)

    for basket, items in by_basket.items():
        if len(items) < 8:
            continue

        vals = [r.midpoint_rial for r in items]
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        p05 = percentile(vals, 0.05)
        p95 = percentile(vals, 0.95)

        for rec in items:
            outlier = False
            if mad > 0:
                rz = 0.6745 * (rec.midpoint_rial - med) / mad
                if abs(rz) > 3.5:
                    outlier = True
            if not outlier and (rec.midpoint_rial < p05 or rec.midpoint_rial > p95):
                outlier = True

            if outlier:
                rec.outlier_flag = True
                outliers_total += 1
                outliers_by_basket[basket] += 1

    cleaned = [r for r in records if not r.outlier_flag]
    return cleaned, outliers_total, outliers_by_basket


def compute_dispersion(values: Sequence[float]) -> Dict[str, Optional[float]]:
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


def compute_concentration(records: Sequence[SegRecord]) -> Dict[str, object]:
    if not records:
        return {
            "top_channel": None,
            "top_channel_share": None,
            "top3_share": None,
            "effective_channel_count": None,
            "hhi": None,
            "dominance_flag": False,
            "concentration_level": "unknown",
        }

    by_ch: Dict[str, float] = {}
    total_w = 0.0
    for r in records:
        by_ch[r.handle] = by_ch.get(r.handle, 0.0) + r.adjusted_weight
        total_w += r.adjusted_weight

    ordered = sorted(by_ch.items(), key=lambda kv: (-kv[1], kv[0]))
    shares = [(h, w / total_w) for h, w in ordered if total_w > 0]
    if not shares:
        return {
            "top_channel": None,
            "top_channel_share": None,
            "top3_share": None,
            "effective_channel_count": None,
            "hhi": None,
            "dominance_flag": False,
            "concentration_level": "unknown",
        }

    top_ch, top_share = shares[0]
    top3_share = sum(s for _, s in shares[:3])
    hhi = sum(s * s for _, s in shares)
    eff_n = (1.0 / hhi) if hhi > 0 else None

    level = "low"
    if top_share > 0.35 or (eff_n is not None and eff_n < 4.0):
        level = "high"
    elif top_share > 0.22 or (eff_n is not None and eff_n < 8.0):
        level = "medium"

    return {
        "top_channel": top_ch,
        "top_channel_share": top_share,
        "top3_share": top3_share,
        "effective_channel_count": eff_n,
        "hhi": hhi,
        "dominance_flag": bool(top_share > 0.35),
        "concentration_level": level,
    }


def score_coherence(
    usable_records: int,
    cv: Optional[float],
    top_share: Optional[float],
    eff_n: Optional[float],
    avg_freshness: float,
    avg_structure: float,
) -> float:
    count_comp = min(1.0, usable_records / 30.0) * 25.0

    if cv is None:
        disp_comp = 0.0
    elif cv <= 0.03:
        disp_comp = 25.0
    elif cv >= 0.20:
        disp_comp = 0.0
    else:
        disp_comp = 25.0 * (1.0 - (cv - 0.03) / 0.17)

    if top_share is None:
        share_comp = 0.0
    elif top_share <= 0.20:
        share_comp = 20.0
    elif top_share >= 0.60:
        share_comp = 0.0
    else:
        share_comp = 20.0 * (1.0 - (top_share - 0.20) / 0.40)

    if eff_n is None:
        eff_comp = 0.0
    elif eff_n >= 8.0:
        eff_comp = 15.0
    elif eff_n <= 2.0:
        eff_comp = 0.0
    else:
        eff_comp = 15.0 * ((eff_n - 2.0) / 6.0)

    fresh_comp = min(1.0, avg_freshness / 85.0) * 20.0
    struct_comp = min(1.0, avg_structure / 90.0) * 15.0

    score = count_comp + disp_comp + share_comp + eff_comp + fresh_comp + struct_comp
    return round(max(0.0, min(100.0, score)), 2)


def stability_label(coherence: float, usable: int, cv: Optional[float], top_share: Optional[float]) -> str:
    if coherence >= 70 and usable >= 20 and cv is not None and cv <= 0.12 and top_share is not None and top_share <= 0.35:
        return "benchmark_candidate"
    if coherence >= 50 and usable >= 10:
        return "shadow_diagnostic_feed"
    return "monitor_only_signal"


def safe_round(v: Optional[float], n: int = 2) -> Optional[float]:
    if v is None:
        return None
    return round(v, n)


def build_basket_rows(all_input: List[SegRecord], usable: List[SegRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    by_input: Dict[str, List[SegRecord]] = {b: [] for b in BASKETS}
    by_usable: Dict[str, List[SegRecord]] = {b: [] for b in BASKETS}
    for r in all_input:
        by_input[r.basket].append(r)
    for r in usable:
        by_usable[r.basket].append(r)

    for basket in BASKETS:
        input_recs = by_input[basket]
        usable_recs = by_usable[basket]

        vals = [r.midpoint_rial for r in usable_recs]
        wts = [r.adjusted_weight for r in usable_recs]

        med = statistics.median(vals) if vals else None
        tmean = trimmed_mean(vals, 0.10) if vals else None
        west = weighted_mean(vals, wts) if vals else None

        disp = compute_dispersion(vals)
        conc = compute_concentration(usable_recs)

        avg_fresh = statistics.mean([r.freshness_score for r in usable_recs]) if usable_recs else 0.0
        avg_struct = statistics.mean([r.structure_score for r in usable_recs]) if usable_recs else 0.0

        coh = score_coherence(
            usable_records=len(usable_recs),
            cv=disp.get("cv"),
            top_share=conc.get("top_channel_share"),
            eff_n=conc.get("effective_channel_count"),
            avg_freshness=avg_fresh,
            avg_structure=avg_struct,
        )
        stability = stability_label(coh, len(usable_recs), disp.get("cv"), conc.get("top_channel_share"))

        rows.append(
            {
                "basket": basket,
                "input_records": len(input_recs),
                "usable_records": len(usable_recs),
                "median": safe_round(med, 2),
                "trimmed_mean": safe_round(tmean, 2),
                "weighted_estimate": safe_round(west, 2),
                "std": safe_round(disp.get("std"), 4),
                "cv": safe_round(disp.get("cv"), 6),
                "iqr": safe_round(disp.get("iqr"), 2),
                "mad": safe_round(disp.get("mad"), 2),
                "top_channel": conc.get("top_channel"),
                "top_channel_share": safe_round(conc.get("top_channel_share"), 6),
                "effective_channel_count": safe_round(conc.get("effective_channel_count"), 4),
                "concentration_level": conc.get("concentration_level"),
                "avg_freshness": safe_round(avg_fresh, 2),
                "avg_structure": safe_round(avg_struct, 2),
                "coherence_score": coh,
                "stability_assessment": stability,
            }
        )

    rows.sort(key=lambda r: (-float(r["coherence_score"]), r["basket"]))
    return rows


def spread_row(rows_by_basket: Dict[str, Dict[str, object]], left: str, right: str) -> Dict[str, object]:
    l = rows_by_basket.get(left, {})
    r = rows_by_basket.get(right, {})
    le = to_opt_float(l.get("weighted_estimate", ""))
    re = to_opt_float(r.get("weighted_estimate", ""))

    spread_irr = None
    spread_pct = None
    if le is not None and re is not None:
        spread_irr = le - re
        if re != 0:
            spread_pct = (spread_irr / re) * 100.0

    return {
        "left_basket": left,
        "right_basket": right,
        "left_weighted_estimate": safe_round(le, 2),
        "right_weighted_estimate": safe_round(re, 2),
        "spread_irr": safe_round(spread_irr, 2),
        "spread_pct": safe_round(spread_pct, 4),
        "left_usable_records": l.get("usable_records", 0),
        "right_usable_records": r.get("usable_records", 0),
    }


def build_channel_ranking(usable: List[SegRecord]) -> List[Dict[str, object]]:
    by_channel: Dict[str, List[SegRecord]] = {}
    by_bucket_weight: Dict[str, float] = {b: 0.0 for b in BASKETS}

    for rec in usable:
        by_channel.setdefault(rec.handle, []).append(rec)
        by_bucket_weight[rec.basket] += rec.adjusted_weight

    rows: List[Dict[str, object]] = []
    for handle, recs in by_channel.items():
        sample = recs[0]
        bucket = sample.basket
        values = [r.midpoint_rial for r in recs]
        weights = [r.adjusted_weight for r in recs]
        est = weighted_mean(values, weights)

        avg_q = statistics.mean([r.overall_record_quality_score for r in recs])
        avg_f = statistics.mean([r.freshness_score for r in recs])
        avg_s = statistics.mean([r.structure_score for r in recs])

        bucket_total_w = by_bucket_weight.get(bucket, 0.0)
        ch_w = sum(weights)
        share = (ch_w / bucket_total_w) if bucket_total_w > 0 else 0.0

        score = 0.4 * avg_q + 0.2 * avg_s + 0.2 * avg_f + 0.2 * min(100.0, len(recs) * 8.0)
        if share > 0.35:
            score -= 12.0

        role = "monitor_only_signal"
        if len(recs) >= 8 and avg_q >= 70 and avg_s >= 80 and share <= 0.35:
            role = "benchmark_support"
        elif len(recs) >= 4 and avg_q >= 60:
            role = "shadow_diagnostic"

        rows.append(
            {
                "handle": handle,
                "title": sample.title,
                "recommended_status": sample.recommended_status,
                "source_priority": sample.source_priority,
                "basket": bucket,
                "record_count": len(recs),
                "weighted_estimate": safe_round(est, 2),
                "avg_quality": safe_round(avg_q, 2),
                "avg_freshness": safe_round(avg_f, 2),
                "avg_structure": safe_round(avg_s, 2),
                "channel_weight_share_in_basket": safe_round(share, 6),
                "channel_coherence_contribution_score": safe_round(score, 2),
                "recommended_role": role,
            }
        )

    rows.sort(key=lambda r: (-float(r["channel_coherence_contribution_score"]), r["basket"], r["handle"]))
    return rows


def load_reference_benchmark(path: Path) -> Dict[str, Optional[float]]:
    values: List[float] = []
    weights: List[float] = []
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            v = to_opt_float(row.get("normalized_to_rial_midpoint", ""))
            w = to_opt_float(row.get("adjusted_weight", ""))
            if v is None:
                continue
            values.append(v)
            weights.append(w if w is not None and w > 0 else 1.0)

    return {
        "reference_record_count": float(len(values)),
        "reference_weighted_estimate": weighted_mean(values, weights),
        "reference_median": statistics.median(values) if values else None,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmented Telegram benchmark analysis")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Survey output directory")
    parser.add_argument(
        "--mode",
        choices=["ready_plus_monitor", "ready_only"],
        default="ready_plus_monitor",
        help="Which channel statuses to include",
    )
    parser.add_argument("--min-quality", type=int, default=60, help="Minimum record quality")
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    benchmark_records_csv = survey_dir / "telegram_research_benchmark_records.csv"
    pilot_channel_metrics_csv = survey_dir / "pilot_channel_metrics.csv"
    pilot_quote_records_csv = survey_dir / "pilot_quote_records.csv"

    for p in (benchmark_records_csv, pilot_channel_metrics_csv, pilot_quote_records_csv):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    channels = load_channel_metrics(pilot_channel_metrics_csv)
    allowed_statuses = {"ready_for_research_ingestion"}
    if args.mode == "ready_plus_monitor":
        allowed_statuses = {"ready_for_research_ingestion", "monitor_only"}

    input_records, reason_counts = load_records(
        pilot_quote_records_csv=pilot_quote_records_csv,
        channels=channels,
        allowed_statuses=allowed_statuses,
        min_quality=args.min_quality,
    )

    usable_records, outliers_removed, outliers_by_basket = apply_outlier_filter_by_basket(input_records)

    basket_rows = build_basket_rows(input_records, usable_records)
    rows_by_basket = {str(r["basket"]): r for r in basket_rows}

    spreads = [
        spread_row(rows_by_basket, "diaspora_direct_shop", "inside_iran_direct_shop"),
        spread_row(rows_by_basket, "network_channel", "inside_iran_direct_shop"),
        spread_row(rows_by_basket, "market_channel", "inside_iran_direct_shop"),
    ]

    channel_rank_rows = build_channel_ranking(usable_records)

    best_basket = basket_rows[0] if basket_rows else None
    inside_row = rows_by_basket.get("inside_iran_direct_shop", {})
    diaspora_row = rows_by_basket.get("diaspora_direct_shop", {})
    network_row = rows_by_basket.get("network_channel", {})
    market_row = rows_by_basket.get("market_channel", {})

    inside_large_enough = bool(int(inside_row.get("usable_records", 0)) >= 20)
    diaspora_secondary_only = bool(float(diaspora_row.get("coherence_score", 0.0)) < 70 or int(diaspora_row.get("usable_records", 0)) < 20)
    network_excludable = bool(
        int(network_row.get("usable_records", 0)) == 0
        or float(network_row.get("coherence_score", 0.0)) < 55
        or str(network_row.get("stability_assessment", "")) != "benchmark_candidate"
    )
    market_excludable = bool(
        int(market_row.get("usable_records", 0)) == 0
        or float(market_row.get("coherence_score", 0.0)) < 55
        or to_float(market_row.get("top_channel_share", "0"), 0.0) > 0.50
        or str(market_row.get("stability_assessment", "")) != "benchmark_candidate"
    )
    exclude_network_market = bool(network_excludable and market_excludable)

    reference = load_reference_benchmark(benchmark_records_csv)

    next_recommendation = "continue segmented research and expand inside-Iran direct-shop coverage"
    if best_basket and best_basket.get("stability_assessment") == "benchmark_candidate":
        next_recommendation = f"promote {best_basket['basket']} to persistent shadow benchmark candidate"
    elif inside_large_enough:
        next_recommendation = "focus on inside_iran_direct_shop as primary dealer-overlay candidate; keep others diagnostic"
    elif exclude_network_market:
        next_recommendation = "exclude network/market baskets from benchmark construction and expand direct-shop sampling"

    summary = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat(),
        "mode": args.mode,
        "input_record_count": len(input_records),
        "usable_record_count": len(usable_records),
        "outliers_removed": outliers_removed,
        "outliers_by_basket": outliers_by_basket,
        "filter_reason_counts": reason_counts,
        "baskets": basket_rows,
        "pairwise_spreads": spreads,
        "best_basket_by_coherence": best_basket,
        "inside_iran_direct_shop_large_enough_to_pursue": inside_large_enough,
        "diaspora_direct_shop_secondary_signal_only": diaspora_secondary_only,
        "network_and_market_should_be_excluded_from_benchmark_construction": exclude_network_market,
        "reference_ready_only_benchmark": {
            "record_count": int(reference.get("reference_record_count") or 0),
            "weighted_estimate": safe_round(reference.get("reference_weighted_estimate"), 2),
            "median": safe_round(reference.get("reference_median"), 2),
        },
        "clear_next_recommendation": next_recommendation,
    }

    summary_json = survey_dir / "telegram_segmented_benchmark_summary.json"
    baskets_csv = survey_dir / "telegram_segmented_benchmark_baskets.csv"
    spreads_csv = survey_dir / "telegram_segmented_spreads.csv"
    ranking_csv = survey_dir / "telegram_segmented_channel_ranking.csv"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(
        baskets_csv,
        fieldnames=[
            "basket",
            "input_records",
            "usable_records",
            "median",
            "trimmed_mean",
            "weighted_estimate",
            "std",
            "cv",
            "iqr",
            "mad",
            "top_channel",
            "top_channel_share",
            "effective_channel_count",
            "concentration_level",
            "avg_freshness",
            "avg_structure",
            "coherence_score",
            "stability_assessment",
        ],
        rows=basket_rows,
    )

    write_csv(
        spreads_csv,
        fieldnames=[
            "left_basket",
            "right_basket",
            "left_weighted_estimate",
            "right_weighted_estimate",
            "spread_irr",
            "spread_pct",
            "left_usable_records",
            "right_usable_records",
        ],
        rows=spreads,
    )

    write_csv(
        ranking_csv,
        fieldnames=[
            "handle",
            "title",
            "recommended_status",
            "source_priority",
            "basket",
            "record_count",
            "weighted_estimate",
            "avg_quality",
            "avg_freshness",
            "avg_structure",
            "channel_weight_share_in_basket",
            "channel_coherence_contribution_score",
            "recommended_role",
        ],
        rows=channel_rank_rows,
    )

    print(json.dumps(
        {
            "best_basket_by_coherence": best_basket["basket"] if best_basket else None,
            "inside_iran_direct_shop_large_enough_to_pursue": inside_large_enough,
            "diaspora_direct_shop_secondary_signal_only": diaspora_secondary_only,
            "exclude_network_and_market_from_benchmark_construction": exclude_network_market,
            "clear_next_recommendation": next_recommendation,
        },
        ensure_ascii=False,
        indent=2,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
