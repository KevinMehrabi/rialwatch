#!/usr/bin/env python3
"""Build a transparent multi-source USD/IRR benchmark from stored RialWatch feeds."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scripts import pipeline
except ImportError:  # pragma: no cover
    import pipeline  # type: ignore


UTC = dt.timezone.utc
DEFAULT_HISTORY_DAYS = 14
TRIM_FRACTION = 0.10

SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bonbast": {
        "source_label": "Bonbast",
        "market_type": "iran_open_market",
        "quote_type": "two_way_public_rate_board",
        "unit_type": "toman",
        "benchmark_eligible": True,
        "diagnostics_only": False,
        "exclusion_reason_default": None,
        "base_weight": 1.00,
    },
    "alanchand_street": {
        "source_label": "AlanChand Street",
        "market_type": "iran_open_market",
        "quote_type": "public_buy_sell_web_quote",
        "unit_type": "rial",
        "benchmark_eligible": True,
        "diagnostics_only": False,
        "exclusion_reason_default": None,
        "base_weight": 0.92,
    },
    "navasan": {
        "source_label": "Navasan",
        "market_type": "mixed_market_api",
        "quote_type": "aggregated_api_quote",
        "unit_type": "mixed",
        "benchmark_eligible": False,
        "diagnostics_only": True,
        "exclusion_reason_default": "open-market quote currently ineligible pending source quality review",
        "base_weight": 0.75,
    },
    "alanchand": {
        "source_label": "AlanChand API",
        "market_type": "regional_transfer_api",
        "quote_type": "regional_transfer_api_quote",
        "unit_type": "toman",
        "benchmark_eligible": False,
        "diagnostics_only": True,
        "exclusion_reason_default": "source family not used for open-market benchmark",
        "base_weight": 0.70,
    },
}

SOURCE_LABELS: Dict[str, str] = {name: str(config["source_label"]) for name, config in SOURCE_REGISTRY.items()}
BASE_WEIGHTS: Dict[str, float] = {name: float(config["base_weight"]) for name, config in SOURCE_REGISTRY.items()}


@dataclass
class SourceObservation:
    source: str
    label: str
    selected_value_rial: Optional[float]
    diagnostic_value_rial: Optional[float]
    buy_rial: Optional[float]
    sell_rial: Optional[float]
    midpoint_rial: Optional[float]
    sample_count: int
    estimate_kind: Optional[str]
    status: str
    reason: Optional[str]
    quote_time: Optional[str]
    sampled_at: Optional[str]
    base_weight: float


@dataclass
class BenchmarkArtifacts:
    benchmark: Dict[str, Any]
    diagnostics: Dict[str, Any]
    card: Dict[str, Any]
    daily_history: Dict[str, Any]


def safe_round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def iso_to_datetime(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def median(values: Sequence[float]) -> Optional[float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not cleaned:
        return None
    return statistics.median(cleaned)


def mean(values: Sequence[float]) -> Optional[float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not cleaned:
        return None
    return statistics.fmean(cleaned)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    cleaned = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    fraction = min(max(fraction, 0.0), 1.0)
    position = (len(cleaned) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return cleaned[lower]
    weight = position - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def trimmed_mean(values: Sequence[float], trim_fraction: float = TRIM_FRACTION) -> Optional[float]:
    cleaned = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not cleaned:
        return None
    if len(cleaned) <= 2:
        return statistics.fmean(cleaned)
    trim_count = int(len(cleaned) * trim_fraction)
    if trim_count <= 0 or (trim_count * 2) >= len(cleaned):
        return statistics.fmean(cleaned)
    return statistics.fmean(cleaned[trim_count:-trim_count])


def coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(cleaned) < 2:
        return 0.0 if cleaned else None
    center = statistics.fmean(cleaned)
    if center == 0:
        return None
    return statistics.pstdev(cleaned) / center


def normalize_with_unit(value: Any, unit: Optional[str]) -> Optional[float]:
    parsed = pipeline.parse_number(value)
    if parsed is None:
        return None
    return pipeline.normalize_unit(parsed, unit or "rial")


def first_number(*candidates: Any) -> Optional[float]:
    for candidate in candidates:
        parsed = pipeline.parse_number(candidate)
        if parsed is not None:
            return parsed
    return None


def extract_sample_point_estimate(sample: Dict[str, Any]) -> Dict[str, Optional[float]]:
    health = sample.get("health", {})
    if not isinstance(health, dict):
        health = {}
    benchmarks = sample.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        benchmarks = {}

    sample_unit = str(sample.get("source_unit") or health.get("source_unit") or "rial")
    benchmark_units = health.get("benchmark_units", {})
    if not isinstance(benchmark_units, dict):
        benchmark_units = {}
    raw_values = health.get("raw_extracted_values", {})
    if not isinstance(raw_values, dict):
        raw_values = {}
    extracted_values = health.get("extracted_values", {})
    if not isinstance(extracted_values, dict):
        extracted_values = {}
    parse_result = health.get("parse_result", {})
    if not isinstance(parse_result, dict):
        parse_result = {}

    buy_rial = first_number(
        parse_result.get("buy_rial"),
        extracted_values.get("open_market_buy"),
        normalize_with_unit(raw_values.get("open_market_buy"), benchmark_units.get("open_market") or sample_unit),
    )
    sell_rial = first_number(
        parse_result.get("sell_rial"),
        extracted_values.get("open_market_sell"),
        normalize_with_unit(raw_values.get("open_market_sell"), benchmark_units.get("open_market") or sample_unit),
        benchmarks.get("open_market"),
        sample.get("value"),
    )
    midpoint_rial = first_number(
        parse_result.get("mid_rial"),
        health.get("bonbast_usd_mid"),
        extracted_values.get("open_market_mid"),
    )
    if midpoint_rial is None and buy_rial is not None and sell_rial is not None:
        midpoint_rial = (buy_rial + sell_rial) / 2.0

    benchmark_open_market_rial = first_number(
        benchmarks.get("open_market"),
        extracted_values.get("open_market"),
        normalize_with_unit(raw_values.get("open_market"), benchmark_units.get("open_market") or sample_unit),
        sample.get("value"),
    )
    preferred_rial = midpoint_rial if midpoint_rial is not None else benchmark_open_market_rial

    return {
        "preferred_rial": preferred_rial,
        "benchmark_open_market_rial": benchmark_open_market_rial,
        "buy_rial": buy_rial,
        "sell_rial": sell_rial,
        "midpoint_rial": midpoint_rial,
    }


def summarize_source_observation(source: str, payload: Dict[str, Any]) -> SourceObservation:
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        samples = []

    valid_estimates: List[float] = []
    diagnostic_estimates: List[float] = []
    valid_buys: List[float] = []
    valid_sells: List[float] = []
    valid_midpoints: List[float] = []
    quote_times: List[dt.datetime] = []
    sampled_times: List[dt.datetime] = []
    midpoint_seen = False
    benchmark_seen = False
    invalid_reasons: List[str] = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        point = extract_sample_point_estimate(sample)
        candidate = point["preferred_rial"]
        diagnostic_candidate = point["benchmark_open_market_rial"] or candidate
        sampled_at = iso_to_datetime(sample.get("sampled_at"))
        quote_time = iso_to_datetime(sample.get("quote_time"))
        if sampled_at is not None:
            sampled_times.append(sampled_at)
        if quote_time is not None:
            quote_times.append(quote_time)

        ok = sample.get("ok") is True
        stale = sample.get("stale") is True
        error_text = str(sample.get("error") or "").strip() or None

        if candidate is not None and ok and not stale:
            valid_estimates.append(candidate)
            if point["buy_rial"] is not None:
                valid_buys.append(point["buy_rial"])
            if point["sell_rial"] is not None:
                valid_sells.append(point["sell_rial"])
            if point["midpoint_rial"] is not None:
                midpoint_seen = True
                valid_midpoints.append(point["midpoint_rial"])
            elif point["benchmark_open_market_rial"] is not None:
                benchmark_seen = True
        elif diagnostic_candidate is not None:
            diagnostic_estimates.append(diagnostic_candidate)
            if stale:
                invalid_reasons.append("stale")
            elif error_text:
                invalid_reasons.append(error_text)
            else:
                invalid_reasons.append("invalid sample")
        elif error_text:
            invalid_reasons.append(error_text)

    selected_value = median(valid_estimates)
    diagnostic_value = selected_value if selected_value is not None else median(diagnostic_estimates)
    if selected_value is not None:
        status = "included_candidate"
        reason = None
    elif diagnostic_value is not None:
        status = "diagnostic_only"
        reason = ", ".join(sorted(set(invalid_reasons))) if invalid_reasons else "open-market quote not eligible"
    else:
        status = "unavailable"
        reason = payload.get("note") or ", ".join(sorted(set(invalid_reasons))) or "no open-market quote found"

    if midpoint_seen:
        estimate_kind = "midpoint"
    elif benchmark_seen:
        estimate_kind = "reported_open_market"
    elif diagnostic_value is not None:
        estimate_kind = "diagnostic_only"
    else:
        estimate_kind = None

    latest_quote_time = max(quote_times).isoformat() if quote_times else None
    latest_sampled_at = max(sampled_times).isoformat() if sampled_times else None

    return SourceObservation(
        source=source,
        label=SOURCE_LABELS.get(source, source.replace("_", " ").title()),
        selected_value_rial=selected_value,
        diagnostic_value_rial=diagnostic_value,
        buy_rial=median(valid_buys),
        sell_rial=median(valid_sells),
        midpoint_rial=median(valid_midpoints),
        sample_count=len(samples),
        estimate_kind=estimate_kind,
        status=status,
        reason=reason,
        quote_time=latest_quote_time,
        sampled_at=latest_sampled_at,
        base_weight=BASE_WEIGHTS.get(source, 0.80),
    )


def detect_outliers(values_by_source: Dict[str, float]) -> Tuple[List[str], Dict[str, Dict[str, Optional[float]]]]:
    if len(values_by_source) < 3:
        return [], {}
    values = list(values_by_source.values())
    center = median(values)
    if center is None:
        return [], {}
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad in (None, 0):
        return [], {}

    outliers: List[str] = []
    details: Dict[str, Dict[str, Optional[float]]] = {}
    for source, value in values_by_source.items():
        modified_z = 0.6745 * (value - center) / mad
        details[source] = {
            "value_rial": value,
            "median_rial": center,
            "mad_rial": mad,
            "modified_z_score": modified_z,
        }
        if abs(modified_z) > 3.5:
            outliers.append(source)
    return sorted(outliers), details


def build_weights(observations: Sequence[SourceObservation]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for observation in observations:
        if observation.selected_value_rial is None:
            continue
        weight = observation.base_weight
        if observation.estimate_kind == "midpoint":
            weight *= 1.08
        elif observation.estimate_kind == "reported_open_market":
            weight *= 0.98
        if observation.buy_rial is not None and observation.sell_rial is not None:
            mid = observation.midpoint_rial or ((observation.buy_rial + observation.sell_rial) / 2.0)
            if mid:
                spread_pct = abs(observation.sell_rial - observation.buy_rial) / mid
                weight *= max(0.70, 1.0 - min(spread_pct, 0.08) * 2.5)
        weights[observation.source] = weight
    return weights


def weighted_mean(values_by_source: Dict[str, float], weights_by_source: Dict[str, float]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for source, value in values_by_source.items():
        weight = weights_by_source.get(source, 0.0)
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def load_recent_history(site_dir: Path, history_days: int, latest_day: dt.date) -> List[Tuple[dt.date, float]]:
    full_history = pipeline.load_benchmark_fix_history(site_dir, "open_market")
    if history_days <= 0:
        return full_history
    cutoff = latest_day - dt.timedelta(days=max(history_days - 1, 0))
    return [(day, value) for day, value in full_history if cutoff <= day <= latest_day]


def summarize_history(history: Sequence[Tuple[dt.date, float]]) -> Dict[str, Optional[float]]:
    values = [value for _, value in history]
    changes: List[float] = []
    for index in range(1, len(history)):
        prior = history[index - 1][1]
        current = history[index][1]
        if prior:
            changes.append(abs((current - prior) / prior))
    return {
        "history_points": float(len(history)),
        "history_cv": coefficient_of_variation(values),
        "median_abs_daily_change_pct": median(changes),
        "max_abs_daily_change_pct": max(changes) if changes else None,
    }


def compute_confidence_score(
    source_count: int,
    dispersion_cv: Optional[float],
    historical_stability: Dict[str, Optional[float]],
    outlier_count: int,
) -> Tuple[float, Dict[str, float]]:
    source_score = min(source_count / 3.0, 1.0) * 40.0

    cv = dispersion_cv if dispersion_cv is not None else 1.0
    dispersion_score = max(0.0, 1.0 - min(cv, 0.08) / 0.08) * 35.0

    history_points = int(historical_stability.get("history_points") or 0)
    history_coverage = min(history_points / 7.0, 1.0)
    daily_change = historical_stability.get("median_abs_daily_change_pct")
    if daily_change is None:
        stability_score = 0.0
    else:
        stability_score = max(0.0, 1.0 - min(daily_change, 0.06) / 0.06) * 25.0 * history_coverage

    penalty = min(float(outlier_count) * 5.0, 15.0)
    raw_score = max(0.0, source_score + dispersion_score + stability_score - penalty)
    if source_count < 2:
        raw_score = min(raw_score, 55.0)
    return round(raw_score, 2), {
        "source_count_component": round(source_score, 2),
        "dispersion_component": round(dispersion_score, 2),
        "historical_stability_component": round(stability_score, 2),
        "outlier_penalty": round(penalty, 2),
    }


def select_latest_fix_path(site_dir: Path, explicit_date: Optional[str]) -> Path:
    fix_dir = site_dir / "fix"
    if explicit_date:
        path = fix_dir / f"{explicit_date}.json"
        if not path.exists():
            raise FileNotFoundError(f"fix file not found: {path}")
        return path

    candidates = sorted(path for path in fix_dir.glob("*.json") if path.stem.count("-") == 2)
    if not candidates:
        raise FileNotFoundError(f"no fix JSON files found under {fix_dir}")
    return candidates[-1]


def summarize_dispersion(values: Sequence[float]) -> Dict[str, Optional[float]]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not cleaned:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "median": None,
            "p25": None,
            "p75": None,
            "range": None,
            "cv": None,
        }
    med = median(cleaned)
    p25 = percentile(cleaned, 0.25)
    p75 = percentile(cleaned, 0.75)
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "median": med,
        "p25": p25,
        "p75": p75,
        "range": (max(cleaned) - min(cleaned)) if len(cleaned) >= 2 else 0.0,
        "cv": coefficient_of_variation(cleaned),
    }


def quote_basis_for_observation(observation: SourceObservation) -> Optional[str]:
    if observation.midpoint_rial is not None:
        return "midpoint"
    if observation.sell_rial is not None and observation.buy_rial is None:
        return "sell"
    if observation.buy_rial is not None and observation.sell_rial is None:
        return "buy"
    if observation.selected_value_rial is not None or observation.diagnostic_value_rial is not None:
        return "inferred"
    return None


def dispersion_level_from_cv(cv: Optional[float]) -> str:
    if cv is None:
        return "unknown"
    if cv <= 0.015:
        return "low"
    if cv <= 0.03:
        return "moderate"
    return "high"


def build_benchmark_status_text(
    benchmark_rate: Optional[float],
    confidence_score: float,
    source_count: int,
    dispersion_level: str,
) -> str:
    if benchmark_rate is None:
        return "Benchmark unavailable"
    if source_count >= 3 and confidence_score >= 80 and dispersion_level == "low":
        return "High-confidence benchmark"
    if source_count >= 2 and confidence_score >= 50 and dispersion_level in {"low", "moderate"}:
        return "Benchmark active with moderate confidence"
    if source_count >= 2:
        return "Benchmark active but elevated dispersion"
    return "Benchmark active but thinly sourced"


def build_diagnostics_warning(
    source_values: Sequence[Dict[str, Any]],
    outliers_removed: Sequence[str],
    source_count: int,
    dispersion_level: str,
) -> Optional[str]:
    diagnostic_only = [row["source_label"] for row in source_values if row.get("eligible_for_benchmark") is False]
    if source_count < 2:
        return "Benchmark is running on fewer than two eligible sources."
    if outliers_removed:
        return f"Outlier screening removed: {', '.join(outliers_removed)}."
    if dispersion_level == "high":
        return "Eligible sources remain materially dispersed."
    if diagnostic_only:
        return f"Diagnostics-only sources excluded from benchmark: {', '.join(diagnostic_only)}."
    return None


def build_daily_history_artifact(
    daily: Dict[str, Any],
    benchmark_json: Dict[str, Any],
    diagnostics_json: Dict[str, Any],
    card_json: Dict[str, Any],
    confidence_components: Dict[str, float],
) -> Dict[str, Any]:
    eligible_sources = [
        row["source_name"]
        for row in diagnostics_json.get("source_values", [])
        if row.get("included_in_benchmark") is True
    ]
    excluded_sources = [
        {
            "source_name": row["source_name"],
            "exclusion_reason": row.get("exclusion_reason"),
        }
        for row in diagnostics_json.get("source_values", [])
        if row.get("included_in_benchmark") is not True
    ]
    return {
        "date": daily.get("date"),
        "timestamp": benchmark_json.get("timestamp"),
        "median_rate": benchmark_json.get("median_rate"),
        "trimmed_mean_rate": benchmark_json.get("trimmed_mean_rate"),
        "weighted_rate": benchmark_json.get("weighted_rate"),
        "confidence_score": benchmark_json.get("confidence_score"),
        "confidence_source_count_component": confidence_components.get("source_count_component"),
        "confidence_dispersion_component": confidence_components.get("dispersion_component"),
        "confidence_stability_component": confidence_components.get("historical_stability_component"),
        "confidence_total": benchmark_json.get("confidence_score"),
        "source_count": benchmark_json.get("source_count"),
        "eligible_sources": eligible_sources,
        "diagnostics_summary": {
            "dispersion_level": card_json.get("dispersion_level"),
            "dispersion_cv": diagnostics_json.get("dispersion", {}).get("cv"),
            "outliers_removed": diagnostics_json.get("outliers_removed", {}).get("sources", []),
            "excluded_sources": excluded_sources,
            "diagnostics_warning": card_json.get("diagnostics_warning"),
        },
    }


def build_source_registry_payload() -> Dict[str, Any]:
    rows = []
    for source_name, config in SOURCE_REGISTRY.items():
        rows.append(
            {
                "source_name": source_name,
                "market_type": config["market_type"],
                "quote_type": config["quote_type"],
                "unit_type": config["unit_type"],
                "benchmark_eligible": config["benchmark_eligible"],
                "diagnostics_only": config["diagnostics_only"],
                "exclusion_reason_default": config["exclusion_reason_default"],
            }
        )
    return {"sources": rows}


def build_methodology_payload(current_diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    registry_payload = build_source_registry_payload()
    current_benchmark_eligible_sources = [
        row["source_name"]
        for row in registry_payload["sources"]
        if row["benchmark_eligible"] is True
    ]
    current_diagnostics_only_sources = [
        row["source_name"]
        for row in registry_payload["sources"]
        if row["diagnostics_only"] is True
    ]
    return {
        "eligibility_rules": [
            "Only open-market USD/IRR quotes with a usable normalized value are considered for benchmark inclusion.",
            "Sources must expose a valid non-stale open-market quote in the stored daily feed to become eligible for a given day.",
            "Diagnostics-only sources remain visible in diagnostics and methodology artifacts but are not included in benchmark aggregation.",
        ],
        "quote_normalization_rules": [
            "All quote values are normalized to IRR before aggregation.",
            "Midpoints are preferred when both buy and sell are available.",
            "If no midpoint exists, the best available open-market quote is used with its quote basis preserved.",
        ],
        "accepted_quote_bases": ["midpoint", "sell", "buy", "inferred"],
        "outlier_handling_method": {
            "method": "median_absolute_deviation",
            "threshold": 3.5,
            "notes": "MAD screening is only applied when at least three eligible benchmark candidates are available.",
        },
        "confidence_score_methodology": {
            "source_count_component_max": 40.0,
            "dispersion_component_max": 35.0,
            "stability_component_max": 25.0,
            "outlier_penalty_max": 15.0,
            "notes": "Confidence score combines eligible source breadth, current dispersion, recent benchmark stability, and outlier penalties.",
        },
        "current_benchmark_eligible_sources": current_benchmark_eligible_sources,
        "current_diagnostics_only_sources": current_diagnostics_only_sources,
        "current_sample_status": current_diagnostics.get("source_values", []),
    }


def build_benchmark_timeseries(history_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for row in history_payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        benchmark_rate = row.get("weighted_rate")
        if benchmark_rate is None:
            benchmark_rate = row.get("median_rate")
        diagnostics_summary = row.get("diagnostics_summary", {})
        if not isinstance(diagnostics_summary, dict):
            diagnostics_summary = {}
        rows.append(
            {
                "date": row.get("date"),
                "benchmark_rate": benchmark_rate,
                "confidence_score": row.get("confidence_score"),
                "source_count": row.get("source_count"),
                "dispersion_level": diagnostics_summary.get("dispersion_level"),
            }
        )
    return {"rows": rows}


def build_benchmark_outputs(daily: Dict[str, Any], site_dir: Path, history_days: int) -> BenchmarkArtifacts:
    daily_sources = daily.get("sources", {})
    if not isinstance(daily_sources, dict):
        daily_sources = {}

    observations = [summarize_source_observation(source, payload) for source, payload in sorted(daily_sources.items())]
    candidate_map = {
        observation.source: observation.selected_value_rial
        for observation in observations
        if observation.selected_value_rial is not None
    }
    outliers_removed, outlier_details = detect_outliers(candidate_map)
    filtered_observations = [obs for obs in observations if obs.selected_value_rial is not None and obs.source not in outliers_removed]
    filtered_map = {obs.source: obs.selected_value_rial for obs in filtered_observations if obs.selected_value_rial is not None}
    filtered_values = list(filtered_map.values())
    filtered_median = median(filtered_values)
    filtered_trimmed_mean = trimmed_mean(filtered_values, TRIM_FRACTION)
    weights = build_weights(filtered_observations)
    filtered_weighted_mean = weighted_mean(filtered_map, weights)

    latest_day = pipeline.parse_iso_date_text(daily.get("date"))
    if latest_day is None:
        raise ValueError("daily fix JSON missing valid date")
    history = load_recent_history(site_dir, history_days, latest_day)
    historical_stability = summarize_history(history)

    confidence_score, confidence_components = compute_confidence_score(
        source_count=len(filtered_observations),
        dispersion_cv=coefficient_of_variation(filtered_values),
        historical_stability=historical_stability,
        outlier_count=len(outliers_removed),
    )

    diagnostic_reference_median = filtered_median or median(
        [obs.diagnostic_value_rial for obs in observations if obs.diagnostic_value_rial is not None]
    )
    source_values: List[Dict[str, Any]] = []
    source_deviation: Dict[str, Dict[str, Optional[float]]] = {}

    for observation in observations:
        comparable_value = observation.selected_value_rial if observation.selected_value_rial is not None else observation.diagnostic_value_rial
        if diagnostic_reference_median not in (None, 0) and comparable_value is not None:
            deviation_from_median = comparable_value - diagnostic_reference_median
            deviation_from_median_pct = deviation_from_median / diagnostic_reference_median
        else:
            deviation_from_median = None
            deviation_from_median_pct = None
        if filtered_weighted_mean not in (None, 0) and comparable_value is not None:
            deviation_from_weighted = comparable_value - filtered_weighted_mean
            deviation_from_weighted_pct = deviation_from_weighted / filtered_weighted_mean
        else:
            deviation_from_weighted = None
            deviation_from_weighted_pct = None
        included_in_benchmark = observation.source in filtered_map
        eligible_for_benchmark = observation.selected_value_rial is not None
        outlier_flag = observation.source in outliers_removed
        if included_in_benchmark:
            exclusion_reason = None
        elif outlier_flag:
            exclusion_reason = "outlier removed by MAD filter"
        else:
            exclusion_reason = observation.reason
        entry = {
            "source_name": observation.source,
            "source_label": observation.label,
            "source": observation.source,
            "label": observation.label,
            "normalized_value_irr": safe_round(comparable_value, 2),
            "selected_value_rial": safe_round(observation.selected_value_rial, 2),
            "diagnostic_value_rial": safe_round(observation.diagnostic_value_rial, 2),
            "buy_rial": safe_round(observation.buy_rial, 2),
            "sell_rial": safe_round(observation.sell_rial, 2),
            "midpoint_rial": safe_round(observation.midpoint_rial, 2),
            "sample_count": observation.sample_count,
            "quote_basis": quote_basis_for_observation(observation),
            "estimate_kind": observation.estimate_kind,
            "status": observation.status,
            "reason": observation.reason,
            "eligible_for_benchmark": eligible_for_benchmark,
            "included_in_benchmark": included_in_benchmark,
            "exclusion_reason": exclusion_reason,
            "quote_time": observation.quote_time,
            "sampled_at": observation.sampled_at,
            "deviation_from_median": safe_round(deviation_from_median, 2),
            "deviation_from_median_pct": safe_round(deviation_from_median_pct, 6),
            "deviation_from_weighted": safe_round(deviation_from_weighted, 2),
            "deviation_from_weighted_pct": safe_round(deviation_from_weighted_pct, 6),
            "outlier_flag": outlier_flag,
            "outlier_removed": outlier_flag,
            "base_weight": safe_round(observation.base_weight, 4),
            "final_weight": safe_round(weights.get(observation.source), 4),
        }
        source_values.append(entry)
        source_deviation[observation.source] = {
            "value_rial": safe_round(comparable_value, 2),
            "deviation_from_median_rial": safe_round(deviation_from_median, 2),
            "deviation_from_median_pct": safe_round(deviation_from_median_pct, 6),
            "deviation_from_weighted_rial": safe_round(deviation_from_weighted, 2),
            "deviation_from_weighted_pct": safe_round(deviation_from_weighted_pct, 6),
        }

    bonbast_obs = next((obs for obs in observations if obs.source == "bonbast"), None)
    navasan_obs = next((obs for obs in observations if obs.source == "navasan"), None)
    bonbast_value = bonbast_obs.selected_value_rial if bonbast_obs is not None else None
    navasan_value = None
    navasan_basis = None
    if navasan_obs is not None:
        if navasan_obs.selected_value_rial is not None:
            navasan_value = navasan_obs.selected_value_rial
            navasan_basis = "eligible_open_market"
        elif navasan_obs.diagnostic_value_rial is not None:
            navasan_value = navasan_obs.diagnostic_value_rial
            navasan_basis = "diagnostic_only"

    bonbast_navasan_difference = None
    bonbast_navasan_difference_pct = None
    if bonbast_value is not None and navasan_value not in (None, 0):
        bonbast_navasan_difference = bonbast_value - navasan_value
        bonbast_navasan_difference_pct = bonbast_navasan_difference / navasan_value

    dispersion = summarize_dispersion(filtered_values)
    benchmark_json = {
        "timestamp": daily.get("as_of") or daily.get("date"),
        "median_rate": safe_round(filtered_median, 2),
        "trimmed_mean_rate": safe_round(filtered_trimmed_mean, 2),
        "weighted_rate": safe_round(filtered_weighted_mean, 2),
        "confidence_score": confidence_score,
        "source_count": len(filtered_observations),
    }
    dispersion_level = dispersion_level_from_cv(dispersion.get("cv"))
    card_benchmark_rate = benchmark_json["weighted_rate"] if benchmark_json["weighted_rate"] is not None else benchmark_json["median_rate"]
    card_json = {
        "benchmark_rate": card_benchmark_rate,
        "confidence_score": confidence_score,
        "source_count": len(filtered_observations),
        "dispersion_level": dispersion_level,
        "benchmark_status_text": build_benchmark_status_text(
            benchmark_rate=card_benchmark_rate,
            confidence_score=confidence_score,
            source_count=len(filtered_observations),
            dispersion_level=dispersion_level,
        ),
        "diagnostics_warning": None,
    }
    card_json["diagnostics_warning"] = build_diagnostics_warning(
        source_values=source_values,
        outliers_removed=outliers_removed,
        source_count=len(filtered_observations),
        dispersion_level=dispersion_level,
    )
    diagnostics_json = {
        "timestamp": daily.get("as_of") or daily.get("date"),
        "input_date": daily.get("date"),
        "confidence_source_count_component": confidence_components.get("source_count_component"),
        "confidence_dispersion_component": confidence_components.get("dispersion_component"),
        "confidence_stability_component": confidence_components.get("historical_stability_component"),
        "confidence_total": confidence_score,
        "source_values": source_values,
        "source_deviation": source_deviation,
        "dispersion": {
            key: safe_round(value, 6) if isinstance(value, float) else value for key, value in dispersion.items()
        },
        "outliers_removed": {
            "sources": outliers_removed,
            "details": {
                key: {inner_key: safe_round(inner_value, 6) for inner_key, inner_value in detail.items()}
                for key, detail in outlier_details.items()
            },
        },
        "divergence": {
            "bonbast_vs_navasan": {
                "difference_rial": safe_round(bonbast_navasan_difference, 2),
                "difference_pct": safe_round(bonbast_navasan_difference_pct, 6),
                "navasan_basis": navasan_basis,
                "navasan_reason": navasan_obs.reason if navasan_obs is not None else "navasan source unavailable",
            }
        },
        "historical_stability": {
            "window_days": history_days,
            "history_points": int(historical_stability.get("history_points") or 0),
            "history_cv": safe_round(historical_stability.get("history_cv"), 6),
            "median_abs_daily_change_pct": safe_round(historical_stability.get("median_abs_daily_change_pct"), 6),
            "max_abs_daily_change_pct": safe_round(historical_stability.get("max_abs_daily_change_pct"), 6),
            "history_rows": [
                {"date": day.isoformat(), "fix": safe_round(value, 2)} for day, value in history
            ],
        },
        "confidence": {
            "score": confidence_score,
            "components": confidence_components,
        },
    }
    daily_history = build_daily_history_artifact(
        daily=daily,
        benchmark_json=benchmark_json,
        diagnostics_json=diagnostics_json,
        card_json=card_json,
        confidence_components=confidence_components,
    )
    return BenchmarkArtifacts(
        benchmark=benchmark_json,
        diagnostics=diagnostics_json,
        card=card_json,
        daily_history=daily_history,
    )


def iter_fix_paths(site_dir: Path) -> List[Path]:
    fix_dir = site_dir / "fix"
    return sorted(path for path in fix_dir.glob("*.json") if path.stem.count("-") == 2)


def write_history_outputs(site_dir: Path, fix_paths: Sequence[Path], history_days: int) -> Dict[str, Any]:
    benchmark_dir = site_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    history_rows: List[Dict[str, Any]] = []

    for fix_path in fix_paths:
        daily = json.loads(fix_path.read_text(encoding="utf-8"))
        artifacts = build_benchmark_outputs(daily, site_dir, history_days)
        day = str(daily.get("date") or fix_path.stem)
        day_path = benchmark_dir / f"{day}.json"
        day_path.write_text(json.dumps(artifacts.daily_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        history_rows.append(
            {
                "date": day,
                "timestamp": artifacts.daily_history.get("timestamp"),
                "median_rate": artifacts.daily_history.get("median_rate"),
                "trimmed_mean_rate": artifacts.daily_history.get("trimmed_mean_rate"),
                "weighted_rate": artifacts.daily_history.get("weighted_rate"),
                "confidence_score": artifacts.daily_history.get("confidence_score"),
                "confidence_source_count_component": artifacts.daily_history.get("confidence_source_count_component"),
                "confidence_dispersion_component": artifacts.daily_history.get("confidence_dispersion_component"),
                "confidence_stability_component": artifacts.daily_history.get("confidence_stability_component"),
                "confidence_total": artifacts.daily_history.get("confidence_total"),
                "source_count": artifacts.daily_history.get("source_count"),
                "eligible_sources": artifacts.daily_history.get("eligible_sources"),
                "diagnostics_summary": artifacts.daily_history.get("diagnostics_summary"),
            }
        )

    history_payload = {"rows": history_rows}
    history_path = site_dir / "api" / "benchmark_history.json"
    history_path.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return history_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a transparent multi-source RialWatch benchmark model")
    parser.add_argument("--site-dir", default="site", help="Path to the RialWatch site directory")
    parser.add_argument("--date", default=None, help="Specific fix date to use (YYYY-MM-DD)")
    parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS, help="Historical window for confidence scoring")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    site_dir = Path(args.site_dir).resolve()
    fix_path = select_latest_fix_path(site_dir, args.date)
    api_dir = site_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    fix_paths = iter_fix_paths(site_dir)
    history_payload = write_history_outputs(site_dir, fix_paths, args.history_days)

    daily = json.loads(fix_path.read_text(encoding="utf-8"))
    artifacts = build_benchmark_outputs(daily, site_dir, args.history_days)
    benchmark_path = api_dir / "benchmark.json"
    diagnostics_path = api_dir / "benchmark_diagnostics.json"
    card_path = api_dir / "benchmark_card.json"
    methodology_path = api_dir / "benchmark_methodology.json"
    source_registry_path = api_dir / "benchmark_source_registry.json"
    timeseries_path = api_dir / "benchmark_timeseries.json"
    benchmark_path.write_text(json.dumps(artifacts.benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics_path.write_text(json.dumps(artifacts.diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card_path.write_text(json.dumps(artifacts.card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    methodology_payload = build_methodology_payload(artifacts.diagnostics)
    source_registry_payload = build_source_registry_payload()
    timeseries_payload = build_benchmark_timeseries(history_payload)
    methodology_path.write_text(json.dumps(methodology_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_registry_path.write_text(json.dumps(source_registry_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timeseries_path.write_text(json.dumps(timeseries_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"input_fix={fix_path.name}")
    print(f"source_count={artifacts.benchmark['source_count']}")
    print(f"median_rate={artifacts.benchmark['median_rate']}")
    print(f"trimmed_mean_rate={artifacts.benchmark['trimmed_mean_rate']}")
    print(f"weighted_rate={artifacts.benchmark['weighted_rate']}")
    print(f"confidence_score={artifacts.benchmark['confidence_score']}")
    print(f"benchmark_json={benchmark_path}")
    print(f"benchmark_diagnostics_json={diagnostics_path}")
    print(f"benchmark_card_json={card_path}")
    print(f"benchmark_history_rows={len(history_payload['rows'])}")
    print(f"benchmark_methodology_json={methodology_path}")
    print(f"benchmark_source_registry_json={source_registry_path}")
    print(f"benchmark_timeseries_json={timeseries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
