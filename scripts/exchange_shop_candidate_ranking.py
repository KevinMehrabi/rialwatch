#!/usr/bin/env python3
"""Rank multilingual exchange-shop discovery candidates for Exchange Shops baskets.

This script is diagnostics-only. It does not modify benchmark artifacts. It
scores newly discovered sources into operational buckets for future locality-
based Exchange Shops baskets.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TARGET_LOCALITIES = ("Iran", "UAE", "Turkey", "Iraq", "Afghanistan", "UK", "Germany", "unknown")

LOCALITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Iran": ("تهران", "tehran", "mashhad", "isfahan", "shiraz", "tabriz", "karaj", "qom", "ahvaz", "rasht", "kerman", "مشهد", "اصفهان", "شیراز", "تبریز", "کرج", "قم", "اهواز", "رشت", "کرمان"),
    "UAE": ("uae", "dubai", "امارات", "دبی", "دوبی"),
    "Turkey": ("turkey", "turkish", "istanbul", "ترکیه", "استانبول"),
    "Iraq": ("iraq", "sulaymaniyah", "sulaimaniyah", "سلیمانیه", "سليمانية", "عراق"),
    "Afghanistan": ("afghanistan", "herat", "افغانستان", "هرات"),
    "UK": ("uk", "london", "britain", "england", "انگلستان", "لندن"),
    "Germany": ("germany", "deutschland", "frankfurt", "hamburg", "آلمان", "فرانکفورت", "هامبورگ"),
}

LOCALITY_USEFULNESS = {
    "Iran": 10,
    "UAE": 9,
    "Turkey": 9,
    "Iraq": 10,
    "Afghanistan": 10,
    "UK": 7,
    "Germany": 7,
    "unknown": 2,
}

SUSPICIOUS_TITLE_KEYWORDS = (
    "pavel durov",
    "hijab girls",
    "school",
    "consulate",
    "council",
    "business council",
)


@dataclass
class RankedCandidate:
    handle_or_url: str
    title: str
    platform: str
    source_type: str
    locality_bucket: str
    country_guess: str
    city_guess: str
    language_guess: str
    parseability_score: int
    quote_post_count: int
    buy_sell_pair_count: int
    likely_individual_shop: bool
    freshness_score: int
    locality_usefulness_score: int
    basket_contribution_score: int
    feed_stability_score: int
    operational_score: int
    operational_bucket: str
    notes: str
    last_seen: str
    status_guess: str


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def safe_int(value: Any) -> int:
    try:
        return int(float(str(value or "0").strip()))
    except ValueError:
        return 0


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def median_or_zero(values: Sequence[float]) -> float:
    cleaned = [float(v) for v in values if v is not None]
    return statistics.median(cleaned) if cleaned else 0.0


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def derive_locality_bucket(title: str, handle_or_url: str, country_guess: str, city_guess: str) -> str:
    text = " ".join(filter(None, [title, handle_or_url])).lower()
    best_bucket = "unknown"
    best_score = 0
    for locality, keywords in LOCALITY_KEYWORDS.items():
        score = sum(text.count(keyword.lower()) for keyword in keywords)
        if score > best_score:
            best_bucket = locality
            best_score = score
    if best_score > 0:
        return best_bucket
    normalized_city = str(city_guess or "").strip()
    if normalized_city in TARGET_LOCALITIES and normalized_city != "unknown":
        return normalized_city
    normalized_country = str(country_guess or "").strip()
    return normalized_country if normalized_country in TARGET_LOCALITIES else "unknown"


def compute_registry_baseline(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {key: 0 for key in TARGET_LOCALITIES}
    for row in rows:
        channel_type = str(row.get("channel_type_guess", "")).lower()
        likely_shop = safe_bool(row.get("likely_individual_shop"))
        if not likely_shop and "market" not in channel_type and "dealer" not in channel_type and "aggregator" not in channel_type:
            continue
        locality = derive_locality_bucket(
            title=str(row.get("title", "")),
            handle_or_url=str(row.get("handle", "")),
            country_guess=str(row.get("country_guess", "")),
            city_guess=str(row.get("city_guess", "")),
        )
        counts[locality] = counts.get(locality, 0) + 1
    return counts


def compute_reference_thresholds(
    pilot_metrics_rows: Sequence[Dict[str, str]],
    pilot_quote_rows: Sequence[Dict[str, str]],
) -> Dict[str, float]:
    ready_rows = [row for row in pilot_metrics_rows if row.get("recommended_status") == "ready_for_research_ingestion"]
    ready_handles = {row["handle"] for row in ready_rows}
    ready_quote_counts = [safe_int(row.get("quote_records_extracted")) for row in ready_rows]
    ready_pair_counts = [safe_int(row.get("buy_sell_pair_records")) for row in ready_rows]
    ready_readiness = [float(row.get("ingestion_readiness_score") or 0) for row in ready_rows]
    ready_quality = [float(row.get("average_record_quality_score") or 0) for row in ready_rows]

    dedup_kept_usd_counts: Dict[str, int] = {}
    for row in pilot_quote_rows:
        if row.get("handle") not in ready_handles:
            continue
        if str(row.get("dedup_keep", "")).lower() != "true":
            continue
        if "USD" not in str(row.get("currency", "")):
            continue
        handle = str(row["handle"])
        dedup_kept_usd_counts[handle] = dedup_kept_usd_counts.get(handle, 0) + 1

    return {
        "ready_quote_records_median": median_or_zero(ready_quote_counts),
        "ready_buy_sell_pairs_median": median_or_zero(ready_pair_counts),
        "ready_ingestion_readiness_median": median_or_zero(ready_readiness),
        "ready_record_quality_median": median_or_zero(ready_quality),
        "ready_deduped_usd_records_median": median_or_zero(list(dedup_kept_usd_counts.values())),
    }


def compute_freshness_score(last_seen: str, status_guess: str, now: dt.datetime) -> int:
    if not last_seen:
        return 0 if status_guess == "ok_no_text" else 1
    try:
        seen_at = dt.datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
    except ValueError:
        return 2
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=dt.timezone.utc)
    age_days = max(0.0, (now - seen_at.astimezone(dt.timezone.utc)).total_seconds() / 86400.0)
    if age_days <= 3:
        return 12
    if age_days <= 14:
        return 10
    if age_days <= 45:
        return 7
    if age_days <= 180:
        return 4
    if age_days <= 365:
        return 2
    return 0


def compute_basket_contribution_score(locality: str, baseline_counts: Dict[str, int]) -> int:
    count = baseline_counts.get(locality, 0)
    if locality == "unknown":
        return 1
    if count <= 1:
        return 10
    if count <= 3:
        return 9
    if count <= 6:
        return 7
    if count <= 10:
        return 5
    return 3


def compute_feed_stability_score(
    status_guess: str,
    quote_post_count: int,
    buy_sell_pair_count: int,
    freshness_score: int,
    parseability_score: int,
    reference_thresholds: Dict[str, float],
) -> int:
    ready_quote_ref = max(reference_thresholds.get("ready_quote_records_median", 0.0), 1.0)
    ready_pair_ref = max(reference_thresholds.get("ready_buy_sell_pairs_median", 0.0), 1.0)
    if status_guess != "ok":
        return 1 if status_guess == "ok_no_text" else 0
    score = 0
    if quote_post_count >= ready_quote_ref * 0.75:
        score += 4
    elif quote_post_count >= ready_quote_ref * 0.40:
        score += 2
    if buy_sell_pair_count >= ready_pair_ref * 0.50:
        score += 4
    elif buy_sell_pair_count > 0:
        score += 2
    if freshness_score >= 7:
        score += 2
    if parseability_score >= 70:
        score += 2
    return min(score, 12)


def compute_operational_score(
    row: Dict[str, str],
    locality_bucket: str,
    baseline_counts: Dict[str, int],
    reference_thresholds: Dict[str, float],
    now: dt.datetime,
) -> Tuple[int, int, int, int]:
    parseability = safe_int(row.get("parseability_score"))
    quote_post_count = safe_int(row.get("quote_post_count"))
    buy_sell_pair_count = safe_int(row.get("buy_sell_pair_count"))
    likely_individual_shop = safe_bool(row.get("likely_individual_shop"))
    freshness_score = compute_freshness_score(str(row.get("last_seen", "")), str(row.get("status_guess", "")), now)
    locality_usefulness = LOCALITY_USEFULNESS.get(locality_bucket, 2)
    basket_contribution = compute_basket_contribution_score(locality_bucket, baseline_counts)
    feed_stability = compute_feed_stability_score(
        status_guess=str(row.get("status_guess", "")),
        quote_post_count=quote_post_count,
        buy_sell_pair_count=buy_sell_pair_count,
        freshness_score=freshness_score,
        parseability_score=parseability,
        reference_thresholds=reference_thresholds,
    )

    ready_quote_ref = max(reference_thresholds.get("ready_quote_records_median", 0.0), 1.0)
    ready_pair_ref = max(reference_thresholds.get("ready_buy_sell_pairs_median", 0.0), 1.0)

    score = 0.0
    score += min(parseability, 100) * 0.35
    score += min((quote_post_count / ready_quote_ref) * 15.0, 15.0)
    score += min((buy_sell_pair_count / ready_pair_ref) * 20.0, 20.0)
    score += 12.0 if likely_individual_shop else 0.0
    score += freshness_score
    score += locality_usefulness
    score += basket_contribution
    score += feed_stability

    source_type = str(row.get("source_type", "unknown"))
    if source_type == "exchange_shop":
        score += 6.0
    elif source_type == "regional_market_channel":
        score += 8.0
    elif source_type == "settlement_exchange":
        score += 4.0
    elif source_type == "aggregator":
        score -= 8.0
    else:
        score -= 10.0

    suspicious_text = f"{row.get('title', '')} {row.get('handle_or_url', '')}".lower()
    if any(token in suspicious_text for token in SUSPICIOUS_TITLE_KEYWORDS):
        score -= 12.0

    return max(0, min(int(round(score)), 100)), freshness_score, locality_usefulness, basket_contribution, feed_stability


def build_notes(
    row: Dict[str, str],
    locality_bucket: str,
    operational_bucket: str,
    freshness_score: int,
    basket_contribution_score: int,
    feed_stability_score: int,
) -> str:
    notes: List[str] = []
    source_type = str(row.get("source_type", "unknown"))
    quote_post_count = safe_int(row.get("quote_post_count"))
    buy_sell_pair_count = safe_int(row.get("buy_sell_pair_count"))
    parseability = safe_int(row.get("parseability_score"))
    if source_type == "exchange_shop":
        notes.append("classified as exchange-shop candidate")
    elif source_type == "regional_market_channel":
        notes.append("classified as regional market-signal feed")
    elif source_type == "settlement_exchange":
        notes.append("classified as settlement/remittance exchange")
    else:
        notes.append(f"classified as {source_type}")

    if buy_sell_pair_count > 0:
        notes.append(f"visible buy/sell pairs: {buy_sell_pair_count}")
    elif quote_post_count > 0:
        notes.append(f"visible quote-like posts: {quote_post_count}")
    else:
        notes.append("no visible quote structure in sampled messages")

    if safe_bool(row.get("likely_individual_shop")):
        notes.append("shop-like contact footprint detected")
    if freshness_score >= 10:
        notes.append("recent activity signal")
    elif freshness_score <= 2:
        notes.append("weak freshness / stale visibility")

    if basket_contribution_score >= 8:
        notes.append(f"fills thin {locality_bucket} basket")
    if feed_stability_score >= 8:
        notes.append("strong stability hints from repeated quote formatting")
    elif feed_stability_score <= 2:
        notes.append("limited stability evidence")

    if parseability >= 75:
        notes.append("high parseability")
    elif parseability < 30:
        notes.append("low parseability")

    notes.append(f"ranked {operational_bucket}")
    return "; ".join(notes)


def assign_operational_bucket(
    row: Dict[str, str],
    operational_score: int,
    freshness_score: int,
) -> str:
    source_type = str(row.get("source_type", "unknown"))
    quote_post_count = safe_int(row.get("quote_post_count"))
    buy_sell_pair_count = safe_int(row.get("buy_sell_pair_count"))
    likely_individual_shop = safe_bool(row.get("likely_individual_shop"))

    if source_type in {"unknown", "aggregator"} and quote_post_count == 0 and not likely_individual_shop:
        return "reject"
    if (
        operational_score >= 75
        and freshness_score >= 4
        and source_type in {"exchange_shop", "regional_market_channel"}
        and (buy_sell_pair_count >= 4 or quote_post_count >= 8)
    ):
        return "P1"
    if (
        operational_score >= 60
        and freshness_score >= 2
        and source_type in {"exchange_shop", "regional_market_channel", "settlement_exchange"}
        and (quote_post_count > 0 or likely_individual_shop or freshness_score >= 7)
    ):
        return "P2"
    if operational_score >= 30 and source_type != "unknown":
        return "monitor_only"
    if operational_score >= 40 and likely_individual_shop:
        return "monitor_only"
    return "reject"


def rank_candidates(
    candidate_rows: Sequence[Dict[str, str]],
    baseline_counts: Dict[str, int],
    reference_thresholds: Dict[str, float],
    now: dt.datetime,
) -> List[RankedCandidate]:
    ranked: List[RankedCandidate] = []
    for row in candidate_rows:
        locality_bucket = derive_locality_bucket(
            title=str(row.get("title", "")),
            handle_or_url=str(row.get("handle_or_url", "")),
            country_guess=str(row.get("country_guess", "")),
            city_guess=str(row.get("city_guess", "")),
        )
        operational_score, freshness_score, locality_usefulness, basket_contribution, feed_stability = compute_operational_score(
            row=row,
            locality_bucket=locality_bucket,
            baseline_counts=baseline_counts,
            reference_thresholds=reference_thresholds,
            now=now,
        )
        operational_bucket = assign_operational_bucket(row, operational_score, freshness_score)
        notes = build_notes(
            row=row,
            locality_bucket=locality_bucket,
            operational_bucket=operational_bucket,
            freshness_score=freshness_score,
            basket_contribution_score=basket_contribution,
            feed_stability_score=feed_stability,
        )
        ranked.append(
            RankedCandidate(
                handle_or_url=str(row.get("handle_or_url", "")),
                title=str(row.get("title", "")),
                platform=str(row.get("platform", "")),
                source_type=str(row.get("source_type", "")),
                locality_bucket=locality_bucket,
                country_guess=str(row.get("country_guess", "")),
                city_guess=str(row.get("city_guess", "")),
                language_guess=str(row.get("language_guess", "")),
                parseability_score=safe_int(row.get("parseability_score")),
                quote_post_count=safe_int(row.get("quote_post_count")),
                buy_sell_pair_count=safe_int(row.get("buy_sell_pair_count")),
                likely_individual_shop=safe_bool(row.get("likely_individual_shop")),
                freshness_score=freshness_score,
                locality_usefulness_score=locality_usefulness,
                basket_contribution_score=basket_contribution,
                feed_stability_score=feed_stability,
                operational_score=operational_score,
                operational_bucket=operational_bucket,
                notes=notes,
                last_seen=str(row.get("last_seen", "")),
                status_guess=str(row.get("status_guess", "")),
            )
        )
    ranked.sort(
        key=lambda item: (
            {"P1": 0, "P2": 1, "monitor_only": 2, "reject": 3}.get(item.operational_bucket, 9),
            -item.operational_score,
            -item.buy_sell_pair_count,
            -item.quote_post_count,
            item.handle_or_url.lower(),
        )
    )
    return ranked


def write_ranking_csv(path: Path, ranked: Sequence[RankedCandidate]) -> None:
    fieldnames = [
        "handle_or_url",
        "title",
        "platform",
        "source_type",
        "locality_bucket",
        "country_guess",
        "city_guess",
        "language_guess",
        "parseability_score",
        "quote_post_count",
        "buy_sell_pair_count",
        "likely_individual_shop",
        "freshness_score",
        "locality_usefulness_score",
        "basket_contribution_score",
        "feed_stability_score",
        "operational_score",
        "operational_bucket",
        "last_seen",
        "status_guess",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranked:
            writer.writerow(
                {
                    "handle_or_url": item.handle_or_url,
                    "title": item.title,
                    "platform": item.platform,
                    "source_type": item.source_type,
                    "locality_bucket": item.locality_bucket,
                    "country_guess": item.country_guess,
                    "city_guess": item.city_guess,
                    "language_guess": item.language_guess,
                    "parseability_score": item.parseability_score,
                    "quote_post_count": item.quote_post_count,
                    "buy_sell_pair_count": item.buy_sell_pair_count,
                    "likely_individual_shop": item.likely_individual_shop,
                    "freshness_score": item.freshness_score,
                    "locality_usefulness_score": item.locality_usefulness_score,
                    "basket_contribution_score": item.basket_contribution_score,
                    "feed_stability_score": item.feed_stability_score,
                    "operational_score": item.operational_score,
                    "operational_bucket": item.operational_bucket,
                    "last_seen": item.last_seen,
                    "status_guess": item.status_guess,
                    "notes": item.notes,
                }
            )


def summarize_ranked(
    ranked: Sequence[RankedCandidate],
    baseline_counts: Dict[str, int],
    reference_thresholds: Dict[str, float],
) -> Dict[str, Any]:
    bucket_counts = {bucket: 0 for bucket in ("P1", "P2", "monitor_only", "reject")}
    locality_summary: Dict[str, Dict[str, Any]] = {}

    for locality in TARGET_LOCALITIES:
        locality_items = [item for item in ranked if item.locality_bucket == locality]
        locality_summary[locality] = {
            "counts": {
                "P1": sum(1 for item in locality_items if item.operational_bucket == "P1"),
                "P2": sum(1 for item in locality_items if item.operational_bucket == "P2"),
                "monitor_only": sum(1 for item in locality_items if item.operational_bucket == "monitor_only"),
                "reject": sum(1 for item in locality_items if item.operational_bucket == "reject"),
            },
            "top_sources": [
                {
                    "handle_or_url": item.handle_or_url,
                    "source_type": item.source_type,
                    "operational_bucket": item.operational_bucket,
                    "operational_score": item.operational_score,
                }
                for item in locality_items[:5]
            ],
        }

    for item in ranked:
        bucket_counts[item.operational_bucket] = bucket_counts.get(item.operational_bucket, 0) + 1

    best_candidates = [
        {
            "handle_or_url": item.handle_or_url,
            "locality_bucket": item.locality_bucket,
            "source_type": item.source_type,
            "operational_bucket": item.operational_bucket,
            "operational_score": item.operational_score,
            "notes": item.notes,
        }
        for item in ranked
        if item.operational_bucket in {"P1", "P2"}
    ][:10]

    locality_improvements: Dict[str, Dict[str, Any]] = {}
    for locality in TARGET_LOCALITIES:
        locality_items = [item for item in ranked if item.locality_bucket == locality]
        p1 = sum(1 for item in locality_items if item.operational_bucket == "P1")
        p2 = sum(1 for item in locality_items if item.operational_bucket == "P2")
        monitor = sum(1 for item in locality_items if item.operational_bucket == "monitor_only")
        locality_improvements[locality] = {
            "baseline_registry_sources": baseline_counts.get(locality, 0),
            "new_candidates": len(locality_items),
            "p1_added": p1,
            "p2_added": p2,
            "monitor_only_added": monitor,
            "projected_operational_additions": p1 + p2,
        }

    return {
        "generated_at": now_utc().replace(microsecond=0).isoformat(),
        "P1_count": bucket_counts.get("P1", 0),
        "P2_count": bucket_counts.get("P2", 0),
        "monitor_only_count": bucket_counts.get("monitor_only", 0),
        "reject_count": bucket_counts.get("reject", 0),
        "ranked_sources_by_locality": locality_summary,
        "best_candidates_for_immediate_basket_inclusion": best_candidates,
        "estimated_basket_improvements_by_locality": locality_improvements,
        "reference_thresholds": reference_thresholds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank multilingual exchange-shop discovery candidates")
    parser.add_argument("--survey-dir", default="survey_outputs", help="Directory containing survey CSV/JSON artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    survey_dir = Path(args.survey_dir)
    if not survey_dir.is_absolute():
        survey_dir = root / survey_dir

    candidates_path = survey_dir / "exchange_shop_multilingual_candidates.csv"
    channel_survey_path = survey_dir / "channel_survey.csv"
    pilot_metrics_path = survey_dir / "pilot_channel_metrics.csv"
    pilot_quote_records_path = survey_dir / "pilot_quote_records.csv"

    candidate_rows = load_csv(candidates_path)
    channel_survey_rows = load_csv(channel_survey_path)
    pilot_metrics_rows = load_csv(pilot_metrics_path)
    pilot_quote_rows = load_csv(pilot_quote_records_path)

    baseline_counts = compute_registry_baseline(channel_survey_rows)
    reference_thresholds = compute_reference_thresholds(pilot_metrics_rows, pilot_quote_rows)
    ranked = rank_candidates(candidate_rows, baseline_counts, reference_thresholds, now_utc())

    ranking_csv = survey_dir / "exchange_shop_candidate_ranking.csv"
    summary_json = survey_dir / "exchange_shop_candidate_ranking_summary.json"
    write_ranking_csv(ranking_csv, ranked)
    summary = summarize_ranked(ranked, baseline_counts, reference_thresholds)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"P1_count={summary['P1_count']}")
    print(f"P2_count={summary['P2_count']}")
    print(f"monitor_only_count={summary['monitor_only_count']}")
    print(f"reject_count={summary['reject_count']}")
    print(f"ranking_csv={ranking_csv}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
