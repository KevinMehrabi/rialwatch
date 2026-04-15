#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TARGET_LOCALITIES = ["Iran", "UAE", "Turkey", "Afghanistan", "UK", "Iraq", "Germany"]
DEFAULT_HISTORY_DAYS = 365
STATE_RANK = {"publish": 3, "monitor": 2, "hide": 1}
SOURCE_RANK = {
    "merged_diagnostics": 4,
    "regional_fx_board_basket_review": 3,
    "exchange_shop_baskets_enriched": 2,
    "exchange_shop_baskets_card": 1,
}
HIDE_REASONS = {"", "no_usable_records", "stale_signal"}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def parse_iso(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str):
        return None
    text = ts.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def resolve_generated_at(payloads: Iterable[Dict[str, Any]]) -> str:
    stamps = [parse_iso(payload.get("generated_at")) for payload in payloads]
    valid = [stamp for stamp in stamps if stamp is not None]
    if valid:
        latest = max(valid)
    else:
        latest = datetime.now(timezone.utc)
    return latest.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def to_int(value: Any) -> int:
    parsed = to_float(value)
    if parsed is None:
        return 0
    return int(round(parsed))


def normalize_state(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"publish", "monitor", "hide"}:
        return text
    return "hide"


def normalize_freshness(freshness_status: Any, suppression_reason: str, display_state: str) -> str:
    text = str(freshness_status or "").strip().lower()
    if text in {"fresh", "recent", "stale", "old"}:
        return text
    reason = str(suppression_reason or "").strip().lower()
    if "stale" in reason:
        return "old"
    if display_state == "publish":
        return "recent"
    if display_state == "monitor":
        return "stale"
    return "unknown"


def display_state_from_publishable(publishable: Any, usable_records: int, suppression_reason: str) -> str:
    if bool(publishable):
        return "publish"
    reason = (suppression_reason or "").strip().lower()
    if usable_records > 0 and reason not in HIDE_REASONS:
        return "monitor"
    return "hide"


def dispersion_level_from_cv(value: Any) -> str:
    cv = to_float(value)
    if cv is None:
        return "unknown"
    if cv < 0.08:
        return "low"
    if cv < 0.18:
        return "medium"
    return "high"


def humanize_token(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip().replace("_", " ")
    return text if text else fallback


def format_rate(value: Any) -> str:
    rate = to_float(value)
    if rate is None:
        return "Unavailable"
    return f"{round(rate):,} IRR"


def format_spread(value: Any) -> str:
    spread = to_float(value)
    if spread is None:
        return "N/A"
    return f"{spread:+.2f}%"


def alignment_label_from_spread(value: Any) -> str:
    spread = to_float(value)
    if spread is None:
        return "Unknown"
    abs_spread = abs(spread)
    if abs_spread < 2.0:
        return "Aligned"
    if abs_spread <= 10.0:
        return "Mild divergence"
    return "Divergent"


def signal_label_for_locality(locality: str, display_state: str, signal_type_used: Any) -> str:
    normalized = locality.strip()
    if normalized in {"Iran", "Turkey", "UK"}:
        return "Exchange network signal"
    if normalized == "UAE":
        return "Dubai settlement signal"
    if normalized == "Afghanistan":
        return "Herat market signal"
    if normalized == "Iraq":
        if display_state == "monitor":
            return "Sulaymaniyah market (monitoring)"
        return "Sulaymaniyah market signal"
    return humanize_token(signal_type_used, fallback="Locality signal")


def normalize_locality(value: Any) -> str:
    return str(value or "").strip()


def build_candidate(
    *,
    source_artifact: str,
    locality: str,
    signal_type_used: Any,
    weighted_rate: Any,
    median_rate: Any,
    spread_vs_benchmark_pct: Any,
    usable_record_count: Any,
    contributing_source_count: Any,
    basket_confidence: Any,
    suppression_reason: Any,
    display_state: Any,
    freshness_status: Any = None,
    dispersion_level: Any = None,
) -> Dict[str, Any]:
    usable = to_int(usable_record_count)
    contributing = to_int(contributing_source_count)
    confidence = to_float(basket_confidence) or 0.0
    suppression = str(suppression_reason or "").strip()
    state = normalize_state(display_state)
    signal_type = str(signal_type_used or "").strip() or "unknown"
    freshness = normalize_freshness(freshness_status, suppression, state)
    dispersion = str(dispersion_level or "").strip() or "unknown"

    return {
        "basket_name": locality,
        "signal_type_used": signal_type,
        "weighted_rate": to_float(weighted_rate),
        "median_rate": to_float(median_rate),
        "spread_vs_benchmark_pct": to_float(spread_vs_benchmark_pct),
        "usable_record_count": usable,
        "contributing_source_count": contributing,
        "basket_confidence": confidence,
        "freshness_status": freshness,
        "dispersion_level": dispersion,
        "display_state": state,
        "suppression_reason": suppression,
        "source_artifact": source_artifact,
    }


def build_regional_candidates(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in payload.get("localities", []):
        locality = normalize_locality(row.get("locality_name"))
        if not locality:
            continue
        candidate = build_candidate(
            source_artifact="regional_fx_board_basket_review",
            locality=locality,
            signal_type_used=row.get("signal_type_used"),
            weighted_rate=row.get("weighted_rate"),
            median_rate=row.get("median_rate"),
            spread_vs_benchmark_pct=row.get("spread_vs_benchmark_pct"),
            usable_record_count=row.get("usable_record_count"),
            contributing_source_count=row.get("contributing_source_count"),
            basket_confidence=row.get("basket_confidence"),
            suppression_reason=row.get("suppression_reason"),
            display_state=row.get("recommended_display_state"),
            freshness_status=row.get("freshness_status"),
            dispersion_level=row.get("dispersion_level"),
        )
        result.setdefault(locality, []).append(candidate)
    return result


def build_enriched_candidates(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in payload.get("baskets", []):
        locality = normalize_locality(row.get("basket_name"))
        if not locality:
            continue
        suppression_reason = str(row.get("suppression_reason") or "").strip()
        usable_records = to_int(row.get("usable_record_count"))
        display_state = display_state_from_publishable(
            row.get("publishable"),
            usable_records,
            suppression_reason,
        )
        candidate = build_candidate(
            source_artifact="exchange_shop_baskets_enriched",
            locality=locality,
            signal_type_used=row.get("signal_type_used"),
            weighted_rate=row.get("weighted_rate"),
            median_rate=row.get("median_rate"),
            spread_vs_benchmark_pct=row.get("spread_vs_benchmark_pct"),
            usable_record_count=usable_records,
            contributing_source_count=row.get("contributing_source_count"),
            basket_confidence=row.get("basket_confidence"),
            suppression_reason=suppression_reason,
            display_state=display_state,
            freshness_status=row.get("freshness_status"),
            dispersion_level=dispersion_level_from_cv(row.get("dispersion_cv")),
        )
        result.setdefault(locality, []).append(candidate)
    return result


def build_legacy_candidates(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in payload.get("cards", []):
        locality = normalize_locality(row.get("basket_name"))
        if not locality:
            continue
        suppression_reason = str(row.get("suppression_reason") or "").strip()
        usable_records = to_int(row.get("usable_record_count"))
        display_state = display_state_from_publishable(
            row.get("publishable"),
            usable_records,
            suppression_reason,
        )
        candidate = build_candidate(
            source_artifact="exchange_shop_baskets_card",
            locality=locality,
            signal_type_used=row.get("signal_type_used"),
            weighted_rate=row.get("weighted_rate"),
            median_rate=row.get("median_rate"),
            spread_vs_benchmark_pct=row.get("spread_vs_benchmark_pct"),
            usable_record_count=usable_records,
            contributing_source_count=row.get("contributing_channel_count"),
            basket_confidence=row.get("basket_confidence"),
            suppression_reason=suppression_reason,
            display_state=display_state,
            freshness_status="unknown",
            dispersion_level="unknown",
        )
        result.setdefault(locality, []).append(candidate)
    return result


def candidate_sort_key(candidate: Dict[str, Any]) -> tuple:
    has_rate = 1 if to_float(candidate.get("weighted_rate")) is not None else 0
    return (
        STATE_RANK.get(candidate.get("display_state"), 0),
        has_rate,
        to_int(candidate.get("contributing_source_count")),
        to_int(candidate.get("usable_record_count")),
        SOURCE_RANK.get(candidate.get("source_artifact"), 0),
        to_float(candidate.get("basket_confidence")) or 0.0,
    )


def merge_locality_candidates(locality: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if locality != "Germany":
        return None
    publishable = [
        row
        for row in candidates
        if normalize_state(row.get("display_state")) == "publish" and to_float(row.get("weighted_rate")) is not None
    ]
    if len(publishable) < 2:
        return None

    weights: List[float] = []
    for row in publishable:
        usable = to_int(row.get("usable_record_count"))
        contributors = to_int(row.get("contributing_source_count"))
        weights.append(float(max(usable, contributors, 1)))

    weight_sum = sum(weights)
    if weight_sum <= 0:
        return None

    def weighted(field: str) -> Optional[float]:
        values = [to_float(row.get(field)) for row in publishable]
        valid = [(v, w) for v, w in zip(values, weights) if v is not None]
        if not valid:
            return None
        return sum(v * w for v, w in valid) / sum(w for _v, w in valid)

    dispersion_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    freshness_rank = {"unknown": 0, "old": 1, "stale": 2, "recent": 3, "fresh": 4}
    dispersion = max(
        (str(row.get("dispersion_level") or "unknown").strip().lower() for row in publishable),
        key=lambda value: dispersion_rank.get(value, 0),
        default="unknown",
    )
    freshness = max(
        (str(row.get("freshness_status") or "unknown").strip().lower() for row in publishable),
        key=lambda value: freshness_rank.get(value, 0),
        default="unknown",
    )

    signal_types = sorted(
        {
            str(row.get("signal_type_used") or "").strip()
            for row in publishable
            if str(row.get("signal_type_used") or "").strip()
        }
    )
    signal_type_label = "+".join(signal_types) if signal_types else "mixed_diagnostics"

    merged = build_candidate(
        source_artifact="merged_diagnostics",
        locality=locality,
        signal_type_used=signal_type_label,
        weighted_rate=weighted("weighted_rate"),
        median_rate=weighted("median_rate"),
        spread_vs_benchmark_pct=weighted("spread_vs_benchmark_pct"),
        usable_record_count=sum(to_int(row.get("usable_record_count")) for row in publishable),
        contributing_source_count=sum(to_int(row.get("contributing_source_count")) for row in publishable),
        basket_confidence=weighted("basket_confidence") or 0.0,
        suppression_reason="",
        display_state="publish",
        freshness_status=freshness,
        dispersion_level=dispersion,
    )
    return merged


def finalize_card(candidate: Dict[str, Any]) -> Dict[str, Any]:
    state = normalize_state(candidate.get("display_state"))
    suppression_reason = str(candidate.get("suppression_reason") or "").strip()
    signal_type = humanize_token(candidate.get("signal_type_used"), fallback="unknown")

    if state == "publish":
        status_text = f"Diagnostics signal available ({signal_type})"
    elif state == "monitor":
        reason_text = humanize_token(suppression_reason, fallback="limited coverage")
        status_text = f"Monitoring ({reason_text})"
    else:
        status_text = "Hidden (no usable diagnostics signal)"

    card = dict(candidate)
    card["alignment_label"] = alignment_label_from_spread(candidate.get("spread_vs_benchmark_pct"))
    card["signal_label"] = signal_label_for_locality(
        locality=str(candidate.get("basket_name") or ""),
        display_state=state,
        signal_type_used=candidate.get("signal_type_used"),
    )
    card["publishable"] = state == "publish"
    card["render_on_homepage"] = state in {"publish", "monitor"}
    card["status_text"] = status_text
    card["rate_text"] = format_rate(candidate.get("weighted_rate"))
    card["spread_text"] = format_spread(candidate.get("spread_vs_benchmark_pct"))
    return card


def build_regional_market_cards_payload(
    regional_payload: Dict[str, Any],
    enriched_payload: Dict[str, Any],
    legacy_payload: Dict[str, Any],
) -> Dict[str, Any]:
    per_locality: Dict[str, List[Dict[str, Any]]] = {}

    for mapping in (
        build_regional_candidates(regional_payload),
        build_enriched_candidates(enriched_payload),
        build_legacy_candidates(legacy_payload),
    ):
        for locality, candidates in mapping.items():
            per_locality.setdefault(locality, []).extend(candidates)

    ordered_localities = list(TARGET_LOCALITIES)
    extras = sorted(
        locality for locality in per_locality.keys() if locality not in set(TARGET_LOCALITIES)
    )
    ordered_localities.extend(extras)

    cards: List[Dict[str, Any]] = []
    for locality in ordered_localities:
        candidates = per_locality.get(locality, [])
        merged_candidate = merge_locality_candidates(locality, candidates)
        if merged_candidate is not None:
            candidates = candidates + [merged_candidate]
        if not candidates:
            best = build_candidate(
                source_artifact="none",
                locality=locality,
                signal_type_used=None,
                weighted_rate=None,
                median_rate=None,
                spread_vs_benchmark_pct=None,
                usable_record_count=0,
                contributing_source_count=0,
                basket_confidence=0.0,
                suppression_reason="no_usable_records",
                display_state="hide",
            )
        else:
            best = max(candidates, key=candidate_sort_key)
        cards.append(finalize_card(best))

    publish_count = len([card for card in cards if card.get("display_state") == "publish"])
    monitor_count = len([card for card in cards if card.get("display_state") == "monitor"])
    hide_count = len([card for card in cards if card.get("display_state") == "hide"])

    return {
        "generated_at": resolve_generated_at((regional_payload, enriched_payload, legacy_payload)),
        "diagnostics_only": True,
        "cards": cards,
        "summary": {
            "publish_count": publish_count,
            "monitor_count": monitor_count,
            "hide_count": hide_count,
        },
        "source_artifacts": {
            "regional_fx_board_basket_review": "site/api/regional_fx_board_basket_review.json",
            "exchange_shop_baskets_enriched": "site/api/exchange_shop_baskets_enriched.json",
            "exchange_shop_baskets_card": "site/api/exchange_shop_baskets_card.json",
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def history_row_from_card(card: Dict[str, Any], generated_at: str) -> Dict[str, Any]:
    ts = generated_at.strip() if isinstance(generated_at, str) else ""
    parsed = parse_iso(ts)
    date_text = parsed.date().isoformat() if parsed else ""
    return {
        "timestamp": ts,
        "date": date_text,
        "basket_name": str(card.get("basket_name") or "").strip(),
        "display_state": str(card.get("display_state") or "").strip(),
        "signal_type_used": str(card.get("signal_type_used") or "").strip(),
        "signal_label": str(card.get("signal_label") or "").strip(),
        "source_artifact": str(card.get("source_artifact") or "").strip(),
        "weighted_rate": to_float(card.get("weighted_rate")),
        "median_rate": to_float(card.get("median_rate")),
        "spread_vs_benchmark_pct": to_float(card.get("spread_vs_benchmark_pct")),
        "usable_record_count": to_int(card.get("usable_record_count")),
        "contributing_source_count": to_int(card.get("contributing_source_count")),
        "basket_confidence": to_float(card.get("basket_confidence")),
        "freshness_status": str(card.get("freshness_status") or "").strip(),
        "dispersion_level": str(card.get("dispersion_level") or "").strip(),
        "suppression_reason": str(card.get("suppression_reason") or "").strip(),
        "alignment_label": str(card.get("alignment_label") or "").strip(),
    }


def _history_key(row: Dict[str, Any]) -> tuple:
    return (
        str(row.get("timestamp") or ""),
        str(row.get("basket_name") or ""),
    )


def _history_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        str(row.get("timestamp") or ""),
        str(row.get("basket_name") or ""),
    )


def prune_history_rows(rows: List[Dict[str, Any]], history_days: int, latest_timestamp: str) -> List[Dict[str, Any]]:
    if history_days <= 0:
        return rows
    latest = parse_iso(latest_timestamp)
    if latest is None:
        return rows
    cutoff = latest - timedelta(days=max(history_days - 1, 0))
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts = parse_iso(row.get("timestamp"))
        if ts is None or ts >= cutoff:
            out.append(row)
    return out


def build_regional_history_payload(
    cards_payload: Dict[str, Any],
    existing_history_payload: Dict[str, Any],
    history_days: int,
) -> Dict[str, Any]:
    generated_at = str(cards_payload.get("generated_at") or "").strip()
    cards = cards_payload.get("cards", [])
    existing_rows = existing_history_payload.get("rows", []) if isinstance(existing_history_payload, dict) else []
    merged: Dict[tuple, Dict[str, Any]] = {}
    if isinstance(existing_rows, list):
        for row in existing_rows:
            if not isinstance(row, dict):
                continue
            key = _history_key(row)
            if key[0] and key[1]:
                merged[key] = row

    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            row = history_row_from_card(card, generated_at)
            key = _history_key(row)
            if key[0] and key[1]:
                merged[key] = row

    rows = sorted(merged.values(), key=_history_sort_key)
    rows = prune_history_rows(rows, history_days, generated_at)

    localities = sorted({str(row.get("basket_name") or "").strip() for row in rows if str(row.get("basket_name") or "").strip()})
    return {
        "generated_at": generated_at,
        "history_days": history_days,
        "rows": rows,
        "row_count": len(rows),
        "localities": localities,
    }


def build_regional_timeseries_payload(history_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = history_payload.get("rows", [])
    series_map: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            locality = str(row.get("basket_name") or "").strip()
            if not locality:
                continue
            point = {
                "timestamp": str(row.get("timestamp") or "").strip(),
                "date": str(row.get("date") or "").strip(),
                "weighted_rate": to_float(row.get("weighted_rate")),
                "median_rate": to_float(row.get("median_rate")),
                "spread_vs_benchmark_pct": to_float(row.get("spread_vs_benchmark_pct")),
                "basket_confidence": to_float(row.get("basket_confidence")),
                "usable_record_count": to_int(row.get("usable_record_count")),
                "contributing_source_count": to_int(row.get("contributing_source_count")),
                "display_state": str(row.get("display_state") or "").strip(),
                "freshness_status": str(row.get("freshness_status") or "").strip(),
                "dispersion_level": str(row.get("dispersion_level") or "").strip(),
                "alignment_label": str(row.get("alignment_label") or "").strip(),
            }
            series_map.setdefault(locality, []).append(point)

    localities_payload: Dict[str, Dict[str, Any]] = {}
    for locality in sorted(series_map.keys()):
        points = sorted(
            series_map[locality],
            key=lambda p: (str(p.get("timestamp") or ""), str(p.get("date") or "")),
        )
        daily_latest: Dict[str, Dict[str, Any]] = {}
        for point in points:
            day = str(point.get("date") or "")
            if not day:
                continue
            prior = daily_latest.get(day)
            if prior is None or str(point.get("timestamp") or "") >= str(prior.get("timestamp") or ""):
                daily_latest[day] = point
        daily_points = [daily_latest[day] for day in sorted(daily_latest.keys())]
        localities_payload[locality] = {
            "point_count": len(points),
            "daily_point_count": len(daily_points),
            "latest_point": points[-1] if points else None,
            "points": points,
            "daily_points": daily_points,
        }

    return {
        "generated_at": str(history_payload.get("generated_at") or ""),
        "history_days": to_int(history_payload.get("history_days")),
        "localities": localities_payload,
        "summary": {
            "locality_count": len(localities_payload),
            "total_points": sum(int(item.get("point_count", 0)) for item in localities_payload.values()),
            "total_daily_points": sum(int(item.get("daily_point_count", 0)) for item in localities_payload.values()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged regional market signals card payload.")
    parser.add_argument(
        "--regional",
        type=Path,
        default=Path("site/api/regional_fx_board_basket_review.json"),
        help="Path to regional FX board basket review artifact.",
    )
    parser.add_argument(
        "--enriched",
        type=Path,
        default=Path("site/api/exchange_shop_baskets_enriched.json"),
        help="Path to enriched exchange shop basket artifact.",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("site/api/exchange_shop_baskets_card.json"),
        help="Path to legacy exchange shop cards artifact.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("site/api/regional_market_signals_card.json"),
        help="Output path for merged homepage card payload.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=Path("site/api/regional_market_signals_history.json"),
        help="Output path for regional signal history artifact.",
    )
    parser.add_argument(
        "--timeseries-out",
        type=Path,
        default=Path("site/api/regional_market_signals_timeseries.json"),
        help="Output path for regional signal timeseries artifact.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Retention window in days for regional signal history (0 = keep all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regional_payload = read_json(args.regional)
    enriched_payload = read_json(args.enriched)
    legacy_payload = read_json(args.legacy)

    payload = build_regional_market_cards_payload(regional_payload, enriched_payload, legacy_payload)
    write_json(args.out, payload)
    existing_history = read_json(args.history_out)
    history_payload = build_regional_history_payload(payload, existing_history, args.history_days)
    write_json(args.history_out, history_payload)
    timeseries_payload = build_regional_timeseries_payload(history_payload)
    write_json(args.timeseries_out, timeseries_payload)

    summary = payload.get("summary", {})
    print(
        "regional_market_signals_card",
        f"publish={summary.get('publish_count', 0)}",
        f"monitor={summary.get('monitor_count', 0)}",
        f"hide={summary.get('hide_count', 0)}",
        f"out={args.out}",
        f"history_rows={history_payload.get('row_count', 0)}",
        f"history_out={args.history_out}",
        f"timeseries_out={args.timeseries_out}",
    )


if __name__ == "__main__":
    main()
