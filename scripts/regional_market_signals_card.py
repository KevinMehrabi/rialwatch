#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TARGET_LOCALITIES = ["Iran", "UAE", "Turkey", "Afghanistan", "UK", "Iraq", "Germany"]
STATE_RANK = {"publish": 3, "monitor": 2, "hide": 1}
SOURCE_RANK = {
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
        SOURCE_RANK.get(candidate.get("source_artifact"), 0),
        to_int(candidate.get("usable_record_count")),
        to_float(candidate.get("basket_confidence")) or 0.0,
    )


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regional_payload = read_json(args.regional)
    enriched_payload = read_json(args.enriched)
    legacy_payload = read_json(args.legacy)

    payload = build_regional_market_cards_payload(regional_payload, enriched_payload, legacy_payload)
    write_json(args.out, payload)

    summary = payload.get("summary", {})
    print(
        "regional_market_signals_card",
        f"publish={summary.get('publish_count', 0)}",
        f"monitor={summary.get('monitor_count', 0)}",
        f"hide={summary.get('hide_count', 0)}",
        f"out={args.out}",
    )


if __name__ == "__main__":
    main()
