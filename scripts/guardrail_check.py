#!/usr/bin/env python3
"""
Guardrail checks for scheduled daily publication consistency.

This script catches situations where current-day published artifacts look stale
or inconsistent with available intraday attempts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

UTC = dt.timezone.utc
try:
    from scripts import pipeline as daily_pipeline
except ImportError:  # pragma: no cover - supports direct execution from scripts/
    import pipeline as daily_pipeline  # type: ignore

NO_INTRADAY_SAMPLES_REASON = "no intraday samples available in publication window"
NO_VALID_INTRADAY_SAMPLES_REASON = "no valid intraday samples in publication window"


def in_publication_window(ts: dt.datetime, day: dt.date) -> bool:
    start = dt.datetime.combine(day, daily_pipeline.WINDOW_START, tzinfo=UTC)
    end = dt.datetime.combine(day, daily_pipeline.WINDOW_END, tzinfo=UTC)
    return daily_pipeline.is_within_window_minute(ts, start, end)


def parse_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if math.isfinite(numeric):
            return numeric
    return None


def parse_iso_datetime(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def collect_intraday_attempts(day_dir: Path) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    if not day_dir.exists():
        return attempts
    for path in sorted(day_dir.glob("*.json")):
        payload = load_json(path)
        if payload is None:
            continue
        computed = payload.get("computed", {})
        if not isinstance(computed, dict):
            computed = {}
        computed_benchmarks = computed.get("benchmarks", {})
        if not isinstance(computed_benchmarks, dict):
            computed_benchmarks = {}
        open_market = computed_benchmarks.get("open_market", {})
        if not isinstance(open_market, dict):
            open_market = {}
        official = computed_benchmarks.get("official", {})
        if not isinstance(official, dict):
            official = {}
        regional_transfer = computed_benchmarks.get("regional_transfer", {})
        if not isinstance(regional_transfer, dict):
            regional_transfer = {}
        crypto_usdt = computed_benchmarks.get("crypto_usdt", {})
        if not isinstance(crypto_usdt, dict):
            crypto_usdt = {}
        attempts.append(
            {
                "file": path.name,
                "collected_at": parse_iso_datetime(payload.get("collected_at")),
                "fix": parse_number(computed.get("fix")),
                "withheld": bool(computed.get("withheld", True)),
                "status": str(computed.get("status") or ""),
                "open_market_available": bool(open_market.get("available")) and parse_number(open_market.get("value")) is not None,
                "open_market_value": parse_number(open_market.get("value")),
                "official_available": bool(official.get("available")) and parse_number(official.get("value")) is not None,
                "official_value": parse_number(official.get("value")),
                "regional_transfer_available": bool(regional_transfer.get("available")) and parse_number(regional_transfer.get("value")) is not None,
                "regional_transfer_value": parse_number(regional_transfer.get("value")),
                "crypto_usdt_available": bool(crypto_usdt.get("available")) and parse_number(crypto_usdt.get("value")) is not None,
                "crypto_usdt_value": parse_number(crypto_usdt.get("value")),
                "source_medians_count": len(
                    [
                        key
                        for key, value in (computed.get("source_medians", {}) or {}).items()
                        if parse_number(value) is not None
                    ]
                ),
            }
        )
    return attempts


def latest_fix_payload_for_day(site_dir: Path, day: dt.date) -> Optional[Dict[str, Any]]:
    fix_dir = site_dir / "fix"
    if not fix_dir.exists():
        return None

    candidates: List[Tuple[dt.date, Path]] = []
    for path in fix_dir.glob("*.json"):
        try:
            stamp = dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        if stamp <= day:
            candidates.append((stamp, path))

    for _stamp, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        payload = load_json(path)
        if payload is not None:
            return payload
    return None


def evaluate_guardrails(site_dir: Path, day: dt.date) -> Tuple[List[str], Dict[str, Any]]:
    day_s = day.isoformat()
    latest_path = site_dir / "api" / "latest.json"
    latest = load_json(latest_path)
    failures: List[str] = []
    context: Dict[str, Any] = {"day": day_s}

    if latest is None:
        failures.append("site/api/latest.json is missing or unreadable after pipeline run.")
        return failures, context

    latest_date = str(latest.get("date") or "")
    context["latest_date"] = latest_date
    current_day_fix_exists = (site_dir / "fix" / f"{day_s}.json").exists()
    official_latest = latest_fix_payload_for_day(site_dir, day)
    official_latest_date = str((official_latest or {}).get("date") or "")
    if official_latest_date:
        context["official_latest_date"] = official_latest_date
        if latest_date != official_latest_date:
            failures.append(
                "latest.json date is "
                f"{latest_date or 'unknown'}, expected latest official daily fix {official_latest_date}."
            )
    elif latest_date != day_s:
        failures.append(f"latest.json date is {latest_date or 'unknown'}, expected current day {day_s}.")

    computed = latest.get("computed", {})
    if not isinstance(computed, dict):
        computed = {}
    publication_selection = latest.get("publication_selection", {})
    if not isinstance(publication_selection, dict):
        publication_selection = {}

    reasons_raw = computed.get("withhold_reasons", [])
    reasons = [str(reason).strip().lower() for reason in reasons_raw if isinstance(reason, str)]
    no_intraday_reason = any(reason == NO_INTRADAY_SAMPLES_REASON for reason in reasons)
    no_valid_intraday_reason = any(reason == NO_VALID_INTRADAY_SAMPLES_REASON for reason in reasons)
    no_valid_sources_reason = any("no valid sources available" in reason for reason in reasons)

    valid_candidate_count = publication_selection.get("valid_candidate_count")
    if isinstance(valid_candidate_count, (int, float)) and math.isfinite(float(valid_candidate_count)):
        valid_candidate_count = int(valid_candidate_count)
    else:
        valid_candidate_count = None

    latest_fix = parse_number(computed.get("fix"))
    latest_withheld = bool(computed.get("withheld", True))
    benchmarks = latest.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        benchmarks = {}
    companion_keys = ("official", "regional_transfer", "crypto_usdt")
    companion_latest: Dict[str, Dict[str, Any]] = {}
    companion_latest_available: Dict[str, bool] = {}
    companion_latest_fix: Dict[str, Optional[float]] = {}
    for key in companion_keys:
        entry = benchmarks.get(key, {})
        if not isinstance(entry, dict):
            entry = {}
        companion_latest[key] = entry
        fix_value = parse_number(entry.get("fix"))
        companion_latest_fix[key] = fix_value
        companion_latest_available[key] = bool(entry.get("available", False)) and fix_value is not None

    intraday_dir = site_dir / "intraday" / day_s
    attempts = collect_intraday_attempts(intraday_dir)
    intraday_count = len(attempts)
    in_window_attempts = [
        attempt
        for attempt in attempts
        if attempt["collected_at"] is not None and in_publication_window(attempt["collected_at"], day)
    ]
    valid_in_window_attempts = [
        attempt for attempt in in_window_attempts if (attempt["fix"] is not None) and (attempt["withheld"] is False)
    ]
    any_valid_attempt = any((attempt["fix"] is not None) and (attempt["withheld"] is False) for attempt in attempts)
    any_open_market_candidate = any(bool(attempt["open_market_available"]) for attempt in attempts)
    any_valid_in_window_attempt = any(
        (attempt["fix"] is not None) and (attempt["withheld"] is False) for attempt in in_window_attempts
    )
    any_open_market_in_window_candidate = any(bool(attempt["open_market_available"]) for attempt in in_window_attempts)
    any_companion_candidate = {
        "official": any(bool(attempt["official_available"]) for attempt in attempts),
        "regional_transfer": any(bool(attempt["regional_transfer_available"]) for attempt in attempts),
        "crypto_usdt": any(bool(attempt["crypto_usdt_available"]) for attempt in attempts),
    }
    any_companion_in_window_candidate = {
        "official": any(bool(attempt["official_available"]) for attempt in in_window_attempts),
        "regional_transfer": any(bool(attempt["regional_transfer_available"]) for attempt in in_window_attempts),
        "crypto_usdt": any(bool(attempt["crypto_usdt_available"]) for attempt in in_window_attempts),
    }

    context.update(
        {
            "intraday_count": intraday_count,
            "in_window_intraday_count": len(in_window_attempts),
            "valid_in_window_candidate_count": len(valid_in_window_attempts),
            "current_day_fix_exists": current_day_fix_exists,
            "latest_withheld": latest_withheld,
            "latest_fix": latest_fix,
            "withhold_reasons": reasons,
            "valid_candidate_count": valid_candidate_count,
            "any_valid_attempt": any_valid_attempt,
            "any_open_market_candidate": any_open_market_candidate,
            "any_valid_in_window_attempt": any_valid_in_window_attempt,
            "any_open_market_in_window_candidate": any_open_market_in_window_candidate,
            "companion_latest_available": companion_latest_available,
            "companion_latest_fix": companion_latest_fix,
            "any_companion_candidate": any_companion_candidate,
            "any_companion_in_window_candidate": any_companion_in_window_candidate,
        }
    )

    if latest_date != day_s and not current_day_fix_exists:
        if valid_in_window_attempts:
            failures.append(
                "Current-day official fix is missing, but intraday attempts contain valid in-window benchmark data."
            )
        return failures, context

    if latest_date == day_s and latest_withheld and no_intraday_reason and intraday_count == 0:
        failures.append(
            "Current-day snapshot is WITHHOLD because no intraday samples were found, "
            "and no same-day intraday artifacts exist. Run auto-heal collection before publishing a blank day."
        )

    if in_window_attempts and no_intraday_reason:
        failures.append(
            "WITHHOLD reason says no intraday samples in publication window, "
            f"but {len(in_window_attempts)} in-window intraday attempt file(s) exist for {day_s}."
        )

    if no_valid_intraday_reason and (any_valid_in_window_attempt or any_open_market_in_window_candidate):
        failures.append(
            "WITHHOLD reason says no valid intraday samples in publication window, "
            "but in-window intraday attempts contain valid benchmark candidate data."
        )

    if latest_withheld and valid_candidate_count is not None and valid_candidate_count > 0:
        failures.append(
            f"publication_selection.valid_candidate_count={valid_candidate_count}, but latest snapshot is still WITHHOLD."
        )

    if latest_withheld and no_valid_sources_reason and (
        any_valid_in_window_attempt or any_open_market_in_window_candidate
    ):
        failures.append(
            "WITHHOLD reason is 'no valid sources available', "
            "but in-window intraday attempts contain valid benchmark candidate data."
        )

    if not latest_withheld and latest_fix is None:
        failures.append("Snapshot is marked published but computed.fix is null.")

    friendly_names = {
        "official": "Official benchmark",
        "regional_transfer": "Regional transfer benchmark",
        "crypto_usdt": "Crypto USDT benchmark",
    }
    for key in companion_keys:
        if not companion_latest_available.get(key, False) and any_companion_in_window_candidate.get(key, False):
            failures.append(
                f"{friendly_names[key]} is unavailable in latest.json, "
                f"but in-window intraday attempts contain usable {key} benchmark values."
            )

    return failures, context


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily publication guardrail checks.")
    parser.add_argument("--site-dir", default="site", help="Static site output directory")
    parser.add_argument("--day", default=None, help="UTC day in YYYY-MM-DD format (defaults to today)")
    args = parser.parse_args()

    if args.day:
        try:
            day = dt.date.fromisoformat(args.day)
        except ValueError:
            print(f"::error::Invalid --day value: {args.day}")
            return 2
    else:
        day = dt.datetime.now(UTC).date()

    failures, context = evaluate_guardrails(Path(args.site_dir), day)
    print(
        "Guardrail context: "
        f"day={context.get('day')} "
        f"latest_date={context.get('latest_date')} "
        f"intraday_count={context.get('intraday_count')} "
        f"in_window_intraday_count={context.get('in_window_intraday_count')} "
        f"latest_withheld={context.get('latest_withheld')} "
        f"latest_fix={context.get('latest_fix')} "
        f"valid_candidate_count={context.get('valid_candidate_count')} "
        f"valid_in_window_candidate_count={context.get('valid_in_window_candidate_count')}"
    )

    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        return 1

    print("Guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
