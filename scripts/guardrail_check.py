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

NO_INTRADAY_REASONS = {
    "no intraday samples available in publication window",
    "no valid intraday samples in publication window",
}

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
        attempts.append(
            {
                "file": path.name,
                "collected_at": parse_iso_datetime(payload.get("collected_at")),
                "fix": parse_number(computed.get("fix")),
                "withheld": bool(computed.get("withheld", True)),
                "status": str(computed.get("status") or ""),
                "open_market_available": bool(open_market.get("available")) and parse_number(open_market.get("value")) is not None,
                "open_market_value": parse_number(open_market.get("value")),
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
    if latest_date != day_s:
        failures.append(f"latest.json date is {latest_date or 'unknown'}, expected current day {day_s}.")

    computed = latest.get("computed", {})
    if not isinstance(computed, dict):
        computed = {}
    publication_selection = latest.get("publication_selection", {})
    if not isinstance(publication_selection, dict):
        publication_selection = {}

    reasons_raw = computed.get("withhold_reasons", [])
    reasons = [str(reason).strip().lower() for reason in reasons_raw if isinstance(reason, str)]
    no_intraday_reason = any(reason in NO_INTRADAY_REASONS for reason in reasons)
    no_valid_sources_reason = any("no valid sources available" in reason for reason in reasons)

    valid_candidate_count = publication_selection.get("valid_candidate_count")
    if isinstance(valid_candidate_count, (int, float)) and math.isfinite(float(valid_candidate_count)):
        valid_candidate_count = int(valid_candidate_count)
    else:
        valid_candidate_count = None

    latest_fix = parse_number(computed.get("fix"))
    latest_withheld = bool(computed.get("withheld", True))

    intraday_dir = site_dir / "intraday" / day_s
    attempts = collect_intraday_attempts(intraday_dir)
    intraday_count = len(attempts)
    any_valid_attempt = any((attempt["fix"] is not None) and (attempt["withheld"] is False) for attempt in attempts)
    any_open_market_candidate = any(bool(attempt["open_market_available"]) for attempt in attempts)

    context.update(
        {
            "intraday_count": intraday_count,
            "latest_withheld": latest_withheld,
            "latest_fix": latest_fix,
            "withhold_reasons": reasons,
            "valid_candidate_count": valid_candidate_count,
            "any_valid_attempt": any_valid_attempt,
            "any_open_market_candidate": any_open_market_candidate,
        }
    )

    if intraday_count > 0 and no_intraday_reason:
        failures.append(
            f"WITHHOLD reason says no intraday samples, but {intraday_count} intraday attempt file(s) exist for {day_s}."
        )

    if latest_withheld and valid_candidate_count is not None and valid_candidate_count > 0:
        failures.append(
            f"publication_selection.valid_candidate_count={valid_candidate_count}, but latest snapshot is still WITHHOLD."
        )

    if latest_withheld and no_valid_sources_reason and (any_valid_attempt or any_open_market_candidate):
        failures.append(
            "WITHHOLD reason is 'no valid sources available', but intraday attempts contain valid benchmark candidate data."
        )

    if not latest_withheld and latest_fix is None:
        failures.append("Snapshot is marked published but computed.fix is null.")

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
        f"latest_withheld={context.get('latest_withheld')} "
        f"latest_fix={context.get('latest_fix')} "
        f"valid_candidate_count={context.get('valid_candidate_count')}"
    )

    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        return 1

    print("Guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
