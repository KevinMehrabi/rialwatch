#!/usr/bin/env python3
"""Build a diagnostics-only UAE stale-signal watchlist artifact."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def infer_signal_type(candidate_records: List[Dict[str, str]]) -> str:
    currencies = {row.get("currency", "") for row in candidate_records}
    has_remittance = any(str(row.get("remittance_quote_detected", "")).lower() == "true" for row in candidate_records)
    if "AED" in currencies and "USD" in currencies:
        return "website_rate_board"
    if has_remittance:
        return "remittance_quote"
    if "USD" in currencies or "AED" in currencies:
        return "numeric_quote_page"
    return "stale_public_signal"


def build_watchlist(review_rows: List[Dict[str, str]], candidate_rows: List[Dict[str, str]], record_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    candidates_by_name = {row.get("business_name", ""): row for row in candidate_rows}
    records_by_name: Dict[str, List[Dict[str, str]]] = {}
    for row in record_rows:
        records_by_name.setdefault(row.get("business_name", ""), []).append(row)

    sources = []
    for row in review_rows:
        if row.get("basket_use_status") != "stale":
            continue
        name = row.get("business_name", "")
        candidate = candidates_by_name.get(name, {})
        related_records = records_by_name.get(name, [])
        last_seen = max([candidate.get("last_seen", "")] + [record.get("timestamp_iso", "") for record in related_records])
        signal_type = infer_signal_type(related_records)
        notes = "Stale numeric public signal; keep tracked but do not render UAE locality card until fresh public quotes reappear."
        if related_records:
            notes += f" Latest sampled surface: {related_records[0].get('surface_type', 'unknown')}."
        sources.append(
            {
                "source_name": name,
                "website": candidate.get("website", ""),
                "signal_type": signal_type,
                "freshness_status": row.get("freshness_status", "unknown"),
                "last_seen": last_seen,
                "notes": notes,
            }
        )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "diagnostics_only": True,
        "locality": "UAE",
        "status": "watchlist_only",
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UAE stale-signal watchlist artifact")
    parser.add_argument("--survey-dir", type=Path, default=Path("survey_outputs"))
    parser.add_argument("--site-api-dir", type=Path, default=Path("site/api"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_dir = args.survey_dir if args.survey_dir.is_absolute() else ROOT_DIR / args.survey_dir
    site_api_dir = args.site_api_dir if args.site_api_dir.is_absolute() else ROOT_DIR / args.site_api_dir

    review_rows = load_csv(survey_dir / "uae_basket_review.csv")
    candidate_rows = load_csv(survey_dir / "uae_exchange_discovery_candidates.csv")
    record_rows = load_csv(survey_dir / "uae_basket_candidate_records.csv")
    payload = build_watchlist(review_rows, candidate_rows, record_rows)
    write_json(site_api_dir / "uae_signal_watchlist.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
