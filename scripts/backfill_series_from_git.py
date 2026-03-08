#!/usr/bin/env python3
"""Backfill site/api/series.json from git history of site/api/latest.json.

This is a one-time recovery utility when daily snapshots were not persisted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Any


def run_git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], text=True)
    return out


def better_row(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer rows with numeric fix; otherwise keep the newer status field.
    c_fix = current.get("fix")
    i_fix = incoming.get("fix")
    if c_fix is None and i_fix is not None:
        return incoming
    if c_fix is not None and i_fix is None:
        return current
    return incoming


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    series_path = repo / "site" / "api" / "series.json"

    rows_by_date: Dict[str, Dict[str, Any]] = {}

    if series_path.exists():
        try:
            existing = json.loads(series_path.read_text(encoding="utf-8"))
            for row in existing.get("rows", []):
                date = row.get("date")
                if not date:
                    continue
                rows_by_date[date] = {
                    "date": date,
                    "fix": row.get("fix"),
                    "p25": row.get("p25"),
                    "p75": row.get("p75"),
                    "status": row.get("status"),
                    "withheld": row.get("withheld"),
                }
        except json.JSONDecodeError:
            pass

    commits = run_git(["rev-list", "--reverse", "HEAD", "--", "site/api/latest.json"]).splitlines()

    for commit in commits:
        try:
            content = run_git(["show", f"{commit}:site/api/latest.json"])
            payload = json.loads(content)
        except Exception:
            continue

        date = payload.get("date")
        computed = payload.get("computed", {})
        if not date:
            continue

        incoming = {
            "date": date,
            "fix": computed.get("fix"),
            "p25": computed.get("band", {}).get("p25"),
            "p75": computed.get("band", {}).get("p75"),
            "status": computed.get("status"),
            "withheld": computed.get("withheld"),
        }

        current = rows_by_date.get(date)
        rows_by_date[date] = incoming if current is None else better_row(current, incoming)

    rows = sorted(rows_by_date.values(), key=lambda r: r["date"])

    series_path.parent.mkdir(parents=True, exist_ok=True)
    series_path.write_text(json.dumps({"rows": rows}, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {series_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
