#!/usr/bin/env python3
"""Verify that the public domain serves the exact pages and data we deployed."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MANIFEST = "api/deployment.json"
REQUIRED_FILES = ("index.html", "api/latest.json", "api/series.json")
OPTIONAL_FILES = (
    "status/index.html",
    "api/chart_history.json",
    "api/intraday/latest.json",
    "api/regional_market_signals_card.json",
    "assets/js/home-dashboard.js",
    "assets/css/dashboard.css",
)


class VerificationError(RuntimeError):
    pass


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def prepare_release(site_dir: Path, application_sha: str, data_sha: str,
                    run_id: str, run_attempt: str) -> dict:
    for name in REQUIRED_FILES:
        if not (site_dir / name).is_file():
            raise VerificationError(f"Required release file is missing: {name}")
    latest = json.loads((site_dir / "api/latest.json").read_text())
    manifest = {
        "schema_version": 1,
        "application_sha": application_sha,
        "data_sha": data_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_date": latest.get("date"),
        "files": {
            name: digest((site_dir / name).read_bytes())
            for name in REQUIRED_FILES + OPTIONAL_FILES
            if (site_dir / name).is_file()
        },
    }
    (site_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fetch_public(url: str, timeout: float) -> bytes:
    request = Request(url, headers={
        "User-Agent": "rialwatch-deployment-verification/1.0",
        "Cache-Control": "no-cache",
    })
    with urlopen(request, timeout=timeout) as response:
        return response.read(2 * 1024 * 1024)


def wait_for_release(expected: dict, base_url: str, wait_seconds: float = 180,
                     *, fetch=fetch_public, clock=time.monotonic, sleep=time.sleep) -> None:
    if wait_seconds <= 0:
        raise ValueError("Verification timeout must be positive")
    deadline = clock() + wait_seconds
    attempt = 0
    last_error = "release not observed"
    while clock() < deadline:
        attempt += 1
        query = urlencode({"rw_release": f"{expected['run_id']}-{expected['run_attempt']}-{attempt}"})

        def read(name: str) -> bytes:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("Public verification deadline reached")
            return fetch(f"{base_url.rstrip('/')}/{name}?{query}", min(10, remaining))

        try:
            actual = json.loads(read(MANIFEST))
            if actual != expected:
                raise VerificationError("Public release manifest is not the deployment being verified")
            # A fresh manifest alone is insufficient: a CDN can still serve an
            # old homepage or rate API. Verify the actual public response bytes.
            for name, expected_hash in expected["files"].items():
                if digest(read(name)) != expected_hash:
                    raise VerificationError(f"Public content does not match the release: {name}")
            print(f"Verified release {expected['run_id']}/{expected['run_attempt']}: "
                  f"{len(expected['files'])} public files; reference {expected['reference_date']}.")
            return
        except (VerificationError, URLError, OSError, ValueError) as exc:
            last_error = str(exc)
            remaining = deadline - clock()
            if remaining > 0:
                print(f"Waiting for public release propagation (attempt {attempt}): {last_error}")
                sleep(min(10, remaining))
    raise VerificationError(f"Public release verification failed after {wait_seconds:g}s: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--site-dir", type=Path, required=True)
    for name in ("application-sha", "data-sha", "run-id", "run-attempt"):
        prepare.add_argument(f"--{name}", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--site-dir", type=Path, required=True)
    check.add_argument("--base-url", required=True)
    check.add_argument("--wait-seconds", type=float, default=180)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            prepare_release(args.site_dir, args.application_sha, args.data_sha, args.run_id, args.run_attempt)
        else:
            expected = json.loads((args.site_dir / MANIFEST).read_text())
            wait_for_release(expected, args.base_url, args.wait_seconds)
    except (VerificationError, OSError, ValueError) as exc:
        print(f"::error::{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
