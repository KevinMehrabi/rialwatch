#!/usr/bin/env python3
"""Wait out a declared Pages/API outage before attempting a real deployment."""

from __future__ import annotations

import argparse
import json
import math
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


STATUS_URL = "https://www.githubstatus.com/api/v2/components.json"
COMPONENTS = {"vg70hn9s2tyj": "Pages", "brv1bkgrwx7q": "API Requests"}
STATES = {"operational", "degraded_performance", "partial_outage", "major_outage", "under_maintenance"}


class RecoveryExpired(RuntimeError):
    pass


def fetch_status(timeout: float) -> dict:
    request = Request(f"{STATUS_URL}?rialwatch_check={time.time_ns()}", headers={
        "User-Agent": "rialwatch-deployment-recovery/1.0",
        "Cache-Control": "no-cache",
    })
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(1024 * 1024))


def degraded_components(payload: dict) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("components"), list):
        raise ValueError("Unrecognized GitHub Status response")
    found = {}
    for item in payload["components"]:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] in COMPONENTS:
            found[item["id"]] = item.get("status")
    if set(found) != set(COMPONENTS) or any(not isinstance(state, str) or state not in STATES for state in found.values()):
        raise ValueError("GitHub Status did not provide recognized Pages and API states")
    return [f"{COMPONENTS[key]}: {state}" for key, state in found.items() if state != "operational"]


def wait_until_ready(deadline: float, *, fetch=fetch_status, wall_clock=time.time,
                     clock=time.monotonic, sleep=time.sleep, log=print) -> str:
    # All workflow checkpoints share one absolute deadline. Monotonic time keeps
    # an individual wait bounded if the runner's wall clock is adjusted.
    budget = deadline - wall_clock()
    if not math.isfinite(budget) or budget <= 0:
        raise RecoveryExpired("Pages recovery deadline has expired; deployment is not verified")
    stop = clock() + budget
    observed_outage = False
    while clock() < stop:
        try:
            degraded = degraded_components(fetch(max(0.001, min(15, stop - clock()))))
        except (URLError, OSError, ValueError) as exc:
            if clock() >= stop:
                break
            if not observed_outage:
                # A broken status feed must not itself stop a healthy deployment.
                # This is permission to TRY, never a successful-deployment signal.
                log(f"::warning::Cannot read GitHub Status ({type(exc).__name__}); attempting deployment and public verification.")
                return "unknown"
            log("GitHub Status unavailable after a confirmed outage; retaining the bounded wait.")
        else:
            if clock() >= stop:
                break
            if not degraded:
                log("GitHub reports Pages and API Requests operational; proceeding to deployment.")
                return "ready"
            observed_outage = True
            log(f"Waiting for GitHub recovery ({'; '.join(degraded)}); collection and publication continue independently.")
        remaining = stop - clock()
        if remaining > 0:
            sleep(min(60, remaining))
    raise RecoveryExpired("GitHub Pages/API outage did not clear before the shared recovery deadline; deployment is not verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True, help="Shared Unix-time recovery deadline")
    args = parser.parse_args()
    try:
        wait_until_ready(args.deadline)
    except RecoveryExpired as exc:
        print(f"::error::{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
