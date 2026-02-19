#!/usr/bin/env python3
"""USD/IRR Open Market Reference pipeline.

This script samples configured sources within the observation window,
computes the daily reference, and renders a static site under /site.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

UTC = dt.timezone.utc
WINDOW_START = dt.time(13, 45)
WINDOW_END = dt.time(14, 15)
PUBLISH_AT = dt.time(14, 20)
SAMPLE_OFFSETS_MIN = (0, 15, 30)  # 13:45, 14:00, 14:15 UTC

REQUIRED_SECRETS = (
    "BONBAST_USERNAME",
    "BONBAST_HASH",
    "NAVASAN_API_KEY",
    "ALANCHAND_API_KEY",
)


@dataclass
class SourceConfig:
    name: str
    url: str
    auth_mode: str  # query_user_hash | query_api_key | header_api_key
    secret_fields: Tuple[str, ...]


@dataclass
class Sample:
    source: str
    sampled_at: dt.datetime
    value: Optional[float]
    quote_time: Optional[dt.datetime]
    ok: bool
    stale: bool
    error: Optional[str] = None


class PipelineError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def parse_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        cleaned = cleaned.replace(" ", "")
        if not cleaned:
            return None
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
            return None
        v = float(cleaned)
    else:
        return None

    if not math.isfinite(v) or v <= 0:
        return None

    # Heuristic: some feeds use toman; normalize to rial when values are clearly too low.
    if 1_000 <= v < 100_000:
        return v * 10.0
    return v


def try_parse_datetime(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, (int, float)):
        iv = int(value)
        # Handle milliseconds epoch.
        if iv > 10_000_000_000:
            iv = iv // 1000
        try:
            return dt.datetime.fromtimestamp(iv, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    # ISO-ish forms.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            parsed = dt.datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            pass

    # Plain epoch in string form.
    if re.fullmatch(r"\d{10,13}", text):
        return try_parse_datetime(int(text))

    return None


def flatten_json(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_json(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            items.extend(flatten_json(value, path))
    else:
        items.append((prefix, obj))
    return items


def extract_quote_time(payload: Any) -> Optional[dt.datetime]:
    candidates = flatten_json(payload)
    scored: List[Tuple[int, dt.datetime]] = []
    for path, raw in candidates:
        path_l = path.lower()
        if not any(tok in path_l for tok in ("time", "date", "updated", "timestamp", "ts")):
            continue
        parsed = try_parse_datetime(raw)
        if not parsed:
            continue
        score = 0
        if "updated" in path_l:
            score += 3
        if "timestamp" in path_l or path_l.endswith(".ts"):
            score += 2
        if "usd" in path_l:
            score += 1
        scored.append((score, parsed))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def extract_usd_irr(payload: Any) -> Optional[float]:
    candidates = flatten_json(payload)
    ranked: List[Tuple[int, float]] = []

    for path, raw in candidates:
        path_l = path.lower()
        num = parse_number(raw)
        if num is None:
            continue

        score = 0
        if "usd" in path_l:
            score += 4
        if "irr" in path_l or "rial" in path_l:
            score += 4
        if "market" in path_l or "open" in path_l:
            score += 2
        if "sell" in path_l or "price" in path_l or path_l.endswith(".value"):
            score += 1
        if "buy" in path_l:
            score -= 1

        # Keep plausible ranges while still allowing future shifts.
        if 150_000 <= num <= 2_500_000:
            score += 2

        if score >= 4:
            ranked.append((score, num))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def median(values: List[float]) -> float:
    return float(statistics.median(values))


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def css_class(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-") or "unknown"


def read_template(templates_dir: Path, name: str) -> Template:
    path = templates_dir / name
    return Template(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def iso_date(d: dt.date) -> str:
    return d.isoformat()


def iso_ts(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def should_sleep_until(target: dt.datetime, skip_waits: bool) -> None:
    if skip_waits:
        return
    now = utc_now()
    if now >= target:
        return
    time.sleep((target - now).total_seconds())


def build_source_configs() -> List[SourceConfig]:
    return [
        SourceConfig(
            name="bonbast",
            url=os.environ.get("BONBAST_API_URL", "https://api.bonbast.com/v1/rates"),
            auth_mode="query_user_hash",
            secret_fields=("BONBAST_USERNAME", "BONBAST_HASH"),
        ),
        SourceConfig(
            name="navasan",
            url=os.environ.get("NAVASAN_API_URL", "https://api.navasan.tech/latest/"),
            auth_mode="query_api_key",
            secret_fields=("NAVASAN_API_KEY",),
        ),
        SourceConfig(
            name="alanchand",
            url=os.environ.get("ALANCHAND_API_URL", "https://api.alanchand.com/v1/rates"),
            auth_mode="header_api_key",
            secret_fields=("ALANCHAND_API_KEY",),
        ),
    ]


def missing_secrets() -> List[str]:
    missing = []
    for key in REQUIRED_SECRETS:
        if not os.environ.get(key):
            missing.append(key)
    return missing


def build_request(config: SourceConfig) -> urllib.request.Request:
    url = config.url
    headers = {
        "Accept": "application/json",
        "User-Agent": "rialwatch-pipeline/0.1",
    }

    if config.auth_mode == "query_user_hash":
        query = {
            "username": os.environ["BONBAST_USERNAME"],
            "hash": os.environ["BONBAST_HASH"],
        }
        url = with_query(url, query)
    elif config.auth_mode == "query_api_key":
        query = {"api_key": os.environ["NAVASAN_API_KEY"]}
        url = with_query(url, query)
    elif config.auth_mode == "header_api_key":
        key = os.environ["ALANCHAND_API_KEY"]
        headers["Authorization"] = f"Bearer {key}"
        headers["X-API-Key"] = key
        url = with_query(url, {"api_key": key})

    return urllib.request.Request(url=url, headers=headers, method="GET")


def with_query(url: str, extra_params: Dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    existing = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key, value in extra_params.items():
        existing[key] = [value]
    query = urllib.parse.urlencode(existing, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=query))


def fetch_one(config: SourceConfig, sampled_at: dt.datetime, window_start_dt: dt.datetime, window_end_dt: dt.datetime) -> Sample:
    try:
        req = build_request(config)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
    except KeyError as exc:
        return Sample(config.name, sampled_at, None, None, ok=False, stale=False, error=f"missing secret: {exc}")
    except urllib.error.HTTPError as exc:
        return Sample(config.name, sampled_at, None, None, ok=False, stale=False, error=f"http {exc.code}")
    except urllib.error.URLError as exc:
        return Sample(config.name, sampled_at, None, None, ok=False, stale=False, error=f"network: {exc.reason}")
    except TimeoutError:
        return Sample(config.name, sampled_at, None, None, ok=False, stale=False, error="timeout")
    except json.JSONDecodeError:
        return Sample(config.name, sampled_at, None, None, ok=False, stale=False, error="invalid json")

    value = extract_usd_irr(payload)
    quote_time = extract_quote_time(payload)

    stale = False
    if quote_time is not None:
        if quote_time < window_start_dt or quote_time > window_end_dt:
            stale = True

    if value is None:
        return Sample(config.name, sampled_at, None, quote_time, ok=False, stale=stale, error="unable to parse USD/IRR")

    return Sample(config.name, sampled_at, value, quote_time, ok=not stale, stale=stale, error="stale quote" if stale else None)


def collect_samples(
    source_configs: List[SourceConfig],
    day: dt.date,
    skip_waits: bool,
    allow_outside_window: bool,
) -> Dict[str, List[Sample]]:
    samples: Dict[str, List[Sample]] = {cfg.name: [] for cfg in source_configs}

    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    window_end_dt = dt.datetime.combine(day, WINDOW_END, tzinfo=UTC)

    for offset in SAMPLE_OFFSETS_MIN:
        target = window_start_dt + dt.timedelta(minutes=offset)
        if not allow_outside_window:
            now = utc_now()
            if now > window_end_dt + dt.timedelta(minutes=5):
                # Too late to satisfy observation-window guarantees.
                break
        should_sleep_until(target, skip_waits=skip_waits)
        sampled_at = utc_now()

        for cfg in source_configs:
            sample = fetch_one(cfg, sampled_at, window_start_dt, window_end_dt)
            if not allow_outside_window:
                if sampled_at < window_start_dt or sampled_at > window_end_dt:
                    sample.ok = False
                    sample.stale = True
                    sample.error = "sample outside observation window"
            samples[cfg.name].append(sample)

    return samples


def summarize_day(samples: Dict[str, List[Sample]], day: dt.date) -> Dict[str, Any]:
    source_medians: Dict[str, float] = {}
    source_notes: Dict[str, str] = {}

    invalid_or_stale = False
    for source, entries in samples.items():
        valid_values = [s.value for s in entries if s.value is not None]
        if not valid_values:
            source_notes[source] = "no valid samples"
            continue
        source_medians[source] = median(valid_values)


    medians = list(source_medians.values())
    reasons: List[str] = []
    withheld = False

    if len(medians) < 2:
        withheld = True
        reasons.append("fewer than 2 sources available")

    fix_value: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    dispersion: Optional[float] = None

    if not withheld:
        fix_value = median(medians)
        p25 = percentile(medians, 0.25)
        p75 = percentile(medians, 0.75)
        dispersion = (p75 - p25) / fix_value if fix_value else None
        if dispersion is not None and dispersion > 0.05:
            withheld = True
            reasons.append("dispersion > 5%")

    if invalid_or_stale:
        withheld = True
        reasons.append("invalid/stale inputs")

    status = "WITHHOLD"
    if not withheld and dispersion is not None:
        if dispersion <= 0.015:
            status = "Green"
        elif dispersion <= 0.035:
            status = "Amber"
        elif dispersion <= 0.05:
            status = "Red"
        else:
            status = "WITHHOLD"

    payload = {
        "date": iso_date(day),
        "as_of": iso_ts(utc_now()),
        "window_utc": {
            "start": WINDOW_START.strftime("%H:%M"),
            "end": WINDOW_END.strftime("%H:%M"),
            "sample_count_per_source": 3,
        },
        "sources": {
            source: {
                "samples": [
                    {
                        "sampled_at": iso_ts(s.sampled_at),
                        "value": s.value,
                        "quote_time": iso_ts(s.quote_time) if s.quote_time else None,
                        "ok": s.ok,
                        "stale": s.stale,
                        "error": s.error,
                    }
                    for s in entries
                ],
                "median": source_medians.get(source),
                "note": source_notes.get(source),
            }
            for source, entries in samples.items()
        },
        "computed": {
            "fix": fix_value,
            "band": {"p25": p25, "p75": p75},
            "dispersion": dispersion,
            "status": status,
            "withheld": withheld,
            "withhold_reasons": reasons,
            "source_medians": source_medians,
        },
    }
    return payload


def load_existing_days(site_dir: Path) -> List[str]:
    fix_dir = site_dir / "fix"
    if not fix_dir.exists():
        return []
    dates = []
    for path in sorted(fix_dir.glob("*.json")):
        name = path.stem
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name):
            dates.append(name)
    return dates


def render_page(templates_dir: Path, page_template: str, title: str, generated_at: str, **kwargs: Any) -> str:
    layout = read_template(templates_dir, "layout.html")
    content = read_template(templates_dir, page_template).safe_substitute(**kwargs)
    return layout.safe_substitute(
        title=title,
        content=content,
        generated_at=generated_at,
    )


def publish_methodology(site_dir: Path, templates_dir: Path, generated_at: str) -> None:
    html = render_page(
        templates_dir,
        "methodology.html",
        title="Methodology",
        generated_at=generated_at,
    )
    write_text(site_dir / "methodology" / "index.html", html)


def publish_status(
    site_dir: Path,
    templates_dir: Path,
    generated_at: str,
    status_title: str,
    status_detail: str,
    missing: Optional[List[str]] = None,
) -> None:
    missing_html = ""
    if missing:
        missing_html = "<ul>" + "".join(f"<li><code>{m}</code></li>" for m in missing) + "</ul>"

    html = render_page(
        templates_dir,
        "status.html",
        title="Status",
        generated_at=generated_at,
        status_title=status_title,
        status_class=css_class(status_title),
        status_detail=status_detail,
        missing_list=missing_html,
    )
    write_text(site_dir / "status" / "index.html", html)


def publish_archive(site_dir: Path, templates_dir: Path, generated_at: str, days: List[str]) -> None:
    if days:
        items = "\n".join(
            f'<li><a href="/fix/{d}/">{d}</a> <span class="muted">(<a href="/fix/{d}.json">json</a>)</span></li>'
            for d in sorted(days, reverse=True)
        )
    else:
        items = "<li>No published references yet.</li>"

    html = render_page(
        templates_dir,
        "archive.html",
        title="Archive",
        generated_at=generated_at,
        archive_items=items,
    )
    write_text(site_dir / "archive" / "index.html", html)


def publish_home(site_dir: Path, templates_dir: Path, generated_at: str, latest: Dict[str, Any]) -> None:
    c = latest.get("computed", {})
    fix = c.get("fix")
    p25 = c.get("band", {}).get("p25")
    p75 = c.get("band", {}).get("p75")
    status = c.get("status", "N/A")
    withheld = c.get("withheld", True)
    reasons = c.get("withhold_reasons", [])

    reasons_html = ""
    if withheld and reasons:
        reasons_html = "<ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>"

    html = render_page(
        templates_dir,
        "index.html",
        title="USD/IRR Open Market Reference",
        generated_at=generated_at,
        date=latest.get("date", "N/A"),
        as_of=latest.get("as_of", "N/A"),
        fix=fmt_rate(fix),
        p25=fmt_rate(p25),
        p75=fmt_rate(p75),
        status=status,
        status_class=css_class(str(status)),
        withheld="Yes" if withheld else "No",
        reasons=reasons_html,
    )
    write_text(site_dir / "index.html", html)


def publish_daily_fix(site_dir: Path, templates_dir: Path, generated_at: str, daily: Dict[str, Any]) -> None:
    day = daily["date"]
    c = daily.get("computed", {})

    reasons = c.get("withhold_reasons", [])
    reasons_html = ""
    if reasons:
        reasons_html = "<ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>"

    html = render_page(
        templates_dir,
        "fix.html",
        title=f"Reference {day}",
        generated_at=generated_at,
        date=day,
        as_of=daily.get("as_of", "N/A"),
        fix=fmt_rate(c.get("fix")),
        p25=fmt_rate(c.get("band", {}).get("p25")),
        p75=fmt_rate(c.get("band", {}).get("p75")),
        dispersion=(f"{c.get('dispersion', 0) * 100:.2f}%" if c.get("dispersion") is not None else "N/A"),
        status=c.get("status", "N/A"),
        status_class=css_class(str(c.get("status", "N/A"))),
        withheld="Yes" if c.get("withheld") else "No",
        reasons=reasons_html,
        source_rows=render_source_table(daily),
    )

    # /fix/YYYY-MM-DD route
    write_text(site_dir / "fix" / day / "index.html", html)
    # /fix/YYYY-MM-DD.json
    write_json(site_dir / "fix" / f"{day}.json", daily)


def render_source_table(daily: Dict[str, Any]) -> str:
    rows: List[str] = []
    for source, data in daily.get("sources", {}).items():
        median_val = data.get("median")
        note = data.get("note") or ""
        ok_count = sum(1 for s in data.get("samples", []) if s.get("ok"))
        rows.append(
            "<tr>"
            f"<td>{source}</td>"
            f"<td>{fmt_rate(median_val)}</td>"
            f"<td>{ok_count}/3</td>"
            f"<td>{note}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def publish_latest(site_dir: Path, daily: Dict[str, Any]) -> None:
    write_json(site_dir / "api" / "latest.json", daily)


def immutable_day_exists(site_dir: Path, day: str) -> bool:
    return (site_dir / "fix" / f"{day}.json").exists() or (site_dir / "fix" / day / "index.html").exists()


def run(args: argparse.Namespace) -> int:
    site_dir = Path(args.site_dir)
    templates_dir = Path(args.templates_dir)
    site_dir.mkdir(parents=True, exist_ok=True)

    now = utc_now()
    day = now.date()
    generated_at = iso_ts(now)

    publish_methodology(site_dir, templates_dir, generated_at)

    missing = missing_secrets()
    if missing:
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="CONFIG NEEDED",
            status_detail="Required API secrets are missing. No daily rate has been published.",
            missing=missing,
        )

        placeholder = {
            "date": iso_date(day),
            "as_of": generated_at,
            "computed": {
                "fix": None,
                "band": {"p25": None, "p75": None},
                "dispersion": None,
                "status": "CONFIG NEEDED",
                "withheld": True,
                "withhold_reasons": ["missing secrets"],
            },
        }
        publish_home(site_dir, templates_dir, generated_at, placeholder)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        # Keep latest as null object rather than fake price.
        publish_latest(site_dir, placeholder)
        return 0

    source_configs = build_source_configs()

    # Immutability: if today's reference exists, do not rewrite it.
    day_s = iso_date(day)
    if immutable_day_exists(site_dir, day_s):
        publish_status(
            site_dir,
            templates_dir,
            generated_at,
            status_title="IMMUTABLE",
            status_detail=f"Reference for {day_s} already exists and was not modified.",
            missing=None,
        )
        latest_path = site_dir / "api" / "latest.json"
        if latest_path.exists():
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            publish_home(site_dir, templates_dir, generated_at, latest)
        publish_archive(site_dir, templates_dir, generated_at, load_existing_days(site_dir))
        return 0

    window_start_dt = dt.datetime.combine(day, WINDOW_START, tzinfo=UTC)
    if now < window_start_dt:
        should_sleep_until(window_start_dt, skip_waits=args.skip_waits)

    samples = collect_samples(
        source_configs,
        day,
        skip_waits=args.skip_waits,
        allow_outside_window=args.allow_outside_window,
    )

    publish_dt = dt.datetime.combine(day, PUBLISH_AT, tzinfo=UTC)
    should_sleep_until(publish_dt, skip_waits=args.skip_waits)

    daily = summarize_day(samples, day)
    publish_daily_fix(site_dir, templates_dir, generated_at=iso_ts(utc_now()), daily=daily)
    publish_latest(site_dir, daily)

    publish_status(
        site_dir,
        templates_dir,
        generated_at=iso_ts(utc_now()),
        status_title="OK",
        status_detail=f"Published {day_s} reference at scheduled time {PUBLISH_AT.strftime('%H:%M')} UTC.",
    )

    publish_home(site_dir, templates_dir, generated_at=iso_ts(utc_now()), latest=daily)
    publish_archive(site_dir, templates_dir, generated_at=iso_ts(utc_now()), days=load_existing_days(site_dir))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USD/IRR Open Market Reference pipeline")
    parser.add_argument("--site-dir", default="site", help="Output directory for static site")
    parser.add_argument("--templates-dir", default="templates", help="Template directory")
    parser.add_argument(
        "--skip-waits",
        action="store_true",
        help="Skip sleeping between sample/publish checkpoints (for local verification)",
    )
    parser.add_argument(
        "--allow-outside-window",
        action="store_true",
        help="Allow sampling outside 13:45-14:15 UTC (for local verification only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
