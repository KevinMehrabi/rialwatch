#!/usr/bin/env python3
"""Standalone Bonbast fetch + parse probe.

This is intentionally isolated from the daily dashboard pipeline so we can
validate Bonbast fetch/parse behavior without touching publication logic.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from scripts import pipeline
except ModuleNotFoundError:  # pragma: no cover - script execution path fallback.
    import pipeline  # type: ignore

DEFAULT_BONBAST_URL = "https://bonbast.com"

# Selector candidates used by the production Bonbast parser for the USD anchor.
USD_SELECTOR_CANDIDATES = pipeline.BONBAST_SELECTOR_MAP.get("open_market", ())

# Regex heuristics for buy/sell channel extraction in raw HTML text.
USD_BUY_PATTERNS = (
    r"(?is)(?:usd|dollar)[^0-9]{0,80}(?:buy|bid|خرید)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
    r"(?is)(?:buy|bid|خرید)[^0-9]{0,80}(?:usd|dollar)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
)
USD_SELL_PATTERNS = (
    r"(?is)(?:usd|dollar)[^0-9]{0,80}(?:sell|offer|فروش)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
    r"(?is)(?:sell|offer|فروش)[^0-9]{0,80}(?:usd|dollar)[^0-9]{0,50}([0-9][0-9,٬،.]*)",
)

EXPECTED_MARKERS = ("usd", "usd1", "sekkeh", "table")
BOT_BLOCK_KEYWORDS = (
    "captcha",
    "cf-chl",
    "cf_clearance",
    "turnstile",
    "hcaptcha",
    "g-recaptcha",
    "verify you are human",
    "attention required",
    "access denied",
    "automated queries",
    "blocked",
)


def detect_bot_block(html: str) -> bool:
    text = html.lower()
    return any(keyword in text for keyword in BOT_BLOCK_KEYWORDS)


def markers_present(html: str) -> Dict[str, bool]:
    text = html.lower()
    return {marker: (marker in text) for marker in EXPECTED_MARKERS}


def extract_number_with_patterns(html: str, patterns: Tuple[str, ...]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, html)
        if not match:
            continue
        value = pipeline.parse_number(match.group(1))
        if value is not None:
            return value
    return None


def extract_value_by_element_id(html: str, element_id: str) -> Optional[float]:
    patterns = (
        rf'(?is)id="{re.escape(element_id)}"[^>]*>([^<]+)',
        rf"(?is)id='{re.escape(element_id)}'[^>]*>([^<]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, html)
        if not match:
            continue
        parsed = pipeline.parse_number_from_text(match.group(1))
        if parsed is not None:
            return parsed
    return None


def parse_bonbast_html(html: str, default_unit: str = "toman") -> Dict[str, Any]:
    marker_map = markers_present(html)
    buy = extract_number_with_patterns(html, USD_BUY_PATTERNS)
    sell = extract_number_with_patterns(html, USD_SELL_PATTERNS)
    selector_sell = extract_value_by_element_id(html, "usd1")
    selector_buy = extract_value_by_element_id(html, "usd2")
    selector_top = extract_value_by_element_id(html, "usd1_top")

    fallback_anchor = pipeline.extract_bonbast_value_from_text(
        html, pipeline.BONBAST_TEXT_HINTS.get("open_market", ("usd",)), 100_000, 5_000_000
    )
    source_unit = pipeline.detect_unit_from_text(html, default_unit)
    chosen = (
        selector_sell
        if selector_sell is not None
        else sell
        if sell is not None
        else buy
        if buy is not None
        else fallback_anchor
    )
    normalized = pipeline.normalize_unit(chosen, source_unit) if chosen is not None else None

    return {
        "expected_marker_presence": marker_map,
        "marker_hit_count": sum(1 for v in marker_map.values() if v),
        "usd_buy_raw": buy,
        "usd_sell_raw": sell,
        "usd_buy_selector_raw": selector_buy,
        "usd_sell_selector_raw": selector_sell,
        "usd_top_selector_raw": selector_top,
        "usd_anchor_fallback_raw": fallback_anchor,
        "source_unit_assumption": source_unit,
        "normalized_unit": "rial",
        "normalized_rial_value": normalized,
        "bot_block_or_captcha": detect_bot_block(html),
        "usd_selector_candidates": list(USD_SELECTOR_CANDIDATES),
        "usd_buy_patterns": list(USD_BUY_PATTERNS),
        "usd_sell_patterns": list(USD_SELL_PATTERNS),
    }


def fetch_http(url: str, timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "rialwatch-bonbast-probe/0.1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        status = getattr(resp, "status", None)
        final_url = resp.geturl()
    return {
        "status": int(status) if status is not None else None,
        "final_url": final_url,
        "html_bytes": body,
        "fetch_mode": "http",
    }


def fetch_playwright(url: str, timeout: int) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
        page.wait_for_timeout(1_000)
        html = page.content().encode("utf-8")
        status = response.status if response is not None else None
        final_url = page.url
        browser.close()
    return {
        "status": status,
        "final_url": final_url,
        "html_bytes": html,
        "fetch_mode": "playwright",
    }


def probe_bonbast(url: str, timeout: int, mode: str) -> Dict[str, Any]:
    if mode == "playwright":
        fetched = fetch_playwright(url, timeout)
    else:
        fetched = fetch_http(url, timeout)

    html_bytes = fetched["html_bytes"]
    html = html_bytes.decode("utf-8", errors="replace")
    parsed = parse_bonbast_html(html)

    tmp = tempfile.NamedTemporaryFile(prefix="bonbast_probe_", suffix=".html", delete=False)
    tmp.write(html_bytes)
    tmp.flush()
    tmp.close()

    return {
        "http_status": fetched["status"],
        "final_url": fetched["final_url"],
        "content_length": len(html_bytes),
        "fetch_mode": fetched["fetch_mode"],
        "raw_html_file": str(Path(tmp.name)),
        **parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bonbast standalone fetch + parse probe")
    parser.add_argument("--url", default=DEFAULT_BONBAST_URL, help="Bonbast URL to fetch")
    parser.add_argument("--timeout", type=int, default=25, help="Request timeout seconds")
    parser.add_argument("--mode", choices=("http", "playwright"), default="http", help="Fetch mode")
    args = parser.parse_args()

    try:
        report = probe_bonbast(url=args.url, timeout=args.timeout, mode=args.mode)
    except urllib.error.HTTPError as exc:
        print(json.dumps({"error": f"http {exc.code}", "url": args.url}, indent=2))
        return 2
    except urllib.error.URLError as exc:
        print(json.dumps({"error": f"network: {exc.reason}", "url": args.url}, indent=2))
        return 2
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc), "url": args.url}, indent=2))
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
