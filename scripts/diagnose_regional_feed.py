#!/usr/bin/env python3
"""Diagnostic probe for Regional Market Feed primary source (AlanChand)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_URL = "https://api.alanchand.com?type=currency&symbols=usd-hav"


def parse_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if out > 0 else None
    if isinstance(value, str):
        cleaned = (
            value.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
            .strip()
            .replace(",", "")
            .replace("٬", "")
            .replace("،", "")
            .replace(" ", "")
        )
        if not cleaned:
            return None
        try:
            out = float(cleaned)
        except ValueError:
            return None
        return out if out > 0 else None
    return None


def normalize_symbol_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def extract_numeric_from_quote_obj(obj: Dict[str, Any]) -> Optional[float]:
    for field in ("value", "price", "last", "close", "sell", "buy", "rate", "amount"):
        if field in obj:
            parsed = parse_float(obj.get(field))
            if parsed is not None:
                return parsed
    return None


def extract_value_by_symbol_candidates(payload: Any, candidates: Tuple[str, ...]) -> Optional[float]:
    target_tokens = {normalize_symbol_token(item) for item in candidates}

    def matches(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        token = normalize_symbol_token(text)
        return bool(token) and token in target_tokens

    def walk(node: Any, results: List[float]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if matches(key):
                    if isinstance(value, dict):
                        parsed = extract_numeric_from_quote_obj(value)
                        if parsed is not None:
                            results.append(parsed)
                    else:
                        parsed = parse_float(value)
                        if parsed is not None:
                            results.append(parsed)

            id_fields = ("symbol", "name", "slug", "code", "title", "label", "item", "id")
            if any(matches(node.get(field)) for field in id_fields):
                parsed = extract_numeric_from_quote_obj(node)
                if parsed is not None:
                    results.append(parsed)

            for value in node.values():
                walk(value, results)
        elif isinstance(node, list):
            for item in node:
                walk(item, results)

    found: List[float] = []
    walk(payload, found)
    if not found:
        return None
    found.sort()
    mid = len(found) // 2
    if len(found) % 2 == 1:
        return found[mid]
    return (found[mid - 1] + found[mid]) / 2.0


def guess_unit(payload_text: str) -> str:
    lowered = payload_text.lower()
    if "toman" in lowered or "tmn" in lowered or "تومان" in payload_text:
        return "toman"
    if "rial" in lowered or "irr" in lowered or "ریال" in payload_text:
        return "rial"
    return "toman"


def normalize_to_rial(value: Optional[float], source_unit: str) -> Optional[float]:
    if value is None:
        return None
    if source_unit == "toman":
        return value * 10.0
    return value


def run_probe(url: str, token: str, timeout: int) -> int:
    request = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "rialwatch-regional-diagnostic/1.0",
            "Authorization": f"Bearer {token}",
        },
    )

    print(f"request_url: {url}")
    print("authorization_header: Bearer <redacted>")

    body = ""
    status: Optional[int] = None
    error_reason: Optional[str] = None
    token_format_hint: Optional[str] = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            status = int(resp.status)
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        body = exc.read().decode("utf-8", errors="replace")
        error_reason = f"http {exc.code}"
    except urllib.error.URLError as exc:
        error_reason = f"network error: {exc.reason}"
    except TimeoutError:
        error_reason = "timeout"

    print(f"http_status: {status if status is not None else 'N/A'}")
    print(f"response_body_length: {len(body.encode('utf-8')) if body else 0}")

    parsed_payload: Any = None
    parse_error: Optional[str] = None
    if body:
        try:
            parsed_payload = json.loads(body)
        except json.JSONDecodeError:
            parse_error = "invalid json"

    parsed_raw = extract_value_by_symbol_candidates(parsed_payload, ("usd-hav", "usd_hav")) if parsed_payload is not None else None
    source_unit = guess_unit(body) if body else "toman"
    normalized_rial = normalize_to_rial(parsed_raw, source_unit)

    print(f"parsed_raw_value: {parsed_raw if parsed_raw is not None else 'N/A'}")
    print(f"parsed_unit_assumption: {source_unit}")
    print(f"normalized_rial_value: {normalized_rial if normalized_rial is not None else 'N/A'}")

    if error_reason:
        print(f"error_reason: {error_reason}")
    elif parse_error:
        print(f"error_reason: {parse_error}")
    else:
        print("error_reason: none")

    if status in (401, 403):
        token_format_hint = (
            "Auth rejected. Verify ALANCHAND_API_KEY validity/scope and ensure header format is exactly "
            "'Authorization: Bearer <token>'."
        )
        print(f"auth_diagnosis: {token_format_hint}")
    elif status and 200 <= status < 300:
        print("auth_diagnosis: authorization accepted")
    else:
        print("auth_diagnosis: inconclusive")

    if parsed_payload is not None and isinstance(parsed_payload, dict) and "error" in parsed_payload:
        print(f"upstream_error_field: {parsed_payload.get('error')}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe AlanChand Regional Market Feed endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL, help="AlanChand endpoint URL")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout (seconds)")
    parser.add_argument(
        "--token",
        default="",
        help="Optional API token override. If omitted, ALANCHAND_API_KEY environment variable is used.",
    )
    args = parser.parse_args()

    token = args.token.strip() or os.environ.get("ALANCHAND_API_KEY", "").strip()
    if not token:
        print("error_reason: missing token (set ALANCHAND_API_KEY or pass --token)", file=sys.stderr)
        return 2

    return run_probe(url=args.url, token=token, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
