import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from scripts import guardrail_check


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def base_latest(day: str) -> dict:
    return {
        "date": day,
        "computed": {
            "fix": None,
            "withheld": True,
            "withhold_reasons": [],
            "source_medians": {},
        },
        "publication_selection": {
            "valid_candidate_count": 0,
        },
    }


def intraday_attempt(open_market_value: Optional[float], fix: Optional[float], withheld: bool) -> dict:
    return {
        "collected_at": "2026-03-24T14:10:00Z",
        "computed": {
            "fix": fix,
            "withheld": withheld,
            "status": "WITHHOLD" if withheld else "Green",
            "source_medians": {"source_a": fix} if fix is not None else {},
            "benchmarks": {
                "open_market": {
                    "available": open_market_value is not None,
                    "value": open_market_value,
                },
                "official": {
                    "available": False,
                    "value": None,
                },
            },
        },
    }


def intraday_attempt_with_official(official_value: Optional[float], fix: Optional[float], withheld: bool) -> dict:
    payload = intraday_attempt(open_market_value=None, fix=fix, withheld=withheld)
    payload["computed"]["benchmarks"]["official"] = {
        "available": official_value is not None,
        "value": official_value,
    }
    return payload


def intraday_attempt_with_companions(
    official_value: Optional[float],
    regional_transfer_value: Optional[float],
    crypto_usdt_value: Optional[float],
    fix: Optional[float],
    withheld: bool,
) -> dict:
    payload = intraday_attempt(open_market_value=None, fix=fix, withheld=withheld)
    payload["computed"]["benchmarks"]["official"] = {
        "available": official_value is not None,
        "value": official_value,
    }
    payload["computed"]["benchmarks"]["regional_transfer"] = {
        "available": regional_transfer_value is not None,
        "value": regional_transfer_value,
    }
    payload["computed"]["benchmarks"]["crypto_usdt"] = {
        "available": crypto_usdt_value is not None,
        "value": crypto_usdt_value,
    }
    return payload


class GuardrailCheckTest(unittest.TestCase):
    def test_fails_when_no_intraday_reason_conflicts_with_existing_intraday_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_dir = Path(temp_dir) / "site"
            day_s = "2026-03-24"
            latest = base_latest(day_s)
            latest["computed"]["withhold_reasons"] = ["no intraday samples available in publication window"]
            write_json(site_dir / "api" / "latest.json", latest)
            write_json(site_dir / "intraday" / day_s / "14-10-00.json", intraday_attempt(None, None, True))

            failures, _ctx = guardrail_check.evaluate_guardrails(site_dir, dt.date(2026, 3, 24))
            self.assertTrue(any("no intraday samples" in failure.lower() for failure in failures))

    def test_fails_when_no_valid_sources_reason_but_intraday_candidate_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_dir = Path(temp_dir) / "site"
            day_s = "2026-03-24"
            latest = base_latest(day_s)
            latest["computed"]["withhold_reasons"] = ["no valid sources available"]
            write_json(site_dir / "api" / "latest.json", latest)
            write_json(
                site_dir / "intraday" / day_s / "14-12-00.json",
                intraday_attempt(open_market_value=1_450_000.0, fix=1_450_000.0, withheld=False),
            )

            failures, _ctx = guardrail_check.evaluate_guardrails(site_dir, dt.date(2026, 3, 24))
            self.assertTrue(any("valid benchmark candidate" in failure.lower() for failure in failures))

    def test_passes_for_consistent_published_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_dir = Path(temp_dir) / "site"
            day_s = "2026-03-24"
            latest = base_latest(day_s)
            latest["computed"]["fix"] = 1_453_000.0
            latest["computed"]["withheld"] = False
            latest["computed"]["withhold_reasons"] = []
            latest["computed"]["source_medians"] = {"street_a": 1_452_000.0, "street_b": 1_454_000.0}
            latest["publication_selection"]["valid_candidate_count"] = 1
            write_json(site_dir / "api" / "latest.json", latest)
            write_json(
                site_dir / "intraday" / day_s / "14-14-00.json",
                intraday_attempt(open_market_value=1_453_000.0, fix=1_453_000.0, withheld=False),
            )

            failures, _ctx = guardrail_check.evaluate_guardrails(site_dir, dt.date(2026, 3, 24))
            self.assertEqual(failures, [])

    def test_fails_when_official_is_unavailable_but_intraday_has_official_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_dir = Path(temp_dir) / "site"
            day_s = "2026-03-24"
            latest = base_latest(day_s)
            latest["computed"]["fix"] = 1_453_000.0
            latest["computed"]["withheld"] = False
            latest["publication_selection"]["valid_candidate_count"] = 1
            latest["benchmarks"] = {
                "official": {
                    "available": False,
                    "fix": None,
                }
            }
            write_json(site_dir / "api" / "latest.json", latest)
            write_json(
                site_dir / "intraday" / day_s / "14-14-00.json",
                intraday_attempt_with_official(official_value=1_325_000.0, fix=1_453_000.0, withheld=False),
            )

            failures, _ctx = guardrail_check.evaluate_guardrails(site_dir, dt.date(2026, 3, 24))
            self.assertTrue(any("official benchmark is unavailable" in failure.lower() for failure in failures))

    def test_fails_when_regional_transfer_is_unavailable_but_intraday_has_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            site_dir = Path(temp_dir) / "site"
            day_s = "2026-03-24"
            latest = base_latest(day_s)
            latest["computed"]["fix"] = 1_453_000.0
            latest["computed"]["withheld"] = False
            latest["publication_selection"]["valid_candidate_count"] = 1
            latest["benchmarks"] = {
                "official": {"available": True, "fix": 1_325_000.0},
                "regional_transfer": {"available": False, "fix": None},
                "crypto_usdt": {"available": True, "fix": 1_510_000.0},
            }
            write_json(site_dir / "api" / "latest.json", latest)
            write_json(
                site_dir / "intraday" / day_s / "14-14-00.json",
                intraday_attempt_with_companions(
                    official_value=1_325_000.0,
                    regional_transfer_value=1_560_000.0,
                    crypto_usdt_value=1_510_000.0,
                    fix=1_453_000.0,
                    withheld=False,
                ),
            )

            failures, _ctx = guardrail_check.evaluate_guardrails(site_dir, dt.date(2026, 3, 24))
            self.assertTrue(any("regional transfer benchmark is unavailable" in failure.lower() for failure in failures))


if __name__ == "__main__":
    unittest.main()
