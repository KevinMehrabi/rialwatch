import argparse
import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from scripts import pipeline


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
SOURCE_CONFIGS = [
    pipeline.SourceConfig(
        name="bonbast",
        url="https://example.test",
        auth_mode="public_json",
        secret_fields=(),
        benchmark_families=("open_market",),
        default_unit="rial",
    )
]


def utc_ts(day: dt.date, hour: int, minute: int) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=pipeline.UTC)


def sample_for(ts: dt.datetime, value: float, ok: bool = True) -> pipeline.Sample:
    benchmarks = {key: None for key in pipeline.BENCHMARK_LABELS}
    benchmarks["open_market"] = value
    return pipeline.Sample(
        source="bonbast",
        sampled_at=ts,
        value=value,
        benchmark_values=benchmarks,
        quote_time=ts,
        ok=ok,
        stale=False,
        error=None if ok else "test invalid sample",
        health={"fetch_success": ok, "validation_result": {"ok": ok}},
        source_unit="rial",
        normalized_unit="rial",
    )


def write_attempt(site_dir: Path, day: dt.date, hour: int, minute: int, value: float, ok: bool = True) -> None:
    ts = utc_ts(day, hour, minute)
    sample = sample_for(ts, value, ok=ok)
    samples = {"bonbast": [sample]}
    summary = pipeline.summarize_day(samples, SOURCE_CONFIGS, day)
    payload = {
        "date": pipeline.iso_date(day),
        "collected_at": pipeline.iso_ts(ts),
        "window_utc": {"start": "13:45", "end": "14:15"},
        "sources": {"bonbast": {"sample": pipeline.serialize_sample(sample), "health": sample.health}},
        "computed": summary.get("computed", {}),
    }
    pipeline.write_json(site_dir / "intraday" / pipeline.iso_date(day) / f"{hour:02d}-{minute:02d}-00.json", payload)


def daily_payload(day: dt.date, fix: float, as_of: str = "2026-05-15T14:10:00Z") -> dict:
    return {
        "date": pipeline.iso_date(day),
        "as_of": as_of,
        "sources": {},
        "benchmarks": {
            "open_market": {"fix": fix, "available": True, "withheld": False},
            "official": {"fix": 1_320_000.0, "available": True, "withheld": False},
            "regional_transfer": {"fix": 1_430_000.0, "available": True, "withheld": False},
            "crypto_usdt": {"fix": 1_415_000.0, "available": True, "withheld": False},
            "emami_gold_coin": {"fix": 1_050_000_000.0, "available": True, "withheld": False},
        },
        "indicators": {
            "street_official_gap_pct": {"value": 6.0606, "available": True},
            "street_transfer_gap_pct": {"value": -2.0979, "available": True},
            "street_crypto_gap_pct": {"value": 1.0714, "available": True},
            "street_gold_gap_pct": {"value": -11.705, "available": True},
            "official_commercial_trend_7d": {"value": 0.8, "available": True},
        },
        "computed": {
            "fix": fix,
            "band": {"p25": fix - 1_000.0, "p75": fix + 1_000.0},
            "dispersion": 0.0014,
            "status": "Green",
            "withheld": False,
            "withhold_reasons": [],
            "source_medians": {"bonbast": fix},
            "source_units": {"bonbast": "rial"},
            "benchmarks": {
                key: {"value": entry["fix"], "available": True, "is_primary": key == "open_market"}
                for key, entry in {
                    "open_market": {"fix": fix},
                    "official": {"fix": 1_320_000.0},
                    "regional_transfer": {"fix": 1_430_000.0},
                    "crypto_usdt": {"fix": 1_415_000.0},
                    "emami_gold_coin": {"fix": 1_050_000_000.0},
                }.items()
            },
            "indicators": {},
        },
        "methodology": {"mapping_fingerprint": "fixture-fingerprint"},
        "publication_selection": {
            "rule": "latest valid intraday attempt in publication window; fallback stays inside the window",
            "selection_scope": "publication_window",
            "selected_collected_at": as_of,
            "selected_attempt_file": "intraday/2026-05-15/14-10-00.json",
        },
    }


class DailyPublicationOutputTests(unittest.TestCase):
    def test_in_window_selection_beats_later_outside_window_sample(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            write_attempt(site_dir, day, 13, 50, 1_390_000.0)
            write_attempt(site_dir, day, 14, 10, 1_410_000.0)
            write_attempt(site_dir, day, 15, 7, 1_550_000.0)

            selected = pipeline.select_daily_from_intraday(site_dir, day, SOURCE_CONFIGS)

            self.assertIsNotNone(selected)
            self.assertEqual(selected["computed"]["fix"], 1_410_000.0)
            selection = selected["publication_selection"]
            self.assertEqual(selection["selected_collected_at"], "2026-05-15T14:10:00Z")
            self.assertEqual(selection["candidate_count"], 2)
            self.assertEqual(selection["valid_candidate_count"], 2)
            self.assertEqual(selection["selection_scope"], "publication_window")

    def test_fallback_stays_inside_publication_window(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            write_attempt(site_dir, day, 13, 50, 1_390_000.0)
            write_attempt(site_dir, day, 14, 10, 1_410_000.0, ok=False)
            write_attempt(site_dir, day, 15, 7, 1_550_000.0)

            selected = pipeline.select_daily_from_intraday(site_dir, day, SOURCE_CONFIGS)

            self.assertIsNotNone(selected)
            self.assertEqual(selected["computed"]["fix"], 1_390_000.0)
            selection = selected["publication_selection"]
            self.assertEqual(selection["selected_collected_at"], "2026-05-15T13:50:00Z")
            self.assertEqual(selection["latest_candidate_collected_at"], "2026-05-15T14:10:00Z")
            self.assertTrue(selection["used_fallback"])
            self.assertEqual(selection["valid_candidate_count"], 1)

    def test_outside_window_only_publishes_withhold(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            write_attempt(site_dir, day, 15, 7, 1_550_000.0)

            pipeline.run_publish_daily(
                argparse.Namespace(),
                site_dir,
                TEMPLATES_DIR,
                "2026-05-15T14:20:00Z",
                day,
            )

            payload = json.loads((site_dir / "fix" / "2026-05-15.json").read_text(encoding="utf-8"))
            self.assertIsNone(payload["computed"]["fix"])
            self.assertTrue(payload["computed"]["withheld"])
            self.assertEqual(payload["computed"]["status"], "WITHHOLD")
            self.assertEqual(payload["publication_selection"]["candidate_count"], 0)
            self.assertIsNone(payload["publication_selection"]["selected_collected_at"])
            self.assertEqual(payload["revision"], 0)
            self.assertIsNone(payload["revision_reason"])
            self.assertIsNone(payload["revised_at"])
            self.assertEqual(payload["original_as_of"], payload["as_of"])
            self.assertEqual(payload["original_publication_selection"], payload["publication_selection"])

    def test_build_only_preserves_existing_official_fix(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            original = daily_payload(day, 1_400_000.0)
            pipeline.write_json(site_dir / "fix" / "2026-05-15.json", original)
            write_attempt(site_dir, day, 15, 7, 1_550_000.0)

            pipeline.run_build_only(site_dir, TEMPLATES_DIR, "2026-05-15T16:00:00Z", day)

            preserved = json.loads((site_dir / "fix" / "2026-05-15.json").read_text(encoding="utf-8"))
            self.assertEqual(preserved["computed"]["fix"], original["computed"]["fix"])
            self.assertEqual(preserved["as_of"], original["as_of"])
            self.assertEqual(preserved["publication_selection"], original["publication_selection"])

    def test_latest_official_is_separate_from_intraday_latest(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            pipeline.write_json(site_dir / "fix" / "2026-05-15.json", daily_payload(day, 1_400_000.0))
            write_attempt(site_dir, day, 15, 7, 1_550_000.0)

            pipeline.run_build_only(site_dir, TEMPLATES_DIR, "2026-05-15T16:00:00Z", day)

            latest = json.loads((site_dir / "api" / "latest.json").read_text(encoding="utf-8"))
            intraday_latest = json.loads((site_dir / "api" / "intraday" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["computed"]["fix"], 1_400_000.0)
            self.assertEqual(intraday_latest["primary_open_market_value"], 1_550_000.0)
            self.assertFalse(intraday_latest["in_publication_window"])
            self.assertEqual(intraday_latest["related_official_fix_date"], "2026-05-15")
            self.assertEqual(intraday_latest["related_official_fix_value"], 1_400_000.0)

    def test_publish_latest_rejects_intraday_pulse_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            with self.assertRaises(pipeline.PipelineError):
                pipeline.publish_latest(
                    site_dir,
                    {
                        "date": "2026-05-15",
                        "collected_at": "2026-05-15T15:07:00Z",
                        "in_publication_window": False,
                        "primary_open_market_value": 1_550_000.0,
                    },
                )

    def test_publication_window_filter_excludes_1507_samples(self) -> None:
        day = dt.date(2026, 5, 15)
        in_window = sample_for(utc_ts(day, 14, 10), 1_410_000.0)
        outside_window = sample_for(utc_ts(day, 15, 7), 1_550_000.0)

        filtered = pipeline.filter_samples_to_publication_window(
            {"bonbast": [in_window, outside_window]},
            day,
        )

        self.assertEqual(filtered["bonbast"], [in_window])

    def test_daily_full_export_is_generated_from_fix_files(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            payload = daily_payload(day, 1_400_000.0)
            expected_selection = copy.deepcopy(payload["publication_selection"])
            pipeline.write_json(site_dir / "fix" / "2026-05-15.json", payload)

            pipeline.publish_public_series_artifacts(site_dir)

            full = json.loads((site_dir / "api" / "daily_full.json").read_text(encoding="utf-8"))
            row = full["rows"][0]
            self.assertEqual(row["date"], "2026-05-15")
            self.assertEqual(row["street_usd_irr"], 1_400_000.0)
            self.assertEqual(row["source_medians"], {"bonbast": 1_400_000.0})
            self.assertEqual(row["source_units"], {"bonbast": "rial"})
            self.assertEqual(row["street_official_gap_pct"], 6.0606)
            self.assertEqual(row["official_commercial_trend_7d"], 0.8)
            self.assertEqual(row["methodology_fingerprint"], "fixture-fingerprint")
            self.assertEqual(row["publication_selection"]["rule"], expected_selection["rule"])
            self.assertEqual(row["publication_selection"]["selected_attempt_file"], expected_selection["selected_attempt_file"])

            revisions = json.loads((site_dir / "api" / "revisions.json").read_text(encoding="utf-8"))
            self.assertEqual(revisions["rows"][0]["revision"], 0)

    def test_republished_daily_fix_increments_revision_and_preserves_original_metadata(self) -> None:
        day = dt.date(2026, 5, 15)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            original = daily_payload(day, 1_400_000.0, as_of="2026-05-15T14:10:00Z")
            changed = daily_payload(day, 1_420_000.0, as_of="2026-05-15T14:12:00Z")
            pipeline.write_json(site_dir / "fix" / "2026-05-15.json", original)

            pipeline.publish_daily_fix(site_dir, TEMPLATES_DIR, "2026-05-15T14:20:00Z", changed)

            revised = json.loads((site_dir / "fix" / "2026-05-15.json").read_text(encoding="utf-8"))
            self.assertEqual(revised["computed"]["fix"], 1_420_000.0)
            self.assertEqual(revised["revision"], 1)
            self.assertIsNotNone(revised["revision_reason"])
            self.assertIsNotNone(revised["revised_at"])
            self.assertEqual(revised["original_as_of"], original["as_of"])
            self.assertEqual(revised["original_publication_selection"], original["publication_selection"])

    def test_daily_reference_workflow_persists_rich_api_exports(self) -> None:
        workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily-reference.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("site/api/daily_full.json", workflow)
        self.assertIn("site/api/revisions.json", workflow)


if __name__ == "__main__":
    unittest.main()
