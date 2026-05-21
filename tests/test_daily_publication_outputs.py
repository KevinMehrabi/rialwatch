import argparse
import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_outside_window_only_repairs_daily_publication_when_valid(self) -> None:
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
            self.assertEqual(payload["computed"]["fix"], 1_550_000.0)
            self.assertFalse(payload["computed"]["withheld"])
            self.assertEqual(payload["computed"]["status"], "Green")
            self.assertEqual(payload["publication_selection"]["candidate_count"], 0)
            self.assertEqual(payload["publication_selection"]["same_day_candidate_count"], 1)
            self.assertEqual(payload["publication_selection"]["valid_same_day_candidate_count"], 1)
            self.assertEqual(payload["publication_selection"]["selection_scope"], "same_day_repair")
            self.assertEqual(payload["publication_selection"]["selected_collected_at"], "2026-05-15T15:07:00Z")
            self.assertEqual(
                payload["publication_selection"]["selection_reason"],
                "no valid publication-window attempt; selected latest valid same-day intraday read",
            )
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

    def test_build_only_revises_withheld_daily_from_valid_same_day_attempt(self) -> None:
        day = dt.date(2026, 5, 21)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            withheld = pipeline.build_placeholder_payload(
                day,
                "2026-05-21T15:32:50Z",
                "WITHHOLD",
                "no intraday samples available in publication window",
            )
            withheld["publication_selection"] = {
                "rule": "latest valid intraday attempt in publication window; fallback stays inside the window",
                "selection_scope": "publication_window",
                "selected_collected_at": None,
            }
            pipeline.write_json(site_dir / "fix" / "2026-05-21.json", withheld)
            write_attempt(site_dir, day, 15, 7, 1_550_000.0)

            pipeline.run_build_only(site_dir, TEMPLATES_DIR, "2026-05-21T16:00:00Z", day)

            repaired = json.loads((site_dir / "fix" / "2026-05-21.json").read_text(encoding="utf-8"))
            latest = json.loads((site_dir / "api" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(repaired["computed"]["fix"], 1_550_000.0)
            self.assertFalse(repaired["computed"]["withheld"])
            self.assertEqual(repaired["publication_selection"]["selection_scope"], "same_day_repair")
            self.assertEqual(repaired["revision"], 1)
            self.assertEqual(
                repaired["revision_reason"],
                "repaired withheld daily fix from valid same-day intraday read",
            )
            self.assertEqual(repaired["original_as_of"], "2026-05-21T15:32:50Z")
            self.assertEqual(repaired["original_publication_selection"], withheld["publication_selection"])
            self.assertEqual(latest["computed"]["fix"], 1_550_000.0)

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

    def test_intraday_latest_keeps_observed_pulse_when_outside_official_window(self) -> None:
        day = dt.date(2026, 5, 21)
        official_day = dt.date(2026, 5, 18)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            pipeline.write_json(site_dir / "fix" / "2026-05-18.json", daily_payload(official_day, 1_797_100.0))
            today_withhold = pipeline.build_placeholder_payload(
                day,
                "2026-05-21T15:32:50Z",
                "WITHHOLD",
                "no intraday samples available in publication window",
            )
            pipeline.write_json(site_dir / "fix" / "2026-05-21.json", today_withhold)
            sampled_at = dt.datetime(2026, 5, 21, 14, 28, 37, tzinfo=pipeline.UTC)
            sample_payload = pipeline.serialize_sample(sample_for(sampled_at, 1_796_000.0))
            sample_payload["ok"] = False
            sample_payload["stale"] = True
            sample_payload["error"] = "sample outside observation window"
            official_sample_payload = pipeline.serialize_sample(sample_for(sampled_at, 1_481_184.0))
            official_sample_payload["benchmarks"] = {key: None for key in pipeline.BENCHMARK_LABELS}
            official_sample_payload["benchmarks"]["official"] = 1_481_184.0
            payload = {
                "date": pipeline.iso_date(day),
                "collected_at": pipeline.iso_ts(sampled_at),
                "window_utc": {"start": "13:45", "end": "14:15"},
                "sources": {
                    "bonbast": {"sample": sample_payload, "health": sample_payload["health"]},
                    "commercial_aux": {
                        "sample": official_sample_payload,
                        "health": official_sample_payload["health"],
                    },
                },
                "computed": {
                    "fix": None,
                    "band": {"p25": None, "p75": None},
                    "dispersion": None,
                    "status": "WITHHOLD",
                    "withheld": True,
                    "withhold_reasons": ["no valid sources available"],
                    "source_medians": {},
                    "source_units": {},
                    "benchmarks": {
                        key: {"value": None, "available": False, "is_primary": key == "open_market"}
                        for key in pipeline.BENCHMARK_LABELS
                    },
                },
            }
            pipeline.write_json(site_dir / "intraday" / "2026-05-21" / "14-28-37.json", payload)

            pipeline.publish_intraday_latest(site_dir, day)

            intraday_latest = json.loads((site_dir / "api" / "intraday" / "latest.json").read_text(encoding="utf-8"))
            self.assertFalse(intraday_latest["in_publication_window"])
            self.assertTrue(intraday_latest["valid"])
            self.assertEqual(intraday_latest["primary_open_market_value"], 1_796_000.0)
            self.assertEqual(intraday_latest["source_count_used"], 1)
            self.assertEqual(intraday_latest["source_medians"], {"bonbast": 1_796_000.0})
            self.assertEqual(intraday_latest["source_units"], {"bonbast": "rial"})
            self.assertEqual(intraday_latest["computed"]["fix"], 1_796_000.0)
            self.assertFalse(intraday_latest["computed"]["withheld"])
            self.assertEqual(intraday_latest["related_official_fix_date"], "2026-05-18")
            self.assertEqual(intraday_latest["related_official_fix_value"], 1_797_100.0)

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

    def test_public_series_is_continuous_with_explicit_carry_forward_rows(self) -> None:
        first_day = dt.date(2026, 5, 18)
        third_day = dt.date(2026, 5, 20)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            pipeline.write_json(site_dir / "fix" / "2026-05-18.json", daily_payload(first_day, 1_797_100.0))
            pipeline.write_json(site_dir / "fix" / "2026-05-20.json", daily_payload(third_day, 1_791_600.0))

            pipeline.publish_series(site_dir)

            series = json.loads((site_dir / "api" / "series.json").read_text(encoding="utf-8"))
            rows = series["rows"]
            self.assertEqual([row["date"] for row in rows], ["2026-05-18", "2026-05-19", "2026-05-20"])
            self.assertFalse(rows[0]["carried_forward"])
            self.assertEqual(rows[0]["source_date"], "2026-05-18")
            self.assertEqual(rows[0]["fill_method"], "observed")
            self.assertEqual(rows[1]["fix"], 1_797_100.0)
            self.assertEqual(rows[1]["p25"], 1_796_100.0)
            self.assertTrue(rows[1]["carried_forward"])
            self.assertEqual(rows[1]["source_date"], "2026-05-18")
            self.assertEqual(rows[1]["fill_method"], "previous_valid_fix")
            self.assertFalse(rows[2]["carried_forward"])
            self.assertEqual(rows[2]["source_date"], "2026-05-20")

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

    def test_intraday_workflow_waits_for_window_checkpoints(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "intraday-collection.yml"
        ).read_text(encoding="utf-8")

        self.assertIn('cron: "30 12 * * *"', workflow)
        self.assertIn("--collect-sample-times", workflow)
        self.assertIn("--sample-times-utc 13:50,14:05,14:14", workflow)

    def test_collect_intraday_sample_times_writes_multiple_attempts(self) -> None:
        day = dt.date(2026, 5, 15)
        first_sampled_at = utc_ts(day, 13, 50)
        second_sampled_at = utc_ts(day, 14, 5)

        def fake_collect_one_attempt(
            source_configs: list[pipeline.SourceConfig],
            sampled_at: dt.datetime,
            day: dt.date,
            allow_outside_window: bool,
        ) -> dict:
            value = 1_400_000.0 + sampled_at.minute
            return {"bonbast": [sample_for(sampled_at, value)]}

        utc_now_values = [
            utc_ts(day, 12, 30),
            utc_ts(day, 12, 31),
            first_sampled_at,
            utc_ts(day, 13, 51),
            second_sampled_at,
        ]

        def fake_utc_now() -> dt.datetime:
            if utc_now_values:
                return utc_now_values.pop(0)
            return second_sampled_at

        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            args = argparse.Namespace(
                allow_outside_window=False,
                collect_sample_times=True,
                sample_times_utc="13:50,14:05",
                skip_waits=True,
            )

            with mock.patch("scripts.pipeline.utc_now", side_effect=fake_utc_now):
                with mock.patch("scripts.pipeline.collect_one_attempt", side_effect=fake_collect_one_attempt):
                    rc = pipeline.run_collect_intraday(args, site_dir, TEMPLATES_DIR, "2026-05-15T12:30:00Z")

            self.assertEqual(rc, 0)
            attempts = sorted((site_dir / "intraday" / "2026-05-15").glob("*.json"))
            self.assertEqual([path.name for path in attempts], ["13-50-00.json", "14-05-00.json"])
            intraday_latest = json.loads((site_dir / "api" / "intraday" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(intraday_latest["collected_at"], "2026-05-15T14:05:00Z")
            self.assertTrue(intraday_latest["in_publication_window"])
            self.assertEqual(intraday_latest["primary_open_market_value"], 1_400_005.0)

    def test_homepage_withhold_shows_last_valid_official_rate(self) -> None:
        valid_day = dt.date(2026, 5, 18)
        withheld_day = dt.date(2026, 5, 21)
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            pipeline.write_json(site_dir / "fix" / "2026-05-18.json", daily_payload(valid_day, 1_797_100.0))
            pipeline.write_json(
                site_dir / "api" / "intraday" / "latest.json",
                {
                    "date": "2026-05-21",
                    "collected_at": "2026-05-21T15:16:50Z",
                    "in_publication_window": False,
                    "valid": True,
                    "primary_open_market_value": 1_791_600.0,
                },
            )
            latest = pipeline.build_placeholder_payload(
                withheld_day,
                "2026-05-21T15:32:50Z",
                "WITHHOLD",
                "no intraday samples available in publication window",
            )

            pipeline.publish_home(site_dir, TEMPLATES_DIR, "2026-05-21T15:46:00Z", latest)

            html = (site_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("WITHHELD TODAY", html)
            self.assertIn("Last valid official", html)
            self.assertIn("1,797,100", html)
            self.assertIn("Last valid daily fix: 2026-05-18", html)
            self.assertIn("Latest intraday pulse", html)
            self.assertIn("1,791,600", html)
            self.assertIn("not official daily fix", html)


if __name__ == "__main__":
    unittest.main()
