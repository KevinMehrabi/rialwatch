import argparse
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import guardrail_check, pipeline


ROOT = Path(__file__).resolve().parents[1]
DAY = dt.date(2026, 11, 1)


def timestamp(hour, minute=0, second=0):
    return dt.datetime.combine(DAY, dt.time(hour, minute, second), tzinfo=pipeline.UTC)


def attempt(site, when, values):
    sources = {}
    for source, value in values.items():
        sample = pipeline.Sample(
            source, when, value, {"open_market": value}, when,
            ok=True, stale=False, error=None,
            health={"fetch_success": True, "validation_result": {"ok": True}},
            source_unit="rial", normalized_unit="rial",
        )
        sources[source] = {"sample": pipeline.serialize_sample(sample)}
    payload = {"date": when.date().isoformat(), "collected_at": pipeline.iso_ts(when), "sources": sources}
    path = pipeline.write_intraday_attempt(site, payload)
    return path


def stored_fix(when, value=2_708_000, scope="same_day_repair", sources=None):
    return {
        "date": DAY.isoformat(), "as_of": pipeline.iso_ts(when), "revision": 0,
        "computed": {
            "fix": value, "withheld": False, "status": "Green",
            "source_medians": sources or {"bonbast": 2_700_000, "alanchand_street": 2_716_000},
            "band": {"p25": value, "p75": value}, "dispersion": 0,
        },
        "publication_selection": {
            "selection_scope": scope,
            "candidate_count": 0 if scope == "same_day_repair" else 1,
            "valid_candidate_count": 0 if scope == "same_day_repair" else 1,
            "selected_collected_at": pipeline.iso_ts(timestamp(0, 25) if scope == "same_day_repair" else timestamp(14, 5)),
            "street_source_count_used": len(sources) if sources else 2,
            "same_as_previous_day": False, "previous_day_fix": None,
            "delta_vs_previous_day": None, "delta_pct_vs_previous_day": None,
        },
    }


class PublicationReliabilityTests(unittest.TestCase):
    def run_at(self, site, now, build_only=False):
        with mock.patch.object(pipeline, "utc_now", return_value=now):
            if build_only:
                return pipeline.run_build_only(site, ROOT / "templates", pipeline.iso_ts(now), DAY)
            return pipeline.run_publish_daily(argparse.Namespace(skip_waits=True), site, ROOT / "templates", pipeline.iso_ts(now), DAY)

    def test_delayed_overnight_publisher_cannot_freeze_day_before_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            raw = attempt(site, timestamp(0, 25), {"bonbast": 2_700_000, "alanchand_street": 2_716_000})
            original = raw.read_bytes()
            for now in (timestamp(0, 27), timestamp(14, 15, 59), timestamp(14, 19, 59)):
                with self.subTest(now=now):
                    self.assertEqual(self.run_at(site, now), 0)
                    self.assertFalse((site / "fix/2026-11-01.json").exists())
                    failures, context = guardrail_check.evaluate_guardrails(site, DAY, now=now)
                    self.assertEqual(failures, [])
                    self.assertFalse(context["publication_due"])
            attempt(site, timestamp(14, 5), {"bonbast": 2_720_000, "alanchand_street": 2_722_000})
            self.run_at(site, timestamp(14, 20))
            daily = json.loads((site / "fix/2026-11-01.json").read_text())
            self.assertEqual(daily["computed"]["fix"], 2_721_000)
            self.assertEqual(daily["publication_selection"]["selection_scope"], "publication_window")
            self.assertEqual(raw.read_bytes(), original)

    def test_build_only_does_not_repair_current_day_withhold_before_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            fix_path = site / "fix/2026-11-01.json"
            pipeline.write_json(fix_path, pipeline.build_placeholder_payload(DAY, pipeline.iso_ts(timestamp(0)), "WITHHOLD", "no valid sources available"))
            original = fix_path.read_bytes()
            attempt(site, timestamp(0, 25), {"bonbast": 2_700_000})
            self.run_at(site, timestamp(0, 27), build_only=True)
            self.assertEqual(fix_path.read_bytes(), original)

    def test_missing_window_repair_and_single_source_are_explicit_without_new_quorum(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            attempt(site, timestamp(21, 21), {"alanchand_street": 2_725_000})
            self.run_at(site, timestamp(21, 23))
            fix_path = site / "fix/2026-11-01.json"
            original = fix_path.read_bytes()
            daily = json.loads(original)
            self.assertEqual(daily["computed"]["status"], "Green")
            self.assertEqual(daily["computed"]["fix"], 2_725_000)
            quality = json.loads((site / "api/publication-quality.json").read_text())
            self.assertFalse(quality["cross_source_confirmation"])
            self.assertEqual(quality["street_source_count_used"], 1)
            codes = {item["code"] for item in quality["findings"]}
            self.assertEqual(codes, {"single_street_source", "missing_publication_window", "outside_publication_window"})
            failures, context = guardrail_check.evaluate_guardrails(site, DAY, now=timestamp(21, 23))
            self.assertEqual(failures, [])
            self.assertTrue(any("No retained observations" in warning for warning in context["warnings"]))
            homepage = (site / "index.html").read_text()
            self.assertNotIn("no cross-source confirmation", homepage)
            self.assertNotIn("outside the publication window", homepage)
            for page in ("fix/2026-11-01/index.html", "status/index.html"):
                html = (site / page).read_text()
                self.assertIn("no cross-source confirmation", html)
                self.assertIn("outside the publication window", html)
            self.assertEqual(fix_path.read_bytes(), original)

    def test_premature_immutable_fix_is_flagged_and_not_replaced_by_later_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            fix_path = site / "fix/2026-11-01.json"
            original_payload = stored_fix(timestamp(0, 27))
            original_payload["original_as_of"] = original_payload["as_of"]
            original_payload["original_publication_selection"] = original_payload["publication_selection"].copy()
            pipeline.write_json(fix_path, original_payload)
            original = fix_path.read_bytes()
            raw = attempt(site, timestamp(21, 21), {"alanchand_street": 2_725_000})
            raw_bytes = raw.read_bytes()
            for build_only in (False, True):
                self.run_at(site, timestamp(21, 23), build_only=build_only)
                self.assertEqual(fix_path.read_bytes(), original)
                self.assertEqual(raw.read_bytes(), raw_bytes)
                latest = json.loads((site / "api/latest.json").read_text())
                self.assertEqual(latest["computed"]["fix"], 2_708_000)
                self.assertEqual(latest["publication_selection"], original_payload["publication_selection"])
                self.assertEqual(latest["revision"], 0)
            failures, context = guardrail_check.evaluate_guardrails(site, DAY, now=timestamp(21, 23))
            self.assertTrue(any("before the 14:20" in failure for failure in failures))
            self.assertTrue(context["warnings"])

    def test_guardrail_before_cutoff_accepts_previous_day_and_rejects_premature_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            previous = stored_fix(timestamp(14, 20), scope="publication_window")
            previous["date"] = "2026-10-31"
            previous["as_of"] = "2026-10-31T14:20:00Z"
            previous["publication_selection"]["selected_collected_at"] = "2026-10-31T14:05:00Z"
            pipeline.write_json(site / "api/latest.json", previous)
            self.assertEqual(guardrail_check.evaluate_guardrails(site, DAY, now=timestamp(0, 27))[0], [])
            pipeline.write_json(site / "api/latest.json", stored_fix(timestamp(0, 27)))
            failures, _ = guardrail_check.evaluate_guardrails(site, DAY, now=timestamp(0, 27))
            self.assertTrue(any("before the 14:20" in failure for failure in failures))

    def test_healthy_two_source_window_fix_has_no_quality_findings(self):
        quality = pipeline.publication_quality(stored_fix(timestamp(14, 20), scope="publication_window"))
        self.assertEqual(quality["findings"], [])
        self.assertTrue(quality["cross_source_confirmation"])

    def test_next_day_rebuild_refreshes_daily_page_and_preserves_overdue_quality_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            fix_path = site / "fix/2026-11-01.json"
            pipeline.write_json(fix_path, stored_fix(timestamp(21, 23)))
            original = fix_path.read_bytes()
            pipeline.write_text(site / "fix/2026-11-01/index.html", "Previous page without quality disclosures")
            now = timestamp(15) + dt.timedelta(days=1)
            with mock.patch.object(pipeline, "utc_now", return_value=now):
                pipeline.run_build_only(site, ROOT / "templates", pipeline.iso_ts(now), now.date())
            homepage = (site / "index.html").read_text()
            self.assertNotIn("No observations from the 13:45-14:15 UTC publication window", homepage)
            self.assertNotIn("outside the publication window", homepage)
            for page in ("fix/2026-11-01/index.html", "status/index.html"):
                html = (site / page).read_text()
                self.assertIn("No observations from the 13:45-14:15 UTC publication window", html)
                self.assertIn("outside the publication window", html)
            self.assertIn("expected 2026-11-02", (site / "status/index.html").read_text())
            self.assertEqual(fix_path.read_bytes(), original)
            self.assertFalse((site / "fix/2026-11-02.json").exists())

    def test_wrong_window_label_or_wrong_day_is_an_error(self):
        payload = stored_fix(timestamp(14, 20))
        payload["publication_selection"]["selection_scope"] = "publication_window"
        self.assertIn("invalid_selection_scope", {x["code"] for x in pipeline.publication_quality(payload)["findings"]})
        payload["publication_selection"]["selection_scope"] = "same_day_repair"
        payload["publication_selection"]["selected_collected_at"] = "2026-10-31T23:59:00Z"
        self.assertIn("invalid_selection_scope", {x["code"] for x in pipeline.publication_quality(payload)["findings"]})

    def test_same_day_selection_does_not_trust_misfiled_previous_day_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            old = attempt(site, timestamp(0) - dt.timedelta(minutes=1), {"bonbast": 2_700_000})
            target = site / "intraday/2026-11-01/23-59-00.json"
            target.parent.mkdir(parents=True)
            old.rename(target)
            with mock.patch.object(pipeline, "utc_now", return_value=timestamp(21, 23)):
                self.assertIsNone(pipeline.select_daily_from_intraday(site, DAY, pipeline.build_source_configs()))

    def test_guardrail_cli_emits_annotations_for_allowed_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            pipeline.write_json(site / "api/latest.json", stored_fix(timestamp(21, 23)))
            stdout = io.StringIO()
            with mock.patch("sys.argv", ["guardrail_check", "--site-dir", str(site), "--day", DAY.isoformat()]), contextlib.redirect_stdout(stdout):
                self.assertEqual(guardrail_check.main(), 0)
            self.assertIn("::warning::No retained observations", stdout.getvalue())
            self.assertIn("passed with publication-quality warnings", stdout.getvalue())
            self.assertNotIn("Guardrails passed.\n", stdout.getvalue())

    def test_guardrail_only_requests_collection_when_revision_policy_allows_recovery(self):
        for withheld in (False, True):
            with self.subTest(withheld=withheld), tempfile.TemporaryDirectory() as tmp:
                site = Path(tmp) / "site"
                latest = stored_fix(timestamp(0, 27))
                if withheld:
                    latest = pipeline.build_placeholder_payload(
                        DAY, pipeline.iso_ts(timestamp(21, 23)), "WITHHOLD",
                        "no intraday samples available in publication window",
                    )
                pipeline.write_json(site / "api/latest.json", latest)
                output, summary = Path(tmp) / "output", Path(tmp) / "summary"
                with (
                    mock.patch("sys.argv", ["guardrail_check", "--site-dir", str(site), "--day", DAY.isoformat()]),
                    mock.patch.dict("os.environ", {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)}),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(guardrail_check.main(), 1)
                self.assertIn(f"auto_heal_required={str(withheld).lower()}", output.read_text())
                self.assertIn("Error:", summary.read_text())
                self.assertIn("Warning:", summary.read_text())


if __name__ == "__main__":
    unittest.main()
