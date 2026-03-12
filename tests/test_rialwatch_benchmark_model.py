import json
import tempfile
import unittest
from pathlib import Path

from scripts import rialwatch_benchmark_model as model


class RialWatchBenchmarkModelTests(unittest.TestCase):
    def make_daily_fix(
        self,
        *,
        date: str = "2026-03-12",
        as_of: str = "2026-03-12T16:42:19Z",
        bonbast_mid: float = 1_464_000.0,
        alanchand_mid: float = 1_434_250.0,
        navasan_raw_open_market_toman: float = 166_400.0,
    ) -> dict:
        return {
            "date": date,
            "as_of": as_of,
            "sources": {
                "bonbast": {
                    "samples": [
                        {
                            "sampled_at": f"{date}T14:08:05Z",
                            "quote_time": f"{date}T14:08:05Z",
                            "value": bonbast_mid + 500.0,
                            "benchmarks": {"open_market": bonbast_mid + 500.0},
                            "ok": True,
                            "stale": False,
                            "source_unit": "toman",
                            "health": {
                                "parse_result": {
                                    "buy_rial": bonbast_mid - 500.0,
                                    "sell_rial": bonbast_mid + 500.0,
                                    "mid_rial": bonbast_mid,
                                }
                            },
                        }
                    ],
                    "note": None,
                },
                "alanchand_street": {
                    "samples": [
                        {
                            "sampled_at": f"{date}T14:08:05Z",
                            "quote_time": f"{date}T14:07:00Z",
                            "value": alanchand_mid + 7_250.0,
                            "benchmarks": {"open_market": alanchand_mid + 7_250.0},
                            "ok": True,
                            "stale": False,
                            "source_unit": "rial",
                            "health": {
                                "extracted_values": {
                                    "open_market_buy": alanchand_mid - 7_250.0,
                                    "open_market_sell": alanchand_mid + 7_250.0,
                                    "open_market_mid": alanchand_mid,
                                }
                            },
                        }
                    ],
                    "note": None,
                },
                "navasan": {
                    "samples": [
                        {
                            "sampled_at": f"{date}T14:08:05Z",
                            "quote_time": f"{date}T14:08:08Z",
                            "value": None,
                            "benchmarks": {"open_market": None},
                            "ok": False,
                            "stale": False,
                            "error": "unable to parse USD/IRR",
                            "source_unit": "mixed",
                            "health": {
                                "raw_extracted_values": {
                                    "open_market": navasan_raw_open_market_toman,
                                },
                                "benchmark_units": {
                                    "open_market": "toman",
                                },
                            },
                        }
                    ],
                    "note": "source excluded from street benchmark universe",
                },
            },
        }

    def test_extract_sample_point_estimate_prefers_midpoint(self) -> None:
        point = model.extract_sample_point_estimate(
            {
                "value": 1_464_500.0,
                "benchmarks": {"open_market": 1_464_500.0},
                "health": {
                    "parse_result": {
                        "buy_rial": 1_463_500.0,
                        "sell_rial": 1_464_500.0,
                        "mid_rial": 1_464_000.0,
                    }
                },
            }
        )
        self.assertEqual(point["preferred_rial"], 1_464_000.0)
        self.assertEqual(point["benchmark_open_market_rial"], 1_464_500.0)

    def test_detect_outliers_flags_extreme_source(self) -> None:
        outliers, details = model.detect_outliers(
            {
                "bonbast": 1_450_000.0,
                "alanchand_street": 1_452_000.0,
                "navasan": 1_780_000.0,
            }
        )
        self.assertEqual(outliers, ["navasan"])
        self.assertIn("navasan", details)

    def test_compute_confidence_score_rewards_more_sources_and_tighter_dispersion(self) -> None:
        low_score, _ = model.compute_confidence_score(
            source_count=1,
            dispersion_cv=0.04,
            historical_stability={"history_points": 2, "median_abs_daily_change_pct": 0.05},
            outlier_count=0,
        )
        high_score, _ = model.compute_confidence_score(
            source_count=3,
            dispersion_cv=0.005,
            historical_stability={"history_points": 10, "median_abs_daily_change_pct": 0.01},
            outlier_count=0,
        )
        self.assertGreater(high_score, low_score)

    def test_source_exclusion_reason_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            (site_dir / "fix").mkdir(parents=True)
            daily = self.make_daily_fix()
            (site_dir / "fix" / "2026-03-12.json").write_text(json.dumps(daily), encoding="utf-8")

            artifacts = model.build_benchmark_outputs(daily, site_dir, history_days=14)
            source_rows = {row["source_name"]: row for row in artifacts.diagnostics["source_values"]}
            navasan = source_rows["navasan"]

            self.assertFalse(navasan["eligible_for_benchmark"])
            self.assertEqual(navasan["quote_basis"], "inferred")
            self.assertEqual(navasan["exclusion_reason"], "unable to parse USD/IRR")
            self.assertFalse(navasan["included_in_benchmark"])
            self.assertIsNotNone(navasan["deviation_from_median"])

    def test_confidence_decomposition_is_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            (site_dir / "fix").mkdir(parents=True)
            daily = self.make_daily_fix()
            (site_dir / "fix" / "2026-03-12.json").write_text(json.dumps(daily), encoding="utf-8")

            artifacts = model.build_benchmark_outputs(daily, site_dir, history_days=14)

            self.assertIn("confidence_source_count_component", artifacts.diagnostics)
            self.assertIn("confidence_dispersion_component", artifacts.diagnostics)
            self.assertIn("confidence_stability_component", artifacts.diagnostics)
            self.assertEqual(artifacts.diagnostics["confidence_total"], artifacts.benchmark["confidence_score"])
            self.assertEqual(artifacts.daily_history["confidence_total"], artifacts.benchmark["confidence_score"])

    def test_benchmark_history_persistence_writes_daily_and_aggregate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            (site_dir / "fix").mkdir(parents=True)
            (site_dir / "api").mkdir(parents=True)
            day1 = self.make_daily_fix(date="2026-03-11", as_of="2026-03-11T16:42:19Z", bonbast_mid=1_479_000.0, alanchand_mid=1_459_750.0)
            day2 = self.make_daily_fix(date="2026-03-12", as_of="2026-03-12T16:42:19Z", bonbast_mid=1_464_000.0, alanchand_mid=1_434_250.0)
            (site_dir / "fix" / "2026-03-11.json").write_text(json.dumps(day1), encoding="utf-8")
            (site_dir / "fix" / "2026-03-12.json").write_text(json.dumps(day2), encoding="utf-8")

            history = model.write_history_outputs(site_dir, model.iter_fix_paths(site_dir), history_days=14)

            self.assertEqual(len(history["rows"]), 2)
            self.assertTrue((site_dir / "benchmark" / "2026-03-11.json").exists())
            self.assertTrue((site_dir / "benchmark" / "2026-03-12.json").exists())
            self.assertTrue((site_dir / "api" / "benchmark_history.json").exists())
            self.assertIn("eligible_sources", history["rows"][0])
            self.assertIn("diagnostics_summary", history["rows"][0])
            self.assertIn("confidence_source_count_component", history["rows"][0])
            self.assertIn("confidence_total", history["rows"][0])

    def test_benchmark_card_payload_generation_surfaces_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            (site_dir / "fix").mkdir(parents=True)
            daily = self.make_daily_fix()
            (site_dir / "fix" / "2026-03-12.json").write_text(json.dumps(daily), encoding="utf-8")

            artifacts = model.build_benchmark_outputs(daily, site_dir, history_days=14)

            self.assertEqual(artifacts.card["benchmark_rate"], artifacts.benchmark["weighted_rate"])
            self.assertEqual(artifacts.card["dispersion_level"], "low")
            self.assertIn("Navasan", artifacts.card["diagnostics_warning"])
            self.assertIn("moderate confidence", artifacts.card["benchmark_status_text"].lower())

    def test_methodology_payload_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_dir = Path(tmpdir)
            (site_dir / "fix").mkdir(parents=True)
            daily = self.make_daily_fix()
            (site_dir / "fix" / "2026-03-12.json").write_text(json.dumps(daily), encoding="utf-8")

            artifacts = model.build_benchmark_outputs(daily, site_dir, history_days=14)
            methodology = model.build_methodology_payload(artifacts.diagnostics)

            self.assertIn("eligibility_rules", methodology)
            self.assertIn("accepted_quote_bases", methodology)
            self.assertIn("bonbast", methodology["current_benchmark_eligible_sources"])
            self.assertIn("navasan", methodology["current_diagnostics_only_sources"])

    def test_timeseries_payload_generation(self) -> None:
        history_payload = {
            "rows": [
                {
                    "date": "2026-03-11",
                    "median_rate": 1_469_375.0,
                    "weighted_rate": 1_469_888.66,
                    "confidence_score": 59.28,
                    "source_count": 2,
                    "diagnostics_summary": {"dispersion_level": "low"},
                },
                {
                    "date": "2026-03-12",
                    "median_rate": 1_449_125.0,
                    "weighted_rate": 1_449_922.07,
                    "confidence_score": 68.19,
                    "source_count": 2,
                    "diagnostics_summary": {"dispersion_level": "low"},
                },
            ]
        }

        timeseries = model.build_benchmark_timeseries(history_payload)

        self.assertEqual(len(timeseries["rows"]), 2)
        self.assertEqual(timeseries["rows"][0]["benchmark_rate"], 1_469_888.66)
        self.assertEqual(timeseries["rows"][1]["dispersion_level"], "low")


if __name__ == "__main__":
    unittest.main()
