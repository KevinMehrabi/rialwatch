import datetime as dt
import unittest

from scripts import pipeline


class RegimeModelTests(unittest.TestCase):
    def test_toman_to_rial_normalization(self) -> None:
        self.assertEqual(pipeline.normalize_unit(136_000.0, "toman"), 1_360_000.0)
        self.assertEqual(pipeline.normalize_unit(1_360_000.0, "rial"), 1_360_000.0)

    def test_gap_calculations(self) -> None:
        benchmark_results = {
            "open_market": {"fix": 1_600_000.0, "available": True},
            "official": {"fix": 1_380_000.0, "available": True},
            "regional_transfer": {"fix": 1_500_000.0, "available": True},
            "crypto_usdt": {"fix": 1_650_000.0, "available": True},
        }
        indicators = pipeline.compute_indicator_results(benchmark_results)
        self.assertAlmostEqual(indicators["street_official_gap_pct"]["value"], ((1_600_000 - 1_380_000) / 1_380_000) * 100)
        self.assertAlmostEqual(indicators["street_transfer_gap_pct"]["value"], ((1_600_000 - 1_500_000) / 1_500_000) * 100)
        self.assertAlmostEqual(indicators["street_crypto_gap_pct"]["value"], ((1_650_000 - 1_600_000) / 1_600_000) * 100)

    def test_official_sanity_warning(self) -> None:
        warnings = pipeline.evaluate_official_sanity_warnings(
            {
                "open_market": {"fix": 1_500_000.0},
                "official": {"fix": 2_100_000.0},
            }
        )
        self.assertTrue(warnings)
        self.assertTrue(any("above street_usd_irr" in msg for msg in warnings))

    def test_deprecated_fields_removed(self) -> None:
        self.assertNotIn("nima", pipeline.BENCHMARK_LABELS)
        self.assertNotIn("street_nima_gap", pipeline.INDICATOR_LABELS)
        self.assertNotIn("street_mobadeleh_gap", pipeline.INDICATOR_LABELS)

    def test_legacy_navasan_samples_apply_symbol_unit_map(self) -> None:
        sample = pipeline.parse_sample_record(
            "navasan",
            {
                "sampled_at": "2026-03-09T14:09:03Z",
                "value": 166400.0,
                "benchmarks": {
                    "open_market": 166400.0,
                    "official": 420000.0,
                    "regional_transfer": 169700.0,
                    "crypto_usdt": 149700.0,
                },
                "ok": True,
                "stale": False,
            },
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.source_unit, "mixed")
        self.assertEqual(sample.value, 1_664_000.0)
        self.assertEqual(sample.benchmark_values["open_market"], 1_664_000.0)
        self.assertEqual(sample.benchmark_values["official"], 420_000.0)

    def test_parse_source_payload_supports_navasan_js_assignment(self) -> None:
        body = (
            'var lastrates = {"mob_usd":{"value":"42,000"}};'
            'var yesterday = {"usd_sell":{"value":166400},"mex_usd_sell":{"value":1325072}};'
        )
        payload, mode = pipeline.parse_source_payload("navasan", body)
        self.assertEqual(mode, "javascript_var:yesterday")
        self.assertIn("mex_usd_sell", payload)
        self.assertEqual(payload["mex_usd_sell"]["value"], 1325072)

    def test_legacy_navasan_unknown_value_assumes_toman(self) -> None:
        sample = pipeline.parse_sample_record(
            "navasan",
            {
                "sampled_at": "2026-03-08T22:22:47Z",
                "value": 153500.0,
                "ok": True,
                "stale": False,
            },
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.value, 1_535_000.0)
        self.assertEqual(sample.source_unit, "mixed")

    def test_extract_value_by_symbol_candidates_supports_tgju_p_field(self) -> None:
        payload = {
            "current": {
                "ice_transfer_usd_sell": {
                    "p": "1,403,083",
                    "ts": "2026-03-27 00:00:00",
                }
            }
        }
        value = pipeline.extract_value_by_symbol_candidates(payload, ("ice_transfer_usd_sell",))
        self.assertEqual(value, 1_403_083.0)

    def test_extract_symbol_quote_time_supports_nested_symbol_paths(self) -> None:
        payload = {
            "current": {
                "ice_transfer_usd_sell": {
                    "p": "1,403,083",
                    "ts": "2026-03-27 00:00:00",
                }
            }
        }
        quote_time = pipeline.extract_symbol_quote_time(payload, ("ice_transfer_usd_sell",))
        self.assertIsNotNone(quote_time)
        assert quote_time is not None
        self.assertEqual(quote_time, dt.datetime(2026, 3, 27, 0, 0, tzinfo=dt.timezone.utc))

    def test_official_benchmark_prefers_freshest_source_quote(self) -> None:
        older = pipeline.Sample(
            source="navasan",
            sampled_at=dt.datetime(2026, 3, 29, 12, 10, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_325_072.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-27T00:00:00Z"},
            },
            source_unit="rial",
        )
        fresher = pipeline.Sample(
            source="tgju_call3",
            sampled_at=dt.datetime(2026, 3, 29, 12, 11, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_403_083.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:11:00Z"},
            },
            source_unit="rial",
        )
        samples = {
            "navasan": [older],
            "tgju_call3": [fresher],
        }
        benchmark_sources = {
            "navasan": ("official",),
            "tgju_call3": ("official",),
        }
        result = pipeline.compute_benchmark_result(samples, "official", benchmark_sources)
        self.assertEqual(result["fix"], 1_403_083.0)
        self.assertEqual(result["selection_method"], "freshest_quote_time")
        self.assertEqual(result["selected_sources"], ["tgju_call3"])
        self.assertEqual(result["source_update_counts"]["tgju_call3"], 1)
        self.assertFalse(result["withheld"])

    def test_official_benchmark_breaks_freshest_ties_by_update_cadence(self) -> None:
        navasan_only_latest = pipeline.Sample(
            source="navasan",
            sampled_at=dt.datetime(2026, 3, 29, 12, 11, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_325_072.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:11:00Z"},
            },
            source_unit="rial",
        )
        tgju_older = pipeline.Sample(
            source="tgju_call4",
            sampled_at=dt.datetime(2026, 3, 29, 12, 9, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_402_500.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:09:00Z"},
            },
            source_unit="rial",
        )
        tgju_latest = pipeline.Sample(
            source="tgju_call4",
            sampled_at=dt.datetime(2026, 3, 29, 12, 11, tzinfo=dt.timezone.utc),
            value=None,
            benchmark_values={"official": 1_403_083.0},
            quote_time=None,
            ok=True,
            stale=False,
            health={
                "fetch_success": True,
                "benchmark_quote_times": {"official": "2026-03-29T12:11:00Z"},
            },
            source_unit="rial",
        )
        samples = {
            "navasan": [navasan_only_latest],
            "tgju_call4": [tgju_older, tgju_latest],
        }
        benchmark_sources = {
            "navasan": ("official",),
            "tgju_call4": ("official",),
        }
        result = pipeline.compute_benchmark_result(samples, "official", benchmark_sources)
        self.assertEqual(result["fix"], 1_403_083.0)
        self.assertEqual(result["selection_method"], "freshest_quote_time_then_update_cadence")
        self.assertEqual(result["selected_sources"], ["tgju_call4"])
        self.assertEqual(result["source_update_counts"]["tgju_call4"], 2)
        self.assertEqual(result["source_update_counts"]["navasan"], 1)
        self.assertFalse(result["withheld"])

    def test_build_source_configs_includes_multiple_tgju_call_hosts(self) -> None:
        source_names = {cfg.name for cfg in pipeline.build_source_configs()}
        self.assertTrue({"tgju_call2", "tgju_call3", "tgju_call4"}.issubset(source_names))


if __name__ == "__main__":
    unittest.main()
