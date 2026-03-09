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


if __name__ == "__main__":
    unittest.main()
