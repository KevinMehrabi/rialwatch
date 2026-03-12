import unittest

from scripts import rialwatch_benchmark_model as model


class RialWatchBenchmarkModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
