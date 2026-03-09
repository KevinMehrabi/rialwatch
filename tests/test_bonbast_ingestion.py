import datetime as dt
import unittest

from scripts import pipeline


class BonbastIngestionTests(unittest.TestCase):
    def test_selector_extraction_and_mid_normalization(self) -> None:
        selector_results = {
            "open_market": [{"selector": "#usd1", "text": "149950"}],
            "open_market_buy": [{"selector": "#usd2", "text": "149850"}],
        }
        sell_raw, sell_selector = pipeline.extract_bonbast_selector_numeric(selector_results, "open_market", 100_000, 3_000_000)
        buy_raw, buy_selector = pipeline.extract_bonbast_selector_numeric(selector_results, "open_market_buy", 100_000, 3_000_000)

        self.assertEqual(sell_selector, "#usd1")
        self.assertEqual(buy_selector, "#usd2")
        self.assertEqual(sell_raw, 149_950.0)
        self.assertEqual(buy_raw, 149_850.0)

        sell_rial = pipeline.normalize_unit(sell_raw, "toman")
        buy_rial = pipeline.normalize_unit(buy_raw, "toman")
        mid_rial = (sell_rial + buy_rial) / 2.0
        self.assertEqual(sell_rial, 1_499_500.0)
        self.assertEqual(buy_rial, 1_498_500.0)
        self.assertEqual(mid_rial, 1_499_000.0)

    def test_validate_bonbast_quote(self) -> None:
        ok = pipeline.validate_bonbast_usd_quote(1_498_500.0, 1_499_500.0, 0.05)
        self.assertTrue(ok["ok"])
        self.assertGreater(ok["spread_pct"], 0)

        bad = pipeline.validate_bonbast_usd_quote(1_600_000.0, 1_500_000.0, 0.05)
        self.assertFalse(bad["ok"])
        self.assertIn("buy quote above sell quote", bad["reason"])

    def test_peer_validation_marks_bonbast_failed_when_outside_band(self) -> None:
        now = dt.datetime(2026, 3, 9, 14, 0, tzinfo=dt.timezone.utc)
        bonbast_sample = pipeline.Sample(
            source="bonbast",
            sampled_at=now,
            value=2_000_000.0,
            benchmark_values={"open_market": 2_000_000.0},
            quote_time=now,
            ok=True,
            stale=False,
            health={"fetch_mode": "playwright", "fetch_success": True},
        )
        navasan_sample = pipeline.Sample(
            source="navasan",
            sampled_at=now,
            value=1_500_000.0,
            benchmark_values={"open_market": 1_500_000.0},
            quote_time=now,
            ok=True,
            stale=False,
        )
        alanchand_sample = pipeline.Sample(
            source="alanchand",
            sampled_at=now,
            value=1_520_000.0,
            benchmark_values={"open_market": 1_520_000.0},
            quote_time=now,
            ok=True,
            stale=False,
        )

        samples = {
            "bonbast": [bonbast_sample],
            "navasan": [navasan_sample],
            "alanchand": [alanchand_sample],
        }
        pipeline.apply_bonbast_peer_validation(samples, max_deviation_pct=0.10)

        self.assertFalse(bonbast_sample.ok)
        self.assertEqual(bonbast_sample.error, "outside peer plausibility band")
        self.assertEqual(bonbast_sample.health.get("failure_reason"), "outside peer plausibility band")
        validation = bonbast_sample.health.get("validation_result", {})
        self.assertEqual(validation.get("peer_band_result"), "failed")


if __name__ == "__main__":
    unittest.main()
