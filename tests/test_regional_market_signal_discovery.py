import unittest

from scripts import regional_market_signal_discovery as regional


class RegionalMarketSignalDiscoveryTests(unittest.TestCase):
    def test_detect_regions_finds_multiple_targets(self) -> None:
        hits = regional.detect_regions("نرخ دلار هرات و دبی امروز")
        self.assertIn("Herat", hits)
        self.assertIn("Dubai", hits)

    def test_summarize_enriched_basket_allows_single_source_diagnostics_when_rich(self) -> None:
        records = [
            regional.BasketRecord(
                handle="source_a",
                title="Source A",
                locality="Afghanistan",
                source_category="regional_market_channel",
                source_priority="regional_discovery",
                likely_individual_shop=False,
                channel_type_guess="regional_market_channel",
                normalized_rate_rial=1_430_000.0 + offset,
                quote_basis="midpoint",
                overall_quality=82.0,
                freshness_score=88.0,
                structure_score=84.0,
                directness_score=70.0,
                timestamp_iso="2026-03-15T12:00:00Z",
                dedup_keep=True,
                duplication_flag="none",
                from_new_p1=False,
                channel_readiness_score=80.0,
            )
            for offset in (0.0, 2_500.0, -2_000.0, 3_000.0)
        ]
        row = regional.summarize_enriched_basket("Afghanistan", records, benchmark_value=1_449_922.07)
        self.assertTrue(row["publishable"])
        self.assertEqual(row["signal_type_used"], "regional_market_channel")
        self.assertEqual(row["contributing_source_count"], 1)


if __name__ == "__main__":
    unittest.main()
