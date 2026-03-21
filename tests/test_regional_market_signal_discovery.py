import unittest
import tempfile
import json
from pathlib import Path

from scripts import regional_market_signal_discovery as regional


class RegionalMarketSignalDiscoveryTests(unittest.TestCase):
    def test_detect_regions_finds_multiple_targets(self) -> None:
        hits = regional.detect_regions("نرخ دلار هرات و دبی امروز")
        self.assertIn("Herat", hits)
        self.assertIn("Dubai", hits)
        germany_hits = regional.detect_regions("حواله یورو فقط بانک آلمان")
        self.assertIn("Germany", germany_hits)
        self.assertEqual(regional.region_to_basket("Germany"), "Germany")

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

    def test_summarize_enriched_basket_preserves_two_source_diversity(self) -> None:
        stale_records = [
            regional.BasketRecord(
                handle="source_old",
                title="Source Old",
                locality="Germany",
                source_category="aggregator",
                source_priority="regional_discovery",
                likely_individual_shop=False,
                channel_type_guess="aggregator",
                normalized_rate_rial=830_000.0 + offset,
                quote_basis="midpoint",
                overall_quality=62.0,
                freshness_score=20.0,
                structure_score=62.0,
                directness_score=55.0,
                timestamp_iso="2026-01-15T12:00:00Z",
                dedup_keep=True,
                duplication_flag="none",
                from_new_p1=False,
                channel_readiness_score=60.0,
            )
            for offset in (0.0, 1_000.0, -1_200.0, 800.0, -600.0, 900.0, -750.0, 650.0)
        ]
        fresh_records = [
            regional.BasketRecord(
                handle="source_new",
                title="Source New",
                locality="Germany",
                source_category="regional_market_channel",
                source_priority="regional_discovery",
                likely_individual_shop=False,
                channel_type_guess="regional_market_channel",
                normalized_rate_rial=1_930_000.0 + offset,
                quote_basis="midpoint",
                overall_quality=78.0,
                freshness_score=82.0,
                structure_score=78.0,
                directness_score=70.0,
                timestamp_iso="2026-03-21T12:00:00Z",
                dedup_keep=True,
                duplication_flag="none",
                from_new_p1=False,
                channel_readiness_score=75.0,
            )
            for offset in (0.0, 3_500.0)
        ]
        row = regional.summarize_enriched_basket(
            "Germany",
            stale_records + fresh_records,
            benchmark_value=1_449_922.07,
        )
        self.assertEqual(row["contributing_source_count"], 2)
        self.assertNotEqual(row["suppression_reason"], "stale_signal")

    def test_seed_from_quote_message_samples_includes_germany_hint_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            survey_dir = Path(tmp_dir)
            payload = [
                {
                    "handle": "eurobazaar",
                    "title": "یورو بازار بانکی",
                    "public_url": "https://t.me/s/eurobazaar",
                    "channel_type_guess": "market_price_channel",
                    "quote_message_records": [
                        {
                            "message_text_sample": "حواله یورو فقط بانک آلمان نرخ 193,000 تومان",
                            "city_mentions": [],
                        }
                    ],
                }
            ]
            (survey_dir / "quote_message_samples.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            seeded = regional.seed_from_quote_message_samples(survey_dir)
            self.assertIn("telegram:eurobazaar", seeded)
            source = seeded["telegram:eurobazaar"]
            self.assertEqual(source.country_guess, "Germany")
            self.assertIn("quote_sample_hint", source.origins)


if __name__ == "__main__":
    unittest.main()
