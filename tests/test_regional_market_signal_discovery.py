import unittest
import tempfile
import json
from pathlib import Path

from scripts import regional_market_signal_discovery as regional


class RegionalMarketSignalDiscoveryTests(unittest.TestCase):
    def test_manual_germany_seeds_include_fresh_hilfen_sources(self) -> None:
        handles = {row["handle"] for row in regional.MANUAL_TELEGRAM_SEEDS if row.get("country_guess") == "Germany"}
        self.assertIn("berlinereuro", handles)
        self.assertIn("helfenassist", handles)
        self.assertIn("nord_deutschland", handles)

    def test_detect_regions_finds_multiple_targets(self) -> None:
        hits = regional.detect_regions("نرخ دلار هرات و دبی امروز")
        self.assertIn("Herat", hits)
        self.assertIn("Dubai", hits)
        germany_hits = regional.detect_regions("حواله یورو فقط بانک آلمان")
        self.assertIn("Germany", germany_hits)
        self.assertEqual(regional.region_to_basket("Germany"), "Germany")
        city_hits = regional.detect_regions("مونیخ یورو و کلن یورو برای حواله بانکی")
        self.assertIn("Munich", city_hits)
        self.assertIn("Cologne", city_hits)
        self.assertEqual(regional.region_to_basket("Munich"), "Germany")
        uk_hits = regional.detect_regions("نرخ پوند امروز لندن و حواله انگلستان")
        self.assertIn("London", uk_hits)
        self.assertIn("UK", uk_hits)
        self.assertEqual(regional.region_to_basket("London"), "UK")
        qatar_hits = regional.detect_regions("ریال قطر و حواله دوحه امروز")
        self.assertIn("Qatar", qatar_hits)
        self.assertIn("Doha", qatar_hits)
        self.assertEqual(regional.region_to_basket("Doha"), "Qatar")
        armenia_hits = regional.detect_regions("درام ارمنستان و حواله ایروان")
        self.assertIn("Armenia", armenia_hits)
        self.assertIn("Yerevan", armenia_hits)
        self.assertEqual(regional.region_to_basket("Yerevan"), "Armenia")

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

    def test_seed_from_manual_handles_includes_berlin_pay(self) -> None:
        seeded = regional.seed_from_manual_handles()
        self.assertIn("telegram:berlin_pay", seeded)
        self.assertIn("telegram:hmtransfer", seeded)
        source = seeded["telegram:berlin_pay"]
        self.assertEqual(source.country_guess, "Germany")
        self.assertIn("manual_germany_seed", source.origins)
        hmtransfer = seeded["telegram:hmtransfer"]
        self.assertEqual(hmtransfer.source_type_guess, "settlement_channel")
        self.assertIn("manual_germany_seed", hmtransfer.origins)
        self.assertIn("telegram:eurobazaar", seeded)
        self.assertIn("telegram:frankfurt_euro", seeded)
        self.assertIn("telegram:munich_euro", seeded)
        self.assertEqual(seeded["telegram:munich_euro"].city_guess, "Munich")

    def test_seed_from_manual_handles_includes_uk_sources(self) -> None:
        seeded = regional.seed_from_manual_handles()
        self.assertIn("telegram:poundbazar", seeded)
        source = seeded["telegram:poundbazar"]
        self.assertEqual(source.country_guess, "UK")
        self.assertEqual(source.city_guess, "London")
        self.assertIn("manual_uk_seed", source.origins)

    def test_seed_from_manual_handles_includes_qatar_armenia_sources(self) -> None:
        seeded = regional.seed_from_manual_handles()
        self.assertIn("telegram:royal_rate", seeded)
        self.assertEqual(seeded["telegram:royal_rate"].country_guess, "Qatar")
        self.assertIn("manual_qatar_seed", seeded["telegram:royal_rate"].origins)
        self.assertIn("telegram:arka_gold", seeded)
        self.assertIn("manual_qatar_armenia_seed", seeded["telegram:arka_gold"].origins)

    def test_seed_from_previous_candidates_replays_prior_regional_sources(self) -> None:
        seeded = regional.seed_from_previous_candidates(
            [
                {
                    "handle_or_url": "hmtransfer",
                    "platform": "telegram",
                    "country_guess": "Germany",
                    "city_guess": "Berlin",
                    "source_type": "settlement_channel",
                    "quote_message_count": "2",
                    "usable_record_count": "1",
                },
                {
                    "handle_or_url": "https://example.com/rates",
                    "platform": "website",
                    "country_guess": "UAE",
                    "city_guess": "Dubai",
                    "source_type": "exchange_shop",
                    "quote_message_count": "0",
                    "usable_record_count": "1",
                },
                {
                    "handle_or_url": "emptychannel",
                    "platform": "telegram",
                    "quote_message_count": "0",
                    "usable_record_count": "0",
                },
            ]
        )
        self.assertIn("telegram:hmtransfer", seeded)
        self.assertEqual(seeded["telegram:hmtransfer"].url, "https://t.me/s/hmtransfer")
        self.assertEqual(seeded["telegram:hmtransfer"].country_guess, "Germany")
        self.assertIn("previous_candidate_registry", seeded["telegram:hmtransfer"].origins)
        self.assertIn("website:https://example.com/rates", seeded)
        self.assertNotIn("telegram:emptychannel", seeded)

    def test_seed_from_source_registry_replays_active_regional_sources(self) -> None:
        payload = {
            "sources": [
                {
                    "platform": "telegram",
                    "handle_or_url": "sarafitehran",
                    "public_url": "https://t.me/s/sarafitehran",
                    "country_guess": "Iran",
                    "city_guess": "Tehran",
                    "source_kind": "exchange_shop",
                    "signal_families": ["direct_shop_expansion"],
                    "last_success_at": "2026-05-01T12:00:00Z",
                },
                {
                    "platform": "telegram",
                    "handle_or_url": "inactive",
                    "signal_families": ["regional_market_signal"],
                },
            ]
        }
        seeded = regional.seed_from_source_registry(payload)
        self.assertIn("telegram:sarafitehran", seeded)
        source = seeded["telegram:sarafitehran"]
        self.assertEqual(source.country_guess, "Iran")
        self.assertEqual(source.source_type_guess, "exchange_shop")
        self.assertIn("source_registry", source.origins)
        self.assertNotIn("telegram:inactive", seeded)

    def test_classify_source_type_honors_regional_market_hint(self) -> None:
        source = regional.DiscoverySource(
            key="telegram:test",
            platform="telegram",
            url="https://t.me/s/test",
            handle_or_url="test",
            source_type_guess="regional_market_channel",
        )
        self.assertEqual(regional.classify_source_type(source, "Berlin Pay", ""), "regional_market_channel")

    def test_summarize_enriched_basket_keeps_min_three_source_coverage(self) -> None:
        records = []
        for offset in (0.0, 1200.0, -900.0, 700.0):
            records.append(
                regional.BasketRecord(
                    handle="source_a",
                    title="A",
                    locality="Germany",
                    source_category="regional_market_channel",
                    source_priority="regional_discovery",
                    likely_individual_shop=False,
                    channel_type_guess="regional_market_channel",
                    normalized_rate_rial=1_550_000.0 + offset,
                    quote_basis="midpoint",
                    overall_quality=76.0,
                    freshness_score=78.0,
                    structure_score=76.0,
                    directness_score=70.0,
                    timestamp_iso="2026-03-21T12:00:00Z",
                    dedup_keep=True,
                    duplication_flag="none",
                    from_new_p1=False,
                    channel_readiness_score=74.0,
                )
            )
        for offset in (0.0, 900.0, -800.0):
            records.append(
                regional.BasketRecord(
                    handle="source_b",
                    title="B",
                    locality="Germany",
                    source_category="regional_market_channel",
                    source_priority="regional_discovery",
                    likely_individual_shop=False,
                    channel_type_guess="regional_market_channel",
                    normalized_rate_rial=1_560_000.0 + offset,
                    quote_basis="midpoint",
                    overall_quality=74.0,
                    freshness_score=74.0,
                    structure_score=74.0,
                    directness_score=68.0,
                    timestamp_iso="2026-03-21T12:00:00Z",
                    dedup_keep=True,
                    duplication_flag="none",
                    from_new_p1=False,
                    channel_readiness_score=72.0,
                )
            )
        # Third source is an outlier candidate that MAD trimming can remove.
        records.append(
            regional.BasketRecord(
                handle="source_c",
                title="C",
                locality="Germany",
                source_category="aggregator",
                source_priority="regional_discovery",
                likely_individual_shop=False,
                channel_type_guess="aggregator",
                normalized_rate_rial=2_450_000.0,
                quote_basis="midpoint",
                overall_quality=65.0,
                freshness_score=62.0,
                structure_score=65.0,
                directness_score=55.0,
                timestamp_iso="2026-03-21T12:00:00Z",
                dedup_keep=True,
                duplication_flag="none",
                from_new_p1=False,
                channel_readiness_score=60.0,
            )
        )
        row = regional.summarize_enriched_basket("Germany", records, benchmark_value=1_449_922.07)
        self.assertGreaterEqual(row["contributing_source_count"], 3)


if __name__ == "__main__":
    unittest.main()
