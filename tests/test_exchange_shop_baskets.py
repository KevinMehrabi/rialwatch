import json
import tempfile
import unittest
from pathlib import Path

from scripts import exchange_shop_baskets as baskets


class ExchangeShopBasketTests(unittest.TestCase):
    def make_record(
        self,
        *,
        handle: str,
        locality: str,
        rate: float,
        category: str = "direct_shop",
        quality: float = 82.0,
        freshness: float = 88.0,
        directness: float = 90.0,
    ) -> baskets.BasketRecord:
        return baskets.BasketRecord(
            handle=handle,
            title=handle,
            locality=locality,
            source_category=category,
            source_priority="P1",
            likely_individual_shop=(category == "direct_shop"),
            channel_type_guess="individual_exchange_shop",
            normalized_rate_rial=rate,
            quote_basis="midpoint",
            overall_quality=quality,
            freshness_score=freshness,
            structure_score=90.0,
            directness_score=directness,
            timestamp_iso="2026-03-15T12:00:00Z",
            dedup_keep=True,
            duplication_flag="none",
            from_new_p1=False,
            channel_readiness_score=85.0,
        )

    def test_rate_from_quote_row_prefers_midpoint(self) -> None:
        rate, basis = baskets.rate_from_quote_row(
            {
                "midpoint_rial": "1450000",
                "sell_quote_rial": "1455000",
                "buy_quote_rial": "1445000",
            }
        )
        self.assertEqual(rate, 1_450_000.0)
        self.assertEqual(basis, "midpoint")

    def test_benchmark_rate_prefers_current_latest_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            site_api_dir = Path(tmpdir)
            (site_api_dir / "benchmark.json").write_text(
                json.dumps({"weighted_rate": 1_449_922.07}),
                encoding="utf-8",
            )
            (site_api_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "computed": {
                            "fix": 1_773_250.0,
                            "benchmarks": {
                                "open_market": {
                                    "value": 1_773_250.0,
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(baskets.benchmark_rate(site_api_dir), 1_773_250.0)

    def test_fallback_usd_rate_from_text_handles_two_column_toman_posts(self) -> None:
        rate, basis = baskets.fallback_usd_rate_from_text(
            "نرخ ارزهای استانبول با تومان نام ارز فروش خرید دلار 149,620 144,400 یورو 174,320 167,800",
            unit_guess="toman",
            min_rate=1_000_000.0,
            max_rate=2_000_000.0,
        )
        self.assertEqual(rate, 1_470_100.0)
        self.assertEqual(basis, "midpoint")

    def test_summarize_basket_flags_single_channel_concentration(self) -> None:
        records = [
            self.make_record(handle="solo", locality="Iran", rate=1_450_000 + offset)
            for offset in (0, 1_000, -800, 700)
        ]
        row = baskets.summarize_basket("Iran", records, benchmark_value=1_449_922.07)
        self.assertFalse(row["publishable"])
        self.assertEqual(row["suppression_reason"], "single_channel_concentration")
        self.assertEqual(row["usable_record_count"], 4)

    def test_summarize_basket_is_publishable_with_two_clean_channels(self) -> None:
        records = [
            self.make_record(handle="iran_a", locality="Iran", rate=1_450_000),
            self.make_record(handle="iran_a", locality="Iran", rate=1_451_000),
            self.make_record(handle="iran_b", locality="Iran", rate=1_448_500),
            self.make_record(handle="iran_b", locality="Iran", rate=1_449_500),
            self.make_record(handle="iran_b", locality="Iran", rate=1_450_500),
        ]
        row = baskets.summarize_basket("Iran", records, benchmark_value=1_449_922.07)
        self.assertTrue(row["publishable"])
        self.assertGreater(row["basket_confidence"], 55.0)
        self.assertEqual(row["contributing_channel_count"], 2)

    def test_summarize_basket_keeps_tight_cluster_source_with_tiny_mad(self) -> None:
        records = [
            self.make_record(handle="uae_a", locality="UAE", rate=1_591_386.06),
            self.make_record(handle="uae_a", locality="UAE", rate=1_592_983.60),
            self.make_record(handle="uae_b", locality="UAE", rate=1_593_865.00),
            self.make_record(handle="uae_c", locality="UAE", rate=1_590_192.50),
            self.make_record(handle="uae_c", locality="UAE", rate=1_590_192.50),
            self.make_record(handle="uae_c", locality="UAE", rate=1_590_192.50),
            self.make_record(handle="uae_d", locality="UAE", rate=1_610_024.00),
            self.make_record(handle="uae_d", locality="UAE", rate=1_610_024.00),
        ]
        row = baskets.summarize_basket("UAE", records, benchmark_value=1_449_922.07)
        self.assertEqual(row["contributing_channel_count"], 4)
        self.assertEqual(row["outliers_removed"], 0)

    def test_build_card_payload_preserves_required_fields(self) -> None:
        payload = baskets.build_card_payload(
            basket_rows=[
                {
                    "basket_name": "Turkey",
                    "weighted_rate": 1_496_200.0,
                    "median_rate": 1_495_800.0,
                    "spread_vs_benchmark_pct": 3.88,
                    "usable_record_count": 8,
                    "contributing_channel_count": 2,
                    "basket_confidence": 67.5,
                    "publishable": True,
                    "suppression_reason": "",
                }
            ],
            network_summary={"generated_at": "2026-03-15T18:10:00Z", "benchmark_weighted_rate": 1_449_922.07},
        )
        self.assertEqual(payload["cards"][0]["basket_name"], "Turkey")
        self.assertTrue(payload["cards"][0]["publishable"])
        self.assertEqual(payload["cards"][0]["rate_text"], "1,496,200 IRR")
        self.assertEqual(payload["cards"][0]["spread_text"], "+3.88%")

    def test_build_card_payload_hides_watchlist_only_locality(self) -> None:
        payload = baskets.build_card_payload(
            basket_rows=[
                {
                    "basket_name": "Turkey",
                    "weighted_rate": 1_496_200.0,
                    "median_rate": 1_495_800.0,
                    "spread_vs_benchmark_pct": 3.88,
                    "usable_record_count": 8,
                    "contributing_channel_count": 2,
                    "basket_confidence": 67.5,
                    "publishable": True,
                    "suppression_reason": "",
                },
                {
                    "basket_name": "UAE",
                    "weighted_rate": None,
                    "median_rate": None,
                    "spread_vs_benchmark_pct": None,
                    "usable_record_count": 0,
                    "contributing_channel_count": 0,
                    "basket_confidence": 0.0,
                    "publishable": False,
                    "suppression_reason": "no_usable_records",
                },
            ],
            network_summary={"generated_at": "2026-03-15T18:10:00Z", "benchmark_weighted_rate": 1_449_922.07},
            locality_internal_status={"UAE": "watchlist_only"},
        )
        self.assertEqual([card["basket_name"] for card in payload["cards"]], ["Turkey"])

    def test_normalize_uae_review_records_converts_aed_to_usd_irr(self) -> None:
        rows = baskets.normalize_uae_review_records(
            candidate_rows=[
                {
                    "business_name": "Dubai AED Desk",
                    "surface_url": "https://hiemarat.com/",
                    "currency": "AED",
                    "normalized_irr_value": "423600",
                    "freshness_status": "fresh",
                    "parseability_score": "74",
                    "quote_basis": "single_value",
                    "timestamp_iso": "2026-04-28T12:00:00Z",
                    "remittance_quote_detected": "true",
                }
            ],
            review_rows=[
                {
                    "business_name": "Dubai AED Desk",
                    "basket_use_status": "monitor_only",
                }
            ],
            benchmark_value=1_550_000.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].locality, "UAE")
        self.assertAlmostEqual(rows[0].normalized_rate_rial, 1_555_671.0, places=1)

    def test_normalize_uae_review_records_accepts_high_quality_usd_quote(self) -> None:
        rows = baskets.normalize_uae_review_records(
            candidate_rows=[
                {
                    "business_name": "Dubai USD Desk",
                    "surface_url": "https://iranianuae.ae/gold-forex",
                    "currency": "USD",
                    "normalized_irr_value": "1606500",
                    "freshness_status": "fresh",
                    "parseability_score": "82",
                    "quote_basis": "single_value",
                    "timestamp_iso": "2026-04-28T12:00:00Z",
                    "remittance_quote_detected": "false",
                },
                {
                    "business_name": "Noisy USD Desk",
                    "surface_url": "https://example.com/rates",
                    "currency": "USD",
                    "normalized_irr_value": "2000000",
                    "freshness_status": "fresh",
                    "parseability_score": "54",
                    "quote_basis": "single_value",
                    "timestamp_iso": "2026-04-28T12:00:00Z",
                    "remittance_quote_detected": "true",
                },
            ],
            review_rows=[
                {
                    "business_name": "Dubai USD Desk",
                    "basket_use_status": "monitor_only",
                },
                {
                    "business_name": "Noisy USD Desk",
                    "basket_use_status": "monitor_only",
                },
            ],
            benchmark_value=1_550_000.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].handle, "uae:iranianuae.ae")
        self.assertEqual(rows[0].quote_basis, "USD_single_value")
        self.assertEqual(rows[0].normalized_rate_rial, 1_606_500.0)


if __name__ == "__main__":
    unittest.main()
