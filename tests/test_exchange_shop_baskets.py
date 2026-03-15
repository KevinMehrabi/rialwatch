import unittest

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


if __name__ == "__main__":
    unittest.main()
