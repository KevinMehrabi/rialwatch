import unittest

from scripts import regional_market_signals_card as cards


class RegionalMarketSignalsCardTests(unittest.TestCase):
    def test_alignment_label_from_spread(self) -> None:
        self.assertEqual(cards.alignment_label_from_spread(1.9), "Aligned")
        self.assertEqual(cards.alignment_label_from_spread(2.0), "Mild divergence")
        self.assertEqual(cards.alignment_label_from_spread(-10.0), "Mild divergence")
        self.assertEqual(cards.alignment_label_from_spread(10.01), "Divergent")

    def test_display_state_from_publishable_hides_stale(self) -> None:
        self.assertEqual(cards.display_state_from_publishable(True, 0, ""), "publish")
        self.assertEqual(cards.display_state_from_publishable(False, 4, "limited_coverage"), "monitor")
        self.assertEqual(cards.display_state_from_publishable(False, 7, "stale_signal"), "hide")
        self.assertEqual(cards.display_state_from_publishable(False, 0, "no_usable_records"), "hide")

    def test_build_payload_selects_best_signal_per_locality(self) -> None:
        regional_payload = {
            "generated_at": "2026-03-16T17:42:27Z",
            "localities": [
                {
                    "locality_name": "Iran",
                    "signal_type_used": "regional_market_channel",
                    "usable_record_count": 2,
                    "contributing_source_count": 2,
                    "weighted_rate": 878396.16,
                    "median_rate": 859250.0,
                    "spread_vs_benchmark_pct": -39.4177,
                    "freshness_status": "old",
                    "dispersion_level": "medium",
                    "basket_confidence": 37.75,
                    "recommended_display_state": "monitor",
                    "suppression_reason": "needs_more_fresh_sources",
                },
                {
                    "locality_name": "Afghanistan",
                    "signal_type_used": "regional_fx_board",
                    "usable_record_count": 33,
                    "contributing_source_count": 5,
                    "weighted_rate": 1276847.69,
                    "median_rate": 1444000.0,
                    "spread_vs_benchmark_pct": -11.9368,
                    "freshness_status": "fresh",
                    "dispersion_level": "high",
                    "basket_confidence": 76.11,
                    "recommended_display_state": "publish",
                    "suppression_reason": "",
                },
                {
                    "locality_name": "Iraq",
                    "signal_type_used": "regional_market_channel",
                    "usable_record_count": 12,
                    "contributing_source_count": 1,
                    "weighted_rate": 1545161.68,
                    "median_rate": 1544000.0,
                    "spread_vs_benchmark_pct": 6.5686,
                    "freshness_status": "fresh",
                    "dispersion_level": "low",
                    "basket_confidence": 76.5,
                    "recommended_display_state": "monitor",
                    "suppression_reason": "limited_coverage",
                },
                {
                    "locality_name": "UAE",
                    "signal_type_used": "exchange_shop",
                    "usable_record_count": 12,
                    "contributing_source_count": 2,
                    "weighted_rate": 1475435.34,
                    "median_rate": 1469000.0,
                    "spread_vs_benchmark_pct": 1.7596,
                    "freshness_status": "recent",
                    "dispersion_level": "low",
                    "basket_confidence": 76.35,
                    "recommended_display_state": "publish",
                    "suppression_reason": "",
                },
                {
                    "locality_name": "Turkey",
                    "recommended_display_state": "hide",
                    "suppression_reason": "no_usable_records",
                },
                {
                    "locality_name": "UK",
                    "recommended_display_state": "hide",
                    "suppression_reason": "no_usable_records",
                },
                {
                    "locality_name": "Germany",
                    "recommended_display_state": "hide",
                    "suppression_reason": "no_usable_records",
                },
            ],
        }

        enriched_payload = {
            "generated_at": "2026-03-16T03:16:43Z",
            "baskets": [
                {
                    "basket_name": "Iran",
                    "signal_type_used": "exchange_shop",
                    "weighted_rate": 1416917.85,
                    "median_rate": 1409950.0,
                    "spread_vs_benchmark_pct": -2.2763,
                    "usable_record_count": 121,
                    "contributing_source_count": 15,
                    "basket_confidence": 64.16,
                    "publishable": True,
                    "suppression_reason": "",
                    "dispersion_cv": 0.248919,
                },
                {
                    "basket_name": "Turkey",
                    "signal_type_used": "exchange_shop",
                    "weighted_rate": 1448558.06,
                    "median_rate": 1448200.0,
                    "spread_vs_benchmark_pct": -0.0941,
                    "usable_record_count": 33,
                    "contributing_source_count": 2,
                    "basket_confidence": 89.07,
                    "publishable": True,
                    "suppression_reason": "",
                    "dispersion_cv": 0.008424,
                },
                {
                    "basket_name": "UK",
                    "signal_type_used": "exchange_shop",
                    "weighted_rate": 1862657.9,
                    "median_rate": 1865000.0,
                    "spread_vs_benchmark_pct": 28.4661,
                    "usable_record_count": 18,
                    "contributing_source_count": 3,
                    "basket_confidence": 87.69,
                    "publishable": True,
                    "suppression_reason": "",
                    "dispersion_cv": 0.029718,
                },
                {
                    "basket_name": "Germany",
                    "signal_type_used": "aggregator",
                    "weighted_rate": 820357.14,
                    "median_rate": 830000.0,
                    "spread_vs_benchmark_pct": -43.4206,
                    "usable_record_count": 7,
                    "contributing_source_count": 1,
                    "basket_confidence": 59.36,
                    "publishable": False,
                    "suppression_reason": "stale_signal",
                    "dispersion_cv": 0.042678,
                },
            ],
        }

        legacy_payload = {"generated_at": "2026-03-16T17:30:39Z", "cards": []}

        payload = cards.build_regional_market_cards_payload(
            regional_payload,
            enriched_payload,
            legacy_payload,
        )
        self.assertEqual(
            [row["basket_name"] for row in payload["cards"]],
            ["Iran", "UAE", "Turkey", "Afghanistan", "UK", "Iraq", "Germany"],
        )
        by_locality = {row["basket_name"]: row for row in payload["cards"]}

        self.assertEqual(by_locality["Iran"]["display_state"], "publish")
        self.assertEqual(by_locality["Iran"]["source_artifact"], "exchange_shop_baskets_enriched")
        self.assertEqual(by_locality["Iran"]["signal_label"], "Exchange network signal")
        self.assertEqual(by_locality["Iran"]["alignment_label"], "Mild divergence")
        self.assertEqual(by_locality["Turkey"]["display_state"], "publish")
        self.assertEqual(by_locality["Turkey"]["signal_label"], "Exchange network signal")
        self.assertEqual(by_locality["Turkey"]["alignment_label"], "Aligned")
        self.assertEqual(by_locality["UK"]["display_state"], "publish")
        self.assertEqual(by_locality["UK"]["signal_label"], "Exchange network signal")
        self.assertEqual(by_locality["UK"]["alignment_label"], "Divergent")
        self.assertEqual(by_locality["UAE"]["display_state"], "publish")
        self.assertEqual(by_locality["UAE"]["signal_label"], "Dubai settlement signal")
        self.assertEqual(by_locality["UAE"]["alignment_label"], "Aligned")
        self.assertEqual(by_locality["Afghanistan"]["display_state"], "publish")
        self.assertEqual(by_locality["Afghanistan"]["signal_label"], "Herat market signal")
        self.assertEqual(by_locality["Afghanistan"]["alignment_label"], "Divergent")
        self.assertEqual(by_locality["Iraq"]["display_state"], "monitor")
        self.assertEqual(by_locality["Iraq"]["signal_label"], "Sulaymaniyah market (monitoring)")
        self.assertEqual(by_locality["Iraq"]["alignment_label"], "Mild divergence")
        self.assertEqual(by_locality["Germany"]["display_state"], "hide")
        self.assertFalse(by_locality["Germany"]["render_on_homepage"])
        self.assertEqual(by_locality["Germany"]["alignment_label"], "Divergent")

        self.assertEqual(payload["summary"]["publish_count"], 5)
        self.assertEqual(payload["summary"]["monitor_count"], 1)
        self.assertEqual(payload["summary"]["hide_count"], 1)


if __name__ == "__main__":
    unittest.main()
