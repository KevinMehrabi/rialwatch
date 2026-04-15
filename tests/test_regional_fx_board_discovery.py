import unittest

from scripts import regional_fx_board_discovery as boards


class RegionalFxBoardDiscoveryTests(unittest.TestCase):
    def test_extract_locality_quotes_handles_multiline_board(self) -> None:
        text = """
        تهران 147300 / 147500
        هرات 147600 / 147700
        سلیمانیه 147800 / 147900
        دبی درهم 40300
        """
        results = boards.extract_locality_quotes(text, benchmark_value=1_449_922.07)
        by_locality = {loc: midpoint for loc, _buy, _sell, midpoint, _unit, _basis, _currency in results}
        self.assertIn("Tehran", by_locality)
        self.assertIn("Herat", by_locality)
        self.assertIn("Sulaymaniyah", by_locality)
        self.assertIn("Dubai", by_locality)
        self.assertGreater(by_locality["Dubai"], 1_400_000.0)

    def test_classify_source_type_prefers_board(self) -> None:
        source_type = boards.classify_source_type(
            "تابلوی نرخ ارز",
            [
                "تهران 147300 / 147500",
                "هرات 147600 / 147700",
                "سلیمانیه 147800 / 147900",
                "دبی 1500000",
            ],
        )
        self.assertEqual(source_type, "regional_fx_board")

    def test_summarize_locality_monitor_when_limited(self) -> None:
        rows = [
            boards.BoardRecord(
                handle="a",
                title="a",
                message_text_sample="تهران 147300 / 147500",
                localities_detected="Tehran",
                tehran_quote="1474000.00",
                herat_quote="",
                sulaymaniyah_quote="",
                erbil_quote="",
                baghdad_quote="",
                iraq_quote="",
                dubai_quote="",
                istanbul_quote="",
                hamburg_quote="",
                berlin_quote="",
                germany_quote="",
                london_quote="",
                frankfurt_quote="",
                inferred_unit="toman",
                normalized_irr_values="{}",
                buy_quote="147300",
                sell_quote="147500",
                midpoint="147400.00",
                freshness_indicator="recent",
                parseability_score=72,
                quote_density_score=70,
                source_type="regional_fx_board",
                timestamp_iso="2026-03-16T00:00:00Z",
                locality_name="Dubai",
                normalized_rate_irr=1_474_000.0,
                quote_basis="midpoint",
                quote_currency_guess="AED",
            ),
            boards.BoardRecord(
                handle="b",
                title="b",
                message_text_sample="دبی 147800 / 148000",
                localities_detected="Dubai",
                tehran_quote="",
                herat_quote="",
                sulaymaniyah_quote="",
                erbil_quote="",
                baghdad_quote="",
                iraq_quote="",
                dubai_quote="1479000.00",
                istanbul_quote="",
                hamburg_quote="",
                berlin_quote="",
                germany_quote="",
                london_quote="",
                frankfurt_quote="",
                inferred_unit="toman",
                normalized_irr_values="{}",
                buy_quote="147800",
                sell_quote="148000",
                midpoint="147900.00",
                freshness_indicator="recent",
                parseability_score=70,
                quote_density_score=66,
                source_type="regional_market_channel",
                timestamp_iso="2026-03-16T00:00:00Z",
                locality_name="Dubai",
                normalized_rate_irr=1_479_000.0,
                quote_basis="midpoint",
                quote_currency_guess="AED",
            ),
        ]
        summary = boards.summarize_locality("Dubai", rows, benchmark_value=1_449_922.07)
        self.assertEqual(summary["recommended_display_state"], "monitor")


if __name__ == "__main__":
    unittest.main()
