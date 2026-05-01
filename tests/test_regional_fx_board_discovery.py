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

    def test_extract_locality_quotes_treats_emirates_dirham_as_dubai(self) -> None:
        text = "حواله درهم امارات 42,360 تومان"
        results = boards.extract_locality_quotes(text, benchmark_value=1_550_000.0)
        by_locality = {loc: midpoint for loc, _buy, _sell, midpoint, _unit, _basis, _currency in results}
        self.assertIn("Dubai", by_locality)
        self.assertGreater(by_locality["Dubai"], 1_500_000.0)

    def test_extract_locality_quotes_keeps_london_pound_signal(self) -> None:
        text = "نرخ پوند امروز لندن خرید حساب رسمی: 190,000 فروش حساب رسمی: 215,000 تومان"
        results = boards.extract_locality_quotes(text, benchmark_value=1_650_000.0)
        by_locality = {loc: (midpoint, currency) for loc, _buy, _sell, midpoint, _unit, _basis, currency in results}
        self.assertIn("London", by_locality)
        self.assertEqual(by_locality["London"][1], "GBP")
        self.assertAlmostEqual(by_locality["London"][0], 1_557_692.31, places=2)

    def test_extract_locality_quotes_keeps_germany_euro_signal(self) -> None:
        text = "حواله بانکی آلمان فروش یورو 185,000 تومان"
        results = boards.extract_locality_quotes(text, benchmark_value=1_650_000.0)
        by_locality = {loc: (midpoint, currency) for loc, _buy, _sell, midpoint, _unit, _basis, currency in results}
        self.assertIn("Germany", by_locality)
        self.assertEqual(by_locality["Germany"][1], "EUR")
        self.assertAlmostEqual(by_locality["Germany"][0], 1_622_807.02, places=2)

    def test_extract_locality_quotes_uses_rate_keyword_over_listing_id(self) -> None:
        text = "حواله 68788 بابت #فروش یورو نرخ پیشنهادی: 200,000 تومان حساب از آلمان"
        results = boards.extract_locality_quotes(text, benchmark_value=1_773_250.0)
        by_locality = {
            loc: (midpoint, basis, currency)
            for loc, _buy, _sell, midpoint, _unit, basis, currency in results
        }
        self.assertIn("Germany", by_locality)
        self.assertEqual(by_locality["Germany"][1], "sell")
        self.assertEqual(by_locality["Germany"][2], "EUR")
        self.assertAlmostEqual(by_locality["Germany"][0], 1_754_385.96, places=2)

    def test_extract_locality_quotes_converts_qatar_riyal_signal(self) -> None:
        text = "ریال قطر : 48,750 تومان"
        results = boards.extract_locality_quotes(text, benchmark_value=1_773_250.0)
        by_locality = {loc: (midpoint, currency) for loc, _buy, _sell, midpoint, _unit, _basis, currency in results}
        self.assertIn("Qatar", by_locality)
        self.assertEqual(by_locality["Qatar"][1], "QAR")
        self.assertAlmostEqual(by_locality["Qatar"][0], 1_774_500.0, places=2)

    def test_extract_locality_quotes_converts_small_armenia_dram_signal(self) -> None:
        text = "درام ارمنستان : 455 تومان"
        results = boards.extract_locality_quotes(text, benchmark_value=1_773_250.0)
        by_locality = {loc: (midpoint, currency) for loc, _buy, _sell, midpoint, _unit, _basis, currency in results}
        self.assertIn("Armenia", by_locality)
        self.assertEqual(by_locality["Armenia"][1], "AMD")
        self.assertAlmostEqual(by_locality["Armenia"][0], 1_774_500.0, places=2)

    def test_detect_localities_maps_german_city_aliases(self) -> None:
        hits = boards.detect_localities("مونیخ یورو و کلن یورو برای حواله آلمان")
        self.assertIn("Munich", hits)
        self.assertIn("Cologne", hits)
        self.assertIn("Germany", hits)

    def test_detect_localities_maps_qatar_and_armenia_currency_aliases(self) -> None:
        hits = boards.detect_localities("ریال قطر و درام ارمنستان روی تابلو")
        self.assertIn("Qatar", hits)
        self.assertIn("Armenia", hits)

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
                doha_quote="",
                qatar_quote="",
                yerevan_quote="",
                armenia_quote="",
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
                doha_quote="",
                qatar_quote="",
                yerevan_quote="",
                armenia_quote="",
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
