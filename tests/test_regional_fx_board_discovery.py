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

    def test_extract_locality_quotes_segments_inline_currency_board(self) -> None:
        text = (
            "قیمت ارزها دلار : 1,779,200 ریال یورو : 2,081,900 ریال "
            "پوند انگلیس : 2,403,500 ریال درهم امارات : 484,780 ریال "
            "لیر ترکیه : 39,300 ریال دینار عراق : 1,360 ریال "
            "ریال قطر : 488,000 ریال درام ارمنستان : 4,550 ریال"
        )
        results = boards.extract_locality_quotes(text, benchmark_value=1_773_250.0)
        by_locality = {loc: (midpoint, currency) for loc, _buy, _sell, midpoint, _unit, _basis, currency in results}
        self.assertEqual(by_locality["Dubai"][1], "AED")
        self.assertAlmostEqual(by_locality["Dubai"][0], 1_780_354.55, places=2)
        self.assertEqual(by_locality["Turkey"][1], "TRY")
        self.assertAlmostEqual(by_locality["Turkey"][0], 1_775_574.0, places=2)
        self.assertEqual(by_locality["Iraq"][1], "IQD")
        self.assertAlmostEqual(by_locality["Iraq"][0], 1_781_600.0, places=2)
        self.assertEqual(by_locality["Qatar"][1], "QAR")
        self.assertAlmostEqual(by_locality["Qatar"][0], 1_776_320.0, places=2)
        self.assertEqual(by_locality["Armenia"][1], "AMD")
        self.assertAlmostEqual(by_locality["Armenia"][0], 1_774_500.0, places=2)

    def test_extract_locality_quotes_handles_hundred_iqd_quote(self) -> None:
        text = "صد دینار عراق : 136,000 ریال"
        results = boards.extract_locality_quotes(text, benchmark_value=1_773_250.0)
        by_locality = {loc: (midpoint, currency) for loc, _buy, _sell, midpoint, _unit, _basis, currency in results}
        self.assertEqual(by_locality["Iraq"][1], "IQD")
        self.assertAlmostEqual(by_locality["Iraq"][0], 1_781_600.0, places=2)

    def test_build_navasan_currency_board_records_adds_provider_qatar_signal(self) -> None:
        payload = {
            "usd": {"value": 175_500, "date": 1_777_568_649},
            "aed": {"value": 48_240, "date": 1_777_656_462},
            "try": {"value": 3_885, "date": 1_777_568_649},
            "iqd": {"value": 133.94, "date": 1_777_568_649},
            "qar": {"value": 47_980, "date": 1_777_568_649},
            "amd": {"value": 473.33, "date": 1_777_568_649},
            "eur": {"value": 205_790, "date": 1_777_568_649},
            "gbp": {"value": 238_340, "date": 1_777_568_649},
        }
        records = boards.build_navasan_currency_board_records(payload, benchmark_value=1_773_250.0)
        by_locality = {record.locality_name: record for record in records}
        self.assertEqual(by_locality["Qatar"].quote_currency_guess, "QAR")
        self.assertAlmostEqual(by_locality["Qatar"].normalized_rate_irr, 1_746_472.0, places=2)
        self.assertEqual(by_locality["Turkey"].quote_currency_guess, "TRY")
        self.assertAlmostEqual(by_locality["Turkey"].normalized_rate_irr, 1_755_243.0, places=2)
        self.assertEqual(by_locality["Iraq"].quote_currency_guess, "IQD")
        self.assertAlmostEqual(by_locality["Iraq"].normalized_rate_irr, 1_754_614.0, places=2)

    def test_seed_from_previous_candidates_replays_search_discovered_handles(self) -> None:
        seeded = boards.seed_from_previous_candidates(
            [
                {
                    "handle": "nrxidolar",
                    "public_url": "https://t.me/s/nrxidolar",
                    "source_type": "regional_fx_board",
                    "quote_message_count": "1",
                    "board_message_count": "1",
                },
                {
                    "handle": "navasan_public_currency_board",
                    "public_url": "https://www.navasan.net/",
                    "quote_message_count": "1",
                    "board_message_count": "1",
                },
                {
                    "handle": "emptyboard",
                    "public_url": "https://t.me/s/emptyboard",
                    "quote_message_count": "0",
                    "board_message_count": "0",
                },
            ]
        )
        self.assertIn("nrxidolar", seeded)
        self.assertEqual(seeded["nrxidolar"].public_url, "https://t.me/s/nrxidolar")
        self.assertIn("previous_candidate_registry", seeded["nrxidolar"].discovery_origins)
        self.assertNotIn("navasan_public_currency_board", seeded)
        self.assertNotIn("emptyboard", seeded)

    def test_seed_from_source_registry_replays_active_telegram_sources(self) -> None:
        payload = {
            "sources": [
                {
                    "platform": "telegram",
                    "handle_or_url": "nrxidolar",
                    "public_url": "https://t.me/s/nrxidolar",
                    "source_kind": "regional_fx_board",
                    "signal_families": ["regional_fx_board"],
                    "last_success_at": "2026-05-01T12:00:00Z",
                },
                {
                    "platform": "telegram",
                    "handle_or_url": "inactive",
                    "signal_families": ["regional_fx_board"],
                },
            ]
        }
        seeded = boards.seed_from_source_registry(payload)
        self.assertIn("nrxidolar", seeded)
        self.assertIn("source_registry", seeded["nrxidolar"].discovery_origins)
        self.assertNotIn("inactive", seeded)

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
