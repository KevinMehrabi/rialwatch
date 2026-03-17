import unittest

from scripts import iraq_signal_enrichment as enrichment


class IraqSignalEnrichmentTests(unittest.TestCase):
    def test_single_value_persian_variant(self) -> None:
        parsed = enrichment.parse_iraq_signal(
            text="سلیمانیه 147800",
            timestamp_iso="2026-03-16T10:00:00+00:00",
            source_type="regional_market_channel",
            benchmark_value=1_449_922.07,
            quote_density_score=80.0,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[2], "single")
        self.assertAlmostEqual(parsed[1], 1_478_000.0)

    def test_single_value_english_variant(self) -> None:
        parsed = enrichment.parse_iraq_signal(
            text="Iraq 147800",
            timestamp_iso="2026-03-16T10:00:00+00:00",
            source_type="regional_market_channel",
            benchmark_value=1_449_922.07,
            quote_density_score=80.0,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[2], "single")
        self.assertAlmostEqual(parsed[1], 1_478_000.0)

    def test_relative_delta_with_tehran_reference(self) -> None:
        parsed = enrichment.parse_iraq_signal(
            text="تهران 147300 | سلیمانیه +2000",
            timestamp_iso="2026-03-16T10:00:00+00:00",
            source_type="regional_fx_board",
            benchmark_value=1_449_922.07,
            quote_density_score=90.0,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[2], "relative")
        self.assertAlmostEqual(parsed[1], 1_493_000.0)
        self.assertEqual(parsed[5], "147300.00")
        self.assertEqual(parsed[6], "2000")

    def test_next_line_value(self) -> None:
        parsed = enrichment.parse_iraq_signal(
            text="سليمانية:\n147900",
            timestamp_iso="2026-03-16T10:00:00+00:00",
            source_type="regional_market_channel",
            benchmark_value=1_449_922.07,
            quote_density_score=60.0,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[2], "single")
        self.assertAlmostEqual(parsed[1], 1_479_000.0)

    def test_summary_publish_with_multiple_sources(self) -> None:
        rows = [
            enrichment.IraqSignalRecord(
                source="a",
                handle="a",
                title="A",
                source_type="regional_market_channel",
                message_text_sample="",
                inferred_value=154_400.0,
                normalized_irr_value=1_544_000.0,
                extraction_type="single",
                freshness="fresh",
                parseability_score=76,
                timestamp_iso="2026-03-16T09:00:00+00:00",
                inferred_unit="toman",
                tehran_reference="",
                delta_value="",
                quote_density_score=85.0,
            ),
            enrichment.IraqSignalRecord(
                source="a",
                handle="a",
                title="A",
                source_type="regional_market_channel",
                message_text_sample="",
                inferred_value=154_600.0,
                normalized_irr_value=1_546_000.0,
                extraction_type="single",
                freshness="fresh",
                parseability_score=76,
                timestamp_iso="2026-03-16T10:00:00+00:00",
                inferred_unit="toman",
                tehran_reference="",
                delta_value="",
                quote_density_score=85.0,
            ),
            enrichment.IraqSignalRecord(
                source="b",
                handle="b",
                title="B",
                source_type="regional_fx_board",
                message_text_sample="",
                inferred_value=154_500.0,
                normalized_irr_value=1_545_000.0,
                extraction_type="single",
                freshness="fresh",
                parseability_score=82,
                timestamp_iso="2026-03-16T11:00:00+00:00",
                inferred_unit="toman",
                tehran_reference="",
                delta_value="",
                quote_density_score=90.0,
            ),
            enrichment.IraqSignalRecord(
                source="b",
                handle="b",
                title="B",
                source_type="regional_fx_board",
                message_text_sample="",
                inferred_value=154_700.0,
                normalized_irr_value=1_547_000.0,
                extraction_type="single",
                freshness="fresh",
                parseability_score=82,
                timestamp_iso="2026-03-16T12:00:00+00:00",
                inferred_unit="toman",
                tehran_reference="",
                delta_value="",
                quote_density_score=90.0,
            ),
        ]
        summary = enrichment.summarize_iraq(rows, benchmark_value=1_449_922.07)
        self.assertEqual(summary["recommended_display_state"], "publish")
        self.assertGreaterEqual(summary["contributing_source_count"], 2)
        self.assertEqual(summary["usable_record_count"], 4)


if __name__ == "__main__":
    unittest.main()
