import unittest
import json
import tempfile
from pathlib import Path

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

    def test_seed_candidates_from_quote_samples_adds_iraq_hint_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            quote_samples_path = Path(tmp_dir) / "quote_message_samples.json"
            quote_samples_path.write_text(
                json.dumps(
                    [
                        {
                            "handle": "live_rate_ir",
                            "title": "Live Rate",
                            "public_url": "https://t.me/s/live_rate_ir",
                            "channel_type_guess": "market_price_channel",
                            "quote_message_records": [
                                {"message_text_sample": "سلیمانیه 154,500"},
                                {"message_text_sample": "Iraq 154800"},
                            ],
                        },
                        {
                            "handle": "not_iraq",
                            "title": "No Iraq",
                            "public_url": "https://t.me/s/not_iraq",
                            "channel_type_guess": "market_price_channel",
                            "quote_message_records": [
                                {"message_text_sample": "تهران 144,000"},
                            ],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            channel_rows = [
                {"handle": "live_rate_ir", "public_url": "https://t.me/s/live_rate_ir", "parseable_score": "71"},
                {"handle": "not_iraq", "public_url": "https://t.me/s/not_iraq", "parseable_score": "50"},
            ]
            seeded = enrichment.seed_candidates_from_quote_samples(
                quote_samples_path=quote_samples_path,
                channel_rows=channel_rows,
                existing_handles=[],
            )
            handles = {row.handle for row in seeded}
            self.assertIn("live_rate_ir", handles)
            self.assertNotIn("not_iraq", handles)
            seeded_row = next(row for row in seeded if row.handle == "live_rate_ir")
            self.assertEqual(seeded_row.source_type, "regional_market_channel")
            self.assertGreaterEqual(seeded_row.quote_density_score, 59.0)


if __name__ == "__main__":
    unittest.main()
