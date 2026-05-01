import tempfile
import unittest
from pathlib import Path

from scripts import regional_source_registry as registry


class RegionalSourceRegistryTests(unittest.TestCase):
    def test_upsert_sources_preserves_best_metrics_and_origins(self) -> None:
        payload = registry.empty_registry()
        payload = registry.upsert_sources(
            payload,
            [
                {
                    "platform": "telegram",
                    "handle_or_url": "NrxiDolar",
                    "public_url": "https://t.me/s/nrxidolar",
                    "title": "Nrxi Dolar",
                    "country_guess": "Iran",
                    "city_guess": "Tehran",
                    "source_kind": "regional_fx_board",
                    "signal_families": ["regional_fx_board"],
                    "origins": ["search"],
                    "quote_message_count": 2,
                    "usable_record_count": 1,
                    "parseability_score": 72,
                }
            ],
            observed_at="2026-05-01T12:00:00Z",
        )
        payload = registry.upsert_sources(
            payload,
            [
                {
                    "platform": "telegram",
                    "handle_or_url": "https://t.me/s/nrxidolar",
                    "country_guess": "Iran",
                    "signal_families": ["regional_market_signal"],
                    "origins": ["source_registry"],
                    "quote_message_count": 1,
                    "usable_record_count": 0,
                    "parseability_score": 55,
                }
            ],
            observed_at="2026-05-01T13:00:00Z",
        )
        self.assertEqual(len(payload["sources"]), 1)
        row = payload["sources"][0]
        self.assertEqual(row["key"], "telegram:nrxidolar")
        self.assertEqual(row["handle_or_url"], "nrxidolar")
        self.assertEqual(row["best_quote_message_count"], 2)
        self.assertEqual(row["best_usable_record_count"], 1)
        self.assertEqual(row["last_quote_message_count"], 1)
        self.assertIn("regional_fx_board", row["signal_families"])
        self.assertIn("regional_market_signal", row["signal_families"])
        self.assertIn("search", row["origins"])
        self.assertIn("source_registry", row["origins"])
        self.assertEqual(payload["summary"]["active_iran_sources"], 1)

    def test_registry_round_trip_and_source_filter(self) -> None:
        payload = registry.upsert_sources(
            registry.empty_registry(),
            [
                {
                    "platform": "telegram",
                    "handle_or_url": "sarafitehran",
                    "country_guess": "Iran",
                    "signal_families": ["direct_shop_expansion"],
                    "quote_message_count": 4,
                },
                {
                    "platform": "provider",
                    "handle_or_url": "navasan_public_currency_board",
                    "country_guess": "Iran",
                    "signal_families": ["regional_fx_board"],
                    "quote_message_count": 4,
                },
            ],
            observed_at="2026-05-01T12:00:00Z",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / registry.REGISTRY_FILENAME
            registry.write_registry(path, payload)
            loaded = registry.load_registry(path)
        rows = registry.registry_sources(
            loaded,
            platform="telegram",
            signal_families={"direct_shop_expansion"},
            active_only=True,
        )
        self.assertEqual([row["handle_or_url"] for row in rows], ["sarafitehran"])


if __name__ == "__main__":
    unittest.main()
