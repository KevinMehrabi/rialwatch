import unittest

from scripts import uae_signal_watchlist as watchlist


class UAESignalWatchlistTests(unittest.TestCase):
    def test_build_watchlist_only_includes_stale_rows(self) -> None:
        payload = watchlist.build_watchlist(
            review_rows=[
                {
                    "business_name": "A",
                    "basket_use_status": "stale",
                    "freshness_status": "stale",
                },
                {
                    "business_name": "B",
                    "basket_use_status": "no_usable_signal",
                    "freshness_status": "unknown",
                },
            ],
            candidate_rows=[
                {"business_name": "A", "website": "https://a.test", "last_seen": "2026-02-01T00:00:00Z"},
                {"business_name": "B", "website": "https://b.test", "last_seen": "2026-02-02T00:00:00Z"},
            ],
            record_rows=[
                {
                    "business_name": "A",
                    "currency": "AED",
                    "surface_type": "website",
                    "remittance_quote_detected": "true",
                    "timestamp_iso": "2026-02-03T00:00:00Z",
                }
            ],
        )
        self.assertEqual(payload["status"], "watchlist_only")
        self.assertEqual(len(payload["sources"]), 1)
        self.assertEqual(payload["sources"][0]["source_name"], "A")
