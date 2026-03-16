import unittest

from scripts import exchange_shop_business_discovery as discovery


class ExchangeShopBusinessDiscoveryTests(unittest.TestCase):
    def test_normalize_instagram_url_keeps_profiles_only(self) -> None:
        self.assertEqual(
            discovery.normalize_instagram_url("https://www.instagram.com/iran_exchange_dubai/"),
            "https://www.instagram.com/iran_exchange_dubai/",
        )
        self.assertIsNone(discovery.normalize_instagram_url("https://www.instagram.com/p/abc123/"))

    def test_detect_source_type_prefers_exchange_shop(self) -> None:
        source_type, likely_shop = discovery.detect_source_type(
            "صرافی ایرانیان دبی نرخ دلار و یورو تماس با ما",
            country="UAE",
            quote_posts=3,
            has_phone=True,
            has_address=False,
        )
        self.assertEqual(source_type, "exchange_shop")
        self.assertTrue(likely_shop)

    def test_merge_records_combines_same_business_name(self) -> None:
        merged = discovery.merge_records(
            [
                discovery.BusinessRecord(
                    business_name="Iran Exchange Dubai",
                    website="https://iranexchange.example",
                    telegram="",
                    instagram="https://www.instagram.com/iranexchange/",
                    country="UAE",
                    city="Dubai",
                    likely_exchange_shop=True,
                    rate_page_detected=False,
                    last_seen_rate="",
                    candidate_score=62,
                    source_type="exchange_shop",
                    sources_with_social_feeds=True,
                    discovery_origins="general",
                    status_guess="ok",
                ),
                discovery.BusinessRecord(
                    business_name="Iran Exchange Dubai LLC",
                    website="",
                    telegram="https://t.me/s/iranexchange_dubai",
                    instagram="",
                    country="UAE",
                    city="Dubai",
                    likely_exchange_shop=True,
                    rate_page_detected=True,
                    last_seen_rate="2026-03-15T12:00:00Z",
                    candidate_score=78,
                    source_type="exchange_shop",
                    sources_with_social_feeds=True,
                    discovery_origins="instagram",
                    status_guess="ok",
                ),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].website, "https://iranexchange.example")
        self.assertEqual(merged[0].telegram, "https://t.me/s/iranexchange_dubai")
        self.assertTrue(merged[0].rate_page_detected)


if __name__ == "__main__":
    unittest.main()
