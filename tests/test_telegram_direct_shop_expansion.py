import unittest

from scripts import telegram_direct_shop_expansion as direct


class TelegramDirectShopExpansionTests(unittest.TestCase):
    def test_search_urls_include_brave_and_duckduckgo_surfaces(self) -> None:
        urls = direct.search_urls_for_query("site:t.me/s صرافی تهران دلار", pages=1)
        self.assertEqual(len(urls), 2)
        self.assertIn("lite.duckduckgo.com", urls[0])
        self.assertIn("search.brave.com", urls[1])

    def test_quote_active_tehran_shop_is_likely_direct_shop(self) -> None:
        self.assertTrue(
            direct.is_likely_direct_shop(
                has_phone=False,
                has_address=False,
                has_map=False,
                has_shop_name=True,
                city_guess="Tehran",
                quote_posts=4,
                pair_posts=0,
                parseability_score=48,
                commentary_heavy=False,
            )
        )

    def test_commentary_heavy_channel_is_not_likely_direct_shop(self) -> None:
        self.assertFalse(
            direct.is_likely_direct_shop(
                has_phone=True,
                has_address=True,
                has_map=False,
                has_shop_name=True,
                city_guess="Tehran",
                quote_posts=6,
                pair_posts=3,
                parseability_score=80,
                commentary_heavy=True,
            )
        )

    def test_contactable_shop_identity_is_likely_even_before_quotes(self) -> None:
        self.assertTrue(
            direct.is_likely_direct_shop(
                has_phone=True,
                has_address=False,
                has_map=False,
                has_shop_name=True,
                city_guess="unknown",
                quote_posts=0,
                pair_posts=0,
                parseability_score=18,
                commentary_heavy=False,
            )
        )

    def test_registry_updates_mark_inside_iran_shop(self) -> None:
        row = direct.ChannelScore(
            handle="sarafitehran",
            public_url="https://t.me/s/sarafitehran",
            title="Sarafi Tehran",
            city_guess="Tehran",
            quote_post_count=7,
            buy_sell_pair_count=3,
            parseability_score=81,
            likely_individual_shop=True,
            has_phone=True,
            has_address=True,
            has_map_link=False,
            has_shop_name=True,
            status="ok",
            commentary_heavy=False,
            last_seen_text_sample="buy sell dollar",
            discovery_queries=["site:t.me/s صرافی تهران دلار"],
        )
        updates = direct.registry_updates_from_scores([row])
        self.assertEqual(updates[0]["country_guess"], "Iran")
        self.assertEqual(updates[0]["source_kind"], "exchange_shop")
        self.assertEqual(updates[0]["usable_record_count"], 7)
        self.assertIn("direct_shop_expansion", updates[0]["signal_families"])


if __name__ == "__main__":
    unittest.main()
