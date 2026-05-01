import unittest

from scripts import telegram_direct_shop_expansion as direct


class TelegramDirectShopExpansionTests(unittest.TestCase):
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
