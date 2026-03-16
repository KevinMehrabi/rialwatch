import unittest

from scripts import uae_exchange_discovery as discovery


class UAEExchangeDiscoveryTests(unittest.TestCase):
    def test_extract_links_reads_markdown_and_uddg_targets(self) -> None:
        page = """
        1.[معتبرترین صرافی ایرانی امارات](http://duckduckgo.com/l/?uddg=https%3A%2F%2Fsarafyhafez.com%2Fdubai-exchange%2F&rut=abc)
        """
        links = discovery.extract_links(page)
        self.assertIn("http://duckduckgo.com/l/?uddg=https%3A%2F%2Fsarafyhafez.com%2Fdubai-exchange%2F&rut=abc", links)
        self.assertIn("https://sarafyhafez.com/dubai-exchange/", links)

    def test_classify_source_type_prefers_exchange_and_remittance(self) -> None:
        text = "صرافی ایرانی دبی حواله ایران دبی تماس واتساپ نرخ دلار امروز"
        self.assertEqual(
            discovery.classify_source_type(
                text,
                has_whatsapp=True,
                has_telegram=False,
                has_instagram=False,
                rate_posts=2,
                iran_transfer_hint=True,
            ),
            "exchange_shop",
        )

    def test_detect_quote_signals_finds_usd_and_aed(self) -> None:
        blocks = [
            "دلار 1,420,000 خرید 1,430,000 فروش",
            "درهم 385,000 / 387,000",
            "تماس واتساپ برای حواله ایران دبی",
        ]
        quote_posts, pair_count, usd_detected, aed_detected = discovery.detect_quote_signals(blocks)
        self.assertEqual(quote_posts, 2)
        self.assertGreaterEqual(pair_count, 2)
        self.assertTrue(usd_detected)
        self.assertTrue(aed_detected)

    def test_compute_candidate_score_rewards_public_signal_surfaces(self) -> None:
        score = discovery.compute_candidate_score(
            source_type="settlement_exchange",
            has_website=True,
            has_instagram=True,
            has_telegram=False,
            has_whatsapp=True,
            iran_transfer_hint=True,
            rate_page_detected=True,
            rate_post_detected=True,
            usd_quote_detected=True,
            aed_quote_detected=True,
            parseability_score=58,
        )
        self.assertGreaterEqual(score, 90)

    def test_is_business_like_candidate_rejects_news_noise(self) -> None:
        record = discovery.UAECandidate(
            business_name="WATCH: Iran strikes Dubai financial hub",
            website="https://news24online.com/world/story",
            instagram="",
            telegram="",
            whatsapp_link="",
            city_or_district="Dubai",
            country="UAE",
            source_type_guess="remittance_exchange",
            rate_page_detected=True,
            rate_post_detected=True,
            iran_transfer_hint=True,
            usd_irr_quote_detected=False,
            aed_irr_quote_detected=False,
            quote_post_count=1,
            parseability_score=42,
            last_seen="",
            candidate_score=100,
            discovery_origins="english",
            status_guess="ok",
        )
        self.assertFalse(discovery.is_business_like_candidate(record))


if __name__ == "__main__":
    unittest.main()
