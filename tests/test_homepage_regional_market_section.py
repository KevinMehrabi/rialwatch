import unittest
from pathlib import Path


class HomepageRegionalMarketSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        template_path = Path(__file__).resolve().parents[1] / "templates" / "index.html"
        cls.template = template_path.read_text(encoding="utf-8")

    def test_section_title_and_subtitle_are_updated(self) -> None:
        self.assertIn("Regional Market Signals", self.template)
        self.assertIn(
            "Best available public locality signals relative to the RialWatch benchmark. Diagnostic only.",
            self.template,
        )
        self.assertNotIn("Exchange Shops</div>", self.template)

    def test_data_source_wired_to_merged_regional_signal_payload(self) -> None:
        self.assertIn("api/regional_market_signals_card.json", self.template)
        self.assertNotIn("api/exchange_shop_baskets_card.json", self.template)

    def test_display_state_filter_and_monitor_copy_present(self) -> None:
        self.assertIn("card.display_state !== 'hide'", self.template)
        self.assertIn("Monitoring only. Reason:", self.template)
        self.assertIn("${card.basket_name}${card.display_state === 'monitor' ? ' (monitoring)' : ''}", self.template)

    def test_alignment_and_high_dispersion_presentation(self) -> None:
        self.assertIn("alignmentBadgeClass(card.alignment_label)", self.template)
        self.assertIn("High dispersion &mdash; indicative signal only", self.template)
        self.assertIn("panel-value-soft", self.template)

    def test_cards_render_in_payload_order(self) -> None:
        self.assertIn("holder.innerHTML = cards.map(card => `", self.template)
        self.assertNotIn("sortedCards", self.template)


if __name__ == "__main__":
    unittest.main()
