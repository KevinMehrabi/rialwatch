import unittest
from pathlib import Path


class HomepageRegionalMarketSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        template_path = Path(__file__).resolve().parents[1] / "templates" / "index.html"
        cls.template = template_path.read_text(encoding="utf-8")
        layout_path = Path(__file__).resolve().parents[1] / "templates" / "layout.html"
        cls.layout = layout_path.read_text(encoding="utf-8")

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
        self.assertIn("formatIrrDisplayValue(card.rate_text || card.weighted_rate)", self.template)
        self.assertIn("High dispersion &mdash; indicative signal only", self.template)
        self.assertIn("panel-value-soft", self.template)
        self.assertNotIn("toLocaleString() + ' IRR'", self.template)

    def test_cards_render_in_payload_order(self) -> None:
        self.assertIn("holder.innerHTML = cards.map(card => `", self.template)
        self.assertNotIn("sortedCards", self.template)

    def test_latest_point_pulse_uses_overlay_not_chart_padding(self) -> None:
        self.assertIn('class="history-chart-wrap"', self.template)
        self.assertIn('id="historyChartPulse"', self.template)
        self.assertIn('class="history-point-marker"', self.template)
        self.assertIn("afterDraw(chart, _args, pluginOptions)", self.template)
        self.assertIn("point.getProps(['x', 'y'], false)", self.template)
        self.assertIn("chart.canvas.offsetLeft + coords.x", self.template)
        self.assertIn("pointRadius: 0", self.template)
        self.assertIn("pointHoverRadius: 0", self.template)
        self.assertNotIn("point.getProps(['x', 'y'], true)", self.template)
        self.assertNotIn("latestPointPulseCanvasPadding", self.template)
        self.assertNotIn("clip: { left: 0, top: 0", self.template)
        self.assertIn(".history-chart-wrap", self.layout)
        self.assertIn("overflow: visible", self.layout)
        self.assertIn(".history-point-pulse", self.layout)
        self.assertIn(".history-point-marker", self.layout)

    def test_primary_rate_value_uses_card_responsive_sizing(self) -> None:
        self.assertIn("container-type: inline-size", self.layout)
        self.assertIn("flex-wrap: wrap", self.layout)
        self.assertIn("font-size: clamp(2.15rem, 17cqw, 3.6rem)", self.layout)
        self.assertIn("font-size: clamp(1.9rem, 14cqw, 2.65rem)", self.layout)
        self.assertIn("font-size: clamp(1.35rem, 7cqw, 2rem)", self.layout)
        self.assertNotIn("4.5vw", self.layout)
        self.assertNotIn("4.1vw", self.layout)
        self.assertNotIn("3.4vw", self.layout)
        self.assertNotIn("5.2vw", self.layout)

    def test_history_chart_fills_responsive_wrapper(self) -> None:
        self.assertIn(".history-chart-wrap", self.layout)
        self.assertIn("height: clamp(250px, 31cqw, 360px)", self.layout)
        self.assertIn("width: 100%", self.layout)
        self.assertIn("height: 100%", self.layout)
        self.assertIn("maintainAspectRatio: false", self.template)
        self.assertNotIn('height="320"', self.template)
        self.assertNotIn("height: auto !important", self.layout)
        self.assertNotIn("width: 100% !important", self.layout)
        self.assertNotIn("height: 100% !important", self.layout)
        self.assertNotIn("aspectRatio: chartAspectRatio", self.template)

    def test_history_chart_refreshes_after_viewport_resizes(self) -> None:
        self.assertIn("let historyChartResizeFrame = null", self.template)
        self.assertIn("let historyChartLastViewportMode = null", self.template)
        self.assertIn("const historyChartCompactQuery = window.matchMedia('(max-width: 700px)')", self.template)
        self.assertIn("function scheduleHistoryChartLayoutRefresh", self.template)
        self.assertIn("function resizeHistoryChartToWrapper", self.template)
        self.assertIn("getBoundingClientRect()", self.template)
        self.assertIn("historyChart.resize(Math.round(rect.width), Math.round(rect.height))", self.template)
        self.assertIn("ResizeObserver", self.template)
        self.assertIn("historyChart.resize()", self.template)
        self.assertIn("historyChart.update('none')", self.template)
        self.assertIn("rebuildOnModeChange", self.template)
        self.assertIn("bindHistoryChartResizeHandling()", self.template)

    def test_homepage_grids_use_stable_responsive_tracks(self) -> None:
        self.assertIn(
            "grid-template-columns: minmax(280px, 360px) minmax(0, 1fr)",
            self.layout,
        )
        self.assertIn(
            "grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
            self.layout,
        )
        self.assertIn(
            "grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr))",
            self.layout,
        )
        self.assertIn("@media (max-width: 900px)", self.layout)
        self.assertIn(".top-grid .primary-card", self.layout)
        self.assertIn("order: 1", self.layout)
        self.assertIn(".top-grid .chart-card", self.layout)
        self.assertIn("order: 2", self.layout)
        self.assertNotIn(".indicators-grid {\n      grid-template-columns: repeat(2", self.layout)

        self.assertLess(
            self.layout.index("justify-content: space-between"),
            self.layout.index("@media (max-width: 900px)"),
        )
        self.assertLess(
            self.layout.index("@media (max-width: 900px)"),
            self.layout.index("justify-content: flex-start"),
        )


if __name__ == "__main__":
    unittest.main()
