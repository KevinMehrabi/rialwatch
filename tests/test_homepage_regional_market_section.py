"""Server-rendered content contracts; interactive coverage is in verify/redesign.cjs."""
import re
import tempfile
import unittest
from pathlib import Path

from scripts import pipeline

ROOT = Path(__file__).resolve().parents[1]


class HomepageRegionalMarketSectionTests(unittest.TestCase):
    def test_homepage_keeps_all_live_indicator_and_reading_bindings(self) -> None:
        # Distinct sentinels detect accidentally dropped/reused values during layout changes.
        template = pipeline.read_template(ROOT / "templates", "index.html")
        keys = set(re.findall(r"\$\{([a-z_0-9]+)\}", template.template))
        values = {key: f"live-{key}" for key in keys}
        rendered = template.substitute(**values)
        for market in ("official", "transfer", "crypto", "gold"):
            for suffix in ("value", "note", "sparkline"):
                self.assertIn(f"live-street_{market}_gap_{suffix}", rendered)
        for market in ("official", "transfer"):
            for suffix in ("value", "note", "sparkline"):
                self.assertIn(f"live-street_{market}_spread_{suffix}", rendered)
        for market in ("official", "regional_transfer", "crypto_usdt", "emami_gold_coin"):
            for suffix in ("value_main", "as_of", "sparkline"):
                self.assertIn(f"live-{market}_{suffix}", rendered)
        self.assertIn("Regional Market Signals", rendered)
        self.assertIn("Diagnostic only.", rendered)
        self.assertIn("Recent Daily References", rendered)

    def test_withheld_primary_keeps_real_publication_state_and_reason(self) -> None:
        latest = {
            "date": "2026-02-05", "as_of": "2026-02-05T14:20:00Z", "sources": {},
            "computed": {"fix": None, "status": "WITHHOLD", "withheld": True,
                         "band": {}, "withhold_reasons": ["No valid sources"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            pipeline.publish_home(site, ROOT / "templates", latest["as_of"], latest)
            rendered = (site / "index.html").read_text()
        self.assertIn('data-publication="Withheld"', rendered)
        self.assertIn('publication-chip Withheld', rendered)
        self.assertIn("Insufficient valid sources", rendered)
        self.assertNotIn('publication-chip Published', rendered)
        self.assertNotRegex(rendered, r"\$\{[a-z_]+\}")

    def test_shared_shell_preserves_supporting_page_body(self) -> None:
        for page in ("methodology.html", "governance.html"):
            source = (ROOT / "templates" / page).read_text()
            rendered = pipeline.render_page(ROOT / "templates", page, "Test", "2026-02-05T14:20:00Z")
            self.assertIn(source, rendered)
            self.assertIn('href="/methodology/"', rendered)
            self.assertIn('href="/status/"', rendered)
            self.assertIn("Site updated: 2026-02-05T14:20:00Z", rendered)

    def test_shell_uses_existing_local_assets_without_chart_dependencies(self) -> None:
        shell = (ROOT / "templates/layout.html").read_text()
        for asset in re.findall(r'(?:src|href)="(/assets/[^\"]+)"', shell):
            self.assertTrue((ROOT / asset.split('?', 1)[0].lstrip('/')).is_file(), asset)
        self.assertNotIn('cdn.jsdelivr.net', shell)
        self.assertNotIn('fonts.googleapis.com', shell)
        self.assertNotIn('chartjs-plugin-zoom', shell)

    def test_reading_timestamps_preserve_quote_and_collection_dates(self) -> None:
        latest = {
            "date": "2026-02-05", "as_of": "2026-02-05T14:20:00Z",
            "computed": {"fix": 1_400_000, "band": {"p25": 1_390_000, "p75": 1_410_000},
                         "status": "Green", "withheld": False},
            "benchmarks": {
                "official": {"fix": 1_000_000, "available": True,
                             "selected_quote_time": "2026-02-03T08:01:12Z"},
                "crypto_usdt": {"fix": 1_390_000, "available": True},
            },
            "sources": {"test": {"samples": [{
                "sampled_at": "2026-02-04T22:45:00Z",
                "benchmarks": {"crypto_usdt": 1_390_000},
            }]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            pipeline.publish_home(site, ROOT / "templates", "2026-02-10T10:00:00Z", latest)
            rendered = (site / "index.html").read_text()
        self.assertIn("Quote observed Feb 3, 2026 · 08:01 UTC", rendered)
        self.assertIn("Collected Feb 4, 2026 · 22:45 UTC", rendered)
        self.assertIn("Site updated: 2026-02-10T10:00:00Z", rendered)


if __name__ == "__main__":
    unittest.main()
