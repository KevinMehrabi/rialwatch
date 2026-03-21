import tempfile
import unittest
from pathlib import Path

from scripts import pipeline


class StatusDiagnosticsSignalCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.templates_dir = cls.repo_root / "templates"
        cls.template_text = (cls.templates_dir / "status.html").read_text(encoding="utf-8")

    def test_status_template_has_diagnostics_signal_section(self) -> None:
        self.assertIn("Diagnostics Signal Coverage", self.template_text)
        self.assertIn("$diagnostics_source_note", self.template_text)
        self.assertIn("$diagnostics_source_rows", self.template_text)

    def test_publish_status_renders_diagnostics_signal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            site_dir = Path(tmp_dir) / "site"
            api_dir = site_dir / "api"
            api_dir.mkdir(parents=True, exist_ok=True)
            (api_dir / "regional_market_signals_card.json").write_text(
                """
                {
                  "cards": [
                    {
                      "basket_name": "UAE",
                      "signal_label": "Dubai settlement signal",
                      "display_state": "publish",
                      "contributing_source_count": 3,
                      "usable_record_count": 12,
                      "suppression_reason": ""
                    },
                    {
                      "basket_name": "Iraq",
                      "signal_label": "Sulaymaniyah market (monitoring)",
                      "display_state": "monitor",
                      "contributing_source_count": 1,
                      "usable_record_count": 6,
                      "suppression_reason": "limited_coverage"
                    },
                    {
                      "basket_name": "Germany",
                      "signal_label": "Exchange network signal",
                      "display_state": "hide",
                      "contributing_source_count": 0,
                      "usable_record_count": 0,
                      "suppression_reason": "no_usable_records"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            pipeline.publish_status(
                site_dir=site_dir,
                templates_dir=self.templates_dir,
                generated_at="2026-03-21T17:36:00Z",
                status_title="OK",
                status_detail="Published daily benchmark.",
                latest={"computed": {}, "sources": {}},
            )

            rendered = (site_dir / "status" / "index.html").read_text(encoding="utf-8")
            self.assertIn("Diagnostics Signal Coverage", rendered)
            self.assertIn("2/3 locality signals active. 1 publish, 1 monitor.", rendered)
            self.assertIn("UAE — Dubai settlement signal", rendered)
            self.assertIn("Iraq — Sulaymaniyah market (monitoring)", rendered)
            self.assertIn("Germany — Exchange network signal", rendered)
            self.assertIn("Diagnostics signal available.", rendered)
            self.assertIn("Monitoring: limited coverage.", rendered)
            self.assertIn("Hidden: no usable records.", rendered)

    def test_publish_status_does_not_show_historical_withheld_day_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            site_dir = Path(tmp_dir) / "site"
            api_dir = site_dir / "api"
            fix_dir = site_dir / "fix"
            api_dir.mkdir(parents=True, exist_ok=True)
            fix_dir.mkdir(parents=True, exist_ok=True)

            (api_dir / "regional_market_signals_card.json").write_text('{"cards": []}', encoding="utf-8")
            (fix_dir / "2026-03-01.json").write_text(
                '{"computed":{"withheld":true}}',
                encoding="utf-8",
            )

            pipeline.publish_status(
                site_dir=site_dir,
                templates_dir=self.templates_dir,
                generated_at="2026-03-21T17:36:00Z",
                status_title="OK",
                status_detail="Published daily benchmark.",
                latest={"computed": {}, "sources": {}},
            )

            rendered = (site_dir / "status" / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("historical day(s) remain withheld", rendered)
            self.assertNotIn("publication-quality rules", rendered)


if __name__ == "__main__":
    unittest.main()
