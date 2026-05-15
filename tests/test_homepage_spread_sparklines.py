import tempfile
import unittest
from pathlib import Path

from scripts import pipeline


class HomepageSpreadSparklineTests(unittest.TestCase):
    def test_negative_prior_day_change_renders_in_primary_card(self) -> None:
        templates_dir = Path(__file__).resolve().parents[1] / "templates"

        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            pipeline.write_json(
                site_dir / "api" / "series.json",
                {
                    "rows": [
                        {
                            "date": "2026-05-14",
                            "fix": 1_807_000.0,
                            "p25": 1_804_725.0,
                            "p75": 1_809_000.0,
                            "status": "Green",
                            "withheld": False,
                        }
                    ]
                },
            )

            latest = {
                "date": "2026-05-15",
                "as_of": "2026-05-15T18:01:26Z",
                "sources": {},
                "benchmarks": {
                    "open_market": {"fix": 1_805_000.0, "available": True},
                    "official": {"fix": 1_479_835.0, "available": True},
                    "regional_transfer": {"fix": 1_834_800.0, "available": True},
                    "crypto_usdt": {"fix": 1_799_905.0, "available": True},
                    "emami_gold_coin": {"fix": 2_794_326.0, "available": True},
                },
                "computed": {
                    "fix": 1_805_000.0,
                    "band": {"p25": 1_804_500.0, "p75": 1_811_500.0},
                    "status": "Green",
                    "withheld": False,
                    "withhold_reasons": [],
                    "source_medians": {"test": 1_805_000.0},
                },
            }

            pipeline.publish_home(
                site_dir=site_dir,
                templates_dir=templates_dir,
                generated_at="2026-05-15T18:01:26Z",
                latest=latest,
            )

            html = (site_dir / "index.html").read_text(encoding="utf-8")
            start = html.index("Prior Day")
            end = html.index("Observed", start)
            prior_day_block = html[start:end]

            self.assertIn("-2,000", prior_day_block)
            self.assertIn("(-0.1%)", prior_day_block)
            self.assertNotIn("History building", prior_day_block)

    def test_negative_street_transfer_spread_still_renders_history(self) -> None:
        templates_dir = Path(__file__).resolve().parents[1] / "templates"

        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp)
            pipeline.write_json(
                site_dir / "api" / "benchmarks" / "open_market.json",
                [
                    {"date": "2026-04-25", "value": 1_556_000.0},
                    {"date": "2026-04-26", "value": 1_572_750.0},
                ],
            )
            pipeline.write_json(
                site_dir / "api" / "benchmarks" / "regional_transfer.json",
                [
                    {"date": "2026-04-25", "value": 1_576_700.0},
                    {"date": "2026-04-26", "value": 1_589_400.0},
                ],
            )

            latest = {
                "date": "2026-04-27",
                "as_of": "2026-04-27T15:41:00Z",
                "sources": {},
                "benchmarks": {
                    "open_market": {"fix": 1_593_250.0, "available": True},
                    "official": {"fix": 1_362_635.0, "available": True},
                    "regional_transfer": {"fix": 1_613_600.0, "available": True},
                    "crypto_usdt": {"fix": 1_579_890.0, "available": True},
                    "emami_gold_coin": {"fix": 905_905_000.0, "available": True},
                },
                "computed": {
                    "fix": 1_593_250.0,
                    "band": {"p25": 1_591_375.0, "p75": 1_595_125.0},
                    "status": "Green",
                    "withheld": False,
                    "withhold_reasons": [],
                    "source_medians": {"test": 1_593_250.0},
                },
            }

            pipeline.publish_home(
                site_dir=site_dir,
                templates_dir=templates_dir,
                generated_at="2026-04-27T15:41:00Z",
                latest=latest,
            )

            html = (site_dir / "index.html").read_text(encoding="utf-8")
            start = html.index("Street-Transfer Spread")
            end = html.index("Additional Rial Readings", start)
            transfer_spread_card = html[start:end]

            self.assertIn("-20,350</div>", transfer_spread_card)
            self.assertIn('<div class="unit-label">IRR per USD</div>', transfer_spread_card)
            self.assertNotIn("-20,350 IRR", transfer_spread_card)
            self.assertIn('aria-label="spread trend sparkline"', transfer_spread_card)
            self.assertNotIn("History building", transfer_spread_card)


if __name__ == "__main__":
    unittest.main()
