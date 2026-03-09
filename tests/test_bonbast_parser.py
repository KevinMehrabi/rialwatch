import json
import os
import unittest
from pathlib import Path

from scripts.bonbast_probe import parse_bonbast_html, probe_bonbast


class BonbastParserTests(unittest.TestCase):
    def test_parser_fixture_extracts_usd_sell_and_normalizes_to_rial(self) -> None:
        fixture_path = Path("tests/fixtures/bonbast_fixture_usd.html")
        html = fixture_path.read_text(encoding="utf-8")
        parsed = parse_bonbast_html(html)

        self.assertTrue(parsed["expected_marker_presence"]["usd"])
        self.assertTrue(parsed["expected_marker_presence"]["usd1"])
        self.assertTrue(parsed["expected_marker_presence"]["table"])
        self.assertEqual(parsed["usd_buy_raw"], 148_500.0)
        self.assertEqual(parsed["usd_sell_raw"], 149_000.0)
        self.assertEqual(parsed["source_unit_assumption"], "toman")
        self.assertEqual(parsed["normalized_unit"], "rial")
        self.assertEqual(parsed["normalized_rial_value"], 1_490_000.0)
        self.assertFalse(parsed["bot_block_or_captcha"])

    def test_live_probe_sanity_bounds(self) -> None:
        if os.environ.get("RUN_LIVE_BONBAST_TEST") != "1":
            self.skipTest("set RUN_LIVE_BONBAST_TEST=1 to run live Bonbast integration test")

        mode = os.environ.get("BONBAST_LIVE_MODE", "playwright")
        try:
            report = probe_bonbast(url=os.environ.get("BONBAST_LIVE_URL", "https://bonbast.com"), timeout=30, mode=mode)
        except ModuleNotFoundError as exc:
            self.skipTest(f"live mode dependency missing: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        self.assertIn(report.get("http_status"), (200, 301, 302))
        self.assertIsInstance(report.get("final_url"), str)
        self.assertGreater(report.get("content_length") or 0, 10_000)
        self.assertTrue(Path(report["raw_html_file"]).exists())

        # If blocked/captcha is detected we surface it explicitly but avoid false-negative CI failures.
        if report.get("bot_block_or_captcha"):
            self.skipTest("live response looks like bot-block/captcha page")

        value = report.get("normalized_rial_value")
        self.assertIsNotNone(value, "expected parsed normalized_rial_value")
        self.assertGreater(value, 100_000)
        self.assertLess(value, 50_000_000)


if __name__ == "__main__":
    unittest.main()
