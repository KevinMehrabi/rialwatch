import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import urlparse

from scripts.verify_deployment import (
    MANIFEST, VerificationError, prepare_release, wait_for_release,
)


class Clock:
    def __init__(self):
        self.now = 0

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class DeploymentVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name)
        (self.site / "api").mkdir()
        (self.site / "index.html").write_text("<html>Current reference</html>")
        (self.site / "api/latest.json").write_text(json.dumps({"date": "2026-11-24", "computed": {"fix": 2837450}}))
        (self.site / "api/series.json").write_text(json.dumps({"rows": [{"date": "2026-11-24", "fix": 2837450}]}))
        self.before = {str(p.relative_to(self.site)): p.read_bytes() for p in self.site.rglob("*") if p.is_file()}
        self.manifest = prepare_release(self.site, "application-sha", "data-sha", "123", "2")
        self.clock = Clock()
        self.requests = []

    def fetch(self, url, timeout):
        self.requests.append(url)
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 10)
        return (self.site / urlparse(url).path.lstrip("/")).read_bytes()

    def verify(self, fetch=None, seconds=25):
        return wait_for_release(self.manifest, "https://rialwatch.example/", seconds,
                                fetch=fetch or self.fetch, clock=self.clock.read, sleep=self.clock.sleep)

    def test_matches_public_content_without_mutating_rates_or_history(self):
        self.verify()
        self.assertEqual(len(self.requests), 4)
        for name, body in self.before.items():
            self.assertEqual((self.site / name).read_bytes(), body)

    def test_fresh_manifest_does_not_mask_stale_rate_api(self):
        def fetch(url, timeout):
            if urlparse(url).path == "/api/latest.json":
                return b'{"date":"2026-11-23"}'
            return self.fetch(url, timeout)
        with self.assertRaisesRegex(VerificationError, "api/latest.json"):
            self.verify(fetch)
        self.assertEqual(self.clock.now, 25)

    def test_a_stale_homepage_fails_even_when_data_is_current(self):
        def fetch(url, timeout):
            if urlparse(url).path == "/index.html":
                return b"<html>Yesterday</html>"
            return self.fetch(url, timeout)
        with self.assertRaisesRegex(VerificationError, "index.html"):
            self.verify(fetch)

    def test_prior_deployment_is_not_accepted(self):
        def fetch(url, timeout):
            if urlparse(url).path == "/" + MANIFEST:
                return json.dumps({**self.manifest, "run_attempt": "1"}).encode()
            return self.fetch(url, timeout)
        with self.assertRaisesRegex(VerificationError, "manifest"):
            self.verify(fetch)

    def test_temporary_missing_release_recovers_and_changes_cache_key(self):
        calls = []
        def fetch(url, timeout):
            calls.append(url)
            if len(calls) == 1:
                raise HTTPError(url, 404, "Not propagated", {}, None)
            return self.fetch(url, timeout)
        self.verify(fetch)
        self.assertEqual(self.clock.now, 10)
        self.assertNotEqual(calls[0], calls[1])

    def test_http_outage_remains_a_failure_after_bounded_retries(self):
        def fetch(url, timeout):
            raise HTTPError(url, 503, "Unavailable", {}, None)
        with self.assertRaisesRegex(VerificationError, "503"):
            self.verify(fetch)
        self.assertEqual(self.clock.now, 25)

    def test_missing_required_file_prevents_release_preparation(self):
        (self.site / "api/series.json").unlink()
        with self.assertRaisesRegex(VerificationError, "series.json"):
            prepare_release(self.site, "app", "data", "123", "3")


if __name__ == "__main__":
    unittest.main()
