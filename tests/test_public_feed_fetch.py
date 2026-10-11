import http.client
import unittest
import urllib.error
from unittest import mock

from scripts.telegram_quote_pilot_ingestion import fetch_url


class PublicFeedFetchTest(unittest.TestCase):
    def test_truncated_response_is_an_unavailable_source_not_a_feed_crash(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = http.client.IncompleteRead(b"partial quote", 100)
        with mock.patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(
                fetch_url("https://example.test/board", timeout=10),
                (None, None, "network_error:IncompleteRead"),
            )

    def test_truncated_http_error_preserves_error_status_without_parsing_partial_body(self) -> None:
        body = mock.Mock()
        body.read.side_effect = http.client.IncompleteRead(b"partial", 100)
        error = urllib.error.HTTPError("https://example.test/board", 503, "Unavailable", {}, body)
        with mock.patch("urllib.request.urlopen", side_effect=error):
            self.assertEqual(fetch_url("https://example.test/board", timeout=10), (None, 503, "http_503"))


if __name__ == "__main__":
    unittest.main()
