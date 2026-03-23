import datetime as dt
import unittest

from scripts import pipeline


class PublicationWindowBoundariesTest(unittest.TestCase):
    def test_in_publication_window_includes_full_end_minute(self) -> None:
        day = dt.date(2026, 3, 23)
        self.assertTrue(
            pipeline.in_publication_window(dt.datetime(2026, 3, 23, 13, 45, 0, tzinfo=pipeline.UTC), day)
        )
        self.assertTrue(
            pipeline.in_publication_window(dt.datetime(2026, 3, 23, 14, 15, 59, tzinfo=pipeline.UTC), day)
        )
        self.assertFalse(
            pipeline.in_publication_window(dt.datetime(2026, 3, 23, 14, 16, 0, tzinfo=pipeline.UTC), day)
        )
        self.assertFalse(
            pipeline.in_publication_window(dt.datetime(2026, 3, 23, 13, 44, 59, tzinfo=pipeline.UTC), day)
        )

    def test_helper_window_minute_boundaries(self) -> None:
        start = dt.datetime(2026, 3, 23, 13, 45, tzinfo=pipeline.UTC)
        end = dt.datetime(2026, 3, 23, 14, 15, tzinfo=pipeline.UTC)
        self.assertTrue(
            pipeline.is_within_window_minute(
                dt.datetime(2026, 3, 23, 14, 15, 53, tzinfo=pipeline.UTC), start, end
            )
        )
        self.assertFalse(
            pipeline.is_within_window_minute(
                dt.datetime(2026, 3, 23, 14, 16, 1, tzinfo=pipeline.UTC), start, end
            )
        )


if __name__ == "__main__":
    unittest.main()
