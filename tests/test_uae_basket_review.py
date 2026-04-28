import datetime as dt
import unittest

from scripts import uae_basket_review as review


class UAEBasketReviewTests(unittest.TestCase):
    def test_basket_use_status_publishable(self) -> None:
        status = review.basket_use_status(
            numeric_count=4,
            repeated=True,
            freshness="fresh",
            reliability=78,
            has_signal=True,
        )
        self.assertEqual(status, "publishable")

    def test_basket_use_status_stale(self) -> None:
        status = review.basket_use_status(
            numeric_count=2,
            repeated=True,
            freshness="stale",
            reliability=80,
            has_signal=True,
        )
        self.assertEqual(status, "stale")

    def test_summarize_support_monitor(self) -> None:
        rows = [
            review.CandidateRecord(
                business_name="A",
                primary_surface_used="website:https://a.test",
                usd_irr_quote_detected=True,
                aed_irr_quote_detected=False,
                remittance_quote_detected=True,
                repeated_quote_signal=False,
                freshness_status="recent",
                parseability_score=55,
                reliability_score=52,
                basket_use_status="monitor_only",
                usable_record_count=1,
                numeric_quote_count=1,
                top_signal_sample="sample",
            ),
            review.CandidateRecord(
                business_name="B",
                primary_surface_used="telegram:https://t.me/s/b",
                usd_irr_quote_detected=True,
                aed_irr_quote_detected=True,
                remittance_quote_detected=False,
                repeated_quote_signal=False,
                freshness_status="recent",
                parseability_score=52,
                reliability_score=50,
                basket_use_status="monitor_only",
                usable_record_count=2,
                numeric_quote_count=2,
                top_signal_sample="sample",
            ),
        ]
        candidate_records = [
            review.BasketCandidateRecord("A", "website", "https://a.test", "USD", "midpoint", "", "", "140000.00", "1400000.00", "toman", "", "recent", 55, "direct", True, "x"),
            review.BasketCandidateRecord("B", "telegram", "https://t.me/s/b", "USD", "midpoint", "", "", "141000.00", "1410000.00", "toman", "", "recent", 52, "direct", False, "y"),
            review.BasketCandidateRecord("B", "telegram", "https://t.me/s/b", "AED", "midpoint", "", "", "38500.00", "385000.00", "toman", "", "recent", 54, "direct", False, "z"),
        ]
        self.assertEqual(review.summarize_support(rows, candidate_records), "monitor_card")

    def test_freshness_status_recent(self) -> None:
        now_dt = dt.datetime(2026, 3, 16, tzinfo=dt.timezone.utc)
        self.assertEqual(review.freshness_status("2026-02-20T00:00:00+00:00", now_dt), "recent")

    def test_visible_jalali_update_timestamp_is_parsed(self) -> None:
        text = "به روز رسانی: سه شنبه 08 اردیبهشت 1405 ساعت 15:04"
        self.assertEqual(review.parse_visible_update_timestamp(text), "2026-04-28T15:04:00Z")

    def test_direct_aed_rate_records_extract_single_value_pages(self) -> None:
        candidate = review.UAECandidate(
            business_name="Dubai AED Desk",
            website="https://rates.example/aed",
            instagram="",
            telegram="",
            whatsapp_link="",
            city_or_district="Dubai",
            country="UAE",
            source_type_guess="settlement_exchange",
            rate_page_detected=True,
            rate_post_detected=True,
            iran_transfer_hint=True,
            usd_irr_quote_detected=False,
            aed_irr_quote_detected=True,
            quote_post_count=1,
            parseability_score=70,
            last_seen="",
            candidate_score=80,
            discovery_origins="manual",
            status_guess="ok",
        )
        surface = review.Surface("Dubai AED Desk", "website", "https://rates.example/aed", "website")
        rows = review.extract_direct_rate_records(
            candidate=candidate,
            surface=surface,
            blocks=["نرخ تبدیل درهم به تومان در حال حاضر 42,360 تومان است"],
            timestamp_iso="2026-04-28T12:00:00Z",
            now_dt=dt.datetime(2026, 4, 28, 12, 10, tzinfo=dt.timezone.utc),
            benchmark_value=1_550_000.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].currency, "AED")
        self.assertEqual(rows[0].normalized_irr_value, "423600.00")


if __name__ == "__main__":
    unittest.main()
