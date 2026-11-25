import datetime as dt
import json
from pathlib import Path
import unittest
from urllib.error import URLError

from scripts.wait_for_pages import RecoveryExpired, degraded_components, wait_until_ready


def payload(pages="operational", api="operational"):
    return {"components": [
        {"id": "vg70hn9s2tyj", "status": pages},
        {"id": "brv1bkgrwx7q", "status": api},
        {"id": "unrelated", "status": "major_outage"},
    ]}


class Clock:
    def __init__(self, now=0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class PagesRecoveryTests(unittest.TestCase):
    def wait(self, fetch, clock, deadline=4500):
        return wait_until_ready(deadline, fetch=fetch, wall_clock=clock,
                                clock=clock, sleep=clock.sleep, log=lambda text: None)

    def test_operational_services_proceed_immediately_despite_unrelated_incident(self):
        clock = Clock()
        self.assertEqual(self.wait(lambda timeout: payload(), clock), "ready")
        self.assertEqual(clock.sleeps, [])

    def test_pages_or_api_outage_blocks_until_recovery(self):
        for initial in (payload(pages="partial_outage"), payload(api="degraded_performance")):
            with self.subTest(initial=initial):
                clock = Clock()
                responses = iter([initial, payload()])
                self.assertEqual(self.wait(lambda timeout: next(responses), clock), "ready")
                self.assertEqual(clock.sleeps, [60])

    def test_replays_the_official_november_25_incident(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/github_pages_incident_2026-11-25.json").read_text())
        epoch = lambda text: dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        clock = Clock(epoch("2026-11-25T16:00:38Z"))
        start = clock.now

        def fetch(timeout):
            snapshot = next(x for x in reversed(fixture["changes"]) if epoch(x["at"]) <= clock.now)
            return payload(snapshot["pages"], snapshot["api"])

        self.assertEqual(self.wait(fetch, clock, start + 4500), "ready")
        self.assertEqual(clock.now - start, 33 * 60)
        self.assertGreaterEqual(clock.now, epoch("2026-11-25T16:33:37.356Z"))

    def test_a_longer_outage_can_recover_without_exhausting_short_action_retries(self):
        clock = Clock()
        self.assertEqual(self.wait(lambda timeout: payload(pages="major_outage") if clock.now < 57 * 60 else payload(), clock), "ready")
        self.assertEqual(clock.now, 57 * 60)

    def test_persistent_outage_fails_at_the_shared_deadline(self):
        clock = Clock()
        with self.assertRaisesRegex(RecoveryExpired, "not verified"):
            self.wait(lambda timeout: payload(api="major_outage"), clock)
        self.assertEqual(clock.now, 4500)
        self.assertTrue(all(delay <= 60 for delay in clock.sleeps))

    def test_later_retry_cannot_restart_the_recovery_budget(self):
        clock = Clock()
        self.wait(lambda timeout: payload(pages="partial_outage") if clock.now < 1200 else payload(), clock)
        self.assertEqual(clock.now, 1200)
        with self.assertRaises(RecoveryExpired):
            self.wait(lambda timeout: payload(pages="partial_outage"), clock)
        self.assertEqual(clock.now, 4500)

    def test_unavailable_status_feed_allows_actual_deployment_to_decide(self):
        clock = Clock()
        def fetch(timeout):
            raise URLError("status feed unavailable")
        self.assertEqual(self.wait(fetch, clock), "unknown")
        self.assertEqual(clock.sleeps, [])

    def test_status_feed_failure_does_not_clear_an_already_confirmed_outage(self):
        clock = Clock()
        def fetch(timeout):
            if clock.now == 0:
                return payload(pages="partial_outage")
            if clock.now == 60:
                raise URLError("temporary status-feed failure")
            return payload()
        self.assertEqual(self.wait(fetch, clock), "ready")
        self.assertEqual(clock.now, 120)

    def test_incomplete_or_malformed_status_is_unknown_not_healthy(self):
        for bad in ({}, None, {"components": []}, {"components": [{"id": []}]}, payload(pages=[]), payload(api="new_state")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    degraded_components(bad)
                self.assertEqual(self.wait(lambda timeout: bad, Clock()), "unknown")

    def test_expired_or_nonfinite_deadline_never_proceeds(self):
        for deadline in (-1, 0, float("inf"), float("nan")):
            with self.subTest(deadline=deadline), self.assertRaises(RecoveryExpired):
                self.wait(lambda timeout: self.fail("should not fetch"), Clock(), deadline)

    def test_network_time_counts_toward_deadline(self):
        clock = Clock()
        def fetch(timeout):
            self.assertEqual(timeout, 5)
            clock.now += timeout
            raise URLError("timeout")
        with self.assertRaises(RecoveryExpired):
            self.wait(fetch, clock, 5)
        self.assertEqual(clock.sleeps, [])

    def test_workflow_waits_before_data_checkout_and_keeps_final_failure_and_verification(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/deploy-pages.yml").read_text()
        self.assertLess(workflow.index("name: Wait for a declared GitHub outage"), workflow.index("name: Checkout latest committed data"))
        self.assertEqual(workflow.count('python3 scripts/wait_for_pages.py --deadline "$PAGES_RECOVERY_DEADLINE"'), 4)
        self.assertEqual(workflow.count("75 * 60"), 1)
        final = workflow.split("name: Final Pages deployment attempt", 1)[1].split("      - name:", 1)[0]
        self.assertNotIn("continue-on-error", final)
        self.assertIn("uses: actions/deploy-pages@v4", final)
        verify = workflow.split("name: Verify the released pages", 1)[1]
        self.assertNotIn("continue-on-error", verify)
        self.assertIn("scripts/verify_deployment.py check", verify)


if __name__ == "__main__":
    unittest.main()
