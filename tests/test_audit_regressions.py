import datetime as dt
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import pipeline


class AuditRegressionTests(unittest.TestCase):
    now = dt.datetime(2026, 10, 11, 21, tzinfo=pipeline.UTC)

    def sample(self, key='open_market', age=dt.timedelta(), healthy=True):
        return pipeline.Sample(
            'bonbast', self.now, 2_640_000, {key: 2_640_000}, self.now-age,
            ok=False, stale=True, error='stale quote',
            health={'fetch_success': healthy, 'validation_result': {'ok': healthy}},
            source_unit='rial', normalized_unit='rial',
        )

    def result(self, sample, key='open_market'):
        return pipeline.compute_benchmark_result(
            {'bonbast': [sample]}, key, {'bonbast': (key,)}, primary_allow_stale=True,
        )

    def test_recovery_rejects_stale_and_future_quotes_but_keeps_recent_late_read(self):
        for age, available in [
            (dt.timedelta(days=30), False), (dt.timedelta(hours=6, seconds=1), False),
            (dt.timedelta(days=-2), False), (dt.timedelta(minutes=-10, seconds=-1), False),
            (dt.timedelta(hours=6), True), (dt.timedelta(minutes=-10), True),
            (dt.timedelta(minutes=3), True),
        ]:
            with self.subTest(age=age):
                self.assertEqual(self.result(self.sample(age=age))['available'], available)

    def test_official_future_quotes_do_not_pass_freshness_or_fallback(self):
        future = self.now + dt.timedelta(days=2)
        self.assertFalse(pipeline.official_quote_is_fresh(future, self.now))
        self.assertFalse(pipeline.official_quote_within_stale_fallback(future, self.now))
        self.assertFalse(self.result(self.sample('official', dt.timedelta(days=-2)), 'official')['available'])

    def test_companion_layers_reject_failed_validation_and_old_quotes(self):
        for key in ('crypto_usdt', 'regional_transfer', 'emami_gold_coin'):
            for sample in (self.sample(key, healthy=False), self.sample(key, dt.timedelta(days=2))):
                with self.subTest(key=key, health=sample.health):
                    self.assertFalse(self.result(sample, key)['available'])
            self.assertTrue(self.result(self.sample(key), key)['available'])

    def test_boolean_and_nonfinite_values_cannot_be_quotes(self):
        for value in (True, False, float('nan'), float('inf'), -1, 0):
            with self.subTest(value=value):
                self.assertIsNone(pipeline.parse_number(value))
                sample = self.sample()
                sample.benchmark_values['open_market'] = value
                self.assertFalse(self.result(sample)['available'])

    def test_one_source_transport_failure_does_not_lose_other_reads(self):
        configs = [pipeline.SourceConfig(n, 'https://example.test', 'public_json', (), ('open_market',), 'rial') for n in ('bonbast','tgju_street')]
        good = self.sample(); good.source = 'tgju_street'
        with mock.patch.object(pipeline, 'fetch_one', side_effect=[http.client.IncompleteRead(b'partial', 30), good]):
            samples = pipeline.collect_one_attempt(configs, self.now, self.now.date(), True)
        self.assertFalse(samples['bonbast'][0].health['fetch_success'])
        self.assertIn('IncompleteRead', samples['bonbast'][0].error)
        self.assertEqual(samples['tgju_street'][0].value, 2_640_000)

    def test_observation_backup_survives_benchmark_calculation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            site, backup = Path(tmp)/'site', Path(tmp)/'backup'
            with mock.patch.object(pipeline, 'collect_one_attempt', return_value={'bonbast':[self.sample()]}), mock.patch.object(pipeline, 'summarize_day', side_effect=RuntimeError('calculation failed')):
                with self.assertRaisesRegex(RuntimeError, 'calculation failed'):
                    pipeline.collect_and_write_intraday_attempt(site, self.now.date(), [], self.now, True, backup)
            retained = list(backup.rglob('*.json'))
            self.assertEqual(len(retained), 1)
            payload = json.loads(retained[0].read_text())
            self.assertEqual(payload['sources']['bonbast']['sample']['value'], 2_640_000)
            self.assertEqual(payload['collected_at'], '2026-10-11T21:00:00Z')
            self.assertEqual(payload['computed'], {})
            self.assertEqual(retained[0].read_bytes(), (site/retained[0].relative_to(backup)).read_bytes())

    def test_atomic_json_write_keeps_previous_observation_when_replacement_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"observation.json"
            pipeline.write_json(path, {"value": 2_640_000})
            with mock.patch.object(pipeline.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    pipeline.write_json(path, {"value": 2_650_000})
            self.assertEqual(json.loads(path.read_text())["value"], 2_640_000)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_status_uses_build_clock_and_recovery_candidate_count(self):
        templates = Path(__file__).resolve().parents[1]/"templates"
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            pipeline.write_json(site/"api/regional_market_signals_card.json", {
                "generated_at":"2026-10-06T19:00:00Z", "cards":[{
                    "basket_name":"UAE", "signal_label":"Transfer", "display_state":"publish",
                    "fresh_contributing_source_count":2, "contributing_source_count":2,
                    "usable_record_count":10,
                }],
            })
            latest = {
                "date":"2026-10-06", "as_of":"2026-10-06T19:18:00Z", "sources":{},
                "computed":{"fix":2_640_000, "withheld":False},
                "publication_selection":{"selection_scope":"same_day_repair",
                    "valid_candidate_count":0, "valid_same_day_candidate_count":2},
            }
            with mock.patch.object(pipeline, "refresh_regional_market_signal_cards"):
                pipeline.publish_status(site, templates, "2026-10-11T21:00:00Z", "OK", "Built", latest=latest)
            html = (site/"status/index.html").read_text()
            self.assertIn("2 valid same-day recovery candidates", html)
            self.assertIn("Latest reference is 2026-10-06; expected 2026-10-11", html)
            self.assertIn("Stale diagnostics artifact", html)
            self.assertNotIn("Dashboard deployment is online", html)

    def test_only_latest_snapshot_workflow_can_deploy(self):
        workflows = Path(__file__).resolve().parents[1]/'.github/workflows'
        publishers = [p.name for p in workflows.glob('*.yml') if 'uses: actions/deploy-pages@' in p.read_text()]
        self.assertEqual(publishers, ['deploy-pages.yml'])
        deploy = (workflows/'deploy-pages.yml').read_text()
        self.assertIn('ref: data/live', deploy)
        self.assertIn('--no-new-reference', deploy)
        self.assertIn('Intraday USD/IRR Collection', deploy)
        self.assertIn('cancel-in-progress: true', deploy)
        for name in ('daily-reference.yml', 'intraday-collection.yml', 'regional-diagnostics-discovery.yml'):
            self.assertNotIn('name: github-pages', (workflows/name).read_text())


if __name__ == '__main__':
    unittest.main()
