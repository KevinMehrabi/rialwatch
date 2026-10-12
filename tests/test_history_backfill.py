import copy
import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import history_backfill as history
from scripts import pipeline

DAY = '2026-10-07'
STAMP = '2026-10-12T02:00:00Z'


def archive(day=DAY, buy='265800', sell='265900'):
    return f'''<title>IRR Exchange Rates Archive {day}- Bonbast.com</title>
    <em class="miladi">{day} .</em>
    All prices are in Iranian Toman (1 Toman = 10 Rials)
    <table><tr><td>Code</td><td>Currency</td><td>Sell</td><td>Buy</td></tr>
    <tr><td><img alt="USD">USD</td><td>US Dollar</td><td>{sell}</td><td>{buy}</td></tr></table>'''.encode()


class HistoricalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name)
        self.rows = [{'date': '2026-10-06', 'fix': 2690000, 'status': 'Green', 'withheld': False},
                     {'date': '2026-10-08', 'fix': 2677500, 'status': 'Green', 'withheld': False}]

    def test_midpoint_converts_toman_to_rial_and_preserves_provenance(self):
        record = history.write_record(self.site, DAY, archive(), STAMP)
        self.assertEqual(record['rate_irr_per_usd'], 2658500)
        self.assertIsNone(record['quote_time'])
        self.assertEqual(record['evidence_sha256'], hashlib.sha256(archive()).hexdigest())
        self.assertEqual(history.read_record(self.site, DAY), record)

    def test_rejects_latest_page_returned_instead_of_requested_date(self):
        with self.assertRaises(ValueError): history.parse_archive(archive('2026-10-11'), DAY)
        raw = archive().replace(b'<em class="miladi">2026-10-07', b'<em class="miladi">2026-10-11')
        with self.assertRaises(ValueError): history.parse_archive(raw, DAY)

    def test_requires_unambiguous_units_columns_and_currency(self):
        for raw in [archive().replace(b'Toman', b'Rials'), archive().replace(b'<td>Sell</td><td>Buy</td>', b'<td>Buy</td><td>Sell</td>'), archive().replace(b'>USD</td>', b'>EUR</td>')]:
            with self.subTest(raw=raw), self.assertRaises(ValueError): history.parse_archive(raw, DAY)

    def test_rejects_invalid_prices(self):
        for buy,sell in [('0','1'),('NaN','Infinity'),('265900','265800'),('100','200'),('-10','-9'),('100','100000000')]:
            with self.subTest(buy=buy,sell=sell), self.assertRaises(ValueError): history.parse_archive(archive(buy=buy,sell=sell), DAY)

    def test_recovery_is_chart_only_and_published_reference_wins(self):
        original = copy.deepcopy(self.rows)
        history.write_record(self.site, DAY, archive(), STAMP)
        chart = history.chart_rows(self.site, self.rows)
        self.assertEqual(self.rows, original)
        self.assertEqual(len(chart), 3)
        self.assertEqual(chart[1]['kind'], 'historical_archive')
        self.assertIsNone(chart[1]['p25'])
        official = {'date': DAY, 'fix': 2660000, 'withheld': False}
        chart = history.chart_rows(self.site, self.rows+[official])
        self.assertEqual(chart[1]['fix'], official['fix'])
        self.assertEqual(chart[1]['kind'], 'daily_reference')

    def test_tampered_evidence_and_rate_are_omitted(self):
        record = history.write_record(self.site, DAY, archive(), STAMP)
        (self.site/record['evidence_path']).write_bytes(archive(buy='265700'))
        self.assertEqual(len(history.chart_rows(self.site, self.rows)),2)
        record = history.write_record(self.site, DAY, archive(), STAMP)
        record['rate_irr_per_usd'] += 1000
        (self.site/'history/backfills'/f'{DAY}.json').write_text(json.dumps(record))
        self.assertEqual(len(history.chart_rows(self.site, self.rows)),2)

    def test_dates_and_file_paths_are_constrained(self):
        for day in ['2026-02-31', '../2026-10-07', '2026-1-7']:
            with self.assertRaises(ValueError): history.read_record(self.site, day)
        record = history.write_record(self.site, DAY, archive(), STAMP)
        record['evidence_path'] = '/etc/passwd'
        (self.site/'history/backfills'/f'{DAY}.json').write_text(json.dumps(record))
        with self.assertRaises(ValueError): history.read_record(self.site, DAY)

    def test_no_current_day_or_trailing_estimates(self):
        self.assertEqual(history.missing_days(self.rows, dt.date(2026,10,7)), [])
        self.assertEqual(history.missing_days(self.rows, dt.date(2026,10,12)), [DAY])
        self.assertEqual(history.missing_days(self.rows[:1]), [])

    def test_pipeline_keeps_benchmark_api_and_fixes_unchanged(self):
        pipeline.write_json(self.site/'api/series.json', {'rows':self.rows})
        history.write_record(self.site, DAY, archive(), STAMP)
        pipeline.publish_series(self.site)
        official = json.loads((self.site/'api/series.json').read_text())['rows']
        chart = json.loads((self.site/'api/chart_history.json').read_text())['rows']
        self.assertEqual([row['date'] for row in official], [row['date'] for row in self.rows])
        self.assertEqual(len(chart),3)
        self.assertFalse((self.site/'fix').exists())
        self.assertFalse((self.site/'api/latest.json').exists())

    def test_saved_records_are_idempotent_and_failed_dates_retry(self):
        pipeline.write_json(self.site/'api/series.json', {'rows':self.rows})
        with patch.object(history.urllib.request, 'urlopen', side_effect=OSError('temporary outage')) as fetch:
            result = history.recover(self.site)
            self.assertIn(DAY, result['failed'])
            self.assertFalse((self.site/'history/backfills'/f'{DAY}.json').exists())
            fetch.assert_called_once()
        with patch.object(history.urllib.request, 'urlopen') as fetch:
            fetch.return_value.__enter__.return_value.status = 200
            fetch.return_value.__enter__.return_value.read.return_value = archive()
            self.assertEqual(history.recover(self.site)['recovered'],[DAY])
            checksum = hashlib.sha256((self.site/'history/backfills'/f'{DAY}.json').read_bytes()).hexdigest()
            self.assertEqual(history.recover(self.site)['existing'],[DAY])
            fetch.assert_called_once()
            self.assertEqual(checksum,hashlib.sha256((self.site/'history/backfills'/f'{DAY}.json').read_bytes()).hexdigest())
