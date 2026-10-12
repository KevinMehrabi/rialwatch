#!/usr/bin/env python3
"""Recover chart-only history from dated public archives; never invent daily fixes."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ARCHIVE_URL = 'https://www.bonbast.com/archive'
KIND = 'historical_archive'


def iso_day(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Invalid calendar date')
    return dt.date.fromisoformat(value)


class ArchiveParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ''
        self.in_title = False
        self.visible_date = ''
        self.in_date = False
        self.rows = []
        self.row = None
        self.cell = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'title': self.in_title = True
        if tag == 'em' and 'miladi' in attrs.get('class', '').split(): self.in_date = True
        if tag == 'tr': self.row = []
        if tag in ('td', 'th'): self.cell = ''

    def handle_data(self, data):
        self.text.append(data)
        if self.in_title: self.title += data
        if self.in_date: self.visible_date += data
        if self.cell is not None: self.cell += data

    def handle_endtag(self, tag):
        if tag == 'title': self.in_title = False
        if tag == 'em': self.in_date = False
        if tag in ('td', 'th') and self.cell is not None:
            if self.row is not None: self.row.append(self.cell.strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_archive(raw: bytes, day: str) -> dict:
    iso_day(day)
    parser = ArchiveParser()
    parser.feed(raw.decode('utf-8-sig'))
    if not re.fullmatch(rf'IRR Exchange Rates Archive {day}\s*-\s*Bonbast\.com', parser.title.strip()):
        raise ValueError('Archive title does not confirm the requested date')
    if parser.visible_date.strip().rstrip(' .') != day:
        raise ValueError('Archive table does not confirm the requested date')
    if 'All prices are in Iranian Toman (1 Toman = 10 Rials)' not in ' '.join(parser.text):
        raise ValueError('Archive currency units are unconfirmed')
    if ['Code', 'Currency', 'Sell', 'Buy'] not in parser.rows:
        raise ValueError('Archive quote columns changed')
    matches = [row for row in parser.rows if row and row[0] == 'USD']
    if len(matches) != 1 or len(matches[0]) != 4 or matches[0][1] != 'US Dollar':
        raise ValueError('Expected one unambiguous USD archive row')
    _, _, sell_text, buy_text = matches[0]
    if not all(re.fullmatch(r'\d+(?:,\d{3})*(?:\.\d+)?', v) for v in (sell_text, buy_text)):
        raise ValueError('Invalid archive price')
    sell, buy = [float(v.replace(',', '')) for v in (sell_text, buy_text)]
    if not (0 < buy <= sell < 100_000_000) or (sell-buy)/buy > .05:
        raise ValueError('Invalid archive bid/ask spread')
    return {'buy_toman': buy, 'sell_toman': sell, 'rate_irr_per_usd': (buy+sell)*5}


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        name = tmp.name
    try:
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def write_record(site_dir: Path, day: str, raw: bytes, retrieved_at: str) -> dict:
    quote = parse_archive(raw, day)
    evidence = f'history/raw/bonbast/{day}.txt'
    record = {
        'schema_version': 1, 'date': day, 'kind': KIND,
        'provider': 'Bonbast', 'source_url': ARCHIVE_URL,
        'request': {'method': 'POST', 'date': day},
        'quote_date': day, 'quote_time': None, 'retrieved_at': retrieved_at,
        'quote_basis': 'Archived daily USD buy/sell midpoint; archive has no intraday timestamp',
        'normalized_unit': 'IRR per USD', **quote,
        'evidence_path': evidence, 'evidence_sha256': hashlib.sha256(raw).hexdigest(),
    }
    # Evidence is saved first. The quote is usable only when both files validate.
    atomic_write(site_dir/evidence, raw)
    atomic_write(site_dir/'history/backfills'/f'{day}.json', (json.dumps(record, indent=2)+'\n').encode())
    return record


def read_record(site_dir: Path, day: str) -> dict:
    iso_day(day)
    record = json.loads((site_dir/'history/backfills'/f'{day}.json').read_text())
    if not isinstance(record, dict):
        raise ValueError('Invalid historical record')
    evidence = f'history/raw/bonbast/{day}.txt'
    if (record.get('date') != day or record.get('quote_date') != day
            or record.get('schema_version') != 1 or record.get('kind') != KIND
            or record.get('source_url') != ARCHIVE_URL or record.get('provider') != 'Bonbast'
            or record.get('normalized_unit') != 'IRR per USD' or record.get('evidence_path') != evidence):
        raise ValueError('Invalid historical provenance')
    raw = (site_dir/evidence).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.get('evidence_sha256'):
        raise ValueError('Historical evidence checksum mismatch')
    expected = parse_archive(raw, day)
    for key, value in expected.items():
        stored = record.get(key)
        if isinstance(stored, bool) or not isinstance(stored, (int, float)) or not math.isfinite(stored) or stored != value:
            raise ValueError('Historical quote does not match its evidence')
    return record


def published_rows(rows):
    result = []
    for row in rows:
        try: iso_day(row.get('date'))
        except (TypeError, ValueError, AttributeError): continue
        value = row.get('fix')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 and not row.get('withheld'):
            result.append(row)
    return result


def missing_days(rows, today=None):
    dates = {iso_day(row['date']) for row in published_rows(rows)}
    if not dates: return []
    stop = min(max(dates), today or dt.datetime.now(dt.timezone.utc).date())
    day = min(dates)
    missing = []
    while day < stop:
        if day not in dates: missing.append(day.isoformat())
        day += dt.timedelta(days=1)
    return missing


def chart_rows(site_dir: Path, rows: list) -> list:
    official = published_rows(rows)
    combined = {row['date']: {**row, 'kind': 'daily_reference'} for row in official}
    for day in missing_days(official):
        if not (site_dir/'history/backfills'/f'{day}.json').exists(): continue
        try:
            record = read_record(site_dir, day)
        except (ValueError, OSError, TypeError):
            continue  # A corrupt archive must never block the live benchmark.
        combined[day] = {
            'date': day, 'fix': record['rate_irr_per_usd'], 'p25': None, 'p75': None,
            'kind': KIND, 'withheld': False,
            'provenance': {'provider': record['provider'], 'source_url': record['source_url'],
                           'retrieved_at': record['retrieved_at'], 'quote_basis': record['quote_basis'],
                           'record_url': f'/history/backfills/{day}.json'},
        }
    return sorted(combined.values(), key=lambda row: row['date'])


def recover(site_dir: Path, limit: int = 30) -> dict:
    rows = json.loads((site_dir/'api/series.json').read_text())['rows']
    result = {'recovered': [], 'existing': [], 'failed': {}, 'deferred': []}
    attempts = 0
    for day in missing_days(rows):
        path = site_dir/'history/backfills'/f'{day}.json'
        if path.exists():
            try: read_record(site_dir, day)
            except (ValueError, OSError, TypeError) as exc: result['failed'][day] = str(exc)
            else: result['existing'].append(day)
            continue
        if attempts >= limit:
            result['deferred'].append(day)
            continue
        attempts += 1
        try:
            req = urllib.request.Request(ARCHIVE_URL, data=urllib.parse.urlencode({'date': day}).encode(),
                                         headers={'User-Agent': 'RialWatch historical recovery (+https://rialwatch.com)'})
            with urllib.request.urlopen(req, timeout=25) as response:
                if response.status != 200: raise ValueError('Archive request failed')
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000: raise ValueError('Unexpected archive response size')
            write_record(site_dir, day, raw, dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'))
            result['recovered'].append(day)
        except Exception as exc:
            result['failed'][day] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100: parser.error('--limit must be between 1 and 100')
    result = recover(args.site_dir, args.limit)
    print(json.dumps(result, indent=2))
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
