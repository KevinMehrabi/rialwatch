# RialWatch: USD/IRR Open Market Reference

Static daily USD/IRR reference site with an institutional dashboard UI and deterministic Python publishing pipeline.

## Architecture

- Static output: `/site`
- Pipeline: `/scripts/pipeline.py`
- Templates: `/templates`
- Local dashboard shell assets (Tabler-style): `/assets/tabler`
- Static assets copied by pipeline: `/assets` -> `/site/assets`
- Intraday collection workflow: `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/intraday-collection.yml`
- Daily publication/deploy workflow: `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/daily-reference.yml`
- Frontend runtime: plain HTML/CSS/JS (no React, no build toolchain)

## Key Outputs

- `/` dashboard homepage
- `/fix/YYYY-MM-DD/` daily permalink page
- `/fix/YYYY-MM-DD.json` daily JSON payload
- `/api/latest.json` latest payload
- `/api/series.json` ordered public historical rows (only valid published fixes: numeric, non-withheld, Green/Amber/Red)
- `/intraday/YYYY-MM-DD/HH-MM-SS.json` timestamped intraday collection attempts
- `/archive/`, `/status/`, `/methodology/`, `/governance/`

## Benchmarks

- Primary benchmark (flagship): `open_market` (`Open Market / Street Rate`)
- Supplementary benchmarks:
  - `nima` (`NIMA Rate`)
  - `official` (`MOB USD Benchmark`)
  - `regional_transfer` (`Regional Transfer Rate`)
  - `crypto_usdt` (`Crypto Dollar (USDT)`)
  - `emami_gold_coin` (`Emami Gold Coin`)
- Derived indicators:
  - `street_nima_gap`
  - `crypto_premium`

Daily JSON (`/fix/YYYY-MM-DD.json` and `/api/latest.json`) includes:

- `computed.fix` for the primary benchmark
- top-level `benchmarks` with per-benchmark results (`fix`, `band`, `status`, `withheld`, etc.)
- `computed.benchmarks` lightweight summary for homepage/UI compatibility
- top-level `indicators` and `computed.indicators` for derived percentage signals

Public historical series (`/api/series.json`) remains strict and primary-only.

## Intraday Collection And Daily Publication

- Intraday collection writes one timestamped file per attempt under:
  - `/site/intraday/YYYY-MM-DD/HH-MM-SS.json`
- Initial cadence (UTC):
  - `13:45`
  - `14:00`
  - `14:15`
- Official daily publication runs once at `14:20 UTC` and selects from intraday attempts in the publication window (`13:45-14:15 UTC`) using this explicit rule:
  - choose the latest valid intraday attempt
  - if the latest attempt is invalid, fall back to the most recent valid attempt
  - if none are valid, publish a WITHHOLD daily snapshot (no fabricated rate)
- Homepage remains daily-only (no live ticker behavior).

### Supplementary Source Wiring

- `nima`:
  - canonical mapping: not yet defined in code (intentionally unavailable)
  - fallback status: heuristic fallback disabled
- `official`:
  - canonical mapping: `navasan -> mob_usd`
  - methodology note: this is a narrower administrative benchmark, not a broad generic "Official Rate"
  - fallback status: heuristic fallback disabled
- `regional_transfer`:
  - canonical mapping: `navasan -> usd_shakhs`, `usd_sherkat`; `alanchand -> usd-hav`
  - methodology note: production-approved under strict canonical mapping
  - fallback status: heuristic fallback disabled
- `crypto_usdt`:
  - source families: `navasan`, `alanchand`, `bonbast`
  - preferred symbols: `usdt`
  - fallback status: enabled (`tether` aliases retained)
- `emami_gold_coin`:
  - source families: `navasan`, `alanchand`, `bonbast`
  - preferred symbol: `sekkeh`
  - fallback status: enabled (legacy aliases retained)

Implementation references:

- Source-family routing: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py` (`build_source_configs`)
- Parsing logic: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py` (`CANONICAL_SOURCE_SYMBOLS`, `STRICT_CANONICAL_BENCHMARKS`, `extract_benchmark_values`, `extract_value_by_symbol_candidates`, `extract_benchmark_value`)
- Validation and withheld rules: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py` (`compute_benchmark_result`)
- Fallback behavior: card shows `Unavailable` if no valid benchmark value (`publish_home`).

## Required GitHub Secrets

Add these in **Settings -> Secrets and variables -> Actions**:

- `BONBAST_USERNAME`
- `BONBAST_HASH`
- `NAVASAN_API_KEY`
- `ALANCHAND_API_KEY`

Optional endpoint overrides:

- `BONBAST_API_URL`
- `NAVASAN_API_URL`
- `ALANCHAND_API_URL`

If required secrets are missing, `/status/` is published as `CONFIG NEEDED` and no fake rate is emitted.

## Local Run

```bash
python scripts/pipeline.py --mode full --site-dir site --templates-dir templates --assets-dir assets
```

Quick local verification (no waiting for UTC window):

```bash
python scripts/pipeline.py --mode full --site-dir site --templates-dir templates --assets-dir assets --skip-waits --allow-outside-window
```

Collect one intraday attempt immediately:

```bash
python scripts/pipeline.py --mode collect-intraday --site-dir site --templates-dir templates --assets-dir assets
```

Publish daily benchmark from collected intraday data:

```bash
python scripts/pipeline.py --mode publish-daily --site-dir site --templates-dir templates --assets-dir assets --skip-waits
```

## GitHub Pages Setup

1. Open **Settings -> Pages**.
2. Set **Source** to **GitHub Actions**.
3. Save.

The workflow publishes `/site` daily.

To increase intraday frequency later, add more cron entries in:

- `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/intraday-collection.yml`

This can be expanded to hourly / every 30 minutes / every 15 minutes without parser changes.

## History Preservation

- Canonical cumulative public history file: `/site/api/series.json`.
- Scheduled runs generate one daily reference and write immutable snapshots to:
  - `/site/fix/YYYY-MM-DD.json`
  - `/site/fix/YYYY-MM-DD/index.html`
- Scheduled runs then update `/site/api/series.json` and auto-commit snapshots/history back to `main`.
- Push/manual runs use build-only mode (`--no-new-reference`) to redeploy UI/template changes without creating a new day record.

## One-Time Backfill (If Early Days Were Lost)

If snapshots were not persisted in earlier runs, recover whatever dates exist in repo history:

```bash
python scripts/backfill_series_from_git.py
```

This rebuilds `/site/api/series.json` from historical versions of `/site/api/latest.json` in git.

## Custom Domain

1. In **Settings -> Pages**, set your **Custom domain**.
2. Add DNS records (`A/AAAA` for apex or `CNAME` for subdomain).
3. Add `/site/CNAME` with your domain, e.g.:

```text
fx.example.com
```

4. Re-run workflow and verify HTTPS is enabled.
