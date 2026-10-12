# RialWatch: USD/IRR Open Market Reference

Static daily USD/IRR reference site with an institutional dashboard UI and deterministic Python publishing pipeline.

## Architecture

- Static output: `/site` (generated at runtime; not tracked on `main`)
- CI state branch for generated runtime artifacts: `data/live` (checked out under `state/` in workflows)
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
- `/api/latest.json` latest official daily fix payload
- `/api/intraday/latest.json` latest observed intraday pulse payload
- `/api/series.json` ordered public historical rows (only actual valid daily fixes: numeric, non-withheld, Green/Amber/Red)
- `/api/daily_full.json` richer historical export generated from `/fix/*.json`
- `/api/revisions.json` revision metadata generated from `/fix/*.json`
- `/api/mapping_audit.json` mapping-version audit for historical backfill/rebuild targeting
- `/intraday/YYYY-MM-DD/HH-MM-SS.json` timestamped intraday collection attempts
- `/archive/`, `/status/`, `/methodology/`, `/governance/`

## Benchmarks

- Primary benchmark (flagship): `open_market` (`Open Market / Street Rate`)
  - primary source universe: `bonbast`, `alanchand_street`, `navasan`, `tgju_street`
- Supplementary benchmarks:
  - `official` (`Official Commercial USD Rate`)
  - `regional_transfer` (`Regional Transfer Rate`)
  - `crypto_usdt` (`Crypto Dollar (USDT)`)
  - `emami_gold_coin` (`Emami Gold Coin`)
- Derived indicators:
  - `street_official_gap_pct`
  - `street_transfer_gap_pct`
  - `street_crypto_gap_pct`
  - `street_gold_gap_pct`
  - `official_commercial_trend_7d`

Daily JSON (`/fix/YYYY-MM-DD.json`, with the newest official daily fix mirrored to `/api/latest.json`) includes:

- `computed.fix` for the primary benchmark
- top-level `benchmarks` with per-benchmark results (`fix`, `band`, `status`, `withheld`, etc.)
- `computed.benchmarks` lightweight summary for homepage/UI compatibility
- top-level `indicators` and `computed.indicators` for derived percentage signals

Public historical series (`/api/series.json`) remains strict and primary-only: it emits actual daily fixes only and does not carry forward missing days. `/api/daily_full.json` is the richer historical export with benchmark layers, source medians/units, indicators, revision metadata, and publication-selection metadata from immutable daily fix files. `/api/intraday/latest.json` is separate and may show a newer observed intraday pulse than the official daily fix.

## Intraday Collection And Daily Publication

- Intraday collection writes one timestamped file per attempt under:
  - `/site/intraday/YYYY-MM-DD/HH-MM-SS.json`
- Bonbast is collected via headless browser scrape (Playwright) during intraday runs and stored as normalized source sample data with health metadata.
- Initial cadence (UTC):
  - `13:50`
  - `14:05`
  - `14:14`
- Official daily publication runs once at `14:20 UTC` and normally selects from intraday attempts collected inside the publication window (`13:45-14:15 UTC`) using this explicit rule:
  - choose the latest valid in-window intraday attempt
  - if the latest in-window attempt is invalid, fall back to the most recent valid in-window attempt
  - if no in-window attempt is valid, repair from the latest valid same-day intraday read and record `publication_selection.selection_scope` as `same_day_repair`
  - if no valid same-day read exists, publish a WITHHOLD daily snapshot (no fabricated rate)
- Later outside-window intraday attempts update `/api/intraday/latest.json`; when they are used to repair an empty daily snapshot, `/fix/YYYY-MM-DD.json` revision metadata records the change.
- Historical missing or WITHHOLD daily files may be backfilled only from captured same-day intraday source-read artifacts; no prior-day carry-forward rows are emitted.
- If a scheduled collector starts after all checkpoints, it records one fresh outside-window recovery attempt with the actual collection time instead of silently skipping the day.
- Scheduled backup publications also create a missing daily reference. A missing current-day fix fails the guardrail even when there are no intraday artifacts, triggering collection and same-day repair.
- Additional immediate recovery collections are scheduled at `05:37`, `09:37`, `17:37`, and `21:37 UTC`. These keep same-day source reads outside the main window without changing the preferred in-window daily selection. Manual collection also runs immediately.
- Homepage remains daily-only (no live ticker behavior).

### Supplementary Source Wiring

- `open_market`:
  - canonical mapping: `bonbast -> usd1/usd2 midpoint`, `alanchand_street -> public USD buy/sell page`, `navasan public site -> usd`, `tgju_street -> price_dollar_rl profile page`
  - selection rule: median of valid, non-stale primary street-source readings after source-level validation
  - methodology note: all primary street inputs are normalized to IRR before aggregation; one-way public-board quotes are accepted when buy/sell midpoint is not available
- `official`:
  - canonical mapping: `commercial_aux|commercial_aux_a|commercial_aux_b|commercial_aux_c -> ice_transfer_usd_sell`
  - selection rule: freshest quote timestamp across official-source families; if multiple sources are equally fresh, pick the source(s) with highest in-window update cadence and median only those ties
  - methodology note: this is the live successor managed-market benchmark (`Official Commercial USD Rate`) from `2025-01-20` onward and is treated as a proxy for the managed commercial FX market that replaced NIMA
  - transition note: after the early-January-2026 hall merger, this track is treated as the unified commercial/remittance successor series when source structure reflects the merge
  - fallback status: heuristic fallback disabled
- `regional_transfer`:
  - canonical mapping: `navasan public site -> usd_shakhs`, `usd_sherkat`; `alanchand -> usd-hav`
  - methodology note: production-approved under strict canonical mapping
  - fallback status: heuristic fallback disabled
- `crypto_usdt`:
  - source families: `navasan`, `alanchand`, `bonbast (Playwright browser collector)`
  - preferred symbols: `usdt`, `usd_usdt`
  - fallback status: enabled (`tether` aliases retained)
- `emami_gold_coin`:
  - source families: `navasan`, `alanchand`, `bonbast (Playwright browser collector)`
  - preferred symbol: `sekkeh`
  - fallback status: enabled (legacy aliases retained)

Implementation references:

- Source-family routing: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py` (`build_source_configs`)
- Parsing logic: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py` (`CANONICAL_SOURCE_SYMBOLS`, `STRICT_CANONICAL_BENCHMARKS`, `extract_benchmark_values`, `extract_value_by_symbol_candidates`, `extract_benchmark_value`)
- Validation and withheld rules: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py` (`compute_benchmark_result`)
- Fallback behavior: card shows `Unavailable` if no valid benchmark value (`publish_home`).

## Optional GitHub Secrets

Optional API secrets can be added in **Settings -> Secrets and variables -> Actions -> Secrets**:

- `ALANCHAND_API_KEY`

Optional endpoint overrides and site overrides:

- `BONBAST_SITE_URL` (repository variable, defaults to `https://bonbast.com`)
- `NAVASAN_PUBLIC_URL` (repository variable, defaults to `https://www.navasan.net/`)
- `TGJU_STREET_PROFILE_URL` (repository variable, defaults to `https://www.tgju.org/profile/price_dollar_rl`)
- `ALANCHAND_API_URL`

No Navasan API secret is required; RialWatch collects Navasan readings from the public website endpoints used by the web UI.

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

Collect the production intraday window checkpoints from one early-starting job:

```bash
python scripts/pipeline.py --mode collect-intraday --site-dir site --templates-dir templates --assets-dir assets --collect-sample-times --sample-times-utc 13:50,14:05,14:14
```

Publish daily benchmark from collected intraday data:

```bash
python scripts/pipeline.py --mode publish-daily --site-dir site --templates-dir templates --assets-dir assets --skip-waits
```

Rebuild one existing day from stored samples using current mapping/normalization logic:

```bash
python scripts/pipeline.py --site-dir site --templates-dir templates --rebuild-day YYYY-MM-DD
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
- Intraday collection starts early and waits inside the job for `13:50`, `14:05`, and `14:14 UTC` checkpoints, then commits `collect:` snapshots to `data/live`. This absorbs GitHub scheduled-run startup delays while keeping official candidates inside the publication window.
- Official publish snapshots (`publish:` at `14:20 UTC`) and Iran discovery outputs (`discover:`) also commit to `data/live` instead of `main`.
- Self-heal schedules (`14:30 UTC`, `15:00 UTC`, `17:47 UTC`, `21:47 UTC`) publish or repair the current-day reference and persist snapshots, including raw recovery observations.
- New daily publication runs supersede obsolete runs, including a run stuck waiting for Pages. Running jobs have a 30-minute timeout.
- `main` intentionally stays code-only; generated `site/*` and `survey_outputs*` artifacts are ignored on `main`.
- Push/manual runs use build-only mode (`--no-new-reference`) to redeploy UI/template changes without creating a new day record.
- Build-only preserves existing `/site/fix/YYYY-MM-DD.json` official fix, `as_of`, and `publication_selection`; it may regenerate pages, archive, series exports, status, and intraday pulse files.
- Every build writes `/site/api/mapping_audit.json` with the current mapping fingerprint and any stale historical day payloads, so mapping changes can be backfilled explicitly instead of silently drifting.

## Observation Durability And Recovery

Each CI collection copies its raw source observations to a separate recovery directory before generating derived outputs. CI uploads that directory as an `intraday-observations-RUN_ID-RUN_ATTEMPT` artifact (or `recovery-observations-RUN_ID-RUN_ATTEMPT` for publisher auto-heal), retained for 90 days. This backup is independent of Git persistence and Pages deployment; successful observations also remain on `data/live`.

A rejected snapshot push or conflicting rebase now fails the workflow instead of claiming success. The collector still attempts Git persistence if the artifact upload fails. Monitor failed collection/publication runs in GitHub Actions; notification delivery depends on the account's GitHub notification settings.

To recover from a failed data-branch push, download that run's observation artifact with `gh run download RUN_ID --name ARTIFACT_NAME --dir recovery`. Restore its `intraday/` files into the `site/intraday/` directory in a `data/live` checkout, preserving their paths and timestamps. Run the existing build-only pipeline against that checkout and persist the recovered observations and generated daily artifacts using the normal data-branch process. Build-only mode backfills missing days from the original same-day source reads; it does not invent values for days with no valid observation.

The redundant schedules and both storage paths still depend on GitHub. They do not guarantee capture during a GitHub-wide outage or when all upstream sources are unavailable. Protection against a full scheduler outage requires an additional collector hosted outside GitHub.

## One-Time Backfill (If Early Days Were Lost)

If snapshots were not persisted in earlier runs, recover whatever dates exist in repo history:

```bash
python scripts/backfill_series_from_git.py
```

This rebuilds `/site/api/series.json` from historical versions of `/site/api/latest.json` in git.

For current production history, run this on the `data/live` branch.

## Custom Domain

1. In **Settings -> Pages**, set your **Custom domain**.
2. Add DNS records (`A/AAAA` for apex or `CNAME` for subdomain).
3. Add `/site/CNAME` with your domain, e.g.:

```text
fx.example.com
```

4. Re-run workflow and verify HTTPS is enabled.

## Dashboard design and verification

The homepage and shared shell use self-hosted Fraunces and IBM Plex Sans fonts,
plain CSS, and SVG charts. The theme still uses `rw-theme` in local storage and
starts in dark mode. All benchmark, indicator, and sparkline values retain their
existing Python template bindings; the timeline, daily ledger, and regional
signals read the existing public JSON endpoints. No market-data pipeline or API
schema changes are required.

The header's **Site updated** time describes the site refresh, while **Observed**
on the primary card describes the benchmark observation. Benchmark snapshots at
least two full days old show their age explicitly; a newer site or regional
diagnostics update never makes an older benchmark current.
Supplementary readings show full quote or collection dates, and regional
diagnostics identify the displayed benchmark used for matching comparisons.

The chart sits directly below the primary rate, with compact range presets and
benchmark details below it. The responsive layout changes at 900px. The timeline supports calendar-based
presets, pointer scrubbing, and a draggable overview. Focus either range handle
and use arrow keys (Shift moves seven observations); Home/End moves to a bound.
Manual ranges contain at least seven points, while presets can show shorter
windows. Focus the chart itself to inspect points with left/right arrows.

Regional signals preserve the existing visibility filter and 30-hour stale-data
warning. The shared Status link intentionally has no health dot because there is
no machine-readable pipeline-health endpoint to bind it to. Methodology and
Status body templates are unchanged.

To validate a generated site locally:

```bash
npm ci
npx playwright install chromium
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
# In another terminal:
npm run verify:header -- http://127.0.0.1:8000/
npm run verify:redesign -- http://127.0.0.1:8000/
npm run verify:accessibility -- http://127.0.0.1:8000/
python3 -m unittest discover -s tests
```

`site/` must contain generated HTML and the existing API artifacts (see the
build-only workflow above). `PW_CHROMIUM=/path/to/chromium` can reuse an installed
browser. Lighthouse reports are written to the OS temporary directory, or to
`RW_AUDIT_OUTPUT` if set. npm dependencies are only for verification; the deployed
site has no npm runtime or CDN dependency. The font licenses are in `assets/fonts/`.

### Audit safeguards (October 2026)

- Market quote recovery retains the intraday six-hour age limit, rejects future
  timestamps beyond ten minutes, and excludes failed source validation.
- Source exceptions are recorded individually; other successful reads survive.
  Raw observations are backed up before benchmark calculations or rendering.
- `deploy-pages.yml` is the only Pages publisher. Successful collection, daily,
  and regional workflows trigger a rebuild from the latest committed `data/live`
  state; obsolete Pages jobs can be cancelled without cancelling collection.
- Pull requests and main pushes run the Python regression suite.
- Gold-implied figures disclose their fixed USD/oz assumption. They are not
  backed by a live international gold-price feed.

GitHub remains a shared failure domain for schedules, storage, and publication.
For independent outage coverage, add a second collector on separately hosted
infrastructure and an external freshness check. Automated recovery cannot invent
observations for dates with no retained source data.
