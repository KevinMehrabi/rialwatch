# RialWatch: USD/IRR Open Market Reference

Static daily USD/IRR reference site with an institutional dashboard UI and deterministic Python publishing pipeline.

## Architecture

- Static output: `/site`
- Pipeline: `/scripts/pipeline.py`
- Templates: `/templates`
- Local dashboard shell assets (Tabler-style): `/assets/tabler`
- Static assets copied by pipeline: `/assets` -> `/site/assets`
- Scheduled workflow: `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/daily-reference.yml`
- Frontend runtime: plain HTML/CSS/JS (no React, no build toolchain)

## Key Outputs

- `/` dashboard homepage
- `/fix/YYYY-MM-DD/` daily permalink page
- `/fix/YYYY-MM-DD.json` daily JSON payload
- `/api/latest.json` latest payload
- `/api/series.json` ordered public historical rows (only valid published fixes: numeric, non-withheld, Green/Amber/Red)
- `/archive/`, `/status/`, `/methodology/`, `/governance/`

## Benchmarks

- Primary benchmark (flagship): `open_market` (`Open Market / Street Rate`)
- Supplementary benchmarks:
  - `retail_seller` (`Retail Seller Rate`)
  - `retail_buyer` (`Retail Buyer Rate`)
  - `retail_mid_composite` (`Retail Mid-Market Composite`)

Daily JSON (`/fix/YYYY-MM-DD.json` and `/api/latest.json`) includes:

- `computed.fix` for the primary benchmark
- `computed.benchmarks` for all benchmark types with `label`, `value`, `available`, and `is_primary`

Public historical series (`/api/series.json`) remains strict and primary-only.

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
python scripts/pipeline.py --site-dir site --templates-dir templates --assets-dir assets
```

Quick local verification (no waiting for UTC window):

```bash
python scripts/pipeline.py --site-dir site --templates-dir templates --assets-dir assets --skip-waits --allow-outside-window
```

## GitHub Pages Setup

1. Open **Settings -> Pages**.
2. Set **Source** to **GitHub Actions**.
3. Save.

The workflow publishes `/site` daily.

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
