# RialWatch v0: USD/IRR Open Market Reference

Static site generator for a daily USD/IRR open-market reference.

## Architecture

- Static output directory: `/site`
- Python pipeline: `/scripts/pipeline.py`
- Templates: `/templates`
- Scheduled publishing: `.github/workflows/daily-reference.yml`

## Daily rules implemented

- One daily reference publication at `14:20 UTC`
- Observation window `13:45-14:15 UTC`
- Three samples per source in the window
- Source medians -> headline FIX
- BAND = p25-p75 of source medians
- DISPERSION = `(p75 - p25) / FIX`
- Status thresholds:
  - Green <= 1.5%
  - Amber <= 3.5%
  - Red <= 5%
- WITHHOLD when:
  - fewer than 2 sources are valid
  - dispersion > 5%
  - invalid/stale inputs
- Governance:
  - no backfills
  - no rewriting historical files
  - immutable daily files once written

## Required GitHub Secrets

Repository Settings -> Secrets and variables -> Actions -> New repository secret:

- `BONBAST_USERNAME`
- `BONBAST_HASH`
- `NAVASAN_API_KEY`
- `ALANCHAND_API_KEY`

Optional endpoint override secrets (if provider URLs differ):

- `BONBAST_API_URL`
- `NAVASAN_API_URL`
- `ALANCHAND_API_URL`

If required secrets are missing, the pipeline publishes `CONFIG NEEDED` on `/status` and does not publish a fake rate.

## Local run

```bash
python scripts/pipeline.py --site-dir site --templates-dir templates
```

For quick local verification (no waiting for UTC window):

```bash
python scripts/pipeline.py --site-dir site --templates-dir templates --skip-waits --allow-outside-window
```

## Enable GitHub Pages

1. Open repository **Settings -> Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Save.

The scheduled workflow commits generated files into `/site` daily and deploys `/site` with the Pages deploy action.

## Set a custom domain

1. In **Settings -> Pages**, set **Custom domain**.
2. Add DNS records at your DNS provider:
   - `A`/`AAAA` for apex domain, or
   - `CNAME` for subdomain
3. Add a file at `/site/CNAME` containing your domain (single line), for example:

```text
fx.example.com
```

4. Re-run workflow and verify HTTPS is enabled in Pages settings.
