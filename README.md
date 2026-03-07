# RialWatch: USD/IRR Open Market Reference

Static daily USD/IRR reference site with an institutional dashboard UI and deterministic Python publishing pipeline.

## Architecture

- Static output: `/site`
- Pipeline: `/scripts/pipeline.py`
- Templates: `/templates`
- Static assets copied by pipeline: `/assets` -> `/site/assets`
- Scheduled workflow: `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/daily-reference.yml`

## Key Outputs

- `/` dashboard homepage
- `/fix/YYYY-MM-DD/` daily permalink page
- `/fix/YYYY-MM-DD.json` daily JSON payload
- `/api/latest.json` latest payload
- `/api/series.json` ordered historical rows built from `/site/fix/*.json`
- `/archive/`, `/status/`, `/methodology/`, `/governance/`

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

## Custom Domain

1. In **Settings -> Pages**, set your **Custom domain**.
2. Add DNS records (`A/AAAA` for apex or `CNAME` for subdomain).
3. Add `/site/CNAME` with your domain, e.g.:

```text
fx.example.com
```

4. Re-run workflow and verify HTTPS is enabled.
