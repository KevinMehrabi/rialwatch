# Diagnostic Report: Why Tabler UI Is Not Appearing on Deployed Pages

Date: 2026-03-08

## Summary
The repository currently **does contain** Tabler-linked templates, dashboard HTML structure, generated `site/` output with Tabler `<link>`/`<script>` tags, and copied frontend assets under `site/assets/tabler`.

Based on repo state, the most likely issue is **deployment path/timing**, not template rendering:
- the Pages workflow is configured to run only on `schedule` and `workflow_dispatch` (no `push` trigger), so UI changes can be committed without immediate redeploy;
- if GitHub Pages source is not set to **GitHub Actions**, the `deploy-pages` artifact path will not control production output.

This explains why DevTools shows no request for `tabler.min.css`/`tabler.min.js`: production is serving older HTML that does not include those tags.

## Findings By Requested Task

### 1) `templates/layout.html` Tabler references
Confirmed present in repository:
- `/Users/kevinmehrabi/Projects/rialwatch/templates/layout.html:8`
- `/Users/kevinmehrabi/Projects/rialwatch/templates/layout.html:10`

It includes:
- `<link rel="stylesheet" href="/assets/tabler/tabler.min.css" />`
- `<script src="/assets/tabler/tabler.min.js" defer></script>`

### 2) `templates/index.html` converted dashboard structure
Confirmed converted to dashboard structure:
- primary reference card (`Today's Reference`)
- historical card with chart container
- recent days table populated from `/api/series.json`

Evidence:
- `/Users/kevinmehrabi/Projects/rialwatch/templates/index.html:6`
- `/Users/kevinmehrabi/Projects/rialwatch/templates/index.html:35`
- `/Users/kevinmehrabi/Projects/rialwatch/templates/index.html:52`
- `/Users/kevinmehrabi/Projects/rialwatch/templates/index.html:129`

### 3) `scripts/pipeline.py` render flow
Pipeline reads templates and writes built HTML into `site/`; it does not overwrite template source files.

Evidence:
- asset copy to build output: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py:518`
- template render call: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py:532`
- homepage write to `site/index.html`: `/Users/kevinmehrabi/Projects/rialwatch/scripts/pipeline.py:605`

### 4) Generated `site/` output contains Tabler refs
Confirmed in generated output:
- `/Users/kevinmehrabi/Projects/rialwatch/site/index.html:8`
- `/Users/kevinmehrabi/Projects/rialwatch/site/index.html:10`

### 5) GitHub Actions artifact coverage
Workflow uploads entire `site` directory as Pages artifact:
- `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/daily-reference.yml:49`
- `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/daily-reference.yml:52`

So assets under `site/assets/**` are included in deploy artifact.

### 6) Frontend asset directory presence
Confirmed present in repo and build output:
- `/Users/kevinmehrabi/Projects/rialwatch/assets/tabler/tabler.min.css`
- `/Users/kevinmehrabi/Projects/rialwatch/assets/tabler/tabler.min.js`
- `/Users/kevinmehrabi/Projects/rialwatch/assets/tabler/fonts/tabler-mono.ttf`
- `/Users/kevinmehrabi/Projects/rialwatch/site/assets/tabler/...`

### 7) Deployed HTML containing Tabler tags
In repository build output, yes (confirmed).
In currently live site, user-reported behavior indicates no (older HTML is being served).

## Direct Answers

- **Why live site still looks unchanged**:
  Production is likely serving an older deployment. The current workflow does not run on push, so template/UI commits do not auto-deploy.

- **Whether Tabler was ever integrated**:
  Yes, integrated in templates and built output (`site/`) with local assets.

- **Whether pipeline overwrote templates**:
  No. Pipeline reads templates and writes rendered files into `site/`.

- **Whether assets are missing from deploy artifact**:
  Not based on repo/workflow. Assets exist and are included under `site/assets`.

- **Exactly what file(s) must change**:
  1. `/Users/kevinmehrabi/Projects/rialwatch/.github/workflows/daily-reference.yml`
     - Add `push` trigger (e.g. on `main`) so UI/template changes redeploy immediately.

## Non-file setting to verify (important)
- GitHub Pages source must be set to **GitHub Actions**. If set to branch deploy, the artifact deployment path is bypassed.
