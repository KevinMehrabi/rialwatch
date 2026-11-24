# November 24: intermittent Pages failures and delayed publication

## Observed failures

The latest failed [Pages run](https://github.com/KevinMehrabi/rialwatch/actions/runs/45459784307)
successfully rebuilt the site and finalized its artifact upload. Less than a second
later, `actions/deploy-pages@v4` listed zero matching artifacts and stopped before
creating a deployment. The [02:05 UTC run](https://github.com/KevinMehrabi/rialwatch/actions/runs/45309739312)
failed at the same lookup. Successful deployments occurred between these failures.
This establishes a transient artifact-discovery handoff failure; it does not prove
which internal GitHub service caused the visibility delay. The action retries
deployment-status polling, but not this initial artifact lookup.

The public site remained reachable at the prior successfully deployed snapshot.
The recurring notices were genuine failed updates, not evidence of a continuous
website outage. The hosting path is GitHub Pages, not Render.

At investigation time, the public daily reference was November 23, while
`data/live` already contained valid November 24 observations collected at
16:15:48 UTC. The publisher had not caught up with the collector. Separately,
an [overnight collection run](https://github.com/KevinMehrabi/rialwatch/actions/runs/45294394817)
started after midnight and timed out two hours later in collection: a delayed
window trigger was waiting for that day's afternoon window.

An older [November 11 collection failure](https://github.com/KevinMehrabi/rialwatch/actions/runs/43095060367)
was an artifact backup service error after five upload attempts. Its separate Git
persistence step succeeded. This was a real backup failure; it is not suppressed
by the changes below.

## Correction

- Retry the official Pages action after 15 and 30 seconds. The final failed
  attempt still fails the workflow. Use a distinct artifact name per run attempt
  so reruns cannot collide with a previous upload.
- Serialize active deployments, with a 20-minute job limit. A queued run rebuilds
  from the latest committed data instead of deploying an old queued artifact.
- Write `api/deployment.json` with application/data revisions, run identity and
  hashes of public pages, APIs and dashboard assets. Check those actual responses
  on `rialwatch.com`, allowing up to three minutes for propagation. A fresh
  manifest with stale page/API content does not pass.
- Trigger daily publication after successful collection. Collection no longer
  triggers Pages directly: publication finishes first. Preserve the 14:20 UTC
  cutoff; near-cutoff collection completions can wait at most 15 minutes for it.
  Manual publisher runs use the same guarded publication path. Code pushes
  remain build-only.
- Window triggers outside 12:30–14:15 UTC collect immediately as recovery
  observations with their actual timestamps. An overnight delayed trigger cannot
  wait hours for a later window and lose its chance to retain an observation.

No benchmark formula, source eligibility rule, immutable historical fix, or
revision policy is changed.

## Validation before release

- 233 Python tests completed: 232 passed, one optional live-source test skipped.
- The collection regression fails against the original workflow, including the
  actual overnight timing case, and passes with the corrected routing.
- Workflow shell tests cover scheduled/manual/collection-triggered publication,
  build-only code pushes, and bounded waiting around the publication cutoff.
- Public verification tests cover stale homepages, stale APIs, prior deployment
  identities, delayed availability, persistent HTTP failures and missing files.
- `actionlint` 1.7.12 validates all three changed workflows.
- A replay of data snapshot `efb5f6ac9cb5783f964348f0bab4765c5d37ab33`
  produced the November 24 reference from the retained 16:15:48 observation and
  passed the guardrails. Hash comparison preserved all 1,120 existing JSON
  observation/publication records; only the missing current-day fix was added.

The replay retains the publication-quality warning: no qualifying observation
from the designated window was available. Recovery cannot manufacture that
missing evidence. GitHub scheduling and service availability remain external
dependencies; passing deployment checks is not proof that a future collection
window will be met. Every actual release now has its own public-content check.

## Live validation and browser-cache follow-up

After PR #15 merged, [collection](https://github.com/KevinMehrabi/rialwatch/actions/runs/45472311651)
succeeded and automatically triggered [publication](https://github.com/KevinMehrabi/rialwatch/actions/runs/45472513979).
Both guardrail passes succeeded with the expected missing-window warnings. The
November 24 reference was published at 2,842,000 IRR/USD from the 19:38:53 UTC
observation. All 1,120 pre-existing immutable observation/publication records
retained identical Git blob hashes; one observation and one daily fix were added.

The resulting [deployment](https://github.com/KevinMehrabi/rialwatch/actions/runs/45472604020)
encountered the same zero-artifact lookup failure. Its automatic retry found the
artifact 15 seconds later, deployed successfully, and verified eight public files.
This exercised the recovery in production, beyond the successful local tests.

A returning-browser check then exposed a separate cache inconsistency: the new
homepage showed November 24 while cached, unversioned API responses still showed
November 23 in the chart and ledger. Browser Resource Timing reported those API
responses as cache deliveries. The follow-up versions dashboard JavaScript and
all dashboard data requests with the homepage build timestamp, and requests API
revalidation. Deployment verification now includes chart-history data as well.
`verify/dashboard-cache.cjs` exercises two releases in the same browser against
real HTTP caching; it does not use route interception, which disables that cache.
