# November 1, 2026 publication-window investigation

The missing window resulted from late delivery of all three window-targeted
GitHub schedules. A separate publication defect finalized November 1 at 00:27
UTC, before its window opened. Existing immutability rules then correctly kept
that non-WITHHOLD fix, while incomplete guardrails incorrectly presented the
publication as an unqualified success.

## Evidence and timing

Application revision throughout the runs:
[`b954d570f80e5410e4a3896e61eb7e5c35ba6946`](https://github.com/KevinMehrabi/rialwatch/commit/b954d570f80e5410e4a3896e61eb7e5c35ba6946).
Retained data inspected at
[`69902b775c27124cfd93dc1ef275f5ca29252763`](https://github.com/KevinMehrabi/rialwatch/commit/69902b775c27124cfd93dc1ef275f5ca29252763).
`api/latest.json` is a moving pointer and had advanced to November 2 during this
investigation. The November 1 ledger is therefore the authoritative incident
record:
[`site/fix/2026-11-01.json`](https://github.com/KevinMehrabi/rialwatch/blob/69902b775c27124cfd93dc1ef275f5ca29252763/site/fix/2026-11-01.json).

All times below are UTC. Cron expressions are confirmed by the collection job
environment logs, not inferred from filenames.

| Scheduled trigger | Run created | Runner started | Captured observation | Run |
| --- | --- | --- | --- | --- |
| 12:30 | 17:59:15 | 17:59:18 | 17:59:53 | [41334233352](https://github.com/KevinMehrabi/rialwatch/actions/runs/41334233352) |
| 13:00 | 18:13:01 | 18:13:04 | 18:13:32 | [41335450669](https://github.com/KevinMehrabi/rialwatch/actions/runs/41335450669) |
| 13:30 | 18:28:34 | 18:28:37 | 18:29:14 | [41336806926](https://github.com/KevinMehrabi/rialwatch/actions/runs/41336806926) |

These are delays of approximately 5h29m, 5h13m, and 4h59m before run creation.
Runner allocation took about three seconds; checkout and Playwright setup took
less than a minute before sampling. The three jobs did not overlap. There is no
evidence of a repository concurrency lock, failed source fetch, lost push, or
slow dependency installation causing this window miss.

The workflow targets 13:50, 14:05, and 14:14 inside the inclusive
13:45-14:15 window. Once all checkpoints are missed, `run_collect_intraday`
deliberately captures a fresh outside-window attempt. Each of the three runs
logged that fallback and successfully persisted its observation. The full day
contains six files: 00:25:19, 11:59:46, 17:59:53, 18:13:32, 18:29:14, and
21:21:30. None is in the window. The other three collections used the distributed
`37 5,9,17,21 * * *` trigger.

[GitHub documents](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows)
that scheduled events can be delayed or dropped under load. These logs locate
the delay before run creation; they do not identify GitHub's internal cause or
prove a particular service incident. Additional cron entries in the same
service cannot guarantee an exact collection window.

## Why the midnight rate became the daily reference

1. [Collection 41256536702](https://github.com/KevinMehrabi/rialwatch/actions/runs/41256536702)
   saved `00-25-19.json`.
2. [Publication 41256773418](https://github.com/KevinMehrabi/rialwatch/actions/runs/41256773418)
   was created at 00:26:41 and ran `publish-daily --skip-waits` at 00:27:06.
   There is no configured midnight publication schedule. A delayed overnight
   schedule is the explanation consistent with the configured cron and the
   run's `schedule` event; the publisher did not log which cron expression fired,
   so the exact original trigger cannot be established from these logs.
3. `run()` chose the actual UTC date, November 1. `--skip-waits` bypassed the
   wait until 14:20, and neither `run_publish_daily` nor the selector enforced
   the cutoff independently.
4. With no window candidates yet, `select_daily_from_intraday` used the
   documented `same_day_repair` path. It reconstructed the early read as
   2,708,000 IRR/USD from Bonbast (2,700,000) and Alanchand Street (2,716,000).
   Dispersion was about 0.2954%, producing Green.
5. Commit
   [`225256876bf51fc5844beb7f9e242f94bf4d20b1`](https://github.com/KevinMehrabi/rialwatch/commit/225256876bf51fc5844beb7f9e242f94bf4d20b1)
   persisted the fix at 00:27:09, revision 0, with `candidate_count=0` and
   `selected_collected_at=2026-11-01T00:25:19Z`.

Recovery was applied before it was possible to determine that the day's window
had failed. That violates the documented publication timing even though
same-day repair itself is intentional policy.

## Why subsequent recovery passed

The later daily runs at 18:58, 19:02, 19:27, and 21:23 all had successful pre- and
post-heal guardrail steps; their auto-heal steps were skipped. In
[run 41351737169](https://github.com/KevinMehrabi/rialwatch/actions/runs/41351737169),
both checks logged six intraday files, zero in-window files, zero valid window
candidates, and a non-WITHHOLD fix of 2,708,000, followed by `Guardrails passed.`

`run_publish_daily` reuses a stored non-WITHHOLD daily fix. Automatic revisions
are restricted to missing or WITHHOLD records, with revision number, reason,
original publication metadata, and revision time retained. A later usable read
does not authorize changing an already published numeric reference.

The guardrail checked missing current-day fixes, contradictory WITHHOLD reasons,
null published rates, and missing companion benchmarks when usable window
evidence exists. It did not check premature publication, warn about absent
window coverage for a published fix, or distinguish dispersion from source
confirmation. Thus it could pass without triggering recovery or visible quality
warnings. Immutability was working; the timing gate and quality reporting were
defective.

## Source validity and the 21:21 observation

The documented quorum is one usable primary street source. Green/Amber/Red are
dispersion thresholds, not confidence grades. A single numeric source produces
zero dispersion and therefore Green under the existing implementation.

Recovery reconstructs rates from retained source samples, not merely an
attempt's stored `computed.fix`. It can relax window eligibility, but still
requires successful fetching and source validation, a positive finite quote,
and known market quote times no older than six hours or more than ten minutes
ahead of collection. A missing provider quote timestamp proves collection time
only. Official commercial rates have separate freshness/fallback rules.

Some source samples use `stale` to mean their quote timestamp lies outside the
fixed publication window. This is distinct from excessive quote age at actual
collection. Consequently the
[21:21:30 observation](https://github.com/KevinMehrabi/rialwatch/blob/69902b775c27124cfd93dc1ef275f5ca29252763/site/intraday/2026-11-01/21-21-30.json)
has a stored computation of 2,725,000 from Alanchand Street alone, but a replay
under the existing recovery rules accepts four retained sources:

| Source | Recovery-eligible IRR/USD |
| --- | ---: |
| Bonbast | 2,718,000 |
| Alanchand Street | 2,725,000 |
| Navasan | 2,718,000 |
| TGJU Street | 2,715,000 |

That reconstruction yields 2,718,000, not 2,725,000. This is a diagnostic replay,
not a revision or a claim that the sources are fully independent. Neither rate
was substituted into the published ledger.

## Correction and validation

- Enforce the 14:20 cutoff independently of `--skip-waits`; current-day repair
  also obeys it. Before the cutoff, the previous reference is expected and a
  missing current-day fix does not trigger auto-heal.
- Reject misfiled observations whose actual UTC collection date differs from
  the date being reconstructed.
- Emit explicit GitHub warnings when collection misses all checkpoints and
  when post-cutoff guardrails find no retained window evidence.
- Report permitted repairs and single-source limitations in a derived
  `/api/publication-quality.json`, the homepage, daily reference page, and
  status page. Preserve the stored rate, dispersion status, and provenance.
- Fail the guardrail for an already premature publication or mismatched
  selection scope. Such immutable records require review. Request automatic
  collection only when an empty or missing publication is eligible for repair.
- Log the publisher's cron expression and actual start time for future audits.

An allowed post-cutoff same-day repair still succeeds **with explicit quality
warnings**. Missing window evidence alone is not made fatal, because that would
silently reverse the documented fallback policy and cannot be repaired by a
late fetch. No source quorum, rate formula, quote-age limit, schedule, or
automatic revision permission was changed.

Validation: all 224 Python tests completed successfully, with one pre-existing
optional Playwright smoke test skipped. Ten new regressions cover midnight and
cutoff boundaries, deferred WITHHOLD repair, legitimate post-cutoff repair,
single-source disclosure, immutable preservation, guardrail annotations and
auto-heal eligibility, healthy multi-source window publication, and wrong-day
provenance. `git diff --check` passed.

An isolated replay used the six real November 1 observations and the October 31
and November 1 daily JSON files from the pinned data commit. After both daily
publication and build-only reruns, all eight input JSON SHA-256 hashes were
unchanged. The latest rate remained 2,708,000, revision 0, selected at 00:25:19;
the new guardrail failed for premature publication and disclosed the missing
window. No live data branch, historical rate, observation, or publication record
was changed by this investigation.

The remaining reliability limit is the shared GitHub scheduling failure domain.
A strict window guarantee requires an independently scheduled collector or
dispatcher. This patch does not provision another service or claim to eliminate
GitHub scheduling delay.
