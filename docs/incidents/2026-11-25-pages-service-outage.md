# November 25: Pages service outage outlasted deployment retries

## Evidence

[Deployment 45650346607](https://github.com/KevinMehrabi/rialwatch/actions/runs/45650346607)
ran the corrected application revision `036c31b701b03e9575574558ea1d0842f1255533`.
The artifact uploaded and was found successfully on all three attempts. The
GitHub Pages deployment-creation API returned HTTP 500 at 16:01:03, 16:01:23,
and 16:01:56 UTC. No public-content verification ran because deployment failed.

This differs from the earlier zero-artifact discovery failure. The November 24
retry correction worked for brief interruptions, including an observed automatic
recovery, but the whole retry sequence here lasted less than one minute.

GitHub's [official incident record](https://www.githubstatus.com/api/v2/incidents/qn8mfv6q5hlx.json)
reports a wider outage from 15:55:19 through 16:52:16 UTC. Pages was marked degraded
at 15:57:42, then partially unavailable at 16:11:14, and operational at 16:28:29.
The API component remained degraded until 16:33:37. The failure's timing and HTTP
500 responses are consistent with that declared service outage; they do not
establish the internal GitHub root cause.

[Deployment 45664384640](https://github.com/KevinMehrabi/rialwatch/actions/runs/45664384640)
subsequently succeeded at 17:24 UTC. At investigation, the live manifest named that
run, and the November 25 reference was 2,922,000 IRR/USD. The email correctly
reported an earlier failed update; it did not mean the site remained down.

## Correction

- Wait for declared Pages or API outages before reading the latest data snapshot.
  Keep collection and daily publication independent of this wait.
- Check service health again before each retry, because an outage may begin
  after the first check. All waits use the same 75-minute recovery deadline.
- Keep the official deployment action, with four attempts separated by 15, 60,
  and 180 seconds. Each action remains limited to five minutes.
- If the status feed is unavailable before any outage is confirmed, try the real
  deployment; do not assume success. If an outage was already confirmed, keep
  waiting until health returns or the recovery deadline expires.
- Preserve fatal final errors and mandatory public-file verification. The job
  has a 110-minute outer limit covering recovery, attempts, build and verification.

## Validation and limits

Regression coverage replays the six relevant component-state transitions in the
official incident record. Starting at the failed run's start time, the new check
waits 33 simulated minutes, until both Pages and API Requests are operational.
Other cases cover persistent outages, status-feed failures, missing/malformed
status, unrelated incidents, and reuse of the same deadline across retries.
The full Python suite completed with 244 passes and one optional live-source test
skipped. `actionlint` 1.7.12 passed, and a live read of the GitHub Status feed
correctly allowed an immediate deployment attempt while both services were healthy.

This replay demonstrates that the new check would have deferred the attempts
made during the declared incident. It cannot prove that a later deployment would
have succeeded. A successful live deployment still requires the real Pages action
and public-content checks. Longer or undeclared outages can still produce failure
notifications. Independent hosting/collection remains the way to reduce reliance
on GitHub's availability; this change does not migrate infrastructure or suppress
notification settings. No historical rates or observations are changed.
