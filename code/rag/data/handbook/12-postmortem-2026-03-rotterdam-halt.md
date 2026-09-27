# Postmortem - Fleet Halt at Veldmark Logistics, Rotterdam

*Incident date: 12 March 2026. Severity: P1. Author: Fleet Platform. Status: closed.*

## Summary

On 12 March 2026, all **47 Atlas robots** at the Veldmark Logistics site near Rotterdam
lost connection to Beacon and parked. The site was halted for **2 hours 13 minutes**
(09:41 to 11:54 CET). No safety incidents occurred. The root cause was an **expired TLS
certificate on the site's fleet gateway**; the automatic renewal job had been failing
silently since 28 February.

## Impact

- 47 robots idle for 2 h 13 min; about 3,100 tasks delayed.
- Veldmark is a Gold-tier customer. The P1 response target (30 minutes) was met: first
  response at 09:52, 11 minutes after the page.
- Service credit issued under the SLA.

## Timeline (CET)

| Time | Event |
|---|---|
| 28 Feb, 02:00 | `cert-renewer` job fails (expired credentials to the certificate authority). Failure logged, no alert. |
| 12 Mar, 09:40 | Fleet gateway certificate expires. Robots fail mTLS handshake. |
| 09:41 | Robots online % for the site drops to 0. PagerDuty pages primary on-call. |
| 09:52 | On-call acknowledges, opens `#incident-2026-03-12`, updates the status page. |
| 10:05 | Rollback considered and rejected: no recent release. |
| 10:40 | Gateway logs show `certificate has expired`. Root cause identified. |
| 11:20 | Manual renewal (`beaconctl certs renew`) fails because the renewer's own credentials had expired; new credentials issued. |
| 11:48 | Certificate renewed and gateway restarted. |
| 11:54 | All 47 robots reconnected and resumed queued tasks. Incident closed. |

## Root cause

The `cert-renewer` job authenticates to the internal certificate authority with a
credential that itself expires yearly. That credential expired on 28 February. The job
logged an error and exited 0, so no alert fired and the fleet gateway certificate was never
renewed.

## What went well

- Robots behaved safely: they parked and waited rather than continuing on stale plans.
- Response target met; customer communication was regular.

## What went badly

- A silent failure went unnoticed for 12 days.
- No alerting on certificate expiry, so the first signal was the outage itself.
- Manual renewal took 28 minutes longer than necessary because the fix needed a second
  credential.

## Action items

| # | Action | Owner | Due | Status |
|---|---|---|---|---|
| 1 | Alert **14 days** before any fleet gateway certificate expires | Fleet Platform | 31 Mar 2026 | Done |
| 2 | `cert-renewer` exits non-zero and pages on failure | Fleet Platform | 31 Mar 2026 | Done |
| 3 | Monitor the renewer's own CA credential expiry | Security | 15 Apr 2026 | Done |
| 4 | Quarterly game day: expire a staging certificate on purpose | Fleet Platform | 30 Jun 2026 | Done |
| 5 | Add certificate checks to the outage runbook | Fleet Platform | 31 Mar 2026 | Done |

---
<!-- nav -->

[← Handbook index](../../README.md)
