# Runbook - Fleet Outage (Robots Offline)

*Owner: Fleet Platform on-call. Last reviewed: June 2026.*

Use this runbook when robots at one or more customer sites stop receiving tasks or drop
offline.

## Severity

| Condition | Severity |
|---|---|
| A single robot offline for more than **60 seconds** | robot alert, no page |
| More than **5% of a site's robots** offline, or any site fully halted | **P1** - pages on-call |
| Beacon API error rate above 2% for 5 minutes | P2 |

## On-call

- On-call is managed in **PagerDuty** with a **primary and a secondary** engineer, rotating
  weekly on Mondays at 10:00 IST.
- Acknowledge a P1 page within **15 minutes**. If the primary does not acknowledge, PagerDuty
  escalates to the secondary after 15 minutes and to the **Fleet Platform lead after 30 minutes**.
- Post in `#incident-<date>` in Slack and update the customer status page within 20 minutes.

## Diagnosis checklist

1. Open the **Fleet Health** dashboard in Grafana. Is "robots online %" dropping for one
   site or all sites?
2. One site → suspect the site's network or the site's fleet gateway pod. Check the
   `fleet-gateway` logs in Loki for TLS or connection errors.
3. All sites in a region → suspect a bad release or a regional infrastructure issue. Check
   the release timeline; if a rollout happened within the last 2 hours, go to Rollback.
4. Robots online but no tasks → check the `planner` service and the Kafka consumer lag.

## Rollback

```
beaconctl rollout status <service>
beaconctl rollout undo <service>
```

Rollback first, investigate later. A rollback takes about 4 minutes per service.

## Certificates

The fleet gateway uses mutual TLS. Certificates are renewed automatically by the
`cert-renewer` job; expiry alerts fire **14 days** before expiry (added after the March
2026 incident). To renew manually:

```
beaconctl certs renew --service fleet-gateway --region <region>
```

## Recovery targets

- RTO **30 minutes**, RPO **5 minutes**.
- Robots automatically reconnect and resume queued tasks once the gateway is healthy; no
  manual robot reset is needed unless an e-stop was pressed.

## After the incident

- Write a postmortem within **5 business days** using the postmortem template.
- Postmortems are blameless and are shared company-wide.

---
<!-- nav -->

[← Handbook index](../../README.md)
