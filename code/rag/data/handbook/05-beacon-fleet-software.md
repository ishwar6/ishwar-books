# Beacon - Fleet Management Software (Architecture and Operations)

*Owner: Fleet Platform. Audience: engineers, solutions engineers, support.*

## What Beacon does

Beacon is the brain of a Lumora deployment. It receives orders from the customer's
warehouse management system (WMS), turns them into robot tasks, assigns tasks to robots,
plans collision-free paths, schedules charging, and reports everything back.

## Architecture

- Runs on **Kubernetes** (GKE for our cloud-hosted customers, or the customer's own cluster
  for on-premise deployments).
- Persistent state in **PostgreSQL 16**; event stream on **Apache Kafka**.
- Robots connect over **gRPC with mutual TLS** through the **fleet gateway** service.
- Customers integrate through a **REST API** (JSON) and outbound webhooks.
- Path planning runs in the **planner** service, which recomputes the fleet plan every
  200 ms.

Services: `api`, `planner`, `fleet-gateway`, `charging-scheduler`, `compass-ingest`,
`notifier`.

## API

- Base URL: `https://api.<region>.beacon.lumora.example/v2`.
- Authentication: per-customer API keys, sent as a bearer token.
- Rate limit: **600 requests per minute per API key**. Above that the API returns HTTP 429
  with a `Retry-After` header.
- Webhooks are retried with exponential backoff for up to **24 hours**.
- **API v1 is deprecated** as of Beacon 4.2 and will be **removed in Beacon 4.4**.

## Availability

- Cloud-hosted Beacon has a **99.9% monthly uptime SLA**.
- Recovery time objective (RTO): **30 minutes**. Recovery point objective (RPO): **5 minutes**.
- If Beacon is unreachable, robots complete their current task and park; they never
  continue on stale plans.

## Release process

- Beacon ships on a **two-week release cadence**, every second **Tuesday**.
- Versions follow semantic versioning (`major.minor.patch`). Minor releases may add API
  fields but never remove them; removals only happen in a minor release announced at least
  two releases in advance.
- Every release goes through: staging soak (48 hours) → canary (5% of cloud customers,
  24 hours) → full rollout.
- Rollback command: `beaconctl rollout undo <service>`.

## Observability

- Metrics in Prometheus, dashboards in Grafana, traces in Tempo, logs in Loki.
- The key fleet health metric is **robots online %**; a fleet-wide drop below **95%** pages
  the on-call engineer (see the incident runbook).

## Environments

| Environment | Purpose |
|---|---|
| `dev` | shared developer environment, reset nightly |
| `staging` | release soak, mirrors production data schema |
| `prod-eu`, `prod-us`, `prod-in` | regional production clusters |

---
<!-- nav -->

[← Handbook index](../../README.md)
