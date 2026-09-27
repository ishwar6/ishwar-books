# Beacon 4.2 - Release Notes

*Released: 16 June 2026. Owner: Fleet Platform.*

## Highlights

- **Multi-warehouse zones.** A single Beacon tenant can now model several buildings on one
  campus, with robots allowed to travel between them through defined transfer corridors.
- **Predictive battery scheduling.** The charging scheduler now uses Compass battery-health
  data to predict runtime per robot instead of assuming a fleet-wide average. In beta
  customers this reduced charging-related idle time by **18%**.
- **New Compass dashboard: Shift Replay.** Scrub through a completed shift and watch every
  robot's path and task on the map.

## Improvements

- Path planner recompute interval lowered from 250 ms to **200 ms**.
- REST API: `GET /v2/robots` now supports filtering by `zone` and `battery_below`.
- Webhook payloads include a `schema_version` field.
- Admin console: API keys can be rotated without downtime.

## Bug fixes

- Fixed a bug where a robot that lost Wi-Fi while lifting a shelf could resume with an
  outdated destination (BEACON-4187).
- Fixed charging stations being marked "occupied" after a robot was e-stopped on the
  charger (BEACON-4203).
- Fixed Compass utilisation totals being off by one shift in time zones east of UTC+8.

## Breaking changes

- **API v1 is deprecated.** It continues to work in 4.2 and 4.3 and will be **removed in
  Beacon 4.4**. Migration guide: `docs/api/v1-to-v2.md`.
- Beacon 4.2 requires **Atlas firmware 3.8 or newer**. Robots on older firmware are
  updated automatically during their next charging window before the fleet is upgraded.

## Upgrade steps (on-premise customers)

1. Confirm all robots report firmware ≥ 3.8 in the admin console.
2. Take a database snapshot (`beaconctl db snapshot`).
3. `beaconctl upgrade --to 4.2.0` - the upgrade takes about 20 minutes and robots keep
   working on their current tasks.
4. Verify the Fleet Health dashboard shows all robots online.

Cloud-hosted customers were upgraded automatically between 16 and 23 June 2026.

## Known issues

- Shift Replay does not yet show robots that were in a transfer corridor between zones.
  Fix planned for 4.3.

## Next release

Beacon **4.3** is planned for **30 June 2026** and will add zone-level throughput targets
in Compass.

---
<!-- nav -->

[← Handbook index](../../README.md)
