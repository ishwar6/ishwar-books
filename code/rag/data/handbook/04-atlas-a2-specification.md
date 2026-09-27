# Atlas A2 - Technical Specification Sheet

*Revision D, May 2026. Owner: Robot Platform.*

The Atlas A2 is Lumora's second-generation autonomous mobile robot for warehouse
transport. It carries shelves, totes and light pallets between pick stations and storage.

## Key specifications

| Property | Atlas A2 | Atlas A2 Lite |
|---|---|---|
| Maximum payload | **250 kg** | 120 kg |
| Maximum speed | **2.0 m/s** | 1.6 m/s |
| Battery runtime (continuous) | **8 hours** | 8 hours |
| Full charge time | **90 minutes** | 75 minutes |
| Fast top-up | **20 minutes for 2 hours of runtime** | same |
| Dimensions (L × W × H) | **1,120 × 720 × 380 mm** | 920 × 620 × 380 mm |
| Robot weight | **145 kg** | 110 kg |
| Ingress protection | **IP54** | IP54 |
| Operating temperature | **0 °C to 40 °C** | 0 °C to 40 °C |
| Lift height | 60 mm | 60 mm |

## Sensing and navigation

- One **270° safety LiDAR** at the front and two **depth cameras** (front and rear).
- Navigation is map-based SLAM with fiducial markers as an optional aid at pick stations.
- Positioning accuracy at a station: **±5 mm**.
- Obstacle detection stops the robot within **0.3 m** at full speed.

## Battery

- Lithium iron phosphate (LFP) pack, **48 V, 60 Ah**.
- Rated for **3,000 full cycles** to 80% capacity.
- Charging is fully automatic: Beacon schedules charging based on the task queue and the
  robot's state of charge. Opportunity charging (fast top-ups) is preferred to full cycles.

## Connectivity

- **Wi-Fi 6** (802.11ax) standard; **optional 5G module** for sites with poor Wi-Fi coverage.
- Robots talk to Beacon over gRPC with mutual TLS. A robot that loses connectivity for more
  than 60 seconds parks itself safely and waits.

## Safety and compliance

- Compliant with **ISO 3691-4** (driverless industrial trucks) and CE marked.
- Emergency-stop button on each corner; a stopped robot must be reset by an operator.
- Speed is automatically reduced to 0.8 m/s in zones marked "shared with people" in Beacon.

## Firmware

- Current firmware: **3.9**. Beacon 4.2 requires firmware **3.8 or newer**.
- Firmware is updated over the air by Beacon during scheduled charging windows.

## Maintenance

- Recommended preventive maintenance every **6 months** or 2,000 operating hours.
- Wheel modules are field-replaceable in under 15 minutes.
- Standard warranty is **24 months**; see the pricing sheet for the extended warranty.

---
<!-- nav -->

[← Handbook index](../../README.md)
