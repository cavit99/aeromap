# AeroMap + AeroCliff Core MVP v0.1

## Executive Summary

`AEROMAP_AEROCLIFF_CORE_MVP_V0_1` combines three pieces:

1. AeroMap Mission Control, an offline active-learning replay for CFD-label budgets.
2. A compact DrivAerML bridge that demonstrates 3D automotive-aero scalar ingestion.
3. AeroCliff Core, a structured Venturi-underfloor pressure/load response benchmark.

The headline claim is:

> AeroMap learns aerodynamic maps under a CFD-label budget and connects that decision loop to a structured Venturi-underfloor pressure/load response map.

## Why AeroCliff Core Exists

The full AeroCliff geometry is the custom 3D transfer lane. AeroCliff Core is the controlled benchmark tier: it keeps the underfloor throat/diffuser mechanism and replaces the unstructured 3D meshing burden with a structured `blockMesh` setup.

Core is intentionally small:

- structured hexahedral mesh;
- explicit moving-belt ground;
- stationary floor underside;
- throat/diffuser profile;
- pressure/load targets;
- fast local runs.

That makes it suitable for a public MVP: the response surface is small enough to audit case by case, but still connected to the underfloor physics that motivated the project.

## AirfRANS Decision Benchmark

The open-CFD benchmark uses 1,000 AirfRANS RANS cases. The feature contract contains 27 geometry and operating-condition features, and the harder split is geometry-disjoint over deterministic geometry descriptors.

On that split, regret-aware utility v2 leads:

- C_D RMSE;
- top-k recovery;
- Pareto recall;
- Spearman rank correlation.

Utility v1 remains best for C_L RMSE, while diversity remains best for absolute best-design regret. This is used directly in the policy selector rather than hidden.

## DrivAerML 3D Bridge

The DrivAerML bridge uses compact root metadata:

- 484 scalar cases;
- 16 geometry/design features;
- integrated C_D and C_L targets;
- three cached DrivAerML STLs parsed into geometry descriptors.

Diversity is currently the recommended acquisition policy for this bridge. The point of the bridge is practicability: the same replay machinery handles compact 3D automotive-aero data without downloading volume fields or boundary fields.

## AeroCliff Core Response Map

The Core response map spans:

```text
ride_height_mm:       50, 60, 70
diffuser_angle_deg:   3, 4, 5, 6, 7
throat_ratio:         0.7
U_inf:                40 m/s
```

Case quality:

- `15 / 15` medium structured cases passed mesh, mass-balance and force-stability gates.
- `3 / 3` representative fine checks passed.

Fine-check differences:

| Case | C_D diff | suction/downforce diff | pressure recovery diff |
|---|---:|---:|---:|
| `70/3` | 0.24% | 0.04% | 0.01% |
| `50/7` | 2.84% | 1.53% | 1.54% |
| `50/6` | 1.28% | 0.76% | 0.76% |

## Core Replay Protocol

The offline replay treats the 15 Core cases as a finite pool of expensive CFD labels.

Inputs:

- ride height;
- diffuser angle;
- derived diffuser/throat area ratio.

Targets:

- C_D;
- suction/downforce coefficient;
- pressure recovery.

Methods:

- random;
- diversity / space filling;
- uncertainty;
- engineering utility;
- cost-aware utility.

The surrogate is a deterministic inverse-distance response interpolator. That keeps the result focused on the decision loop rather than model architecture.

## Core Replay Result

Classification:

```text
AEROCLIFF_CORE_2D_PRESSURE_LOAD_RESPONSE_REPLAY_V0
```

Replay summary:

| Method | Curve-error area | Budget-8 normalised RMSE | Budget-8 high-gradient recall |
|---|---:|---:|---:|
| engineering utility | 1.065504 | 0.108925 | 0.667 |
| cost-aware utility | 1.065504 | 0.108925 | 0.667 |
| diversity / space filling | 1.075858 | 0.101231 | 0.667 |
| uncertainty | 1.075858 | 0.101231 | 0.667 |
| random | 1.154690 | 0.129759 | 0.733 |

Engineering utility and cost-aware utility tie for best curve-error area. Diversity and uncertainty remain competitive at the final budget. The useful conclusion is that Mission Control can operate on a custom structured underfloor response map, and that simple baselines remain important.

## Scope

This MVP establishes:

- open-CFD budgeted acquisition on AirfRANS;
- compact 3D automotive scalar replay on DrivAerML;
- structured AeroCliff Core pressure/load response mapping;
- an offline acquisition replay over the Core response surface.

Follow-on work:

- live CFD scheduling;
- richer 3D open-data benchmarks;
- pressure-field or component-level metrics;
- higher-fidelity custom AeroCliff transfer studies.

## Artifact Index

- Core dataset: `docs/evidence/cfd/aerocliff_core/core_2d_response_map_dataset_v0.json`
- Core replay: `docs/evidence/cfd/aerocliff_core/core_2d_response_map_active_replay_v0.json`
- AirfRANS replay: `docs/evidence/aeromap/airfrans_decision_replay_v03.json`
- DrivAerML bridge replay: `docs/evidence/aeromap3d/drivaerml_scalar_bridge_replay.json`
- Demo: `docs/demo/aeromap_mission_control.html`
- Main figures:
  - `docs/assets/aeromap/aeromap_headline_geometry_heldout.png`
  - `docs/assets/aeromap/aerocliff_core_response_surface.png`
  - `docs/assets/aeromap/aerocliff_core_pressure_recovery_surface.png`
  - `docs/assets/aeromap/aerocliff_core_active_replay.png`
  - `docs/assets/aeromap/aeromap_aerocliff_core_story.png`
