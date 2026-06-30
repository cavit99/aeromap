# Project Specification

## Product

AeroMap Mission Control is a simulation-budget decision system for aerodynamic maps. It treats CFD labels as a limited resource and recommends the next case to evaluate based on prediction uncertainty, design-decision utility and domain-specific acquisition evidence.

The current public package is:

```text
AEROMAP_AEROCLIFF_CORE_MVP_V0_1
```

## Release Components

### AeroMap Open-CFD Benchmark

The headline benchmark uses AirfRANS:

- 1,000 real RANS airfoil simulations;
- scalar lift and drag targets;
- 27 compact geometry/operating-condition features;
- map-completion and geometry-disjoint split modes;
- acquisition replay with random, diversity, uncertainty, uncertainty plus diversity, engineering utility v1 and regret-aware utility v2.

On the geometry-disjoint split, regret-aware utility v2 leads C_D RMSE, top-k recovery, Pareto recall and efficiency-rank correlation. Utility v1 leads C_L RMSE. Diversity leads absolute best-design regret.

### DrivAerML 3D Bridge

The compact 3D bridge uses DrivAerML root metadata:

- 484 scalar cases;
- 16 geometry/design features;
- integrated drag and lift targets;
- three real cached DrivAerML STLs ingested for geometry-readiness descriptors.

This bridge is used to show that the AeroMap replay machinery works on compact automotive-aero 3D metadata. Diversity is the recommended policy for this bridge.

### AeroCliff Core

AeroCliff Core is a structured Venturi-underfloor response benchmark. It uses `blockMesh` hexahedra, fixed geometry conventions and scalar pressure/load targets.

The current Core response map uses:

```text
ride_height_mm:       50, 60, 70
diffuser_angle_deg:   3, 4, 5, 6, 7
throat_ratio:         0.7
U_inf:                40 m/s
```

All 15 medium cases pass mesh, mass-balance and force-stability gates. Three representative fine checks pass the pressure/load sanity thresholds. The offline Core replay uses C_D, suction/downforce and pressure recovery as targets.

## Coordinate And Coefficient Contract

- SI units internally.
- `x`: freestream direction.
- `y`: vehicle left.
- `z`: upward.
- `L_ref = 2.0 m`, `W_ref = 1.0 m`, `A_ref = 2.0 m^2`.
- `U_inf = 40 m/s`, `rho = 1.225 kg/m^3`, `nu = 1.5e-5 m^2/s`.
- `q_inf = 0.5 * rho * U_inf^2`.
- Downforce coefficient is positive downward: `C_DF = -F_z / (q_inf A_ref)`.
- Drag coefficient is positive opposing freestream.

## Data Contract

Compact replay datasets are committed as JSON metadata plus compressed NPZ arrays. Each artifact records:

- dataset/source identifier;
- feature names and normalisation;
- target names;
- split definition;
- acquisition protocol;
- source hashes where applicable.

Large raw CFD, cached downloads, generated OpenFOAM cases and model checkpoints stay out of Git.

## Acquisition Metrics

AeroMap reports both prediction error and design-decision quality:

- C_D RMSE and C_L RMSE;
- top-k design recovery;
- Pareto-front recall;
- Spearman rank correlation;
- best-design regret;
- learning-curve area;
- cost-proxy variants where a transparent proxy is available.

## Public Scope

This repository is an offline benchmark and reproducible prototype. It demonstrates budgeted simulation selection, compact 3D metadata ingestion and a structured underfloor pressure/load response map.

The release does not present production F1 geometry, live solver scheduling, field prediction, wall-shear/separation labels or DoMINO accuracy as current results. Those are extension paths once matching validation data are available.

## Reproduction Targets

Primary commands:

```sh
make lint
make test
uv run python scripts/generate_aeromap_portfolio_figures.py
uv run aeromap benchmark aeromap-decision-replay-v03 \
  --config configs/benchmark/aeromap_mission_control_v03.yaml \
  --dataset-npz docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz \
  --out docs/evidence/aeromap/airfrans_decision_replay_v03.json \
  --svg-dir docs/evidence/aeromap
uv run scripts/run_venturi_core_2d_response_map_replay.py --overwrite
```
