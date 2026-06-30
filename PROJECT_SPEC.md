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

### NASA Hump Methodology Smoke

The NASA/TMR wall-mounted hump slice is a CFD-methodology extension, not a new
headline benchmark. It ingests experimental Cp/Cf data and published CFL3D SA/SST
curves for a recognised separated-flow validation case, then checks whether the
small no-plenum PLOT3D grid can enter the OpenFOAM workflow.

Current status:

- reference data ingestion and SA/SST-vs-experiment metric plumbing are implemented;
- `plot3dToFoam` can ingest the `103 x 28` grid after `-noBlank -2D 0.1`;
- prototype patch splitting creates front/back/inlet/outlet/top/hump-wall patches;
- the global AeroMap mesh gate remains strict;
- a separate NASA/TMR methodology gate allows the official boundary-layer grid only
  for conversion, boundary-condition and single-solver smoke work;
- a bounded OpenFOAM v13 `kOmegaSST` smoke run completes after `potentialFoam`
  initialisation and exports hump-wall pressure/shear fields;
- the exported smoke wall fields can be converted into NASA/TMR-style `C_p(x)`
  and `C_f(x)` curves with an explicit `C_f` sign audit;
- the official `409 x 109` no-plenum grid can run through the same local SST
  pipeline with wall-tangent-projected `C_f`;
- that medium-grid SST candidate is not correlation-plausible yet: it still
  lacks the expected `C_f` zero-crossing behaviour and remains a methodology
  finding, not a model comparison;
- no OpenFOAM correlation result or turbulence-model recommendation is claimed.

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

This repository is an offline benchmark and reproducible prototype. It demonstrates budgeted simulation selection, compact 3D metadata ingestion, a field-level AirfRANS surface-pressure baseline, a structured underfloor pressure/load response map and a NASA/TMR separated-flow methodology smoke with Cp/Cf extraction plus a bounded medium-grid SST candidate.

The release does not present production F1 geometry, industrial live solver scheduling, 3D field prediction, wall-shear/separation labels, OpenFOAM NASA hump correlation accuracy, turbulence-model recommendation or DoMINO accuracy as current results. Those are extension paths once matching validation data are available.

The NASA hump extension does not claim OpenFOAM correlation accuracy or a
turbulence-model recommendation. The current evidence includes a single-grid
smoke run, smoke-grid Cp/Cf overlay and a `409 x 109` SST candidate that is not
yet correlation-plausible. Boundary-condition, grid or numerics work is required
before any SA/SST model-comparison branch.

## Reproduction Targets

Primary commands:

```sh
make lint
make test
uv run python scripts/generate_aeromap_release_figures.py
uv run aeromap benchmark aeromap-decision-replay-v03 \
  --config configs/benchmark/aeromap_mission_control_v03.yaml \
  --dataset-npz docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz \
  --out docs/evidence/aeromap/airfrans_decision_replay_v03.json \
  --svg-dir docs/evidence/aeromap
uv run scripts/run_venturi_core_2d_response_map_replay.py --overwrite
uv run python scripts/prepare_nasa_hump_methodology.py
uv run python scripts/convert_tmr_nasa_hump_to_openfoam.py
uv run python scripts/run_nasa_hump_sst_smoke.py --overwrite
uv run python scripts/report_nasa_hump_sst_smoke.py
uv run python scripts/extract_nasa_hump_cp_cf.py
uv run python scripts/run_nasa_hump_medium_grid_sst.py --overwrite --end-time 200
uv run python scripts/report_nasa_hump_medium_grid_sst.py
```
