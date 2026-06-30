# AeroMap 3D Practicability Bridge

## Executive Summary

This bridge tested whether AeroMap Mission Control can move beyond the 2D
AirfRANS benchmark into realistic 3D aerodynamic design data without reopening
the large-data, cloud, NIM or custom-CFD lanes.

Result:

```text
AEROMAP_3D_OPEN_CFD_SCALAR_BRIDGE
selected_dataset = DrivAerML
scope = compact_3d_scalar_metadata_replay
geometry_sample = real_drivaerml_stl_ingestion
compute = local_no_cloud_replay
```

The bridge is successful as a practicability result:

- compact 3D scalar metadata was obtained without cloning a full dataset;
- DrivAerML was selected by the predeclared selection rule;
- 484 real 3D automotive CFD cases were converted into a compact scalar replay
  dataset;
- three real DrivAerML STL surfaces were ingested from existing local cache and
  reduced to compact descriptors and point samples;
- the existing AeroMap active-learning replay ran on the 3D scalar dataset.

The bridge is not a new custom-acquisition win. On the harder
geometry-heldout DrivAerML replay, simple diversity is currently the best
recommended acquisition policy across most decision metrics. The regret-aware
utility v2 improves best-design regret versus random, but it does not beat
diversity.

## Scope

The bridge uses compact metadata and scalar targets only. It is a local
practicability result for 3D aerodynamic design data: geometry parameters,
integrated force/moment summaries and a tiny STL readiness sample. Bulk field
data, live solver coupling and higher-fidelity custom Core/3D transfer studies
belong to later work.

## Phase 1: Metadata-Only Triage

The inspected datasets were HiLiftAeroML, DrivAerML and AhmedML. The triage was
performed through metadata/API and HTTP HEAD checks only. No volume fields,
boundary fields or full dataset clones were downloaded.

| Dataset | Licence | Cases | Compact scalar/geometry bytes | Estimated 3-STL sample | Scalar replay feasible? | Main relevance |
|---|---|---:|---:|---:|---|---|
| DrivAerML | CC BY-SA 4.0 | 500 geometries, 484 labelled scalar rows | 145,463 | 427,155,558 bytes | yes | vehicle/motorsport bridge |
| HiLiftAeroML | CC BY 4.0 | 1,800 samples | 300,882 | 616,652,652 bytes | yes | real-wing credibility |
| AhmedML | CC BY-SA 4.0 | 500 variants | 110,448 | 22,492,452 bytes | yes | compact bluff-body fallback |

Triage output:

```text
docs/evidence/aeromap3d/metadata_triage.json
classification = AEROMAP_3D_COMPACT_PRACTICABILITY_TRIAGE
selected_dataset = DrivAerML
```

Selection reason:

```text
DrivAerML satisfies the explicit selection rule: compact all-case geometry
parameters and force/moment summaries are accessible.
```

## Phase 2: Dataset Selection

DrivAerML was selected because it satisfies the predeclared rule:

```text
If DrivAerML all-case geometry parameters and force/moment summaries can be
downloaded compactly, choose DrivAerML.
```

Why this is the right first 3D bridge:

- strongest vehicle/motorsport relevance among the three candidates;
- root all-case force/moment CSVs exist;
- root all-case geometry parameter CSV exists;
- the scalar bridge can be built without boundary or volume fields;
- existing local DrivAerML STLs from earlier work avoid any new large STL
  download for the tiny geometry readiness sample.

HiLiftAeroML remains attractive for a later wing-focused slice. AhmedML remains
the safest compact fallback if DrivAerML becomes unavailable or too large.

## Phase 3: Compact 3D Scalar Dataset

Created:

```text
AEROMAP_3D_SCALAR_BRIDGE_DATASET
docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.json
docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.npz
```

Dataset summary:

| Field | Value |
|---|---|
| Source dataset | DrivAerML |
| Case count | 484 |
| Feature count | 16 |
| Targets | `integrated_cd`, `integrated_cl` |
| Target source | `force_mom_constref_all.csv` |
| Geometry source | `geo_parameters_all.csv` |
| Compact NPZ size | about 46 KiB |

Feature contract:

The compact feature vector is built from DrivAerML geometry/design parameters:
vehicle length, width, height, front overhang, front planform, hood angle,
approach angle, windscreen angle, greenhouse tapering, backlight angle,
decklid height, rear-end tapering, rear overhang, rear diffuser angle, ride
height and pitch.

Release scope:

```text
compact_3d_scalar_bridge = true
dataset = DrivAerML compact metadata
targets = integrated scalar force coefficients
geometry = STL descriptor readiness sample
compute = local offline replay
```

## Phase 4: Tiny 3D Geometry Readiness Sample

Created:

```text
AEROMAP_3D_GEOMETRY_READINESS_SAMPLE
docs/evidence/aeromap3d/drivaerml_geometry_readiness_sample.json
docs/evidence/aeromap3d/drivaerml_geometry_readiness_sample.points.npz
```

No new STLs were downloaded. The sample used three existing cached DrivAerML
STLs:

| STL | Triangles | Surface area | Bounding box |
|---|---:|---:|---|
| `drivaer_2.stl` | 753,246 | 37.672 | `[4.728, 2.147, 1.384]` |
| `drivaer_10.stl` | 753,232 | 35.053 | `[4.807, 1.896, 1.448]` |
| `drivaer_102.stl` | 753,238 | 37.345 | `[4.671, 2.130, 1.462]` |

For each STL the bridge computed bounding box, scale diagonal, surface area,
triangle-area statistics, normal statistics, deterministic sampled
surface-centroid point cloud and compact geometry embedding.

An optional Venturi Core STL comparison can be generated on demand for
descriptor-level context. The previous descriptor distance to the three-DrivAer
sample was:

```text
ood_distance_to_drivaerml_sample = 120.690
```

This is a descriptor-level OOD indication only. It is not a calibrated OOD
probability and not an accuracy result.

## Phase 5: 3D Scalar Active-Learning Replay

Created:

```text
AEROMAP_3D_OPEN_CFD_SCALAR_BRIDGE
docs/evidence/aeromap3d/drivaerml_scalar_bridge_replay.json
```

Configuration:

```text
config = configs/benchmark/aeromap_3d_bridge.yaml
initial_labels = 32
acquisition_batch = 16
max_labels = 128
replay_seeds = 20260629, 20260630, 20260631
methods = random, diversity, uncertainty, uncertainty_plus_diversity,
          engineering_decision_utility_v2_regret_aware
```

### Map-Completion Split

| Method | C_D RMSE | C_L RMSE | Top-k | Pareto | Spearman | Regret |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.017030 | 0.036747 | 0.638889 | 0.462963 | 0.871980 | 0.095782 |
| diversity | **0.016506** | **0.035566** | **0.722222** | 0.392593 | **0.886506** | 0.137813 |
| uncertainty | 0.017206 | 0.035914 | **0.722222** | 0.429630 | 0.869496 | 0.117040 |
| uncertainty_plus_diversity | 0.017470 | 0.039249 | 0.638889 | **0.540741** | 0.855054 | 0.156579 |
| utility v2 regret-aware | 0.020250 | 0.039077 | 0.666667 | 0.362963 | 0.859649 | **0.081781** |

Interpretation:

- diversity is strongest on drag RMSE, lift RMSE, top-k overlap and ranking;
- uncertainty-plus-diversity is strongest on Pareto recall;
- v2 is strongest on best-design regret.

### Geometry-Heldout Split

| Method | C_D RMSE | C_L RMSE | Top-k | Pareto | Spearman | Regret |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.019208 | 0.044236 | 0.527778 | 0.400000 | 0.840916 | 0.106066 |
| diversity | 0.018364 | **0.039675** | **0.750000** | **0.633333** | **0.873061** | **0.000000** |
| uncertainty | 0.018491 | 0.040942 | 0.666667 | **0.633333** | 0.856429 | 0.076195 |
| uncertainty_plus_diversity | **0.017638** | 0.042219 | 0.583333 | 0.500000 | 0.855180 | 0.118990 |
| utility v2 regret-aware | 0.019792 | 0.046471 | 0.527778 | 0.566667 | 0.819810 | 0.038517 |

Interpretation:

- diversity is currently the best recommended DrivAerML bridge acquisition
  policy for broad design-decision quality;
- uncertainty-plus-diversity wins drag RMSE;
- v2 improves regret versus random but does not beat diversity;
- therefore this bridge proves practicability, not a new custom-method
  leadership claim.

## Practicability Answers

1. AeroMap now has a plausible path beyond 2D AirfRANS: compact 3D automotive
   CFD scalar metadata can be ingested, replayed and paired with real 3D STL
   descriptor ingestion.
2. DrivAerML was selected because compact all-case geometry and force/moment
   summaries were available and it is the most vehicle-relevant candidate.
3. The 3D geometry evidence is three real DrivAerML STLs with about 753k
   triangles each, with an optional generated Venturi Core descriptor
   comparison.
4. The scalar replay ran on 484 cases. Diversity is recommended for this bridge;
   v2 is not the winner.
5. Missing for real motorsport wing/floor analysis: F1-relevant wing/floor data,
   validated custom Core/3D targets, surface/field targets, component metrics,
   live CFD coupling and calibrated simulation-cost models.
6. Inside an F1 team, this would need validated internal CFD labels, real CAD
   design variables, component IDs, cost estimates, uncertainty calibration,
   label governance and scheduler/post-processing integration.

## Portfolio Positioning

The updated positioning should be:

1. AeroMap v0.3 proves simulation-budget learning on AirfRANS.
2. AeroMap-3D bridge proves compact 3D aero scalar practicability on DrivAerML.
3. Venturi Core remains the custom underfloor benchmark tier, with a future
   full-3D extension lane.

Do not lead with the DrivAerML bridge as the headline. It is a credibility
upgrade and a practical answer to "does this go beyond 2D?", not the strongest
decision-policy result.

## Commands

```sh
uv run aeromap benchmark aeromap-3d-triage \
  --out docs/evidence/aeromap3d/metadata_triage.json

uv run aeromap benchmark aeromap-3d-drivaerml-scalars \
  --cache-dir artifacts/benchmark/aeromap3d/drivaerml \
  --out docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.json

uv run aeromap benchmark aeromap-decision-replay-v03 \
  --config configs/benchmark/aeromap_3d_bridge.yaml \
  --dataset-npz docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.npz \
  --out docs/evidence/aeromap3d/drivaerml_scalar_bridge_replay.json \
  --svg-dir docs/evidence/aeromap3d
```

The committed geometry-readiness sample is retained as evidence. Regenerating it
requires explicit local STL paths from a DrivAerML sample cache; the public
clean-checkout path does not assume those bulky STL files are present.

## Release Scope

Included:

- compact DrivAerML 3D scalar metadata bridge;
- offline active-learning replay on 3D automotive open-CFD scalar labels;
- real 3D STL ingestion readiness sample;
- diversity currently recommended for the DrivAerML bridge.

Extension paths:

- richer field-level prediction;
- measured solver-cost savings;
- full custom Venturi Core or 3D transfer;
- extra 3D datasets;
- acquisition retuning on a dedicated calibration split.
