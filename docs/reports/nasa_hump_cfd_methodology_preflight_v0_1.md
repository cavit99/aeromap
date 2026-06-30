# NASA Hump CFD Methodology Preflight v0.1

## Question

Can AeroMap add a compact CFD-methodology slice around a recognised separated-flow validation case without turning the public repo into a large CFD validation framework?

## Current Answer

Yes for preflight. The NASA/TMR wall-mounted hump gives a compact separated-flow validation target with experimental data and published SA/SST reference curves. The small no-plenum grid can enter the OpenFOAM workflow, but it is not correlation-eligible yet.

## Case

- Source: NASA/TMR 2D wall-mounted hump, no flow control.
- Validation focus: smooth-body separation, reattachment and recovery.
- Reynolds number: `936000`.
- Reference data: experimental `Cp` and `Cf` curves.
- Published CFD comparison data: CFL3D SA and SST curves on the no-plenum grid.
- Small grid inspected: `hump2newtop_noplenumZ103x28.p2dfmt.gz`.

## Reference Correlation Metrics

These numbers compare NASA/TMR-published CFL3D curves against the experimental curves. They validate the correlation-metric plumbing; they are not AeroMap/OpenFOAM results.

| Published curve | Cp RMSE | Cp MAE | Cf RMSE | Cf MAE |
|---|---:|---:|---:|---:|
| CFL3D SA | 0.02842 | 0.01947 | 0.000897 | 0.000704 |
| CFL3D SST | 0.04189 | 0.02818 | 0.000834 | 0.000658 |

## OpenFOAM Ingest Smoke

- `plot3dToFoam -noBlank -2D 0.1` reads the `103 x 28` grid correctly.
- Prototype patch split produced `front/back/inlet/outlet/top_slip/hump_wall`.
- Converted mesh cells: `2754`.
- Boundary patches after split: `6`.
- Failed mesh checks: `2`.
- Max aspect ratio: `19660.6`.
- Small-determinant cells: `382`.
- Methodology gate: `accepted_with_methodology_warning`.

## Mesh Policy

Do not weaken the global AeroMap mesh gate. The NASA/TMR hump uses a separate methodology gate for official boundary-layer validation grids. The `103 x 28` grid is accepted only for conversion, boundary-condition and single-solver smoke work with explicit warnings. It is not accepted for headline correlation or turbulence-model recommendation.

| Gate result | Status |
|---|---|
| Global AeroMap gate | `False` |
| NASA/TMR methodology gate | `True` |
| Mesh quality class | `accepted_with_methodology_warning` |

## Correlation Eligibility

| Requirement | Status |
|---|---|
| `experimental_cp_cf_parsed` | Pass |
| `published_cfl3d_sa_sst_references_parsed` | Pass |
| `plot3d_grid_converted` | Pass |
| `patch_split_audited` | Pass |
| `methodology_mesh_gate_defined` | Pass |
| `openfoam_sst_setup_generated` | Pass: single-grid smoke setup |
| `solver_run_completed` | Pass: single-grid smoke only |
| `cp_cf_extracted_from_openfoam` | Pass: smoke-grid overlay only |
| `openfoam_vs_experiment_compared` | Pass: smoke-grid overlay metrics only |
| `medium_grid_sst_candidate_checked` | Pass: 409 x 109 candidate not correlation-plausible |
| `grid_sensitivity_checked` | Not yet |

## Next Step

The follow-up SST smoke and Cp/Cf extraction artifacts show that the OpenFOAM wall-field overlay pipeline works. The 409 x 109 SST candidate now tests the next methodology question and is not correlation-plausible yet, so a model recommendation should wait for boundary-condition, grid or numerics improvements.

## Claim Boundary

- Established: NASA/TMR reference ingestion, correlation metrics and OpenFOAM grid-ingest feasibility.
- Not established: OpenFOAM hump correlation, SA/SST recommendation, production CFD accuracy.

## Evidence

- JSON: `docs/evidence/methodology/nasa_hump_methodology_preflight_v0_1.json`
