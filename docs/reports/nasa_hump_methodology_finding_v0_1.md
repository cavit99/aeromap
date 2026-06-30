# NASA Hump Methodology Finding v0.1

## Question

Can AeroMap run a recognised separated-flow CFD validation case through a local
OpenFOAM methodology pipeline and produce comparable `C_p(x)` / `C_f(x)`
evidence?

## What Passed

- NASA/TMR wall-mounted-hump reference data were ingested.
- Experimental and published CFL3D SA/SST `C_p` and `C_f` curves were parsed.
- Official NASA/TMR no-plenum PLOT3D grids were converted into OpenFOAM.
- The boundary patches were split into front/back, inlet, outlet, top and
  hump-wall patches.
- A separate methodology mesh gate was defined for highly stretched official
  boundary-layer validation grids without weakening the global AeroMap mesh gate.
- A bounded `103 x 28` OpenFOAM SST smoke run completed and exported hump-wall
  pressure and wall-shear fields.
- The smoke-grid wall fields were converted into NASA/TMR-style `C_p(x)` and
  `C_f(x)` overlays.
- The official `409 x 109` grid ran through the same OpenFOAM SST pipeline.
- Wall-tangent-projected `C_f` was implemented for the medium-grid candidate,
  with global-x `C_f` retained only as a diagnostic.

## What Blocked Model Comparison

The `409 x 109` SST candidate completed mechanically, but it did not pass the
minimum separated-flow plausibility gate for a NASA hump model-comparison branch.

Key evidence:

| Check | Result |
|---|---:|
| Grid cells | 44,064 |
| `foamRun` final time | 200 iterations |
| Non-finite residuals | none detected |
| Final local continuity | `2.36563e-05` |
| Wall-tangent `C_f` zero crossings | none |
| Experiment `C_p` RMSE | 0.54268 |
| Experiment `C_f` RMSE | 0.012280 |

For this case, the absence of a `C_f` zero crossing is not a complete validation
metric, but it is a first blocker: the current SST setup is not reproducing the
separated-flow signature that makes the NASA hump useful for turbulence-model
correlation.

## Engineering Interpretation

The methodology pipeline works, but the current OpenFOAM SST setup is not yet a
valid basis for turbulence-model comparison. Running an SA case or ranking SA
against SST from this baseline would produce more plots, not better evidence.

The next real investigation would be CFD-methodology work, not model comparison:

- verify the NASA/TMR boundary-condition details against the OpenFOAM setup;
- inspect patch assignment and top-boundary treatment;
- audit pressure reference and outlet treatment;
- check inlet turbulence quantities and wall treatment;
- inspect `y+` and near-wall resolution on the hump wall;
- compare wall-tangent direction and wall-shear sign conventions;
- revisit numerical schemes or run length only after the setup is physically
  consistent.

## Decision

Stop the public NASA hump lane at this methodology finding.

Do not claim:

- NASA hump validation accuracy;
- grid convergence;
- SA/SST turbulence-model recommendation;
- production CFD methodology;
- F1-specific accuracy.

Allowed claim:

> AeroMap includes a reproducible NASA/TMR hump methodology pipeline through
> reference ingestion, OpenFOAM grid conversion, SST smoke and medium-grid
> candidate runs, wall-field `C_p/C_f` extraction and an evidence-based decision
> that the current SST setup is not correlation-plausible enough for turbulence
> model comparison.

## Evidence

- Smoke run report: `docs/reports/nasa_hump_sst_smoke_v0_1.md`
- Smoke-grid Cp/Cf report: `docs/reports/nasa_hump_cp_cf_extraction_v0_1.md`
- Medium-grid SST report: `docs/reports/nasa_hump_medium_grid_sst_v0_1.md`
- Medium-grid evidence: `docs/evidence/methodology/nasa_hump_medium_grid_sst_v0_1.json`
