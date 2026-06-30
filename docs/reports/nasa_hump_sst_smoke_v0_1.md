# NASA Hump SST Smoke v0.1

## Result

A bounded OpenFOAM v13 SST smoke run now completes locally on the NASA/TMR wall-mounted hump `103 x 28` no-plenum grid after potential-flow initialisation.

| Item | Status |
|---|---|
| Classification | `OPENFOAM_NASA_HUMP_SST_SMOKE_V0_1` |
| Solver smoke passed | `True` |
| Final iteration | `80.0` |
| Non-finite residuals detected | `False` |
| Final local continuity | `0.00012311464` |
| Hump-wall VTK cells | `102` |

## Setup

- Solver: OpenFOAM Foundation v13 `incompressibleFluid`.
- Turbulence model: `kOmegaSST`.
- Initialisation: `potentialFoam` before `foamRun`.
- Reynolds number: `936000` with `U_inf = 1`, chord `1`, `nu = 1 / 936000`.
- Grid: NASA/TMR no-plenum `103 x 28` PLOT3D grid converted with `plot3dToFoam`.

## Mesh Gate

The official tiny boundary-layer grid is still treated under the NASA/TMR methodology gate, not the global AeroMap production mesh gate.

- Cells: `2754`.
- Boundary patches: `6`.
- Failed mesh checks: `2`.
- Max aspect ratio: `19660.642`.
- Determinant-warning faces: `3054`.
- Mesh policy: `accepted_with_methodology_warning`.

## Field Export

- Hump-wall pressure range: `-1.14653` to `0.34513`.
- Hump-wall wall-shear-stress x range: `-0.00723225` to `-0.000127333`.
- `foamPostProcess -solver incompressibleFluid -func wallShearStress` min/max: `[-0.007232248, -0.0014768575, -4.0577909e-19]` to `[-0.00012733299, 0.0026129569, 4.1588951e-19]`.

## Claim Boundary

- Established: local OpenFOAM SST smoke execution and hump-wall field export.
- Not established: NASA correlation, turbulence-model recommendation, grid convergence or production CFD accuracy.

## Evidence

- JSON: `docs/evidence/methodology/nasa_hump_sst_smoke_v0_1.json`
- Generated case directory: `artifacts/methodology/nasa_hump/sst_smoke_case`
