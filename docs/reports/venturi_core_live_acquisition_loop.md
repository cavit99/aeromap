# Venturi Core live acquisition loop

## Executive summary

This report records the first minimal live/replay Mission Control loop on the
structured Venturi Core pressure/load response map.

The loop starts with three labelled Core cases, fits a lightweight response
surrogate, selects the next case, ingests the committed Core evidence, and
updates the map metrics. It proves the local acquisition workflow without
claiming live industrial CFD savings.

Classification: `AEROMAP_LIVE_CORE_ACQUISITION_LOOP_V0_1`

## Loop

```text
labelled Core cases
        -> response surrogate
        -> acquisition policy
        -> selected Core simulation
        -> committed evidence ingestion or local OpenFOAM run
        -> updated pressure/load map
```

Mode requested: `replay-live`
Mode executed: `replay-live`
Live execution status: existing_committed_core_evidence_reused; no OpenFOAM case was regenerated

## Initial labelled set

- `50mm/3deg` (`venturi_core_653f895037de3465`)
- `60mm/5deg` (`venturi_core_5b33557f601a1c7b`)
- `70mm/7deg` (`venturi_core_1299ced75c5bace8`)

## Primary selections

Primary policy: `engineering_utility`

| Iteration | Selected case | Reason |
|---:|---|---|
| 1 | 50mm/7deg | balances design-space coverage, response-gradient proxy and high-suction relevance |
| 2 | 50mm/5deg | balances design-space coverage, response-gradient proxy and high-suction relevance |
| 3 | 70mm/3deg | balances design-space coverage, response-gradient proxy and high-suction relevance |
| 4 | 60mm/3deg | balances design-space coverage, response-gradient proxy and high-suction relevance |

## Metrics

| Method | Curve-error area | Final normalised RMSE | Final C_D RMSE | Final suction RMSE | Final pressure-recovery RMSE |
|---|---:|---:|---:|---:|---:|
| random | 0.908394 | 0.204808 | 0.002809 | 0.053868 | 0.049559 |
| diversity | 0.741718 | 0.144426 | 0.001513 | 0.041217 | 0.038413 |
| engineering_utility | 0.747198 | 0.142798 | 0.001460 | 0.039260 | 0.039705 |
| cost_aware_utility | 0.747198 | 0.142798 | 0.001460 | 0.039260 | 0.039705 |

Best method by curve-error area:
`diversity`

## Claim boundary

Allowed:

- local Core live/replay acquisition loop;
- pressure/load response mapping;
- OpenFOAM result ingestion when selected cases have to be generated locally.

Not claimed:

- field-level surrogate;
- wall-shear or continuous separation-fraction labels;
- validated cliff boundary;
- full 3D extension accuracy;
- F1 floor accuracy;
- external predictor accuracy;
- industrial live CFD savings.

## Artifacts

- Manifest: `docs/evidence/cfd/venturi_core/live_core_loop_v0_1/live_core_loop_manifest.json`
- Learning curve: `docs/evidence/cfd/venturi_core/live_core_loop_v0_1/live_core_loop_learning_curve.svg`
