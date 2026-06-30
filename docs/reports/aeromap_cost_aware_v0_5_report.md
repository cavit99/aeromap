# AeroMap v0.5 Cost-Aware Acquisition and 3D Field Feasibility

## Classification

`AEROMAP_V0_5_COST_PROXY_AWARE_REPLAY` and `AEROMAP_3D_SURFACE_FIELD_READINESS_SAMPLE`.

## Claim Boundaries

- No CFD was run.
- No EC2, NIM, cloud or new dataset download was used.
- Cost values are proxies, not measured live solver savings.
- DrivAerML boundary fields were inspected only because they were already cached locally.
- No AeroCliff, F1, DoMINO accuracy or field-prediction claim is made.

## Cost Proxy Sources

- AirfRANS: `real_observed_cost_proxy` from local AirfRANS internal.vtu file size.
- DrivAerML: `derived_complexity_proxy` from compact geometry descriptors.

## Geometry-Heldout Winners

AirfRANS final decision winners:

```json
{
  "best_design_regret": [
    "engineering_decision_utility_v2_cost_aware"
  ],
  "pareto_recall": [
    "engineering_decision_utility_v2_regret_aware"
  ],
  "rmse_cd": [
    "engineering_decision_utility_v2_regret_aware"
  ],
  "rmse_cl": [
    "diversity"
  ],
  "spearman_efficiency": [
    "engineering_decision_utility_v2_regret_aware"
  ],
  "top_k_efficiency_overlap": [
    "engineering_decision_utility_v2_regret_aware"
  ]
}
```

AirfRANS cost-normalised winners:

```json
{
  "cumulative_cost_proxy": [
    "engineering_decision_utility_v2_cost_aware"
  ],
  "pareto_recall_per_cost_proxy": [
    "engineering_decision_utility_v2_regret_aware"
  ],
  "regret_per_cost_proxy": [
    "engineering_decision_utility_v2_cost_aware"
  ],
  "rmse_cd_per_cost_proxy": [
    "engineering_decision_utility_v2_regret_aware"
  ],
  "rmse_cl_per_cost_proxy": [
    "diversity"
  ],
  "top_k_overlap_per_cost_proxy": [
    "engineering_decision_utility_v2_regret_aware"
  ]
}
```

DrivAerML compact 3D scalar final decision winners:

```json
{
  "best_design_regret": [
    "diversity"
  ],
  "pareto_recall": [
    "diversity",
    "uncertainty"
  ],
  "rmse_cd": [
    "uncertainty_plus_diversity"
  ],
  "rmse_cl": [
    "diversity"
  ],
  "spearman_efficiency": [
    "diversity"
  ],
  "top_k_efficiency_overlap": [
    "diversity"
  ]
}
```

DrivAerML compact 3D scalar cost-normalised winners:

```json
{
  "cumulative_cost_proxy": [
    "engineering_decision_utility_v2_cost_aware"
  ],
  "pareto_recall_per_cost_proxy": [
    "uncertainty"
  ],
  "regret_per_cost_proxy": [
    "diversity"
  ],
  "rmse_cd_per_cost_proxy": [
    "uncertainty_plus_diversity"
  ],
  "rmse_cl_per_cost_proxy": [
    "diversity"
  ],
  "top_k_overlap_per_cost_proxy": [
    "diversity"
  ]
}
```

## 3D Surface-Field Feasibility

- Classification: `AEROMAP_3D_SURFACE_FIELD_READINESS_SAMPLE`.
- Cached boundary VTPs found: 24.
- Cached boundary VTPs inspected: 1.

This proves local 3D surface-field ingestion/readiness only. It is not a surface-field prediction benchmark.
