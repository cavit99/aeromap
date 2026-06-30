from __future__ import annotations

from pathlib import Path

import numpy as np

from aeromap.benchmarks.aeromap import AeroMapConfig, DatasetArrays
from aeromap.benchmarks.aeromap_cost import (
    COST_AWARE_METHOD,
    CostProxy,
    run_cost_aware_decision_replay_v05,
    write_surface_field_feasibility_precheck,
)


def _tiny_dataset(case_count: int = 64) -> DatasetArrays:
    rng = np.random.default_rng(20260629)
    x = rng.normal(size=(case_count, 4)).astype(np.float64)
    cd = 0.02 + 0.003 * x[:, 0] ** 2 + 0.001 * x[:, 1]
    cl = 0.4 + 0.04 * x[:, 2] - 0.02 * x[:, 3]
    return DatasetArrays(
        case_ids=[f"case_{idx:03d}" for idx in range(case_count)],
        features=x,
        targets=np.stack([cd, cl], axis=1).astype(np.float64),
        feature_names=["geom_x0", "geom_x1", "aoa", "reynolds_proxy"],
        target_names=["integrated_cd", "integrated_cl"],
        classification="TEST_AEROMAP_DATASET",
        open_cfd_result=True,
        group_ids=[f"group_{idx:03d}" for idx in range(case_count)],
    )


def test_cost_aware_replay_reports_proxy_metrics() -> None:
    dataset = _tiny_dataset()
    config = AeroMapConfig(
        name="test_cost_aware",
        fixture_case_count=64,
        initial_labels=8,
        acquisition_batch=4,
        max_labels=16,
        test_fraction=0.25,
        ensemble_members=2,
        replay_seeds=(1, 2),
        acquisition_methods=("random", COST_AWARE_METHOD),
    )
    costs = np.linspace(0.75, 1.5, dataset.features.shape[0], dtype=np.float64)
    proxy = CostProxy(
        values=costs,
        kind="simulated_cost_proxy",
        source="unit test",
        description="unit-test-only positive costs",
        available_for_all_cases=True,
        evidence={"case_count": int(costs.shape[0])},
    )

    report = run_cost_aware_decision_replay_v05(
        dataset,
        config,
        proxy,
        split_modes=("geometry_heldout",),
    )

    split = report["split_reports"]["geometry_heldout"]
    assert report["classification"] == "AEROMAP_V0_5_COST_PROXY_AWARE_REPLAY"
    assert report["claim_boundary"]["live_cfd_savings"] is False
    assert COST_AWARE_METHOD in split["final_metrics_by_method"]
    assert "cumulative_cost_proxy" in split["final_metrics_by_method"][COST_AWARE_METHOD]
    assert "regret_per_cost_proxy" in split["final_metrics_by_method"][COST_AWARE_METHOD]


def test_surface_field_precheck_handles_empty_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    visual_dir = tmp_path / "visuals"
    cache_root.mkdir()

    out = write_surface_field_feasibility_precheck(
        drivaerml_cache_root=cache_root,
        out=tmp_path / "surface.json",
        visual_dir=visual_dir,
        max_cases=1,
    )

    payload = out.read_text(encoding="utf-8")
    assert "AEROMAP_3D_SURFACE_FIELD_METADATA_ONLY_PRECHECK" in payload
    assert '"new_downloads": false' in payload
