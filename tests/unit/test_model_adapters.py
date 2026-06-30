from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from aeromap.models.adapters import NimPredictorAdapter, PhysicsNeMoCorrector
from aeromap.models.core import (
    CorrectorCoreConfig,
    ResidualCorrectorCore,
    corrector_input_sensitivity,
    load_coefficient_loss,
    load_device_neutral_checkpoint,
    physics_residual_loss,
    residual_mse_loss,
    save_device_neutral_checkpoint,
    surface_load_coefficients,
)


def test_nim_predictor_refuses_training(tmp_path: Path) -> None:
    adapter = NimPredictorAdapter(
        endpoint="http://localhost:8000",
        cache_dir=tmp_path,
        image_ref="nvcr.io/nim/nvidia/domino-automotive-aero:2.1.0-41313772",
    )
    with pytest.raises(RuntimeError, match="frozen predictor"):
        adapter.train()


def test_corrector_requires_cuda_for_training_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_resolve_device(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        message = "PhysicsNeMo/DoMINO requires a Linux NVIDIA CUDA host."
        raise RuntimeError(message)

    monkeypatch.setattr("aeromap.models.adapters.resolve_device", fake_resolve_device)
    corrector = PhysicsNeMoCorrector(config_path=tmp_path / "corrector.yaml")
    with pytest.raises(RuntimeError, match="requires a Linux NVIDIA CUDA host"):
        corrector.train_step()
    assert calls == [(("auto",), {"require_physicsnemo": True})]


def _run_forward_backward(device: str) -> ResidualCorrectorCore:
    torch.manual_seed(7)
    config = CorrectorCoreConfig(input_dim=6, output_dim=4, hidden_dim=16, depth=2)
    model = ResidualCorrectorCore(config).to(device)
    features = torch.randn(9, config.input_dim, device=device)
    target = torch.randn(9, config.output_dim, device=device)

    prediction = model(features)
    residual = prediction - target
    loss = (
        residual_mse_loss(prediction, target)
        + 0.1 * load_coefficient_loss(prediction[:, :2], target[:, :2])
        + 0.1 * physics_residual_loss({"project_residual": residual})
    )
    loss.backward()  # type: ignore[no-untyped-call]

    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
    return model


def test_corrector_core_cpu_forward_backward_smoke() -> None:
    _run_forward_backward("cpu")


def test_residual_loss_accepts_sample_column_mask() -> None:
    prediction = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[1.0], [0.0]])

    loss = residual_mse_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(float((prediction[0].square()).mean()))


def test_residual_loss_rejects_ambiguous_mask_shape() -> None:
    prediction = torch.zeros((2, 3))
    target = torch.zeros_like(prediction)
    mask = torch.ones((3, 1))

    with pytest.raises(ValueError, match="mask must have shape"):
        residual_mse_loss(prediction, target, mask)


def test_surface_load_coefficients_match_wall_force_sign_convention() -> None:
    loads = surface_load_coefficients(
        surface_cp=torch.tensor([2.0]),
        surface_cf=torch.tensor([[-0.5, 0.0, 0.25]]),
        surface_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        surface_area_m2=torch.tensor([1.0]),
        a_ref_m2=2.0,
    )

    assert loads.pressure_xyz.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert loads.viscous_xyz.tolist() == pytest.approx([0.25, 0.0, -0.125])
    assert loads.total_xyz.tolist() == pytest.approx([0.25, 0.0, 0.875])
    assert loads.drag.item() == pytest.approx(-0.25)
    assert loads.downforce.item() == pytest.approx(-0.875)


def test_physics_residual_loss_rejects_unknown_weight() -> None:
    with pytest.raises(ValueError, match="unknown residuals"):
        physics_residual_loss(
            {"continuity": torch.ones(4)},
            weights={"momentum": 0.5},
        )


def test_corrector_input_sensitivity_returns_feature_gradients() -> None:
    torch.manual_seed(11)
    config = CorrectorCoreConfig(input_dim=3, output_dim=2, hidden_dim=8)
    model = ResidualCorrectorCore(config)
    features = torch.randn(5, config.input_dim)
    weights = torch.ones(5, config.output_dim)

    result = corrector_input_sensitivity(model, features, output_weights=weights)

    assert result.prediction.shape == (5, config.output_dim)
    assert result.input_gradient.shape == features.shape
    assert torch.all(torch.isfinite(result.input_gradient))


def test_corrector_core_mps_forward_backward_smoke_when_available() -> None:
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this host")

    _run_forward_backward("mps")


def test_corrector_checkpoint_round_trip_is_device_neutral(tmp_path: Path) -> None:
    model = _run_forward_backward("cpu")
    checkpoint = tmp_path / "corrector.pt"

    save_device_neutral_checkpoint(
        checkpoint,
        model=model,
        metadata={"sample_schema": "aeromap_sample_v0.2.0"},
    )
    restored, metadata = load_device_neutral_checkpoint(checkpoint)

    assert metadata == {"sample_schema": "aeromap_sample_v0.2.0"}
    assert restored.config == model.config
    for original, reloaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert original.device.type == "cpu"
        assert reloaded.device.type == "cpu"
        assert torch.allclose(original, reloaded)


def test_cpu_mps_config_is_project_owned_not_domino_cuda_evidence() -> None:
    payload = yaml.safe_load(
        Path("configs/model/corrector_core_smoke.yaml").read_text(encoding="utf-8"),
    )

    assert payload["devices"]["required"] == ["cpu"]
    assert payload["devices"]["optional"] == ["mps"]
    assert (
        payload["devices"]["claim_scope"] == "project_owned_pytorch_only_not_domino_cuda_evidence"
    )
