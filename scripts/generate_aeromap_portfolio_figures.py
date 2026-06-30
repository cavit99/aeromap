"""Generate AeroMap Mission Control portfolio figures from v0.3 evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/aeromap/airfrans_decision_replay_v03.json"
DRIVAER3D = ROOT / "docs/evidence/aeromap3d/drivaerml_scalar_bridge_replay.json"
CORE_2D_DATASET = ROOT / "docs/evidence/cfd/aerocliff_core/core_2d_response_map_dataset_v0.json"
CORE_2D_REPLAY = (
    ROOT / "docs/evidence/cfd/aerocliff_core/core_2d_response_map_active_replay_v0.json"
)
OUT_DIR = ROOT / "docs/assets/aeromap"
MANIFEST = OUT_DIR / "figures_manifest.json"

FIGURES = {
    "headline": "aeromap_headline_geometry_heldout.png",
    "learning": "label_budget_learning_curves.png",
    "comparison": "acquisition_method_comparison.png",
    "decision": "decision_metrics_panel.png",
    "flow": "mission_control_flow.png",
    "two_tier": "aeromap_two_tier_evidence.png",
    "drivaer3d": "drivaerml_3d_bridge_metrics.png",
    "core_surface": "aerocliff_core_response_surface.png",
    "core_recovery": "aerocliff_core_pressure_recovery_surface.png",
    "core_replay": "aerocliff_core_active_replay.png",
    "core_story": "aeromap_aerocliff_core_story.png",
}

METHOD_LABELS = {
    "random": "Random",
    "diversity": "Diversity",
    "uncertainty": "Uncertainty",
    "uncertainty_plus_diversity": "Uncertainty + Diversity",
    "engineering_decision_utility_v1": "Utility v1",
    "engineering_decision_utility_v2_regret_aware": "Regret-aware Utility v2",
    "diversity_space_filling": "Diversity",
    "engineering_utility": "Engineering utility",
    "cost_aware_utility": "Cost-aware utility",
}

PALETTE = {
    "random": "#6b7280",
    "diversity": "#2563eb",
    "uncertainty": "#d97706",
    "uncertainty_plus_diversity": "#059669",
    "engineering_decision_utility_v1": "#be185d",
    "engineering_decision_utility_v2_regret_aware": "#5b21b6",
    "diversity_space_filling": "#2563eb",
    "engineering_utility": "#5b21b6",
    "cost_aware_utility": "#059669",
}

TEXT = "#172026"
MUTED = "#5b6670"
GRID = "#d8dee5"
PANEL = "#f6f8fb"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4af")
    ax.spines["bottom"].set_color("#9aa4af")
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)


def title(fig: plt.Figure, main: str, sub: str) -> None:
    fig.text(0.04, 0.955, main, fontsize=24, fontweight="bold", color=TEXT)
    fig.text(0.04, 0.922, sub, fontsize=11, color=MUTED)


def final_metrics(data: dict[str, Any], split: str) -> dict[str, dict[str, float]]:
    return data["split_reports"][split]["final_metrics_by_method"]


def records(data: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return data["split_reports"][split]["records"]


def figure_headline(data: dict[str, Any]) -> Path:
    metrics = final_metrics(data, "geometry_heldout")
    methods = [
        "random",
        "diversity",
        "uncertainty_plus_diversity",
        "engineering_decision_utility_v1",
        "engineering_decision_utility_v2_regret_aware",
    ]
    short_labels = {
        "random": "Random",
        "diversity": "Diversity",
        "uncertainty_plus_diversity": "Unc + div",
        "engineering_decision_utility_v1": "Utility v1",
        "engineering_decision_utility_v2_regret_aware": "Utility v2",
    }
    metric_specs = [
        ("rmse_cd", "C_D RMSE", True),
        ("rmse_cl", "C_L RMSE", True),
        ("top_k_efficiency_overlap", "Top-k overlap", False),
        ("pareto_recall", "Pareto recall", False),
        ("spearman_efficiency", "Rank correlation", False),
        ("best_design_regret", "Best-design regret", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=False)
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, wspace=0.42, hspace=0.42)
    title(
        fig,
        "AeroMap Mission Control v0.3: geometry-disjoint decision benchmark",
        (
            "Real AirfRANS CFD, 1000 cases. Lower is better for RMSE/regret; "
            "higher is better for decision metrics."
        ),
    )
    for ax, (metric, label, lower_better) in zip(axes.ravel(), metric_specs, strict=True):
        values = [metrics[method][metric] for method in methods]
        colors = [PALETTE[method] for method in methods]
        y_pos = np.arange(len(methods))
        ax.barh(y_pos, values, color=colors, height=0.64)
        style_axes(ax)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_yticks(y_pos, [short_labels[m] for m in methods], fontsize=8)
        best_idx = int(np.argmin(values) if lower_better else np.argmax(values))
        ax.barh(
            best_idx,
            values[best_idx],
            color=colors[best_idx],
            edgecolor="#111827",
            linewidth=1.5,
            height=0.64,
        )
        ax.invert_yaxis()
        for y, value in zip(y_pos, values, strict=True):
            ax.text(value, y, f" {value:.3g}", va="center", fontsize=8, color=MUTED)
    fig.text(
        0.04,
        0.035,
        (
            "Headline: v2 leads C_D RMSE, top-k, Pareto and rank correlation. "
            "Caveat: v1 leads C_L RMSE; diversity leads absolute regret."
        ),
        fontsize=10,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["headline"]
    save(fig, path)
    return path


def figure_learning_curves(data: dict[str, Any]) -> Path:
    recs = records(data, "geometry_heldout")
    metrics = [
        ("rmse_cd", "C_D RMSE", "lower better"),
        ("rmse_cl", "C_L RMSE", "lower better"),
        ("top_k_efficiency_overlap", "Top-k overlap", "higher better"),
        ("pareto_recall", "Pareto recall", "higher better"),
    ]
    methods = [
        "random",
        "diversity",
        "uncertainty_plus_diversity",
        "engineering_decision_utility_v1",
        "engineering_decision_utility_v2_regret_aware",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=False)
    fig.subplots_adjust(top=0.82, left=0.07, right=0.98, wspace=0.22, hspace=0.34)
    title(
        fig,
        "Label-budget learning curves",
        (
            "Offline pool replay: each curve shows mean performance across "
            "five deterministic acquisition seeds."
        ),
    )
    for ax, (metric, label, direction) in zip(axes.ravel(), metrics, strict=True):
        for method in methods:
            rows = sorted(
                [row for row in recs if row["method"] == method],
                key=lambda row: int(row["label_count"]),
            )
            ax.plot(
                [row["label_count"] for row in rows],
                [row[metric] for row in rows],
                color=PALETTE[method],
                linewidth=2.6 if method.endswith("v2_regret_aware") else 1.9,
                marker="o" if method.endswith("v2_regret_aware") else None,
                markersize=4,
                label=METHOD_LABELS[method],
            )
        style_axes(ax)
        ax.set_title(
            f"{label} ({direction})", loc="left", fontsize=12, fontweight="bold", color=TEXT
        )
        ax.set_xlabel("Labelled CFD cases", fontsize=10, color=MUTED)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper right")
    path = OUT_DIR / FIGURES["learning"]
    save(fig, path)
    return path


def figure_acquisition_comparison(data: dict[str, Any]) -> Path:
    metrics = final_metrics(data, "geometry_heldout")
    methods = list(metrics)
    score_metrics = [
        ("rmse_cd", True),
        ("rmse_cl", True),
        ("top_k_efficiency_overlap", False),
        ("pareto_recall", False),
        ("spearman_efficiency", False),
        ("best_design_regret", True),
    ]
    scores = {}
    for method in methods:
        method_scores = []
        for metric, lower_better in score_metrics:
            vals = np.array([metrics[m][metric] for m in methods], dtype=float)
            lo, hi = float(vals.min()), float(vals.max())
            raw = metrics[method][metric]
            norm = 0.5 if np.isclose(lo, hi) else (raw - lo) / (hi - lo)
            method_scores.append(1.0 - norm if lower_better else norm)
        scores[method] = float(np.mean(method_scores))
    ordered = sorted(methods, key=lambda method: scores[method])
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(top=0.78, left=0.27, right=0.96, bottom=0.12)
    title(
        fig,
        "Acquisition method comparison",
        (
            "Composite final-budget score over six geometry-disjoint metrics, "
            "scaled within this benchmark only."
        ),
    )
    ypos = np.arange(len(ordered))
    ax.barh(ypos, [scores[m] for m in ordered], color=[PALETTE[m] for m in ordered])
    ax.set_yticks(ypos, [METHOD_LABELS[m] for m in ordered], fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Composite decision score", fontsize=10, color=MUTED)
    style_axes(ax)
    for y, method in zip(ypos, ordered, strict=True):
        ax.text(
            scores[method] + 0.015, y, f"{scores[method]:.2f}", va="center", fontsize=10, color=TEXT
        )
    fig.text(
        0.04,
        0.04,
        (
            "Composite scores aid presentation only. Primary claims are "
            "metric-specific and reported in the technical report."
        ),
        fontsize=10,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["comparison"]
    save(fig, path)
    return path


def figure_decision_panel(data: dict[str, Any]) -> Path:
    metrics = final_metrics(data, "geometry_heldout")
    methods = [
        "random",
        "diversity",
        "uncertainty_plus_diversity",
        "engineering_decision_utility_v1",
        "engineering_decision_utility_v2_regret_aware",
    ]
    short_labels = ["Rnd", "Div", "U+D", "V1", "V2"]
    panel_metrics = [
        ("top_k_efficiency_overlap", "Top-k recovery", False),
        ("pareto_recall", "Pareto recall", False),
        ("spearman_efficiency", "Design ranking", False),
        ("best_design_regret", "Best-design regret", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, left=0.05, right=0.98, wspace=0.34)
    title(
        fig,
        "Decision metrics panel",
        "Aero teams care about choosing the right simulation, not only reducing average error.",
    )
    for ax, (metric, label, lower_better) in zip(axes, panel_metrics, strict=True):
        values = [metrics[method][metric] for method in methods]
        ax.bar(np.arange(len(methods)), values, color=[PALETTE[method] for method in methods])
        best = int(np.argmin(values) if lower_better else np.argmax(values))
        ax.bar(best, values[best], color=PALETTE[methods[best]], edgecolor="#111827", linewidth=1.4)
        ax.set_title(label, fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xticks(np.arange(len(methods)), short_labels, fontsize=8)
        style_axes(ax)
    fig.text(
        0.05,
        0.045,
        (
            "Method key: Rnd=random, Div=diversity, U+D=uncertainty plus "
            "diversity, V1=engineering utility, V2=regret-aware utility."
        ),
        fontsize=9,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["decision"]
    save(fig, path)
    return path


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, *, color: str = "#eef2ff") -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        0.18,
        0.18,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#c7d2fe",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.09, y + 0.09, text, ha="center", va="center", fontsize=11, color=TEXT, weight="bold"
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color="#334155",
            shrinkA=6,
            shrinkB=6,
        ),
    )


def figure_flow() -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()
    title(
        fig,
        "Mission Control flow",
        (
            "The product is a decision loop: spend the next CFD label where "
            "it most improves the aero map."
        ),
    )
    boxes = [
        ((0.05, 0.54), "Current\nlabelled set", "#f8fafc"),
        ((0.28, 0.54), "Surrogate\nmodel", "#eef2ff"),
        ((0.51, 0.54), "Uncertainty +\ndecision utility", "#ecfdf5"),
        ((0.74, 0.54), "Recommended\nnext CFD case", "#fff7ed"),
        ((0.74, 0.22), "Updated\naero map", "#fdf2f8"),
        ((0.28, 0.22), "Decision metrics:\ntop-k, Pareto,\nrank, regret", "#f8fafc"),
    ]
    for xy, label, color in boxes:
        add_box(ax, xy, label, color=color)
    arrow(ax, (0.23, 0.63), (0.28, 0.63))
    arrow(ax, (0.46, 0.63), (0.51, 0.63))
    arrow(ax, (0.69, 0.63), (0.74, 0.63))
    arrow(ax, (0.83, 0.54), (0.83, 0.40))
    arrow(ax, (0.74, 0.31), (0.46, 0.31))
    arrow(ax, (0.37, 0.40), (0.37, 0.54))
    ax.text(
        0.05,
        0.13,
        (
            "Current release uses offline replay: the same interface can be "
            "wired to live CFD scheduling later."
        ),
        fontsize=11,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["flow"]
    save(fig, path)
    return path


def figure_two_tier_evidence() -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_axis_off()
    title(
        fig,
        "AeroMap evidence tiers",
        (
            "AeroMap starts with an open-CFD decision benchmark, extends to compact "
            "3D automotive data and connects to a structured underfloor response map."
        ),
    )
    boxes = [
        (
            (0.05, 0.50),
            "Tier 1\nAirfRANS",
            (
                "1000 real RANS cases\n27 geometry-aware features\n"
                "v2 leads several geometry-disjoint\ndecision metrics"
            ),
            "#eef2ff",
        ),
        (
            (0.37, 0.50),
            "Tier 2\nDrivAerML bridge",
            (
                "484 3D automotive scalar cases\n16 geometry/design features\n"
                "3 real STLs ingested\ndiversity currently recommended"
            ),
            "#ecfdf5",
        ),
        (
            (0.69, 0.50),
            "Core\nAeroCliff",
            (
                "structured Venturi-underfloor map\n"
                "15 medium cases, 3 fine checks\ncustom pressure/load replay"
            ),
            "#fff7ed",
        ),
    ]
    for (x, y), header, body, color in boxes:
        rect = FancyBboxPatch(
            (x, y),
            0.25,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#cbd5e1",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.125, y + 0.205, header, ha="center", fontsize=17, color=TEXT, weight="bold")
        ax.text(x + 0.125, y + 0.095, body, ha="center", va="center", fontsize=10.5, color=MUTED)
    arrow(ax, (0.30, 0.64), (0.37, 0.64))
    arrow(ax, (0.62, 0.64), (0.69, 0.64))
    ax.text(
        0.05,
        0.25,
        (
            "Product lesson: Mission Control does not force one acquisition policy everywhere.\n"
            "It selects the strategy supported by the current aero domain evidence."
        ),
        fontsize=13,
        color=TEXT,
        weight="bold",
    )
    ax.text(
        0.05,
        0.17,
        (
            "The public release focuses on budgeted simulation selection and "
            "pressure/load response mapping."
        ),
        fontsize=11,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["two_tier"]
    save(fig, path)
    return path


def figure_drivaer3d_metrics(data: dict[str, Any]) -> Path:
    metrics = final_metrics(data, "geometry_heldout")
    methods = [
        "random",
        "diversity",
        "uncertainty",
        "uncertainty_plus_diversity",
        "engineering_decision_utility_v2_regret_aware",
    ]
    short_labels = {
        "random": "Random",
        "diversity": "Diversity",
        "uncertainty": "Uncertainty",
        "uncertainty_plus_diversity": "Unc + div",
        "engineering_decision_utility_v2_regret_aware": "Utility v2",
    }
    metric_specs = [
        ("rmse_cd", "C_D RMSE", True),
        ("rmse_cl", "C_L RMSE", True),
        ("top_k_efficiency_overlap", "Top-k overlap", False),
        ("pareto_recall", "Pareto recall", False),
        ("spearman_efficiency", "Rank correlation", False),
        ("best_design_regret", "Best-design regret", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=False)
    fig.subplots_adjust(top=0.82, left=0.09, right=0.98, wspace=0.42, hspace=0.42)
    title(
        fig,
        "DrivAerML compact 3D bridge: geometry-heldout replay",
        (
            "484 automotive CFD scalar cases. Diversity is currently recommended; "
            "utility v2 is not the winner on this bridge."
        ),
    )
    for ax, (metric, label, lower_better) in zip(axes.ravel(), metric_specs, strict=True):
        values = [metrics[method][metric] for method in methods]
        y_pos = np.arange(len(methods))
        ax.barh(y_pos, values, color=[PALETTE[method] for method in methods], height=0.64)
        best_idx = int(np.argmin(values) if lower_better else np.argmax(values))
        ax.barh(
            best_idx,
            values[best_idx],
            color=PALETTE[methods[best_idx]],
            edgecolor="#111827",
            linewidth=1.5,
            height=0.64,
        )
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_yticks(y_pos, [short_labels[m] for m in methods], fontsize=8)
        ax.invert_yaxis()
        style_axes(ax)
        for y, value in zip(y_pos, values, strict=True):
            ax.text(value, y, f" {value:.3g}", va="center", fontsize=8, color=MUTED)
    fig.text(
        0.04,
        0.035,
        (
            "Bridge interpretation: compact 3D automotive scalar replay works, "
            "but broad geometry coverage currently beats the custom utility policy."
        ),
        fontsize=10,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["drivaer3d"]
    save(fig, path)
    return path


def _core_grid(dataset: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cases = dataset["cases"]
    ride_heights = np.array(sorted({float(case["ride_height_mm"]) for case in cases}), dtype=float)
    angles = np.array(sorted({float(case["diffuser_angle_deg"]) for case in cases}), dtype=float)
    values = np.full((len(ride_heights), len(angles)), np.nan, dtype=float)
    ride_index = {value: idx for idx, value in enumerate(ride_heights)}
    angle_index = {value: idx for idx, value in enumerate(angles)}
    for case in cases:
        values[
            ride_index[float(case["ride_height_mm"])],
            angle_index[float(case["diffuser_angle_deg"])],
        ] = float(
            case[target],
        )
    return ride_heights, angles, values


def _annotate_heatmap(ax: plt.Axes, values: np.ndarray, *, precision: int = 2) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.{precision}f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value > np.nanmean(values) else TEXT,
                    weight="bold",
                )


def figure_core_response_surface(dataset: dict[str, Any]) -> Path:
    ride_heights, angles, values = _core_grid(dataset, "suction_downforce")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    fig.subplots_adjust(top=0.80, left=0.14, right=0.92, bottom=0.14)
    title(
        fig,
        "AeroCliff Core response surface: suction/downforce",
        (
            "Structured Core medium map: 15/15 cases passed gates. Values are "
            "pressure/load response targets for the offline replay."
        ),
    )
    image = ax.imshow(values, cmap="viridis", aspect="auto", origin="lower")
    _annotate_heatmap(ax, values, precision=2)
    ax.set_xticks(np.arange(len(angles)), [f"{angle:.0f}°" for angle in angles])
    ax.set_yticks(np.arange(len(ride_heights)), [f"{height:.0f} mm" for height in ride_heights])
    ax.set_xlabel("Diffuser angle", fontsize=11, color=MUTED)
    ax.set_ylabel("Ride height", fontsize=11, color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.03)
    cbar.set_label("Suction/downforce coefficient", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    fig.text(
        0.14,
        0.055,
        (
            "Top suction in this bounded map occurs at 60 mm / 4 deg. "
            "The map is intentionally small enough to audit case by case."
        ),
        fontsize=10,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["core_surface"]
    save(fig, path)
    return path


def figure_core_pressure_recovery(dataset: dict[str, Any]) -> Path:
    ride_heights, angles, values = _core_grid(dataset, "pressure_recovery")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    fig.subplots_adjust(top=0.80, left=0.14, right=0.92, bottom=0.14)
    title(
        fig,
        "AeroCliff Core pressure recovery map",
        (
            "Pressure recovery is one of the accepted Core pressure/load response "
            "targets used by the active replay."
        ),
    )
    image = ax.imshow(values, cmap="cividis", aspect="auto", origin="lower")
    _annotate_heatmap(ax, values, precision=2)
    ax.set_xticks(np.arange(len(angles)), [f"{angle:.0f}°" for angle in angles])
    ax.set_yticks(np.arange(len(ride_heights)), [f"{height:.0f} mm" for height in ride_heights])
    ax.set_xlabel("Diffuser angle", fontsize=11, color=MUTED)
    ax.set_ylabel("Ride height", fontsize=11, color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.03)
    cbar.set_label("Cp_exit - Cp_throat", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)
    fig.text(
        0.14,
        0.055,
        (
            "Pressure recovery is evaluated as a scalar response target across "
            "the same 3 x 5 design space."
        ),
        fontsize=10,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["core_recovery"]
    save(fig, path)
    return path


def figure_core_active_replay(replay: dict[str, Any]) -> Path:
    summary = replay["replay"]["summary_by_method"]
    methods = [
        "random",
        "diversity_space_filling",
        "uncertainty",
        "engineering_utility",
        "cost_aware_utility",
    ]
    labels = {
        "random": "Random",
        "diversity_space_filling": "Diversity",
        "uncertainty": "Uncertainty",
        "engineering_utility": "Engineering",
        "cost_aware_utility": "Cost-aware",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(top=0.78, left=0.08, right=0.98, bottom=0.18, wspace=0.26)
    title(
        fig,
        "AeroCliff Core active replay",
        (
            "Offline budgeted replay on the 3 x 5 structured Core pressure/load "
            "response map. Lower curve-error area is better."
        ),
    )
    areas = [summary[method]["area_under_normalised_rmse_mean_curve"] for method in methods]
    y_pos = np.arange(len(methods))
    axes[0].barh(y_pos, areas, color=[PALETTE[method] for method in methods], height=0.62)
    best = int(np.argmin(areas))
    axes[0].barh(
        best,
        areas[best],
        color=PALETTE[methods[best]],
        edgecolor="#111827",
        linewidth=1.4,
        height=0.62,
    )
    axes[0].set_yticks(y_pos, [labels[method] for method in methods])
    axes[0].invert_yaxis()
    axes[0].set_title("Curve-error area", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    axes[0].set_xlabel("Normalised response error area", color=MUTED)
    style_axes(axes[0])
    for y, value in zip(y_pos, areas, strict=True):
        axes[0].text(value, y, f" {value:.3f}", va="center", fontsize=9, color=MUTED)

    for method in methods:
        curve = summary[method]["budget_curve"]
        axes[1].plot(
            [row["label_count"] for row in curve],
            [row["normalised_rmse_mean"] for row in curve],
            color=PALETTE[method],
            label=labels[method],
            linewidth=2.8 if method in {"engineering_utility", "cost_aware_utility"} else 2.0,
            marker="o" if method in {"engineering_utility", "cost_aware_utility"} else None,
            markersize=4,
        )
    axes[1].set_title("Learning curve", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    axes[1].set_xlabel("Labelled Core CFD cases", color=MUTED)
    axes[1].set_ylabel("Mean normalised RMSE", color=MUTED)
    style_axes(axes[1])
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(
        0.08,
        0.055,
        (
            "Engineering and cost-aware utility tie on curve-error area for "
            "this compact response-map replay."
        ),
        fontsize=10,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["core_replay"]
    save(fig, path)
    return path


def figure_core_story() -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_axis_off()
    title(
        fig,
        "AeroMap connects to AeroCliff Core",
        (
            "The project now links simulation-budget selection to a custom structured "
            "Venturi-underfloor pressure/load benchmark."
        ),
    )
    boxes = [
        (
            (0.05, 0.52),
            "AeroMap\nMission Control",
            "open-CFD decision benchmark\nAirfRANS + DrivAerML bridge",
            "#eef2ff",
        ),
        (
            (0.38, 0.52),
            "AeroCliff Core\nVenturi Lab",
            "structured blockMesh benchmark\naccepted pressure/load references",
            "#ecfdf5",
        ),
        (
            (0.71, 0.52),
            "2D Response\nReplay",
            "15/15 clean medium cases\n3/3 representative fine checks",
            "#fff7ed",
        ),
    ]
    for (x, y), header, body, color in boxes:
        rect = FancyBboxPatch(
            (x, y),
            0.24,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#cbd5e1",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.12, y + 0.19, header, ha="center", fontsize=18, color=TEXT, weight="bold")
        ax.text(x + 0.12, y + 0.09, body, ha="center", va="center", fontsize=10.5, color=MUTED)
    arrow(ax, (0.29, 0.66), (0.38, 0.66))
    arrow(ax, (0.62, 0.66), (0.71, 0.66))
    ax.text(
        0.05,
        0.28,
        "Public claim",
        fontsize=13,
        color=TEXT,
        weight="bold",
    )
    ax.text(
        0.05,
        0.22,
        (
            "AeroMap maps a structured Venturi-underfloor pressure/load response "
            "under a CFD-label budget."
        ),
        fontsize=13,
        color=TEXT,
    )
    ax.text(
        0.05,
        0.13,
        (
            "Scope: pressure/load response mapping on the structured Core tier; "
            "higher-fidelity transfer and live solver scheduling are follow-on work."
        ),
        fontsize=10.5,
        color=MUTED,
    )
    path = OUT_DIR / FIGURES["core_story"]
    save(fig, path)
    return path


def current_commit() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required to stamp figure manifests")
    return subprocess.check_output(  # noqa: S603 - fixed git command, no user input.
        [git, "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write_manifest(paths: list[Path]) -> None:
    captions = {
        FIGURES[
            "headline"
        ]: "Geometry-disjoint final metrics summary for AeroMap Mission Control v0.3.",
        FIGURES[
            "learning"
        ]: "Label-budget learning curves for drag, lift, top-k recovery and Pareto recall.",
        FIGURES[
            "comparison"
        ]: "Composite acquisition method comparison over six final geometry-disjoint metrics.",
        FIGURES[
            "decision"
        ]: "Decision-focused metric panel: top-k, Pareto recall, rank correlation and regret.",
        FIGURES[
            "flow"
        ]: "Mission Control decision loop from labelled cases to recommended next CFD simulation.",
        FIGURES[
            "two_tier"
        ]: "AeroMap evidence story: AirfRANS, DrivAerML bridge and AeroCliff Core.",
        FIGURES[
            "drivaer3d"
        ]: "DrivAerML compact 3D scalar replay metrics; diversity is currently recommended.",
        FIGURES["core_surface"]: "AeroCliff Core 3 x 5 suction/downforce response surface.",
        FIGURES[
            "core_recovery"
        ]: "AeroCliff Core pressure recovery surface over ride height and diffuser angle.",
        FIGURES[
            "core_replay"
        ]: "AeroCliff Core offline response-map active replay method comparison.",
        FIGURES[
            "core_story"
        ]: "AeroMap to AeroCliff Core workflow for bounded pressure/load response mapping.",
    }
    payload = {
        "schema_version": "aerocliff_aeromap_portfolio_figures_v1",
        "classification": "AEROMAP_AEROCLIFF_CORE_MVP_V0_1_PORTFOLIO_FIGURES",
        "source_evidence": {
            "airfrans": str(EVIDENCE.relative_to(ROOT)),
            "drivaerml_bridge": str(DRIVAER3D.relative_to(ROOT)),
            "aerocliff_core_dataset": str(CORE_2D_DATASET.relative_to(ROOT)),
            "aerocliff_core_replay": str(CORE_2D_REPLAY.relative_to(ROOT)),
        },
        "evidence_generation_git_sha": current_commit(),
        "git_sha_note": (
            "This SHA records the commit used to generate the figure manifest; "
            "the containing release commit may be later."
        ),
        "figures": [
            {
                "path": str(path.relative_to(ROOT)),
                "caption": captions[path.name],
                "source_case": _figure_source_case(path.name),
            }
            for path in paths
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _figure_source_case(path_name: str) -> str:
    if path_name.startswith(("aerocliff_core", "aeromap_aerocliff")):
        return "AeroCliff Core 2D pressure/load response replay"
    if "drivaerml" in path_name or "two_tier" in path_name:
        return "AeroMap 3D bridge and portfolio context"
    return "AirfRANS v0.3 open-CFD replay"


def main() -> None:
    data = load_json(EVIDENCE)
    drivaer3d = load_json(DRIVAER3D)
    core_dataset = load_json(CORE_2D_DATASET)
    core_replay = load_json(CORE_2D_REPLAY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_headline(data),
        figure_learning_curves(data),
        figure_acquisition_comparison(data),
        figure_decision_panel(data),
        figure_flow(),
        figure_two_tier_evidence(),
        figure_drivaer3d_metrics(drivaer3d),
        figure_core_response_surface(core_dataset),
        figure_core_pressure_recovery(core_dataset),
        figure_core_active_replay(core_replay),
        figure_core_story(),
    ]
    write_manifest(paths)
    print(json.dumps({"figures": [str(path.relative_to(ROOT)) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
