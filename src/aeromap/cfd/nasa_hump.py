"""NASA/TMR wall-mounted hump methodology helpers."""

from __future__ import annotations

import gzip
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

METHODOLOGY_GATE_ID = "NASA_TMR_BOUNDARY_LAYER_METHODOLOGY_GATE_V0_1"
CLASSIFICATION = "NASA_HUMP_METHODOLOGY_PREFLIGHT_V0_1"
SST_SMOKE_CLASSIFICATION = "OPENFOAM_NASA_HUMP_SST_SMOKE_V0_1"
RECORDED_TMR_WARNING_COUNT = 2
HUMP_REYNOLDS_NUMBER = 936_000
HUMP_CHORD = 1.0
HUMP_U_INF = 1.0
HUMP_NU = HUMP_U_INF * HUMP_CHORD / HUMP_REYNOLDS_NUMBER

PATCH_CONTRACT: dict[str, str] = {
    "front": "empty",
    "back": "empty",
    "inlet": "patch",
    "outlet": "patch",
    "top_slip": "patch/slip",
    "hump_wall": "wall",
}

PATCH_ORDER = ["front", "back", "inlet", "outlet", "top_slip", "hump_wall"]
COORDINATE_COMPONENTS = 3
BOUNDARY_EXTENT_TOLERANCE = 1e-5
FOAM_FILE_TRAILER = f"// {'*' * 73} //"

WAIVED_WARNINGS = [
    "high_aspect_ratio_expected_for_near_wall_RANS_grid",
    "small_determinant_cells_expected_from_boundary_layer_stretching",
]

FATAL_CHECKS = [
    "negative_or_zero_cell_volume",
    "invalid_or_nonfinite_coordinates",
    "broken_boundary_patch_topology",
    "missing_hump_wall_patch",
    "failed_plot3d_conversion",
    "solver_crash_before_first_iteration",
]

CORRELATION_REQUIREMENTS = [
    ("experimental_cp_cf_parsed", "Pass"),
    ("published_cfl3d_sa_sst_references_parsed", "Pass"),
    ("plot3d_grid_converted", "Pass"),
    ("patch_split_audited", "Pass"),
    ("methodology_mesh_gate_defined", "Pass"),
    ("openfoam_sst_setup_generated", "Pass: single-grid smoke setup"),
    ("solver_run_completed", "Pass: single-grid smoke only"),
    ("cp_cf_extracted_from_openfoam", "Pass: smoke-grid overlay only"),
    ("openfoam_vs_experiment_compared", "Pass: smoke-grid overlay metrics only"),
    ("grid_sensitivity_checked", "Not yet"),
]

RECORDED_103X28_CHECKMESH: dict[str, Any] = {
    "boundary_patches": 6,
    "cells": 2754,
    "failed_mesh_checks": 2,
    "log_path": "artifacts/methodology/nasa_hump/smoke_case_p2d_noblank/checkMesh_split.log",
    "max_aspect_ratio": 19660.6,
    "mesh_ok": False,
    "small_determinant_cells": 382,
    "source": "recorded_local_plot3dToFoam_smoke_summary",
}

FOAM_HEADER = r"""/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  13
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
"""


def methodology_mesh_policy(strict_mesh_gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Classify the official TMR near-wall grid without weakening the global gate."""

    if strict_mesh_gate is None:
        return {
            "gate": METHODOLOGY_GATE_ID,
            "official_tmr_boundary_layer_grid": True,
            "global_aeromap_gate_passed": False,
            "methodology_gate_passed": False,
            "mesh_quality": "not_evaluated",
            "accepted_for": [],
            "not_accepted_for": [
                "solver_smoke",
                "headline_correlation",
                "turbulence_model_recommendation",
            ],
            "waived_warnings": [],
            "fatal_checks": FATAL_CHECKS,
            "rationale": "No checkMesh evidence was supplied.",
        }

    failed_checks = strict_mesh_gate.get("failed_mesh_checks")
    mesh_ok = bool(strict_mesh_gate.get("mesh_ok"))
    global_gate_passed = mesh_ok and (failed_checks in (0, None))

    methodology_warning = not global_gate_passed and failed_checks == RECORDED_TMR_WARNING_COUNT
    return {
        "gate": METHODOLOGY_GATE_ID,
        "official_tmr_boundary_layer_grid": True,
        "global_aeromap_gate_passed": global_gate_passed,
        "methodology_gate_passed": methodology_warning or global_gate_passed,
        "mesh_quality": (
            "accepted_with_methodology_warning"
            if methodology_warning
            else ("accepted" if global_gate_passed else "fatal_or_unclassified")
        ),
        "accepted_for": [
            "conversion_smoke",
            "boundary_condition_smoke",
            "single_solver_smoke",
        ]
        if methodology_warning or global_gate_passed
        else [],
        "not_accepted_for": [
            "headline_correlation",
            "turbulence_model_recommendation",
            "grid_convergence_claim",
        ],
        "waived_warnings": WAIVED_WARNINGS if methodology_warning else [],
        "fatal_checks": FATAL_CHECKS,
        "rationale": (
            "The official 103 x 28 TMR boundary-layer grid is intentionally stretched. "
            "High aspect ratio and small determinant warnings are therefore treated as "
            "methodology warnings for smoke work only, not as correlation eligibility."
        ),
    }


def correlation_eligibility() -> list[dict[str, str]]:
    """Return the preflight status table used by reports and evidence."""

    return [
        {"requirement": requirement, "status": status}
        for requirement, status in CORRELATION_REQUIREMENTS
    ]


def write_conversion_scaffold(
    *,
    grid_path: Path,
    out_dir: Path,
    case_name: str = "nasa_hump_no_plenum_103x28",
) -> dict[str, Any]:
    """Write a small OpenFOAM conversion scaffold for the NASA/TMR hump grid.

    The scaffold does not run OpenFOAM. It records the patch contract, mesh policy and
    commands required for a local smoke conversion.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "input"
    input_dir.mkdir(exist_ok=True)
    target_grid = input_dir / "hump2newtop_noplenumZ103x28.p2dfmt"
    if grid_path.suffix == ".gz":
        target_grid.write_bytes(gzip.decompress(grid_path.read_bytes()))
    else:
        shutil.copyfile(grid_path, target_grid)

    policy = methodology_mesh_policy(RECORDED_103X28_CHECKMESH)
    patch_path = out_dir / "patch_contract.json"
    policy_path = out_dir / "methodology_mesh_policy.json"
    commands_path = out_dir / "Allrun.convert"

    patch_path.write_text(
        json.dumps(PATCH_CONTRACT, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "# Source OpenFOAM before running this script, for example:",
                "#   source /opt/openfoam13/etc/bashrc",
                "",
                f"case_name={case_name!r}",
                "grid=input/hump2newtop_noplenumZ103x28.p2dfmt",
                "",
                'plot3dToFoam -noBlank -2D 0.1 "$grid"',
                "# Apply the patch split described in patch_contract.json before solving.",
                "checkMesh > checkMesh.log 2>&1",
                "",
            ],
        ),
        encoding="utf-8",
    )
    commands_path.chmod(0o755)

    manifest = {
        "schema_version": "nasa_hump_openfoam_conversion_scaffold_v0.1.0",
        "case_name": case_name,
        "grid_path": str(target_grid),
        "patch_contract_path": str(patch_path),
        "methodology_mesh_policy_path": str(policy_path),
        "commands_path": str(commands_path),
        "runs_openfoam": False,
        "next_step": (
            "Run Allrun.convert in an OpenFOAM v13 environment, then audit boundary patches."
        ),
    }
    manifest_path = out_dir / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest | {"manifest_path": str(manifest_path)}


def _foam_dict_header(*, class_name: str, location: str | None, object_name: str) -> str:
    location_line = f'    location    "{location}";\n' if location else ""
    return (
        f"{FOAM_HEADER}"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def _foam_body(text: str) -> str:
    start = text.index("\n(") + 2
    end = text.rfind("\n)")
    return text[start:end].strip()


def _parse_points(path: Path) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for line in _foam_body(path.read_text(encoding="utf-8")).splitlines():
        values = [float(item) for item in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]
        if len(values) == COORDINATE_COMPONENTS:
            points.append((values[0], values[1], values[2]))
    return points


def _parse_faces(path: Path) -> list[list[int]]:
    faces: list[list[int]] = []
    for line in _foam_body(path.read_text(encoding="utf-8")).splitlines():
        values = [int(item) for item in re.findall(r"\d+", line)]
        if values:
            faces.append(values[1:])
    return faces


def _parse_owner(path: Path) -> list[int]:
    return [
        int(line.strip())
        for line in _foam_body(path.read_text(encoding="utf-8")).splitlines()
        if line.strip().isdigit()
    ]


def _boundary_start(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"startFace\s+(\d+);", text)
    if not match:
        msg = f"could not find startFace in {path}"
        raise ValueError(msg)
    return int(match.group(1))


def _write_faces(path: Path, faces: list[list[int]]) -> None:
    lines = [
        _foam_dict_header(class_name="faceList", location="constant/polyMesh", object_name="faces"),
        str(len(faces)),
        "(",
    ]
    lines.extend(f"{len(face)}({' '.join(str(item) for item in face)})" for face in faces)
    lines.extend([")", "", FOAM_FILE_TRAILER, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_owner(path: Path, owners: list[int]) -> None:
    lines = [
        _foam_dict_header(
            class_name="labelList",
            location="constant/polyMesh",
            object_name="owner",
        ),
        str(len(owners)),
        "(",
    ]
    lines.extend(str(item) for item in owners)
    lines.extend([")", "", FOAM_FILE_TRAILER, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_boundary(path: Path, patch_counts: Mapping[str, int], start_face: int) -> None:
    lines = [
        _foam_dict_header(
            class_name="polyBoundaryMesh",
            location="constant/polyMesh",
            object_name="boundary",
        ),
        str(len(PATCH_ORDER)),
        "(",
    ]
    cursor = start_face
    for name in PATCH_ORDER:
        patch_type = PATCH_CONTRACT[name].split("/")[0]
        count = patch_counts[name]
        lines.extend(
            [
                f"    {name}",
                "    {",
                f"        type            {patch_type};",
                f"        nFaces          {count};",
                f"        startFace       {cursor};",
                "    }",
            ],
        )
        cursor += count
    lines.extend([")", "", FOAM_FILE_TRAILER, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def split_plot3d_default_patch(case_dir: Path, *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Split the converted NASA hump single-patch PLOT3D mesh into named patches."""

    mesh_dir = case_dir / "constant" / "polyMesh"
    points = _parse_points(mesh_dir / "points")
    faces = _parse_faces(mesh_dir / "faces")
    owners = _parse_owner(mesh_dir / "owner")
    start_face = _boundary_start(mesh_dir / "boundary")
    x_min = min(point[0] for point in points)
    x_max = max(point[0] for point in points)
    z_min = min(point[2] for point in points)
    z_max = max(point[2] for point in points)

    grouped: dict[str, list[tuple[list[int], int]]] = {name: [] for name in PATCH_ORDER}
    remaining: list[tuple[list[int], int, float]] = []
    for face, owner in zip(faces[start_face:], owners[start_face:], strict=True):
        coordinates = [points[index] for index in face]
        xs = [item[0] for item in coordinates]
        ys = [item[1] for item in coordinates]
        zs = [item[2] for item in coordinates]
        x_span = max(xs) - min(xs)
        z_span = max(zs) - min(zs)
        x_center = sum(xs) / len(xs)
        y_center = sum(ys) / len(ys)
        z_center = sum(zs) / len(zs)
        if z_span <= tolerance and abs(z_center - z_min) <= tolerance:
            grouped["front"].append((face, owner))
        elif z_span <= tolerance and abs(z_center - z_max) <= tolerance:
            grouped["back"].append((face, owner))
        elif x_span <= tolerance and abs(x_center - x_min) <= BOUNDARY_EXTENT_TOLERANCE:
            grouped["inlet"].append((face, owner))
        elif x_span <= tolerance and abs(x_center - x_max) <= BOUNDARY_EXTENT_TOLERANCE:
            grouped["outlet"].append((face, owner))
        else:
            remaining.append((face, owner, y_center))

    if len(remaining) % 2 != 0:
        msg = f"expected an even number of upper/lower boundary faces, got {len(remaining)}"
        raise ValueError(msg)
    y_sorted = sorted(item[2] for item in remaining)
    threshold = (y_sorted[(len(y_sorted) // 2) - 1] + y_sorted[len(y_sorted) // 2]) / 2
    for face, owner, y_center in remaining:
        grouped["top_slip" if y_center > threshold else "hump_wall"].append((face, owner))

    new_faces = faces[:start_face]
    new_owners = owners[:start_face]
    counts: dict[str, int] = {}
    for name in PATCH_ORDER:
        counts[name] = len(grouped[name])
        new_faces.extend(face for face, _owner in grouped[name])
        new_owners.extend(owner for _face, owner in grouped[name])

    _write_faces(mesh_dir / "faces", new_faces)
    _write_owner(mesh_dir / "owner", new_owners)
    _write_boundary(mesh_dir / "boundary", counts, start_face)
    return {
        "patch_counts": counts,
        "start_face": start_face,
        "split_threshold_y": threshold,
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
    }


def _field_header(*, class_name: str, object_name: str) -> str:
    return _foam_dict_header(class_name=class_name, location="0", object_name=object_name)


def write_sst_smoke_case_template(case_dir: Path, *, end_time: int = 200) -> dict[str, Any]:
    """Write a bounded kOmegaSST OpenFOAM v13 smoke setup for the NASA hump case."""

    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(exist_ok=True)
    (case_dir / "system").mkdir(exist_ok=True)
    nu = HUMP_NU
    u_inf = HUMP_U_INF
    intensity = 0.01
    length_scale = 0.07
    k_value = 1.5 * (u_inf * intensity) ** 2
    omega_value = (k_value**0.5) / (0.09**0.25 * length_scale)

    (case_dir / "system" / "controlDict").write_text(
        _foam_dict_header(class_name="dictionary", location="system", object_name="controlDict")
        + f"""solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {end_time};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
functions       {{}};

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSchemes").write_text(
        _foam_dict_header(class_name="dictionary", location="system", object_name="fvSchemes")
        + """ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSolution").write_text(
        _foam_dict_header(class_name="dictionary", location="system", object_name="fvSolution")
        + """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.05;
    }

    Phi
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    fields
    {
        p               0.1;
    }
    equations
    {
        U               0.3;
        k               0.3;
        omega           0.3;
    }
}

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "meshQualityDict").write_text(
        _foam_dict_header(class_name="dictionary", location="system", object_name="meshQualityDict")
        + """#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    (case_dir / "constant" / "physicalProperties").write_text(
        _foam_dict_header(
            class_name="dictionary",
            location="constant",
            object_name="physicalProperties",
        )
        + f"""viscosityModel  constant;
nu              [0 2 -1 0 0 0 0] {nu:.12g};

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    (case_dir / "constant" / "momentumTransport").write_text(
        _foam_dict_header(
            class_name="dictionary",
            location="constant",
            object_name="momentumTransport",
        )
        + """simulationType RAS;

RAS
{
    model           kOmegaSST;
    turbulence      on;
}

// ************************************************************************* //
""",
        encoding="utf-8",
    )

    boundary_common = """
    front
    {
        type            empty;
    }
    back
    {
        type            empty;
    }
"""
    (case_dir / "0" / "U").write_text(
        _field_header(class_name="volVectorField", object_name="U")
        + f"""dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({u_inf:g} 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({u_inf:g} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top_slip
    {{
        type            slip;
    }}
    hump_wall
    {{
        type            noSlip;
    }}
{boundary_common}}}

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    (case_dir / "0" / "p").write_text(
        _field_header(class_name="volScalarField", object_name="p")
        + """dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    top_slip
    {
        type            zeroGradient;
    }
    hump_wall
    {
        type            zeroGradient;
    }
"""
        + boundary_common
        + """}

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    for name, dimensions, internal, wall_type in [
        ("k", "[0 2 -2 0 0 0 0]", f"{k_value:.12g}", "kqRWallFunction"),
        ("omega", "[0 0 -1 0 0 0 0]", f"{omega_value:.12g}", "omegaWallFunction"),
    ]:
        (case_dir / "0" / name).write_text(
            _field_header(class_name="volScalarField", object_name=name)
            + f"""dimensions      {dimensions};
internalField   uniform {internal};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {internal};
    }}
    outlet
    {{
        type            zeroGradient;
        value           uniform {internal};
    }}
    top_slip
    {{
        type            zeroGradient;
    }}
    hump_wall
    {{
        type            {wall_type};
        value           uniform {internal};
    }}
{boundary_common}}}

// ************************************************************************* //
""",
            encoding="utf-8",
        )
    (case_dir / "0" / "nut").write_text(
        _field_header(class_name="volScalarField", object_name="nut")
        + """dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    top_slip
    {
        type            calculated;
        value           uniform 0;
    }
    hump_wall
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
"""
        + boundary_common
        + """}

// ************************************************************************* //
""",
        encoding="utf-8",
    )
    return {
        "solver": "incompressibleFluid",
        "turbulence_model": "kOmegaSST",
        "end_time": end_time,
        "u_inf": u_inf,
        "nu": nu,
        "reynolds_number": HUMP_REYNOLDS_NUMBER,
        "turbulence_intensity": intensity,
        "turbulence_length_scale": length_scale,
        "k": k_value,
        "omega": omega_value,
        "potential_flow_initialisation_required": True,
    }
