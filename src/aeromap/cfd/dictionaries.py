"""OpenFOAM v13 dictionary rendering."""

from __future__ import annotations

import math

from aeromap.cfd.patch_surface import (
    ZERO_LAYER_PATCHES,
    article_patch_names,
    critical_core_patches,
    is_multi_patch_mode,
)
from aeromap.cfd.schema import CfdConfig
from aeromap.constants import REF
from aeromap.parameters import AeroParams
from aeromap.transforms import inlet_unit_vector


def header(foam_class: str, location: str, obj: str) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version: 13                                     |
|   \\\\  /    A nd           | Website: https://openfoam.org                   |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {foam_class};
    location    "{location}";
    object      {obj};
}}
// ************************************************************************* //
"""


def vec(values: tuple[float, float, float]) -> str:
    return f"({values[0]:.10g} {values[1]:.10g} {values[2]:.10g})"


def inlet_velocity(params: AeroParams) -> tuple[float, float, float]:
    direction = inlet_unit_vector(params.yaw_deg)
    return (
        float(REF.u_inf_m_s * direction[0]),
        float(REF.u_inf_m_s * direction[1]),
        0.0,
    )


def turbulence_values() -> tuple[float, float]:
    intensity = 0.01
    length_scale = 0.07 * REF.l_ref_m
    c_mu = 0.09
    k = 1.5 * (REF.u_inf_m_s * intensity) ** 2
    omega = math.sqrt(k) / (c_mu**0.25 * length_scale)
    return k, omega


def _refinement_box_geometry(config: CfdConfig) -> str:
    return "".join(
        [
            f"""
    {box.name}
    {{
        type box;
        min {vec(box.bounds_min)};
        max {vec(box.bounds_max)};
    }}
"""
            for box in config.mesh.refinement_boxes
        ],
    )


def _refinement_regions(config: CfdConfig) -> str:
    if not config.mesh.refinement_boxes and not config.mesh.span_refinements:
        return "refinementRegions {}"
    rendered = [
        "refinementRegions\n    {",
        *[
            f"""
        {box.name}
        {{
            mode inside;
            level {box.level};
        }}"""
            for box in config.mesh.refinement_boxes
        ],
    ]
    rendered.extend(
        [
            f"""
        {span.surface}
        {{
            mode insideSpan;
            level ({span.distance_m:g} {span.level});
            cellsAcrossSpan {span.cells_across_span};
        }}"""
            for span in config.mesh.span_refinements
        ],
    )
    rendered.append("\n    }")
    return "".join(rendered)


def _surface_file(config: CfdConfig) -> str:
    if is_multi_patch_mode(config.surface_export.openfoam_patch_mode):
        return "article.obj"
    return "article.stl"


def _geometry_regions(config: CfdConfig) -> str:
    if not is_multi_patch_mode(config.surface_export.openfoam_patch_mode):
        return ""
    body = "".join(
        [
            f"""
            {patch}
            {{
                name {patch};
            }}"""
            for patch in article_patch_names(
                patch_mode=config.surface_export.openfoam_patch_mode,
            )
        ],
    )
    return f"""
        regions
        {{{body}
        }}"""


def _refinement_surface_regions(config: CfdConfig, level_min: int, level_max: int) -> str:
    if not is_multi_patch_mode(config.surface_export.openfoam_patch_mode):
        return ""
    body = "".join(
        [
            f"""
                {patch}
                {{
                    level ({level_min} {level_max});
                    patchInfo {{ type wall; }}
                }}"""
            for patch in article_patch_names(
                patch_mode=config.surface_export.openfoam_patch_mode,
            )
        ],
    )
    return f"""
            regions
            {{{body}
            }}"""


def _patch_layers(config: CfdConfig) -> str:
    if config.mesh.patch_layers:
        return "".join(
            [
                f"""
        {patch.patch} {{ nSurfaceLayers {patch.n_surface_layers}; }}"""
                for patch in config.mesh.patch_layers
            ],
        )
    if is_multi_patch_mode(config.surface_export.openfoam_patch_mode):
        body = []
        critical_patches = critical_core_patches(
            patch_mode=config.surface_export.openfoam_patch_mode,
        )
        for patch in article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode):
            if patch in critical_patches:
                layers = config.mesh.n_surface_layers
            elif patch in ZERO_LAYER_PATCHES:
                layers = 0
            else:
                layers = 1
            body.append(
                f"""
        {patch} {{ nSurfaceLayers {layers}; }}""",
            )
        return "".join(body)
    return f"""
        article {{ nSurfaceLayers {config.mesh.n_surface_layers}; }}"""


def _layer_thickness_controls(config: CfdConfig) -> str:
    if config.mesh.first_layer_thickness is not None:
        return f"""
    expansionRatio {config.mesh.layer_expansion_ratio:g};
    firstLayerThickness {config.mesh.first_layer_thickness:g};
    minThickness {config.mesh.min_layer_thickness:g};"""
    return f"""
    expansionRatio {config.mesh.layer_expansion_ratio:g};
    finalLayerThickness {config.mesh.final_layer_thickness:g};
    minThickness {config.mesh.min_layer_thickness:g};"""


def block_mesh_dict(config: CfdConfig) -> str:
    nx, ny, nz = config.mesh.block_cells
    return (
        header("dictionary", "system", "blockMeshDict")
        + f"""
scale 1;

vertices
(
    (-4 -2 0)
    (10 -2 0)
    (10 2 0)
    (-4 2 0)
    (-4 -2 4)
    (10 -2 4)
    (10 2 4)
    (-4 2 4)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    ground
    {{
        type wall;
        faces ((0 1 2 3));
    }}
    top
    {{
        type patch;
        faces ((4 5 6 7));
    }}
    side_y_min
    {{
        type patch;
        faces ((0 1 5 4));
    }}
    side_y_max
    {{
        type patch;
        faces ((3 7 6 2));
    }}
);

mergePatchPairs
(
);
"""
    )


def snappy_hex_mesh_dict(config: CfdConfig) -> str:
    level_min, level_max = config.mesh.surface_level
    refinement_box_geometry = _refinement_box_geometry(config)
    refinement_regions = _refinement_regions(config)
    surface_file = _surface_file(config)
    geometry_regions = _geometry_regions(config)
    refinement_surface_regions = _refinement_surface_regions(config, level_min, level_max)
    patch_layers = _patch_layers(config)
    layer_thickness_controls = _layer_thickness_controls(config)
    features_block = (
        f"""
        {{
            file "article.eMesh";
            level {level_max};
        }}"""
        if config.mesh.explicit_feature_snap
        else ""
    )
    add_layers = "true" if config.mesh.add_layers else "false"
    layer_relative_sizes = "true" if config.mesh.layer_relative_sizes else "false"
    implicit_feature_snap = "true" if config.mesh.implicit_feature_snap else "false"
    explicit_feature_snap = "true" if config.mesh.explicit_feature_snap else "false"
    layer_slip_feature_angle = (
        f"    slipFeatureAngle {config.mesh.layer_slip_feature_angle_deg:g};\n"
        if config.mesh.layer_slip_feature_angle_deg is not None
        else ""
    )
    layer_n_relaxed_iter = (
        f"    nRelaxedIter {config.mesh.layer_n_relaxed_iter};\n"
        if config.mesh.layer_n_relaxed_iter is not None
        else ""
    )
    layer_n_medial_axis_iter = (
        f"    nMedialAxisIter {config.mesh.layer_n_medial_axis_iter};\n"
        if config.mesh.layer_n_medial_axis_iter is not None
        else ""
    )
    layer_additional_reporting = "true" if config.mesh.layer_additional_reporting else "false"
    layer_optional_controls = (
        layer_slip_feature_angle
        + layer_n_relaxed_iter
        + layer_n_medial_axis_iter
        + f"    additionalReporting {layer_additional_reporting};\n"
    )
    return (
        header("dictionary", "system", "snappyHexMeshDict")
        + f"""
castellatedMesh true;
snap            true;
addLayers       {add_layers};

geometry
{{
    article
    {{
        type triSurface;
        file "{surface_file}";
        name article;
{geometry_regions}
    }}
{refinement_box_geometry}
}}

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells {config.mesh.max_global_cells};
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {config.mesh.n_cells_between_levels};

    features
    (
{features_block}
    );

    refinementSurfaces
    {{
        article
        {{
            level ({level_min} {level_max});
            patchInfo {{ type wall; }}
{refinement_surface_regions}
        }}
    }}

    resolveFeatureAngle {config.mesh.feature_angle_deg:g};
    {refinement_regions}
    locationInMesh (0 0 2);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter {config.mesh.snap_solve_iterations};
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap {implicit_feature_snap};
    explicitFeatureSnap {explicit_feature_snap};
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes {layer_relative_sizes};
    layers
    {{
{patch_layers}
    }}
{layer_thickness_controls}
    nGrow {config.mesh.layer_n_grow};
    featureAngle {config.mesh.layer_feature_angle_deg:g};
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {config.mesh.max_thickness_to_medial_ratio:g};
    minMedialAxisAngle 90;
{layer_optional_controls}\
    nBufferCellsNoExtrude {config.mesh.layer_n_buffer_cells_no_extrude};
    nLayerIter 50;
}}

meshQualityControls
{{
    #include "$FOAM_ETC/caseDicts/mesh/generation/meshQualityDict"
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;
"""
    )


def mesh_quality_dict() -> str:
    return (
        header("dictionary", "system", "meshQualityDict")
        + """
#include "$FOAM_ETC/caseDicts/mesh/generation/meshQualityDict.cfg"
"""
    )


def _patch_list(config: CfdConfig) -> str:
    return (
        "("
        + " ".join(article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode))
        + ")"
    )


def _surface_features_closeness(config: CfdConfig) -> str:
    if not config.mesh.span_refinements:
        return ""
    return """
closeness
{
    pointCloseness yes;
}
"""


def surface_features_dict(config: CfdConfig | None = None) -> str:
    config = config or CfdConfig()
    return (
        header("dictionary", "system", "surfaceFeaturesDict")
        + f"""
surfaces ("{_surface_file(config)}");

includedAngle 150;
{_surface_features_closeness(config)}
"""
    )


def control_dict(config: CfdConfig) -> str:
    patches = _patch_list(config)
    return (
        header("dictionary", "system", "controlDict")
        + f"""
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {config.solver.max_iterations};
deltaT          1;

writeControl    timeStep;
writeInterval   {config.solver.write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         {patches};
        rho             rhoInf;
        rhoInf          {REF.rho_kg_m3:g};
        CofR            (1 0 0);
        writeControl    timeStep;
        writeInterval   1;
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         {patches};
        rho             rhoInf;
        rhoInf          {REF.rho_kg_m3:g};
        liftDir         (0 0 -1);
        dragDir         (1 0 0);
        CofR            (1 0 0);
        pitchAxis       (0 1 0);
        magUInf         {REF.u_inf_m_s:g};
        lRef            {REF.l_ref_m:g};
        Aref            {REF.a_ref_m2:g};
        writeControl    timeStep;
        writeInterval   1;
    }}

    inletFlowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           inlet;
        fields          (phi);
        operation       sum;
    }}

    outletFlowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           outlet;
        fields          (phi);
        operation       sum;
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        patches         {patches};
        writeControl    writeTime;
    }}

    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
}}
"""
    )


def fv_schemes() -> str:
    return (
        header("dictionary", "system", "fvSchemes")
        + """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(phi,U)      bounded Gauss linearUpwind grad(U);
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
    method          meshWave;
}
"""
    )


def fv_solution() -> str:
    return (
        header("dictionary", "system", "fvSolution")
        + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    pcorr
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0;
        smoother        GaussSeidel;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl
    {
        p       1e-2;
        U       1e-3;
        "(k|omega)" 1e-3;
    }
}

relaxationFactors
{
    equations
    {
        U       0.9;
        ".*"    0.9;
    }
}
"""
    )


def physical_properties() -> str:
    return (
        header("dictionary", "constant", "physicalProperties")
        + f"""
viscosityModel  constant;
nu              {REF.nu_m2_s};
"""
    )


def momentum_transport() -> str:
    return (
        header("dictionary", "constant", "momentumTransport")
        + """
simulationType  RAS;

RAS
{
    model           kOmegaSST;
    turbulence      on;
    viscosityModel  Newtonian;
}
"""
    )


def _wall_boundary_entries(config: CfdConfig | None, entry: str) -> str:
    config = config or CfdConfig()
    return "\n".join(
        [
            f"    {patch:<22} {{ {entry} }}"
            for patch in article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)
        ],
    )


def field_u(params: AeroParams, config: CfdConfig | None = None) -> str:
    u = inlet_velocity(params)
    wall_entries = _wall_boundary_entries(config, "type noSlip;")
    return (
        header("volVectorField", "0", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform {vec(u)};

boundaryField
{{
    inlet       {{ type fixedValue; value uniform {vec(u)}; }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type movingWallVelocity; value uniform {vec(u)}; }}
    top         {{ type slip; }}
    side_y_min  {{ type slip; }}
    side_y_max  {{ type slip; }}
{wall_entries}
}}
"""
    )


def field_p(config: CfdConfig | None = None) -> str:
    wall_entries = _wall_boundary_entries(config, "type zeroGradient;")
    return (
        header("volScalarField", "0", "p")
        + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet       {{ type zeroGradient; }}
    outlet      {{ type fixedValue; value uniform 0; }}
    ground      {{ type zeroGradient; }}
    top         {{ type zeroGradient; }}
    side_y_min  {{ type zeroGradient; }}
    side_y_max  {{ type zeroGradient; }}
{wall_entries}
}}
"""
    )


def field_k(config: CfdConfig | None = None) -> str:
    k, _omega = turbulence_values()
    wall_entries = _wall_boundary_entries(
        config,
        f"type kqRWallFunction; value uniform {k:.10g};",
    )
    return (
        header("volScalarField", "0", "k")
        + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k:.10g};

boundaryField
{{
    inlet       {{ type fixedValue; value uniform {k:.10g}; }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type kqRWallFunction; value uniform {k:.10g}; }}
    top         {{ type zeroGradient; }}
    side_y_min  {{ type zeroGradient; }}
    side_y_max  {{ type zeroGradient; }}
{wall_entries}
}}
"""
    )


def field_omega(config: CfdConfig | None = None) -> str:
    _k, omega = turbulence_values()
    wall_entries = _wall_boundary_entries(
        config,
        f"type omegaWallFunction; value uniform {omega:.10g};",
    )
    return (
        header("volScalarField", "0", "omega")
        + f"""
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega:.10g};

boundaryField
{{
    inlet       {{ type fixedValue; value uniform {omega:.10g}; }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type omegaWallFunction; value uniform {omega:.10g}; }}
    top         {{ type zeroGradient; }}
    side_y_min  {{ type zeroGradient; }}
    side_y_max  {{ type zeroGradient; }}
{wall_entries}
}}
"""
    )


def field_nut(config: CfdConfig | None = None) -> str:
    wall_entries = _wall_boundary_entries(config, "type nutkWallFunction; value uniform 0;")
    return (
        header("volScalarField", "0", "nut")
        + f"""
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet       {{ type calculated; value uniform 0; }}
    outlet      {{ type calculated; value uniform 0; }}
    ground      {{ type nutkWallFunction; value uniform 0; }}
    top         {{ type calculated; value uniform 0; }}
    side_y_min  {{ type calculated; value uniform 0; }}
    side_y_max  {{ type calculated; value uniform 0; }}
{wall_entries}
}}
"""
    )
