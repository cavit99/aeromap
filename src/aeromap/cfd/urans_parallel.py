"""Prepare disposable OpenFOAM MPI benchmark continuations."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeromap.cfd.dictionaries import header
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file

SCHEMA_VERSION = "aerocliff_urans_parallel_benchmark_plan_v0.1.0"
DEFAULT_RANKS = (1, 4, 8, 16)
DEFAULT_CONTINUATION_S = 0.002


@dataclass(frozen=True)
class UransParallelBenchmarkPlanArtifacts:
    manifest_path: Path
    run_scripts: tuple[Path, ...]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _repo_relative_required(path: Path, *, label: str) -> str:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        message = f"{label} must be inside project root for Docker Compose: {path}"
        raise ValueError(message) from exc
    if rel in {"", "."} or rel.startswith("../") or "/../" in rel or rel.endswith("/.."):
        message = f"{label} resolved to an unsafe repository-relative path: {rel}"
        raise ValueError(message)
    return rel


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _decompose_par_dict(rank: int) -> str:
    return (
        header("dictionary", "system", "decomposeParDict")
        + f"""
numberOfSubdomains {rank};

method          hierarchical;

hierarchicalCoeffs
{{
    n           ({rank} 1 1);
    delta       0.001;
    order       xyz;
}}

distributed     no;

// ************************************************************************* //
"""
    )


def _rank_script(
    *,
    source_rel: str,
    benchmark_case_rel: str,
    decompose_rel: str | None,
    rank: int,
    start_time_s: float,
    end_time_s: float,
) -> str:
    source_q = shlex.quote(source_rel)
    bench_q = shlex.quote(benchmark_case_rel)
    decompose_q = shlex.quote(decompose_rel) if decompose_rel is not None else ""
    container_dir = shlex.quote(f"/work/{benchmark_case_rel}/openfoam")
    foam_run = "foamRun -solver incompressibleFluid"
    timing_prefix = "TIMEFORMAT='real %3R user %3U sys %3S'"
    if rank > 1:
        solver_line = (
            f"decomposePar -latestTime -force > ../logs/decomposePar_rank{rank}.log 2>&1; "
            f"{timing_prefix}; "
            f"{{ time mpirun --allow-run-as-root -np {rank} {foam_run} -parallel "
            f"> ../logs/foamRun_parallel_rank{rank}.log 2>&1; }} "
            f"2> ../logs/wall_time_rank{rank}.txt; "
            f"reconstructPar -latestTime > ../logs/reconstructPar_rank{rank}.log 2>&1"
        )
    else:
        solver_line = (
            f"{timing_prefix}; "
            f"{{ time {foam_run} > ../logs/foamRun_serial_rank{rank}.log 2>&1; }} "
            f"2> ../logs/wall_time_rank{rank}.txt"
        )
    return f"""#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${{AEROMAP_REPO_ROOT:-$(pwd)}}"
SOURCE_WORK_CASE_REL={source_q}
BENCH_CASE_REL={bench_q}
DECOMPOSE_REL={decompose_q}
RANK={rank}
START_TIME_S={start_time_s:g}
END_TIME_S={end_time_s:g}

REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
SOURCE_WORK_CASE="$REPO_ROOT/$SOURCE_WORK_CASE_REL"
BENCH_CASE="$REPO_ROOT/$BENCH_CASE_REL"
LOCK_DIR="$BENCH_CASE.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Benchmark case is already locked: $BENCH_CASE" >&2
    exit 2
fi
cleanup_lock() {{
    rm -f "$LOCK_DIR/host_pid" "$LOCK_DIR/started_at_utc"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}}
trap cleanup_lock EXIT
printf '%s\\n' "$$" > "$LOCK_DIR/host_pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK_DIR/started_at_utc"

if [[ -e "$BENCH_CASE" && "${{AEROMAP_PARALLEL_BENCH_OVERWRITE:-0}}" != "1" ]]; then
    echo "Refusing to overwrite existing benchmark case: $BENCH_CASE" >&2
    echo "Set AEROMAP_PARALLEL_BENCH_OVERWRITE=1 to replace it." >&2
    exit 2
fi

rm -rf "$BENCH_CASE"
mkdir -p "$(dirname "$BENCH_CASE")"
cp -a "$SOURCE_WORK_CASE" "$BENCH_CASE"
rm -rf "$BENCH_CASE/openfoam"/processor*
mkdir -p "$BENCH_CASE/logs" "$BENCH_CASE/quality"
if [[ "$RANK" -gt 1 ]]; then
    cp "$REPO_ROOT/$DECOMPOSE_REL" "$BENCH_CASE/openfoam/system/decomposeParDict"
fi

container_cmd="set +u"
container_cmd="$container_cmd; source /opt/openfoam13/etc/bashrc"
container_cmd="$container_cmd; set -euo pipefail"
container_cmd="$container_cmd; cd {container_dir}"
container_cmd="$container_cmd; foamDictionary system/controlDict -entry startFrom -set latestTime"
container_cmd="$container_cmd; foamDictionary system/controlDict -entry endTime -set $END_TIME_S"
container_cmd="$container_cmd; foamDictionary system/controlDict -entry purgeWrite -set 0"
container_cmd="$container_cmd; {solver_line}"

docker compose run --rm cfd "$container_cmd"
"""


def write_urans_parallel_benchmark_plan(
    *,
    work_case: Path,
    out_dir: Path,
    checkpoint_report_path: Path,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    continuation_s: float = DEFAULT_CONTINUATION_S,
) -> UransParallelBenchmarkPlanArtifacts:
    """Write disposable per-rank OpenFOAM benchmark scripts without running them."""

    if continuation_s <= 0.0:
        message = "continuation_s must be positive"
        raise ValueError(message)
    if not ranks or any(rank < 1 for rank in ranks):
        message = "ranks must contain positive MPI rank counts"
        raise ValueError(message)
    if not work_case.exists():
        message = f"URANS work case does not exist: {work_case}"
        raise FileNotFoundError(message)
    if not checkpoint_report_path.exists():
        message = f"checkpoint report does not exist: {checkpoint_report_path}"
        raise FileNotFoundError(message)

    checkpoint = _load_json(checkpoint_report_path)
    start_time_s = float(checkpoint["latest_complete_written_time_s"])
    end_time_s = start_time_s + continuation_s
    work_case_rel = _repo_relative_required(work_case, label="work_case")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_scripts: list[Path] = []
    planned_cases: list[dict[str, Any]] = []
    for rank in ranks:
        case_dir = out_dir / f"{work_case.name}_rank{rank}"
        case_rel = _repo_relative_required(case_dir, label=f"benchmark_case_rank{rank}")
        script_path = out_dir / f"run_rank{rank}.sh"
        decompose_path = out_dir / f"decomposeParDict_rank{rank}"
        if rank > 1:
            atomic_write_text(decompose_path, _decompose_par_dict(rank))
        script = _rank_script(
            source_rel=work_case_rel,
            benchmark_case_rel=case_rel,
            decompose_rel=_repo_relative_required(decompose_path, label=f"decompose_rank{rank}")
            if rank > 1
            else None,
            rank=rank,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        atomic_write_text(script_path, script)
        script_path.chmod(0o755)
        run_scripts.append(script_path)
        planned_cases.append(
            {
                "rank": rank,
                "benchmark_case": case_rel,
                "run_script": _repo_relative(script_path),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "continuation_s": continuation_s,
                "decomposeParDict_template": _repo_relative(decompose_path) if rank > 1 else None,
                "expected_logs": [
                    f"logs/wall_time_rank{rank}.txt",
                    (
                        f"logs/foamRun_parallel_rank{rank}.log"
                        if rank > 1
                        else f"logs/foamRun_serial_rank{rank}.log"
                    ),
                ],
            },
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARED_NOT_RUN",
        "accepted": False,
        "training_eligible": False,
        "source_work_case": work_case_rel,
        "checkpoint_report": _repo_relative(checkpoint_report_path),
        "checkpoint_report_sha256": sha256_file(checkpoint_report_path),
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "continuation_s": continuation_s,
        "rank_counts": list(ranks),
        "current_run_serial_evidence": {
            "processor_directories_present": bool(
                list((work_case / "openfoam").glob("processor*")),
            ),
            "existing_logs_use_parallel_flag": any(
                "-parallel" in path.read_text(encoding="utf-8", errors="ignore")
                for path in sorted((work_case / "logs").glob("foamRun_urans_recon*.log"))
            ),
        },
        "planned_cases": planned_cases,
        "selection_rule": {
            "primary": "fastest wall-clock time with finite OpenFOAM completion",
            "numerical_consistency": (
                "final force means over the 0.002 s continuation must agree with the "
                "single-rank continuation within 0.5% before a rank count is used for "
                "the next physical-time segment"
            ),
            "stability": "max Courant remains below 1 and no solver failure occurs",
        },
        "claims_not_established": [
            "No parallel benchmark has been run by this preparation command.",
            "No MPI rank count is selected until the generated scripts run and are analyzed.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return UransParallelBenchmarkPlanArtifacts(
        manifest_path=manifest_path,
        run_scripts=tuple(run_scripts),
    )
