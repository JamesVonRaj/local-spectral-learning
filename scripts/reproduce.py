"""Single entry point for reproducing the numerical results.

Typical use::

    python scripts/reproduce.py check       # archive and result validation
    python scripts/reproduce.py artifacts   # archived data -> plots/tables
    python scripts/reproduce.py experiments # rerun fixed-seed experiments
    python scripts/reproduce.py full        # experiments, then artifacts

The ``full`` workflow can take several hours. Every
subprocess is printed before execution, and a failure stops the workflow.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from validation.data_archive import extract as extract_publication_data

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_ROOT = Path(os.environ.get("LSL_DATA_DIR", SCRIPTS_DIR / "outputs"))
VECTOR_EXEMPLAR = (
    DATA_ROOT
    / "prl_vector_periodic"
    / "vector_periodic_s5_net0_train0_c50_w10.npz"
)
LARGE_VECTOR_EXEMPLAR = (
    DATA_ROOT
    / "prl_vector_periodic"
    / "vector_periodic_s20_net0_train0_c50_w10.npz"
)
REQUIRED_DATA = (
    VECTOR_EXEMPLAR,
    DATA_ROOT / "prl_bandgap" / "fig1_exemplar.npz",
    DATA_ROOT / "prl_propagation" / "summary.json",
    DATA_ROOT / "prl_v7_adaptive" / "retargeting.npz",
)


def module(name: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", name, *arguments)


def experiment_commands(workers: int) -> list[tuple[str, ...]]:
    """Commands that regenerate every retained numerical evidence family."""
    vector = "publication.bloch_gap"
    exemplar = str(VECTOR_EXEMPLAR)
    return [
        module("publication.scalar_gap", "--force"),
        module("publication.scalar_validation", "--force", "--only", "nontrivial"),
        module("publication.scalar_validation", "--force", "--only", "controls"),
        module("publication.scalar_validation", "--force", "--only", "targets"),
        module("publication.scalar_validation", "--force", "--only", "finite"),
        module("publication.nonspatial", "--force"),
        module(
            vector,
            "--mode", "single",
            "--response-mode", "reciprocal",
            "--frequency-sampling", "random_grid",
            "--frequencies-per-step", "1",
            "--material-update", "response_conditioned_log",
            "--response-metric-eta", "0.02",
            "--response-metric-lambda-ratio", "0.025",
            "--response-bound-mode", "clip",
            "--size", "5",
            "--net-seed", "0",
            "--train-seed", "0",
            "--center-percentile", "50",
            "--width-percentile", "10",
            "--steps", "3000",
            "--k-train-grid", "5",
            "--k-eval-grid", "9",
            "--k-dense-grid", "17",
            "--eval-every", "100",
        ),
        module(
            vector,
            "--mode", "bz-convergence",
            "--input", exemplar,
            "--bz-grids", "17", "33", "65",
        ),
        module(
            vector,
            "--mode", "refine-gap",
            "--input", exemplar,
            "--refine-grid", "33",
            "--refine-candidates", "7",
            "--refine-maxiter", "400",
        ),
        module(
            vector,
            "--mode", "ensemble",
            "--response-mode", "reciprocal",
            "--material-update", "response_conditioned_log",
            "--response-metric-eta", "0.02",
            "--response-metric-lambda-ratio", "0.025",
            "--response-bound-mode", "clip",
            "--ensemble-sizes", "5", "6", "8", "10", "12", "16", "20",
            "--ensemble-net-seeds", "10",
            "--steps", "3000",
            "--k-dense-grid", "33",
            "--eval-every", "3000",
        ),
        module(
            vector,
            "--mode", "single",
            "--response-mode", "reciprocal",
            "--frequency-sampling", "random_grid",
            "--frequencies-per-step", "1",
            "--material-update", "response_conditioned_log",
            "--response-metric-eta", "0.02",
            "--response-metric-lambda-ratio", "0.025",
            "--response-bound-mode", "clip",
            "--size", "20",
            "--net-seed", "0",
            "--train-seed", "0",
            "--center-percentile", "50",
            "--width-percentile", "10",
            "--steps", "3000",
            "--k-train-grid", "5",
            "--k-eval-grid", "9",
            "--k-dense-grid", "33",
            "--eval-every", "3000",
        ),
        module(
            vector,
            "--mode", "refine-gap",
            "--input", str(LARGE_VECTOR_EXEMPLAR),
            "--refine-grid", "9",
            "--refine-candidates", "5",
            "--refine-maxiter", "200",
            "--refine-output-name", "refined_gap_s20_net0",
        ),
        module(
            vector,
            "--mode", "controls",
            "--response-mode", "reciprocal",
            "--material-update", "response_conditioned_log",
            "--response-metric-eta", "0.02",
            "--response-metric-lambda-ratio", "0.025",
            "--response-bound-mode", "clip",
            "--size", "5",
            "--control-net-seeds", "8",
            "--steps", "3000",
            "--k-dense-grid", "33",
        ),
        module(
            "publication.propagation", "full",
            "--workers", str(workers), "--force",
        ),
        module(
            "publication.adaptive_control", "experiments",
            "--workers", str(min(workers, 8)), "--force",
        ),
    ]


def artifact_commands() -> list[tuple[str, ...]]:
    """Commands that render derived artifacts from existing numerical data."""
    return [
        module(
            "publication.scalar_validation", "--skip-experiments",
            "--only", "nontrivial",
        ),
        module(
            "publication.scalar_validation", "--skip-experiments",
            "--only", "controls",
        ),
        module(
            "publication.scalar_validation", "--skip-experiments",
            "--only", "targets",
        ),
        module(
            "publication.scalar_validation", "--skip-experiments",
            "--only", "finite",
        ),
        module("publication.bloch_gap", "--mode", "tables"),
        module(
            "publication.render_figures",
            "--figures",
            "scalar", "bloch",
        ),
        module("publication.propagation", "render"),
        module("publication.adaptive_control", "render"),
    ]


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    # Stable metadata for byte-for-byte figure rebuilds.
    env.setdefault("SOURCE_DATE_EPOCH", "1787961600")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SCRIPTS_DIR}{os.pathsep}{existing}" if existing else str(SCRIPTS_DIR)
    )
    return env


def run(commands: Iterable[Sequence[str]]) -> None:
    env = environment()
    for command in commands:
        printable = " ".join(command)
        print(f"\n>>> {printable}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def prepare_data() -> None:
    if all(path.exists() for path in REQUIRED_DATA):
        return
    if DATA_ROOT != SCRIPTS_DIR / "outputs":
        missing = [str(path) for path in REQUIRED_DATA if not path.exists()]
        raise FileNotFoundError(
            "LSL_DATA_DIR is incomplete; missing:\n  " + "\n  ".join(missing)
        )
    print("Restoring curated numerical data from data/publication-data.tar.gz")
    extract_publication_data()


def check_release() -> None:
    prepare_data()
    run((module("validation.release"),))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        choices=("check", "artifacts", "figures", "experiments", "full"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(40, os.cpu_count() or 1),
        help="worker processes for propagation calculations (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    if args.workflow == "check":
        check_release()
    elif args.workflow in {"artifacts", "figures"}:
        prepare_data()
        run(artifact_commands())
    elif args.workflow == "experiments":
        run(experiment_commands(args.workers))
    elif args.workflow == "full":
        run(experiment_commands(args.workers))
        run(artifact_commands())


if __name__ == "__main__":
    main()
