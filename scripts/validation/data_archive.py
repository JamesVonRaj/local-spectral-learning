"""Build, verify, or extract the curated publication-data archive.

The working ``scripts/outputs`` directory contains exploratory runs in addition
to the data used by the paper.  This module selects only the evidence needed to
render every retained figure and table, records a SHA-256 digest for every
file, and writes a deterministic compressed archive for a public release.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "scripts" / "outputs"
DATA_DIR = REPO_ROOT / "data"
ARCHIVE = DATA_DIR / "publication-data.tar.gz"
MANIFEST = DATA_DIR / "manifest.json"

# These patterns are intentionally explicit.  They exclude exploratory logs,
# diagnostic plots, smoke-test caches, and inactive supplemental experiments
# while retaining the numerical evidence behind every cited paper asset.
PUBLICATION_FILES = {
    "prl_bandgap": (
        "fig1_exemplar.npz",
        "fig1_exemplar_config.json",
        "fig2_gradient_check.npz",
        "prl_figure_summary.json",
    ),
    "prl_nonspatial": ("ws_screen.npz",),
    "prl_propagation": (
        "summary.json",
        "convergence_net0_nk*.npz",
        "spectrum_s5_net0_nk129_*.npz",
        "target_s5_net0_nk129_*.npz",
        "target_s5_net[0-9]_nk81_*.npz",
        "trained_cells/vector_periodic_s5_net[0-9]_train0_c50_w10.npz",
    ),
    "prl_readiness": (
        "reciprocal_nontrivial_gap_runs.npz",
        "expanded_controls_real.npz",
        "finite_size_scaling_real.npz",
        "target_robustness_real.npz",
    ),
    "prl_vector_periodic": (
        "bz_convergence.json",
        "bz_convergence.npz",
        "latest_vector_periodic_validation.json",
        "refined_gap.json",
        "size_scan_summary.json",
        "vector_controls.json",
        "vector_controls.npz",
        "vector_gap_ensemble.json",
        "vector_gap_ensemble.npz",
        "vector_periodic_s5_net[0-9]_train0_c50_w10.json",
        "vector_periodic_s5_net[0-9]_train0_c50_w10.npz",
        "vector_periodic_s6_net[0-9]_train0_c50_w10.json",
        "vector_periodic_s6_net[0-9]_train0_c50_w10.npz",
        "vector_periodic_s8_net[0-9]_train0_c50_w10.json",
        "vector_periodic_s8_net[0-9]_train0_c50_w10.npz",
    ),
    "prl_v7_adaptive": (
        "absolute_windows.json",
        "retargeting.npz",
        "retargeting_summary.json",
        "scalar_damping_scan.json",
        "stiffness_calibration_disorder.json",
        "vector_damping_radii.npz",
        "vector_damping_scan.json",
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def selected_files() -> list[Path]:
    """Return the unique, sorted set of files included in a release."""
    paths: set[Path] = set()
    missing: list[str] = []
    for family, patterns in PUBLICATION_FILES.items():
        root = OUTPUT_ROOT / family
        for pattern in patterns:
            matches = [path for path in root.glob(pattern) if path.is_file()]
            if not matches:
                missing.append(f"{family}/{pattern}")
            paths.update(matches)
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"publication data are missing:\n  {joined}")
    return sorted(paths, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def manifest_record(path: Path) -> dict[str, object]:
    relative = path.relative_to(REPO_ROOT).as_posix()
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def deterministic_archive(paths: Iterable[Path], archive: Path) -> None:
    """Write a byte-reproducible gzip-compressed POSIX tar archive."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:  # noqa: SIM117 - Python 3.8 compatible
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in paths:
                    relative = path.relative_to(REPO_ROOT).as_posix()
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    info.mtime = 0
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)


def build() -> None:
    paths = selected_files()
    deterministic_archive(paths, ARCHIVE)
    records = [manifest_record(path) for path in paths]
    payload = {
        "schema_version": 1,
        "description": "Numerical data used to verify results and render derived artifacts.",
        "archive": {
            "path": ARCHIVE.relative_to(REPO_ROOT).as_posix(),
            "bytes": ARCHIVE.stat().st_size,
            "sha256": digest(ARCHIVE),
        },
        "files": records,
        "totals": {
            "files": len(records),
            "uncompressed_bytes": sum(int(record["bytes"]) for record in records),
        },
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {ARCHIVE.relative_to(REPO_ROOT)} "
        f"({ARCHIVE.stat().st_size / 1024**2:.2f} MiB, {len(records)} files)"
    )


def load_manifest() -> dict[str, object]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing {MANIFEST.relative_to(REPO_ROOT)}")
    return json.loads(MANIFEST.read_text())


def check(check_extracted: bool = True) -> None:
    payload = load_manifest()
    archive_info = payload["archive"]
    expected_archive = str(archive_info["sha256"])
    actual_archive = digest(ARCHIVE)
    if actual_archive != expected_archive:
        raise ValueError(
            f"archive digest mismatch: expected {expected_archive}, got {actual_archive}"
        )

    if check_extracted:
        problems = []
        for record in payload["files"]:
            path = REPO_ROOT / str(record["path"])
            if not path.exists():
                problems.append(f"missing {record['path']}")
            elif path.stat().st_size != int(record["bytes"]):
                problems.append(f"size mismatch {record['path']}")
            elif digest(path) != str(record["sha256"]):
                problems.append(f"digest mismatch {record['path']}")
        if problems:
            raise ValueError("data-manifest failures:\n  " + "\n  ".join(problems))
    print(
        f"PASS data archive: {payload['totals']['files']} files, "
        f"SHA-256 {actual_archive[:12]}..."
    )


def extract() -> None:
    """Safely restore publication data into an otherwise clean checkout."""
    check(check_extracted=False)
    root = REPO_ROOT.resolve()
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            destination = (REPO_ROOT / member.name).resolve()
            if root not in destination.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
            if not member.isfile():
                raise ValueError(f"unexpected non-file archive member: {member.name}")
        for member in members:
            destination = REPO_ROOT / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            destination.write_bytes(source.read())
    check(check_extracted=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "extract"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build()
    elif args.command == "check":
        check()
    else:
        extract()


if __name__ == "__main__":
    main()
