"""Fast integrity checks for the code and numerical-data release.

This is deliberately a plain Python program rather than a test-runner suite so
that a reader can run ``make check`` immediately after installing the runtime
requirements.  It validates the curated data archive, central numerical
results, deterministic network construction, and reciprocal-gradient identities.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys

import matplotlib
import networkx
import numpy as np
import scipy
import yaml
from mechanical.topology import make_network

from validation.damped_reciprocity import run_checks as check_damped_reciprocity
from validation.data_archive import REPO_ROOT
from validation.data_archive import check as check_data_archive

PUBLICATION_MODULES = (
    "publication.scalar_gap",
    "publication.scalar_validation",
    "publication.nonspatial",
    "publication.bloch_gap",
    "publication.propagation",
    "publication.adaptive_control",
    "publication.render_figures",
)
NETWORK_FIXTURES = {
    (5, 0): "1e87a596b9c23a56899c3e71b226a877993e581684086d805f68da4ff39edee2",
    (5, 99): "9127b6c7e11e93cc9d3922f4172d720a165115de8d9479513add1826172a564f",
    (20, 5): "da5c34d2127bfc6701834170955f8e8afeda2cb42623b2bcdba7b958aaf53648",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(relative: str):
    return json.loads((REPO_ROOT / relative).read_text())


def check_imports() -> None:
    for name in PUBLICATION_MODULES:
        importlib.import_module(name)
    print(f"PASS imports: {len(PUBLICATION_MODULES)} publication modules")


def check_network_generator() -> None:
    for (size, seed), expected in NETWORK_FIXTURES.items():
        arrays = make_network("rand-del", size=size, seed=seed)
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(np.ascontiguousarray(array).tobytes())
        require(
            digest.hexdigest() == expected,
            f"periodic network fixture changed for size={size}, seed={seed}",
        )
    print(f"PASS periodic generator: {len(NETWORK_FIXTURES)} exact fixtures")


def check_reciprocal_damping() -> None:
    results = check_damped_reciprocity()
    largest = max(value for metrics in results.values() for value in metrics.values())
    print(f"PASS damped reciprocity and gradients: max error={largest:.2e}")


def check_headline_results() -> None:
    scalar = read_json("scripts/outputs/prl_bandgap/prl_figure_summary.json")
    require(scalar["fig1"]["n_in_initial"] == 119, "unexpected scalar initial count")
    require(scalar["fig1"]["n_in_final"] == 0, "scalar prescribed spectral window is not empty")
    require(
        scalar["gradient_check"]["gradient_max_relative_error"] < 2.0e-6,
        "scalar analytic-gradient check exceeds tolerance",
    )

    refined = read_json("scripts/outputs/prl_vector_periodic/refined_gap.json")
    gap = refined["refined_gap"]
    require(gap["lower_band_1based"] == 25, "unexpected lower Bloch band")
    require(gap["upper_band_1based"] == 26, "unexpected upper Bloch band")
    require(0.380 < gap["normalized_gap"] < 0.382, "unexpected normalized Bloch gap")
    require(gap["contains_target_window"], "Bloch gap does not contain the prescribed spectral window")

    exemplar_path = (
        REPO_ROOT / "scripts/outputs/prl_vector_periodic/"
        "vector_periodic_s5_net0_train0_c50_w10.npz"
    )
    with np.load(exemplar_path, allow_pickle=False) as exemplar:
        require(
            str(exemplar["material_update"]) == "response_conditioned_log",
            "periodic exemplar does not use the response-conditioned material law",
        )
        require(int(exemplar["n_steps"]) == 3000, "unexpected periodic training length")
        require(
            np.isnan(float(exemplar["grad_clip"])),
            "periodic exemplar unexpectedly uses gradient clipping",
        )
        require(
            np.isclose(float(exemplar["inflation_strength"]), 0.0),
            "periodic exemplar unexpectedly uses inflation",
        )

    controls = read_json("scripts/outputs/prl_vector_periodic/vector_controls.json")
    control_success = {
        row["control"]: row["success_rate"] for row in controls["summary"]
    }
    require(control_success.pop("paired-response") == 1.0, "paired response control failed")
    require(
        control_success.pop("paired-no-bounds") == 1.0,
        "unbounded paired-response control failed",
    )
    require(
        all(rate == 0.0 for rate in control_success.values()),
        "a periodic lesion or uniform-material control unexpectedly succeeded",
    )

    sizes = read_json("scripts/outputs/prl_vector_periodic/size_scan_summary.json")
    require([row["size"] for row in sizes] == [5, 6, 8], "unexpected size ensemble")
    require(all(row["success"] == row["n"] == 10 for row in sizes), "size scan failed")

    propagation = read_json("scripts/outputs/prl_propagation/summary.json")
    suppression = propagation["learned_minus_initial_db"]
    require(propagation["n_materials"] == 10, "unexpected propagation ensemble")
    require(propagation["n_coordinate_sources"] == 500, "unexpected source count")
    require(suppression["median"] < -75.0, "median suppression is too small")
    require(suppression["max"] < -32.0, "worst-case suppression is too small")

    damping = read_json("scripts/outputs/prl_v7_adaptive/scalar_damping_scan.json")
    require(
        all(row["success"] == row["n"] == 10 for row in damping["summary"]),
        "scalar damping scan failed",
    )
    absolute = read_json("scripts/outputs/prl_v7_adaptive/absolute_windows.json")
    require(
        all(row["success"] == row["n"] == 10 for row in absolute["summary"]),
        "absolute-window control failed",
    )
    require(
        all(row["online_eigensolves"] == 0 for row in absolute["rows"]),
        "absolute-window control used an online eigensolve",
    )
    retarget = read_json("scripts/outputs/prl_v7_adaptive/retargeting_summary.json")
    require(retarget["counts"]["after_A"]["A"] == 0, "first A phase failed")
    require(retarget["counts"]["after_B"]["B"] == 0, "B phase failed")
    require(retarget["counts"]["after_return_A"]["A"] == 0, "return to A failed")
    vector_damping = read_json("scripts/outputs/prl_v7_adaptive/vector_damping_scan.json")
    require(all(row["success"] for row in vector_damping["rows"]), "Bloch damping scan failed")
    calibration = read_json(
        "scripts/outputs/prl_v7_adaptive/stiffness_calibration_disorder.json"
    )
    require(calibration["success"] == 9, "unexpected calibration-disorder success count")
    require(
        all(not row["calibration_supplied_to_controller"] for row in calibration["rows"]),
        "calibration factors were supplied to a local controller",
    )
    print("PASS numerical claims: gaps, damping, absolute windows, calibration disorder, retargeting, and propagation")


def check_repository_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    require("scripts/reproduce.py" in readme, "README omits the reproduction entry point")
    require("publication-data.tar.gz" in readme, "README omits the archived data")
    require((REPO_ROOT / "requirements-lock.txt").exists(), "missing locked environment")
    require((REPO_ROOT / "CITATION.cff").exists(), "missing citation metadata")
    require((REPO_ROOT / "LICENSE").exists(), "missing software license")
    print("PASS repository contract: documentation, environment, citation, license")


def print_environment() -> None:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "networkx": networkx.__version__,
        "pyyaml": yaml.__version__,
    }
    print("Environment: " + ", ".join(f"{key}={value}" for key, value in versions.items()))


def main() -> None:
    print_environment()
    check_imports()
    check_data_archive()
    check_network_generator()
    check_reciprocal_damping()
    check_headline_results()
    check_repository_contract()
    print("PASS code/data release checks")


if __name__ == "__main__":
    main()
