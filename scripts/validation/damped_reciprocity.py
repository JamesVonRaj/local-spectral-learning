"""Deterministic checks for reciprocal learning with physical damping.

The checks compare the phase-conjugate re-drive against a conventional
algebraic adjoint, verify scalar and Bloch gradients by finite differences,
and confirm that the viscous implementation reduces to the v6 lossless rule
at zero damping.
"""

from __future__ import annotations

import mechanical.learners_local as learners_local
import numpy as np
from mechanical.learners import (
    REAL_SHIFT,
    VISCOUS_DAMPING,
    build_K,
    local_step_with_moments,
)
from mechanical.topology import make_network
from publication.bloch_gap import (
    bloch_stiffness,
    edge_extensions,
    local_bloch_step,
    make_periodic_vector_cell,
    response_matrix,
)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(expected)), 1.0e-14)
    return float(np.linalg.norm(actual - expected) / scale)


def _centered_difference(cost, values: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (cost(plus) - cost(minus)) / (2.0 * step)
    return result


def check_scalar() -> dict[str, float]:
    pos, edges, lengths, _ = make_network("rand-del", size=4, seed=7)
    n_nodes = len(pos)
    radii = 0.82 + 0.31 * np.random.RandomState(4).rand(len(edges))
    omega = 1.47
    gamma = 0.08 * omega
    force = np.random.RandomState(9).normal(size=n_nodes)

    def fields(test_radii: np.ndarray):
        stiffness = build_K(edges, test_radii, lengths, n_nodes).toarray()
        operator = (
            stiffness - omega**2 * np.eye(n_nodes)
            + 1j * gamma * omega * np.eye(n_nodes)
        )
        displacement = np.linalg.solve(operator, force)
        adjoint = np.linalg.solve(operator.conjugate().T, displacement)
        physical_second = np.linalg.solve(operator, np.conjugate(displacement))
        return operator, displacement, adjoint, physical_second

    operator, displacement, adjoint, physical_second = fields(radii)
    adjoint_error = _relative_error(physical_second, np.conjugate(adjoint))
    i_edge, j_edge = edges[:, 0], edges[:, 1]
    du = displacement[i_edge] - displacement[j_edge]
    dw = physical_second[i_edge] - physical_second[j_edge]
    analytic = -(4.0 * radii / lengths) * np.real(dw * du)
    finite = _centered_difference(
        lambda test_radii: float(np.vdot(fields(test_radii)[1], fields(test_radii)[1]).real),
        radii,
    )
    gradient_error = _relative_error(analytic, finite)

    eigenvalues = np.linalg.eigvalsh(operator.real + omega**2 * np.eye(n_nodes))
    spectral_trace = np.sum(
        1.0 / ((eigenvalues - omega**2) ** 2 + gamma**2 * omega**2)
    )
    direct_trace = np.trace(
        np.linalg.inv(operator).conjugate().T @ np.linalg.inv(operator)
    ).real
    trace_error = abs(float(direct_trace - spectral_trace)) / float(spectral_trace)

    arguments = dict(
        edges=edges,
        radii=radii,
        lengths=lengths,
        N=n_nodes,
        M_node=1.0,
        omegas=np.array([omega]),
        batch=3,
        force_distribution="gaussian",
    )
    lossless = local_step_with_moments(
        rng=np.random.RandomState(12), damping=REAL_SHIFT, **arguments
    )
    zero_damping = local_step_with_moments(
        rng=np.random.RandomState(12), damping=VISCOUS_DAMPING,
        damping_gamma=0.0, **arguments
    )
    zero_limit_error = max(
        abs(float(lossless[0] - zero_damping[0])) / max(abs(float(lossless[0])), 1.0),
        _relative_error(lossless[1], zero_damping[1]),
        *(
            _relative_error(lossless[2][key], zero_damping[2][key])
            for key in ("U", "V", "D")
        ),
    )
    damped_moments = local_step_with_moments(
        rng=np.random.RandomState(12), damping=VISCOUS_DAMPING,
        damping_gamma=gamma, **arguments
    )[2]
    local_lambda = 0.06
    denominator = damped_moments["U"] + local_lambda**2 * damped_moments["V"]
    drive = np.divide(
        2.0 * local_lambda * damped_moments["D"], denominator,
        out=np.zeros_like(denominator), where=denominator > 0.0,
    )
    drive_excess = max(0.0, float(np.max(np.abs(drive))) - 1.0)

    return {
        "adjoint_error": adjoint_error,
        "gradient_error": gradient_error,
        "trace_error": trace_error,
        "zero_damping_error": zero_limit_error,
        "bounded_drive_excess": drive_excess,
    }


def check_bloch() -> dict[str, float]:
    cell = make_periodic_vector_cell(size=3, seed=5)
    radii = 0.84 + 0.27 * np.random.RandomState(6).rand(cell.n_edges)
    kred = np.array([0.71, -1.13])
    omega = 1.38
    gamma = 0.07 * omega
    force = (
        np.random.RandomState(14).normal(size=(cell.n_dof, 2))
        + 1j * np.random.RandomState(15).normal(size=(cell.n_dof, 2))
    ) / np.sqrt(2.0 * cell.n_dof)

    def fields(test_radii: np.ndarray):
        operator = response_matrix(
            cell, test_radii, kred, omega, 0.0, "viscous",
            damping_gamma=gamma,
        )
        opposite_operator = response_matrix(
            cell, test_radii, -kred, omega, 0.0, "viscous",
            damping_gamma=gamma,
        )
        displacement = np.linalg.solve(operator, force)
        adjoint = np.linalg.solve(operator.conjugate().T, displacement)
        physical_second = np.linalg.solve(
            opposite_operator, np.conjugate(displacement)
        )
        return operator, opposite_operator, displacement, adjoint, physical_second

    operator, opposite, displacement, adjoint, physical_second = fields(radii)
    reciprocity_error = _relative_error(operator.T, opposite)
    adjoint_error = _relative_error(physical_second, np.conjugate(adjoint))
    du = edge_extensions(cell, displacement, kred)
    dw = edge_extensions(cell, physical_second, -kred)
    analytic = -(4.0 * radii / cell.lengths) * np.sum(
        np.real(dw * du), axis=1
    )
    finite = _centered_difference(
        lambda test_radii: float(
            np.sum(np.abs(fields(test_radii)[2]) ** 2).real
        ),
        radii,
    )
    gradient_error = _relative_error(analytic, finite)

    arguments = dict(
        cell=cell,
        radii=radii,
        kpoints=np.array([kred]),
        omegas=np.array([omega]),
        force_batch=2,
        eps_reg=0.0,
        update_rule="adjoint",
        return_moments=True,
    )
    lossless = local_bloch_step(
        rng=np.random.RandomState(18), response_mode="reciprocal", **arguments
    )
    zero_damping = local_bloch_step(
        rng=np.random.RandomState(18), response_mode="viscous",
        damping_gamma=0.0, **arguments
    )
    zero_limit_error = max(
        abs(float(lossless[0] - zero_damping[0])) / max(abs(float(lossless[0])), 1.0),
        _relative_error(lossless[1], zero_damping[1]),
        *(
            _relative_error(lossless[2][key], zero_damping[2][key])
            for key in ("U", "V", "D")
        ),
    )
    hermitian_error = _relative_error(
        bloch_stiffness(cell, radii, kred).conjugate().T,
        bloch_stiffness(cell, radii, kred),
    )
    return {
        "reciprocity_error": reciprocity_error,
        "adjoint_error": adjoint_error,
        "gradient_error": gradient_error,
        "zero_damping_error": zero_limit_error,
        "hermitian_error": hermitian_error,
    }


def check_eigensolve_free_schedule() -> dict[str, float]:
    pos, edges, lengths, _ = make_network("rand-del", size=3, seed=2)
    original = learners_local.eigenfreqs
    calls = 0

    def forbidden_eigensolve(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("scheduled absolute-window learning called an eigensolver")

    learners_local.eigenfreqs = forbidden_eigensolve
    try:
        result = learners_local.train_local(
            edges, lengths, len(pos),
            n_steps=6,
            batch=1,
            n_freq=2,
            material_update="response_conditioned_log",
            regularizers=[],
            target_window_schedule=[
                (0, 1.20, 1.50),
                (2, 1.70, 2.00),
                (4, 1.20, 1.50),
            ],
            damping="viscous",
            damping_gamma=0.02,
            spectrum_diagnostics=False,
            snapshot_every=2,
            eval_every=1,
        )
    finally:
        learners_local.eigenfreqs = original
    expected_steps = np.array([0, 2, 4, 6])
    expected_final_lambda = 0.025 * (1.50**2 - 1.20**2)
    schedule_error = float(
        not np.array_equal(result["radius_snapshot_steps"], expected_steps)
        or result["window_schedule"].shape != (3, 3)
        or result["f0"].size != 0
        or result["ff"].size != 0
        or not np.isclose(result["response_metric_lambda"], expected_final_lambda)
    )
    return {
        "eigensolver_calls": float(calls),
        "schedule_error": schedule_error,
    }


def run_checks(tolerance: float = 2.0e-6) -> dict[str, dict[str, float]]:
    results = {
        "scalar": check_scalar(),
        "bloch": check_bloch(),
        "absolute_schedule": check_eigensolve_free_schedule(),
    }
    for family, metrics in results.items():
        for name, value in metrics.items():
            if value > tolerance:
                raise AssertionError(
                    f"{family} damped-reciprocity check {name}={value:.3e} "
                    f"exceeds {tolerance:.1e}"
                )
    return results


def main() -> None:
    results = run_checks()
    summary = ", ".join(
        f"{family} max={max(metrics.values()):.2e}"
        for family, metrics in results.items()
    )
    print(f"PASS damped reciprocity: {summary}")


if __name__ == "__main__":
    main()
