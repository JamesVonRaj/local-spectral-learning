"""Legacy global-constraint band-gap training with splu factorization caching.

This module is kept for old projection/conservation baselines.  New active
band-gap runs should use ``learners_local.train_local`` unless a legacy
comparison is being made explicitly.

For mass-proportional viscous damping, reciprocity realizes the adjoint in the
same physical network.  With

    H_gamma(omega) = K - omega^2 M + i gamma omega M,
    H_gamma u = F,
    H_gamma w = conj(u),

the second physical response is ``w = conj(v)``, where
``H_gamma^H v = u``.  The exact local rule is therefore

    g_e = -2 Re[(2 r_e / L_e) (w_i - w_j) (u_i - u_j)].

The legacy ``complex`` mode instead uses a small numerical term
``i eps_reg omega^2 I`` and an explicit adjoint solve.  It is retained only for
backward-compatible controls and is not the physical damping model.

The production reciprocal variant keeps all fields real and uses the bare
undamped operator (``eps_reg`` applies only to complex damping; the
``real_shift`` mode name is historical):
    g_e = -2 (2 r_e / L_e) (v_i - v_j) (u_i - u_j)
with u = H^{-1} F and v = H^{-1} u,
    H(omega) = K - omega^2 M.

The cost gradient is augmented by a list of strictly-local regularizers
(see ``regularizers.py``) and then projected + rescaled onto a global
conservation constraint (see ``constraints.py``).  The original
``vanilla`` / ``rad-L1`` / ``mass-cons`` modes correspond to:

    vanilla   : regularizers=[],                constraint=stiffness_conservation()
    rad-L1    : regularizers=[l1_sparsity(0.02)], constraint=stiffness_conservation()
    mass-cons : regularizers=[],                constraint=mass_conservation()

For the strictly-local variant (no constraint, default inflation
regularizer), see ``learners_local.train_local``.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from .constraints import stiffness_conservation
from .objectives import (
    adaptive_pressure_history,
    adaptive_pressure_state_from_spec,
    apply_source,
    calibrate_target_distribution_terms,
    default_objective_terms,
    evaluate_term,
    objective_term_arrays,
    objective_terms_from_spec,
    term_response,
    update_adaptive_pressure_terms,
)

_DEFAULT_CONSTRAINT = object()
REAL_SHIFT = 'real_shift'
COMPLEX_DAMPING = 'complex'
VISCOUS_DAMPING = 'viscous'


def build_K(edges, radii, lengths, N):
    ke = radii**2 / lengths
    i, j = edges[:, 0], edges[:, 1]
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    vals = np.concatenate([ke, ke, -ke, -ke])
    return coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def eigenfreqs(edges, radii, lengths, N, M_node):
    K = build_K(edges, radii, lengths, N)
    lam = eigh(K.toarray(), np.diag(np.full(N, M_node)), eigvals_only=True)
    return np.sqrt(np.maximum(lam, 0))


def local_step(edges, radii, lengths, N, M_node, omegas, rng,
               batch=10, eps_reg=1e-6, damping=REAL_SHIFT,
               damping_gamma=0.0,
               objective_terms=None, return_term_responses=False,
               force_distribution='gaussian'):
    """One training step.  Factor each H(omega) once, reuse across batch.

    ``objective_terms`` enables generalized real-valued response losses.
    When omitted, this is exactly the original band-gap step:
    ``sum_omega ||H(omega)^-1 F||^2``.
    """
    if objective_terms is None:
        cost, grad = _local_step_legacy(
            edges, radii, lengths, N, M_node, omegas, rng,
            batch=batch, eps_reg=eps_reg, damping=damping,
            damping_gamma=damping_gamma,
            force_distribution=force_distribution,
        )
        if return_term_responses:
            return cost, grad, np.zeros(0, dtype=np.float64)
        return cost, grad
    elif damping != REAL_SHIFT:
        raise ValueError('generalized objective_terms currently require '
                         'damping="real_shift"')

    K = build_K(edges, radii, lengths, N)
    M_sp = diags(np.full(N, M_node))
    i_e, j_e = edges[:, 0], edges[:, 1]
    Me = len(edges)
    n_terms = len(objective_terms)

    terms_by_omega = {}
    for idx, term in enumerate(objective_terms):
        terms_by_omega.setdefault(float(term.omega), []).append((idx, term))

    solvers = {}
    for omega in terms_by_omega:
        H = (K - omega**2 * M_sp).tocsc()
        solvers[omega] = (splu(H), None)

    inv_bn = 1.0 / (batch * n_terms)
    cost = 0.0
    grad = np.zeros(Me)
    term_responses = (np.zeros(n_terms, dtype=np.float64)
                      if return_term_responses else None)
    two_r_over_L = 2 * radii / lengths

    for _ in range(batch):
        F = _random_force(rng, N, force_distribution)
        for omega, terms in terms_by_omega.items():
            solv, solv_adj = solvers[omega]
            solved_u = []
            for idx, term in terms:
                rhs = apply_source(term, F)
                u = solv.solve(rhs)
                solved_u.append((idx, term, u))

            for idx, term, u in solved_u:
                if solv_adj is None:
                    response = term_response(term, u)
                    if term_responses is not None:
                        term_responses[idx] += response / batch
                    term_cost, adj_rhs = evaluate_term(
                        term, u, response_value=response,
                    )
                    v = solv.solve(adj_rhs)
                    cost += float(term_cost) * inv_bn
                    du = u[i_e] - u[j_e]
                    dv = v[i_e] - v[j_e]
                    grad += (-1.0 * inv_bn) * two_r_over_L * dv * du
                else:
                    # Back-compat path for the original complex-damped
                    # squared-norm objective only.
                    v = solv_adj.solve(u)
                    cost += float(np.vdot(u, u).real) * inv_bn
                    du = u[i_e] - u[j_e]
                    dv = v[i_e] - v[j_e]
                    grad += (-2.0 * inv_bn) * two_r_over_L * np.real(np.conj(dv) * du)
    if return_term_responses:
        return cost, grad, term_responses
    return cost, grad


def local_step_with_moments(edges, radii, lengths, N, M_node, omegas, rng,
                            batch=10, eps_reg=1e-6, damping=REAL_SHIFT,
                            damping_gamma=0.0,
                            force_distribution='gaussian'):
    """Squared-response step plus edge-local response moments.

    Returns ``(cost, grad_r, moments)`` with

    ``U_e = <|du_e|^2>``, ``V_e = <|dv_e|^2>``, and
    ``D_e = <Re(conj(dv_e) du_e)>``.

    The same force and frequency weights are used for all three moments, so
    ``|D_e| <= sqrt(U_e V_e)``.  These quantities are already available at
    edge ``e`` during the paired response and support bounded local material
    kinetics without a network-wide gradient norm.
    """
    return _local_step_legacy(
        edges, radii, lengths, N, M_node, omegas, rng,
        batch=batch, eps_reg=eps_reg, damping=damping,
        damping_gamma=damping_gamma,
        force_distribution=force_distribution, return_edge_moments=True,
    )


def objective_cost(edges, radii, lengths, N, M_node, objective_terms, forces,
                   eps_reg=1e-6, damping=REAL_SHIFT, damping_gamma=0.0):
    """Cost-only objective estimate on a fixed force batch.

    This is intended for monitoring.  It uses only forward response solves,
    so it is cheaper and less noisy than reading the stochastic training
    mini-batch cost at every step.
    """
    if damping not in {REAL_SHIFT, VISCOUS_DAMPING}:
        raise ValueError('objective_cost supports undamped and viscous modes')
    damping_gamma = float(damping_gamma)
    if damping == VISCOUS_DAMPING and damping_gamma < 0.0:
        raise ValueError('damping_gamma must be nonnegative')
    K = build_K(edges, radii, lengths, N)
    M_sp = diags(np.full(N, M_node))
    n_terms = len(objective_terms)
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != N:
        raise ValueError(f'forces must have shape (batch, {N})')

    terms_by_omega = {}
    for term in objective_terms:
        terms_by_omega.setdefault(float(term.omega), []).append(term)

    solvers = {}
    for omega in terms_by_omega:
        H = K - omega**2 * M_sp
        if damping == VISCOUS_DAMPING:
            H = H + 1j * damping_gamma * omega * M_sp
        H = H.tocsc()
        solvers[omega] = splu(H)

    inv_bn = 1.0 / (len(forces) * n_terms)
    cost = 0.0
    for F in forces:
        for omega, terms in terms_by_omega.items():
            solv = solvers[omega]
            for term in terms:
                u = solv.solve(apply_source(term, F))
                if damping == VISCOUS_DAMPING:
                    if (
                        term.transform != 'quadratic'
                        or term.readout_weights is not None
                    ):
                        raise ValueError(
                            'viscous monitoring currently supports only the '
                            'full-field quadratic response objective'
                        )
                    cost += (
                        float(term.coefficient)
                        * float(np.vdot(u, u).real)
                        * inv_bn
                    )
                else:
                    term_cost, _ = evaluate_term(term, u)
                    cost += float(term_cost) * inv_bn
    return cost


def _local_step_legacy(edges, radii, lengths, N, M_node, omegas, rng,
                       batch=10, eps_reg=1e-6, damping=REAL_SHIFT,
                       damping_gamma=0.0,
                       force_distribution='gaussian',
                       return_edge_moments=False):
    """Original squared-response step, kept for exact back-compat."""
    K = build_K(edges, radii, lengths, N)
    M_sp = diags(np.full(N, M_node))
    I_sp = diags(np.ones(N))
    i_e, j_e = edges[:, 0], edges[:, 1]
    Me = len(edges)
    n_freq = len(omegas)
    damping_gamma = float(damping_gamma)
    if damping == VISCOUS_DAMPING and damping_gamma < 0.0:
        raise ValueError('damping_gamma must be nonnegative')

    solvers = []
    for omega in omegas:
        if damping == REAL_SHIFT:
            H = (K - omega**2 * M_sp).tocsc()
            solvers.append((splu(H), None, 'real'))
        elif damping == COMPLEX_DAMPING:
            H = (K - omega**2 * M_sp
                 + (1j * eps_reg * omega**2) * I_sp).tocsc()
            solvers.append((
                splu(H), splu(H.conjugate().transpose().tocsc()),
                'sesquilinear',
            ))
        elif damping == VISCOUS_DAMPING:
            H = (
                K - omega**2 * M_sp
                + (1j * damping_gamma * omega) * M_sp
            ).tocsc()
            # Reciprocity gives H^T = H in real space.  Solving the same
            # physical operator with conj(u) returns w = conj(v), where
            # H^H v = u is the conventional adjoint equation.
            solvers.append((splu(H), None, 'bilinear'))
        else:
            raise ValueError(f'unknown damping mode {damping!r}; '
                             f'valid: {REAL_SHIFT!r}, {COMPLEX_DAMPING!r}, '
                             f'{VISCOUS_DAMPING!r}')

    inv_bn = 1.0 / (batch * n_freq)
    cost = 0.0
    grad = np.zeros(Me)
    moment_u = np.zeros(Me) if return_edge_moments else None
    moment_v = np.zeros(Me) if return_edge_moments else None
    moment_d = np.zeros(Me) if return_edge_moments else None
    two_r_over_L = 2 * radii / lengths

    for _ in range(batch):
        F = _random_force(rng, N, force_distribution)
        for solv, solv_adj, product_mode in solvers:
            u = solv.solve(F)
            if product_mode == 'real':
                second = solv.solve(u)
                cost += float(np.sum(u * u)) * inv_bn
            elif product_mode == 'bilinear':
                second = solv.solve(np.conjugate(u))
                cost += float(np.vdot(u, u).real) * inv_bn
            else:
                second = solv_adj.solve(u)
                cost += float(np.vdot(u, u).real) * inv_bn
            du = u[i_e] - u[j_e]
            dsecond = second[i_e] - second[j_e]
            if product_mode == 'real':
                edge_product = dsecond * du
            elif product_mode == 'bilinear':
                edge_product = np.real(dsecond * du)
            else:
                edge_product = np.real(np.conj(dsecond) * du)
            grad += (-2.0 * inv_bn) * two_r_over_L * edge_product
            if return_edge_moments:
                moment_u += inv_bn * np.real(np.conj(du) * du)
                moment_v += inv_bn * np.real(np.conj(dsecond) * dsecond)
                moment_d += inv_bn * edge_product
    if return_edge_moments:
        return cost, grad, {
            'U': moment_u,
            'V': moment_v,
            'D': moment_d,
        }
    return cost, grad


def train(edges, lengths, N, M_node=1.0,
          R_INIT=1.0, R_MIN=0.5, R_MAX=2.0,
          n_steps=3000, batch=10, n_freq=8,
          alpha=0.05, grad_clip=3.0, damping=REAL_SHIFT,
          force_distribution='gaussian',
          regularizers=None, constraint=_DEFAULT_CONSTRAINT,
          objective=None, pos=None, box=None,
          train_seed=0, warmup_steps=0,
          eval_every=500,
          monitor_batch=0, monitor_every=None, monitor_seed=None,
          monitor_force_distribution=None,
          snapshot_every=0, snapshot_dtype='float32'):
    """Run the full training loop and return a dict of results.

    Parameters
    ----------
    regularizers : list of callables, optional
        Each callable has signature ``reg(radii, lengths) -> ndarray`` of
        shape ``(M_edges,)`` and returns a per-edge gradient
        contribution.  Contributions are summed into the cost gradient
        before constraint projection and clipping.  ``None`` defaults to
        ``[]``.  See ``mechanical.regularizers`` for ready-made
        factories (inflation, l1_sparsity, l2_pull).
    constraint : constraint object, optional
        Conservation constraint with ``bind``, ``project``, ``enforce``
        methods.  Omitted defaults to ``stiffness_conservation()``.
        Explicit ``None`` disables projection/rescaling.  See
        ``mechanical.constraints``.  For the strictly-local variant with
        only per-edge regularizers, use ``learners_local.train_local``.
    """
    if regularizers is None:
        regularizers = []
    if constraint is _DEFAULT_CONSTRAINT:
        constraint = stiffness_conservation()

    Me = len(edges)
    radii = np.full(Me, R_INIT, dtype=np.float64)
    rng = np.random.RandomState(train_seed)

    # Initial spectrum and target/evaluation window.  With objective=None
    # this is the original [P35, P65] band-gap objective.
    f0 = eigenfreqs(edges, radii, lengths, N, M_node)
    if objective is None:
        objective_terms, wlo, whi = default_objective_terms(f0, n_freq=n_freq)
        objective_terms_for_step = None
        adaptive_pressure_state = None
        target_distribution_initial_response = None
    else:
        objective_terms, wlo, whi = objective_terms_from_spec(
            objective, f0=f0, N=N, pos=pos, box=box, default_n_freq=n_freq,
        )
        objective_terms_for_step = objective_terms
        calibration_batch = int(_target_calibration_batch(objective, batch))
        target_distribution_initial_response = None
        if calibration_batch > 0:
            cal_rng = np.random.RandomState(train_seed + 104729)
            _, _, target_distribution_initial_response = local_step(
                edges, radii, lengths, N, M_node,
                np.array([t.omega for t in objective_terms_for_step],
                         dtype=np.float64),
                cal_rng, batch=calibration_batch, damping=damping,
                objective_terms=objective_terms_for_step,
                return_term_responses=True,
                force_distribution=force_distribution,
            )
            objective_terms_for_step = calibrate_target_distribution_terms(
                objective, objective_terms_for_step,
                target_distribution_initial_response,
            )
            objective_terms = objective_terms_for_step
        adaptive_pressure_state = adaptive_pressure_state_from_spec(
            objective, objective_terms_for_step,
        )
    omegas_in = np.array([t.omega for t in objective_terms], dtype=np.float64)
    n0 = int(np.sum((f0 > wlo) & (f0 < whi)))

    monitor_forces = None
    monitor_samples = []
    if int(monitor_batch) > 0:
        monitor_rng = np.random.RandomState(
            train_seed + 1000003 if monitor_seed is None else int(monitor_seed)
        )
        monitor_dist = monitor_force_distribution or force_distribution
        monitor_forces = _force_matrix(
            monitor_rng, int(monitor_batch), N, monitor_dist,
        )
        monitor_every = int(monitor_every or eval_every)

    if constraint is not None:
        constraint.bind(radii, lengths)

    cost_history = np.zeros(n_steps, dtype=np.float64)
    nin_samples = []  # (step, n_in, f_min_nonzero, f_max)
    snapshot_steps, radii_snapshots = _init_radius_snapshots(
        n_steps, Me, snapshot_every, snapshot_dtype,
    )
    snapshot_i = _record_radius_snapshot(
        snapshot_steps, radii_snapshots, 0, 0, radii,
    )

    for step in range(n_steps):
        if adaptive_pressure_state is None:
            cost, grad = local_step(
                edges, radii, lengths, N, M_node, omegas_in, rng, batch=batch,
                damping=damping,
                objective_terms=objective_terms_for_step,
                force_distribution=force_distribution,
            )
            term_responses = None
        else:
            cost, grad, term_responses = local_step(
                edges, radii, lengths, N, M_node, omegas_in, rng, batch=batch,
                damping=damping,
                objective_terms=objective_terms_for_step,
                return_term_responses=True,
                force_distribution=force_distribution,
            )
        cost_history[step] = cost

        # Per-edge regularizer contributions (added before projection so
        # the tangent component survives; the normal component cancels
        # against the rescale, leaving a differential pressure on the
        # constraint surface).
        for reg in regularizers:
            grad = grad + reg(radii, lengths)

        # Project onto constraint tangent plane (after optional warmup).
        if constraint is not None and step >= warmup_steps:
            grad = constraint.project(grad, radii, lengths)

        # Clip gradient magnitude.
        gmax = float(np.abs(grad).max())
        if gmax > grad_clip:
            grad = grad * (grad_clip / gmax)

        # Step, then rescale to enforce the constraint exactly.
        radii = np.clip(radii - alpha * grad, R_MIN, R_MAX)
        if constraint is not None and step >= warmup_steps:
            radii = constraint.enforce(radii, lengths)
            radii = np.clip(radii, R_MIN, R_MAX)

        if adaptive_pressure_state is not None:
            objective_terms_for_step = update_adaptive_pressure_terms(
                objective_terms_for_step, term_responses,
                adaptive_pressure_state, step=step + 1,
            )

        if monitor_forces is not None and (step + 1) % monitor_every == 0:
            monitor_cost = objective_cost(
                edges, radii, lengths, N, M_node,
                objective_terms_for_step or objective_terms,
                monitor_forces, damping=damping,
            )
            monitor_samples.append((step + 1, monitor_cost))

        if (step + 1) % eval_every == 0:
            ff = eigenfreqs(edges, radii, lengths, N, M_node)
            ni = int(np.sum((ff > wlo) & (ff < whi)))
            nin_samples.append((step + 1, ni, float(ff[1]), float(ff[-1])))
        snapshot_i = _record_radius_snapshot(
            snapshot_steps, radii_snapshots, snapshot_i, step + 1, radii,
        )

    ff = eigenfreqs(edges, radii, lengths, N, M_node)
    n_in_final = int(np.sum((ff > wlo) & (ff < whi)))

    ff_sorted = np.sort(ff[ff > 0])
    below = ff_sorted[ff_sorted <= wlo]
    above = ff_sorted[ff_sorted >= whi]
    if len(below) and len(above):
        gap_lo, gap_hi = float(below[-1]), float(above[0])
        gap_ratio = (gap_hi - gap_lo) / ((gap_hi + gap_lo) / 2)
    else:
        gap_lo = gap_hi = 0.0
        gap_ratio = 0.0

    result = dict(
        radii=radii.astype(np.float64),
        f0=f0.astype(np.float64),
        ff=ff.astype(np.float64),
        wlo=wlo, whi=whi,
        n_in_initial=n0,
        n_in_final=n_in_final,
        gap_lo=gap_lo, gap_hi=gap_hi, gap_ratio=gap_ratio,
        cost_history=cost_history,
        nin_samples=np.array(nin_samples, dtype=np.float64) if nin_samples
                    else np.zeros((0, 4), dtype=np.float64),
    )
    if adaptive_pressure_state is not None:
        result['adaptive_pressure_history'] = adaptive_pressure_history(
            adaptive_pressure_state,
        )
        result['adaptive_pressure_final'] = np.array(
            [t.coefficient for t in objective_terms_for_step], dtype=np.float64,
        )
    if objective_terms_for_step is not None:
        result.update(objective_term_arrays(objective_terms_for_step))
    if target_distribution_initial_response is not None:
        result['target_distribution_initial_response'] = (
            target_distribution_initial_response.astype(np.float64)
        )
    if monitor_samples:
        result['monitor_cost_samples'] = np.array(monitor_samples,
                                                  dtype=np.float64)
    if radii_snapshots is not None:
        result['radius_snapshot_steps'] = snapshot_steps
        result['radii_snapshots'] = radii_snapshots
    return result


def _target_calibration_batch(spec, default_batch):
    """Return calibration batch when a target-distribution block opts in."""
    blocks = spec.get('terms', []) if isinstance(spec, dict) and spec.get('type') == 'composite' else spec
    if isinstance(blocks, dict):
        blocks = [blocks]
    if not isinstance(blocks, list):
        return 0
    batches = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get('type') != 'target_response_distribution':
            continue
        target = block.get('target_response', block.get('target', 1.0))
        if isinstance(target, dict) and target.get('relative_to_initial', False):
            batches.append(int(block.get('calibration_batch', default_batch)))
    return max(batches) if batches else 0


def _random_force(rng, N, force_distribution):
    if force_distribution in (None, 'gaussian', 'normal'):
        return rng.randn(N)
    if force_distribution in ('rademacher', 'sign', 'binary'):
        return rng.choice((-1.0, 1.0), size=N)
    raise ValueError(f'unknown force_distribution {force_distribution!r}; '
                     'expected "gaussian" or "rademacher"')


def _force_matrix(rng, batch, N, force_distribution):
    return np.vstack([
        _random_force(rng, N, force_distribution)
        for _ in range(int(batch))
    ]).astype(np.float64)


def _init_radius_snapshots(n_steps, n_edges, snapshot_every, snapshot_dtype):
    every = int(snapshot_every or 0)
    if every <= 0:
        return None, None

    steps = [0]
    steps.extend(range(every, int(n_steps) + 1, every))
    if steps[-1] != int(n_steps):
        steps.append(int(n_steps))
    steps = np.array(steps, dtype=np.int64)
    dtype = _radius_snapshot_dtype(snapshot_dtype)
    return steps, np.zeros((len(steps), int(n_edges)), dtype=dtype)


def _record_radius_snapshot(snapshot_steps, radii_snapshots, snapshot_i, step, radii):
    if radii_snapshots is None:
        return snapshot_i
    if snapshot_i < len(snapshot_steps) and int(snapshot_steps[snapshot_i]) == int(step):
        radii_snapshots[snapshot_i] = radii
        return snapshot_i + 1
    return snapshot_i


def _radius_snapshot_dtype(snapshot_dtype):
    dtype = np.dtype(snapshot_dtype)
    if dtype.kind != 'f':
        raise ValueError('snapshot_dtype must be a floating-point dtype')
    return dtype
