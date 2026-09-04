"""Strictly local scalar band-gap learning.

The production path learns each edge's log stiffness from two response solves.
It combines three edge-local response moments into an analytically bounded
descent drive.  This path uses no global material constraint, projection,
normalization, initial-stiffness memory, or post-step rescaling.

The module also retains the original additive-radius update and optional local
regularizers for legacy comparisons.  Those controls are separate from the
production response-conditioned rule.
"""
import numpy as np

from .learners import (
    VISCOUS_DAMPING,
    _force_matrix,
    _init_radius_snapshots,
    _record_radius_snapshot,
    _target_calibration_batch,
    eigenfreqs,
    local_step,
    local_step_with_moments,
    objective_cost,
)
from .objectives import (
    ObjectiveTerm,
    adaptive_pressure_history,
    adaptive_pressure_state_from_spec,
    calibrate_target_distribution_terms,
    default_objective_terms,
    objective_term_arrays,
    objective_terms_from_spec,
    update_adaptive_pressure_terms,
)
from .regularizers import inflation


def saturate_components(values, c):
    r"""Apply the local saturation function :math:`\operatorname{sat}_c`.

    For every edge independently,

    .. math::

        \operatorname{sat}_c(g_e) =
        \begin{cases}
        -c, & g_e < -c, \\
        g_e, & |g_e| \le c, \\
        c, & g_e > c.
        \end{cases}

    Unlike normalization by a vector norm or by ``max(abs(values))``, this
    operation reads only the component being updated.  It therefore caps an
    edge's radius increment at ``alpha * c`` without coupling that increment
    to gradients elsewhere in the network.
    """
    c = float(c)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("saturation threshold c must be finite and positive")
    return np.clip(np.asarray(values), -c, c)


def train_local(edges, lengths, N, M_node=1.0,
                R_INIT=1.0, R_MIN=0.5, R_MAX=2.0,
                n_steps=3000, batch=10, n_freq=8,
                alpha=0.05, grad_clip=None, damping='real_shift',
                damping_gamma=0.0,
                force_distribution='gaussian',
                regularizers=None,
                material_update='radius_euler',
                response_metric_eta=0.02,
                response_metric_lambda_ratio=0.025,
                response_drive='balanced',
                response_bound_mode='clip',
                radius_parameterization='clip',
                objective=None, target_window=None, target_window_schedule=None,
                pos=None, box=None,
                train_seed=0,
                frequency_sampling='random_grid',
                frequencies_per_step=1, frequency_seed=None,
                warmup_steps=0,
                eval_every=500,
                monitor_batch=0, monitor_every=None, monitor_seed=None,
                monitor_force_distribution=None,
                snapshot_every=0, snapshot_dtype='float32',
                spectrum_diagnostics=True, initial_radii=None):
    """Train a scalar network without global stiffness/mass conservation.

    Parameters
    ----------
    regularizers : list of callables, optional
        Each callable has signature ``reg(radii, lengths) -> ndarray``
        of shape ``(M_edges,)`` and returns a per-edge gradient
        contribution.  Contributions are summed into the cost gradient
        before componentwise clipping and the radius update.  ``None`` defaults to
        ``[inflation(strength=0.03)]`` for ``radius_euler`` (the original
        behavior) and ``[]`` for ``response_conditioned_log``.  Pass
        ``[]`` to disable all regularization (cost gradient only).
        See ``mechanical.regularizers`` for ready-made factories.
    material_update : {'radius_euler', 'response_conditioned_log'}, optional
        ``'radius_euler'`` retains the original additive radius update.
        ``'response_conditioned_log'`` learns ``theta_e=log(r_e^2/L_e)``
        using a scaled second response ``v_tilde = Lambda v`` and the bounded,
        dimensionless edge drive
        ``2 D_tilde_e / (U_e + V_tilde_e)``.  This mode uses no
        global gradient norm or material total and requires ``regularizers=[]``.
    response_metric_eta : float, optional
        Maximum absolute log-stiffness change per material update in the
        response-conditioned mode.
    response_metric_lambda_ratio : float, optional
        Sets ``Lambda`` as this fraction of the spectral-window width in squared
        frequency, ``whi^2-wlo^2``.
    response_drive : {'balanced', 'correlation', 'sign'}, optional
        Local log-stiffness drive used by ``response_conditioned_log``.
        ``'balanced'`` is the production response-balanced rule.
        ``'correlation'`` uses ``D_e / sqrt(U_e V_e)`` and ``'sign'`` uses
        ``sign(D_e)``; these two experimental alternatives contain no response
        scale parameter.
    response_bound_mode : {'clip', 'none'}, optional
        Whether the response-conditioned log-stiffness update is clipped at
        the componentwise radius bounds.  ``'none'`` is an experimental
        unbounded material trajectory; it does not add a global stopping rule.
    target_window : tuple(float, float) or None, optional
        Optional externally prescribed absolute frequency window.  This keeps
        the default full-network squared-response objective while bypassing
        percentile selection; it is therefore compatible with the
        response-conditioned material update.  Generalized ``objective``
        specifications remain a
        separate path.
    target_window_schedule : sequence, optional
        Abrupt absolute-window schedule.  Each entry is ``(start_step, lo,
        hi)``; the first start step must be zero.  Changing stages changes only
        the physical probe frequencies and continues from the current material
        state.  It is incompatible with ``objective`` and ``target_window``.
    damping_gamma : float, optional
        Coefficient ``gamma`` in the mass-proportional viscous operator
        ``H = K - omega**2 M + 1j * gamma * omega * M``.  It is used only
        when ``damping='viscous'``.
    grad_clip : float or None, optional
        Threshold ``c`` in the optional local saturation
        ``sat_c(g_e) = clip(g_e, -c, c)``.  Each edge component is saturated
        independently; ``None`` disables saturation.  No network-wide norm
        or maximum is read.
    frequency_sampling : {'random_grid', 'all'}, optional
        ``'random_grid'`` samples a fresh subset of the fixed objective
        frequencies each step.  Its expectation is the same finite-grid
        objective as ``'all'``, which evaluates every frequency each step.
    frequencies_per_step : int, optional
        Number of fixed-grid frequencies sampled without replacement when
        ``frequency_sampling='random_grid'``.
    frequency_seed : int or None, optional
        Seed for frequency-index sampling.  By default it is deterministically
        offset from ``train_seed`` so force and frequency streams are separate.
    warmup_steps : int, optional
        Accepted for config compatibility with ``learners.train``.  The
        strictly-local learner has no global projection/rescale warmup, so
        this value is intentionally ignored.
    spectrum_diagnostics : bool, optional
        If false with an absolute window or schedule, perform no eigensolve
        before, during, or after learning.  Radius snapshots can be analyzed
        spectrally afterward without entering the update loop.
    initial_radii : array-like or None, optional
        Optional positive starting radii.  This supports continued adaptation
        without reinitializing the material.
    """
    material_update = str(material_update)
    if material_update not in {'radius_euler', 'response_conditioned_log'}:
        raise ValueError(
            "material_update must be 'radius_euler' or "
            "'response_conditioned_log'"
        )
    response_conditioned_update = material_update == 'response_conditioned_log'
    response_drive = str(response_drive)
    response_bound_mode = str(response_bound_mode)
    if regularizers is None:
        regularizers = (
            [] if response_conditioned_update else [inflation(strength=0.03)]
        )
    if response_conditioned_update:
        if response_drive not in {'balanced', 'correlation', 'sign'}:
            raise ValueError(
                "response_drive must be 'balanced', 'correlation', or 'sign'"
            )
        if response_bound_mode not in {'clip', 'none'}:
            raise ValueError("response_bound_mode must be 'clip' or 'none'")
        if response_bound_mode == 'clip':
            if not 0.0 < float(R_MIN) <= float(R_MAX):
                raise ValueError('response_conditioned_log requires positive '
                                 'ordered radius bounds')
            starting_radii = (
                np.asarray(initial_radii, dtype=np.float64)
                if initial_radii is not None else np.asarray([R_INIT], dtype=np.float64)
            )
            if np.any(starting_radii < float(R_MIN)) or np.any(
                starting_radii > float(R_MAX)
            ):
                raise ValueError('initial radii must lie within R_MIN and R_MAX')
        if objective is not None:
            raise ValueError('response_conditioned_log currently requires the '
                             'default squared-response objective')
        if damping not in {'real_shift', VISCOUS_DAMPING}:
            raise ValueError(
                'response_conditioned_log requires undamped or viscous response'
            )
        if radius_parameterization != 'clip':
            raise ValueError('response_conditioned_log manages log stiffness '
                             'internally and requires radius_parameterization="clip"')
        if grad_clip is not None:
            raise ValueError('response_conditioned_log supplies its own analytic '
                             'rate bound; grad_clip must be None')
        if len(regularizers):
            raise ValueError('response_conditioned_log does not use additive '
                             'radius regularizers')
        response_metric_eta = float(response_metric_eta)
        response_metric_lambda_ratio = float(response_metric_lambda_ratio)
        if response_metric_eta <= 0.0:
            raise ValueError('response-metric eta must be positive')
        if response_drive == 'balanced' and response_metric_lambda_ratio <= 0.0:
            raise ValueError('balanced response drive requires a positive lambda ratio')
    damping_gamma = float(damping_gamma)
    if damping == VISCOUS_DAMPING and damping_gamma < 0.0:
        raise ValueError('damping_gamma must be nonnegative')
    if damping != VISCOUS_DAMPING and damping_gamma != 0.0:
        raise ValueError('damping_gamma is only valid with damping="viscous"')

    if target_window_schedule is not None:
        if target_window is not None or objective is not None:
            raise ValueError(
                'target_window_schedule is incompatible with target_window '
                'and generalized objective specifications'
            )
        window_schedule = _normalize_window_schedule(
            target_window_schedule, n_steps,
        )
    else:
        window_schedule = None

    Me = len(edges)
    radius_state = _init_radius_state(
        Me, R_INIT, radius_parameterization, initial_radii=initial_radii,
    )
    radii = _radii_from_state(radius_state, radius_parameterization)
    rng = np.random.RandomState(train_seed)
    frequency_sampling = str(frequency_sampling)
    if frequency_sampling not in {'random_grid', 'all'}:
        raise ValueError("frequency_sampling must be 'random_grid' or 'all'")
    frequencies_per_step = int(frequencies_per_step)
    if frequencies_per_step < 1:
        raise ValueError('frequencies_per_step must be positive')
    frequency_rng = np.random.RandomState(
        int(train_seed) + 918273 if frequency_seed is None else int(frequency_seed)
    )

    spectrum_diagnostics = bool(spectrum_diagnostics)
    externally_positioned = target_window is not None or window_schedule is not None
    needs_initial_spectrum = objective is not None or not externally_positioned
    f0 = (
        eigenfreqs(edges, radii, lengths, N, M_node)
        if needs_initial_spectrum or spectrum_diagnostics
        else np.zeros(0, dtype=np.float64)
    )
    if objective is None:
        if window_schedule is not None:
            _, wlo, whi = window_schedule[0]
            objective_terms = _absolute_window_terms(wlo, whi, n_freq)
        elif target_window is None:
            objective_terms, wlo, whi = default_objective_terms(f0, n_freq=n_freq)
        else:
            wlo, whi = _validate_window(target_window)
            objective_terms = _absolute_window_terms(wlo, whi, n_freq)
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
    if adaptive_pressure_state is not None and frequency_sampling != 'all':
        raise ValueError('adaptive-pressure objectives currently require frequency_sampling="all"')
    n0 = (
        int(np.sum((f0 > wlo) & (f0 < whi)))
        if len(f0) else -1
    )
    if response_conditioned_update:
        stiffness_initial = radii * radii / lengths
        theta_state = np.log(stiffness_initial)
        if response_bound_mode == 'clip':
            theta_min = np.log((float(R_MIN) ** 2) / lengths)
            theta_max = np.log((float(R_MAX) ** 2) / lengths)
        if response_drive == 'balanced':
            response_metric_lambda = (
                response_metric_lambda_ratio * float(whi**2 - wlo**2)
            )
        else:
            response_metric_lambda = np.nan

    cost_history = np.zeros(n_steps, dtype=np.float64)
    nin_samples = []
    snapshot_steps, radii_snapshots = _init_radius_snapshots(
        n_steps, Me, snapshot_every, snapshot_dtype,
    )
    snapshot_i = _record_radius_snapshot(
        snapshot_steps, radii_snapshots, 0, 0, radii,
    )
    monitor_forces = None
    monitor_samples = []
    window_eval_samples = []
    if int(monitor_batch) > 0:
        monitor_rng = np.random.RandomState(
            train_seed + 1000003 if monitor_seed is None else int(monitor_seed)
        )
        monitor_dist = monitor_force_distribution or force_distribution
        monitor_forces = _force_matrix(
            monitor_rng, int(monitor_batch), N, monitor_dist,
        )
        monitor_every = int(monitor_every or eval_every)

    schedule_index = 0
    for step in range(n_steps):
        if (
            window_schedule is not None
            and schedule_index + 1 < len(window_schedule)
            and step == int(window_schedule[schedule_index + 1][0])
        ):
            schedule_index += 1
            _, wlo, whi = window_schedule[schedule_index]
            objective_terms = _absolute_window_terms(wlo, whi, n_freq)
            omegas_in = np.array(
                [term.omega for term in objective_terms], dtype=np.float64,
            )
            if response_conditioned_update and response_drive == 'balanced':
                response_metric_lambda = (
                    response_metric_lambda_ratio * float(whi**2 - wlo**2)
                )
        step_omegas = omegas_in
        step_terms = objective_terms_for_step
        if frequency_sampling == 'random_grid':
            n_pick = min(frequencies_per_step, len(omegas_in))
            term_indices = frequency_rng.choice(
                len(omegas_in), size=n_pick, replace=False,
            )
            step_omegas = omegas_in[term_indices]
            if objective_terms_for_step is not None:
                step_terms = [objective_terms_for_step[int(i)] for i in term_indices]
        edge_moments = None
        if response_conditioned_update:
            cost, grad, edge_moments = local_step_with_moments(
                edges, radii, lengths, N, M_node, step_omegas, rng,
                batch=batch, damping=damping,
                damping_gamma=damping_gamma,
                force_distribution=force_distribution,
            )
            term_responses = None
        elif adaptive_pressure_state is None:
            cost, grad = local_step(
                edges, radii, lengths, N, M_node, step_omegas, rng, batch=batch,
                damping=damping,
                damping_gamma=damping_gamma,
                objective_terms=step_terms,
                force_distribution=force_distribution,
            )
            term_responses = None
        else:
            cost, grad, term_responses = local_step(
                edges, radii, lengths, N, M_node, step_omegas, rng, batch=batch,
                damping=damping,
                damping_gamma=damping_gamma,
                objective_terms=step_terms,
                return_term_responses=True,
                force_distribution=force_distribution,
            )
        cost_history[step] = cost

        for reg in regularizers:
            grad = grad + reg(radii, lengths)

        if response_conditioned_update:
            moment_u = np.asarray(edge_moments['U'], dtype=np.float64)
            moment_v = np.asarray(edge_moments['V'], dtype=np.float64)
            moment_d = np.asarray(edge_moments['D'], dtype=np.float64)
            if response_drive == 'balanced':
                # Absorb the externally fixed re-drive gain into the second
                # response.  The edge law then uses only its three measured
                # moments: 2 D_tilde / (U + V_tilde).
                moment_v_scaled = response_metric_lambda**2 * moment_v
                moment_d_scaled = response_metric_lambda * moment_d
                denominator = moment_u + moment_v_scaled
                local_drive = np.zeros_like(moment_d)
                np.divide(
                    2.0 * moment_d_scaled,
                    denominator,
                    out=local_drive,
                    where=denominator > 0.0,
                )
            elif response_drive == 'correlation':
                denominator = np.sqrt(moment_u * moment_v)
                local_drive = np.zeros_like(moment_d)
                np.divide(
                    moment_d,
                    denominator,
                    out=local_drive,
                    where=denominator > 0.0,
                )
            else:
                local_drive = np.sign(moment_d)
            if np.max(np.abs(local_drive)) > 1.0 + 5e-12:
                raise AssertionError('response-conditioned local drive exceeded '
                                     'its Cauchy--Schwarz bound')
            theta_state = theta_state + response_metric_eta * local_drive
            if response_bound_mode == 'clip':
                theta_state = np.clip(theta_state, theta_min, theta_max)
            radii = np.sqrt(np.exp(theta_state) * lengths)
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise FloatingPointError(
                    f'nonfinite response-conditioned material state at step {step + 1}'
                )
            radius_state = radii
        elif radius_parameterization == 'clip':
            grad_update = grad
            if grad_clip is not None:
                grad_update = saturate_components(grad_update, grad_clip)
            radii = np.clip(radii - alpha * grad_update, R_MIN, R_MAX)
            radius_state = radii
        else:
            grad_state = _radius_state_gradient(
                grad, radius_state, radius_parameterization,
            )
            if grad_clip is not None:
                grad_state = saturate_components(grad_state, grad_clip)
            radius_state = radius_state - alpha * grad_state
            radii = _radii_from_state(radius_state, radius_parameterization)

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
                damping_gamma=damping_gamma,
            )
            monitor_samples.append((step + 1, monitor_cost))

        if spectrum_diagnostics and (step + 1) % eval_every == 0:
            ff = eigenfreqs(edges, radii, lengths, N, M_node)
            ni = int(np.sum((ff > wlo) & (ff < whi)))
            nin_samples.append((step + 1, ni, float(ff[1]), float(ff[-1])))
            window_eval_samples.append((step + 1, wlo, whi, ni))
        snapshot_i = _record_radius_snapshot(
            snapshot_steps, radii_snapshots, snapshot_i, step + 1, radii,
        )

    if spectrum_diagnostics:
        ff = eigenfreqs(edges, radii, lengths, N, M_node)
        n_in_final = int(np.sum((ff > wlo) & (ff < whi)))
        ff_sorted = np.sort(ff[ff > 0])
        below = ff_sorted[ff_sorted <= wlo]
        above = ff_sorted[ff_sorted >= whi]
        if n_in_final == 0 and len(below) and len(above):
            gap_lo, gap_hi = float(below[-1]), float(above[0])
            gap_ratio = (gap_hi - gap_lo) / ((gap_hi + gap_lo) / 2)
        else:
            gap_lo = gap_hi = 0.0
            gap_ratio = 0.0
    else:
        ff = np.zeros(0, dtype=np.float64)
        n_in_final = -1
        gap_lo = gap_hi = gap_ratio = np.nan

    result = dict(
        radii=radii.astype(np.float64),
        f0=f0.astype(np.float64),
        ff=ff.astype(np.float64),
        wlo=wlo, whi=whi,
        n_in_initial=n0,
        n_in_final=n_in_final,
        gap_lo=gap_lo, gap_hi=gap_hi, gap_ratio=gap_ratio,
        frequency_sampling=np.asarray(frequency_sampling),
        frequencies_per_step=np.int64(
            len(omegas_in) if frequency_sampling == 'all' else
            min(frequencies_per_step, len(omegas_in))
        ),
        frequency_seed=np.int64(
            int(train_seed) + 918273 if frequency_seed is None else int(frequency_seed)
        ),
        cost_history=cost_history,
        material_update=np.asarray(material_update),
        damping=np.asarray(damping),
        damping_gamma=np.float64(damping_gamma),
        spectrum_diagnostics=np.bool_(spectrum_diagnostics),
        response_metric_eta=np.float64(
            response_metric_eta if response_conditioned_update else np.nan
        ),
        response_metric_lambda_ratio=np.float64(
            response_metric_lambda_ratio if response_conditioned_update else np.nan
        ),
        response_metric_lambda=np.float64(
            response_metric_lambda if response_conditioned_update else np.nan
        ),
        response_drive=np.asarray(
            response_drive if response_conditioned_update else ''
        ),
        response_bound_mode=np.asarray(
            response_bound_mode if response_conditioned_update else ''
        ),
        nin_samples=np.array(nin_samples, dtype=np.float64) if nin_samples
                    else np.zeros((0, 4), dtype=np.float64),
        window_eval_samples=(
            np.array(window_eval_samples, dtype=np.float64)
            if window_eval_samples else np.zeros((0, 4), dtype=np.float64)
        ),
    )
    result['window_schedule'] = np.asarray(
        window_schedule if window_schedule is not None else [(0, wlo, whi)],
        dtype=np.float64,
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


def _validate_window(window):
    try:
        wlo, whi = (float(value) for value in window)
    except (TypeError, ValueError) as exc:
        raise ValueError('spectral window must contain exactly two numbers') from exc
    if not 0.0 < wlo < whi:
        raise ValueError('spectral window must satisfy 0 < lo < hi')
    return wlo, whi


def _absolute_window_terms(wlo, whi, n_freq):
    wlo, whi = _validate_window((wlo, whi))
    if int(n_freq) < 1:
        raise ValueError('n_freq must be positive')
    return [
        ObjectiveTerm(float(omega), label='absolute-bandgap')
        for omega in np.linspace(wlo, whi, int(n_freq))
    ]


def _normalize_window_schedule(schedule, n_steps):
    normalized = []
    for entry in schedule:
        if isinstance(entry, dict):
            start = entry.get('start_step')
            window = entry.get('window')
            if window is None:
                window = (entry.get('wlo'), entry.get('whi'))
            wlo, whi = _validate_window(window)
        else:
            try:
                start, wlo, whi = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'each window schedule entry must be (start_step, lo, hi)'
                ) from exc
            wlo, whi = _validate_window((wlo, whi))
        start = int(start)
        normalized.append((start, wlo, whi))
    if not normalized or normalized[0][0] != 0:
        raise ValueError('window schedule must begin at start_step 0')
    starts = [entry[0] for entry in normalized]
    if starts != sorted(set(starts)):
        raise ValueError('window schedule start steps must be strictly increasing')
    if starts[-1] >= int(n_steps):
        raise ValueError('window schedule start steps must be smaller than n_steps')
    return normalized


def _init_radius_state(n_edges, R_INIT, parameterization, initial_radii=None):
    parameterization = str(parameterization)
    if initial_radii is None:
        r0 = np.full(n_edges, float(R_INIT), dtype=np.float64)
    else:
        r0 = np.asarray(initial_radii, dtype=np.float64)
        if r0.shape != (n_edges,):
            raise ValueError(f'initial_radii must have shape {(n_edges,)}')
    if np.any(~np.isfinite(r0)) or np.any(r0 <= 0.0):
        raise ValueError('initial radii must be finite and positive')
    if parameterization == 'clip':
        return r0.copy()
    if parameterization == 'log':
        return np.log(r0)
    if parameterization == 'softplus':
        return _inv_softplus(r0)
    raise ValueError(f'unknown radius_parameterization {parameterization!r}; '
                     'expected "clip", "log", or "softplus"')


def _radii_from_state(state, parameterization):
    if parameterization == 'clip':
        return np.asarray(state, dtype=np.float64)
    if parameterization == 'log':
        return np.exp(state)
    if parameterization == 'softplus':
        x = np.asarray(state, dtype=np.float64)
        return np.logaddexp(0.0, x)
    raise ValueError(f'unknown radius_parameterization {parameterization!r}')


def _radius_state_gradient(grad_r, state, parameterization):
    if parameterization == 'log':
        return grad_r * np.exp(state)
    if parameterization == 'softplus':
        return grad_r * _sigmoid(state)
    raise ValueError(f'unknown unconstrained parameterization {parameterization!r}')


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _inv_softplus(r):
    # Stable inverse of log(1 + exp(x)) for positive scalar or array input.
    r = np.asarray(r, dtype=np.float64)
    return r + np.log(-np.expm1(-r))
