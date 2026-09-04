"""Configurable real-valued response objectives for mechanical learning.

The original band-gap learner minimizes ``||H(omega)^{-1} F||^2`` at a
fixed grid of in-band probe frequencies.  This module keeps that case as
the default, but exposes the more general real-adjoint structure:

    objective term:
        loss = coefficient * phi(u^T Q u)
        H(omega) u = P_source F
        H(omega) v = d loss / d u

``Q`` is diagonal here, represented by per-node readout weights.  The
edge-local gradient is still computed in ``learners.local_step`` from
endpoint products of ``u`` and ``v``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ObjectiveTerm:
    """One real response objective at one probe frequency."""

    omega: float
    coefficient: float = 1.0
    readout_weights: np.ndarray | None = None
    source_weights: np.ndarray | None = None
    transform: str = 'quadratic'
    scale: float = 1.0
    label: str = ''


def default_objective_terms(f0, n_freq=8):
    """Original band-gap objective: suppress response in P35--P65."""
    wlo = float(np.percentile(np.asarray(f0)[1:], 35))
    whi = float(np.percentile(np.asarray(f0)[1:], 65))
    return [
        ObjectiveTerm(float(w), label='default-bandgap')
        for w in np.linspace(wlo, whi, int(n_freq))
    ], wlo, whi


def objective_terms_from_spec(spec, *, f0, N, pos=None, box=None, default_n_freq=8):
    """Build objective terms from a JSON-friendly spec.

    Parameters
    ----------
    spec : dict or list or None
        ``None`` recovers the original band-gap objective.  A list is
        treated as a list of objective blocks.  A dict with
        ``{"type": "composite", "terms": [...]}`` does the same.
    f0 : ndarray
        Initial eigenfrequencies; percentile windows are taken from this.
    N : int
        Number of nodes.
    pos, box : ndarray, optional
        Node positions and simulation box, used for spatial node selectors.
    default_n_freq : int
        Back-compat value used when a block does not provide ``n_freq``.

    Returns
    -------
    terms, wlo, whi
        ``wlo`` and ``whi`` are the first stopband window, retained as
        the legacy band-gap evaluation window.
    """
    if spec is None:
        return default_objective_terms(f0, n_freq=default_n_freq)

    blocks = _normalize_blocks(spec)
    terms: list[ObjectiveTerm] = []
    eval_window = None
    chosen_windows: dict[str, tuple[float, float]] = {}
    for block in blocks:
        block_terms, block_window = _terms_from_block(
            block, f0=np.asarray(f0), N=N, pos=pos, box=box,
            default_n_freq=default_n_freq, chosen_windows=chosen_windows,
        )
        if eval_window is None and block_window is not None:
            eval_window = block_window
        terms.extend(block_terms)
        if block_window is not None:
            label = block.get('label') or block.get('type', 'objective')
            chosen_windows[str(label)] = block_window

    if not terms:
        raise ValueError('objective spec produced no probe terms')
    if eval_window is None:
        omegas = np.array([t.omega for t in terms], dtype=float)
        eval_window = (float(omegas.min()), float(omegas.max()))
    return terms, float(eval_window[0]), float(eval_window[1])


def term_response(term: ObjectiveTerm, u):
    """Measured scalar response ``u^T Q u`` for one objective term."""
    q_u = apply_readout(term, u)
    return float(np.dot(u, q_u))


def evaluate_term(term: ObjectiveTerm, u, response_value=None):
    """Return ``(loss, adjoint_rhs)`` for one term and response field."""
    q_u = apply_readout(term, u)
    s = term_response(term, u) if response_value is None else float(response_value)
    phi, dphi_ds = _transform_value_and_slope(term.transform, s, term.scale)
    coeff = float(term.coefficient)
    loss = coeff * phi
    rhs = coeff * dphi_ds * 2.0 * q_u
    return loss, rhs


def apply_source(term: ObjectiveTerm, F):
    """Apply a diagonal source mask/weight to a random force vector."""
    if term.source_weights is None:
        return F
    return term.source_weights * F


def apply_readout(term: ObjectiveTerm, u):
    """Apply diagonal readout weights Q to a response vector."""
    if term.readout_weights is None:
        return u
    return term.readout_weights * u


def describe_terms(terms: Iterable[ObjectiveTerm]):
    """Compact labels for logging."""
    labels = []
    for term in terms:
        label = term.label or term.transform
        labels.append(f'{label}@{term.omega:.4g}')
    return labels


def _normalize_blocks(spec):
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f'objective spec must be dict/list/None, got {type(spec).__name__}')
    typ = spec.get('type', 'bandgap')
    if typ == 'composite':
        return list(spec.get('terms', []))
    return [spec]


def _terms_from_block(block, *, f0, N, pos, box, default_n_freq, chosen_windows):
    if not isinstance(block, dict):
        raise TypeError(f'objective block must be a dict, got {type(block).__name__}')

    typ = block.get('type', 'bandgap')
    if typ in ('bandgap', 'stopband', 'multi_stopband',
               'passband', 'weighted_response', 'source_target',
               'spectral_pressure', 'adaptive_pressure',
               'target_response_distribution'):
        return _response_terms_from_block(block, f0=f0, N=N, pos=pos, box=box,
                                          default_n_freq=default_n_freq,
                                          chosen_windows=chosen_windows)
    raise ValueError(f'unknown objective type {typ!r}')


def _response_terms_from_block(block, *, f0, N, pos, box, default_n_freq,
                               chosen_windows):
    typ = block.get('type', 'bandgap')
    if typ == 'spectral_pressure':
        return _spectral_pressure_terms_from_block(
            block, f0=f0, N=N, pos=pos, box=box,
            default_n_freq=default_n_freq, chosen_windows=chosen_windows,
        )
    if typ == 'target_response_distribution':
        return _target_response_distribution_terms_from_block(
            block, f0=f0, N=N, pos=pos, box=box,
            default_n_freq=default_n_freq, chosen_windows=chosen_windows,
        )

    windows = block.get('windows')
    if windows is None:
        windows = [block.get('window', _default_window_for_type(typ))]
    if not isinstance(windows, list):
        raise TypeError('objective windows must be a list when provided')

    if typ == 'adaptive_pressure':
        default_coefficient = float(block.get('initial_pressure', 1.0))
    else:
        default_coefficient = -1.0 if typ == 'passband' else 1.0
    coefficient = float(block.get('coefficient', default_coefficient))
    transform = block.get('transform', 'log' if typ == 'passband' else 'quadratic')
    scale = float(block.get('scale', 1.0))
    n_freq = int(block.get('n_freq', default_n_freq))
    schedule = block.get('schedule')

    readout_weights = _selector_weights(
        block.get('readout') or block.get('target'), N=N, pos=pos, box=box,
        none_means_all=True,
    )
    source_weights = _selector_weights(
        block.get('source'), N=N, pos=pos, box=box,
        none_means_all=False,
    )

    all_terms: list[ObjectiveTerm] = []
    eval_window = block.get('eval_window')
    first_window = (_resolve_window(eval_window, f0, chosen_windows=chosen_windows)
                    if eval_window is not None else None)
    for wi, window in enumerate(windows):
        lo, hi = _resolve_window(window, f0, chosen_windows=chosen_windows)
        if first_window is None and coefficient > 0:
            first_window = (lo, hi)
        omegas = _frequency_grid(lo, hi, n_freq=n_freq, schedule=schedule)
        for oi, omega in enumerate(omegas):
            all_terms.append(ObjectiveTerm(
                omega=float(omega),
                coefficient=coefficient,
                readout_weights=readout_weights,
                source_weights=source_weights,
                transform=transform,
                scale=scale,
                label=block.get(
                    'label',
                    'adaptive-pressure' if typ == 'adaptive_pressure'
                    else f'{typ}{wi}:{oi}',
                ),
            ))
    if first_window is None and windows:
        first_window = _resolve_window(windows[0], f0)
    return all_terms, first_window


def _spectral_pressure_terms_from_block(block, *, f0, N, pos, box,
                                        default_n_freq, chosen_windows):
    """Build response terms with frequency-dependent fixed coefficients."""
    windows = block.get('windows')
    if windows is None:
        windows = [block.get('window', {'kind': 'percentile', 'lo': 5, 'hi': 95})]
    if not isinstance(windows, list):
        raise TypeError('spectral_pressure windows must be a list when provided')

    transform = block.get('transform', 'quadratic')
    scale = float(block.get('scale', 1.0))
    n_freq = int(block.get('n_freq', default_n_freq))
    schedule = block.get('schedule')
    pressure = block.get('pressure', 1.0)

    readout_weights = _selector_weights(
        block.get('readout') or block.get('target'), N=N, pos=pos, box=box,
        none_means_all=True,
    )
    source_weights = _selector_weights(
        block.get('source'), N=N, pos=pos, box=box,
        none_means_all=False,
    )

    all_terms: list[ObjectiveTerm] = []
    eval_window = block.get('eval_window')
    first_window = (_resolve_window(eval_window, f0, chosen_windows=chosen_windows)
                    if eval_window is not None else None)
    for wi, window in enumerate(windows):
        lo, hi = _resolve_window(window, f0, chosen_windows=chosen_windows)
        if first_window is None:
            first_window = (lo, hi)
        omegas = np.asarray(_frequency_grid(lo, hi, n_freq=n_freq,
                                            schedule=schedule), dtype=float)
        coefficients = _pressure_values(
            omegas, pressure, f0=f0, chosen_windows=chosen_windows,
        )
        for oi, (omega, coeff) in enumerate(zip(omegas, coefficients)):
            all_terms.append(ObjectiveTerm(
                omega=float(omega),
                coefficient=float(coeff),
                readout_weights=readout_weights,
                source_weights=source_weights,
                transform=transform,
                scale=scale,
                label=block.get('label', f'spectral-pressure{wi}:{oi}'),
            ))
    return all_terms, first_window


def _target_response_distribution_terms_from_block(block, *, f0, N, pos, box,
                                                   default_n_freq,
                                                   chosen_windows):
    """Build log-target response terms with a frequency-dependent target."""
    windows = block.get('windows')
    if windows is None:
        windows = [block.get('window', {'kind': 'percentile', 'lo': 5, 'hi': 95})]
    if not isinstance(windows, list):
        raise TypeError('target_response_distribution windows must be a list')

    coefficient = float(block.get('coefficient', 1.0))
    n_freq = int(block.get('n_freq', default_n_freq))
    schedule = block.get('schedule')
    target = block.get('target_response', block.get('target', 1.0))
    min_target = float(block.get('min_target', _profile_floor(target, default=1e-12)))

    readout_weights = _selector_weights(
        block.get('readout') or block.get('target_nodes'), N=N, pos=pos, box=box,
        none_means_all=True,
    )
    source_weights = _selector_weights(
        block.get('source'), N=N, pos=pos, box=box,
        none_means_all=False,
    )

    all_terms: list[ObjectiveTerm] = []
    eval_window = block.get('eval_window')
    first_window = (_resolve_window(eval_window, f0, chosen_windows=chosen_windows)
                    if eval_window is not None else None)
    for window in windows:
        lo, hi = _resolve_window(window, f0, chosen_windows=chosen_windows)
        if first_window is None:
            first_window = (lo, hi)
        omegas = np.asarray(_frequency_grid(lo, hi, n_freq=n_freq,
                                            schedule=schedule), dtype=float)
        targets = _profile_values(omegas, target, f0=f0,
                                  chosen_windows=chosen_windows)
        targets = np.maximum(targets, min_target)
        for omega, target_value in zip(omegas, targets):
            all_terms.append(ObjectiveTerm(
                omega=float(omega),
                coefficient=coefficient,
                readout_weights=readout_weights,
                source_weights=source_weights,
                transform='log_target',
                scale=float(target_value),
                label=block.get('label', 'target-response'),
            ))
    return all_terms, first_window


def _pressure_values(omegas, pressure, *, f0, chosen_windows):
    """Evaluate a JSON-friendly scalar pressure profile on probe omegas."""
    return _profile_values(omegas, pressure, f0=f0, chosen_windows=chosen_windows)


def _profile_values(omegas, pressure, *, f0, chosen_windows):
    """Evaluate a JSON-friendly scalar profile on probe omegas."""
    omegas = np.asarray(omegas, dtype=float)
    if np.isscalar(pressure):
        return np.full_like(omegas, float(pressure), dtype=float)
    if not isinstance(pressure, dict):
        raise TypeError('pressure must be a scalar or dict')

    typ = pressure.get('type', 'constant')
    if typ in ('constant', 'uniform'):
        return np.full_like(omegas, float(pressure.get('value', 1.0)), dtype=float)
    if typ == 'table':
        xp = np.asarray(pressure['omegas'], dtype=float)
        yp = np.asarray(
            pressure.get('values', pressure.get('pressures', pressure.get('weights'))),
            dtype=float,
        )
        if xp.ndim != 1 or yp.ndim != 1 or len(xp) != len(yp):
            raise ValueError('table pressure requires equal-length 1D omegas and values')
        order = np.argsort(xp)
        return np.interp(omegas, xp[order], yp[order],
                         left=float(yp[order][0]), right=float(yp[order][-1]))
    if typ == 'piecewise':
        vals = np.full_like(omegas, float(pressure.get('default', 0.0)), dtype=float)
        for part in pressure.get('windows', []):
            if not isinstance(part, dict):
                raise TypeError('piecewise pressure windows must be dicts')
            window = part.get('window', part.get('range'))
            if window is None:
                raise ValueError('piecewise pressure entry needs a window/range')
            lo, hi = _resolve_window(window, f0, chosen_windows=chosen_windows)
            value = part.get('pressure', part.get('value', part.get('coefficient')))
            if value is None:
                raise ValueError('piecewise pressure entry needs pressure/value')
            mask = (omegas >= lo) & (omegas <= hi)
            vals[mask] = float(value)
        return vals
    if typ in ('gaussian', 'gaussian_mixture'):
        vals = np.full_like(omegas, float(pressure.get('baseline', 0.0)), dtype=float)
        components = pressure.get('components', pressure.get('peaks', []))
        if typ == 'gaussian' and not components:
            components = [pressure]
        for comp in components:
            if not isinstance(comp, dict):
                raise TypeError('gaussian pressure components must be dicts')
            if 'window' in comp:
                lo, hi = _resolve_window(comp['window'], f0,
                                         chosen_windows=chosen_windows)
                center = 0.5 * (lo + hi)
                width = float(comp.get('width', comp.get('sigma', 0.5 * (hi - lo))))
            else:
                center = float(comp['center'])
                width = float(comp.get('width', comp.get('sigma')))
            if width <= 0:
                raise ValueError('gaussian pressure width/sigma must be positive')
            amplitude = float(comp.get('height', comp.get('amplitude',
                                                         comp.get('pressure', 1.0))))
            vals += amplitude * np.exp(-0.5 * ((omegas - center) / width) ** 2)
        floor = pressure.get('floor')
        if floor is not None:
            vals = np.maximum(vals, float(floor))
        return vals
    if typ in ('gaussian_notch', 'notch', 'notches'):
        vals = np.full_like(omegas, float(pressure.get('baseline', 1.0)), dtype=float)
        notches = pressure.get('notches', pressure.get('components', [pressure]))
        for notch in notches:
            if 'window' in notch:
                lo, hi = _resolve_window(notch['window'], f0,
                                         chosen_windows=chosen_windows)
                center = 0.5 * (lo + hi)
                width = float(notch.get('width', notch.get('sigma', 0.5 * (hi - lo))))
            else:
                center = float(notch['center'])
                width = float(notch.get('width', notch.get('sigma')))
            if width <= 0:
                raise ValueError('notch width/sigma must be positive')
            depth = float(notch.get('depth', 0.9))
            vals *= 1.0 - depth * np.exp(-0.5 * ((omegas - center) / width) ** 2)
        floor = float(pressure.get('floor', 1e-12))
        vals = np.maximum(vals, floor)
        return vals
    raise ValueError(f'unknown pressure profile type {typ!r}')


def _profile_floor(profile, default):
    if isinstance(profile, dict) and profile.get('floor') is not None:
        return profile['floor']
    return default


def calibrate_target_distribution_terms(spec, terms, term_responses):
    """Scale dimensionless target profiles using initial measured response.

    Blocks opt in with ``target_response.relative_to_initial``.  The
    existing per-frequency term scale is treated as a dimensionless target
    shape and multiplied by a response reference measured at initialization.
    """
    blocks = _target_distribution_blocks(spec)
    if not blocks:
        return terms
    responses = np.asarray(term_responses, dtype=float)
    if responses.shape != (len(terms),):
        raise ValueError('term_responses shape must match objective terms')

    new_terms = list(terms)
    labels = [term.label for term in terms]
    for block in blocks:
        profile = block.get('target_response', block.get('target', 1.0))
        if not (isinstance(profile, dict) and profile.get('relative_to_initial', False)):
            continue
        label = block.get('label', 'target-response')
        idx = np.array([i for i, term_label in enumerate(labels)
                        if term_label == label], dtype=int)
        if len(idx) == 0:
            raise ValueError(f'target_response_distribution label {label!r} '
                             'did not match any objective terms')

        observed = np.maximum(responses[idx], float(profile.get('eps', 1e-300)))
        reference = profile.get('reference', 'median')
        if reference in ('median', 'initial_median'):
            ref = float(np.median(observed))
        elif reference in ('mean', 'initial_mean'):
            ref = float(np.mean(observed))
        elif reference in ('quantile', 'initial_quantile'):
            ref = float(np.quantile(observed, float(profile.get('quantile', 0.5))))
        elif reference in ('pointwise', 'pointwise_initial'):
            ref = None
        else:
            raise ValueError(f'unknown target calibration reference {reference!r}')

        reference_scale = float(profile.get('reference_scale', 1.0))
        min_target = float(block.get('min_target', _profile_floor(profile, 1e-12)))
        for local_i, term_i in enumerate(idx):
            multiplier = observed[local_i] if ref is None else ref
            scale = max(float(new_terms[term_i].scale) * multiplier * reference_scale,
                        min_target)
            new_terms[term_i] = replace(new_terms[term_i], scale=scale)
    return new_terms


def objective_term_arrays(terms):
    """Serialisable arrays describing the objective terms used in training."""
    return {
        'objective_term_omegas': np.array([t.omega for t in terms],
                                         dtype=np.float64),
        'objective_term_coefficients': np.array([t.coefficient for t in terms],
                                                dtype=np.float64),
        'objective_term_scales': np.array([t.scale for t in terms],
                                          dtype=np.float64),
        'objective_term_labels': np.array([t.label for t in terms], dtype='<U128'),
        'objective_term_transforms': np.array([t.transform for t in terms],
                                             dtype='<U64'),
    }


def adaptive_pressure_state_from_spec(spec, terms):
    """Return mutable state for response-adaptive pressure blocks.

    The update uses only measured driven responses for the existing probe
    terms.  Frequencies and readout/source masks remain fixed; only the
    scalar pressure coefficient multiplying each local adjoint term is
    adapted between training steps.
    """
    blocks = _adaptive_blocks(spec)
    if not blocks:
        return None

    states = []
    labels = [term.label for term in terms]
    for block in blocks:
        label = block.get('label', 'adaptive-pressure')
        indices = np.array([i for i, term_label in enumerate(labels)
                            if term_label == label], dtype=int)
        if len(indices) == 0:
            raise ValueError(f'adaptive_pressure block label {label!r} '
                             'did not match any objective terms')
        states.append({
            'label': label,
            'indices': indices,
            'eta': float(block.get('eta', block.get('learning_rate', 0.05))),
            'min_pressure': float(block.get('min_pressure', 0.0)),
            'max_pressure': float(block.get('max_pressure', np.inf)),
            'update_every': int(block.get('update_every', 10)),
            'target_response': (None if block.get('target_response') is None
                                else float(block.get('target_response'))),
            'target_quantile': float(block.get('target_quantile', 0.5)),
            'target_scale': float(block.get('target_scale', 1.0)),
            'ema_decay': float(block.get('ema_decay', 0.0)),
            'eps': float(block.get('eps', 1e-300)),
            'zero_mean': bool(block.get('zero_mean', False)),
            'update_rule': block.get('update_rule', 'additive'),
            'target': None,
            'ema': None,
        })
    return {'blocks': states, 'history': []}


def update_adaptive_pressure_terms(terms, term_responses, state, step=None):
    """Update adaptive pressure coefficients from measured responses."""
    if state is None:
        return terms
    new_terms = list(terms)
    responses = np.asarray(term_responses, dtype=float)
    if responses.shape != (len(new_terms),):
        raise ValueError('term_responses shape must match objective terms')

    changed = False
    for block in state['blocks']:
        if step is not None and int(step) % block['update_every'] != 0:
            continue
        idx = block['indices']
        observed = responses[idx].copy()
        observed = np.maximum(observed, 0.0)
        decay = block['ema_decay']
        if decay < 0.0 or decay >= 1.0:
            raise ValueError('adaptive_pressure ema_decay must be in [0, 1)')
        if block['ema'] is None:
            block['ema'] = observed
        else:
            block['ema'] = decay * block['ema'] + (1.0 - decay) * observed
        observed = block['ema']

        target = block['target_response']
        if target is None:
            if block['target'] is None:
                q = min(max(block['target_quantile'], 0.0), 1.0)
                block['target'] = (
                    float(np.quantile(observed, q)) * block['target_scale']
                )
            target = block['target']
        target = max(float(target), block['eps'])

        error = np.log((observed + block['eps']) / target)
        if block['zero_mean']:
            error = error - float(np.mean(error))

        pressures = np.array([new_terms[i].coefficient for i in idx], dtype=float)
        if block['update_rule'] == 'multiplicative':
            pressures = pressures * np.exp(block['eta'] * error)
        elif block['update_rule'] == 'additive':
            pressures = pressures + block['eta'] * error
        else:
            raise ValueError(f"unknown adaptive pressure update_rule "
                             f"{block['update_rule']!r}")
        pressures = np.clip(pressures, block['min_pressure'], block['max_pressure'])

        for i, pressure in zip(idx, pressures):
            new_terms[i] = replace(new_terms[i], coefficient=float(pressure))
        changed = True

    if changed:
        state['history'].append(np.array([t.coefficient for t in new_terms],
                                         dtype=np.float64))
    return new_terms


def adaptive_pressure_history(state):
    """Return pressure snapshots recorded after adaptive updates."""
    if state is None or not state.get('history'):
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(state['history']).astype(np.float64)


def _adaptive_blocks(spec):
    if spec is None:
        return []
    if isinstance(spec, list):
        blocks = spec
    elif isinstance(spec, dict) and spec.get('type') == 'composite':
        blocks = spec.get('terms', [])
    else:
        blocks = [spec]
    return [
        block for block in blocks
        if isinstance(block, dict) and block.get('type') == 'adaptive_pressure'
    ]


def _target_distribution_blocks(spec):
    if spec is None:
        return []
    if isinstance(spec, list):
        blocks = spec
    elif isinstance(spec, dict) and spec.get('type') == 'composite':
        blocks = spec.get('terms', [])
    else:
        blocks = [spec]
    return [
        block for block in blocks
        if isinstance(block, dict)
        and block.get('type') == 'target_response_distribution'
    ]


def _default_window_for_type(typ):
    if typ == 'passband':
        return {'kind': 'percentile', 'lo': 10, 'hi': 25}
    return {'kind': 'percentile', 'lo': 35, 'hi': 65}


def _resolve_window(window, f0, chosen_windows=None):
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return float(window[0]), float(window[1])
    if not isinstance(window, dict):
        raise TypeError('window must be a [lo, hi] pair or dict')
    kind = window.get('kind', 'absolute')
    if kind == 'absolute':
        return float(window['lo']), float(window['hi'])
    if kind == 'percentile':
        fp = np.asarray(f0)[1:]
        return (float(np.percentile(fp, float(window['lo']))),
                float(np.percentile(fp, float(window['hi']))))
    if kind == 'center_width':
        center = float(window['center'])
        width = float(window['width'])
        return center - 0.5 * width, center + 0.5 * width
    if kind in ('density_peak', 'density_valley', 'density_sparse'):
        return _density_window(window, f0, chosen_windows or {})
    raise ValueError(f'unknown window kind {kind!r}')


def _density_window(window, f0, chosen_windows):
    """Pick a fixed-width frequency window from the initial DOS.

    ``density_peak`` maximizes the eigenfrequency count in the window.
    ``density_valley`` / ``density_sparse`` minimize it, optionally while
    staying far from earlier named windows.
    """
    fp = np.sort(np.asarray(f0, dtype=float)[1:])
    fp = fp[np.isfinite(fp)]
    if len(fp) == 0:
        raise ValueError('cannot choose density window from empty spectrum')
    fmin, fmax = float(fp.min()), float(fp.max())
    span = fmax - fmin
    if 'width' in window:
        width = float(window['width'])
    else:
        width = float(window.get('width_fraction', 0.10)) * span
    if width <= 0:
        raise ValueError('density window width must be positive')

    n_scan = int(window.get('n_scan', 400))
    centers = np.linspace(fmin + 0.5 * width, fmax - 0.5 * width, n_scan)
    if len(centers) == 0:
        centers = np.array([0.5 * (fmin + fmax)])
    counts = np.array([
        np.sum((fp >= c - 0.5 * width) & (fp <= c + 0.5 * width))
        for c in centers
    ], dtype=float)

    avoid = list(window.get('avoid_windows', []))
    if window.get('avoid_previous', False):
        avoid.extend(chosen_windows.keys())
    min_sep = float(window.get('min_separation', width))
    valid = np.ones_like(centers, dtype=bool)
    for name in avoid:
        if name not in chosen_windows:
            continue
        lo, hi = chosen_windows[name]
        ref_center = 0.5 * (lo + hi)
        valid &= np.abs(centers - ref_center) >= min_sep
    if not np.any(valid):
        valid[:] = True

    kind = window.get('kind')
    score = counts.copy()
    if kind == 'density_peak':
        score[~valid] = -np.inf
        idx = int(np.argmax(score))
    else:
        score[~valid] = np.inf
        # Prefer valleys far from the spectrum boundaries unless the user
        # explicitly allows edge windows.
        if not window.get('allow_edges', False):
            margin = float(window.get('edge_margin_fraction', 0.08)) * span
            interior = ((centers >= fmin + margin) & (centers <= fmax - margin))
            if np.any(valid & interior):
                score[~interior] = np.inf
        idx = int(np.argmin(score))
    center = float(centers[idx])
    return center - 0.5 * width, center + 0.5 * width


def _frequency_grid(lo, hi, *, n_freq, schedule=None):
    if schedule is None:
        return np.linspace(lo, hi, int(n_freq))
    if isinstance(schedule, str):
        schedule = {'type': schedule}
    if not isinstance(schedule, dict):
        raise TypeError('schedule must be a string, dict, or None')
    typ = schedule.get('type', 'uniform')
    n_freq = int(schedule.get('n_freq', n_freq))
    if typ == 'uniform':
        return np.linspace(lo, hi, n_freq)
    if typ == 'edges':
        frac = float(schedule.get('edge_fraction', 0.15))
        n_each = max(1, n_freq // 2)
        left = np.linspace(lo, lo + frac * (hi - lo), n_each)
        right = np.linspace(hi - frac * (hi - lo), hi, n_freq - n_each)
        return np.concatenate([left, right])
    if typ == 'center':
        center = 0.5 * (lo + hi)
        width_fraction = float(schedule.get('width_fraction', 0.25))
        half = 0.5 * width_fraction * (hi - lo)
        return np.linspace(center - half, center + half, n_freq)
    if typ == 'curriculum':
        # Static representation of a curriculum phase; callers can swap
        # specs between runs, while the inner loop remains simple.
        phase = float(schedule.get('phase', 1.0))
        phase = min(max(phase, 0.0), 1.0)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * phase
        return np.linspace(center - half, center + half, n_freq)
    raise ValueError(f'unknown frequency schedule {typ!r}')


def _selector_weights(selector, *, N, pos, box, none_means_all):
    if selector is None:
        return None if not none_means_all else None
    weights = np.zeros(int(N), dtype=np.float64)
    if selector == 'all':
        weights[:] = 1.0
        return weights
    if isinstance(selector, list):
        weights[np.asarray(selector, dtype=int)] = 1.0
        return weights
    if not isinstance(selector, dict):
        raise TypeError('node selector must be "all", list, dict, or None')

    if selector.get('all', False):
        weights[:] = 1.0
    if 'indices' in selector:
        weights[np.asarray(selector['indices'], dtype=int)] = 1.0
    if 'weights' in selector:
        arr = np.asarray(selector['weights'], dtype=np.float64)
        if arr.shape != (N,):
            raise ValueError(f'selector weights shape {arr.shape} != ({N},)')
        weights += arr
    if any(k in selector for k in ('x_lt', 'x_gt', 'y_lt', 'y_gt',
                                   'x_between', 'y_between')):
        if pos is None:
            raise ValueError('spatial selector requires node positions')
        p = np.asarray(pos, dtype=np.float64)
        b = np.asarray(box, dtype=np.float64) if box is not None else None
        x = _normalized_coord(p[:, 0], b[0] if b is not None and b.size else None)
        y = _normalized_coord(p[:, 1], b[1] if b is not None and b.size > 1 else None)
        mask = np.ones(N, dtype=bool)
        if 'x_lt' in selector:
            mask &= x < float(selector['x_lt'])
        if 'x_gt' in selector:
            mask &= x > float(selector['x_gt'])
        if 'y_lt' in selector:
            mask &= y < float(selector['y_lt'])
        if 'y_gt' in selector:
            mask &= y > float(selector['y_gt'])
        if 'x_between' in selector:
            lo, hi = selector['x_between']
            mask &= (x >= float(lo)) & (x <= float(hi))
        if 'y_between' in selector:
            lo, hi = selector['y_between']
            mask &= (y >= float(lo)) & (y <= float(hi))
        weights[mask] = float(selector.get('value', 1.0))
    if not np.any(weights):
        raise ValueError(f'node selector selected no nodes: {selector}')
    return weights


def _normalized_coord(values, box_length):
    values = np.asarray(values, dtype=np.float64)
    if box_length is not None and float(box_length) > 0:
        return values / float(box_length)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _transform_value_and_slope(transform, s, scale):
    scale = float(scale)
    if scale <= 0:
        raise ValueError('objective transform scale must be positive')
    if transform in ('quadratic', 'linear_response'):
        return s, 1.0
    if transform == 'log':
        z = 1.0 + s / scale
        return scale * np.log(z), 1.0 / z
    if transform == 'exp_saturating':
        z = np.exp(-s / scale)
        return scale * (1.0 - z), z
    if transform == 'log_target':
        # Penalize response away from a finite target scale:
        #   0.5 * (log((s+eps)/scale))^2
        # This avoids pure passband rewards gaming the objective with
        # isolated resonant spikes.
        eps = 1e-300
        z = max(s, eps) / scale
        logz = np.log(z)
        return 0.5 * logz * logz, logz / max(s, eps)
    raise ValueError(f'unknown objective transform {transform!r}')
