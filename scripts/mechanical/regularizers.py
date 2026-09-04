"""Modular per-edge regularizers for the strictly-local training rule.

A regularizer is any callable

    reg(radii, lengths) -> ndarray of shape (M_edges,)

that returns a per-edge contribution to the gradient.  The strictly-local
update rule (see ``learners_local.train_local``) is

    r_e <- clip(r_e - alpha * (g_e + sum_i reg_i(r, L)_e), R_MIN, R_MAX)

so a regularizer's sign convention follows that of a cost gradient: positive
values push the corresponding radii down, negative values push them up.

Unlike the global-constraint modes in ``learners.train`` (vanilla / rad-L1
/ mass-cons), no projection or rescaling is applied, so the rules below are
strictly local: each edge sees only its own radius and length.

The factories return fresh callables that close over their parameters, so a
training run can stack any combination, for example

    train_local(..., regularizers=[inflation(0.03), l1_sparsity(0.01)])
"""
import numpy as np


def inflation(strength=0.03):
    """Constant upward pressure on every radius.

    This is the "stiffness-increasing" rule from Nachi.  It replaces the
    global stiffness/mass conservation: edges that strongly contribute to
    the driven modes (large negative cost gradient) win against the
    inflation and are pushed to R_MIN, while irrelevant edges drift up
    to R_MAX.  The result is a bipolar radius distribution that opens
    the band gap without any global readout.

    Cost interpretation: -strength * sum(r_e), so the gradient is
    -strength on every edge.
    """
    def reg(radii, lengths):
        return np.full_like(radii, -float(strength))
    reg.__name__ = f'inflation(strength={strength})'
    return reg


def l1_sparsity(strength=0.02, length_weighted=True):
    """Strictly-local L1 push toward smaller radii.

    The local analog of the ``rad-L1`` mode in ``learners.train``: a
    length-weighted (or uniform) downward pressure that drives weakly-
    loaded edges toward R_MIN without any global constraint.  Pair with
    ``inflation`` for a competing pair of pressures, or use alone to
    bias the spectrum toward a sparser final network.

    Cost interpretation: strength * sum(w_e * r_e) with w_e = L_e
    (length-weighted) or w_e = 1 (uniform).
    """
    def reg(radii, lengths):
        w = lengths if length_weighted else np.ones_like(radii)
        return float(strength) * w
    reg.__name__ = f'l1_sparsity(strength={strength}, length_weighted={length_weighted})'
    return reg


def l2_pull(target=1.0, strength=0.01):
    """Quadratic spring pulling each radius toward ``target``.

    Homeostatic: dampens both the bipolar collapse (driven by the cost
    gradient) and runaway inflation, keeping the radius distribution
    centered on ``target``.  Useful as a soft prior when neither
    extreme of the bounded interval is desired.

    Cost interpretation: 0.5 * strength * sum((r_e - target)^2), so the
    gradient is strength * (r_e - target) per edge.
    """
    def reg(radii, lengths):
        return float(strength) * (radii - float(target))
    reg.__name__ = f'l2_pull(target={target}, strength={strength})'
    return reg


def material_volume(strength=0.01):
    """Penalize local strut material volume, proportional to ``r^2 L``."""
    def reg(radii, lengths):
        return 2.0 * float(strength) * radii * lengths
    reg.__name__ = f'material_volume(strength={strength})'
    return reg


def stiffness_penalty(strength=0.01):
    """Penalize local axial stiffness contribution, proportional to ``r^2/L``."""
    def reg(radii, lengths):
        return 2.0 * float(strength) * radii / lengths
    reg.__name__ = f'stiffness_penalty(strength={strength})'
    return reg


def binary_radius(strength=0.01, low=0.5, high=2.0):
    """Double-well prior that discourages intermediate radii.

    Cost interpretation:
        strength * (r-low)^2 * (r-high)^2
    with minima at ``low`` and ``high``.
    """
    def reg(radii, lengths):
        a = radii - float(low)
        b = radii - float(high)
        return 2.0 * float(strength) * a * b * (a + b)
    reg.__name__ = f'binary_radius(strength={strength}, low={low}, high={high})'
    return reg


_REGISTRY = {
    'inflation': inflation,
    'l1_sparsity': l1_sparsity,
    'l2_pull': l2_pull,
    'material_volume': material_volume,
    'stiffness_penalty': stiffness_penalty,
    'binary_radius': binary_radius,
}


def regularizer_from_spec(spec):
    """Instantiate a regularizer from a JSON-friendly dict spec.

    The spec must contain a ``type`` key naming one of the registered
    factories (``inflation``, ``l1_sparsity``, ``l2_pull``); all other
    keys are passed as keyword arguments.  Example::

        {"type": "l1_sparsity", "strength": 0.02, "length_weighted": true}
    """
    if not isinstance(spec, dict):
        raise TypeError(f'regularizer spec must be a dict, got {type(spec).__name__}')
    if 'type' not in spec:
        raise ValueError(f'regularizer spec missing "type" key: {spec}')
    reg_type = spec['type']
    if reg_type not in _REGISTRY:
        raise ValueError(f'unknown regularizer type {reg_type!r}; '
                         f'available: {sorted(_REGISTRY)}')
    kwargs = {k: v for k, v in spec.items() if k != 'type'}
    return _REGISTRY[reg_type](**kwargs)


def regularizers_from_specs(specs):
    """Instantiate a list of regularizers from a list of dict specs."""
    if specs is None:
        return None
    return [regularizer_from_spec(s) for s in specs]
