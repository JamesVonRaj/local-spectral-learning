"""Legacy global constraints for the conservation-based training rule.

These constraints are kept for reproducing and interpreting older
projection/conservation baselines.  They are not part of the active
local-only band-gap method.

A constraint is an object with three methods:

    bind(radii_init, lengths) -> None
        Called once at the start of training to anchor the conserved
        quantity at its initial value.
    project(grad, radii, lengths) -> ndarray of shape (M_edges,)
        Project the gradient onto the constraint tangent plane, removing
        the component that would change the conserved quantity.
    enforce(radii, lengths) -> ndarray of shape (M_edges,)
        Rescale the radii to land exactly on the constraint surface.

The factories below (``stiffness_conservation``, ``mass_conservation``)
return fresh, unbound constraint instances.  Pass one to
``learners.train`` via the ``constraint=`` parameter; ``train`` will call
``bind`` once and then ``project`` / ``enforce`` each step.

Sign convention for ``project`` matches that of regularizers: the
returned gradient is what gets used by the radius update,
``r <- clip(r - alpha * grad)``.
"""
import numpy as np


class _ConservationConstraint:
    """Scalar conservation: enforce a single positive scalar quantity
    ``c(radii, lengths)`` to remain at its initial value.  Subclasses
    define ``value`` and the per-edge ``grad`` of c with respect to r.
    """
    def __init__(self):
        self.target = None

    def value(self, radii, lengths):
        raise NotImplementedError

    def grad(self, radii, lengths):
        raise NotImplementedError

    def bind(self, radii_init, lengths):
        self.target = self.value(radii_init, lengths)

    def project(self, grad, radii, lengths):
        cg = self.grad(radii, lengths)
        proj = float(np.dot(grad, cg) / (np.dot(cg, cg) + 1e-30))
        return grad - proj * cg

    def enforce(self, radii, lengths):
        c_cur = self.value(radii, lengths)
        if c_cur > 0:
            return radii * np.sqrt(self.target / c_cur)
        return radii


class _StiffnessConservation(_ConservationConstraint):
    """Conserve the total stiffness ``sum(r_e^2 / L_e)``."""
    def value(self, radii, lengths):
        return float(np.sum(radii**2 / lengths))

    def grad(self, radii, lengths):
        return 2 * radii / lengths


class _MassConservation(_ConservationConstraint):
    """Conserve the total mass ``sum(r_e^2 * L_e)``."""
    def value(self, radii, lengths):
        return float(np.sum(radii**2 * lengths))

    def grad(self, radii, lengths):
        return 2 * radii * lengths


def stiffness_conservation():
    """Factory: a fresh stiffness-conservation constraint."""
    return _StiffnessConservation()


def mass_conservation():
    """Factory: a fresh mass-conservation constraint."""
    return _MassConservation()


_REGISTRY = {
    'stiffness_conservation': stiffness_conservation,
    'mass_conservation': mass_conservation,
}


def constraint_from_spec(spec):
    """Instantiate a constraint from a JSON-friendly dict spec, or return
    ``None`` if ``spec`` is ``None``.

    The spec must contain a ``type`` key naming one of the registered
    factories (``stiffness_conservation``, ``mass_conservation``); other
    keys are passed as keyword arguments.  Example::

        {"type": "stiffness_conservation"}
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError(f'constraint spec must be a dict or None, '
                        f'got {type(spec).__name__}')
    if 'type' not in spec:
        raise ValueError(f'constraint spec missing "type" key: {spec}')
    cons_type = spec['type']
    if cons_type not in _REGISTRY:
        raise ValueError(f'unknown constraint type {cons_type!r}; '
                         f'available: {sorted(_REGISTRY)}')
    kwargs = {k: v for k, v in spec.items() if k != 'type'}
    return _REGISTRY[cons_type](**kwargs)
