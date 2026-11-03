"""Counterfactual intervention regimes for Layer 1 structural counterfactuals.

Defines the δ_t intervention regimes that are injected into the structural
equation for counterfactual treatment generation:

    T^{cf}_t = Φ_T · T^{cf}_{t-1} + ε_{T,t} + δ_t

Three regimes are supported:
    - PUNCTUAL:    One-time shock at t = t_int
    - CONTINUOUS:  Sustained constant intensity from t_int onward
    - GRADUAL:     Phased ramp starting at t_int
"""

from enum import Enum
from typing import Callable, Dict


class InterventionRegime(Enum):
    """Enumeration of supported counterfactual intervention regimes.

    Each regime defines how δ_t is computed at each time step,
    representing a different treatment policy for the counterfactual world.
    """

    PUNCTUAL = "punctual"
    CONTINUOUS = "continuous"
    GRADUAL = "gradual"


# ---------------------------------------------------------------------------
# δ_t generator registry (Strategy Pattern)
# ---------------------------------------------------------------------------

_DELTA_REGIMES: Dict[InterventionRegime, Callable[..., float]] = {}


def _register_delta(regime: InterventionRegime) -> Callable[[Callable], Callable]:
    """Decorator to register a δ_t generator for an intervention regime."""

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        _DELTA_REGIMES[regime] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# δ_t generators
# ---------------------------------------------------------------------------


@_register_delta(InterventionRegime.PUNCTUAL)
def _delta_punctual(
    t: int,
    tint: int,
    alpha: float,
    k: float = 1.0,
    n_days: int = 0,
) -> float:
    """Punctual intervention: one-time shock at the exact moment t_int.

    δ_t = α  if t == t_int
    δ_t = 0  if t ≠ t_int

    This simulates acute, one-time shocks (e.g. a single-day campaign).

    Args:
        t: Current time step.
        tint: Intervention time step.
        alpha: Intervention intensity.
        k: Not used for punctual (kept for uniform interface).
        n_days: Not used for punctual (kept for uniform interface).

    Returns:
        δ_t value for this time step.
    """
    return alpha if t == tint else 0.0


@_register_delta(InterventionRegime.CONTINUOUS)
def _delta_continuous(
    t: int,
    tint: int,
    alpha: float,
    k: float = 1.0,
    n_days: int = 0,
) -> float:
    """Continuous intervention: constant intensity from t_int onward.

    δ_t = 0  if t < t_int
    δ_t = α  if t ≥ t_int

    This simulates chronic treatments or sustained policies
    (e.g. always-on advertising).

    Args:
        t: Current time step.
        tint: Intervention time step.
        alpha: Intervention intensity.
        k: Not used for continuous (kept for uniform interface).
        n_days: Not used for continuous (kept for uniform interface).

    Returns:
        δ_t value for this time step.
    """
    return alpha if t >= tint else 0.0


@_register_delta(InterventionRegime.GRADUAL)
def _delta_gradual(
    t: int,
    tint: int,
    alpha: float,
    k: float = 1.0,
    n_days: int = 0,
) -> float:
    """Gradual intervention: phased ramp starting at t_int.

    δ_t = 0                if t < t_int
    δ_t = α · (t - t_int) / k  if t ≥ t_int

    The intervention scales progressively from t_int, incrementing at each
    subsequent time step scaled by factor k. This simulates accumulative or
    phased treatment plans (e.g. gradually increasing ad spend).

    Args:
        t: Current time step.
        tint: Intervention time step.
        alpha: Maximum intervention intensity.
        k: Scaling factor controlling ramp steepness.
            Larger k = slower ramp. Default: 1.0.
        n_days: Not used for gradual (kept for uniform interface).

    Returns:
        δ_t value for this time step.
    """
    if t < tint:
        return 0.0
    return alpha * (t - tint) / k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_delta_fn(regime: InterventionRegime) -> Callable[..., float]:
    """Get the δ_t generator function for an intervention regime.

    Args:
        regime: The intervention regime.

    Returns:
        Callable with signature:
            (t, tint, alpha, k, n_days) -> float

    Raises:
        ValueError: If no generator is registered for the regime.
    """
    fn = _DELTA_REGIMES.get(regime)
    if fn is None:
        raise ValueError(f"No delta_t generator registered for regime '{regime.value}'")
    return fn


def list_available_regimes() -> list[str]:
    """List all registered intervention regime names.

    Returns:
        Sorted list of regime names (e.g. ['continuous', 'gradual', 'punctual']).
    """
    return sorted(r.value for r in _DELTA_REGIMES)
