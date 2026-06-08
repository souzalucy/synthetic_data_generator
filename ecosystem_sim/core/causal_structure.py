"""Configurable causal structures for the first-layer SCM.

Defines the causal graph topologies that govern how treatment (T),
mediator/confounder (X), and outcomes (Y) relate to each other.
The structure is sampled aleatorily per user, creating realistic
heterogeneity in the data-generating process.

Supported structures (3 total, per user specification):
    - DIRECT:     T → Y
                  Tt = ΦT·Tt−1 + εT,t
                  Yt = ΦY·Yt−1 + βTY·f(Tt) + εY,t

    - CHAIN:      T → X → Y  (mediation)
                  Tt = ΦT·Tt−1 + εT,t
                  Xt = βTX·f(Tt) + εX,t
                  Yt = ΦY·Yt−1 + βXY·f(Xt) + εY,t

    - CONFOUNDER: X → T, X → Y  (common cause)
                  Xt = ΦX·Xt−1 + εX,t
                  Tt = ΦT·Tt−1 + βXT·f(Xt) + εT,t
                  Yt = ΦY·Yt−1 + βXY·f(Xt) + βTY·g(Tt) + εY,t
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class CausalStructure(Enum):
    """Enumeration of supported causal graph topologies."""

    DIRECT = "direct"
    CHAIN = "chain"
    CONFOUNDER = "confounder"


# ---------------------------------------------------------------------------
# Parameter sampling strategy: maps each structure to its sampler function
# ---------------------------------------------------------------------------

_STRUCTURE_PARAM_SAMPLERS: Dict[CausalStructure, Callable[..., Dict[str, float]]] = {}


def _register_sampler(structure: CausalStructure) -> Callable[[Callable], Callable]:
    """Decorator to register a parameter sampler for a causal structure."""

    def decorator(
        fn: Callable[..., Dict[str, float]],
    ) -> Callable[..., Dict[str, float]]:
        _STRUCTURE_PARAM_SAMPLERS[structure] = fn
        return fn

    return decorator


@dataclass
class CausalStructureConfig:
    """Configuration for aleatory causal structure selection.

    Controls which structures are available, their sampling probabilities,
    and the parameter ranges for each structure's coefficients.
    """

    available_structures: List[CausalStructure] = field(
        default_factory=lambda: [
            CausalStructure.DIRECT,
            CausalStructure.CHAIN,
            CausalStructure.CONFOUNDER,
        ]
    )
    structure_probs: Optional[List[float]] = None
    randomize_per: str = "user"

    # Parameter ranges (aleatory)
    # Shared
    persistence_factor_range: Tuple[float, float] = (0.6, 0.9)
    noise_std_range: Tuple[float, float] = (0.02, 0.10)

    # Direct: T → Y
    direct_treatment_effect_range: Tuple[float, float] = (0.1, 0.5)

    # Chain (Mediation): T → X → Y
    chain_tx_effect_range: Tuple[float, float] = (0.2, 0.6)
    chain_xy_effect_range: Tuple[float, float] = (0.1, 0.4)

    # Confounder: X → T, X → Y
    confounder_xt_effect_range: Tuple[float, float] = (0.3, 0.7)
    confounder_xy_effect_range: Tuple[float, float] = (0.2, 0.5)
    confounder_ty_effect_range: Tuple[float, float] = (0.1, 0.4)

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.structure_probs is not None:
            n_avail = len(self.available_structures)
            n_probs = len(self.structure_probs)
            if n_probs != n_avail:
                raise ValueError(
                    f"structure_probs length ({n_probs}) must match "
                    f"available_structures length ({n_avail})"
                )
            if not np.isclose(sum(self.structure_probs), 1.0):
                raise ValueError(
                    f"structure_probs must sum to 1.0, got {sum(self.structure_probs)}"
                )

        valid_modes = ("simulation", "user")
        if self.randomize_per not in valid_modes:
            raise ValueError(
                f"randomize_per must be one of {valid_modes}, "
                f"got '{self.randomize_per}'"
            )

        for name, (lo, hi) in [
            ("persistence_factor_range", self.persistence_factor_range),
            ("noise_std_range", self.noise_std_range),
            ("direct_treatment_effect_range", self.direct_treatment_effect_range),
            ("chain_tx_effect_range", self.chain_tx_effect_range),
            ("chain_xy_effect_range", self.chain_xy_effect_range),
            ("confounder_xt_effect_range", self.confounder_xt_effect_range),
            ("confounder_xy_effect_range", self.confounder_xy_effect_range),
            ("confounder_ty_effect_range", self.confounder_ty_effect_range),
        ]:
            if lo < 0 or hi > 1.0 or lo >= hi:
                raise ValueError(
                    f"{name} must satisfy 0 <= lo < hi <= 1, got ({lo}, {hi})"
                )


# ---------------------------------------------------------------------------
# Parameter samplers per structure (Strategy Pattern via registry)
# ---------------------------------------------------------------------------


@_register_sampler(CausalStructure.DIRECT)
def _sample_direct_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for DIRECT structure.

    Tt = ΦT·Tt−1 + εT,t
    Yt = ΦY·Yt−1 + βTY·f(Tt) + εY,t
    """
    return {
        "persistence_factor_T": float(rng.uniform(*config.persistence_factor_range)),
        "persistence_factor_Y": float(rng.uniform(*config.persistence_factor_range)),
        "noise_std_T": float(rng.uniform(*config.noise_std_range)),
        "noise_std_Y": float(rng.uniform(*config.noise_std_range)),
        "beta_TY": float(rng.uniform(*config.direct_treatment_effect_range)),
    }


@_register_sampler(CausalStructure.CHAIN)
def _sample_chain_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for CHAIN (mediation) structure.

    Tt = ΦT·Tt−1 + εT,t
    Xt = βTX·f(Tt) + εX,t
    Yt = ΦY·Yt−1 + βXY·f(Xt) + εY,t
    """
    return {
        "persistence_factor_T": float(rng.uniform(*config.persistence_factor_range)),
        "persistence_factor_Y": float(rng.uniform(*config.persistence_factor_range)),
        "noise_std_T": float(rng.uniform(*config.noise_std_range)),
        "noise_std_X": float(rng.uniform(*config.noise_std_range)),
        "noise_std_Y": float(rng.uniform(*config.noise_std_range)),
        "beta_TX": float(rng.uniform(*config.chain_tx_effect_range)),
        "beta_XY": float(rng.uniform(*config.chain_xy_effect_range)),
    }


@_register_sampler(CausalStructure.CONFOUNDER)
def _sample_confounder_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for CONFOUNDER structure.

    Xt = ΦX·Xt−1 + εX,t
    Tt = ΦT·Tt−1 + βXT·f(Xt) + εT,t
    Yt = ΦY·Yt−1 + βXY·f(Xt) + βTY·g(Tt) + εY,t
    """
    return {
        "persistence_factor_X": float(rng.uniform(*config.persistence_factor_range)),
        "persistence_factor_T": float(rng.uniform(*config.persistence_factor_range)),
        "persistence_factor_Y": float(rng.uniform(*config.persistence_factor_range)),
        "noise_std_X": float(rng.uniform(*config.noise_std_range)),
        "noise_std_T": float(rng.uniform(*config.noise_std_range)),
        "noise_std_Y": float(rng.uniform(*config.noise_std_range)),
        "beta_XT": float(rng.uniform(*config.confounder_xt_effect_range)),
        "beta_XY": float(rng.uniform(*config.confounder_xy_effect_range)),
        "beta_TY": float(rng.uniform(*config.confounder_ty_effect_range)),
    }


# ---------------------------------------------------------------------------
# Non-linear transformation functions f() and g()
# ---------------------------------------------------------------------------


def _default_f(x: np.ndarray) -> np.ndarray:
    """Default non-linear transformation f: sigmoid squash.

    Maps treatment/confounder values to [0, 1] with a soft S-curve.
    """
    return 1.0 / (1.0 + np.exp(-5.0 * (x - 0.5)))


def _default_g(x: np.ndarray) -> np.ndarray:
    """Default non-linear transformation g: quadratic amplification.

    Allows treatment to have a non-linear (amplifying) effect on outcome.
    """
    return np.clip(x**2, 0, 1)


# ---------------------------------------------------------------------------
# Outcome equation strategy: maps each structure to its equation function
# ---------------------------------------------------------------------------

_OUTCOME_EQUATIONS: Dict[CausalStructure, Callable[..., np.ndarray]] = {}


def _register_outcome(structure: CausalStructure) -> Callable[[Callable], Callable]:
    """Decorator to register an outcome equation for a causal structure."""

    def decorator(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        _OUTCOME_EQUATIONS[structure] = fn
        return fn

    return decorator


@_register_outcome(CausalStructure.DIRECT)
def _outcome_direct(
    prev_interest: np.ndarray,
    confounder: np.ndarray,
    treatment: float,
    treatment_effect: float,
    params: Dict[str, float],
    persistence_factor: float,
    mean_reversion_strength: float,
) -> np.ndarray:
    """Yt = ΦY·Yt−1 + βTY·f(Tt) + εY,t

    Args:
        prev_interest: Yt−1 — previous day's outcome (n_cats,).
        confounder: Not used in DIRECT (kept for uniform interface).
        treatment: Tt — current day's treatment value (scalar).
        treatment_effect: Not used directly; beta_TY from params is used.
        params: Must contain 'persistence_factor_Y', 'beta_TY', 'noise_std_Y'.
        persistence_factor: Fallback if not in params.
        mean_reversion_strength: Not used in DIRECT (no mean reversion).

    Returns:
        Yt — outcome vector (n_cats,).
    """
    phi_Y = params.get("persistence_factor_Y", persistence_factor)
    beta_TY = params.get("beta_TY", treatment_effect)
    noise_std = params.get("noise_std_Y", 0.05)

    ar = phi_Y * prev_interest
    treatment_component = beta_TY * _default_f(np.full_like(prev_interest, treatment))
    noise = np.random.normal(0, noise_std, prev_interest.shape)

    return np.clip(ar + treatment_component + noise, 0, 1)


@_register_outcome(CausalStructure.CHAIN)
def _outcome_chain(
    prev_interest: np.ndarray,
    confounder: np.ndarray,
    treatment: float,
    treatment_effect: float,
    params: Dict[str, float],
    persistence_factor: float,
    mean_reversion_strength: float,
) -> np.ndarray:
    """Yt = ΦY·Yt−1 + βXY·f(Xt) + εY,t

    The mediator Xt is computed from the treatment: Xt = βTX·f(Tt) + εX,t.
    The confounder argument is used to pass the mediator Xt (pre-computed
    by the caller in AleatoryCausalMachine for the CHAIN structure).

    Args:
        prev_interest: Yt−1 — previous day's outcome (n_cats,).
        confounder: Xt — mediator value (n_cats,), pre-computed from treatment.
        treatment: Tt — not used directly in CHAIN outcome (mediated via X).
        treatment_effect: Not used directly; beta_XY from params is used.
        params: Must contain 'persistence_factor_Y', 'beta_XY', 'noise_std_Y'.
        persistence_factor: Fallback if not in params.
        mean_reversion_strength: Not used in CHAIN (no mean reversion).

    Returns:
        Yt — outcome vector (n_cats,).
    """
    phi_Y = params.get("persistence_factor_Y", persistence_factor)
    beta_XY = params.get("beta_XY", treatment_effect)
    noise_std = params.get("noise_std_Y", 0.05)

    ar = phi_Y * prev_interest
    mediator_component = beta_XY * _default_f(confounder)
    noise = np.random.normal(0, noise_std, prev_interest.shape)

    return np.clip(ar + mediator_component + noise, 0, 1)


@_register_outcome(CausalStructure.CONFOUNDER)
def _outcome_confounder(
    prev_interest: np.ndarray,
    confounder: np.ndarray,
    treatment: float,
    treatment_effect: float,
    params: Dict[str, float],
    persistence_factor: float,
    mean_reversion_strength: float,
) -> np.ndarray:
    """Yt = ΦY·Yt−1 + βXY·f(Xt) + βTY·g(Tt) + εY,t

    Args:
        prev_interest: Yt−1 — previous day's outcome (n_cats,).
        confounder: Xt — confounder value (n_cats,).
        treatment: Tt — current day's treatment value (scalar).
        treatment_effect: Not used directly; beta_TY from params is used.
        params: Must contain 'persistence_factor_Y', 'beta_XY', 'beta_TY',
                'noise_std_Y'.
        persistence_factor: Fallback if not in params.
        mean_reversion_strength: Not used in CONFOUNDER (no mean reversion).

    Returns:
        Yt — outcome vector (n_cats,).
    """
    phi_Y = params.get("persistence_factor_Y", persistence_factor)
    beta_XY = params.get("beta_XY", 0.3)
    beta_TY = params.get("beta_TY", treatment_effect)
    noise_std = params.get("noise_std_Y", 0.05)

    ar = phi_Y * prev_interest
    confounder_component = beta_XY * _default_f(confounder)
    treatment_component = beta_TY * _default_g(np.full_like(prev_interest, treatment))
    noise = np.random.normal(0, noise_std, prev_interest.shape)

    return np.clip(ar + confounder_component + treatment_component + noise, 0, 1)


# ---------------------------------------------------------------------------
# Treatment generation strategy: maps each structure to its treatment fn
# ---------------------------------------------------------------------------

_TREATMENT_GENERATORS: Dict[CausalStructure, Callable[..., float]] = {}


def _register_treatment(structure: CausalStructure) -> Callable[[Callable], Callable]:
    """Decorator to register a treatment generator for a causal structure."""

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        _TREATMENT_GENERATORS[structure] = fn
        return fn

    return decorator


@_register_treatment(CausalStructure.DIRECT)
def _treatment_direct(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Tt = ΦT·Tt−1 + εT,t — treatment evolves as its own AR(1) process.

    Note: The caller (AleatoryCausalMachine) handles the AR(1) recursion.
    This function generates the noise term εT,t for the current day.

    Args:
        base_exposure: Base treatment level from persona.
        confounder_mean: Not used in DIRECT.
        params: Must contain 'noise_std_T'.
        rng: Random generator.

    Returns:
        Treatment value in [0, 1].
    """
    noise_std = params.get("noise_std_T", 0.05)
    noise = rng.normal(0, noise_std)
    return float(np.clip(base_exposure + noise, 0, 1))


@_register_treatment(CausalStructure.CHAIN)
def _treatment_chain(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Tt = ΦT·Tt−1 + εT,t — same as DIRECT (no confounder influence on T).

    In the CHAIN (mediation) structure, treatment is independent of
    confounders. The mediation happens via T → X → Y.

    Args:
        base_exposure: Base treatment level from persona.
        confounder_mean: Not used in CHAIN.
        params: Must contain 'noise_std_T'.
        rng: Random generator.

    Returns:
        Treatment value in [0, 1].
    """
    noise_std = params.get("noise_std_T", 0.05)
    noise = rng.normal(0, noise_std)
    return float(np.clip(base_exposure + noise, 0, 1))


@_register_treatment(CausalStructure.CONFOUNDER)
def _treatment_confounder(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Tt = ΦT·Tt−1 + βXT·f(Xt) + εT,t — confounder influences treatment.

    Note: The AR(1) part ΦT·Tt−1 is handled by the caller. This function
    generates the confounder influence βXT·f(Xt) + εT,t.

    Args:
        base_exposure: Base treatment level from persona.
        confounder_mean: Mean of confounder Xt across categories.
        params: Must contain 'beta_XT', 'noise_std_T'.
        rng: Random generator.

    Returns:
        Treatment value in [0, 1].
    """
    beta_XT = params.get("beta_XT", 0.3)
    noise_std = params.get("noise_std_T", 0.05)
    confounder_influence = beta_XT * float(_default_f(np.array([confounder_mean]))[0])
    noise = rng.normal(0, noise_std)
    return float(np.clip(base_exposure + confounder_influence + noise, 0, 1))


# ---------------------------------------------------------------------------
# Mediator generation for CHAIN structure
# ---------------------------------------------------------------------------

_MEDIATOR_GENERATORS: Dict[CausalStructure, Callable[..., np.ndarray]] = {}


def _register_mediator(structure: CausalStructure) -> Callable[[Callable], Callable]:
    """Decorator to register a mediator generator for a causal structure."""

    def decorator(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        _MEDIATOR_GENERATORS[structure] = fn
        return fn

    return decorator


@_register_mediator(CausalStructure.CHAIN)
def _mediator_chain(
    treatment: float,
    params: Dict[str, float],
    n_categories: int,
) -> np.ndarray:
    """Xt = βTX·f(Tt) + εX,t — mediator is a function of treatment.

    Args:
        treatment: Tt — current day's treatment value (scalar).
        params: Must contain 'beta_TX', 'noise_std_X'.
        n_categories: Number of categories (dimensionality of X).

    Returns:
        Xt — mediator vector (n_categories,).
    """
    beta_TX = params.get("beta_TX", 0.3)
    noise_std = params.get("noise_std_X", 0.05)
    tx_component = beta_TX * _default_f(np.full(n_categories, treatment))
    noise = np.random.normal(0, noise_std, n_categories)
    return np.clip(tx_component + noise, 0, 1)


@_register_mediator(CausalStructure.CONFOUNDER)
def _mediator_confounder(
    treatment: float,
    params: Dict[str, float],
    n_categories: int,
) -> np.ndarray:
    """Xt = ΦX·Xt−1 + εX,t — confounder evolves as its own AR(1) process.

    Note: The AR(1) recursion is handled by the caller. This function
    generates the noise term for the current day.

    Args:
        treatment: Not used in CONFOUNDER mediator (X is independent of T).
        params: Must contain 'noise_std_X'.
        n_categories: Number of categories.

    Returns:
        Xt — confounder noise vector (n_categories,).
    """
    noise_std = params.get("noise_std_X", 0.05)
    return np.random.normal(0, noise_std, n_categories)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CausalStructureSampler:
    """Samples causal structures and their parameters aleatorily.

    This is the core aleatory factor — it introduces randomness into
    which causal graph governs each user's trajectory.
    """

    def __init__(self, config: CausalStructureConfig, rng: np.random.Generator):
        """Initialize sampler.

        Args:
            config: Configuration for structure selection and parameter ranges.
            rng: NumPy random generator for reproducibility.

        Raises:
            ValueError: If config is invalid.
        """
        self._config = config
        self._rng = rng
        self._probs = self._resolve_probs(config)

    @staticmethod
    def _resolve_probs(config: CausalStructureConfig) -> np.ndarray:
        """Resolve sampling probabilities, defaulting to uniform."""
        if config.structure_probs is not None:
            return np.array(config.structure_probs)
        n = len(config.available_structures)
        return np.ones(n) / n

    def sample_structure(self) -> Tuple[CausalStructure, Dict[str, float]]:
        """Sample a causal structure and its random parameters.

        Returns:
            (structure, params) where params is a dict of structure-specific
            coefficients sampled from the configured ranges.
        """
        idx = self._rng.choice(len(self._config.available_structures), p=self._probs)
        structure = self._config.available_structures[idx]
        params = self._sample_params(structure)
        return structure, params

    def _sample_params(self, structure: CausalStructure) -> Dict[str, float]:
        """Sample random parameters for the given structure via strategy registry."""
        sampler = _STRUCTURE_PARAM_SAMPLERS.get(structure)
        if sampler is None:
            raise ValueError(f"No parameter sampler registered for {structure}")
        return sampler(self._config, self._rng)


def get_outcome_equation(structure: CausalStructure) -> Callable[..., np.ndarray]:
    """Get the outcome equation function for a causal structure.

    Args:
        structure: The causal structure.

    Returns:
        Callable with signature:
            (prev_interest, confounder, treatment, treatment_effect,
             params, persistence_factor, mean_reversion_strength) -> np.ndarray

    Raises:
        ValueError: If no equation is registered for the structure.
    """
    equation = _OUTCOME_EQUATIONS.get(structure)
    if equation is None:
        raise ValueError(f"No outcome equation registered for {structure}")
    return equation


def get_treatment_generator(structure: CausalStructure) -> Callable[..., float]:
    """Get the treatment generation function for a causal structure.

    Args:
        structure: The causal structure.

    Returns:
        Callable with signature:
            (base_exposure, confounder_mean, params, rng) -> float

    Raises:
        ValueError: If no generator is registered for the structure.
    """
    generator = _TREATMENT_GENERATORS.get(structure)
    if generator is None:
        raise ValueError(f"No treatment generator registered for {structure}")
    return generator


def get_mediator_generator(structure: CausalStructure) -> Callable[..., np.ndarray]:
    """Get the mediator/confounder generation function for a causal structure.

    For CHAIN: generates Xt = βTX·f(Tt) + εX,t (mediator from treatment).
    For CONFOUNDER: generates noise for Xt = ΦX·Xt−1 + εX,t (AR(1) confounder).
    For DIRECT: raises ValueError (no mediator in DIRECT).

    Args:
        structure: The causal structure.

    Returns:
        Callable with signature:
            (treatment, params, n_categories) -> np.ndarray

    Raises:
        ValueError: If no mediator generator is registered for the structure.
    """
    generator = _MEDIATOR_GENERATORS.get(structure)
    if generator is None:
        raise ValueError(f"No mediator generator registered for {structure}")
    return generator
