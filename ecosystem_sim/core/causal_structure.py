"""Configurable causal structures for the first-layer SCM.

Defines the causal graph topologies that govern how confounders (Z),
treatment (T), and outcomes (Y) relate to each other. The structure
is sampled aleatorily per simulation or per user, creating realistic
heterogeneity in the data-generating process.

Supported structures:
    - NONE:       Independent categories, no causal links
    - DIRECT:     T → Y (treatment directly affects outcome)
    - CHAIN:      Z → T → Y (confounder → treatment → outcome)
    - CONFOUNDER: Z → T, Z → Y (common cause affects both)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class CausalStructure(Enum):
    """Enumeration of supported causal graph topologies."""

    NONE = "none"
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
            CausalStructure.NONE,
            CausalStructure.DIRECT,
            CausalStructure.CHAIN,
            CausalStructure.CONFOUNDER,
        ]
    )
    structure_probs: Optional[List[float]] = None
    randomize_per: str = "simulation"

    # Parameter ranges (aleatory)
    direct_effect_range: Tuple[float, float] = (0.1, 0.5)
    chain_confounder_effect_range: Tuple[float, float] = (0.2, 0.6)
    chain_treatment_effect_range: Tuple[float, float] = (0.1, 0.4)
    confounder_effect_on_treatment: Tuple[float, float] = (0.3, 0.7)
    confounder_effect_on_outcome: Tuple[float, float] = (0.2, 0.5)
    noise_std_range: Tuple[float, float] = (0.02, 0.10)

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
            ("direct_effect_range", self.direct_effect_range),
            ("chain_confounder_effect_range", self.chain_confounder_effect_range),
            ("chain_treatment_effect_range", self.chain_treatment_effect_range),
            ("confounder_effect_on_treatment", self.confounder_effect_on_treatment),
            ("confounder_effect_on_outcome", self.confounder_effect_on_outcome),
            ("noise_std_range", self.noise_std_range),
        ]:
            if lo < 0 or hi > 1.0 or lo >= hi:
                raise ValueError(
                    f"{name} must satisfy 0 <= lo < hi <= 1, got ({lo}, {hi})"
                )


# ---------------------------------------------------------------------------
# Parameter samplers per structure (Strategy Pattern via registry)
# ---------------------------------------------------------------------------


@_register_sampler(CausalStructure.NONE)
def _sample_none_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for NONE structure — only noise."""
    return {"noise_std": float(rng.uniform(*config.noise_std_range))}


@_register_sampler(CausalStructure.DIRECT)
def _sample_direct_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for DIRECT structure — direct effect + noise."""
    return {
        "noise_std": float(rng.uniform(*config.noise_std_range)),
        "direct_effect": float(rng.uniform(*config.direct_effect_range)),
    }


@_register_sampler(CausalStructure.CHAIN)
def _sample_chain_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for CHAIN structure — Z→T→Y."""
    return {
        "noise_std": float(rng.uniform(*config.noise_std_range)),
        "confounder_effect": float(rng.uniform(*config.chain_confounder_effect_range)),
        "treatment_effect": float(rng.uniform(*config.chain_treatment_effect_range)),
    }


@_register_sampler(CausalStructure.CONFOUNDER)
def _sample_confounder_params(
    config: CausalStructureConfig, rng: np.random.Generator
) -> Dict[str, float]:
    """Sample parameters for CONFOUNDER structure — Z→T and Z→Y."""
    return {
        "noise_std": float(rng.uniform(*config.noise_std_range)),
        "confounder_on_treatment": float(
            rng.uniform(*config.confounder_effect_on_treatment)
        ),
        "confounder_on_outcome": float(
            rng.uniform(*config.confounder_effect_on_outcome)
        ),
    }


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


@_register_outcome(CausalStructure.NONE)
def _outcome_none(
    prev_interest: np.ndarray,
    confounder: np.ndarray,
    treatment: float,
    treatment_effect: float,
    params: Dict[str, float],
    persistence_factor: float,
    mean_reversion_strength: float,
) -> np.ndarray:
    """Y = AR(1) + beta*T + epsilon — no confounder path."""
    ar = persistence_factor * prev_interest
    treatment_component = treatment_effect * treatment
    reversion = mean_reversion_strength * (confounder - prev_interest)
    noise = np.random.normal(0, params["noise_std"], prev_interest.shape)
    return np.clip(ar + treatment_component + reversion + noise, 0, 1)


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
    """Y = AR(1) + beta*T + epsilon — beta sampled from direct_effect_range."""
    ar = persistence_factor * prev_interest
    beta = params["direct_effect"]
    treatment_component = beta * treatment
    reversion = mean_reversion_strength * (confounder - prev_interest)
    noise = np.random.normal(0, params["noise_std"], prev_interest.shape)
    return np.clip(ar + treatment_component + reversion + noise, 0, 1)


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
    """Y = AR(1) + alpha*Z + beta*T + epsilon — Z->T->Y, Z->Y path open."""
    ar = persistence_factor * prev_interest
    alpha = params["confounder_effect"]
    confounder_component = alpha * confounder
    treatment_component = treatment_effect * treatment
    reversion = mean_reversion_strength * (confounder - prev_interest)
    noise = np.random.normal(0, params["noise_std"], prev_interest.shape)
    return np.clip(
        ar + confounder_component + treatment_component + reversion + noise, 0, 1
    )


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
    """Y = AR(1) + alpha*Z + beta*T + epsilon — Z->T and Z->Y, both open."""
    ar = persistence_factor * prev_interest
    alpha = params["confounder_on_outcome"]
    confounder_component = alpha * confounder
    treatment_component = treatment_effect * treatment
    reversion = mean_reversion_strength * (confounder - prev_interest)
    noise = np.random.normal(0, params["noise_std"], prev_interest.shape)
    return np.clip(
        ar + confounder_component + treatment_component + reversion + noise, 0, 1
    )


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


@_register_treatment(CausalStructure.NONE)
def _treatment_none(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Treatment independent of confounders."""
    campaign_noise = rng.beta(2, 5) if rng.random() < 0.1 else rng.beta(1, 3) * 0.3
    return float(np.clip(base_exposure + campaign_noise, 0, 1))


@_register_treatment(CausalStructure.DIRECT)
def _treatment_direct(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Treatment independent of confounders (direct effect is in outcome eq)."""
    campaign_noise = rng.beta(2, 5) if rng.random() < 0.1 else rng.beta(1, 3) * 0.3
    return float(np.clip(base_exposure + campaign_noise, 0, 1))


@_register_treatment(CausalStructure.CHAIN)
def _treatment_chain(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Z -> T: confounders influence treatment assignment."""
    confounder_influence = params["confounder_effect"] * confounder_mean
    campaign_noise = rng.beta(2, 5) if rng.random() < 0.1 else rng.beta(1, 3) * 0.3
    return float(np.clip(base_exposure + confounder_influence + campaign_noise, 0, 1))


@_register_treatment(CausalStructure.CONFOUNDER)
def _treatment_confounder(
    base_exposure: float,
    confounder_mean: float,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Z -> T: confounders influence treatment assignment."""
    confounder_influence = params["confounder_on_treatment"] * confounder_mean
    campaign_noise = rng.beta(2, 5) if rng.random() < 0.1 else rng.beta(1, 3) * 0.3
    return float(np.clip(base_exposure + confounder_influence + campaign_noise, 0, 1))


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
