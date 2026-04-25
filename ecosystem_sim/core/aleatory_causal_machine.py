"""Aleatory causal machine — first-layer orchestrator with configurable structures.

This module provides the top-level entry point for generating causal trajectories
with aleatory (randomized) causal structures. It samples a causal graph topology
(NONE / DIRECT / CHAIN / CONFOUNDER) per simulation or per user, then generates
confounders, treatment, and outcome trajectories that respect the sampled structure.

Supports do-operator counterfactual interventions:
    - Pointwise:   do(T = c) — set treatment to a constant
    - Continuous:  do(T = T + delta) — additive shift
    - Gradual:     do(T = T * ramp) — linear ramp over time
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .causal_structure import (
    CausalStructure,
    CausalStructureConfig,
    CausalStructureSampler,
    get_outcome_equation,
    get_treatment_generator,
)
from .causal_engine import CausalEngine, CausalConfig


@dataclass
class AleatoryCausalConfig:
    """Top-level configuration for the aleatory causal machine."""

    structure_config: CausalStructureConfig = field(
        default_factory=CausalStructureConfig
    )
    engine_config: CausalConfig = field(default_factory=CausalConfig)
    random_seed: Optional[int] = None


class AleatoryCausalMachine:
    """First-layer causal machine with aleatory structure selection.

    This is the entry point that:
    1. Samples a causal structure (NONE/DIRECT/CHAIN/CONFOUNDER)
       either per-simulation or per-user
    2. Generates confounders via the SCM (category correlations)
    3. Generates treatment respecting the structure (Z->T for CHAIN/CONFOUNDER)
    4. Generates outcome trajectories via structure-aware equations
    5. Supports do-operator counterfactuals (pointwise/continuous/gradual)

    The aleatory factor creates realistic variability between users
    by randomizing which causal graph governs their behavior.
    """

    def __init__(
        self,
        config: AleatoryCausalConfig,
        users: list,
        priors: Optional[dict] = None,
    ):
        """Initialize the aleatory causal machine.

        Args:
            config: Top-level configuration.
            users: List of User objects.
            priors: Optional compiled dataset priors.
        """
        self._config = config
        self._users = users
        self._priors = priors
        self._rng = np.random.default_rng(config.random_seed)

        # Initialize the structure sampler
        self._structure_sampler = CausalStructureSampler(
            config.structure_config, self._rng
        )

        # Sample causal structure(s)
        self._user_structures: List[Tuple[CausalStructure, Dict[str, float]]] = (
            self._sample_structures()
        )

        # Initialize the base causal engine
        self._engine = CausalEngine(config.engine_config, users, priors)

    # ------------------------------------------------------------------
    # Structure sampling
    # ------------------------------------------------------------------

    def _sample_structures(self) -> List[Tuple[CausalStructure, Dict[str, float]]]:
        """Sample causal structures per the randomization strategy.

        Returns:
            List of (structure, params) tuples, one per user.
        """
        if self._config.structure_config.randomize_per == "simulation":
            structure, params = self._structure_sampler.sample_structure()
            return [(structure, params) for _ in self._users]

        # Per-user randomization
        return [self._structure_sampler.sample_structure() for _ in self._users]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> CausalEngine:
        """The underlying causal engine."""
        return self._engine

    @property
    def user_structures(self) -> List[Tuple[CausalStructure, Dict[str, float]]]:
        """Per-user (structure, params) assignments."""
        return list(self._user_structures)

    @property
    def dominant_structure(self) -> CausalStructure:
        """The most common causal structure across users."""
        from collections import Counter

        counts = Counter(s for s, _ in self._user_structures)
        return counts.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Factual trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectories(self) -> np.ndarray:
        """Generate factual trajectories under the aleatory structures.

        Delegates to CausalEngine but overrides the treatment generation
        and outcome equation per user based on their sampled structure.

        Returns:
            Array of shape (n_users, n_days, n_categories).
        """
        n_users = self._engine.n_users
        n_days = self._engine.config.n_days
        n_cats = self._engine.config.n_categories

        interest_matrix = np.zeros((n_users, n_days, n_cats))
        interest_matrix[:, 0, :] = self._engine.confounders.copy()

        for day in range(1, n_days):
            for user_idx in range(n_users):
                structure, params = self._user_structures[user_idx]
                prev = interest_matrix[user_idx, day - 1, :]
                conf = self._engine.confounders[user_idx, :]

                # Generate treatment respecting the causal structure
                treatment = self._generate_treatment_for_user(
                    user_idx, day, structure, params
                )

                # Compute outcome via structure-aware equation
                equation = get_outcome_equation(structure)
                interest_matrix[user_idx, day, :] = equation(
                    prev_interest=prev,
                    confounder=conf,
                    treatment=treatment,
                    treatment_effect=self._engine.treatment_effects[user_idx],
                    params=params,
                    persistence_factor=self._engine.config.persistence_factor,
                    mean_reversion_strength=self._engine.config.mean_reversion_strength,
                )

        return interest_matrix

    def _generate_treatment_for_user(
        self,
        user_idx: int,
        day: int,
        structure: CausalStructure,
        params: Dict[str, float],
    ) -> float:
        """Generate a single treatment value respecting the causal structure.

        Args:
            user_idx: User index.
            day: Day index.
            structure: The causal structure for this user.
            params: Structure-specific parameters.

        Returns:
            Treatment value in [0, 1].
        """
        base_exposure = self._users[user_idx].persona.ad_click_propensity * 0.5
        confounder_mean = float(self._engine.confounders[user_idx].mean())

        generator = get_treatment_generator(structure)
        return generator(base_exposure, confounder_mean, params, self._rng)

    # ------------------------------------------------------------------
    # Counterfactual trajectory generation
    # ------------------------------------------------------------------

    def generate_counterfactual(
        self,
        intervention_type: str = "pointwise",
        intervention_strength: float = 0.0,
        intervention_start_day: int = 0,
        intervention_end_day: Optional[int] = None,
    ) -> np.ndarray:
        """Generate counterfactual trajectories under do-operator.

        The do-operator blocks T<-Z paths (in CHAIN and CONFOUNDER)
        but leaves the Z->Y path open — correct Pearlian causality.

        Args:
            intervention_type: One of "pointwise", "continuous", "gradual".
            intervention_strength:
                - pointwise:  constant value to set T to (0-1)
                - continuous: additive shift to apply to T (0-1)
                - gradual:    multiplier for linear ramp (0-3)
            intervention_start_day: Day index when intervention begins.
            intervention_end_day: Day index when intervention ends.
                None means end of simulation.

        Returns:
            Counterfactual interest matrix of shape
            (n_users, n_days, n_categories).
        """
        n_users = self._engine.n_users
        n_days = self._engine.config.n_days
        n_cats = self._engine.config.n_categories

        if intervention_end_day is None:
            intervention_end_day = n_days

        # Build counterfactual treatment matrix
        cf_treatment = self._build_counterfactual_treatment(
            intervention_type,
            intervention_strength,
            intervention_start_day,
            intervention_end_day,
        )

        # Generate trajectories with counterfactual treatment, keeping Z fixed
        interest_matrix = np.zeros((n_users, n_days, n_cats))
        interest_matrix[:, 0, :] = self._engine.confounders.copy()

        for day in range(1, n_days):
            for user_idx in range(n_users):
                structure, params = self._user_structures[user_idx]
                prev = interest_matrix[user_idx, day - 1, :]
                conf = self._engine.confounders[user_idx, :]

                equation = get_outcome_equation(structure)
                interest_matrix[user_idx, day, :] = equation(
                    prev_interest=prev,
                    confounder=conf,
                    treatment=cf_treatment[user_idx, day],
                    treatment_effect=self._engine.treatment_effects[user_idx],
                    params=params,
                    persistence_factor=self._engine.config.persistence_factor,
                    mean_reversion_strength=self._engine.config.mean_reversion_strength,
                )

        return interest_matrix

    def _build_counterfactual_treatment(
        self,
        intervention_type: str,
        intervention_strength: float,
        intervention_start_day: int,
        intervention_end_day: int,
    ) -> np.ndarray:
        """Build the counterfactual treatment matrix under do-operator.

        Args:
            intervention_type: "pointwise", "continuous", or "gradual".
            intervention_strength: Magnitude of the intervention.
            intervention_start_day: When intervention begins.
            intervention_end_day: When intervention ends.

        Returns:
            Modified treatment matrix of shape (n_users, n_days).
        """
        cf_treatment = self._engine.treatment_matrix.copy()

        for day in range(intervention_start_day, intervention_end_day):
            if intervention_type == "pointwise":
                # do(T = c): set treatment to a constant value
                cf_treatment[:, day] = intervention_strength

            elif intervention_type == "continuous":
                # do(T = T + delta): additive shift
                cf_treatment[:, day] = np.clip(
                    cf_treatment[:, day] + intervention_strength, 0, 1
                )

            elif intervention_type == "gradual":
                # do(T = T * ramp): linear ramp from 0 to strength
                ramp_progress = (day - intervention_start_day) / max(
                    1, intervention_end_day - intervention_start_day
                )
                ramp_factor = float(1.0 + intervention_strength * ramp_progress)
                cf_treatment[:, day] = np.clip(
                    cf_treatment[:, day] * ramp_factor,
                    0,
                    1,
                )

            else:
                raise ValueError(
                    f"Unknown intervention_type '{intervention_type}'. "
                    f"Must be one of: pointwise, continuous, gradual."
                )

        return cf_treatment

    # ------------------------------------------------------------------
    # Convenience counterfactual methods
    # ------------------------------------------------------------------

    def generate_no_treatment_counterfactual(self) -> np.ndarray:
        """do(T=0): remove all treatment."""
        return self.generate_counterfactual(
            intervention_type="pointwise",
            intervention_strength=0.0,
            intervention_start_day=0,
        )

    def generate_double_treatment_counterfactual(self) -> np.ndarray:
        """do(T=2T): double treatment (capped at 1)."""
        return self.generate_counterfactual(
            intervention_type="continuous",
            intervention_strength=1.0,
            intervention_start_day=0,
        )

    def generate_gradual_ramp_counterfactual(
        self, multiplier: float = 2.0, start_day: int = 14
    ) -> np.ndarray:
        """do(T = T * ramp): gradually increase treatment.

        Args:
            multiplier: Maximum multiplier at end of ramp.
            start_day: Day to begin the ramp.

        Returns:
            Counterfactual interest matrix.
        """
        return self.generate_counterfactual(
            intervention_type="gradual",
            intervention_strength=multiplier - 1.0,
            intervention_start_day=start_day,
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_structure_summary(self) -> Dict[str, int]:
        """Get a summary of how many users have each causal structure.

        Returns:
            Dict mapping structure name -> count.
        """
        from collections import Counter

        counts: Counter = Counter(s.value for s, _ in self._user_structures)
        return dict(counts)
