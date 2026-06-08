"""Aleatory causal machine — first-layer orchestrator with configurable structures.

This module provides the top-level entry point for generating causal trajectories
with aleatory (randomized) causal structures. It samples a causal graph topology
(DIRECT / CHAIN / CONFOUNDER) per simulation or per user, then generates
treatment, mediator/confounder, and outcome trajectories that respect the
sampled structure's equations.

Supported structures (per user specification):
    - DIRECT:     Tt = ΦT·Tt−1 + εT,t
                  Yt = ΦY·Yt−1 + βTY·f(Tt) + εY,t

    - CHAIN:      Tt = ΦT·Tt−1 + εT,t
                  Xt = βTX·f(Tt) + εX,t
                  Yt = ΦY·Yt−1 + βXY·f(Xt) + εY,t

    - CONFOUNDER: Xt = ΦX·Xt−1 + εX,t
                  Tt = ΦT·Tt−1 + βXT·f(Xt) + εT,t
                  Yt = ΦY·Yt−1 + βXY·f(Xt) + βTY·g(Tt) + εY,t

Supports do-operator counterfactual interventions:
    - Pointwise:   do(T = c) — set treatment to a constant
    - Continuous:  do(T = T + delta) — additive shift
    - Gradual:     do(T = T * ramp) — linear ramp over time

Structural counterfactual generation (CSDG framework):
    Generates counterfactual trajectories by injecting δ_t intervention
    regimes directly into the structural equations:

        T^{cf}_t = Φ_T · T^{cf}_{t-1} + ε_{T,t} + δ_t          (Eq 6)
        Y^{cf}_t = Φ_Y · Y^{cf}_{t-1} + β_{TY} · f(T^{cf}_t) + ε_{Y,t}  (Eq 7)

    Three δ_t regimes:
        - PUNCTUAL:    δ_t = α if t == t_int else 0
        - CONTINUOUS:  δ_t = α if t >= t_int else 0
        - GRADUAL:     δ_t = α · (t - t_int) / k if t >= t_int else 0

    Ground Truth ATE:
        ATE_t = E[Y^{factual}_t - Y^{cf}_t]                     (Eq 11)
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
    get_mediator_generator,
)
from .causal_engine import CausalEngine, CausalConfig
from .counterfactual_regimes import (
    InterventionRegime,
    get_delta_fn,
)


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
    1. Samples a causal structure (DIRECT/CHAIN/CONFOUNDER)
       either per-simulation or per-user
    2. Generates confounders via the SCM (category correlations)
    3. Generates treatment respecting the structure
       (AR(1) for DIRECT/CHAIN, AR(1)+confounder for CONFOUNDER)
    4. Generates mediator X for CHAIN (T→X) and CONFOUNDER (X AR(1))
    5. Generates outcome trajectories via structure-aware equations
    6. Supports do-operator counterfactuals (pointwise/continuous/gradual)

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

        # Storage for mediator/confounder trajectories (used by CHAIN and CONFOUNDER)
        self._mediator_matrix: Optional[np.ndarray] = None

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

    @property
    def mediator_matrix(self) -> Optional[np.ndarray]:
        """The mediator/confounder trajectory matrix, if generated.

        For CHAIN: Xt = βTX·f(Tt) + εX,t (mediator from treatment).
        For CONFOUNDER: Xt = ΦX·Xt−1 + εX,t (confounder AR(1)).

        Shape: (n_users, n_days, n_categories) or None if not yet generated.
        """
        return self._mediator_matrix

    # ------------------------------------------------------------------
    # Factual trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectories(self) -> np.ndarray:
        """Generate factual trajectories under the aleatory structures.

        For each user, applies the structure-specific equations:
        - DIRECT:     Tt AR(1) → Yt = ΦY·Yt−1 + βTY·f(Tt) + ε
        - CHAIN:      Tt AR(1) → Xt = βTX·f(Tt) + ε → Yt = ΦY·Yt−1 + βXY·f(Xt) + ε
        - CONFOUNDER: Xt AR(1) → Tt = ΦT·Tt−1 + βXT·f(Xt) + ε → Yt = ...

        Returns:
            Array of shape (n_users, n_days, n_categories).
        """
        n_users = self._engine.n_users
        n_days = self._engine.config.n_days
        n_cats = self._engine.config.n_categories

        # Initialize outcome matrix
        interest_matrix = np.zeros((n_users, n_days, n_cats))
        interest_matrix[:, 0, :] = self._engine.confounders.copy()

        # Initialize mediator/confounder matrix (used by CHAIN and CONFOUNDER)
        self._mediator_matrix = np.zeros((n_users, n_days, n_cats))
        self._mediator_matrix[:, 0, :] = self._engine.confounders.copy()

        # Initialize treatment matrix with AR(1) base values
        treatment_matrix = np.zeros((n_users, n_days))

        for day in range(1, n_days):
            for user_idx in range(n_users):
                structure, params = self._user_structures[user_idx]
                prev_interest = interest_matrix[user_idx, day - 1, :]

                # ----------------------------------------------------------
                # Step 1: Generate/update mediator/confounder Xt
                # ----------------------------------------------------------
                if structure == CausalStructure.CHAIN:
                    # Xt = βTX·f(Tt) + εX,t
                    # Mediator depends on treatment — compute after treatment
                    # For now, use previous day's treatment to compute mediator
                    prev_treatment = treatment_matrix[user_idx, day - 1]
                    mediator_gen = get_mediator_generator(structure)
                    self._mediator_matrix[user_idx, day, :] = mediator_gen(
                        treatment=prev_treatment,
                        params=params,
                        n_categories=n_cats,
                    )

                elif structure == CausalStructure.CONFOUNDER:
                    # Xt = ΦX·Xt−1 + εX,t — confounder AR(1)
                    prev_X = self._mediator_matrix[user_idx, day - 1, :]
                    phi_X = params.get("persistence_factor_X", 0.7)
                    mediator_gen = get_mediator_generator(structure)
                    noise = mediator_gen(
                        treatment=0.0,  # not used for CONFOUNDER
                        params=params,
                        n_categories=n_cats,
                    )
                    self._mediator_matrix[user_idx, day, :] = np.clip(
                        phi_X * prev_X + noise, 0, 1
                    )

                else:  # DIRECT — no mediator, copy previous confounder
                    self._mediator_matrix[user_idx, day, :] = self._mediator_matrix[
                        user_idx, day - 1, :
                    ]

                # ----------------------------------------------------------
                # Step 2: Generate treatment Tt
                # ----------------------------------------------------------
                base_exposure = self._users[user_idx].persona.ad_click_propensity * 0.5

                if structure == CausalStructure.CONFOUNDER:
                    # Tt = ΦT·Tt−1 + βXT·f(Xt) + εT,t
                    prev_T = treatment_matrix[user_idx, day - 1]
                    phi_T = params.get("persistence_factor_T", 0.7)
                    confounder_mean = float(
                        self._mediator_matrix[user_idx, day, :].mean()
                    )
                    generator = get_treatment_generator(structure)
                    noise_component = generator(
                        base_exposure, confounder_mean, params, self._rng
                    )
                    # noise_component = base_exposure + βXT·f(Xt) + ε
                    # We need to add the AR(1) part
                    treatment_matrix[user_idx, day] = float(
                        np.clip(
                            phi_T * prev_T
                            + (
                                noise_component - base_exposure
                            ),  # just the βXT·f(Xt) + ε part
                            0,
                            1,
                        )
                    )

                else:  # DIRECT or CHAIN
                    # Tt = ΦT·Tt−1 + εT,t
                    prev_T = treatment_matrix[user_idx, day - 1]
                    phi_T = params.get("persistence_factor_T", 0.7)
                    generator = get_treatment_generator(structure)
                    noise_component = generator(base_exposure, 0.0, params, self._rng)
                    # noise_component = base_exposure + ε
                    # We need: Tt = ΦT·Tt−1 + ε
                    treatment_matrix[user_idx, day] = float(
                        np.clip(
                            phi_T * prev_T
                            + (noise_component - base_exposure),  # just the ε part
                            0,
                            1,
                        )
                    )

                # ----------------------------------------------------------
                # Step 3: Compute outcome Yt
                # ----------------------------------------------------------
                treatment = treatment_matrix[user_idx, day]

                if structure == CausalStructure.CHAIN:
                    # Yt = ΦY·Yt−1 + βXY·f(Xt) + εY,t
                    # Pass mediator Xt as the 'confounder' argument
                    confounder_for_outcome = self._mediator_matrix[user_idx, day, :]
                else:
                    # DIRECT: confounder not used
                    # CONFOUNDER: confounder Xt is used
                    confounder_for_outcome = self._mediator_matrix[user_idx, day, :]

                equation = get_outcome_equation(structure)
                interest_matrix[user_idx, day, :] = equation(
                    prev_interest=prev_interest,
                    confounder=confounder_for_outcome,
                    treatment=treatment,
                    treatment_effect=self._engine.treatment_effects[user_idx],
                    params=params,
                    persistence_factor=self._engine.config.persistence_factor,
                    mean_reversion_strength=self._engine.config.mean_reversion_strength,
                )

        # Store treatment matrix for counterfactual use
        self._treatment_matrix = treatment_matrix

        return interest_matrix

    def _generate_treatment_for_user(
        self,
        user_idx: int,
        day: int,
        structure: CausalStructure,
        params: Dict[str, float],
    ) -> float:
        """Generate a single treatment value respecting the causal structure.

        This is a simplified version used by counterfactual generation.
        For factual generation, the full AR(1) recursion is used in
        generate_trajectories().

        Args:
            user_idx: User index.
            day: Day index.
            structure: The causal structure for this user.
            params: Structure-specific parameters.

        Returns:
            Treatment value in [0, 1].
        """
        base_exposure = self._users[user_idx].persona.ad_click_propensity * 0.5

        if structure == CausalStructure.CONFOUNDER:
            confounder_mean = float(
                self._mediator_matrix[user_idx, day, :].mean()
                if self._mediator_matrix is not None
                else self._engine.confounders[user_idx].mean()
            )
        else:
            confounder_mean = 0.0

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

        The do-operator blocks T←X paths (in CONFOUNDER) and T←... paths
        but leaves the X→Y path open — correct Pearlian causality.

        For CHAIN: do(T=t) blocks T→X→Y path entirely (mediator depends on T).
        For CONFOUNDER: do(T=t) blocks T←X but X→Y path remains open.

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

        # Generate trajectories with counterfactual treatment
        interest_matrix = np.zeros((n_users, n_days, n_cats))
        interest_matrix[:, 0, :] = self._engine.confounders.copy()

        # Recompute mediator under counterfactual treatment for CHAIN
        cf_mediator = np.zeros((n_users, n_days, n_cats))
        cf_mediator[:, 0, :] = self._engine.confounders.copy()

        for day in range(1, n_days):
            for user_idx in range(n_users):
                structure, params = self._user_structures[user_idx]
                prev = interest_matrix[user_idx, day - 1, :]
                treatment = cf_treatment[user_idx, day]

                # Recompute mediator under counterfactual treatment
                if structure == CausalStructure.CHAIN:
                    # Under do(T=t), mediator Xt = βTX·f(t) + εX,t
                    mediator_gen = get_mediator_generator(structure)
                    cf_mediator[user_idx, day, :] = mediator_gen(
                        treatment=treatment,
                        params=params,
                        n_categories=n_cats,
                    )
                    confounder_for_outcome = cf_mediator[user_idx, day, :]

                elif structure == CausalStructure.CONFOUNDER:
                    # Under do(T=t), X→Y path remains open.
                    # Xt still evolves as AR(1) independent of T.
                    prev_X = cf_mediator[user_idx, day - 1, :]
                    phi_X = params.get("persistence_factor_X", 0.7)
                    mediator_gen = get_mediator_generator(structure)
                    noise = mediator_gen(
                        treatment=0.0,
                        params=params,
                        n_categories=n_cats,
                    )
                    cf_mediator[user_idx, day, :] = np.clip(
                        phi_X * prev_X + noise, 0, 1
                    )
                    confounder_for_outcome = cf_mediator[user_idx, day, :]

                else:  # DIRECT
                    confounder_for_outcome = cf_mediator[user_idx, day - 1, :]

                equation = get_outcome_equation(structure)
                interest_matrix[user_idx, day, :] = equation(
                    prev_interest=prev,
                    confounder=confounder_for_outcome,
                    treatment=treatment,
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
        cf_treatment = self._treatment_matrix.copy()

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
    # Structural counterfactual generation (CSDG framework)
    # ------------------------------------------------------------------

    def generate_counterfactual_structural(
        self,
        regime: InterventionRegime,
        alpha: float,
        tint: int,
        k: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate counterfactual trajectories using structural equations.

        Implements the CSDG counterfactual generation from first principles.
        The counterfactual treatment T^{cf}_t is generated by injecting the
        δ_t intervention regime directly into the structural AR(1) equation,
        rather than post-hoc modifying the factual treatment matrix.

        T^{cf}_t = Φ_T · T^{cf}_{t-1} + ε_{T,t} + δ_t          (Eq 6)

        Y^{cf}_t = Φ_Y · Y^{cf}_{t-1} + β_{TY} · f(T^{cf}_t) + ε_{Y,t}  (Eq 7)

        By holding the fundamental structural parameters (Φ_T, Φ_Y, β_TY)
        constant while altering the treatment trajectory via δ_t, this
        provides a mathematically rigorous Ground Truth for ATE estimation.

        Args:
            regime: The intervention regime (PUNCTUAL, CONTINUOUS, GRADUAL).
            alpha: Intervention intensity. Interpretation depends on regime:
                - PUNCTUAL:   magnitude of the one-time shock at t=tint
                - CONTINUOUS: constant intensity applied from tint onward
                - GRADUAL:    maximum intensity at end of ramp
            tint: Intervention time step. The intervention "occurs" at this
                time point. Before tint, δ_t = 0 for all regimes.
            k: Scaling factor for GRADUAL regime only.
                δ_t = α · (t - tint) / k. Larger k = slower ramp.
                Default: 1.0 (full intensity reached after 1 step).

        Returns:
            Tuple of (T_cf, Y_cf):
                T_cf: ndarray of shape (n_users, n_days)
                    Counterfactual treatment trajectory.
                Y_cf: ndarray of shape (n_users, n_days, n_categories)
                    Counterfactual outcome (interest) trajectory.

        Raises:
            ValueError: If tint is out of bounds or regime is unknown.

        Examples:
            >>> # Punctual: one-time shock of 0.5 at day 14
            >>> T_cf, Y_cf = machine.generate_counterfactual_structural(
            ...     regime=InterventionRegime.PUNCTUAL,
            ...     alpha=0.5,
            ...     tint=14,
            ... )

            >>> # Continuous: sustained intensity of 0.3 from day 7 onward
            >>> T_cf, Y_cf = machine.generate_counterfactual_structural(
            ...     regime=InterventionRegime.CONTINUOUS,
            ...     alpha=0.3,
            ...     tint=7,
            ... )

            >>> # Gradual: phased ramp reaching 0.8 over 10 steps from day 14
            >>> T_cf, Y_cf = machine.generate_counterfactual_structural(
            ...     regime=InterventionRegime.GRADUAL,
            ...     alpha=0.8,
            ...     tint=14,
            ...     k=10.0,
            ... )
        """
        n_users = self._engine.n_users
        n_days = self._engine.config.n_days
        n_cats = self._engine.config.n_categories

        # Validate tint
        if tint < 0 or tint >= n_days:
            raise ValueError(f"tint must be in [0, {n_days - 1}], got {tint}")

        # Resolve the δ_t generator for the requested regime
        delta_fn = get_delta_fn(regime)

        # ------------------------------------------------------------------
        # Step 1: Generate counterfactual treatment T^{cf}_t (Eq 6)
        # ------------------------------------------------------------------
        T_cf = np.zeros((n_users, n_days))

        for day in range(1, n_days):
            for user_idx in range(n_users):
                structure, params = self._user_structures[user_idx]
                phi_T = params.get("persistence_factor_T", 0.7)
                noise_std_T = params.get("noise_std_T", 0.05)

                # AR(1) component: Φ_T · T^{cf}_{t-1}
                ar = phi_T * T_cf[user_idx, day - 1]

                # Fresh noise: ε_{T,t}
                eps = self._rng.normal(0, noise_std_T)

                # Intervention delta: δ_t
                delta = delta_fn(
                    t=day,
                    tint=tint,
                    alpha=alpha,
                    k=k,
                    n_days=n_days,
                )

                # Structural equation (Eq 6)
                T_cf[user_idx, day] = float(np.clip(ar + eps + delta, 0, 1))

        # ------------------------------------------------------------------
        # Step 2: Generate counterfactual outcome Y^{cf}_t (Eq 7)
        # ------------------------------------------------------------------
        Y_cf = np.zeros((n_users, n_days, n_cats))
        Y_cf[:, 0, :] = self._engine.confounders.copy()

        for day in range(1, n_days):
            for user_idx in range(n_users):
                structure, params = self._user_structures[user_idx]
                prev = Y_cf[user_idx, day - 1, :]
                treatment = T_cf[user_idx, day]

                # Use the structure-specific outcome equation
                # For DIRECT:   Yt = ΦY·Yt−1 + βTY·f(Tt) + εY,t
                # For CHAIN:    Yt = ΦY·Yt−1 + βXY·f(Xt) + εY,t
                # For CONFOUNDER: Yt = ΦY·Yt−1 + βXY·f(Xt) + βTY·g(Tt) + εY,t
                equation = get_outcome_equation(structure)

                # Determine the confounder/mediator to pass
                if structure == CausalStructure.CHAIN:
                    # For CHAIN, the mediator Xt depends on T^{cf}_t
                    # Recompute Xt = βTX·f(T^{cf}_t) + εX,t
                    mediator_gen = get_mediator_generator(structure)
                    confounder_for_outcome = mediator_gen(
                        treatment=treatment,
                        params=params,
                        n_categories=n_cats,
                    )
                elif structure == CausalStructure.CONFOUNDER:
                    # For CONFOUNDER, Xt evolves as AR(1) independent of T
                    # Use the factual mediator trajectory (X is not intervened on)
                    confounder_for_outcome = (
                        self._mediator_matrix[user_idx, day, :]
                        if self._mediator_matrix is not None
                        else self._engine.confounders[user_idx]
                    )
                else:  # DIRECT
                    confounder_for_outcome = self._engine.confounders[user_idx]

                Y_cf[user_idx, day, :] = equation(
                    prev_interest=prev,
                    confounder=confounder_for_outcome,
                    treatment=treatment,
                    treatment_effect=self._engine.treatment_effects[user_idx],
                    params=params,
                    persistence_factor=self._engine.config.persistence_factor,
                    mean_reversion_strength=self._engine.config.mean_reversion_strength,
                )

        return T_cf, Y_cf

    # ------------------------------------------------------------------
    # Ground Truth ATE computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ground_truth_ate(
        factual_Y: np.ndarray,
        counterfactual_Y: np.ndarray,
    ) -> np.ndarray:
        """Compute the Ground Truth Average Treatment Effect over time.

        ATE_t = E[Y^{factual}_t - Y^{cf}_t]                     (Eq 11)

        The expectation is taken over both users and categories, yielding
        a per-time-step ATE that can be used as ground truth for evaluating
        causal inference methods.

        Args:
            factual_Y: Factual outcome trajectories.
                Shape: (n_users, n_days, n_categories).
            counterfactual_Y: Counterfactual outcome trajectories.
                Shape: (n_users, n_days, n_categories).

        Returns:
            ATE_t: ndarray of shape (n_days,)
                Per-day average treatment effect. Positive values mean
                the factual outcome was higher than counterfactual
                (i.e., the treatment had a positive effect).

        Examples:
            >>> T_cf, Y_cf = machine.generate_counterfactual_structural(
            ...     regime=InterventionRegime.PUNCTUAL,
            ...     alpha=0.5,
            ...     tint=14,
            ... )
            >>> ate = AleatoryCausalMachine.compute_ground_truth_ate(
            ...     factual_Y=factual_interest,
            ...     counterfactual_Y=Y_cf,
            ... )
            >>> # ATE should be ~0 before tint, positive after tint
            >>> assert ate.shape == (n_days,)
        """
        # Average over categories → (n_users, n_days)
        factual_avg = factual_Y.mean(axis=2)
        cf_avg = counterfactual_Y.mean(axis=2)

        # Difference then average over users → (n_days,)
        ate = (factual_avg - cf_avg).mean(axis=0)

        return ate

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
