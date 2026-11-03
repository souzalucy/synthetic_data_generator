"""Causal inference engine for interest trajectory generation.

Uses a Structural Causal Model (SCM) for principled confounder generation
and supports do-operator interventions for counterfactual scenarios.

GPU acceleration is used automatically for the trajectory generation loop
when PyTorch is available and the number of users exceeds a threshold.
"""

import csv
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Union
import numpy as np
from .user_agent import User
from .scm_generator import SCMConfounderGenerator, SCMConfig
from ecosystem_sim.core.device_utils import detect_device, get_torch_device


# Threshold for GPU-accelerated trajectory generation
GPU_TRAJECTORY_MIN_USERS = 500


@dataclass
class CausalConfig:
    """Configuration for causal inference."""

    n_categories: int = 10
    n_days: int = 365
    treatment_effect_range: Tuple[float, float] = (0.1, 0.3)
    persistence_factor: float = 0.7
    mean_reversion_strength: float = 0.1
    noise_std: float = 0.05
    random_seed: Optional[int] = None
    use_scm: bool = True  # Use SCM-based confounder generation


class CausalEngine:
    """Generates causal trajectories for user interests.

    Uses an SCM for confounder generation and supports do-operator
    interventions for counterfactual scenario generation.
    """

    def __init__(
        self, config: CausalConfig, users: List[User], priors: Optional[dict] = None
    ):
        """Initialize causal engine.

        Args:
            config: Causal configuration
            users: List of users
            priors: Optional compiled dataset priors
        """
        self.config = config
        self.users = users
        self.n_users = len(users)
        self.priors = priors

        self._device_type = detect_device()
        self._torch_device = get_torch_device()

        self._apply_priors()

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Initialize SCM confounder generator
        scm_config = SCMConfig(
            n_categories=config.n_categories,
            use_scm=config.use_scm,
        )
        self.scm_generator = SCMConfounderGenerator(scm_config, priors=priors)

        # Generate confounders using SCM
        self._generate_confounders()
        self._generate_treatment_trajectories()
        self._generate_causal_parameters()

    def _apply_priors(self):
        """Apply dataset priors to causal configuration."""
        from .prior_loader import get_prior

        p = self.priors

        pf = get_prior(
            p, "layer_3_intelligence.causal_engine.persistence_factor.fit_params", None
        )
        if pf:
            self.config.persistence_factor = float(
                np.clip(
                    np.random.beta(pf.get("alpha", 6.8), pf.get("beta", 3.2)), 0.3, 0.95
                )
            )

        noise = get_prior(p, "layer_3_intelligence.causal_engine.noise_std", None)
        if noise is not None:
            self.config.noise_std = float(noise)

    def _generate_confounders(self):
        """Generate confounders using the SCM.

        Delegates to SCMConfounderGenerator which builds a structured
        covariance matrix from dataset priors.
        """
        self.confounders = self.scm_generator.generate_confounders(self.users)

    def _generate_treatment_trajectories(self):
        """Generate ad exposure trajectories (treatment)."""
        self.treatment_matrix = np.zeros((self.n_users, self.config.n_days))

        for i, user in enumerate(self.users):
            # Treatment intensity based on persona
            base_exposure = user.persona.ad_click_propensity * 0.5

            for day in range(self.config.n_days):
                # Simulate campaign bursts with random timing
                if np.random.random() < 0.1:  # 10% chance of campaign on any day
                    burst_intensity = np.random.beta(2, 5)  # mostly low, some high
                else:
                    burst_intensity = np.random.beta(1, 3) * 0.3  # baseline

                self.treatment_matrix[i, day] = np.clip(
                    base_exposure + burst_intensity, 0, 1
                )

    def _generate_causal_parameters(self):
        """Generate treatment effect heterogeneity."""
        from .prior_loader import get_prior

        te_params = get_prior(
            self.priors,
            "layer_3_intelligence.causal_engine.treatment_effect_distribution.fit_params",
            None,
        )
        if te_params and te_params.get("shape") and te_params.get("scale"):
            self.treatment_effects = np.random.gamma(
                te_params["shape"], te_params["scale"], size=self.n_users
            )
            self.treatment_effects = np.clip(self.treatment_effects, 0, 1)
        else:
            self.treatment_effects = np.random.uniform(
                self.config.treatment_effect_range[0],
                self.config.treatment_effect_range[1],
                size=self.n_users,
            )

    # ------------------------------------------------------------------
    # Factual trajectory generation
    # ------------------------------------------------------------------

    def generate_causal_trajectories(self) -> np.ndarray:
        """Generate interest trajectories using AR(1) + treatment.

        Uses GPU-accelerated vectorized operations when PyTorch is available
        and the number of users exceeds ``GPU_TRAJECTORY_MIN_USERS``.

        Returns:
            Array of shape (n_users, n_days, n_categories)
        """
        interest_matrix = np.zeros(
            (self.n_users, self.config.n_days, self.config.n_categories)
        )

        # Initialize day 0 with base affinity from SCM
        interest_matrix[:, 0, :] = self.confounders.copy()

        # GPU-accelerated path for large simulations
        if self.n_users >= GPU_TRAJECTORY_MIN_USERS and self._device_type != "cpu":
            return self._gpu_generate_trajectories(interest_matrix)

        # CPU path (original nested loop)
        for day in range(1, self.config.n_days):
            for user_idx in range(self.n_users):
                # AR(1) component: persistence from previous day
                prev_interest = interest_matrix[user_idx, day - 1, :]
                ar_component = self.config.persistence_factor * prev_interest

                # Treatment effect: causal impact
                treatment = self.treatment_matrix[user_idx, day]
                treatment_component = self.treatment_effects[user_idx] * treatment

                # Mean reversion: pull towards base affinity
                confounder = self.confounders[user_idx, :]
                mean_reversion = self.config.mean_reversion_strength * (
                    confounder - prev_interest
                )

                # Noise
                noise = np.random.normal(
                    0, self.config.noise_std, self.config.n_categories
                )

                # Combine components
                new_interest = (
                    ar_component + treatment_component + mean_reversion + noise
                )

                # Enforce bounds
                interest_matrix[user_idx, day, :] = np.clip(new_interest, 0, 1)

        return interest_matrix

    def _gpu_generate_trajectories(self, interest_matrix: np.ndarray) -> np.ndarray:
        """Generate trajectories using GPU-vectorized operations.

        The AR(1) + treatment + mean reversion loop is replaced by
        batched tensor operations on the GPU.

        Args:
            interest_matrix: Pre-allocated array of shape
                (n_users, n_days, n_categories) with day 0 initialized.

        Returns:
            Completed interest matrix.
        """
        try:
            import torch

            n_days = self.config.n_days
            n_cats = self.config.n_categories
            pf = self.config.persistence_factor
            mr = self.config.mean_reversion_strength
            noise_std = self.config.noise_std

            # Move all arrays to GPU once
            interest_t = torch.tensor(
                interest_matrix, dtype=torch.float32, device=self._torch_device
            )
            confounders_t = torch.tensor(
                self.confounders, dtype=torch.float32, device=self._torch_device
            )
            treatment_t = torch.tensor(
                self.treatment_matrix, dtype=torch.float32, device=self._torch_device
            )
            treatment_effects_t = torch.tensor(
                self.treatment_effects,
                dtype=torch.float32,
                device=self._torch_device,
            ).reshape(-1, 1)  # (n_users, 1) for broadcasting

            for day in range(1, n_days):
                prev = interest_t[:, day - 1, :]  # (n_users, n_cats)

                # Vectorized AR(1) + treatment + mean reversion
                ar = pf * prev
                treatment_effect = treatment_effects_t * treatment_t[:, day : day + 1]
                reversion = mr * (confounders_t - prev)
                noise = (
                    torch.randn(self.n_users, n_cats, device=self._torch_device)
                    * noise_std
                )

                new_interest = ar + treatment_effect + reversion + noise
                interest_t[:, day, :] = torch.clamp(new_interest, 0, 1)

            print(
                f"  [CAUSAL] Trajectories generated on {self._device_type.upper()} "
                f"({self.n_users} users x {n_days} days x {n_cats} categories)"
            )
            return interest_t.cpu().numpy()

        except Exception as exc:
            print(
                f"  [CAUSAL] GPU trajectory generation failed ({exc}), "
                f"falling back to CPU"
            )

        # CPU fallback
        for day in range(1, self.config.n_days):
            for user_idx in range(self.n_users):
                prev_interest = interest_matrix[user_idx, day - 1, :]
                ar_component = self.config.persistence_factor * prev_interest
                treatment = self.treatment_matrix[user_idx, day]
                treatment_component = self.treatment_effects[user_idx] * treatment
                confounder = self.confounders[user_idx, :]
                mean_reversion = self.config.mean_reversion_strength * (
                    confounder - prev_interest
                )
                noise = np.random.normal(
                    0, self.config.noise_std, self.config.n_categories
                )
                new_interest = (
                    ar_component + treatment_component + mean_reversion + noise
                )
                interest_matrix[user_idx, day, :] = np.clip(new_interest, 0, 1)

        return interest_matrix

    # ------------------------------------------------------------------
    # Counterfactual trajectory generation (do-operator)
    # ------------------------------------------------------------------

    def generate_counterfactual_trajectories(
        self,
        treatment_intervention: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        confounder_intervention: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """Generate counterfactual trajectories under do-operator interventions.

        Supports two types of interventions:
        1. treatment_intervention: modifies the treatment matrix (do(T = f(T)))
        2. confounder_intervention: modifies the confounders (do(Z = f(Z)))

        Args:
            treatment_intervention: Callable(treatment_matrix) -> modified_treatment_matrix
            confounder_intervention: Callable(confounders) -> modified_confounders

        Returns:
            Array of shape (n_users, n_days, n_categories) with counterfactual interests
        """
        # Apply confounder intervention if provided
        if confounder_intervention is not None:
            counterfactual_confounders = confounder_intervention(
                self.confounders.copy()
            )
        else:
            counterfactual_confounders = self.confounders.copy()

        # Apply treatment intervention if provided
        if treatment_intervention is not None:
            counterfactual_treatment = treatment_intervention(
                self.treatment_matrix.copy()
            )
        else:
            counterfactual_treatment = self.treatment_matrix.copy()

        # Generate counterfactual trajectories
        interest_matrix = np.zeros(
            (self.n_users, self.config.n_days, self.config.n_categories)
        )

        interest_matrix[:, 0, :] = counterfactual_confounders.copy()

        # GPU-accelerated path for large simulations
        if self.n_users >= GPU_TRAJECTORY_MIN_USERS and self._device_type != "cpu":
            return self._gpu_generate_counterfactual(
                interest_matrix, counterfactual_confounders, counterfactual_treatment
            )

        # CPU path
        for day in range(1, self.config.n_days):
            for user_idx in range(self.n_users):
                prev_interest = interest_matrix[user_idx, day - 1, :]
                ar_component = self.config.persistence_factor * prev_interest

                treatment = counterfactual_treatment[user_idx, day]
                treatment_component = self.treatment_effects[user_idx] * treatment

                confounder = counterfactual_confounders[user_idx, :]
                mean_reversion = self.config.mean_reversion_strength * (
                    confounder - prev_interest
                )

                noise = np.random.normal(
                    0, self.config.noise_std, self.config.n_categories
                )

                new_interest = (
                    ar_component + treatment_component + mean_reversion + noise
                )
                interest_matrix[user_idx, day, :] = np.clip(new_interest, 0, 1)

        return interest_matrix

    def _gpu_generate_counterfactual(
        self,
        interest_matrix: np.ndarray,
        counterfactual_confounders: np.ndarray,
        counterfactual_treatment: np.ndarray,
    ) -> np.ndarray:
        """Generate counterfactual trajectories on GPU.

        Args:
            interest_matrix: Pre-allocated array with day 0 initialized.
            counterfactual_confounders: Modified confounders from do-operator.
            counterfactual_treatment: Modified treatment matrix from do-operator.

        Returns:
            Completed counterfactual interest matrix.
        """
        try:
            import torch

            n_days = self.config.n_days
            n_cats = self.config.n_categories
            pf = self.config.persistence_factor
            mr = self.config.mean_reversion_strength
            noise_std = self.config.noise_std

            interest_t = torch.tensor(
                interest_matrix, dtype=torch.float32, device=self._torch_device
            )
            confounders_t = torch.tensor(
                counterfactual_confounders,
                dtype=torch.float32,
                device=self._torch_device,
            )
            treatment_t = torch.tensor(
                counterfactual_treatment,
                dtype=torch.float32,
                device=self._torch_device,
            )
            treatment_effects_t = torch.tensor(
                self.treatment_effects,
                dtype=torch.float32,
                device=self._torch_device,
            ).reshape(-1, 1)

            for day in range(1, n_days):
                prev = interest_t[:, day - 1, :]
                ar = pf * prev
                treatment_effect = treatment_effects_t * treatment_t[:, day : day + 1]
                reversion = mr * (confounders_t - prev)
                noise = (
                    torch.randn(self.n_users, n_cats, device=self._torch_device)
                    * noise_std
                )

                new_interest = ar + treatment_effect + reversion + noise
                interest_t[:, day, :] = torch.clamp(new_interest, 0, 1)

            return interest_t.cpu().numpy()

        except Exception as exc:
            print(
                f"  [CAUSAL] GPU counterfactual generation failed ({exc}), "
                f"falling back to CPU"
            )

        # CPU fallback
        for day in range(1, self.config.n_days):
            for user_idx in range(self.n_users):
                prev_interest = interest_matrix[user_idx, day - 1, :]
                ar_component = self.config.persistence_factor * prev_interest
                treatment = counterfactual_treatment[user_idx, day]
                treatment_component = self.treatment_effects[user_idx] * treatment
                confounder = counterfactual_confounders[user_idx, :]
                mean_reversion = self.config.mean_reversion_strength * (
                    confounder - prev_interest
                )
                noise = np.random.normal(
                    0, self.config.noise_std, self.config.n_categories
                )
                new_interest = (
                    ar_component + treatment_component + mean_reversion + noise
                )
                interest_matrix[user_idx, day, :] = np.clip(new_interest, 0, 1)

        return interest_matrix

    def generate_no_treatment_counterfactual(self) -> np.ndarray:
        """Generate counterfactual trajectories with no treatment (do(T=0)).

        Returns:
            Counterfactual interest matrix of shape (n_users, n_days, n_categories)
        """
        return self.generate_counterfactual_trajectories(
            treatment_intervention=lambda t: np.zeros_like(t)
        )

    def generate_double_treatment_counterfactual(self) -> np.ndarray:
        """Generate counterfactual trajectories with doubled treatment (do(T=2T)).

        Returns:
            Counterfactual interest matrix
        """
        return self.generate_counterfactual_trajectories(
            treatment_intervention=lambda t: np.clip(t * 2, 0, 1)
        )

    def generate_optimal_treatment_counterfactual(
        self, threshold: float = 0.5
    ) -> np.ndarray:
        """Generate counterfactual with optimal targeted treatment.

        Only treats users whose base affinity is above threshold.

        Args:
            threshold: Base affinity threshold for treatment

        Returns:
            Counterfactual interest matrix
        """

        def targeted_treatment(treatment: np.ndarray) -> np.ndarray:
            cf = treatment.copy()
            for i, user in enumerate(self.users):
                base_affinity = self.confounders[i].mean()
                if base_affinity < threshold:
                    cf[i, :] = 0  # No treatment for low-affinity users
                else:
                    cf[i, :] = np.clip(cf[i, :] * 1.5, 0, 1)  # Boost for high-affinity
            return cf

        return self.generate_counterfactual_trajectories(
            treatment_intervention=targeted_treatment
        )

    def generate_confounder_intervention_counterfactual(
        self, shift: float = 0.2
    ) -> np.ndarray:
        """Generate counterfactual with shifted confounders (do(Z = Z + shift)).

        Simulates what would happen if all users had higher base affinity.

        Args:
            shift: Amount to shift confounders by

        Returns:
            Counterfactual interest matrix
        """
        return self.generate_counterfactual_trajectories(
            confounder_intervention=lambda z: np.clip(z + shift, 0, 1)
        )

    # ------------------------------------------------------------------
    # Time-localized counterfactual (intervention = marketing action)
    # ------------------------------------------------------------------

    def _validate_intervention_day(self, intervention_day: int) -> None:
        """Validate that intervention_day is within the simulation range.

        Args:
            intervention_day: Day index for the intervention

        Raises:
            ValueError: If intervention_day is out of bounds
        """
        if intervention_day < 0:
            raise ValueError(f"intervention_day must be >= 0, got {intervention_day}")
        if intervention_day >= self.config.n_days:
            raise ValueError(
                f"intervention_day ({intervention_day}) must be < "
                f"n_days ({self.config.n_days})"
            )

    def _compute_single_day(
        self,
        prev_interest: np.ndarray,
        treatment: float,
        treatment_effect: float,
        confounder: np.ndarray,
    ) -> np.ndarray:
        """Compute a single day's interest from AR(1) + treatment + reversion + noise.

        Args:
            prev_interest: Previous day interest vector (n_categories,)
            treatment: Treatment value for this day
            treatment_effect: User-specific treatment effect coefficient
            confounder: Base confounder vector (n_categories,)

        Returns:
            New interest vector (n_categories,)
        """
        ar_component = self.config.persistence_factor * prev_interest
        treatment_component = treatment_effect * treatment
        mean_reversion = self.config.mean_reversion_strength * (
            confounder - prev_interest
        )
        noise = np.random.normal(0, self.config.noise_std, prev_interest.shape)
        new_interest = ar_component + treatment_component + mean_reversion + noise
        return np.clip(new_interest, 0, 1)

    def generate_time_localized_counterfactual(
        self,
        factual_interest_matrix: np.ndarray,
        intervention_day: int,
        treatment_override: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        """Generate a counterfactual branching from the factual trajectory.

        The intervention occurs *at* intervention_day — it represents the
        marketing action itself happening at that point in time. Everything
        before intervention_day is kept identical to the factual trajectory.
        From intervention_day onward, the treatment is overridden and the
        trajectory is recomputed forward.

        Args:
            factual_interest_matrix: Factual trajectory of shape
                (n_users, n_days, n_categories) from generate_causal_trajectories()
            intervention_day: Day index at which the intervention (marketing
                action) occurs. Must satisfy 0 <= intervention_day < n_days.
            treatment_override: The alternative treatment value to apply from
                intervention_day onward. Can be a scalar (same value for all
                users) or an array of shape (n_users,) for per-user overrides.
                Default: 0.0 (i.e. do(T=0)).

        Returns:
            Counterfactual interest matrix of shape
            (n_users, n_days, n_categories), identical to factual up to
            intervention_day - 1, then recomputed from intervention_day forward.
        """
        self._validate_intervention_day(intervention_day)

        # Branch: copy factual trajectory up to intervention_day - 1
        counterfactual = factual_interest_matrix.copy()

        # Override treatment at intervention_day and beyond
        modified_treatment = self.treatment_matrix.copy()
        if isinstance(treatment_override, np.ndarray):
            modified_treatment[:, intervention_day:] = treatment_override[:, np.newaxis]
        else:
            modified_treatment[:, intervention_day:] = treatment_override

        # Recompute from intervention_day forward
        for day in range(intervention_day, self.config.n_days):
            if day == 0:
                continue
            for user_idx in range(self.n_users):
                prev_interest = counterfactual[user_idx, day - 1, :]
                treatment = modified_treatment[user_idx, day]
                treatment_effect = self.treatment_effects[user_idx]
                confounder = self.confounders[user_idx, :]
                counterfactual[user_idx, day, :] = self._compute_single_day(
                    prev_interest, treatment, treatment_effect, confounder
                )

        return counterfactual

    def generate_time_localized_no_treatment_counterfactual(
        self,
        factual_interest_matrix: np.ndarray,
        intervention_day: int,
    ) -> np.ndarray:
        """Generate counterfactual with no treatment from intervention_day onward.

        The intervention (do(T=0)) occurs at intervention_day, representing
        the marketing action being turned off from that point forward.

        Args:
            factual_interest_matrix: Factual trajectory matrix
            intervention_day: Day at which treatment is set to zero

        Returns:
            Counterfactual interest matrix
        """
        return self.generate_time_localized_counterfactual(
            factual_interest_matrix,
            intervention_day=intervention_day,
            treatment_override=0.0,
        )

    def generate_time_localized_double_treatment_counterfactual(
        self,
        factual_interest_matrix: np.ndarray,
        intervention_day: int,
    ) -> np.ndarray:
        """Generate counterfactual with doubled treatment from intervention_day onward.

        The intervention (do(T=2T)) occurs at intervention_day, representing
        a campaign intensity increase from that point forward.

        Args:
            factual_interest_matrix: Factual trajectory matrix
            intervention_day: Day at which treatment is doubled

        Returns:
            Counterfactual interest matrix
        """
        # Compute the override value per user as 2x their factual treatment
        # at intervention_day, clipped to [0, 1]
        override = np.clip(self.treatment_matrix[:, intervention_day] * 2.0, 0, 1)
        return self.generate_time_localized_counterfactual(
            factual_interest_matrix,
            intervention_day=intervention_day,
            treatment_override=override,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_interest_at_day(
        self, user_idx: int, day: int, interest_matrix: np.ndarray
    ) -> Dict[int, float]:
        """Get interest scores for a user at a specific day.

        Args:
            user_idx: User index
            day: Day index
            interest_matrix: Pre-computed interest matrix

        Returns:
            Dictionary mapping category_idx -> interest_score
        """
        scores = {}
        for cat_idx in range(self.config.n_categories):
            scores[cat_idx] = float(interest_matrix[user_idx, day, cat_idx])
        return scores

    def get_counterfactual_interest(
        self,
        user_idx: int,
        day: int,
        treatment_override: float,
        interest_matrix: np.ndarray,
    ) -> Dict[int, float]:
        """Get counterfactual interests if treatment were different.

        Args:
            user_idx: User index
            day: Day index
            treatment_override: Alternative treatment value
            interest_matrix: Pre-computed interest matrix

        Returns:
            Dictionary mapping category_idx -> counterfactual_interest_score
        """
        factual_treatment = self.treatment_matrix[user_idx, day]
        actual_interest = interest_matrix[user_idx, day, :].copy()

        treatment_diff = treatment_override - factual_treatment
        effect_adjustment = self.treatment_effects[user_idx] * treatment_diff

        counterfactual = np.clip(actual_interest - effect_adjustment, 0, 1)

        return {
            cat_idx: float(counterfactual[cat_idx])
            for cat_idx in range(self.config.n_categories)
        }

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def export_causal_table_with_counterfactual(
        self,
        filepath: str,
        interest_matrix: np.ndarray,
        counterfactual_matrix: np.ndarray,
        intervention_type: str = "no_treatment",
        intervention_value: str = "do(T=0)",
    ):
        """Export a wide-format CSV with both factual and counterfactual base affinities.

        Columns: user_id, persona, treatment_effect, intervention_type, intervention_value,
                 day_0_base_affinity_factual, ..., day_N_base_affinity_factual,
                 day_0_base_affinity_counterfactual, ..., day_N_base_affinity_counterfactual

        Args:
            filepath: Output CSV file path
            interest_matrix: Factual interest trajectory matrix (n_users, n_days, n_categories)
            counterfactual_matrix: Counterfactual interest trajectory matrix (n_users, n_days, n_categories)
            intervention_type: Label for the counterfactual intervention (e.g. 'no_treatment')
            intervention_value: Description of the intervention (e.g. 'do(T=0)')
        """
        n_days = self.config.n_days

        factual_affinity = interest_matrix.mean(axis=2)  # (n_users, n_days)
        counterfactual_affinity = counterfactual_matrix.mean(
            axis=2
        )  # (n_users, n_days)

        fieldnames = (
            [
                "user_id",
                "persona",
                "treatment_effect",
                "intervention_type",
                "intervention_value",
            ]
            + [f"day_{d}_base_affinity_factual" for d in range(n_days)]
            + [f"day_{d}_base_affinity_counterfactual" for d in range(n_days)]
        )

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, user in enumerate(self.users):
                row = {
                    "user_id": user.user_id,
                    "persona": user.persona.name,
                    "treatment_effect": float(self.treatment_effects[i]),
                    "intervention_type": intervention_type,
                    "intervention_value": intervention_value,
                }
                for d in range(n_days):
                    row[f"day_{d}_base_affinity_factual"] = float(
                        factual_affinity[i, d]
                    )
                    row[f"day_{d}_base_affinity_counterfactual"] = float(
                        counterfactual_affinity[i, d]
                    )
                writer.writerow(row)

    def export_ground_truth(self, filepath: str, interest_matrix: np.ndarray):
        """Export ground truth for validation.

        Args:
            filepath: Output file path
            interest_matrix: Interest trajectory matrix of shape
                (n_users, n_days, n_categories)
        """
        # Store the full 3-D interest matrix as a nested list so that
        # downstream consumers (e.g. plot_interest_heatmap) can average
        # over any axis they need.
        interest_matrix_3d = interest_matrix.tolist()  # (n_users, n_days, n_categories)

        ground_truth = {
            "config": {
                "n_users": self.n_users,
                "n_days": self.config.n_days,
                "n_categories": self.config.n_categories,
                "persistence_factor": self.config.persistence_factor,
                "treatment_effect_mean": float(self.treatment_effects.mean()),
                "treatment_effect_std": float(self.treatment_effects.std()),
                "use_scm": self.config.use_scm,
            },
            "users": [
                {
                    "user_id": user.user_id,
                    "persona": user.persona.name,
                    "treatment_effect": float(self.treatment_effects[i]),
                    "base_affinity": float(self.confounders[i].mean()),
                }
                for i, user in enumerate(self.users)
            ],
            "statistics": {
                "mean_interest": float(interest_matrix.mean()),
                "std_interest": float(interest_matrix.std()),
                "mean_treatment": float(self.treatment_matrix.mean()),
                "std_treatment": float(self.treatment_matrix.std()),
            },
            "interest_matrix": interest_matrix_3d,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

    def export_causal_table(self, filepath: str, interest_matrix: np.ndarray):
        """Export a wide-format causal ground truth CSV with per-day base_affinity.

        Columns: user_id, persona, treatment_effect, day_0_base_affinity, ..., day_N_base_affinity

        Args:
            filepath: Output CSV file path
            interest_matrix: Interest trajectory matrix of shape (n_users, n_days, n_categories)
        """
        n_days = self.config.n_days
        n_cats = self.config.n_categories

        daily_affinity = interest_matrix.mean(axis=2)  # (n_users, n_days)

        fieldnames = ["user_id", "persona", "treatment_effect"] + [
            f"day_{d}_base_affinity" for d in range(n_days)
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, user in enumerate(self.users):
                row = {
                    "user_id": user.user_id,
                    "persona": user.persona.name,
                    "treatment_effect": float(self.treatment_effects[i]),
                }
                for d in range(n_days):
                    row[f"day_{d}_base_affinity"] = float(daily_affinity[i, d])
                writer.writerow(row)
