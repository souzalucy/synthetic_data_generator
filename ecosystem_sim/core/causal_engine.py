"""Causal inference engine for interest trajectory generation."""

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from .user_agent import User


@dataclass
class CausalConfig:
    """Configuration for causal inference."""
    n_categories: int = 10
    n_days: int = 365
    treatment_effect_range: Tuple[float, float] = (0.1, 0.3)
    persistence_factor: float = 0.7
    mean_reversion_strength: float = 0.1
    noise_std: float = 0.05
    random_seed: int = None


class CausalEngine:
    """Generates causal trajectories for user interests."""
    
    def __init__(self, config: CausalConfig, users: List[User]):
        """Initialize causal engine.
        
        Args:
            config: Causal configuration
            users: List of users
        """
        self.config = config
        self.users = users
        self.n_users = len(users)
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Generate causal parameters
        self._generate_confounders()
        self._generate_treatment_trajectories()
        self._generate_causal_parameters()
    
    def _generate_confounders(self):
        """Map user attributes to numerical confounder values."""
        self.confounders = np.zeros((self.n_users, self.config.n_categories))
        
        for i, user in enumerate(self.users):
            for j in range(self.config.n_categories):
                # Base affinity determined by persona and latent interests
                base_affinity = (
                    user.latent_tech_interest * 0.5 +
                    user.latent_finance_interest * 0.3 +
                    user.persona.income_level.value * 0.2
                )
                
                # Add category-specific noise
                affinity = base_affinity + np.random.normal(0, 0.1)
                self.confounders[i, j] = np.clip(affinity, 0, 1)
    
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
                
                self.treatment_matrix[i, day] = np.clip(base_exposure + burst_intensity, 0, 1)
    
    def _generate_causal_parameters(self):
        """Generate treatment effect heterogeneity."""
        # Each user has a treatment effect (heterogeneous)
        self.treatment_effects = np.random.uniform(
            self.config.treatment_effect_range[0],
            self.config.treatment_effect_range[1],
            size=self.n_users
        )
    
    def generate_causal_trajectories(self) -> np.ndarray:
        """Generate interest trajectories using AR(1) + treatment.
        
        Returns:
            Array of shape (n_users, n_days, n_categories)
        """
        interest_matrix = np.zeros(
            (self.n_users, self.config.n_days, self.config.n_categories)
        )
        
        # Initialize day 0 with base affinity
        interest_matrix[:, 0, :] = self.confounders.copy()
        
        # Generate trajectory for each day
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
                mean_reversion = (
                    self.config.mean_reversion_strength * 
                    (confounder - prev_interest)
                )
                
                # Noise
                noise = np.random.normal(0, self.config.noise_std, self.config.n_categories)
                
                # Combine components
                new_interest = (
                    ar_component +
                    treatment_component +
                    mean_reversion +
                    noise
                )
                
                # Enforce bounds
                interest_matrix[user_idx, day, :] = np.clip(new_interest, 0, 1)
        
        return interest_matrix
    
    def get_interest_at_day(self, user_idx: int, day: int, 
                            interest_matrix: np.ndarray) -> Dict[str, float]:
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
            scores[cat_idx] = interest_matrix[user_idx, day, cat_idx]
        return scores
    
    def get_counterfactual_interest(self, user_idx: int, day: int,
                                    treatment_override: float,
                                    interest_matrix: np.ndarray) -> Dict[str, float]:
        """Get counterfactual interests if treatment were different.
        
        Args:
            user_idx: User index
            day: Day index
            treatment_override: Alternative treatment value
            interest_matrix: Pre-computed interest matrix
            
        Returns:
            Dictionary mapping category_idx -> counterfactual_interest_score
        """
        # Get factual treatment
        factual_treatment = self.treatment_matrix[user_idx, day]
        
        # Get actual interest
        actual_interest = interest_matrix[user_idx, day, :].copy()
        
        # Calculate treatment effect difference
        treatment_diff = treatment_override - factual_treatment
        effect_adjustment = self.treatment_effects[user_idx] * treatment_diff
        
        # Adjust down (but don't go below 0)
        counterfactual = np.clip(actual_interest - effect_adjustment, 0, 1)
        
        return {cat_idx: counterfactual[cat_idx] 
                for cat_idx in range(self.config.n_categories)}
    
    def export_ground_truth(self, filepath: str, interest_matrix: np.ndarray):
        """Export ground truth for validation.
        
        Args:
            filepath: Output file path
            interest_matrix: Interest trajectory matrix
        """
        ground_truth = {
            "config": {
                "n_users": self.n_users,
                "n_days": self.config.n_days,
                "n_categories": self.config.n_categories,
                "persistence_factor": self.config.persistence_factor,
                "treatment_effect_mean": float(self.treatment_effects.mean()),
                "treatment_effect_std": float(self.treatment_effects.std()),
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
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(ground_truth, f, indent=2)
