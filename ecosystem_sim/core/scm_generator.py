"""Structural Causal Model (SCM) confounder generator.

Implements a formal SCM for generating correlated confounders (base affinities)
using a multivariate normal distribution with a covariance matrix built from
real dataset priors (Amazon co-purchasing, Wikispeedia transitions, etc.).

This replaces the independent-per-category confounder generation in CausalEngine
with a principled causal generative model.

GPU acceleration is used automatically when PyTorch is available and the
matrix/user sizes exceed configurable thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.linalg import cholesky as cpu_cholesky

from ecosystem_sim.core.device_utils import detect_device, get_torch_device


# Thresholds for GPU acceleration
GPU_CHOLESKY_MIN_SIZE = 50  # n_categories >= this → GPU Cholesky
GPU_SAMPLING_MIN_USERS = 1000  # n_users >= this → GPU confounder sampling


@dataclass
class SCMConfig:
    """Configuration for the SCM confounder generator."""

    n_categories: int = 10
    base_diagonal: float = 0.5
    prior_correlation_scale: float = 0.3
    off_diagonal_regularizer: float = 0.1
    use_scm: bool = True


class SCMConfounderGenerator:
    """Generates confounders using a formal Structural Causal Model.

    The SCM is defined as:
        Z = μ + L @ ε,   ε ~ N(0, I)

    where:
    - μ is the mean vector derived from persona attributes
    - L is the Cholesky factor of the covariance matrix Σ = LLᵀ
    - Σ encodes causal relationships between categories from real data

    This enables principled do-operator interventions for counterfactual generation.

    GPU acceleration is transparently applied when:
    - PyTorch is installed (optional dependency ``[gpu]``)
    - The matrix size or user count exceeds internal thresholds
    """

    def __init__(self, config: SCMConfig, priors: Optional[Dict] = None):
        """Initialize SCM confounder generator.

        Args:
            config: SCM configuration
            priors: Optional compiled dataset priors for building Σ
        """
        self.config = config
        self.priors = priors
        self._device_type = detect_device()
        self._torch_device = get_torch_device()

        print(f"  [SCM] Device: {self._device_type.upper()} ({self._torch_device})")

        # Build the structured covariance matrix from priors
        self.Sigma = self._build_covariance_matrix()

        # Precompute Cholesky factor for efficient sampling
        self.L = self._compute_cholesky(self.Sigma)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_confounders(
        self, users: List, n_categories: Optional[int] = None
    ) -> np.ndarray:
        """Generate confounders using the SCM: Z = μ + L @ ε.

        Args:
            users: List of User objects with persona attributes
            n_categories: Override number of categories (default: from config)

        Returns:
            Array of shape (n_users, n_categories) with confounder values in [0, 1]
        """
        if not users:
            return np.zeros((0, self.config.n_categories))

        n_users = len(users)
        n_cats = n_categories or self.config.n_categories

        # Compute mean vector per user from persona attributes
        mu_matrix = self._compute_mean_matrix(users, n_cats)

        # Use GPU for large batches
        if n_users >= GPU_SAMPLING_MIN_USERS and self._device_type != "cpu":
            return self._gpu_sample_confounders(mu_matrix, n_cats)

        # CPU path
        epsilon = np.random.normal(0, 1, (n_users, n_cats))
        confounders = mu_matrix + epsilon @ self.L[:n_cats, :n_cats].T

        return np.clip(confounders, 0, 1)

    def generate_counterfactual_confounders(
        self,
        users: List,
        intervention_fn,
        n_categories: Optional[int] = None,
    ) -> np.ndarray:
        """Generate counterfactual confounders under a do-operator intervention.

        The intervention_fn receives the base mean vector for each user and
        returns a modified mean vector, representing a do-operator on the SCM.

        Args:
            users: List of User objects
            intervention_fn: Callable(mean_vector) -> modified_mean_vector
            n_categories: Override number of categories

        Returns:
            Array of shape (n_users, n_categories) with counterfactual values
        """
        if not users:
            return np.zeros((0, self.config.n_categories))

        n_users = len(users)
        n_cats = n_categories or self.config.n_categories

        # Compute base mean matrix
        mu_matrix = self._compute_mean_matrix(users, n_cats)

        # Apply do-operator intervention
        mu_counterfactual = np.array([intervention_fn(mu) for mu in mu_matrix])

        # Use GPU for large batches
        if n_users >= GPU_SAMPLING_MIN_USERS and self._device_type != "cpu":
            return self._gpu_sample_confounders(mu_counterfactual, n_cats)

        # CPU path
        epsilon = np.random.normal(0, 1, (n_users, n_cats))
        confounders = mu_counterfactual + epsilon @ self.L[:n_cats, :n_cats].T

        return np.clip(confounders, 0, 1)

    def get_covariance_matrix(self) -> np.ndarray:
        """Return the structured covariance matrix Σ."""
        return self.Sigma.copy()

    def get_cholesky_factor(self) -> np.ndarray:
        """Return the Cholesky factor L (lower triangular)."""
        return self.L.copy()

    # ------------------------------------------------------------------
    # GPU-accelerated helpers
    # ------------------------------------------------------------------

    def _compute_cholesky(self, Sigma: np.ndarray) -> np.ndarray:
        """Compute Cholesky factor, using GPU for large matrices.

        Args:
            Sigma: Covariance matrix of shape (n, n)

        Returns:
            Lower-triangular Cholesky factor L
        """
        n = Sigma.shape[0]

        if n >= GPU_CHOLESKY_MIN_SIZE and self._device_type != "cpu":
            try:
                import torch

                Sigma_t = torch.tensor(
                    Sigma, dtype=torch.float32, device=self._torch_device
                )
                L_t = torch.linalg.cholesky(Sigma_t)
                print(
                    f"  [SCM] Cholesky decomposition on {self._device_type.upper()} "
                    f"({n}x{n})"
                )
                return L_t.cpu().numpy()
            except Exception as exc:
                print(f"  [SCM] GPU Cholesky failed ({exc}), falling back to CPU")

        # CPU fallback
        return cpu_cholesky(Sigma, lower=True)

    def _gpu_sample_confounders(self, mu_matrix: np.ndarray, n_cats: int) -> np.ndarray:
        """Sample confounders on GPU.

        Args:
            mu_matrix: Mean matrix of shape (n_users, n_cats)
            n_cats: Number of categories

        Returns:
            Confounder array of shape (n_users, n_cats), clipped to [0, 1]
        """
        try:
            import torch

            mu_t = torch.tensor(
                mu_matrix, dtype=torch.float32, device=self._torch_device
            )
            L_t = torch.tensor(
                self.L[:n_cats, :n_cats],
                dtype=torch.float32,
                device=self._torch_device,
            )
            epsilon_t = torch.randn(
                mu_matrix.shape[0], n_cats, device=self._torch_device
            )
            confounders_t = mu_t + epsilon_t @ L_t.T
            return torch.clamp(confounders_t, 0, 1).cpu().numpy()
        except Exception as exc:
            print(f"  [SCM] GPU sampling failed ({exc}), falling back to CPU")

        # CPU fallback
        n_users = mu_matrix.shape[0]
        epsilon = np.random.normal(0, 1, (n_users, n_cats))
        confounders = mu_matrix + epsilon @ self.L[:n_cats, :n_cats].T
        return np.clip(confounders, 0, 1)

    # ------------------------------------------------------------------
    # Covariance matrix construction
    # ------------------------------------------------------------------

    def _build_covariance_matrix(self) -> np.ndarray:
        """Build a structured covariance matrix from dataset priors.

        Strategy:
        1. Start with a diagonal base (independent per category)
        2. Add off-diagonal correlations from Amazon co-purchasing data
        3. Add off-diagonal correlations from Wikispeedia topic transitions
        4. Add correlations from SNAP social network patterns
        5. Ensure positive definiteness
        """
        n = self.config.n_categories
        Sigma = np.eye(n) * self.config.base_diagonal

        if self.priors is None:
            Sigma = self._ensure_positive_definite(Sigma)
            return Sigma

        Sigma = self._add_amazon_copurchasing_correlations(Sigma)
        Sigma = self._add_wikispeedia_transition_correlations(Sigma)
        Sigma = self._add_snap_network_correlations(Sigma)
        Sigma = self._ensure_positive_definite(Sigma)

        return Sigma

    def _add_amazon_copurchasing_correlations(self, Sigma: np.ndarray) -> np.ndarray:
        """Add off-diagonal correlations from Amazon co-purchasing network."""
        amazon = self.priors.get("layer_1_persona.product_affinity_network", {}).get(
            "product_category_transitions", {}
        )

        if not amazon:
            return Sigma

        scale = self.config.prior_correlation_scale
        n = self.config.n_categories

        for transition, prob in amazon.items():
            parts = transition.split("->")
            if len(parts) != 2:
                continue
            # Map category names to indices using hash for deterministic mapping
            i = hash(parts[0].strip()) % n
            j = hash(parts[1].strip()) % n
            if i != j:
                correlation = prob * scale
                Sigma[i, j] += correlation
                Sigma[j, i] += correlation

        return Sigma

    def _add_wikispeedia_transition_correlations(self, Sigma: np.ndarray) -> np.ndarray:
        """Add off-diagonal correlations from Wikispeedia topic transitions."""
        wiki_matrix = self.priors.get(
            "layer_2_streams.search.category_transition_matrix", {}
        )

        if not wiki_matrix:
            return Sigma

        scale = self.config.prior_correlation_scale
        n = self.config.n_categories

        for from_cat, transitions in wiki_matrix.items():
            if not isinstance(transitions, dict):
                continue
            for to_cat, prob in transitions.items():
                i = hash(from_cat) % n
                j = hash(to_cat) % n
                if i != j and isinstance(prob, (int, float)):
                    correlation = prob * scale
                    Sigma[i, j] += correlation
                    Sigma[j, i] += correlation

        return Sigma

    def _add_snap_network_correlations(self, Sigma: np.ndarray) -> np.ndarray:
        """Add correlations from SNAP social network degree patterns."""
        degree_prior = self.priors.get("layer_1_persona.contact_network_degree", {})

        clustering = degree_prior.get("clustering_coefficient", 0.1)
        if not isinstance(clustering, (int, float)):
            clustering = 0.1

        # Clustering coefficient creates small-world correlations
        # between nearby category indices
        n = self.config.n_categories
        scale = clustering * self.config.prior_correlation_scale

        for i in range(n):
            for j in range(i + 1, n):
                distance = abs(i - j)
                if distance <= 2:  # Nearby categories are more correlated
                    correlation = scale / (distance + 1)
                    Sigma[i, j] += correlation
                    Sigma[j, i] += correlation

        return Sigma

    @staticmethod
    def _ensure_positive_definite(Sigma: np.ndarray) -> np.ndarray:
        """Ensure the covariance matrix is positive definite.

        Adds a small regularization term to the diagonal if needed.
        """
        n = Sigma.shape[0]
        try:
            cpu_cholesky(Sigma, lower=True)
            return Sigma
        except np.linalg.LinAlgError:
            # Add regularization to diagonal
            eigvals = np.linalg.eigvalsh(Sigma)
            min_eig = eigvals.min()
            if min_eig < 1e-10:
                Sigma += np.eye(n) * (abs(min_eig) + 1e-6)
            return Sigma

    # ------------------------------------------------------------------
    # Mean vector computation
    # ------------------------------------------------------------------

    def _compute_mean_matrix(self, users: List, n_cats: int) -> np.ndarray:
        """Compute the mean vector μ for each user from persona attributes.

        The mean represents the expected base affinity before category-specific
        correlations are applied via the covariance matrix.

        Args:
            users: List of User objects
            n_cats: Number of categories

        Returns:
            Array of shape (n_users, n_cats)
        """
        n_users = len(users)
        mu_matrix = np.zeros((n_users, n_cats))

        for i, user in enumerate(users):
            # Base affinity from persona attributes (same formula as before,
            # but now it's the mean of the SCM rather than the final value)
            base = (
                user.latent_tech_interest * 0.5
                + user.latent_finance_interest * 0.3
                + user.persona.income_level.value * 0.2
            )
            mu_matrix[i, :] = base

        return mu_matrix
