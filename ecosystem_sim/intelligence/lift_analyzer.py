"""Lift analysis for campaign effectiveness with Criteo uplift benchmarks."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import ttest_ind, mannwhitneyu
import math


class LiftAnalyzer:
    """Counterfactual analysis and lift calculation with Criteo uplift benchmarks."""

    def __init__(self, causal_engine, priors: Optional[Dict] = None):
        """Initialize lift analyzer with Criteo benchmarks.

        Args:
            causal_engine: Causal inference engine
            priors: Optional compiled dataset priors
        """
        self.causal_engine = causal_engine
        self.priors = priors

        from ecosystem_sim.core.prior_loader import get_prior

        self.base_conversion = get_prior(
            priors, "layer_3_intelligence.propensity.conversion_rate_by_persona", None
        )

        # Criteo uplift benchmark parameters
        self.criteo_benchmarks = self._load_criteo_benchmarks()

        # Uplift modeling parameters
        self.uplift_decay_rate = 0.85  # How quickly uplift decays over time
        self.min_sample_size = 100  # Minimum users for reliable uplift estimation

    def _load_criteo_benchmarks(self) -> Dict:
        """Load Criteo uplift modeling benchmarks."""
        # Default Criteo-style benchmarks
        benchmarks = {
            "baseline_conversion_rate": 0.02,  # 2% baseline
            "average_uplift": 0.18,  # 18% average uplift
            "uplift_std": 0.08,  # Standard deviation of uplift
            "persistence_factor": 0.68,  # How persistent treatment effects are
            "max_uplift": 0.35,  # Maximum realistic uplift
            "min_uplift": 0.05,  # Minimum detectable uplift
        }

        # Override with priors if available
        if self.priors:
            from ecosystem_sim.core.prior_loader import get_prior

            uplift_priors = get_prior(
                self.priors, "layer_3_intelligence.causal_engine", {}
            )
            if uplift_priors:
                if "treatment_effect_distribution" in uplift_priors:
                    te_dist = uplift_priors["treatment_effect_distribution"]
                    benchmarks["average_uplift"] = te_dist.get("mean", 0.18)
                    benchmarks["uplift_std"] = te_dist.get("std", 0.08)

                if "persistence_factor" in uplift_priors:
                    pf = uplift_priors["persistence_factor"]
                    benchmarks["persistence_factor"] = pf.get("mean", 0.68)

        return benchmarks

    def calculate_lift(
        self,
        campaign_id: str,
        users: List,
        interest_matrix: np.ndarray,
        campaign_day: int,
        campaign_category: int = None,
    ) -> Dict:
        """Calculate campaign lift vs counterfactual with Criteo benchmarks.

        Args:
            campaign_id: Campaign identifier
            users: List of users
            interest_matrix: Interest trajectory matrix
            campaign_day: Day campaign ran
            campaign_category: Target category index (None = all)

        Returns:
            Lift analysis report with Criteo benchmarks
        """
        # Clamp campaign_day to valid range for the interest matrix
        n_days = interest_matrix.shape[1]
        if campaign_day < 0 or campaign_day >= n_days:
            print(
                f"  [WARN] campaign_day={campaign_day} out of bounds "
                f"(matrix has {n_days} days). Clamping to {n_days - 1}."
            )
            campaign_day = n_days - 1

        results = {
            "campaign_id": campaign_id,
            "exposed_users": [],
            "control_users": [],
            "lift_percent": 0,
            "p_value": 1.0,
            "auuc": 0,
            "qini_coefficient": 0,
            "criteo_benchmark_comparison": {},
            "uplift_by_decile": [],
        }

        exposed_purchases = []
        counterfactual_purchases = []
        uplift_scores = []

        for user_idx, user in enumerate(users):
            # Get actual outcome (with ads)
            actual_interest = interest_matrix[user_idx, campaign_day, :]

            if campaign_category is not None:
                actual_purchases = actual_interest[campaign_category]
            else:
                actual_purchases = actual_interest.mean()

            # Get counterfactual outcome (no ads)
            exposure = self.causal_engine.treatment_matrix[user_idx, campaign_day]
            treatment_effect = self.causal_engine.treatment_effects[user_idx]

            # Counterfactual: reduce interest by treatment effect
            counterfactual_interest = actual_interest - (treatment_effect * exposure)
            counterfactual_interest = np.clip(counterfactual_interest, 0, 1)

            if campaign_category is not None:
                counterfactual_purchases_val = counterfactual_interest[
                    campaign_category
                ]
            else:
                counterfactual_purchases_val = counterfactual_interest.mean()

            # Assume conversion from interest
            actual_conv = actual_purchases * 0.3
            counterfactual_conv = counterfactual_purchases_val * 0.3

            # Calculate uplift score
            uplift = actual_conv - counterfactual_conv
            uplift_scores.append((user.user_id, uplift))

            exposed_purchases.append(actual_conv)
            counterfactual_purchases.append(counterfactual_conv)

            results["exposed_users"].append(
                {
                    "user_id": user.user_id,
                    "actual": float(actual_conv),
                    "counterfactual": float(counterfactual_conv),
                    "incremental": float(uplift),
                    "uplift_score": float(uplift),
                }
            )

        # Calculate aggregate lift
        total_actual = sum(exposed_purchases)
        total_counterfactual = sum(counterfactual_purchases)

        if total_counterfactual > 0:
            results["lift_percent"] = (
                (total_actual - total_counterfactual) / total_counterfactual * 100
            )
        else:
            results["lift_percent"] = 0

        # Statistical significance
        if len(exposed_purchases) > 1 and len(counterfactual_purchases) > 1:
            try:
                t_stat, p_value = ttest_ind(exposed_purchases, counterfactual_purchases)
                results["p_value"] = p_value
            except:
                results["p_value"] = 1.0

        # Calculate AUUC (Area Under Uplift Curve)
        results["auuc"] = self._calculate_auuc(uplift_scores, exposed_purchases)

        # Calculate Qini coefficient
        results["qini_coefficient"] = self._calculate_qini_coefficient(uplift_scores)

        # Calculate uplift by decile
        results["uplift_by_decile"] = self._calculate_uplift_by_decile(uplift_scores)

        # Compare with Criteo benchmarks
        results["criteo_benchmark_comparison"] = self._compare_with_criteo_benchmarks(
            results["lift_percent"] / 100,  # Convert to proportion
            len(users),
            results["p_value"],
        )

        return results

    def _calculate_auuc(
        self, uplift_scores: List[Tuple[str, float]], actual_conversions: List[float]
    ) -> float:
        """Calculate Area Under Uplift Curve."""
        if not uplift_scores:
            return 0.0

        # Sort by uplift score descending
        sorted_scores = sorted(uplift_scores, key=lambda x: x[1], reverse=True)

        # Calculate cumulative uplift
        n = len(sorted_scores)
        cumulative_uplift = 0
        auuc = 0

        for i, (user_id, uplift) in enumerate(sorted_scores):
            cumulative_uplift += uplift
            auuc += cumulative_uplift

        # Normalize
        if n > 0:
            auuc = auuc / n

        return auuc

    def _calculate_qini_coefficient(
        self, uplift_scores: List[Tuple[str, float]]
    ) -> float:
        """Calculate Qini coefficient for uplift model."""
        if not uplift_scores or len(uplift_scores) < 10:
            return 0.0

        # Sort by uplift score
        sorted_scores = sorted(uplift_scores, key=lambda x: x[1], reverse=True)
        n = len(sorted_scores)

        # Calculate random targeting baseline
        total_uplift = sum(score for _, score in sorted_scores)
        random_uplift_per_step = total_uplift / n

        # Calculate model uplift curve
        cumulative_model = 0
        cumulative_random = 0
        qini_area = 0

        # Initialize previous values
        cumulative_model_prev = 0
        cumulative_random_prev = 0

        for i, (_, uplift) in enumerate(sorted_scores):
            cumulative_model += uplift
            cumulative_random += random_uplift_per_step

            # Add trapezoid area (starting from second point)
            if i > 0:
                qini_area += (cumulative_model + cumulative_model_prev) / 2
                qini_area -= (cumulative_random + cumulative_random_prev) / 2

            # Update previous values for next iteration
            cumulative_model_prev = cumulative_model
            cumulative_random_prev = cumulative_random

        # Normalize
        qini_coefficient = qini_area / n if n > 0 else 0

        return qini_coefficient

    def _calculate_uplift_by_decile(
        self, uplift_scores: List[Tuple[str, float]]
    ) -> List[Dict]:
        """Calculate uplift statistics by decile."""
        if not uplift_scores:
            return []

        # Sort by uplift score
        sorted_scores = sorted(uplift_scores, key=lambda x: x[1], reverse=True)
        n = len(sorted_scores)
        decile_size = max(1, n // 10)

        uplift_by_decile = []

        for decile in range(10):
            start_idx = decile * decile_size
            end_idx = min((decile + 1) * decile_size, n)

            if start_idx >= n:
                break

            decile_scores = [score for _, score in sorted_scores[start_idx:end_idx]]

            uplift_by_decile.append(
                {
                    "decile": decile + 1,
                    "user_count": len(decile_scores),
                    "avg_uplift": np.mean(decile_scores) if decile_scores else 0,
                    "std_uplift": np.std(decile_scores)
                    if len(decile_scores) > 1
                    else 0,
                    "min_uplift": min(decile_scores) if decile_scores else 0,
                    "max_uplift": max(decile_scores) if decile_scores else 0,
                    "cumulative_uplift": sum(decile_scores),
                }
            )

        return uplift_by_decile

    def _compare_with_criteo_benchmarks(
        self, observed_uplift: float, sample_size: int, p_value: float
    ) -> Dict:
        """Compare results with Criteo uplift benchmarks."""
        benchmark = self.criteo_benchmarks

        comparison = {
            "observed_uplift": observed_uplift,
            "benchmark_average": benchmark["average_uplift"],
            "difference_from_benchmark": observed_uplift - benchmark["average_uplift"],
            "relative_difference": (observed_uplift - benchmark["average_uplift"])
            / benchmark["average_uplift"]
            * 100
            if benchmark["average_uplift"] > 0
            else 0,
            "statistically_significant": p_value < 0.05,
            "sample_size_adequate": sample_size >= self.min_sample_size,
            "benchmark_notes": [],
        }

        # Add notes based on comparison
        if observed_uplift > benchmark["max_uplift"]:
            comparison["benchmark_notes"].append(
                f"Uplift ({observed_uplift:.1%}) exceeds Criteo maximum benchmark "
                f"({benchmark['max_uplift']:.1%}). May indicate overestimation."
            )
        elif observed_uplift < benchmark["min_uplift"]:
            comparison["benchmark_notes"].append(
                f"Uplift ({observed_uplift:.1%}) below Criteo minimum benchmark "
                f"({benchmark['min_uplift']:.1%}). Campaign may be ineffective."
            )

        if not comparison["statistically_significant"]:
            comparison["benchmark_notes"].append(
                "Results not statistically significant (p > 0.05)."
            )

        if not comparison["sample_size_adequate"]:
            comparison["benchmark_notes"].append(
                f"Sample size ({sample_size}) below minimum ({self.min_sample_size}) "
                "for reliable uplift estimation."
            )

        # Calculate confidence interval
        if sample_size >= 30:
            se = benchmark["uplift_std"] / math.sqrt(sample_size)
            comparison["confidence_interval"] = {
                "lower": observed_uplift - 1.96 * se,
                "upper": observed_uplift + 1.96 * se,
                "width": 3.92 * se,
            }

        return comparison

    def run_ab_test_simulation(
        self,
        control_users: List,
        treatment_users: List,
        interest_matrix: np.ndarray,
        campaign_day: int,
    ) -> Dict:
        """Simulate A/B test outcome with Criteo methodology.

        Args:
            control_users: Control group users
            treatment_users: Treatment group users
            interest_matrix: Interest matrix
            campaign_day: Campaign day

        Returns:
            A/B test results with Criteo metrics
        """
        # Clamp campaign_day to valid range for the interest matrix
        n_days = interest_matrix.shape[1]
        if campaign_day < 0 or campaign_day >= n_days:
            print(
                f"  [WARN] run_ab_test_simulation: campaign_day={campaign_day} "
                f"out of bounds (matrix has {n_days} days). Clamping to {n_days - 1}."
            )
            campaign_day = n_days - 1

        control_conversions = []
        treatment_conversions = []

        for user in control_users:
            user_idx = next(
                i for i, u in enumerate(control_users) if u.user_id == user.user_id
            )
            avg_interest = interest_matrix[user_idx, campaign_day, :].mean()
            # Apply Criteo baseline conversion rate
            conversion = (
                avg_interest * self.criteo_benchmarks["baseline_conversion_rate"] * 10
            )
            control_conversions.append(conversion)

        for user in treatment_users:
            user_idx = next(
                i for i, u in enumerate(treatment_users) if u.user_id == user.user_id
            )
            avg_interest = interest_matrix[user_idx, campaign_day, :].mean()
            # Apply uplift
            uplift = self.criteo_benchmarks["average_uplift"]
            conversion = (
                avg_interest
                * self.criteo_benchmarks["baseline_conversion_rate"]
                * 10
                * (1 + uplift)
            )
            treatment_conversions.append(conversion)

        control_rate = np.mean(control_conversions) if control_conversions else 0
        treatment_rate = np.mean(treatment_conversions) if treatment_conversions else 0

        # Statistical test (Mann-Whitney U for non-normal distributions)
        if len(control_conversions) > 1 and len(treatment_conversions) > 1:
            try:
                u_stat, p_value = mannwhitneyu(
                    treatment_conversions, control_conversions
                )
            except:
                p_value = 1.0
        else:
            p_value = 1.0

        # Calculate relative risk and number needed to treat (NNT)
        relative_risk = treatment_rate / control_rate if control_rate > 0 else 0
        absolute_risk_reduction = treatment_rate - control_rate
        nnt = (
            1 / absolute_risk_reduction if absolute_risk_reduction > 0 else float("inf")
        )

        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "lift_bps": (treatment_rate - control_rate) * 10000,
            "relative_risk": relative_risk,
            "absolute_risk_reduction": absolute_risk_reduction,
            "nnt": nnt if nnt != float("inf") else "Inf",
            "p_value": p_value,
            "significant": p_value < 0.05,
            "criteo_benchmark_uplift": self.criteo_benchmarks["average_uplift"],
            "benchmark_comparison": self._compare_ab_with_benchmark(
                treatment_rate - control_rate,
                len(control_users) + len(treatment_users),
                p_value,
            ),
        }

    def _compare_ab_with_benchmark(
        self, observed_lift: float, total_sample: int, p_value: float
    ) -> Dict:
        """Compare A/B test results with Criteo benchmarks."""
        benchmark = self.criteo_benchmarks

        return {
            "observed_lift": observed_lift,
            "benchmark_lift": benchmark["average_uplift"],
            "within_benchmark_range": (
                benchmark["min_uplift"] <= observed_lift <= benchmark["max_uplift"]
            ),
            "power_analysis": self._calculate_power_analysis(
                total_sample, observed_lift
            ),
            "recommended_sample_size": self._calculate_required_sample_size(
                observed_lift
            ),
            "statistical_power": 1
            - self._calculate_type2_error(total_sample, observed_lift, p_value),
        }

    def _calculate_power_analysis(self, sample_size: int, effect_size: float) -> Dict:
        """Calculate statistical power for the test."""
        # Simplified power calculation
        if sample_size < 30:
            power = 0.3
        elif sample_size < 100:
            power = 0.5
        elif sample_size < 500:
            power = 0.8
        else:
            power = 0.95

        return {
            "sample_size": sample_size,
            "effect_size": effect_size,
            "statistical_power": power,
            "adequate_power": power >= 0.8,
        }

    def _calculate_type2_error(
        self, sample_size: int, effect_size: float, p_value: float
    ) -> float:
        """Calculate Type II error (beta) probability."""
        # Simplified Type II error calculation
        if sample_size < 30:
            return 0.7  # High error rate for small samples
        elif sample_size < 100:
            return 0.5  # Moderate error rate
        elif sample_size < 500:
            return 0.2  # Low error rate
        else:
            return 0.05  # Very low error rate for large samples

    def _calculate_required_sample_size(
        self, effect_size: float, power: float = 0.8, alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for given effect size."""
        # Simplified sample size calculation based on effect size
        if effect_size <= 0:
            return 1000  # Large sample needed for small effects

        # Sample size formula: n = (Z_alpha + Z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p1-p2)^2
        # Simplified version for typical conversion rates
        if effect_size < 0.05:
            return 2000  # Very small effect, need large sample
        elif effect_size < 0.1:
            return 1000  # Small effect
        elif effect_size < 0.2:
            return 500  # Moderate effect
        elif effect_size < 0.3:
            return 250  # Large effect
        else:
            return 100  # Very large effect
