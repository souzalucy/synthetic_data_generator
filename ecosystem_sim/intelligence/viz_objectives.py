"""Visualization objective registry for explanation generation.

Each entry defines:
- ``title``: Human-readable chart title.
- ``objective``: The analytical purpose of this visualization — *why* it exists.
- ``key_questions``: Specific questions the visualization answers.
- ``interpretation_rules``: Heuristics for turning numbers into insights.

This module is the single source of truth for what each chart type
is trying to communicate, decoupled from the explanation generation logic.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VIZ_OBJECTIVES: Dict[str, Dict] = {}

# ── Lift / Uplift Analysis ────────────────────────────────────────────────


VIZ_OBJECTIVES["lift_deciles"] = {
    "title": "Campaign Uplift by Decile",
    "objective": (
        "Identify which user deciles respond most strongly to treatment, "
        "so the campaign can target high-uplift segments and avoid "
        "low-uplift or negative-uplift segments to maximise ROI."
    ),
    "key_questions": [
        "What is the uplift range across deciles (min -> max)?",
        "Which deciles show positive vs negative average uplift?",
        "How much of the total uplift is concentrated in the top 3 deciles?",
        "Is there a clear monotonic trend (higher decile -> higher uplift)?",
    ],
    "interpretation_rules": {
        "positive_uplift_deciles": ("Deciles with avg_uplift > 0 are worth targeting."),
        "top_decile_concentration": (
            "If top 3 deciles capture >60 % of total uplift, "
            "targeting is highly efficient."
        ),
        "negative_uplift_deciles": (
            "Deciles with avg_uplift < 0 should be excluded from treatment "
            "to avoid harming campaign performance."
        ),
    },
}


VIZ_OBJECTIVES["qini_curve"] = {
    "title": "Qini Curve: Model vs Random Targeting",
    "objective": (
        "Quantify how much better the model's targeting is compared to "
        "random targeting, to justify using the model for campaign allocation."
    ),
    "key_questions": [
        "What is the Qini coefficient? Is it > 0.15 (strong)?",
        "At what population percentage does the model diverge most from random?",
        "What is the AUUC and how does it compare to benchmarks?",
        "What is the overall campaign lift percentage?",
    ],
    "interpretation_rules": {
        "qini_strong": ("Qini > 0.15 indicates strong discriminatory power."),
        "qini_moderate": (
            "Qini between 0.05 and 0.15 indicates moderate discriminatory power."
        ),
        "qini_weak": (
            "Qini < 0.05 indicates weak discriminatory power - "
            "the model adds little value over random targeting."
        ),
    },
}


VIZ_OBJECTIVES["lift_summary"] = {
    "title": "Campaign Lift Analysis Dashboard",
    "objective": (
        "Provide an executive summary of campaign performance including "
        "overall lift, statistical significance, and comparison to "
        "industry benchmarks (Criteo)."
    ),
    "key_questions": [
        "What is the overall campaign lift percentage?",
        "Is the lift statistically significant (p < 0.05)?",
        "How does performance compare to the Criteo benchmark average?",
        "What are the AUUC and Qini coefficient values?",
    ],
    "interpretation_rules": {
        "significant": ("p < 0.05 means the observed lift is unlikely due to chance."),
        "above_benchmark": (
            "Lift above the Criteo benchmark average indicates "
            "above-industry campaign performance."
        ),
    },
}


# ── Causal Trajectories ───────────────────────────────────────────────────


VIZ_OBJECTIVES["causal_trajectories"] = {
    "title": "Causal Interest Trajectories",
    "objective": (
        "Understand how user interest (base affinity) evolves over time "
        "across different persona types, revealing temporal dynamics, "
        "persistence effects, and divergence between user segments."
    ),
    "key_questions": [
        "What is the overall mean affinity and how does it trend over days?",
        "Do different personas follow distinct trajectory patterns?",
        "Are confidence bands widening over time (indicating divergence)?",
        "What is the effect of the AR(1) persistence process on trajectories?",
    ],
    "interpretation_rules": {
        "widening_bands": (
            "Widening 95 % confidence bands indicate increasing "
            "divergence between users within a persona."
        ),
        "upward_trend": (
            "A sustained upward trend suggests positive persistence "
            "effects reinforcing user interest."
        ),
        "persona_separation": (
            "Clear separation between persona mean trajectories "
            "validates that personas capture meaningful heterogeneity."
        ),
    },
}


VIZ_OBJECTIVES["persona_trajectories"] = {
    "title": "Interest Trajectories by Persona (Faceted)",
    "objective": (
        "Provide a detailed per-persona view of interest trajectories, "
        "showing both individual user paths and the persona average, "
        "to assess within-persona homogeneity and between-persona separation."
    ),
    "key_questions": [
        "How much within-persona variability exists (spread of thin lines)?",
        "Do persona mean trajectories differ meaningfully from each other?",
        "Which personas show the highest / lowest affinity over time?",
    ],
    "interpretation_rules": {
        "tight_clusters": (
            "Thin lines tightly clustered around the mean indicate "
            "a homogeneous persona with predictable behaviour."
        ),
        "wide_spread": (
            "Wide spread of individual lines suggests the persona "
            "may need further sub-segmentation."
        ),
    },
}


# ── Counterfactual Analysis ───────────────────────────────────────────────


VIZ_OBJECTIVES["counterfactual_deltas"] = {
    "title": "Counterfactual Graph Analysis Dashboard",
    "objective": (
        "Compare how different counterfactual interventions change "
        "the user interaction graph relative to the factual (observed) world, "
        "identifying which interventions have the strongest structural impact."
    ),
    "key_questions": [
        "Which scenarios were tested and how do they compare?",
        "Which intervention causes the largest delta in graph metrics?",
        "Are the effects consistent across metrics (edges, density, degree)?",
        "Which metrics are most sensitive to interventions?",
    ],
    "interpretation_rules": {
        "positive_delta": (
            "Positive delta means the intervention increases "
            "graph connectivity (more edges, higher engagement)."
        ),
        "negative_delta": (
            "Negative delta means the intervention reduces "
            "graph connectivity (sparser interactions)."
        ),
        "largest_effect": (
            "The scenario with the largest absolute delta "
            "is the most impactful intervention."
        ),
    },
}


VIZ_OBJECTIVES["causal_pathways"] = {
    "title": "Discovered Causal Pathways",
    "objective": (
        "Reveal the causal mechanisms through which interventions "
        "affect the user engagement graph, quantifying each pathway's "
        "effect size and providing an interpretation."
    ),
    "key_questions": [
        "How many causal pathways were discovered?",
        "Which pathway has the strongest effect on graph structure?",
        "What is the direction and magnitude of each pathway's effect?",
        "Do the pathways suggest a coherent causal story?",
    ],
    "interpretation_rules": {
        "strongest_pathway": (
            "The pathway with the largest absolute relative change "
            "is the dominant causal mechanism."
        ),
        "multiple_pathways": (
            "Multiple pathways suggest the intervention operates "
            "through several distinct mechanisms."
        ),
    },
}


# ── Device Stitching ──────────────────────────────────────────────────────


VIZ_OBJECTIVES["device_stitching"] = {
    "title": "Device Stitching Clusters",
    "objective": (
        "Visualise how individual devices are stitched together into "
        "user-level clusters based on co-occurrence patterns, revealing "
        "multi-device behaviour and cross-device identity resolution quality."
    ),
    "key_questions": [
        "How many devices and clusters were identified?",
        "What is the average cluster size?",
        "Are there unusually large clusters (potential stitching errors)?",
        "What is the distribution of devices per cluster?",
    ],
    "interpretation_rules": {
        "avg_cluster_size": (
            "Average cluster size of 1-3 devices is typical for individual users."
        ),
        "large_clusters": (
            "Clusters with >5 devices may represent shared devices "
            "(family tablets) or stitching errors."
        ),
    },
}


VIZ_OBJECTIVES["interaction_graphs"] = {
    "title": "Interaction Graph Summary Metrics",
    "objective": (
        "Summarise the overall structure of user interaction graphs, "
        "showing how users engage with digital services and the "
        "connectivity patterns that emerge."
    ),
    "key_questions": [
        "How many users have interaction graphs?",
        "What is the total number of edges and average degree?",
        "What is the distribution of edge counts across users?",
        "How do events per user and edges per user compare?",
    ],
    "interpretation_rules": {
        "high_degree": (
            "High average degree indicates users with diverse "
            "and connected digital behaviour."
        ),
        "skewed_distribution": (
            "A right-skewed edge count distribution means most users "
            "have few connections while a few have many."
        ),
    },
}


# ── Ground Truth ──────────────────────────────────────────────────────────


VIZ_OBJECTIVES["ground_truth_summary"] = {
    "title": "Ground Truth Summary Dashboard",
    "objective": (
        "Provide an overview of the generated ground truth data, "
        "including simulation configuration, persona distributions, "
        "treatment effect variability, and base affinity ranges."
    ),
    "key_questions": [
        "What is the simulation scale (users, days, categories)?",
        "How are users distributed across persona types?",
        "What is the range and variability of treatment effects?",
        "What is the range and variability of base affinities?",
    ],
    "interpretation_rules": {
        "balanced_personas": (
            "Even persona distributions indicate a well-balanced simulation design."
        ),
        "treatment_variability": (
            "High variability in treatment effects across personas "
            "means there is strong potential for targeted campaigns."
        ),
    },
}


# ── Propensity / Distributions ────────────────────────────────────────────


VIZ_OBJECTIVES["propensity_distributions"] = {
    "title": "Propensity & Treatment Effect Distributions",
    "objective": (
        "Examine how treatment effects and base affinities vary "
        "across the user population and between persona types, "
        "to understand heterogeneity and targeting potential."
    ),
    "key_questions": [
        "How do treatment effects differ across personas?",
        "How do base affinities differ across personas?",
        "What is the overall distribution shape for each metric?",
        "Are there personas with consistently high or low values?",
    ],
    "interpretation_rules": {
        "persona_separation": (
            "Clear separation between persona box plots validates "
            "that personas capture meaningful behavioural differences."
        ),
        "wide_distributions": (
            "Wide distributions within a persona indicate "
            "heterogeneous responses that may need sub-segmentation."
        ),
    },
}


VIZ_OBJECTIVES["price_sensitivity"] = {
    "title": "Persona-Level Metrics (Price Sensitivity Proxy)",
    "objective": (
        "Use persona-level treatment effects and base affinities as "
        "a proxy for price sensitivity analysis, identifying which "
        "personas are most responsive to advertising interventions."
    ),
    "key_questions": [
        "Which personas have the highest mean treatment effect?",
        "Which personas have the highest mean base affinity?",
        "How much within-persona variability exists (error bars)?",
    ],
    "interpretation_rules": {
        "high_te": (
            "Personas with high treatment effects are more responsive "
            "to advertising, which may correlate with price sensitivity."
        ),
        "wide_error_bars": (
            "Wide error bars indicate heterogeneous responses within a persona group."
        ),
    },
}


VIZ_OBJECTIVES["cluster_size_distribution"] = {
    "title": "Device Cluster Size Distribution",
    "objective": (
        "Analyse the distribution of device cluster sizes to assess "
        "stitching quality and understand multi-device behaviour patterns."
    ),
    "key_questions": [
        "What is the average, minimum, and maximum cluster size?",
        "Are most clusters small (1-3 devices) or are there outliers?",
        "What does the distribution shape tell us about stitching quality?",
    ],
    "interpretation_rules": {
        "typical_size": (
            "Most clusters with 1-3 devices is expected for individual user behaviour."
        ),
        "outliers": (
            "Outliers (>5 devices) may indicate shared devices "
            "or potential stitching errors worth investigating."
        ),
    },
}


# ── Treatment Matrix ──────────────────────────────────────────────────────


VIZ_OBJECTIVES["treatment_matrix"] = {
    "title": "Treatment Effects & Base Affinity Distributions",
    "objective": (
        "Visualise the distributions of treatment effects and base "
        "affinities across all users, revealing the overall population "
        "heterogeneity that determines campaign targeting potential."
    ),
    "key_questions": [
        "What is the mean and spread of treatment effects?",
        "What is the mean and spread of base affinities?",
        "Are the distributions approximately normal or skewed?",
        "How much variation exists for potential targeting gains?",
    ],
    "interpretation_rules": {
        "high_variance": (
            "More variation in treatment effects means more "
            "opportunity for targeted treatment and uplift gains."
        ),
        "skewed": (
            "Skewed distributions indicate that most users cluster "
            "at one end of the spectrum, with a long tail of outliers."
        ),
    },
}


# ── Interest Heatmap ──────────────────────────────────────────────────────


VIZ_OBJECTIVES["interest_heatmap"] = {
    "title": "Average Interest: Days x Categories",
    "objective": (
        "Reveal temporal and categorical patterns in user interest, "
        "showing how average interest evolves across days and content "
        "categories simultaneously."
    ),
    "key_questions": [
        "Which categories have the highest average interest?",
        "Are there temporal trends (increasing/decreasing over days)?",
        "Are there category x day interaction patterns?",
        "What is the overall mean interest level?",
    ],
    "interpretation_rules": {
        "bright_columns": (
            "Consistently bright columns indicate categories with "
            "sustained high interest over time."
        ),
        "vertical_patterns": (
            "Vertical gradients reveal temporal trends - "
            "warming colours over time indicate growing interest."
        ),
        "horizontal_patterns": (
            "Horizontal gradients show which categories are most engaging overall."
        ),
    },
}


# ── Counterfactual Table / User Comparison ────────────────────────────────


VIZ_OBJECTIVES["user_counterfactual_comparison"] = {
    "title": "Per-User Factual vs Counterfactual Comparison",
    "objective": (
        "Compare each user's factual interest trajectory against their "
        "counterfactual trajectory under a specific intervention, "
        "revealing individual-level treatment effects over time."
    ),
    "key_questions": [
        "How does the counterfactual trajectory differ from factual for each user?",
        "Is the intervention effect consistent across users or highly variable?",
        "Which users show the largest positive or negative deltas?",
        "Does the effect grow, shrink, or remain stable over time?",
    ],
    "interpretation_rules": {
        "positive_delta": (
            "Counterfactual above factual means the intervention "
            "increases user interest."
        ),
        "negative_delta": (
            "Counterfactual below factual means the intervention "
            "decreases user interest."
        ),
        "growing_delta": (
            "A widening gap over time suggests the intervention "
            "has a cumulative or compound effect."
        ),
    },
}


VIZ_OBJECTIVES["counterfactual_table"] = {
    "title": "Layer 1 User Table: Factual vs Counterfactual",
    "objective": (
        "Provide a detailed, per-user, per-day tabular comparison of "
        "factual and counterfactual base affinities, colour-coded by "
        "delta magnitude for quick visual scanning."
    ),
    "key_questions": [
        "What is the distribution of deltas across users?",
        "Are most deltas positive, negative, or near zero?",
        "Which users have the most extreme deltas?",
        "How does the intervention type affect the delta pattern?",
    ],
    "interpretation_rules": {
        "green_cells": (
            "Green delta cells (positive) indicate users whose "
            "interest increases under the intervention."
        ),
        "red_cells": (
            "Red delta cells (negative) indicate users whose "
            "interest decreases under the intervention."
        ),
        "mostly_green": (
            "A predominantly green table means the intervention "
            "is broadly effective across the user population."
        ),
    },
}


# ── Helper ────────────────────────────────────────────────────────────────


def get_objective(chart_type: str) -> Dict:
    """Return the objective entry for a chart type, or a default.

    Args:
        chart_type: The registered chart type key.

    Returns:
        Dict with at least ``title``, ``objective``, ``key_questions``.
    """
    return VIZ_OBJECTIVES.get(
        chart_type,
        {
            "title": chart_type.replace("_", " ").title(),
            "objective": (
                f"Explore the **{chart_type.replace('_', ' ').title()}** "
                f"visualisation from the simulation output."
            ),
            "key_questions": [
                "What patterns or insights does this visualisation reveal?",
                "What are the key numbers and how should they be interpreted?",
            ],
            "interpretation_rules": {},
        },
    )


def list_chart_types() -> List[str]:
    """Return all registered chart type keys."""
    return list(VIZ_OBJECTIVES.keys())
