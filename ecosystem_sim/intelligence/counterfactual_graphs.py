"""Counterfactual graph generation and comparison engine.

Extends GraphStitcher to generate multiple graph worlds from factual and
counterfactual interest trajectories, then compares them to discover
causal pathways and measure graph-level treatment effects.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import math

from .graph_stitcher import GraphStitcher


class CounterfactualGraphStitcher(GraphStitcher):
    """Extends GraphStitcher to generate and compare counterfactual graph worlds.

    Creates separate interaction graphs for each counterfactual scenario
    (no treatment, double treatment, optimal targeting, etc.) and computes
    graph-level causal metrics:

    - Graph edit distance between factual and counterfactual worlds
    - Community structure changes (Δ modularity, Δ communities)
    - Centrality shifts (Δ PageRank, Δ degree centrality)
    - Edge density changes (engagement impact)
    - Per-user graph-level treatment effects
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        network_priors: Optional[Dict] = None,
    ):
        """Initialize counterfactual graph stitcher.

        Args:
            confidence_threshold: Threshold for device stitching confidence
            network_priors: Optional SNAP network priors
        """
        super().__init__(confidence_threshold, network_priors)
        self.scenario_graphs: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Multi-world graph generation
    # ------------------------------------------------------------------

    def generate_all_scenario_graphs(
        self,
        causal_engine,
        users: List,
        factual_interest_matrix: np.ndarray,
        events_by_scenario: Optional[Dict[str, List[Dict]]] = None,
        intervention_day: int = 14,
    ) -> Dict[str, Any]:
        """Generate interaction graphs for all counterfactual scenarios.

        If events_by_scenario is provided, uses those pre-generated events.
        Otherwise, generates events from the interest matrices internally.

        Args:
            causal_engine: CausalEngine instance with counterfactual methods
            users: List of User objects
            factual_interest_matrix: Factual interest trajectory matrix
            events_by_scenario: Optional pre-generated events per scenario
            intervention_day: Day at which the intervention (marketing action)
                occurs. Before this day, trajectories match the factual world.
                Default: 14.

        Returns:
            Dict with keys:
              - "scenarios": Dict[str, Dict] - graph data per scenario
              - "comparison": Dict - cross-scenario graph metrics
              - "causal_pathways": List[Dict] - discovered causal pathways
        """
        # Define scenarios to generate
        scenarios = self._define_scenarios(
            causal_engine, factual_interest_matrix, intervention_day
        )

        # Build graphs for each scenario
        self.scenario_graphs = {}
        for name, interest_matrix in scenarios.items():
            if events_by_scenario and name in events_by_scenario:
                events = events_by_scenario[name]
            else:
                events = self._interest_matrix_to_events(
                    interest_matrix, users, scenario_name=name
                )

            graph_data = self.build_interaction_graphs(events)
            self.scenario_graphs[name] = {
                "graph": graph_data,
                "interest_matrix": interest_matrix,
                "n_events": len(events),
            }

        # Compute cross-scenario comparisons
        comparison = self._compare_all_scenarios()

        # Discover causal pathways
        causal_pathways = self._discover_causal_pathways()

        return {
            "scenarios": {
                name: {
                    "metrics": data["graph"]["metrics"],
                    "n_events": data["n_events"],
                }
                for name, data in self.scenario_graphs.items()
            },
            "comparison": comparison,
            "causal_pathways": causal_pathways,
        }

    def _define_scenarios(
        self,
        causal_engine,
        factual_interest_matrix: np.ndarray,
        intervention_day: int = 14,
    ) -> Dict[str, np.ndarray]:
        """Define counterfactual scenarios and generate their interest matrices.

        All counterfactual scenarios branch from the factual trajectory at
        intervention_day, meaning the intervention (marketing action) occurs
        at that point in time. Everything before intervention_day is kept
        identical to the factual world.

        Args:
            causal_engine: CausalEngine instance with time-localized methods
            factual_interest_matrix: Factual interest trajectory matrix
            intervention_day: Day at which the intervention occurs

        Returns:
            Dict mapping scenario name -> interest matrix
        """
        scenarios: Dict[str, np.ndarray] = {
            "factual": factual_interest_matrix,
        }

        # Scenario: No treatment from intervention_day onward (do(T=0))
        scenarios["no_treatment"] = (
            causal_engine.generate_time_localized_no_treatment_counterfactual(
                factual_interest_matrix, intervention_day
            )
        )

        # Scenario: Double treatment from intervention_day onward (do(T=2T))
        scenarios["double_treatment"] = (
            causal_engine.generate_time_localized_double_treatment_counterfactual(
                factual_interest_matrix, intervention_day
            )
        )

        # Scenario: Optimal targeted treatment (keeps global do-operator for now)
        scenarios["optimal_treatment"] = (
            causal_engine.generate_optimal_treatment_counterfactual(threshold=0.5)
        )

        # Scenario: Confounder shift (keeps global do-operator for now)
        scenarios["confounder_shift"] = (
            causal_engine.generate_confounder_intervention_counterfactual(shift=0.2)
        )

        return scenarios

    # ------------------------------------------------------------------
    # Cross-scenario comparison
    # ------------------------------------------------------------------

    def _compare_all_scenarios(self) -> Dict[str, Any]:
        """Compare graph metrics across all counterfactual worlds.

        Returns:
            Dict with per-world metrics and delta analysis
        """
        if "factual" not in self.scenario_graphs:
            return {"error": "No factual scenario found"}

        factual_metrics = self.scenario_graphs["factual"]["graph"]["metrics"]

        per_world: Dict[str, Dict] = {}
        deltas: Dict[str, Dict[str, float]] = {}

        for name, data in self.scenario_graphs.items():
            metrics = data["graph"]["metrics"]
            per_world[name] = metrics

            if name != "factual":
                deltas[name] = self._compute_delta(factual_metrics, metrics)

        return {
            "per_world": per_world,
            "deltas": deltas,
            "most_affected_users": self._find_highly_affected_users(),
        }

    @staticmethod
    def _compute_delta(
        factual: Dict[str, Any], counterfactual: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute percentage change between factual and counterfactual metrics.

        Args:
            factual: Factual graph metrics
            counterfactual: Counterfactual graph metrics

        Returns:
            Dict of metric -> percentage change
        """
        delta = {}
        for key in factual:
            f_val = factual[key]
            cf_val = counterfactual.get(key)
            if cf_val is None:
                continue
            f_num = float(f_val)
            cf_num = float(cf_val)
            if f_num != 0:
                delta[f"delta_{key}"] = round((cf_num - f_num) / f_num * 100, 2)
            else:
                delta[f"delta_{key}"] = 0.0
        return delta

    def _find_highly_affected_users(
        self, threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """Find users whose graph structure changes most under counterfactuals.

        Compares edge counts per user between factual and each counterfactual
        scenario to identify the most affected individuals.

        Args:
            threshold: Minimum relative change to be considered "highly affected"

        Returns:
            List of {user_id, scenario, delta_edges, delta_percent} dicts
        """
        if "factual" not in self.scenario_graphs:
            return []

        factual_graphs = self.scenario_graphs["factual"]["graph"]["graphs"]
        affected = []

        for scenario_name, data in self.scenario_graphs.items():
            if scenario_name == "factual":
                continue

            cf_graphs = data["graph"]["graphs"]

            for uid in factual_graphs:
                f_edges = factual_graphs[uid].get("n_edges", 0)
                cf_data = cf_graphs.get(uid)
                if cf_data is None:
                    continue
                cf_edges = cf_data.get("n_edges", 0)

                if f_edges > 0:
                    delta_pct = (cf_edges - f_edges) / f_edges
                    if abs(delta_pct) >= threshold:
                        affected.append(
                            {
                                "user_id": uid,
                                "scenario": scenario_name,
                                "factual_edges": f_edges,
                                "counterfactual_edges": cf_edges,
                                "delta_edges": cf_edges - f_edges,
                                "delta_percent": round(delta_pct * 100, 2),
                            }
                        )

        # Sort by absolute delta percent descending
        affected.sort(key=lambda x: abs(x["delta_percent"]), reverse=True)
        return affected

    # ------------------------------------------------------------------
    # Causal pathway discovery
    # ------------------------------------------------------------------

    def _discover_causal_pathways(self) -> List[Dict[str, Any]]:
        """Discover which causal pathways are most impactful.

        Analyzes how each counterfactual intervention changes graph structure
        and attributes changes to specific causal mechanisms.

        Returns:
            List of pathway descriptions with effect sizes
        """
        if "factual" not in self.scenario_graphs:
            return []

        pathways = []
        factual_metrics = self.scenario_graphs["factual"]["graph"]["metrics"]

        # Define pathway interpretations for each scenario
        pathway_definitions = {
            "no_treatment": {
                "name": "ad_removal",
                "label": "Ad Removal Effect",
                "description": "Impact of removing all advertising on user engagement graph",
                "expected_sign": -1,  # Negative delta expected
            },
            "double_treatment": {
                "name": "ad_doubling",
                "label": "Ad Doubling Effect",
                "description": "Impact of doubling ad exposure on user engagement graph",
                "expected_sign": 1,  # Positive delta expected
            },
            "optimal_treatment": {
                "name": "targeted_treatment",
                "label": "Targeted Treatment Effect",
                "description": "Impact of targeting only high-affinity users with boosted ads",
                "expected_sign": 1,
            },
            "confounder_shift": {
                "name": "affinity_shift",
                "label": "Base Affinity Shift Effect",
                "description": "Impact of increasing all users' base category affinity",
                "expected_sign": 1,
            },
        }

        for scenario_name, data in self.scenario_graphs.items():
            if scenario_name == "factual":
                continue

            definition = pathway_definitions.get(scenario_name, {})
            cf_metrics = data["graph"]["metrics"]

            pathway = {
                "scenario": scenario_name,
                "pathway_name": definition.get("name", scenario_name),
                "label": definition.get(
                    "label", scenario_name.replace("_", " ").title()
                ),
                "description": definition.get(
                    "description", f"Effect of {scenario_name} intervention"
                ),
                "effect_sizes": {},
            }

            # Compute effect sizes for key metrics
            for metric_key in ["total_edges", "edge_density", "avg_degree"]:
                f_val = float(factual_metrics.get(metric_key, 0))
                cf_val = float(cf_metrics.get(metric_key, 0))
                if f_val != 0:
                    pathway["effect_sizes"][metric_key] = {
                        "factual": f_val,
                        "counterfactual": cf_val,
                        "absolute_change": round(cf_val - f_val, 4),
                        "relative_change_pct": round((cf_val - f_val) / f_val * 100, 2),
                    }

            # Interpret the effect
            pathway["interpretation"] = self._interpret_pathway(pathway, definition)

            pathways.append(pathway)

        # Sort by absolute effect size (most impactful first)
        pathways.sort(
            key=lambda p: abs(
                p["effect_sizes"].get("total_edges", {}).get("relative_change_pct", 0)
            ),
            reverse=True,
        )

        return pathways

    @staticmethod
    def _interpret_pathway(pathway: Dict, definition: Dict) -> str:
        """Generate a human-readable interpretation of a causal pathway.

        Args:
            pathway: Pathway data with effect_sizes
            definition: Pathway definition with expected_sign

        Returns:
            Interpretive string
        """
        edges_effect = pathway["effect_sizes"].get("total_edges", {})
        rel_change = edges_effect.get("relative_change_pct", 0)
        expected_sign = definition.get("expected_sign", 0)

        if abs(rel_change) < 1.0:
            return (
                f"{pathway['label']}: Negligible impact on graph structure "
                f"({rel_change:+.2f}% edge change)."
            )

        direction = "increases" if rel_change > 0 else "decreases"
        magnitude = (
            "strong"
            if abs(rel_change) > 20
            else "moderate"
            if abs(rel_change) > 5
            else "weak"
        )

        consistent = (rel_change > 0 and expected_sign > 0) or (
            rel_change < 0 and expected_sign < 0
        )

        consistency_note = (
            " (consistent with expected direction)"
            if consistent
            else " (opposite to expected direction)"
        )

        return (
            f"{pathway['label']}: {magnitude.title()} {direction} "
            f"user engagement graph by {abs(rel_change):.1f}%{consistency_note}."
        )

    # ------------------------------------------------------------------
    # Event generation from interest matrices
    # ------------------------------------------------------------------

    def _interest_matrix_to_events(
        self,
        interest_matrix: np.ndarray,
        users: List,
        scenario_name: str = "factual",
    ) -> List[Dict]:
        """Convert an interest matrix to synthetic events for graph construction.

        This bridges Layer 1 (interest trajectories) to Layer 3 (event graphs)
        by generating timestamped events whose frequency is proportional to
        interest levels.

        Args:
            interest_matrix: Array of shape (n_users, n_days, n_categories)
            users: List of User objects
            scenario_name: Label for the scenario (for event metadata)

        Returns:
            List of event dicts suitable for GraphStitcher.build_interaction_graphs()
        """
        events = []
        n_users, n_days, n_cats = interest_matrix.shape

        for user_idx in range(n_users):
            user = users[user_idx]
            for day in range(n_days):
                for cat_idx in range(n_cats):
                    interest = interest_matrix[user_idx, day, cat_idx]

                    # Generate events proportional to interest level
                    # Higher interest → more events
                    n_events = np.random.poisson(interest * 3)
                    if n_events == 0 and interest > 0.3:
                        n_events = 1  # At least one event for moderate interest

                    for e in range(n_events):
                        hour = np.random.randint(0, 24)
                        minute = np.random.randint(0, 60)
                        second = np.random.randint(0, 60)

                        timestamp = (
                            f"2024-01-{day % 30 + 1:02d}T"
                            f"{hour:02d}:{minute:02d}:{second:02d}Z"
                        )

                        event = {
                            "event_id": f"{scenario_name}_{user_idx}_{day}_{cat_idx}_{e}",
                            "user_id": user.user_id,
                            "device_id": (
                                user.devices[0].device_id
                                if user.devices
                                else f"device_{user_idx}"
                            ),
                            "timestamp": timestamp,
                            "service": self._category_to_service(cat_idx),
                            "event_type": "interest_event",
                            "category": cat_idx,
                            "interest_score": float(interest),
                            "scenario": scenario_name,
                        }
                        events.append(event)

        return events

    @staticmethod
    def _category_to_service(cat_idx: int) -> str:
        """Map category index to a service name for event diversity."""
        services = [
            "search",
            "commerce",
            "media",
            "social",
            "email",
            "geo",
            "network",
            "urban",
            "news",
            "entertainment",
        ]
        return services[cat_idx % len(services)]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_counterfactual_report(self, report_data: Dict, output_path: str):
        """Export the full counterfactual analysis report as JSON.

        Args:
            report_data: Output from generate_all_scenario_graphs()
            output_path: Path to output JSON file
        """
        # Convert numpy types for JSON serialization
        cleaned = json.loads(json.dumps(report_data, default=self._json_serialize))

        with open(output_path, "w") as f:
            json.dump(cleaned, f, indent=2)

    @staticmethod
    def _json_serialize(obj):
        """Serialize numpy types for JSON export."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
