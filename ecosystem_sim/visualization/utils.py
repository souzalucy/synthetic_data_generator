"""Shared utility functions for visualization module."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


# ── Color palettes ──────────────────────────────────────────────────────────

PERSONA_COLORS = {
    "Tech-Savvy Millennial": "#636EFA",
    "Budget-Conscious Parent": "#EF553B",
    "Business Professional": "#00CC96",
    "Casual Browser": "#AB63FA",
    "Privacy-Focused User": "#FFA15A",
}

SERVICE_COLORS = {
    "search": "#636EFA",
    "commerce": "#EF553B",
    "geo": "#00CC96",
    "media": "#AB63FA",
    "social": "#FFA15A",
    "email": "#19D3F3",
    "network": "#FF6692",
    "urban": "#B6E880",
    "news": "#FF97FF",
    "entertainment": "#FECB52",
}

SCENARIO_COLORS = {
    "factual": "#636EFA",
    "no_treatment": "#EF553B",
    "double_treatment": "#00CC96",
    "optimal_treatment": "#AB63FA",
    "confounder_shift": "#FFA15A",
}

METRIC_LABELS = {
    "total_edges": "Total Edges",
    "edge_density": "Edge Density",
    "avg_degree": "Avg Degree",
    "total_events": "Total Events",
    "n_users_with_graphs": "Users with Graphs",
    "avg_events_per_user": "Avg Events/User",
    "avg_edges_per_user": "Avg Edges/User",
}


# ── File loading helpers ────────────────────────────────────────────────────


def load_json(filepath: str) -> Dict:
    """Load a JSON file and return its contents.

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON dictionary (empty dict if file not found)
    """
    path = Path(filepath)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_csv_to_dicts(filepath: str) -> List[Dict[str, Any]]:
    """Load a CSV file as a list of dictionaries.

    Args:
        filepath: Path to CSV file

    Returns:
        List of row dicts (empty list if file not found)
    """
    path = Path(filepath)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_jsonl(filepath: str, max_lines: Optional[int] = None) -> List[Dict]:
    """Load a JSONL file as a list of dictionaries.

    Args:
        filepath: Path to JSONL file
        max_lines: Maximum number of lines to read (None = all)

    Returns:
        List of event dicts
    """
    path = Path(filepath)
    if not path.exists():
        return []
    events = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ── Data transformation helpers ─────────────────────────────────────────────


def melt_causal_csv(csv_rows: List[Dict]) -> List[Dict]:
    """Melt wide-format causal CSV to long format for plotting.

    Transforms columns day_0_base_affinity ... day_N_base_affinity
    into (day, affinity) rows.

    Args:
        csv_rows: List of row dicts from causal_ground_truth.csv

    Returns:
        List of dicts with keys: user_id, persona, treatment_effect, day, affinity
    """
    melted = []
    for row in csv_rows:
        user_id = row["user_id"]
        persona = row["persona"]
        treatment_effect = float(row.get("treatment_effect", 0))
        for key, val in row.items():
            if key.startswith("day_") and key.endswith("_base_affinity"):
                day = int(key.split("_")[1])
                affinity = float(val) if val else 0.0
                melted.append(
                    {
                        "user_id": user_id,
                        "persona": persona,
                        "treatment_effect": treatment_effect,
                        "day": day,
                        "affinity": affinity,
                    }
                )
    return melted


def melt_causal_counterfactual_csv(csv_rows: List[Dict]) -> List[Dict]:
    """Melt wide-format causal counterfactual CSV to long format for plotting.

    Transforms dual columns:
      day_X_base_affinity_factual       -> (day, affinity, scenario='factual')
      day_X_base_affinity_counterfactual -> (day, affinity, scenario='counterfactual')

    Args:
        csv_rows: List of row dicts from causal_table_with_counterfactual.csv

    Returns:
        List of dicts with keys: user_id, persona, treatment_effect,
                                 intervention_type, intervention_value,
                                 day, affinity, scenario
    """
    melted = []
    for row in csv_rows:
        user_id = row["user_id"]
        persona = row["persona"]
        treatment_effect = float(row.get("treatment_effect", 0))
        intervention_type = row.get("intervention_type", "")
        intervention_value = row.get("intervention_value", "")

        for key, val in row.items():
            if key.startswith("day_") and key.endswith("_base_affinity_factual"):
                day = int(key.split("_")[1])
                affinity = float(val) if val else 0.0
                melted.append(
                    {
                        "user_id": user_id,
                        "persona": persona,
                        "treatment_effect": treatment_effect,
                        "intervention_type": intervention_type,
                        "intervention_value": intervention_value,
                        "day": day,
                        "affinity": affinity,
                        "scenario": "factual",
                    }
                )
            elif key.startswith("day_") and key.endswith(
                "_base_affinity_counterfactual"
            ):
                day = int(key.split("_")[1])
                affinity = float(val) if val else 0.0
                melted.append(
                    {
                        "user_id": user_id,
                        "persona": persona,
                        "treatment_effect": treatment_effect,
                        "intervention_type": intervention_type,
                        "intervention_value": intervention_value,
                        "day": day,
                        "affinity": affinity,
                        "scenario": "counterfactual",
                    }
                )
    return melted


def extract_decile_data(lift_report: Dict) -> List[Dict]:
    """Extract and validate decile data from lift report.

    Args:
        lift_report: Loaded lift_report.json dict

    Returns:
        List of decile dicts with keys: decile, avg_uplift, std_uplift, etc.
    """
    return lift_report.get("uplift_by_decile", [])


def extract_scenario_metrics(cf_report: Dict) -> Dict[str, Dict]:
    """Extract per-scenario metrics from counterfactual report.

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict

    Returns:
        Dict mapping scenario name -> metrics dict
    """
    return cf_report.get("scenarios", {})


def extract_deltas(cf_report: Dict) -> Dict[str, Dict[str, float]]:
    """Extract delta comparisons from counterfactual report.

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict

    Returns:
        Dict mapping scenario name -> {metric: delta_percent}
    """
    comparison = cf_report.get("comparison", {})
    return comparison.get("deltas", {})


def extract_causal_pathways(cf_report: Dict) -> List[Dict]:
    """Extract causal pathway interpretations.

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict

    Returns:
        List of pathway dicts
    """
    return cf_report.get("causal_pathways", [])


def extract_most_affected_users(cf_report: Dict) -> List[Dict]:
    """Extract most affected users from counterfactual comparison.

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict

    Returns:
        List of affected user dicts
    """
    comparison = cf_report.get("comparison", {})
    return comparison.get("most_affected_users", [])


# ── Color helpers ───────────────────────────────────────────────────────────


def get_persona_color(persona_name: str) -> str:
    """Get a consistent color for a persona name.

    Args:
        persona_name: Persona display name

    Returns:
        Hex color string
    """
    return PERSONA_COLORS.get(persona_name, "#999999")


def get_service_color(service_name: str) -> str:
    """Get a consistent color for a service name.

    Args:
        service_name: Service name

    Returns:
        Hex color string
    """
    return SERVICE_COLORS.get(service_name, "#999999")


def get_scenario_color(scenario_name: str) -> str:
    """Get a consistent color for a counterfactual scenario.

    Args:
        scenario_name: Scenario name

    Returns:
        Hex color string
    """
    return SCENARIO_COLORS.get(scenario_name, "#999999")


def get_metric_label(metric_key: str) -> str:
    """Get a human-readable label for a metric key.

    Args:
        metric_key: Internal metric key (e.g. 'total_edges')

    Returns:
        Human-readable label
    """
    return METRIC_LABELS.get(metric_key, metric_key.replace("_", " ").title())


# ── Output path helpers ─────────────────────────────────────────────────────


def ensure_output_dir(base_dir: str, subdir: str = "viz") -> Path:
    """Ensure a visualization output subdirectory exists.

    Args:
        base_dir: Base output directory
        subdir: Subdirectory name for visualizations

    Returns:
        Path to the visualization output directory
    """
    viz_dir = Path(base_dir) / subdir
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir
