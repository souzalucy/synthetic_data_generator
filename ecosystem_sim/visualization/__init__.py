"""Visualization module for ecosystem data simulator outputs.

Provides interactive HTML and static figure generation for:
- Network graphs (interaction graphs, counterfactual worlds, device stitching)
- Time-series plots (causal trajectories, interest curves)
- Distribution plots (lift deciles, propensity scores, cluster sizes)
- Report dashboards (counterfactual comparison, ground truth summaries)
"""

from .graph_viz import (
    plot_interaction_graphs,
    plot_counterfactual_comparison,
    plot_device_stitching_graph,
    render_graph_tab,
)
from .timeseries_viz import (
    plot_causal_trajectories,
    plot_interest_heatmap,
    plot_treatment_matrix,
    plot_persona_trajectories,
    render_timeseries_tab,
)
from .distribution_viz import (
    plot_lift_deciles,
    plot_qini_curve,
    plot_propensity_distributions,
    plot_cluster_size_distribution,
    plot_price_sensitivity,
    render_distribution_tab,
)
from .report_viz import (
    plot_counterfactual_deltas,
    plot_ground_truth_summary,
    plot_causal_pathways,
    render_report_tab,
)
from .dashboard import (
    generate_all_visualizations,
    run_dashboard,
)

__all__ = [
    # Graph visualizations
    "plot_interaction_graphs",
    "plot_counterfactual_comparison",
    "plot_device_stitching_graph",
    "render_graph_tab",
    # Time-series visualizations
    "plot_causal_trajectories",
    "plot_interest_heatmap",
    "plot_treatment_matrix",
    "plot_persona_trajectories",
    "render_timeseries_tab",
    # Distribution visualizations
    "plot_lift_deciles",
    "plot_qini_curve",
    "plot_propensity_distributions",
    "plot_cluster_size_distribution",
    "plot_price_sensitivity",
    "render_distribution_tab",
    # Report visualizations
    "plot_counterfactual_deltas",
    "plot_ground_truth_summary",
    "plot_causal_pathways",
    "render_report_tab",
    # Dashboard
    "generate_all_visualizations",
    "run_dashboard",
]
