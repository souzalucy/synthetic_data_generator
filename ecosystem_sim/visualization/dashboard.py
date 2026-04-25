"""Streamlit integrated dashboard and visualization orchestrator.

Provides:
- `generate_all_visualizations()`: Standalone HTML export of all visualizations
- `run_dashboard()`: Interactive Streamlit dashboard
- `render_dashboard()`: Full-page Streamlit layout with tabs
"""

import json
import webbrowser
from pathlib import Path
from typing import Dict, Optional

from ecosystem_sim.visualization.utils import (
    ensure_output_dir,
    load_json,
    load_csv_to_dicts,
)
from ecosystem_sim.visualization.graph_viz import (
    plot_interaction_graphs,
    plot_counterfactual_comparison,
    plot_device_stitching_graph,
    render_graph_tab,
)
from ecosystem_sim.visualization.timeseries_viz import (
    plot_causal_trajectories,
    plot_persona_trajectories,
    plot_interest_heatmap,
    plot_treatment_matrix,
    render_timeseries_tab,
)
from ecosystem_sim.visualization.distribution_viz import (
    plot_lift_deciles,
    plot_qini_curve,
    plot_lift_summary,
    plot_propensity_distributions,
    plot_cluster_size_distribution,
    plot_price_sensitivity,
    render_distribution_tab,
)
from ecosystem_sim.visualization.report_viz import (
    plot_counterfactual_deltas,
    plot_ground_truth_summary,
    plot_causal_pathways,
    render_report_tab,
)


# ---------------------------------------------------------------------------
# Orchestrator: generate all visualizations as standalone HTML files
# ---------------------------------------------------------------------------


def generate_all_visualizations(
    output_dir: str,
    ground_truth_path: Optional[str] = None,
    causal_csv_path: Optional[str] = None,
    interaction_graphs_path: Optional[str] = None,
    stitched_data_path: Optional[str] = None,
    lift_report_path: Optional[str] = None,
    counterfactual_report_path: Optional[str] = None,
) -> Dict[str, str]:
    """Generate every visualization as a standalone HTML file.

    Each function auto-discovers its input file from *output_dir* when the
    corresponding path argument is ``None``.

    Args:
        output_dir: Root simulation output directory (e.g. ``ecosystem_sim/outputs``).
        ground_truth_path: Path to ``ground_truth.json``.
        causal_csv_path: Path to ``causal_ground_truth.csv``.
        interaction_graphs_path: Path to ``interaction_graphs.json``.
        stitched_data_path: Path to ``stitched_data.json``.
        lift_report_path: Path to ``lift_report.json``.
        counterfactual_report_path: Path to ``counterfactual_graph_report.json``.

    Returns:
        Dictionary mapping logical keys to the generated HTML file paths.
    """
    out = Path(output_dir)
    ensure_output_dir(str(out / "viz"))

    # Resolve default paths -------------------------------------------------
    gt_path = ground_truth_path or str(out / "ground_truth.json")
    csv_path = causal_csv_path or str(out / "causal_ground_truth.csv")
    ig_path = interaction_graphs_path or str(out / "interaction_graphs.json")
    st_path = stitched_data_path or str(out / "stitched_data.json")
    lr_path = lift_report_path or str(out / "lift_report.json")
    cf_path = counterfactual_report_path or str(
        out / "counterfactual_graph_report.json"
    )

    generated: Dict[str, str] = {}

    # 1. Graph visualizations -----------------------------------------------
    print("\n[VIZ] Generating graph visualizations...")

    ig_data = load_json(ig_path)
    if ig_data:
        files = plot_interaction_graphs(ig_data, str(out / "viz"))
        if files:
            generated["interaction_graphs"] = files[0]  # summary page
        print(f"  * Interaction graphs: {len(files)} pages generated")

    st_data = load_json(st_path)
    if st_data:
        filepath = plot_device_stitching_graph(st_data, str(out / "viz"))
        generated["device_stitching"] = filepath
        print(f"  * Device stitching graph: {filepath}")

    cf_data = load_json(cf_path)
    if cf_data:
        files = plot_counterfactual_comparison(cf_data, str(out / "viz"))
        if files:
            generated["counterfactual_comparison"] = files[0]
        print(f"  * Counterfactual comparison: {len(files)} pages generated")

    # 2. Time-series visualizations -----------------------------------------
    print("\n[VIZ] Generating time-series visualizations...")

    if Path(csv_path).exists():
        result = plot_causal_trajectories(csv_path, str(out / "viz"))
        generated["causal_trajectories"] = result
        print(f"  * Causal trajectories: {result}")

        result = plot_persona_trajectories(csv_path, str(out / "viz"))
        generated["persona_trajectories"] = result
        print(f"  * Persona trajectories: {result}")

    gt_data = load_json(gt_path)
    if gt_data:
        interest_matrix = gt_data.get(
            "interest_matrix", gt_data.get("interest_trajectories")
        )
        if interest_matrix:
            result = plot_interest_heatmap(interest_matrix, gt_path, str(out / "viz"))
            generated["interest_heatmap"] = result
            print(f"  * Interest heatmap: {result}")

        result = plot_treatment_matrix(gt_path, str(out / "viz"))
        generated["treatment_matrix"] = result
        print(f"  * Treatment matrix: {result}")

    # 3. Distribution visualizations ----------------------------------------
    print("\n[VIZ] Generating distribution visualizations...")

    lr_data = load_json(lr_path)
    if lr_data:
        result = plot_lift_deciles(lr_data, str(out / "viz"))
        generated["lift_deciles"] = result
        print(f"  * Lift deciles: {result}")

        result = plot_qini_curve(lr_data, str(out / "viz"))
        generated["qini_curve"] = result
        print(f"  * Qini curve: {result}")

        result = plot_lift_summary(lr_data, str(out / "viz"))
        generated["lift_summary"] = result
        print(f"  * Lift summary: {result}")

    if gt_data:
        result = plot_propensity_distributions(gt_path, str(out / "viz"))
        generated["propensity_distributions"] = result
        print(f"  * Propensity distributions: {result}")

        result = plot_price_sensitivity(gt_path, str(out / "viz"))
        generated["price_sensitivity"] = result
        print(f"  * Price sensitivity: {result}")

    if st_data:
        result = plot_cluster_size_distribution(st_data, str(out / "viz"))
        generated["cluster_sizes"] = result
        print(f"  * Cluster size distribution: {result}")

    # 4. Report visualizations ----------------------------------------------
    print("\n[VIZ] Generating report visualizations...")

    if cf_data:
        result = plot_counterfactual_deltas(cf_data, str(out / "viz"))
        generated["counterfactual_deltas"] = result
        print(f"  * Counterfactual deltas: {result}")

        result = plot_causal_pathways(cf_data, str(out / "viz"))
        generated["causal_pathways"] = result
        print(f"  * Causal pathways: {result}")

    if gt_data:
        result = plot_ground_truth_summary(gt_path, str(out / "viz"))
        generated["ground_truth_summary"] = result
        print(f"  * Ground truth summary: {result}")

    # 5. Generate index page ------------------------------------------------
    index_path = str(out / "viz" / "index.html")
    _generate_index_page(generated, index_path)
    generated["index"] = index_path
    print(f"\n[VIZ] Index page: {index_path}")

    print("[VIZ] All visualizations generated successfully.")
    return generated


def _generate_index_page(generated: Dict[str, str], output_path: str):
    """Create a simple HTML index linking to all generated visualizations."""
    sections = {
        "Graph Visualizations": [
            ("Interaction Graphs", generated.get("interaction_graphs")),
            ("Device Stitching", generated.get("device_stitching")),
            ("Counterfactual Comparison", generated.get("counterfactual_comparison")),
        ],
        "Time-Series Visualizations": [
            ("Causal Trajectories", generated.get("causal_trajectories")),
            ("Persona Trajectories", generated.get("persona_trajectories")),
            ("Interest Heatmap", generated.get("interest_heatmap")),
            ("Treatment Matrix", generated.get("treatment_matrix")),
        ],
        "Distribution Visualizations": [
            ("Lift Deciles", generated.get("lift_deciles")),
            ("Qini Curve", generated.get("qini_curve")),
            ("Lift Summary", generated.get("lift_summary")),
            ("Propensity Distributions", generated.get("propensity_distributions")),
            ("Price Sensitivity", generated.get("price_sensitivity")),
            ("Cluster Size Distribution", generated.get("cluster_sizes")),
        ],
        "Report Visualizations": [
            ("Counterfactual Deltas", generated.get("counterfactual_deltas")),
            ("Causal Pathways", generated.get("causal_pathways")),
            ("Ground Truth Summary", generated.get("ground_truth_summary")),
        ],
    }

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "  <title>Ecosystem Data Simulator - Visualization Index</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
        "           max-width: 960px; margin: 0 auto; padding: 2rem; background: #f5f5f5; }",
        "    h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 0.5rem; }",
        "    h2 { color: #16213e; margin-top: 2rem; }",
        "    ul { list-style: none; padding: 0; }",
        "    li { margin: 0.5rem 0; }",
        "    a { display: block; padding: 0.75rem 1rem; background: white; border-radius: 8px;",
        "       color: #0f3460; text-decoration: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1);",
        "       transition: transform 0.1s, box-shadow 0.1s; }",
        "    a:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(0,0,0,0.15); }",
        "    a.disabled { opacity: 0.4; pointer-events: none; }",
        "    .badge { font-size: 0.75rem; background: #e94560; color: white;",
        "             padding: 0.15rem 0.5rem; border-radius: 10px; margin-left: 0.5rem; }",
        "    .footer { margin-top: 3rem; font-size: 0.85rem; color: #666; text-align: center; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>📊 Ecosystem Data Simulator</h1>",
        "  <p>Visualization index for simulation outputs.</p>",
    ]

    for section_title, links in sections.items():
        html_parts.append(f"  <h2>{section_title}</h2>")
        html_parts.append("  <ul>")
        for label, path in links:
            if path and Path(path).exists():
                rel = Path(path).name
                html_parts.append(
                    f'    <li><a href="{rel}" target="_blank">{label} <span class="badge">HTML</span></a></li>'
                )
            else:
                html_parts.append(
                    f'    <li><a class="disabled">{label} <span class="badge">N/A</span></a></li>'
                )
        html_parts.append("  </ul>")

    html_parts.extend(
        [
            "  <div class='footer'>",
            "    Generated by ecosystem_sim.visualization.dashboard",
            "  </div>",
            "</body>",
            "</html>",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


# ---------------------------------------------------------------------------
# Streamlit interactive dashboard
# ---------------------------------------------------------------------------


def run_dashboard(output_dir: str, open_browser: bool = True):
    """Launch the Streamlit dashboard for a given simulation output directory.

    Args:
        output_dir: Path to the simulation output directory.
        open_browser: Whether to automatically open the browser.
    """
    import streamlit as st

    st.set_page_config(
        page_title="Ecosystem Data Simulator - Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_dashboard(output_dir)

    if open_browser:
        # Streamlit auto-opens by default; this is a safety fallback
        webbrowser.open("http://localhost:8501")


def render_dashboard(output_dir: str):
    """Render the full Streamlit dashboard layout.

    This function can be called directly from a ``streamlit run`` entry point
    or imported and composed into a larger Streamlit app.

    Args:
        output_dir: Path to the simulation output directory.
    """
    import streamlit as st

    out = Path(output_dir)

    # Sidebar ---------------------------------------------------------------
    st.sidebar.title("📊 Dashboard")
    st.sidebar.markdown("**Simulation Outputs**")
    st.sidebar.markdown(f"`{output_dir}`")

    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Output Files**")
    files = {
        "Ground Truth": out / "ground_truth.json",
        "Causal CSV": out / "causal_ground_truth.csv",
        "Interaction Graphs": out / "interaction_graphs.json",
        "Stitched Data": out / "stitched_data.json",
        "Lift Report": out / "lift_report.json",
        "Counterfactual Report": out / "counterfactual_graph_report.json",
    }
    for label, fpath in files.items():
        exists = fpath.exists()
        icon = "✅" if exists else "❌"
        st.sidebar.markdown(f"{icon} **{label}**")
        if exists:
            size_kb = fpath.stat().st_size / 1024
            st.sidebar.caption(f"  {size_kb:.1f} KB")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with Plotly • Streamlit • NetworkX",
        help="Technology stack used for visualizations",
    )

    # Main area -------------------------------------------------------------
    st.title("📊 Ecosystem Data Simulator")
    st.markdown(
        "Interactive dashboard for exploring simulation outputs. "
        "Each tab provides a different analytical perspective on the generated data."
    )

    tab_names = [
        "📈 Time Series",
        "🌐 Graphs",
        "📊 Distributions",
        "📋 Reports",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_timeseries_tab(output_dir)

    with tabs[1]:
        render_graph_tab(output_dir)

    with tabs[2]:
        render_distribution_tab(output_dir)

    with tabs[3]:
        render_report_tab(output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point: ``python -m ecosystem_sim.visualization.dashboard``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ecosystem Data Simulator - Visualization Dashboard"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ecosystem_sim/outputs",
        help="Simulation output directory",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate standalone HTML files without launching the dashboard",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open browser when launching dashboard",
    )
    args = parser.parse_args()

    if args.generate_only:
        generate_all_visualizations(args.output_dir)
    else:
        run_dashboard(args.output_dir, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
