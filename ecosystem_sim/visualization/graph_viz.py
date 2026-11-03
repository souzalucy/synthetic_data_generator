"""Network graph visualizations for interaction graphs, counterfactual worlds,
and device stitching clusters.

Uses Plotly for interactive HTML output and NetworkX for graph layout algorithms.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import networkx as nx
import plotly.graph_objects as go

from .utils import (
    load_json,
    get_service_color,
    get_scenario_color,
    get_metric_label,
    ensure_output_dir,
    extract_scenario_metrics,
    extract_deltas,
    extract_most_affected_users,
)
from ecosystem_sim.intelligence.explanation_agent import VisualizationExplainer


# ── Interaction Graph Visualizations ────────────────────────────────────────

# Edge line width is quantised into this many Plotly traces. One trace per edge
# is a rendering trap: a dense interaction graph (tens of thousands of edges)
# produced HTML files of ~90 MB and took minutes to serialise.
N_EDGE_BUCKETS = 4

# Upper bound on edges drawn per figure (highest-weight edges win). Keeps the
# exported HTML readable and small on very dense graphs.
MAX_EDGES_PER_FIGURE = 2000


def _build_nx_graph_from_edges(edges: List[Dict]) -> nx.Graph:
    """Build a NetworkX graph from an edge list.

    Args:
        edges: List of edge dicts with 'source', 'target', 'weight'

    Returns:
        NetworkX Graph
    """
    G = nx.Graph()
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        weight = edge.get("weight", 0.5)
        G.add_edge(source, target, weight=weight)
    return G


def _create_network_figure(
    G: nx.Graph,
    title: str,
    node_colors: Optional[Dict[str, str]] = None,
    show_labels: bool = False,
    layout_seed: int = 42,
) -> go.Figure:
    """Create a Plotly network figure from a NetworkX graph.

    Args:
        G: NetworkX graph
        title: Plot title
        node_colors: Optional dict mapping node_id -> color string
        show_labels: Whether to show node labels
        layout_seed: Random seed for layout reproducibility

    Returns:
        Plotly Figure
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[
                dict(text="No nodes to display", showarrow=False, x=0.5, y=0.5)
            ],
        )
        return fig

    # Compute layout (large graphs get fewer iterations to keep rendering fast)
    iterations = 50 if G.number_of_nodes() <= 500 else 30
    pos = nx.spring_layout(G, seed=layout_seed, k=0.3, iterations=iterations)

    # Extract edge traces -- one trace per line-width bucket, with gaps encoded
    # as ``None`` so every edge still renders as a separate segment.
    edge_data = list(G.edges(data=True))
    edge_weights = [data.get("weight", 0.5) for _, _, data in edge_data]
    max_weight = max(edge_weights) if edge_weights else 1.0
    max_weight = max_weight or 1.0

    buckets: Dict[int, Dict[str, list]] = {
        index: {"x": [], "y": [], "text": []} for index in range(N_EDGE_BUCKETS)
    }
    for (u, v, _data), weight in zip(edge_data, edge_weights):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # Line width proportional to weight
        index = min(int(weight / max_weight * N_EDGE_BUCKETS), N_EDGE_BUCKETS - 1)
        bucket = buckets[index]
        bucket["x"].extend((x0, x1, None))
        bucket["y"].extend((y0, y1, None))
        label = f"{u} -> {v}<br>Weight: {weight:.3f}"
        bucket["text"].extend((label, label, None))

    edge_traces = [
        go.Scatter(
            x=bucket["x"],
            y=bucket["y"],
            mode="lines",
            line=dict(
                width=max(0.5, (index + 1) / N_EDGE_BUCKETS * 4),
                color="rgba(150,150,150,0.6)",
            ),
            hoverinfo="text",
            text=bucket["text"],
            showlegend=False,
        )
        for index, bucket in buckets.items()
        if bucket["x"]
    ]

    # Extract node trace
    node_x = []
    node_y = []
    node_text = []
    node_color_list = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if node_colors and node in node_colors:
            node_color_list.append(node_colors[node])
        else:
            node_color_list.append("#636EFA")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers" + ("+text" if show_labels else ""),
        marker=dict(
            size=10,
            color=node_color_list,
            line=dict(width=1, color="rgba(0,0,0,0.3)"),
        ),
        text=node_text if show_labels else None,
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        width=800,
        height=600,
    )
    return fig


def plot_interaction_graphs(
    graph_data: Dict,
    output_dir: str,
    max_users: int = 10,
    explainer: Optional["VisualizationExplainer"] = None,
    max_edges: int = MAX_EDGES_PER_FIGURE,
) -> List[str]:
    """Render per-user interaction graphs as Plotly network diagrams.

    Args:
        graph_data: Loaded interaction_graphs.json dict
        output_dir: Output directory for HTML files
        max_users: Maximum number of user graphs to render
        max_edges: Maximum edges drawn per graph (highest weights win)

    Returns:
        List of generated HTML file paths
    """
    viz_dir = ensure_output_dir(output_dir, "graphs")
    generated_files = []

    users_data = graph_data.get("users", {})
    metrics = graph_data.get("metrics", {})

    # Generate summary metrics page
    summary_path = viz_dir / "interaction_graphs_summary.html"
    _create_graph_summary_page(metrics, users_data, summary_path)
    generated_files.append(str(summary_path))

    # Generate per-user graphs
    rendered = 0
    for uid, graph in users_data.items():
        if rendered >= max_users:
            break
        # ``unidentified`` is an aggregate bucket for events whose user could not
        # be inferred, not a user graph -- it can hold hundreds of thousands of
        # edges, so it is not worth rendering.
        if uid == "unidentified":
            continue

        edges = graph.get("edges", [])
        if not edges:
            continue

        total_edges = graph.get("n_edges", len(edges))
        if max_edges and len(edges) > max_edges:
            edges = sorted(
                edges, key=lambda edge: edge.get("weight", 0.5), reverse=True
            )[:max_edges]

        G = _build_nx_graph_from_edges(edges)

        # Color nodes by service
        service_map = {}
        for edge in edges:
            src_service = edge.get("source_service", "")
            tgt_service = edge.get("target_service", "")
            if src_service:
                service_map[edge["source"]] = get_service_color(src_service)
            if tgt_service:
                service_map[edge["target"]] = get_service_color(tgt_service)

        shown_edges = len(edges)
        title = (
            f"Interaction Graph: User {uid} "
            f"({graph.get('n_events', 0)} events, {total_edges} edges)"
        )
        if shown_edges < total_edges:
            title += f"<br><sub>top {shown_edges} edges by weight</sub>"

        fig = _create_network_figure(
            G,
            title=title,
            node_colors=service_map,
            show_labels=False,
        )

        filepath = viz_dir / f"user_{uid}_graph.html"
        fig.write_html(str(filepath))
        if explainer:
            html = Path(filepath).read_text(encoding="utf-8")
            explanation = explainer.explain(
                "interaction_graphs",
                {
                    "user_id": uid,
                    "graph": graph,
                    "edges": edges,
                    "n_edges": total_edges,
                    "n_edges_shown": shown_edges,
                    "n_events": graph.get("n_events", 0),
                },
            )
            html = VisualizationExplainer.inject_explanation_button(
                html, explanation, f"User {uid} Interaction Graph"
            )
            Path(filepath).write_text(html, encoding="utf-8")
        generated_files.append(str(filepath))
        rendered += 1

    return generated_files


def _create_graph_summary_page(metrics: Dict, users_data: Dict, output_path: Path):
    """Create a summary page with graph-level metrics as a bar chart.

    Args:
        metrics: Graph-level metrics dict
        users_data: Per-user graph data
        output_path: Output HTML path
    """
    # Aggregate metrics bar chart
    metric_keys = [
        "total_events",
        "total_edges",
        "n_users_with_graphs",
        "avg_degree",
        "avg_events_per_user",
        "avg_edges_per_user",
    ]
    labels = [get_metric_label(k) for k in metric_keys if k in metrics]
    values = [metrics.get(k, 0) for k in metric_keys if k in metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#636EFA",
            text=[f"{v:,.1f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Interaction Graph Summary Metrics",
        title_x=0.5,
        yaxis_title="Value",
        plot_bgcolor="rgba(0,0,0,0)",
        width=900,
        height=500,
    )

    # Add per-user edge count histogram as subplot
    edge_counts = [g.get("n_edges", 0) for g in users_data.values()]
    if edge_counts:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=edge_counts,
                nbinsx=min(30, len(set(edge_counts))),
                marker_color="#EF553B",
                name="Edge Count Distribution",
            )
        )
        fig_hist.update_layout(
            title="Per-User Edge Count Distribution",
            title_x=0.5,
            xaxis_title="Number of Edges",
            yaxis_title="Number of Users",
            plot_bgcolor="rgba(0,0,0,0)",
            width=600,
            height=400,
        )

        # Combine into one HTML with both figures
        combined_html = (
            fig.to_html(full_html=False, include_plotlyjs="cdn")
            + "<hr>"
            + fig_hist.to_html(full_html=False, include_plotlyjs=False)
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                "<html><head><title>Interaction Graphs Summary</title></head><body>"
                + combined_html
                + "</body></html>"
            )
    else:
        fig.write_html(str(output_path))


# ── Counterfactual Graph Comparison ─────────────────────────────────────────


def plot_counterfactual_comparison(
    cf_report: Dict,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
) -> List[str]:
    """Render multi-scenario counterfactual comparison visualizations.

    Generates:
    1. Delta bar chart comparing metrics across scenarios
    2. Per-scenario metric comparison table as heatmap
    3. Most affected users table

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict
        output_dir: Output directory for HTML files

    Returns:
        List of generated HTML file paths
    """
    viz_dir = ensure_output_dir(output_dir, "counterfactual")
    generated_files = []

    scenarios = extract_scenario_metrics(cf_report)
    deltas = extract_deltas(cf_report)

    if not scenarios:
        return generated_files

    # 1. Delta bar chart
    delta_path = viz_dir / "counterfactual_deltas.html"
    _plot_delta_comparison(deltas, delta_path)
    generated_files.append(str(delta_path))

    # 2. Per-scenario metric heatmap
    heatmap_path = viz_dir / "counterfactual_metrics_heatmap.html"
    _plot_metrics_heatmap(scenarios, heatmap_path)
    generated_files.append(str(heatmap_path))

    # 3. Most affected users
    affected = extract_most_affected_users(cf_report)
    if affected:
        affected_path = viz_dir / "most_affected_users.html"
        _plot_affected_users(affected, affected_path)
        generated_files.append(str(affected_path))

    if explainer and generated_files:
        # Inject explanation into the first generated file (delta comparison)
        first = generated_files[0]
        html = Path(first).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "counterfactual_deltas",
            {
                "cf_report": cf_report,
                "deltas": deltas,
                "scenarios": list(
                    cf_report.get("scenarios", cf_report.get("worlds", {}).keys())
                ),
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Counterfactual Comparison"
        )
        Path(first).write_text(html, encoding="utf-8")

    return generated_files


def _plot_delta_comparison(
    deltas: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Create a grouped bar chart of metric deltas across scenarios.

    Args:
        deltas: Dict mapping scenario -> {metric: delta_percent}
        output_path: Output HTML path
    """
    if not deltas:
        fig = go.Figure()
        fig.update_layout(title="No Delta Data Available")
        fig.write_html(str(output_path))
        return

    # Collect all metric keys across scenarios
    all_metrics = set()
    for scenario_deltas in deltas.values():
        all_metrics.update(scenario_deltas.keys())
    all_metrics = sorted(all_metrics)

    fig = go.Figure()
    for scenario_name, scenario_deltas in deltas.items():
        values = [scenario_deltas.get(m, 0) for m in all_metrics]
        fig.add_trace(
            go.Bar(
                name=scenario_name.replace("_", " ").title(),
                x=[get_metric_label(m.replace("delta_", "")) for m in all_metrics],
                y=values,
                marker_color=get_scenario_color(scenario_name),
                text=[f"{v:+.1f}%" for v in values],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Graph Metric Deltas vs Factual (Percentage Change)",
        title_x=0.5,
        barmode="group",
        yaxis_title="Delta (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=1000,
        height=500,
        legend_title="Scenario",
    )
    fig.write_html(str(output_path))


def _plot_metrics_heatmap(
    scenarios: Dict[str, Dict],
    output_path: Path,
):
    """Create a heatmap of metrics across scenarios.

    Args:
        scenarios: Dict mapping scenario -> {metrics: {...}}
        output_path: Output HTML path
    """
    metric_keys = [
        "total_edges",
        "edge_density",
        "avg_degree",
        "total_events",
        "avg_events_per_user",
        "avg_edges_per_user",
    ]
    scenario_names = list(scenarios.keys())

    # Build matrix: rows = scenarios, cols = metrics
    matrix = []
    for sname in scenario_names:
        metrics = scenarios[sname].get("metrics", {})
        row = [metrics.get(k, 0) for k in metric_keys]
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[get_metric_label(k) for k in metric_keys],
            y=[s.replace("_", " ").title() for s in scenario_names],
            colorscale="Viridis",
            hoverongaps=False,
            text=[[f"{v:.4f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        title="Metric Values Across Counterfactual Scenarios",
        title_x=0.5,
        width=900,
        height=400 + len(scenario_names) * 30,
        xaxis_title="Metric",
        yaxis_title="Scenario",
    )
    fig.write_html(str(output_path))


def _plot_affected_users(
    affected_users: List[Dict],
    output_path: Path,
):
    """Create a bar chart of most affected users.

    Args:
        affected_users: List of {user_id, scenario, delta_percent, ...}
        output_path: Output HTML path
    """
    if not affected_users:
        return

    # Limit to top 20
    top_users = affected_users[:20]

    fig = go.Figure()
    for scenario in set(u["scenario"] for u in top_users):
        scenario_users = [u for u in top_users if u["scenario"] == scenario]
        fig.add_trace(
            go.Bar(
                name=scenario.replace("_", " ").title(),
                x=[u["user_id"] for u in scenario_users],
                y=[u["delta_percent"] for u in scenario_users],
                marker_color=get_scenario_color(scenario),
                text=[f"{u['delta_percent']:+.1f}%" for u in scenario_users],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Most Graph-Affected Users Under Counterfactuals",
        title_x=0.5,
        barmode="group",
        yaxis_title="Edge Count Change (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=1000,
        height=500,
    )
    fig.write_html(str(output_path))


# ── Device Stitching Graph ──────────────────────────────────────────────────


def plot_device_stitching_graph(
    stitched_data: Dict,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
    accuracy_data_path: Optional[str] = None,
) -> str:
    """Visualize ground-truth user→device mapping as a bipartite graph.

    Reads the ground-truth device mapping (``user_to_devices``) from
    *stitched_data* and renders a single deterministic graph showing
    which devices belong to each user.

    Args:
        stitched_data: Loaded ground_truth_device_mapping.json dict
        output_dir: Output directory for HTML file
        explainer: Optional VisualizationExplainer for explanation injection
        accuracy_data_path: Ignored (kept for backward compatibility).

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "stitching")

    # Extract ground-truth clusters from the mapping
    user_to_devices = stitched_data.get("user_to_devices", {})
    n_users = stitched_data.get("n_users", 0)
    n_devices = stitched_data.get("n_devices", 0)

    if not user_to_devices:
        fig = go.Figure()
        fig.update_layout(title="No Ground-Truth Device Mapping Available")
        filepath = viz_dir / "device_stitching.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # ── Build ground-truth graph ───────────────────────────────────────────
    G = nx.Graph()
    colors = {}

    for user_id, devices in user_to_devices.items():
        node_id = f"user_{user_id}"
        G.add_node(node_id, type="user", size=20)
        colors[node_id] = "#00CC96"
        for device in devices:
            G.add_node(device, type="device", size=8)
            colors[device] = "#AB63FA"
            G.add_edge(node_id, device, weight=1.0)

    fig = _create_network_figure(
        G,
        title=f"Ground-Truth User->Device Mapping ({n_users} users, {n_devices} devices)",
        node_colors=colors,
        show_labels=False,
        layout_seed=42,
    )

    # Add summary annotation
    cluster_sizes = [len(devices) for devices in user_to_devices.values()]
    size_text = (
        f"Users: {n_users}<br>"
        f"Devices: {n_devices}<br>"
        f"Avg devices/user: {np.mean(cluster_sizes):.1f}<br>"
        f"Max devices/user: {max(cluster_sizes)}<br>"
        f"Min devices/user: {min(cluster_sizes)}"
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=size_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
    )

    # ── Write output ───────────────────────────────────────────────────────
    filepath = viz_dir / "device_stitching.html"
    fig.write_html(str(filepath))
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "device_stitching",
            {
                "user_to_devices": user_to_devices,
                "n_users": n_users,
                "n_devices": n_devices,
                "is_ground_truth": True,
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Ground-Truth Device Mapping"
        )
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Streamlit Tab Renderer ──────────────────────────────────────────────────


def render_graph_tab(output_dir: str):
    """Render the Graph Visualizations tab for Streamlit dashboard.

    Args:
        output_dir: Simulation output directory
    """
    # Lazy import streamlit only when this function is called
    import streamlit as st

    st.header("[chart] Network Graph Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interaction Graphs")
        graph_path = Path(output_dir) / "interaction_graphs.json"
        if graph_path.exists():
            data = load_json(str(graph_path))
            metrics = data.get("metrics", {})
            st.metric("Total Users with Graphs", metrics.get("n_users_with_graphs", 0))
            st.metric("Total Edges", metrics.get("total_edges", 0))
            st.metric("Avg Degree", f"{metrics.get('avg_degree', 0):.3f}")

            # Show summary iframe
            summary_path = (
                Path(output_dir) / "viz/graphs/interaction_graphs_summary.html"
            )
            if summary_path.exists():
                with open(summary_path, encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
        else:
            st.info("No interaction graphs found. Run simulation first.")

    with col2:
        st.subheader("Device Stitching")
        stitch_path = Path(output_dir) / "stitched_data.json"
        if stitch_path.exists():
            data = load_json(str(stitch_path))
            metrics = data.get("metrics", {})
            st.metric("Devices Stitched", metrics.get("n_devices", 0))
            st.metric("Clusters Formed", metrics.get("n_clusters", 0))
            st.metric("Avg Cluster Size", f"{metrics.get('avg_cluster_size', 0):.1f}")

            stitch_viz = Path(output_dir) / "viz/stitching/device_stitching.html"
            if stitch_viz.exists():
                with open(stitch_viz, encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
        else:
            st.info("No stitching data found.")

    st.subheader("Counterfactual Graph Comparison")
    cf_path = Path(output_dir) / "counterfactual_graph_report.json"
    if cf_path.exists():
        data = load_json(str(cf_path))
        deltas_path = Path(output_dir) / "viz/counterfactual/counterfactual_deltas.html"
        if deltas_path.exists():
            with open(deltas_path, encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=550, scrolling=True)

        # Show causal pathways
        pathways = data.get("causal_pathways", [])
        if pathways:
            st.subheader("Causal Pathways")
            for pw in pathways:
                with st.expander(pw.get("label", pw.get("pathway_name", "Unknown"))):
                    st.write(pw.get("interpretation", ""))
                    effects = pw.get("effect_sizes", {})
                    for metric, vals in effects.items():
                        st.metric(
                            get_metric_label(metric),
                            f"{vals.get('counterfactual', 0):.4f}",
                            delta=f"{vals.get('relative_change_pct', 0):+.2f}%",
                        )
    else:
        st.info("No counterfactual graph report found.")
