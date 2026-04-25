"""Report-style visualizations for counterfactual analysis, ground truth summaries,
and causal pathway discovery.

Generates comprehensive HTML dashboards with tables, gauges, and annotated charts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    load_json,
    get_persona_color,
    get_scenario_color,
    get_metric_label,
    ensure_output_dir,
    extract_scenario_metrics,
    extract_deltas,
    extract_causal_pathways,
    extract_most_affected_users,
)
from ecosystem_sim.intelligence.explanation_agent import VisualizationExplainer


# ── Counterfactual Delta Visualizations ─────────────────────────────────────


def plot_counterfactual_deltas(
    cf_report: Dict,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Comprehensive counterfactual delta dashboard.

    Generates a multi-panel dashboard with:
    1. Delta bar chart (all metrics, all scenarios)
    2. Per-world metric comparison table
    3. Scenario summary cards

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/reports")
    scenarios = extract_scenario_metrics(cf_report)
    deltas = extract_deltas(cf_report)
    pathways = extract_causal_pathways(cf_report)

    if not scenarios:
        fig = go.Figure()
        fig.update_layout(title="No Counterfactual Data Available")
        filepath = viz_dir / "counterfactual_dashboard.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # Build multi-panel figure
    n_pathways = len(pathways)
    n_rows = 2 + n_pathways

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=(
            "Metric Deltas vs Factual (%)",
            "Absolute Metric Values",
            "Edge Count Comparison",
            "Density Comparison",
            *(
                f"Causal Pathway: {p.get('label', p.get('pathway_name', 'Unknown'))}"
                for p in pathways
            ),
        ),
        specs=(
            [[{"type": "bar"}, {"type": "heatmap"}]]
            + [[{"type": "bar"}, {"type": "bar"}]]
            + [[{"type": "bar"}, {"type": "indicator"}] for _ in range(n_pathways)]
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.12,
    )

    # ── Panel 1: Delta bar chart ──
    if deltas:
        all_metrics = sorted(set(m for sd in deltas.values() for m in sd.keys()))
        metric_labels = [get_metric_label(m.replace("delta_", "")) for m in all_metrics]

        for s_idx, (sname, sdeltas) in enumerate(deltas.items()):
            values = [sdeltas.get(m, 0) for m in all_metrics]
            fig.add_trace(
                go.Bar(
                    name=sname.replace("_", " ").title(),
                    x=metric_labels,
                    y=values,
                    marker_color=get_scenario_color(sname),
                    text=[f"{v:+.1f}%" for v in values],
                    textposition="outside",
                    textfont=dict(size=9),
                    showlegend=s_idx == 0,
                ),
                row=1,
                col=1,
            )

    # ── Panel 2: Absolute metric heatmap ──
    metric_keys = [
        "total_edges",
        "edge_density",
        "avg_degree",
        "total_events",
        "avg_events_per_user",
        "avg_edges_per_user",
    ]
    snames = list(scenarios.keys())
    matrix = []
    for sname in snames:
        metrics = scenarios[sname].get("metrics", {})
        row = [metrics.get(k, 0) for k in metric_keys]
        matrix.append(row)

    if matrix:
        matrix = np.array(matrix, dtype=float)
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[get_metric_label(k) for k in metric_keys],
                y=[s.replace("_", " ").title() for s in snames],
                colorscale="Viridis",
                text=[[f"{v:.4f}" for v in row] for row in matrix],
                texttemplate="%{text}",
                textfont=dict(size=9),
                hoverongaps=False,
                showscale=True,
            ),
            row=1,
            col=2,
        )

    # ── Panel 3: Edge count comparison ──
    for sname in snames:
        metrics = scenarios[sname].get("metrics", {})
        fig.add_trace(
            go.Bar(
                name=sname.replace("_", " ").title(),
                x=[sname.replace("_", " ").title()],
                y=[metrics.get("total_edges", 0)],
                marker_color=get_scenario_color(sname),
                text=[f"{metrics.get('total_edges', 0):,}"],
                textposition="auto",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # ── Panel 4: Density comparison ──
    for sname in snames:
        metrics = scenarios[sname].get("metrics", {})
        fig.add_trace(
            go.Bar(
                name=sname.replace("_", " ").title(),
                x=[sname.replace("_", " ").title()],
                y=[metrics.get("edge_density", 0)],
                marker_color=get_scenario_color(sname),
                text=[f"{metrics.get('edge_density', 0):.6f}"],
                textposition="auto",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # ── Panels 5+: Causal pathway effect sizes ──
    for p_idx, pathway in enumerate(pathways[:4]):
        row = 3 + p_idx
        effects = pathway.get("effect_sizes", {})

        # Effect size bar chart
        effect_metrics = list(effects.keys())
        effect_labels = [get_metric_label(m) for m in effect_metrics]
        rel_changes = [effects[m].get("relative_change_pct", 0) for m in effect_metrics]

        fig.add_trace(
            go.Bar(
                x=effect_labels,
                y=rel_changes,
                marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in rel_changes],
                text=[f"{v:+.2f}%" for v in rel_changes],
                textposition="outside",
                name=pathway.get("label", f"Pathway {p_idx + 1}"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Interpretation indicator
        interpretation = pathway.get("interpretation", "")
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=abs(rel_changes[0]) if rel_changes else 0,
                number=dict(
                    suffix="%",
                    font=dict(size=24, color="#636EFA"),
                ),
                title=dict(
                    text=f"<b>{pathway.get('label', 'Effect')}</b><br>"
                    f"<span style='font-size:12px'>{interpretation[:80]}...</span>"
                ),
            ),
            row=row,
            col=2,
        )

    fig.update_layout(
        title="Counterfactual Graph Analysis Dashboard",
        title_x=0.5,
        height=300 * n_rows,
        width=1200,
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        hovermode="x unified",
    )

    filepath = viz_dir / "counterfactual_dashboard.html"
    fig.write_html(str(filepath))
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        deltas = extract_deltas(cf_report)
        explanation = explainer.explain("counterfactual_deltas", {"scenarios": list(deltas.keys()) if deltas else []})
        html = VisualizationExplainer.inject_explanation_button(html, explanation, "Counterfactual Deltas")
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Ground Truth Summary ────────────────────────────────────────────────────


def plot_ground_truth_summary(
    ground_truth_path: str,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Create a comprehensive ground truth summary dashboard.

    Generates:
    1. Persona distribution pie chart
    2. Treatment effect histogram by persona
    3. Base affinity distribution
    4. Summary statistics table

    Args:
        ground_truth_path: Path to ground_truth.json
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/reports")
    gt = load_json(ground_truth_path)

    if not gt:
        fig = go.Figure()
        fig.update_layout(title="No Ground Truth Data Available")
        filepath = viz_dir / "ground_truth_summary.html"
        fig.write_html(str(filepath))
        return str(filepath)

    users = gt.get("users", [])
    config = gt.get("config", {})
    stats = gt.get("statistics", {})

    if not users:
        filepath = viz_dir / "ground_truth_summary.html"
        go.Figure().update_layout(title="No Users").write_html(str(filepath))
        return str(filepath)

    df = pd.DataFrame(users)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Persona Distribution",
            "Treatment Effect by Persona",
            "Base Affinity by Persona",
            "Treatment Effect Histogram",
            "Base Affinity Histogram",
            "Summary Statistics",
        ),
        specs=[
            [{"type": "pie"}, {"type": "box"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "domain"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Pie chart: Persona distribution ──
    persona_counts = df["persona"].value_counts()
    colors = [get_persona_color(p) for p in persona_counts.index]
    fig.add_trace(
        go.Pie(
            labels=persona_counts.index,
            values=persona_counts.values,
            marker=dict(colors=colors),
            textinfo="label+percent",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ── Box plots by persona ──
    personas = df["persona"].unique()
    for persona in personas:
        p_df = df[df["persona"] == persona]
        color = get_persona_color(persona)

        fig.add_trace(
            go.Box(
                y=p_df["treatment_effect"],
                name=persona,
                marker_color=color,
                boxmean=True,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Box(
                y=p_df["base_affinity"],
                name=persona,
                marker_color=color,
                boxmean=True,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # ── Histograms ──
    fig.add_trace(
        go.Histogram(
            x=df["treatment_effect"],
            nbinsx=30,
            marker_color="#636EFA",
            name="Treatment Effect",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df["base_affinity"],
            nbinsx=30,
            marker_color="#EF553B",
            name="Base Affinity",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # ── Summary statistics ──
    stat_text = (
        f"<b>Configuration</b><br>"
        f"Users: {config.get('n_users', 'N/A')}<br>"
        f"Days: {config.get('n_days', 'N/A')}<br>"
        f"Categories: {config.get('n_categories', 'N/A')}<br>"
        f"Persistence: {config.get('persistence_factor', 'N/A')}<br>"
        f"SCM: {'Yes' if config.get('use_scm', False) else 'No'}<br><br>"
        f"<b>Statistics</b><br>"
        f"Mean Interest: {stats.get('mean_interest', 0):.4f}<br>"
        f"Std Interest: {stats.get('std_interest', 0):.4f}<br>"
        f"Mean Treatment: {stats.get('mean_treatment', 0):.4f}<br>"
        f"Std Treatment: {stats.get('std_treatment', 0):.4f}<br>"
        f"Mean TE: {stats.get('mean_treatment_effect', df['treatment_effect'].mean()):.4f}"
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=config.get("n_users", 0),
            number=dict(font=dict(size=28)),
            title=dict(text=stat_text, font=dict(size=12)),
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        title="Ground Truth Summary Dashboard",
        title_x=0.5,
        height=700,
        width=1200,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    filepath = viz_dir / "ground_truth_summary.html"
    fig.write_html(str(filepath))
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain("ground_truth_summary", {"n_users": len(gt.get("users", [])), "n_days": gt.get("config", {}).get("n_days", 0)})
        html = VisualizationExplainer.inject_explanation_button(html, explanation, "Ground Truth Summary")
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Causal Pathway Visualizations ───────────────────────────────────────────


def plot_causal_pathways(
    cf_report: Dict,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Visualize discovered causal pathways with effect sizes and interpretations.

    Args:
        cf_report: Loaded counterfactual_graph_report.json dict
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/reports")
    pathways = extract_causal_pathways(cf_report)

    if not pathways:
        fig = go.Figure()
        fig.update_layout(title="No Causal Pathways Discovered")
        filepath = viz_dir / "causal_pathways.html"
        fig.write_html(str(filepath))
        return str(filepath)

    n_pathways = len(pathways)
    fig = make_subplots(
        rows=n_pathways,
        cols=2,
        subplot_titles=[
            p.get("label", p.get("pathway_name", f"Pathway {i + 1}"))
            for i, p in enumerate(pathways)
        ]
        + [""] * n_pathways,  # Second column titles
        specs=[[{"type": "bar"}, {"type": "indicator"}] for _ in range(n_pathways)],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )

    for idx, pathway in enumerate(pathways):
        row = idx + 1
        effects = pathway.get("effect_sizes", {})

        # Effect size bar chart
        effect_metrics = list(effects.keys())
        effect_labels = [get_metric_label(m) for m in effect_metrics]
        rel_changes = [effects[m].get("relative_change_pct", 0) for m in effect_metrics]

        colors = ["#00CC96" if v >= 0 else "#EF553B" for v in rel_changes]
        fig.add_trace(
            go.Bar(
                x=effect_labels,
                y=rel_changes,
                marker_color=colors,
                text=[f"{v:+.2f}%" for v in rel_changes],
                textposition="outside",
                name=pathway.get("label", f"Pathway {idx + 1}"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Interpretation indicator
        interpretation = pathway.get("interpretation", "No interpretation available.")
        primary_effect = effects.get("total_edges", {})
        primary_change = primary_effect.get("relative_change_pct", 0)

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=primary_effect.get("counterfactual", 0),
                number=dict(font=dict(size=20)),
                delta=dict(
                    reference=primary_effect.get("factual", 0),
                    relative=True,
                    valueformat=".1%",
                ),
                title=dict(
                    text=f"<b>{pathway.get('label', 'Effect')}</b><br>"
                    f"<span style='font-size:11px'>{interpretation}</span>"
                ),
            ),
            row=row,
            col=2,
        )

    fig.update_layout(
        title="Discovered Causal Pathways",
        title_x=0.5,
        height=250 * n_pathways,
        width=1100,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    filepath = viz_dir / "causal_pathways.html"
    fig.write_html(str(filepath))
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        pathways = extract_causal_pathways(cf_report)
        explanation = explainer.explain("causal_pathways", {"n_pathways": len(pathways)})
        html = VisualizationExplainer.inject_explanation_button(html, explanation, "Causal Pathways")
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Streamlit Tab Renderer ──────────────────────────────────────────────────


def render_report_tab(output_dir: str):
    """Render the Report Visualizations tab for Streamlit dashboard.

    Args:
        output_dir: Simulation output directory
    """
    import streamlit as st

    st.header("📋 Report Dashboards")

    gt_path = Path(output_dir) / "ground_truth.json"
    cf_path = Path(output_dir) / "counterfactual_graph_report.json"

    tab1, tab2 = st.tabs(["Ground Truth Summary", "Counterfactual Analysis"])

    with tab1:
        st.subheader("Ground Truth Summary Dashboard")
        summary_path = Path(output_dir) / "viz/reports/ground_truth_summary.html"
        if summary_path.exists():
            with open(summary_path) as f:
                st.components.v1.html(f.read(), height=750, scrolling=True)
        elif gt_path.exists():
            st.info("Summary not yet generated. Run visualization phase.")
        else:
            st.info("No ground truth data found.")

    with tab2:
        st.subheader("Counterfactual Analysis Dashboard")
        cf_dash_path = Path(output_dir) / "viz/reports/counterfactual_dashboard.html"
        if cf_dash_path.exists():
            with open(cf_dash_path) as f:
                st.components.v1.html(f.read(), height=900, scrolling=True)
        else:
            st.info("No counterfactual dashboard available.")

        st.subheader("Causal Pathways")
        pathways_path = Path(output_dir) / "viz/reports/causal_pathways.html"
        if pathways_path.exists():
            with open(pathways_path) as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

        # Show pathway interpretations as expandable cards
        if cf_path.exists():
            data = load_json(str(cf_path))
            pathways = data.get("causal_pathways", [])
            if pathways:
                st.subheader("Pathway Interpretations")
                for pw in pathways:
                    with st.expander(
                        pw.get("label", pw.get("pathway_name", "Unknown"))
                    ):
                        st.write(pw.get("interpretation", ""))
                        effects = pw.get("effect_sizes", {})
                        for metric, vals in effects.items():
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Factual", f"{vals.get('factual', 0):.4f}")
                            col2.metric(
                                "Counterfactual", f"{vals.get('counterfactual', 0):.4f}"
                            )
                            col3.metric(
                                "Change",
                                f"{vals.get('relative_change_pct', 0):+.2f}%",
                                delta=f"{vals.get('absolute_change', 0):+.4f}",
                            )
