"""Distribution visualizations for lift analysis, propensity scores,
device cluster statistics, and price sensitivity analysis.

Uses Plotly for interactive histograms, box plots, bar charts, and Qini curves.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    load_json,
    get_persona_color,
    ensure_output_dir,
    extract_decile_data,
)


# ── Lift Analysis Visualizations ────────────────────────────────────────────


def plot_lift_deciles(
    lift_report: Dict,
    output_dir: str,
) -> str:
    """Bar chart of average uplift by decile with error bars.

    Args:
        lift_report: Loaded lift_report.json dict
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/distributions")
    deciles = extract_decile_data(lift_report)

    if not deciles:
        fig = go.Figure()
        fig.update_layout(title="No Lift Decile Data Available")
        filepath = viz_dir / "lift_deciles.html"
        fig.write_html(str(filepath))
        return str(filepath)

    df = pd.DataFrame(deciles)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["decile"],
            y=df["avg_uplift"],
            error_y=dict(
                type="data",
                array=df["std_uplift"],
                visible=True,
                color="rgba(0,0,0,0.5)",
            ),
            marker=dict(
                color=df["avg_uplift"],
                colorscale="RdYlGn",
                cmin=min(0, df["avg_uplift"].min()),
                cmax=max(0, df["avg_uplift"].max()),
                showscale=True,
                colorbar=dict(title="Avg Uplift"),
            ),
            text=df["avg_uplift"].round(4),
            textposition="outside",
            hovertemplate=(
                "Decile %{x}<br>"
                "Avg Uplift: %{y:.4f}<br>"
                "Std: %{error_y.array:.4f}<br>"
                "Users: %{customdata[0]}<br>"
                "Cumulative: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
            customdata=df[["user_count", "cumulative_uplift"]],
        )
    )

    fig.update_layout(
        title="Campaign Uplift by Decile",
        title_x=0.5,
        xaxis=dict(
            title="Decile",
            dtick=1,
            tickmode="linear",
        ),
        yaxis_title="Average Uplift",
        plot_bgcolor="rgba(0,0,0,0)",
        width=800,
        height=500,
    )

    filepath = viz_dir / "lift_deciles.html"
    fig.write_html(str(filepath))
    return str(filepath)


def plot_qini_curve(
    lift_report: Dict,
    output_dir: str,
) -> str:
    """Qini curve showing model performance vs. random targeting.

    Reconstructs the cumulative uplift curve from decile data and
    compares it against the random targeting baseline.

    Args:
        lift_report: Loaded lift_report.json dict
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/distributions")
    deciles = extract_decile_data(lift_report)

    if not deciles:
        fig = go.Figure()
        fig.update_layout(title="No Qini Curve Data Available")
        filepath = viz_dir / "qini_curve.html"
        fig.write_html(str(filepath))
        return str(filepath)

    df = pd.DataFrame(deciles)

    # Compute cumulative uplift
    df["cumulative_uplift_model"] = df["cumulative_uplift"].cumsum()
    total_uplift = df["cumulative_uplift"].sum()
    n_deciles = len(df)
    df["cumulative_uplift_random"] = (
        np.arange(1, n_deciles + 1) / n_deciles
    ) * total_uplift

    # Qini = model - random
    df["qini"] = df["cumulative_uplift_model"] - df["cumulative_uplift_random"]

    # Population percentage
    df["population_pct"] = (np.arange(1, n_deciles + 1) / n_deciles) * 100

    fig = go.Figure()

    # Model curve
    fig.add_trace(
        go.Scatter(
            x=df["population_pct"],
            y=df["cumulative_uplift_model"],
            mode="lines+markers",
            name="Model (Targeted)",
            line=dict(color="#636EFA", width=3),
            marker=dict(size=8),
        )
    )

    # Random baseline
    fig.add_trace(
        go.Scatter(
            x=df["population_pct"],
            y=df["cumulative_uplift_random"],
            mode="lines+markers",
            name="Random Targeting",
            line=dict(color="#EF553B", width=3, dash="dash"),
            marker=dict(size=8),
        )
    )

    # Qini area (shaded)
    fig.add_trace(
        go.Scatter(
            x=list(df["population_pct"]) + list(df["population_pct"][::-1]),
            y=list(df["cumulative_uplift_model"])
            + list(df["cumulative_uplift_random"][::-1]),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Qini Area",
            showlegend=True,
        )
    )

    # Add Qini coefficient annotation
    qini_coef = lift_report.get("qini_coefficient", 0)
    auuc = lift_report.get("auuc", 0)

    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            f"Qini Coefficient: {qini_coef:.4f}<br>"
            f"AUUC: {auuc:.4f}<br>"
            f"Lift: {lift_report.get('lift_percent', 0):.1f}%"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
    )

    fig.update_layout(
        title="Qini Curve: Model vs. Random Targeting",
        title_x=0.5,
        xaxis_title="Population Targeted (%)",
        yaxis_title="Cumulative Uplift",
        plot_bgcolor="rgba(0,0,0,0)",
        width=800,
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    filepath = viz_dir / "qini_curve.html"
    fig.write_html(str(filepath))
    return str(filepath)


def plot_lift_summary(
    lift_report: Dict,
    output_dir: str,
) -> str:
    """Create a lift summary dashboard with key metrics and benchmark comparison.

    Args:
        lift_report: Loaded lift_report.json dict
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/distributions")

    lift_pct = lift_report.get("lift_percent", 0)
    p_value = lift_report.get("p_value", 1.0)
    auuc = lift_report.get("auuc", 0)
    qini = lift_report.get("qini_coefficient", 0)
    benchmark = lift_report.get("criteo_benchmark_comparison", {})

    # Gauge chart for lift
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Campaign Lift (%)",
            "Statistical Significance",
            "AUUC & Qini Coefficient",
            "Benchmark Comparison",
        ),
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Lift gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=lift_pct,
            delta=dict(
                reference=benchmark.get("benchmark_average", 18) * 100,
                increasing=dict(color="#00CC96"),
                decreasing=dict(color="#EF553B"),
            ),
            gauge=dict(
                axis=dict(range=[0, max(50, lift_pct * 1.5)]),
                bar=dict(color="#636EFA"),
                steps=[
                    dict(range=[0, 10], color="rgba(239, 85, 59, 0.2)"),
                    dict(range=[10, 20], color="rgba(255, 161, 90, 0.2)"),
                    dict(range=[20, 50], color="rgba(0, 204, 150, 0.2)"),
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=benchmark.get("benchmark_average", 18) * 100,
                ),
            ),
            title=dict(text="Lift %"),
        ),
        row=1,
        col=1,
    )

    # P-value indicator
    significant = p_value < 0.05
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=round(p_value, 4),
            number=dict(
                suffix="",
                font=dict(color="#00CC96" if significant else "#EF553B", size=24),
            ),
            gauge=dict(
                axis=dict(range=[0, 1], visible=True),
                bar=dict(color="#00CC96" if significant else "#EF553B"),
                steps=[
                    dict(range=[0, 0.05], color="rgba(0, 204, 150, 0.3)"),
                    dict(range=[0.05, 1], color="rgba(239, 85, 59, 0.2)"),
                ],
            ),
            title=dict(
                text=f"p-value<br>{'✅ Significant' if significant else '❌ Not Significant'}"
            ),
        ),
        row=1,
        col=2,
    )

    # AUUC & Qini bar chart
    fig.add_trace(
        go.Bar(
            x=["AUUC", "Qini Coefficient"],
            y=[auuc, qini],
            marker_color=["#636EFA", "#AB63FA"],
            text=[f"{auuc:.4f}", f"{qini:.4f}"],
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    # Benchmark comparison
    bench_diff = benchmark.get("difference_from_benchmark", 0) * 100
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=benchmark.get("observed_uplift", 0) * 100,
            number=dict(suffix="%", font=dict(size=28)),
            delta=dict(
                reference=benchmark.get("benchmark_average", 18) * 100,
                increasing=dict(color="#00CC96"),
                decreasing=dict(color="#EF553B"),
            ),
            title=dict(text="Observed vs Benchmark"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Campaign Lift Analysis Dashboard",
        title_x=0.5,
        height=600,
        width=1000,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    filepath = viz_dir / "lift_summary.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Propensity Score Visualizations ─────────────────────────────────────────


def plot_propensity_distributions(
    ground_truth_path: str,
    output_dir: str,
) -> str:
    """Visualize propensity-related distributions from ground truth data.

    Generates histograms of treatment effects and base affinities
    broken down by persona.

    Args:
        ground_truth_path: Path to ground_truth.json
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/distributions")
    gt = load_json(ground_truth_path)

    if not gt:
        fig = go.Figure()
        fig.update_layout(title="No Ground Truth Data")
        filepath = viz_dir / "propensity_distributions.html"
        fig.write_html(str(filepath))
        return str(filepath)

    users = gt.get("users", [])
    if not users:
        filepath = viz_dir / "propensity_distributions.html"
        go.Figure().update_layout(title="No Users").write_html(str(filepath))
        return str(filepath)

    df = pd.DataFrame(users)

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Treatment Effect by Persona",
            "Base Affinity by Persona",
            "Treatment Effect Distribution",
            "Base Affinity Distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Box plots by persona
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
            col=1,
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
            col=2,
        )

    # Overall histograms
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

    fig.update_layout(
        title="Propensity & Treatment Effect Distributions",
        title_x=0.5,
        height=700,
        width=1000,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    filepath = viz_dir / "propensity_distributions.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Cluster Size Distribution ───────────────────────────────────────────────


def plot_cluster_size_distribution(
    stitched_data: Dict,
    output_dir: str,
) -> str:
    """Visualize device cluster size distribution.

    Args:
        stitched_data: Loaded stitched_data.json dict
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/distributions")
    clusters = stitched_data.get("clusters", {})
    metrics = stitched_data.get("metrics", {})

    if not clusters:
        fig = go.Figure()
        fig.update_layout(title="No Cluster Data Available")
        filepath = viz_dir / "cluster_size_distribution.html"
        fig.write_html(str(filepath))
        return str(filepath)

    cluster_sizes = [len(devices) for devices in clusters.values()]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cluster Size Distribution", "Cluster Size Box Plot"),
        specs=[[{"type": "bar"}, {"type": "box"}]],
    )

    # Histogram
    size_counts = pd.Series(cluster_sizes).value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=size_counts.index,
            y=size_counts.values,
            marker_color="#636EFA",
            text=size_counts.values,
            textposition="outside",
            name="Cluster Sizes",
        ),
        row=1,
        col=1,
    )

    # Box plot
    fig.add_trace(
        go.Box(
            y=cluster_sizes,
            name="Cluster Size",
            boxmean=True,
            marker_color="#EF553B",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Device Cluster Size Distribution "
        f"({metrics.get('n_clusters', 0)} clusters, "
        f"{metrics.get('n_devices', 0)} devices)",
        title_x=0.5,
        height=450,
        width=900,
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Cluster Size", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Devices per Cluster", row=1, col=2)

    filepath = viz_dir / "cluster_size_distribution.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Price Sensitivity Visualizations ────────────────────────────────────────


def plot_price_sensitivity(
    ground_truth_path: str,
    output_dir: str,
) -> str:
    """Visualize price sensitivity metrics from ground truth data.

    Since price sensitivity is computed per-persona, this generates
    persona-level comparison charts.

    Args:
        ground_truth_path: Path to ground_truth.json
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/distributions")
    gt = load_json(ground_truth_path)

    if not gt:
        fig = go.Figure()
        fig.update_layout(title="No Ground Truth Data")
        filepath = viz_dir / "price_sensitivity.html"
        fig.write_html(str(filepath))
        return str(filepath)

    users = gt.get("users", [])
    if not users:
        filepath = viz_dir / "price_sensitivity.html"
        go.Figure().update_layout(title="No Users").write_html(str(filepath))
        return str(filepath)

    df = pd.DataFrame(users)

    # Aggregate by persona
    persona_stats = (
        df.groupby("persona")
        .agg(
            {
                "treatment_effect": ["mean", "std", "count"],
                "base_affinity": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten columns
    persona_stats.columns = [
        "persona",
        "te_mean",
        "te_std",
        "count",
        "ba_mean",
        "ba_std",
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Avg Treatment Effect by Persona",
            "Avg Base Affinity by Persona",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    colors = [get_persona_color(p) for p in persona_stats["persona"]]

    fig.add_trace(
        go.Bar(
            x=persona_stats["persona"],
            y=persona_stats["te_mean"],
            error_y=dict(
                type="data",
                array=persona_stats["te_std"],
                visible=True,
            ),
            marker_color=colors,
            text=persona_stats["te_mean"].round(3),
            textposition="outside",
            name="Treatment Effect",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=persona_stats["persona"],
            y=persona_stats["ba_mean"],
            error_y=dict(
                type="data",
                array=persona_stats["ba_std"],
                visible=True,
            ),
            marker_color=colors,
            text=persona_stats["ba_mean"].round(3),
            textposition="outside",
            name="Base Affinity",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Persona-Level Metrics (Proxy for Price Sensitivity Analysis)",
        title_x=0.5,
        height=450,
        width=1000,
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    filepath = viz_dir / "price_sensitivity.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Streamlit Tab Renderer ──────────────────────────────────────────────────


def render_distribution_tab(output_dir: str):
    """Render the Distribution Visualizations tab for Streamlit dashboard.

    Args:
        output_dir: Simulation output directory
    """
    import streamlit as st

    st.header("📊 Distribution Visualizations")

    lift_path = Path(output_dir) / "lift_report.json"
    gt_path = Path(output_dir) / "ground_truth.json"
    stitch_path = Path(output_dir) / "stitched_data.json"

    tab1, tab2, tab3 = st.tabs(["Lift Analysis", "Propensity Scores", "Cluster Stats"])

    with tab1:
        st.subheader("Lift Analysis")
        if lift_path.exists():
            data = load_json(str(lift_path))
            col1, col2, col3 = st.columns(3)
            col1.metric("Campaign Lift", f"{data.get('lift_percent', 0):.1f}%")
            col2.metric("AUUC", f"{data.get('auuc', 0):.4f}")
            col3.metric("Qini Coefficient", f"{data.get('qini_coefficient', 0):.4f}")

            # Show decile chart
            decile_path = Path(output_dir) / "viz/distributions/lift_deciles.html"
            if decile_path.exists():
                with open(decile_path) as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)

            # Show Qini curve
            qini_path = Path(output_dir) / "viz/distributions/qini_curve.html"
            if qini_path.exists():
                with open(qini_path) as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)

            # Show summary dashboard
            summary_path = Path(output_dir) / "viz/distributions/lift_summary.html"
            if summary_path.exists():
                with open(summary_path) as f:
                    st.components.v1.html(f.read(), height=600, scrolling=True)
        else:
            st.info("No lift report found.")

    with tab2:
        st.subheader("Propensity Score Distributions")
        prop_path = Path(output_dir) / "viz/distributions/propensity_distributions.html"
        if prop_path.exists():
            with open(prop_path) as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        else:
            st.info("No propensity distributions available.")

        price_path = Path(output_dir) / "viz/distributions/price_sensitivity.html"
        if price_path.exists():
            with open(price_path) as f:
                st.components.v1.html(f.read(), height=450, scrolling=True)

    with tab3:
        st.subheader("Device Cluster Statistics")
        if stitch_path.exists():
            data = load_json(str(stitch_path))
            metrics = data.get("metrics", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Devices", metrics.get("n_devices", 0))
            col2.metric("Clusters", metrics.get("n_clusters", 0))
            col3.metric("Avg Size", f"{metrics.get('avg_cluster_size', 0):.1f}")

            cluster_path = (
                Path(output_dir) / "viz/distributions/cluster_size_distribution.html"
            )
            if cluster_path.exists():
                with open(cluster_path) as f:
                    st.components.v1.html(f.read(), height=450, scrolling=True)
        else:
            st.info("No stitching data found.")
