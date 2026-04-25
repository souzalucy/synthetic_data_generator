"""Time-series visualizations for causal interest trajectories, treatment matrices,
and persona-level trend analysis.

Uses Plotly Express for interactive line charts, heatmaps, and area plots.
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
    load_csv_to_dicts,
    melt_causal_csv,
    get_persona_color,
    ensure_output_dir,
)


# ── Causal Trajectory Plots ─────────────────────────────────────────────────


def plot_causal_trajectories(
    csv_path: str,
    output_dir: str,
    max_users: int = 50,
) -> str:
    """Plot per-day base affinity trajectories from the causal CSV.

    Generates:
    1. All-user trajectories (sampled) colored by persona
    2. Mean trajectory per persona with confidence bands

    Args:
        csv_path: Path to causal_ground_truth.csv
        output_dir: Output directory for HTML files
        max_users: Maximum number of individual user lines to show

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/timeseries")
    csv_rows = load_csv_to_dicts(csv_path)

    if not csv_rows:
        fig = go.Figure()
        fig.update_layout(title="No Causal Trajectory Data Available")
        filepath = viz_dir / "causal_trajectories.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # Melt to long format
    melted = melt_causal_csv(csv_rows)
    df = pd.DataFrame(melted)

    # Create figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Individual Interest Trajectories (Sampled)",
            "Mean Interest Trajectory by Persona",
        ),
        vertical_spacing=0.15,
        shared_xaxes=True,
    )

    # ── Subplot 1: Individual trajectories (sampled) ──
    unique_users = df["user_id"].unique()
    if len(unique_users) > max_users:
        sampled_users = np.random.choice(unique_users, max_users, replace=False)
    else:
        sampled_users = unique_users

    df_sample = df[df["user_id"].isin(sampled_users)]

    for persona in df_sample["persona"].unique():
        persona_df = df_sample[df_sample["persona"] == persona]
        color = get_persona_color(persona)

        # Group by day, compute mean and std for this persona
        agg = persona_df.groupby("day")["affinity"].agg(["mean", "std"]).reset_index()

        fig.add_trace(
            go.Scatter(
                x=agg["day"],
                y=agg["mean"],
                mode="lines",
                name=persona,
                line=dict(color=color, width=2),
                legendgroup=persona,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # ── Subplot 2: Mean per persona with confidence bands ──
    for persona in df["persona"].unique():
        persona_df = df[df["persona"] == persona]
        color = get_persona_color(persona)
        agg = (
            persona_df.groupby("day")["affinity"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"])

        fig.add_trace(
            go.Scatter(
                x=agg["day"],
                y=agg["mean"],
                mode="lines",
                name=f"{persona} (mean)",
                line=dict(color=color, width=3),
                legendgroup=persona,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([agg["day"], agg["day"][::-1]]),
                y=pd.concat([agg["mean"] + agg["ci"], (agg["mean"] - agg["ci"])[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.2),
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{persona} (95% CI)",
                legendgroup=persona,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Causal Interest Trajectories",
        title_x=0.5,
        height=700,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="Base Affinity", row=1, col=1)
    fig.update_yaxes(title_text="Base Affinity", row=2, col=1)

    filepath = viz_dir / "causal_trajectories.html"
    fig.write_html(str(filepath))
    return str(filepath)


def plot_persona_trajectories(
    csv_path: str,
    output_dir: str,
) -> str:
    """Create faceted line plots, one subplot per persona.

    Args:
        csv_path: Path to causal_ground_truth.csv
        output_dir: Output directory for HTML files

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/timeseries")
    csv_rows = load_csv_to_dicts(csv_path)

    if not csv_rows:
        filepath = viz_dir / "persona_trajectories.html"
        go.Figure().update_layout(title="No Data").write_html(str(filepath))
        return str(filepath)

    melted = melt_causal_csv(csv_rows)
    df = pd.DataFrame(melted)

    personas = df["persona"].unique()
    n_personas = len(personas)
    n_cols = min(3, n_personas)
    n_rows = int(np.ceil(n_personas / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[p for p in personas],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for idx, persona in enumerate(personas):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        persona_df = df[df["persona"] == persona]
        color = get_persona_color(persona)

        # Plot individual user lines (thin, low opacity)
        for uid in persona_df["user_id"].unique()[:20]:  # Max 20 per persona
            user_df = persona_df[persona_df["user_id"] == uid]
            fig.add_trace(
                go.Scatter(
                    x=user_df["day"],
                    y=user_df["affinity"],
                    mode="lines",
                    line=dict(color=color, width=0.5),
                    opacity=0.3,
                    name=uid,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # Mean line (bold)
        agg = persona_df.groupby("day")["affinity"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=agg["day"],
                y=agg["affinity"],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{persona} (mean)",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Interest Trajectories by Persona (Faceted)",
        title_x=0.5,
        height=250 * n_rows,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Day", row=n_rows, col=1)
    fig.update_yaxes(title_text="Affinity")

    filepath = viz_dir / "persona_trajectories.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Interest Heatmap ────────────────────────────────────────────────────────


def plot_interest_heatmap(
    interest_matrix: Optional[np.ndarray] = None,
    ground_truth_path: Optional[str] = None,
    output_dir: str = ".",
) -> str:
    """Plot a heatmap of average interest across days and categories.

    Args:
        interest_matrix: Optional numpy array (n_users, n_days, n_categories)
        ground_truth_path: Optional path to ground_truth.json (loaded if no matrix)
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/timeseries")

    if interest_matrix is None and ground_truth_path:
        # Try to reconstruct from ground truth (limited info)
        gt = load_json(ground_truth_path)
        if gt:
            stats = gt.get("statistics", {})
            config = gt.get("config", {})
            n_days = config.get("n_days", 30)
            n_cats = config.get("n_categories", 10)
            mean_val = stats.get("mean_interest", 0.5)
            std_val = stats.get("std_interest", 0.1)
            # Generate synthetic matrix matching statistics for visualization
            interest_matrix = np.random.normal(mean_val, std_val, (10, n_days, n_cats))
            interest_matrix = np.clip(interest_matrix, 0, 1)

    if interest_matrix is None:
        fig = go.Figure()
        fig.update_layout(title="No Interest Data Available")
        filepath = viz_dir / "interest_heatmap.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # Average across users
    avg_interest = interest_matrix.mean(axis=0)  # (n_days, n_categories)

    fig = px.imshow(
        avg_interest.T,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(x="Day", y="Category", color="Avg Interest"),
        title="Average Interest: Days × Categories",
    )
    fig.update_layout(
        title_x=0.5,
        width=900,
        height=400,
        xaxis_title="Day",
        yaxis_title="Category",
    )

    filepath = viz_dir / "interest_heatmap.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Treatment Matrix ────────────────────────────────────────────────────────


def plot_treatment_matrix(
    ground_truth_path: str,
    output_dir: str,
) -> str:
    """Visualize the treatment assignment matrix as a heatmap.

    Args:
        ground_truth_path: Path to ground_truth.json
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "viz/timeseries")
    gt = load_json(ground_truth_path)

    if not gt:
        fig = go.Figure()
        fig.update_layout(title="No Ground Truth Data")
        filepath = viz_dir / "treatment_matrix.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # Extract user-level stats
    users = gt.get("users", [])
    config = gt.get("config", {})
    n_days = config.get("n_days", 30)

    if not users:
        filepath = viz_dir / "treatment_matrix.html"
        go.Figure().update_layout(title="No Users").write_html(str(filepath))
        return str(filepath)

    # Build treatment effect histogram
    treatment_effects = [u.get("treatment_effect", 0) for u in users]
    base_affinities = [u.get("base_affinity", 0) for u in users]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Treatment Effect Distribution", "Base Affinity Distribution"),
    )

    fig.add_trace(
        go.Histogram(
            x=treatment_effects,
            nbinsx=30,
            marker_color="#636EFA",
            name="Treatment Effect",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=base_affinities,
            nbinsx=30,
            marker_color="#EF553B",
            name="Base Affinity",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Treatment Effects & Base Affinity Distributions",
        title_x=0.5,
        height=400,
        width=900,
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Treatment Effect", row=1, col=1)
    fig.update_xaxes(title_text="Base Affinity", row=1, col=2)

    filepath = viz_dir / "treatment_matrix.html"
    fig.write_html(str(filepath))
    return str(filepath)


# ── Streamlit Tab Renderer ──────────────────────────────────────────────────


def render_timeseries_tab(output_dir: str):
    """Render the Time-Series Visualizations tab for Streamlit dashboard.

    Args:
        output_dir: Simulation output directory
    """
    import streamlit as st

    st.header("📈 Time-Series Visualizations")

    csv_path = Path(output_dir) / "causal_ground_truth.csv"
    gt_path = Path(output_dir) / "ground_truth.json"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Causal Trajectories")
        traj_path = Path(output_dir) / "viz/timeseries/causal_trajectories.html"
        if traj_path.exists():
            with open(traj_path) as f:
                st.components.v1.html(f.read(), height=500, scrolling=True)
        elif csv_path.exists():
            st.info("Trajectories not yet generated. Run visualization phase.")
        else:
            st.info("No causal trajectory data found.")

    with col2:
        st.subheader("Interest Heatmap")
        heatmap_path = Path(output_dir) / "viz/timeseries/interest_heatmap.html"
        if heatmap_path.exists():
            with open(heatmap_path) as f:
                st.components.v1.html(f.read(), height=400, scrolling=True)
        else:
            st.info("No interest heatmap available.")

    st.subheader("Persona Trajectories")
    persona_path = Path(output_dir) / "viz/timeseries/persona_trajectories.html"
    if persona_path.exists():
        with open(persona_path) as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    else:
        st.info("No persona trajectory plots available.")

    st.subheader("Treatment & Affinity Distributions")
    treatment_path = Path(output_dir) / "viz/timeseries/treatment_matrix.html"
    if treatment_path.exists():
        with open(treatment_path) as f:
            st.components.v1.html(f.read(), height=450, scrolling=True)
    else:
        st.info("No treatment distribution data.")


# ── Internal helpers ────────────────────────────────────────────────────────


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string.

    Args:
        hex_color: Hex color string (e.g. '#636EFA')
        alpha: Alpha value (0-1)

    Returns:
        RGBA string: 'rgba(R, G, B, A)'
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"
