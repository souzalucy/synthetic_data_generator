"""Time-series visualizations for causal interest trajectories, treatment matrices,
persona-level trend analysis, and per-user factual vs counterfactual comparisons.

Uses Plotly Express for interactive line charts, heatmaps, area plots,
and per-user subplot grids with delta shading.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    load_json,
    load_csv_to_dicts,
    melt_causal_csv,
    melt_causal_counterfactual_csv,
    get_persona_color,
    get_persona_dash,
    get_persona_symbol,
    get_fill_alpha,
    get_scenario_color,
    ensure_output_dir,
)
from ecosystem_sim.intelligence.explanation_agent import VisualizationExplainer


# ── Causal Trajectory Plots ─────────────────────────────────────────────────


def plot_causal_trajectories(
    csv_path: str,
    output_dir: str,
    max_users: int = 50,
    explainer: Optional["VisualizationExplainer"] = None,
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
    viz_dir = ensure_output_dir(output_dir, "timeseries")
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
        dash = get_persona_dash(persona)
        symbol = get_persona_symbol(persona)

        # Group by day, compute mean and std for this persona
        agg = persona_df.groupby("day")["affinity"].agg(["mean", "std"]).reset_index()

        fig.add_trace(
            go.Scatter(
                x=agg["day"],
                y=agg["mean"],
                mode="lines+markers",
                name=persona,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(
                    symbol=symbol,
                    size=5,
                    color=color,
                ),
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
        dash = get_persona_dash(persona)
        symbol = get_persona_symbol(persona)
        fill_alpha = get_fill_alpha(persona)
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
                mode="lines+markers",
                name=f"{persona} (mean)",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(
                    symbol=symbol,
                    size=6,
                    color=color,
                ),
                legendgroup=persona,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Confidence band with varied fill opacity
        fig.add_trace(
            go.Scatter(
                x=pd.concat([agg["day"], agg["day"][::-1]]),
                y=pd.concat([agg["mean"] + agg["ci"], (agg["mean"] - agg["ci"])[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, fill_alpha),
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
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "causal_trajectories",
            {
                "df": df,
                "unique_users": unique_users,
                "n_users": len(unique_users),
                "n_personas": len(df["persona"].unique()),
                "n_days": int(df["day"].max()),
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Causal Interest Trajectories"
        )
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


def plot_persona_trajectories(
    csv_path: str,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Create faceted line plots, one subplot per persona.

    Args:
        csv_path: Path to causal_ground_truth.csv
        output_dir: Output directory for HTML files

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "timeseries")
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

    # Pool of dash patterns for individual user lines
    _user_dash_pool = ["solid", "dot", "dash", "dashdot"]

    for idx, persona in enumerate(personas):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        persona_df = df[df["persona"] == persona]
        color = get_persona_color(persona)
        dash = get_persona_dash(persona)
        symbol = get_persona_symbol(persona)

        # Plot individual user lines (thin, low opacity, varied dashes)
        for j, uid in enumerate(
            persona_df["user_id"].unique()[:20]
        ):  # Max 20 per persona
            user_df = persona_df[persona_df["user_id"] == uid]
            fig.add_trace(
                go.Scatter(
                    x=user_df["day"],
                    y=user_df["affinity"],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=0.5,
                        dash=_user_dash_pool[j % len(_user_dash_pool)],
                    ),
                    opacity=0.3,
                    name=uid,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # Mean line (bold, with dash and marker)
        agg = persona_df.groupby("day")["affinity"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=agg["day"],
                y=agg["affinity"],
                mode="lines+markers",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(
                    symbol=symbol,
                    size=6,
                    color=color,
                ),
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
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "persona_trajectories",
            {
                "df": df,
                "personas": list(personas),
                "n_personas": len(personas),
                "n_days": int(df["day"].max()),
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Persona Trajectories"
        )
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Per-User Factual vs Counterfactual Comparison ──────────────────────────


def plot_user_counterfactual_comparison(
    csv_path: str,
    output_dir: str,
    max_users: int = 12,
    max_days: Optional[int] = None,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Create a grid of per-user subplots comparing factual vs counterfactual trajectories.

    Each subplot shows:
      - Solid blue line for factual base affinity
      - Dashed orange line for counterfactual base affinity
      - Semi-transparent delta shading between curves
      - Title showing user_id and persona_type
      - Unified hover with factual, counterfactual, and delta values

    Args:
        csv_path: Path to causal_table_with_counterfactual.csv
        output_dir: Output directory for HTML files
        max_users: Maximum number of users to show (default 12, creates a 4×3 grid)
        max_days: Optional cap on days to display (None = all days)
        explainer: Optional VisualizationExplainer for AI explanations

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "timeseries")
    csv_rows = load_csv_to_dicts(csv_path)

    if not csv_rows:
        fig = go.Figure()
        fig.update_layout(title="No Counterfactual Comparison Data Available")
        filepath = viz_dir / "user_counterfactual_comparison.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # Melt the dual-column CSV to long format
    melted = melt_causal_counterfactual_csv(csv_rows)
    df = pd.DataFrame(melted)

    # Determine unique users
    unique_users = df["user_id"].unique()
    if len(unique_users) > max_users:
        # Sample deterministically for reproducibility
        rng = np.random.default_rng(42)
        selected_users = list(rng.choice(unique_users, max_users, replace=False))
    else:
        selected_users = list(unique_users)

    # Build a lookup: user_id -> persona
    user_persona_map = (
        df[["user_id", "persona"]]
        .drop_duplicates()
        .set_index("user_id")["persona"]
        .to_dict()
    )

    n_users = len(selected_users)
    n_cols = min(4, n_users)
    n_rows = int(np.ceil(n_users / n_cols))

    # Cap days if requested
    if max_days is not None:
        df = df[df["day"] < max_days]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{uid}<br><sup>{user_persona_map.get(uid, '')}</sup>"
            for uid in selected_users
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for idx, uid in enumerate(selected_users):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        user_df = df[df["user_id"] == uid]

        # Split by scenario
        factual = user_df[user_df["scenario"] == "factual"].sort_values("day")
        counterfactual = user_df[user_df["scenario"] == "counterfactual"].sort_values(
            "day"
        )

        if factual.empty or counterfactual.empty:
            continue

        days_f = factual["day"].values
        aff_f = factual["affinity"].values
        days_cf = counterfactual["day"].values
        aff_cf = counterfactual["affinity"].values

        # Compute delta = counterfactual - factual (interpolated to common days)
        common_days = np.intersect1d(days_f, days_cf)
        f_interp = np.interp(common_days, days_f, aff_f)
        cf_interp = np.interp(common_days, days_cf, aff_cf)
        deltas = cf_interp - f_interp

        # ── Factual line (solid blue) ──
        fig.add_trace(
            go.Scatter(
                x=days_f,
                y=aff_f,
                mode="lines",
                name=f"{uid} factual",
                line=dict(color="#636EFA", width=2),
                legendgroup=f"factual_{uid}",
                showlegend=(idx == 0),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>Factual: %{{y:.4f}}<br><extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # ── Counterfactual line (dashed orange) ──
        fig.add_trace(
            go.Scatter(
                x=days_cf,
                y=aff_cf,
                mode="lines",
                name=f"{uid} counterfactual",
                line=dict(color="#FF8C00", width=2, dash="dash"),
                legendgroup=f"counterfactual_{uid}",
                showlegend=(idx == 0),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>Counterfactual: %{{y:.4f}}<br><extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # ── Delta shading (semi-transparent fill between curves) ──
        # Build a combined polygon: factual forward, counterfactual backward
        all_days = np.concatenate([common_days, common_days[::-1]])
        all_y = np.concatenate([f_interp, cf_interp[::-1]])

        fig.add_trace(
            go.Scatter(
                x=all_days,
                y=all_y,
                fill="toself",
                fillcolor="rgba(255, 140, 0, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{uid} delta",
                legendgroup=f"delta_{uid}",
                showlegend=(idx == 0),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # ── Add a custom hover trace for delta values (invisible points) ──
        fig.add_trace(
            go.Scatter(
                x=common_days,
                y=(f_interp + cf_interp) / 2,  # midpoint for hover placement
                mode="markers",
                marker=dict(size=0, opacity=0),
                name=f"{uid} delta hover",
                legendgroup=f"delta_hover_{uid}",
                showlegend=False,
                hovertemplate=(
                    "<b>Day %{x}</b><br>"
                    f"Factual: %{{customdata[0]:.4f}}<br>"
                    f"Counterfactual: %{{customdata[1]:.4f}}<br>"
                    f"Delta: %{{customdata[2]:.4f}}<br>"
                    "<extra></extra>"
                ),
                customdata=np.column_stack([f_interp, cf_interp, deltas]),
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        title="Per-User Factual vs Counterfactual Comparison",
        title_x=0.5,
        height=250 * n_rows,
        width=300 * n_cols,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    # Add a horizontal reference line at y=0 for delta context
    fig.update_yaxes(
        title_text="Base Affinity", zeroline=True, zerolinecolor="rgba(0,0,0,0.1)"
    )

    # Label only bottom-row x-axes
    for i in range(n_users):
        r = i // n_cols + 1
        c = i % n_cols + 1
        if r < n_rows:
            fig.update_xaxes(showticklabels=False, row=r, col=c)
        else:
            fig.update_xaxes(title_text="Day", row=r, col=c)

    filepath = viz_dir / "user_counterfactual_comparison.html"
    fig.write_html(str(filepath))

    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "user_counterfactual_comparison",
            {
                "df": df,
                "selected_users": selected_users,
                "n_users": n_users,
                "n_days": int(df["day"].max()) if not df.empty else 0,
                "intervention_type": csv_rows[0].get("intervention_type", "unknown")
                if csv_rows
                else "unknown",
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "User Counterfactual Comparison"
        )
        Path(filepath).write_text(html, encoding="utf-8")

    return str(filepath)


# ── Interest Heatmap ────────────────────────────────────────────────────────


def plot_interest_heatmap(
    interest_matrix: Optional[np.ndarray] = None,
    ground_truth_path: Optional[str] = None,
    output_dir: str = ".",
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Plot a heatmap of average interest across days and categories.

    Args:
        interest_matrix: Optional numpy array (n_users, n_days, n_categories).
            If a list is passed (e.g. from JSON deserialization), it is
            converted to ndarray automatically.
        ground_truth_path: Optional path to ground_truth.json. When
            *interest_matrix* is None, the function reconstructs a synthetic
            matrix matching the simulation statistics stored in the JSON.
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "timeseries")

    # Guard: convert list → ndarray if needed (JSON deserialization returns lists)
    if interest_matrix is not None and not isinstance(interest_matrix, np.ndarray):
        interest_matrix = np.asarray(interest_matrix, dtype=np.float64)

    if interest_matrix is None and ground_truth_path:
        # Reconstruct synthetic matrix from ground-truth summary statistics
        gt = load_json(ground_truth_path)
        if gt:
            stats = gt.get("statistics", {})
            cfg = gt.get("config", {})
            n_users = cfg.get("n_users", 100)
            n_days = cfg.get("n_days", 30)
            n_cats = cfg.get("n_categories", 10)
            mean_val = stats.get("mean_interest", 0.5)
            std_val = stats.get("std_interest", 0.1)
            rng = np.random.default_rng(42)
            interest_matrix = rng.normal(mean_val, std_val, (n_users, n_days, n_cats))
            interest_matrix = np.clip(interest_matrix, 0, 1)

    if interest_matrix is None:
        fig = go.Figure()
        fig.update_layout(title="No Interest Data Available")
        filepath = viz_dir / "interest_heatmap.html"
        fig.write_html(str(filepath))
        return str(filepath)

    # Average across users → (n_days, n_categories)
    avg_interest = interest_matrix.mean(axis=0)

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
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        n_days, n_cats = avg_interest.shape
        explanation = explainer.explain(
            "interest_heatmap",
            {
                "avg_interest": avg_interest,
                "n_days": n_days,
                "n_categories": n_cats,
                "mean_interest": float(avg_interest.mean()),
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Interest Heatmap"
        )
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Treatment Matrix ────────────────────────────────────────────────────────


def plot_treatment_matrix(
    ground_truth_path: str,
    output_dir: str,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Visualize the treatment assignment matrix as a heatmap.

    Args:
        ground_truth_path: Path to ground_truth.json
        output_dir: Output directory for HTML file

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "timeseries")
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
    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "treatment_matrix",
            {
                "users": users,
                "treatment_effects": treatment_effects,
                "base_affinities": base_affinities,
                "n_users": len(users),
                "n_days": n_days,
                "mean_treatment_effect": float(np.mean(treatment_effects)),
                "mean_base_affinity": float(np.mean(base_affinities)),
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Treatment Matrix"
        )
        Path(filepath).write_text(html, encoding="utf-8")
    return str(filepath)


# ── Counterfactual Table Visualization (Layer 1 User Table) ────────────────


def plot_counterfactual_table(
    csv_path: str,
    output_dir: str,
    max_users: int = 20,
    sample_days: Optional[int] = None,
    explainer: Optional["VisualizationExplainer"] = None,
) -> str:
    """Create a color-coded Plotly table comparing factual vs counterfactual affinities.

    Columns: user_id, persona, intervention_type, intervention_value,
             day_0_factual, day_0_counterfactual, day_0_delta, ...

    Cells are color-coded: green for positive delta, red for negative, gray for neutral.

    Args:
        csv_path: Path to causal_table_with_counterfactual.csv
        output_dir: Output directory for HTML files
        max_users: Maximum number of users to display
        sample_days: Optional number of days to sample (e.g. every 7th day) to keep table readable

    Returns:
        Path to generated HTML file
    """
    viz_dir = ensure_output_dir(output_dir, "timeseries")
    filepath = viz_dir / "counterfactual_table.html"

    # ── Guard: no CSV file ──
    csv_rows = load_csv_to_dicts(csv_path)
    if not csv_rows:
        fig = go.Figure()
        fig.update_layout(title="No Counterfactual Table Data Available")
        fig.write_html(str(filepath))
        return str(filepath)

    # ── Guard: missing expected columns ──
    all_keys = list(csv_rows[0].keys())
    factual_keys = sorted(
        [k for k in all_keys if k.endswith("_base_affinity_factual")],
        key=lambda x: int(x.split("_")[1]),
    )
    counterfactual_keys = sorted(
        [k for k in all_keys if k.endswith("_base_affinity_counterfactual")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not factual_keys or not counterfactual_keys:
        print(
            f"  [WARN] Counterfactual table: no day columns found. "
            f"Expected suffixes '_base_affinity_factual' / '_base_affinity_counterfactual'. "
            f"Available keys: {list(all_keys)[:10]}..."
        )
        fig = go.Figure()
        fig.update_layout(
            title="Counterfactual Table: Required Columns Not Found in CSV"
        )
        fig.write_html(str(filepath))
        return str(filepath)

    # ── Guard: mismatched key counts ──
    if len(factual_keys) != len(counterfactual_keys):
        print(
            f"  [WARN] Counterfactual table: mismatched column counts "
            f"(factual={len(factual_keys)}, counterfactual={len(counterfactual_keys)}). "
            f"Truncating to shorter list."
        )
        n = min(len(factual_keys), len(counterfactual_keys))
        factual_keys = factual_keys[:n]
        counterfactual_keys = counterfactual_keys[:n]

    # Limit users
    if len(csv_rows) > max_users:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(csv_rows), max_users, replace=False)
        csv_rows = [csv_rows[i] for i in indices]

    # Sample days if requested
    if sample_days is not None:
        step = max(1, len(factual_keys) // sample_days)
        factual_keys = factual_keys[::step]
        counterfactual_keys = counterfactual_keys[::step]

    # Build header
    headers = ["user_id", "persona", "intervention_type", "intervention_value"]
    day_labels = []
    for fk in factual_keys:
        day_num = fk.split("_")[1]
        day_labels.append(day_num)
        headers.append(f"Day {day_num}<br>Factual")
        headers.append(f"Day {day_num}<br>Counterfactual")
        headers.append(f"Day {day_num}<br>Δ")

    # Build cell values and colors
    cell_values = {h: [] for h in headers}
    cell_colors = {h: [] for h in headers}

    for row in csv_rows:
        cell_values["user_id"].append(row.get("user_id", "?"))
        cell_values["persona"].append(row.get("persona", "?"))
        cell_values["intervention_type"].append(row.get("intervention_type", ""))
        cell_values["intervention_value"].append(row.get("intervention_value", ""))

        for fk, cfk, day_num in zip(factual_keys, counterfactual_keys, day_labels):
            try:
                f_val = float(row.get(fk, 0))
                cf_val = float(row.get(cfk, 0))
            except (ValueError, TypeError):
                f_val = 0.0
                cf_val = 0.0
            delta = cf_val - f_val

            # Factual cell
            cell_values[f"Day {day_num}<br>Factual"].append(f"{f_val:.4f}")
            cell_colors[f"Day {day_num}<br>Factual"].append("white")

            # Counterfactual cell
            cell_values[f"Day {day_num}<br>Counterfactual"].append(f"{cf_val:.4f}")
            cell_colors[f"Day {day_num}<br>Counterfactual"].append("white")

            # Delta cell with color coding
            cell_values[f"Day {day_num}<br>Δ"].append(f"{delta:+.4f}")
            if delta > 0.01:
                cell_colors[f"Day {day_num}<br>Δ"].append(
                    "rgba(0, 200, 83, 0.2)"
                )  # green
            elif delta < -0.01:
                cell_colors[f"Day {day_num}<br>Δ"].append(
                    "rgba(255, 82, 82, 0.2)"
                )  # red
            else:
                cell_colors[f"Day {day_num}<br>Δ"].append(
                    "rgba(200, 200, 200, 0.2)"
                )  # gray

    # Build the table
    table_data = [cell_values[h] for h in headers]
    table_colors = [cell_colors[h] for h in headers]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    align="center",
                    font=dict(size=10, color="white"),
                    fill_color="#2c3e50",
                    height=40,
                ),
                cells=dict(
                    values=table_data,
                    align="center",
                    font=dict(size=9),
                    fill_color=table_colors,
                    height=25,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Layer 1 User Table: Factual vs Counterfactual Base Affinity",
            x=0.5,
        ),
        height=40 + min(len(csv_rows), max_users) * 28 + 60,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    fig.write_html(str(filepath))
    print(
        f"  * Counterfactual table written to {filepath} ({len(csv_rows)} users, {len(day_labels)} days)"
    )

    if explainer:
        html = Path(filepath).read_text(encoding="utf-8")
        explanation = explainer.explain(
            "counterfactual_table",
            {
                "csv_rows": csv_rows,
                "day_labels": day_labels,
                "factual_keys": factual_keys,
                "counterfactual_keys": counterfactual_keys,
                "n_users": len(csv_rows),
                "n_days": len(day_labels),
                "intervention_type": csv_rows[0].get("intervention_type", "unknown"),
            },
        )
        html = VisualizationExplainer.inject_explanation_button(
            html, explanation, "Counterfactual Table"
        )
        Path(filepath).write_text(html, encoding="utf-8")

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
    cf_csv_path = Path(output_dir) / "causal_table_with_counterfactual.csv"
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

    # ── New: Counterfactual comparison sections ──
    st.subheader("Per-User Factual vs Counterfactual Comparison")
    user_cf_path = (
        Path(output_dir) / "viz/timeseries/user_counterfactual_comparison.html"
    )
    if user_cf_path.exists():
        with open(user_cf_path) as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    elif cf_csv_path.exists():
        st.info("User counterfactual comparison not yet generated.")
    else:
        st.info("No counterfactual comparison data available.")

    st.subheader("Layer 1 User Table: Factual + Counterfactual")
    table_path = Path(output_dir) / "viz/timeseries/counterfactual_table.html"
    if table_path.exists():
        with open(table_path) as f:
            st.components.v1.html(f.read(), height=500, scrolling=True)
    elif cf_csv_path.exists():
        st.info("Counterfactual table not yet generated.")
    else:
        st.info("No counterfactual table data available.")


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
