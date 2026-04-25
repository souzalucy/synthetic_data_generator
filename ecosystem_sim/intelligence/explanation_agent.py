"""LangChain + DeepSeek explanation agent for visualization insights.

Provides a ``VisualizationExplainer`` that generates brief, non-technical
explanations for each visualization type using a DeepSeek LLM via LangChain.

Two modes:
- **LangChain mode** (default): Uses ``langchain`` + ``ChatDeepSeek`` for
  real LLM-generated explanations.
- **Template mode** (fallback): Uses rule-based templates when LangChain
  is not installed or no API key is configured. This ensures the system
  always produces explanations.

Usage::

    from ecosystem_sim.intelligence.explanation_agent import VisualizationExplainer

    explainer = VisualizationExplainer(api_key="sk-...")
    html = explainer.inject_explanation_button(
        fig_html="<div>...</div>",
        explanation=explainer.explain_lift_deciles(lift_data),
        chart_title="Lift Deciles",
    )
"""

import json
import os
import re
from typing import Dict, List, Optional, Any, Callable

# ---------------------------------------------------------------------------
# Template-based fallback explanations (used when LangChain is unavailable)
# ---------------------------------------------------------------------------

_TEMPLATE_EXPLANATIONS: Dict[str, Callable[[Dict], str]] = {}


def _register_template(key: str):
    """Decorator to register a template explanation function."""

    def decorator(func):
        _TEMPLATE_EXPLANATIONS[key] = func
        return func

    return decorator


# ── Lift Analysis Templates ────────────────────────────────────────────────


@_register_template("lift_deciles")
def _template_lift_deciles(data: Dict) -> str:
    """Template explanation for lift decile chart."""
    deciles = data.get("uplift_by_decile", [])
    if not deciles:
        return (
            "This lift decile chart shows how campaign effectiveness varies "
            "across user segments. No decile data is available yet — run the "
            "lift analysis phase first."
        )
    best_decile = max(deciles, key=lambda d: d.get("avg_uplift", 0))
    worst_decile = min(deciles, key=lambda d: d.get("avg_uplift", 0))
    return (
        f"This chart breaks users into 10 equal-sized groups (deciles) ranked "
        f"by their predicted uplift from the campaign. The **top decile** shows "
        f"an average uplift of **{best_decile.get('avg_uplift', 0):.4f}**, "
        f"meaning these users respond most strongly to treatment. The "
        f"**bottom decile** averages **{worst_decile.get('avg_uplift', 0):.4f}** — "
        f"these users may be better left untreated. Targeting the top deciles "
        f"maximizes return on ad spend."
    )


@_register_template("qini_curve")
def _template_qini_curve(data: Dict) -> str:
    """Template explanation for Qini curve."""
    qini = data.get("qini_coefficient", 0)
    auuc = data.get("auuc", 0)
    lift_pct = data.get("lift_percent", 0)
    quality = "strong" if qini > 0.15 else "moderate" if qini > 0.05 else "weak"
    return (
        f"The **Qini curve** compares the model's targeting performance against "
        f"random targeting. The blue area between the two curves represents the "
        f"model's advantage. With a Qini coefficient of **{qini:.4f}** and AUUC "
        f"of **{auuc:.4f}**, this model shows **{quality}** discriminatory power. "
        f"The overall campaign lift is **{lift_pct:.1f}%** above the no-treatment "
        f"baseline."
    )


@_register_template("lift_summary")
def _template_lift_summary(data: Dict) -> str:
    """Template explanation for lift summary dashboard."""
    lift_pct = data.get("lift_percent", 0)
    p_value = data.get("p_value", 1.0)
    significant = p_value < 0.05
    benchmark = data.get("criteo_benchmark_comparison", {})
    bench_avg = benchmark.get("benchmark_average", 0.18) * 100
    return (
        f"This dashboard summarizes the campaign's overall performance. "
        f"The observed lift of **{lift_pct:.1f}%** is "
        f"{'statistically significant' if significant else 'not statistically significant'} "
        f"(p={p_value:.4f}). "
        f"Compared to the Criteo benchmark average of **{bench_avg:.1f}%**, "
        f"this campaign is performing "
        f"{'above' if lift_pct > bench_avg else 'below'} "
        f"the industry standard."
    )


# ── Causal Trajectory Templates ────────────────────────────────────────────


@_register_template("causal_trajectories")
def _template_causal_trajectories(data: Dict) -> str:
    """Template explanation for causal trajectory plot."""
    n_users = data.get("n_users", 0)
    n_days = data.get("n_days", 0)
    mean_affinity = data.get("mean_affinity", 0)
    return (
        f"This chart shows how user interest (base affinity) evolves over "
        f"**{n_days} days** across **{n_users} users**. The top panel displays "
        f"individual trajectories colored by persona type, while the bottom panel "
        f"shows the average trajectory per persona with 95% confidence bands. "
        f"The overall mean affinity is **{mean_affinity:.3f}**. "
        f"Widening confidence bands indicate increasing divergence between users "
        f"over time, driven by the AR(1) persistence process and treatment effects."
    )


@_register_template("persona_trajectories")
def _template_persona_trajectories(data: Dict) -> str:
    """Template explanation for faceted persona trajectories."""
    personas = data.get("personas", [])
    return (
        f"This faceted view separates interest trajectories by persona type "
        f"({', '.join(personas) if personas else 'N/A'}). "
        f"Each subplot shows individual users as thin translucent lines and the "
        f"persona average as a bold line. This reveals whether personas follow "
        f"distinct interest patterns — for example, whether 'Tech-Savvy Millennials' "
        f"show consistently higher affinity than 'Budget-Conscious Parents'."
    )


# ── Counterfactual Templates ───────────────────────────────────────────────


@_register_template("counterfactual_deltas")
def _template_counterfactual_deltas(data: Dict) -> str:
    """Template explanation for counterfactual delta chart."""
    deltas = data.get("deltas", {})
    scenarios = list(deltas.keys())
    if not scenarios:
        return (
            "This chart compares graph metrics across counterfactual scenarios. "
            "No delta data is available — enable counterfactual graph analysis "
            "in the configuration."
        )
    return (
        f"This chart shows how different counterfactual interventions change "
        f"the user interaction graph relative to the factual (observed) world. "
        f"Scenarios analyzed: **{', '.join(s.replace('_', ' ').title() for s in scenarios)}**. "
        f"Positive values mean the intervention increases graph connectivity "
        f"(more edges, higher engagement), while negative values indicate reduced "
        f"connectivity. The magnitude tells us how impactful each intervention is "
        f"on the overall network structure."
    )


@_register_template("causal_pathways")
def _template_causal_pathways(data: Dict) -> str:
    """Template explanation for causal pathway discovery."""
    pathways = data.get("causal_pathways", [])
    if not pathways:
        return (
            "This section displays discovered causal pathways — the mechanisms "
            "through which interventions affect the user engagement graph. "
            "No pathways were found; this may indicate weak treatment effects "
            "or insufficient graph structure."
        )
    strongest = max(
        pathways,
        key=lambda p: abs(
            p.get("effect_sizes", {})
            .get("total_edges", {})
            .get("relative_change_pct", 0)
        ),
    )
    return (
        f"This analysis discovered **{len(pathways)} causal pathways** — the "
        f"mechanisms linking interventions to changes in user engagement. "
        f"The strongest pathway is "
        f"**'{strongest.get('label', 'Unknown')}'** with a "
        f"{strongest.get('effect_sizes', {}).get('total_edges', {}).get('relative_change_pct', 0):+.1f}% "
        f"effect on total graph edges. Each pathway quantifies how a specific "
        f"intervention propagates through the user network."
    )


@_register_template("ground_truth_summary")
def _template_ground_truth_summary(data: Dict) -> str:
    """Template explanation for ground truth summary."""
    config = data.get("config", {})
    stats = data.get("statistics", {})
    n_users = config.get("n_users", 0)
    n_days = config.get("n_days", 0)
    return (
        f"This dashboard summarizes the ground truth data generated by the "
        f"simulation. It covers **{n_users} users** over **{n_days} days** "
        f"with persona distributions, treatment effect variability, and base "
        f"affinity ranges. The persona pie chart shows how users are distributed "
        f"across types, while the box plots reveal how treatment effects and "
        f"base affinities vary within each persona group."
    )


@_register_template("interaction_graphs")
def _template_interaction_graphs(data: Dict) -> str:
    """Template explanation for interaction graph summary."""
    metrics = data.get("metrics", {})
    n_users = metrics.get("n_users_with_graphs", 0)
    n_edges = metrics.get("total_edges", 0)
    return (
        f"These network graphs visualize user interaction patterns across "
        f"digital services. **{n_users} users** have interaction graphs with "
        f"a total of **{n_edges} edges** connecting their service usage. "
        f"Each node represents a service (colored by type), and edge thickness "
        f"reflects interaction frequency. Denser graphs indicate users with "
        f"more diverse and connected digital behavior."
    )


@_register_template("device_stitching")
def _template_device_stitching(data: Dict) -> str:
    """Template explanation for device stitching graph."""
    metrics = data.get("metrics", {})
    n_devices = metrics.get("n_devices", 0)
    n_clusters = metrics.get("n_clusters", 0)
    return (
        f"This graph shows how devices are stitched together into user clusters. "
        f"**{n_devices} devices** were grouped into **{n_clusters} clusters** "
        f"based on co-occurrence patterns. Large blue nodes represent clusters; "
        f"small red nodes are individual devices. This cross-device identity "
        f"resolution is critical for understanding true user-level behavior "
        f"across multiple touchpoints."
    )


@_register_template("propensity_distributions")
def _template_propensity_distributions(data: Dict) -> str:
    """Template explanation for propensity distributions."""
    return (
        "These distributions show how treatment effects and base affinities "
        "vary across the user population. The box plots reveal differences "
        "between persona types — some personas may have inherently higher "
        "affinity or be more responsive to treatment. The histograms show "
        "the overall population distribution, which helps identify whether "
        "the treatment effect is concentrated in a subset of users or "
        "spread evenly."
    )


@_register_template("price_sensitivity")
def _template_price_sensitivity(data: Dict) -> str:
    """Template explanation for price sensitivity chart."""
    return (
        "This chart visualizes persona-level metrics as a proxy for price "
        "sensitivity analysis. Personas with higher treatment effects are "
        "more responsive to advertising interventions, which may correlate "
        "with price sensitivity. The error bars show within-persona "
        "variability — wide bars indicate heterogeneous responses within "
        "a persona group."
    )


@_register_template("cluster_size_distribution")
def _template_cluster_size_distribution(data: Dict) -> str:
    """Template explanation for cluster size distribution."""
    metrics = data.get("metrics", {})
    avg_size = metrics.get("avg_cluster_size", 0)
    return (
        f"This chart shows how device cluster sizes are distributed. "
        f"The average cluster size is **{avg_size:.1f} devices**. "
        f"Most users have 1-3 devices, but some clusters may be larger, "
        f"indicating users with many connected devices. The box plot "
        f"highlights outliers — unusually large clusters that may represent "
        f"shared devices (e.g., family tablets) or stitching errors."
    )


# ── Treatment Matrix Template ──────────────────────────────────────────────


@_register_template("treatment_matrix")
def _template_treatment_matrix(data: Dict) -> str:
    """Template explanation for treatment matrix."""
    return (
        "These histograms show the distribution of treatment effects and "
        "base affinities across all users. Treatment effects represent how "
        "strongly each user responds to advertising, while base affinity is "
        "their inherent interest level. The spread of these distributions "
        "determines how much uplift a campaign can achieve — more variation "
        "means more opportunity for targeted treatment."
    )


# ── Interest Heatmap Template ──────────────────────────────────────────────


@_register_template("interest_heatmap")
def _template_interest_heatmap(data: Dict) -> str:
    """Template explanation for interest heatmap."""
    return (
        "This heatmap shows average user interest across days (x-axis) and "
        "content categories (y-axis). Brighter colors indicate higher interest. "
        "Vertical patterns reveal temporal trends — for example, interest may "
        "grow over time due to persistence effects. Horizontal patterns show "
        "which categories are most engaging overall. This helps identify both "
        "seasonal and structural interest patterns."
    )


# ---------------------------------------------------------------------------
# Main Explainer Class
# ---------------------------------------------------------------------------


class VisualizationExplainer:
    """Generates brief explanations for simulation visualizations.

    Uses LangChain + DeepSeek when available, falling back to curated
    template-based explanations that are data-aware.

    Args:
        api_key: DeepSeek API key. If ``None``, reads from
            ``DEEPSEEK_API_KEY`` environment variable. If neither is set,
            uses template-based explanations.
        model: DeepSeek model name (default: ``deepseek-chat``).
        temperature: LLM temperature for generation (default: ``0.3``).
        max_explanations: Maximum number of LLM-generated explanations
            per session (default: ``20``). Beyond this, templates are used
            to avoid excessive API costs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_explanations: int = 20,
    ):
        # Auto-load .env file from project root if python-dotenv is available
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._model = model
        self._temperature = temperature
        self._max_explanations = max_explanations
        self._explanation_count = 0
        self._llm = None
        self._langchain_available = False

        # Try to initialize LangChain
        if self._api_key:
            self._init_langchain()
        else:
            print(
                "  [EXPLAIN] No DEEPSEEK_API_KEY found. "
                "Using template-based explanations."
            )

    def _init_langchain(self):
        """Attempt to initialize LangChain with DeepSeek."""
        try:
            from langchain.chat_models import ChatDeepSeek
            from langchain.prompts import PromptTemplate

            self._llm = ChatDeepSeek(
                model=self._model,
                api_key=self._api_key,
                temperature=self._temperature,
            )
            self._PromptTemplate = PromptTemplate
            self._langchain_available = True
            print(f"  [EXPLAIN] LangChain + DeepSeek initialized (model={self._model})")
        except ImportError:
            print(
                "  [EXPLAIN] LangChain not installed. "
                "Install with: pip install langchain langchain-community"
            )
        except Exception as exc:
            print(f"  [EXPLAIN] Failed to initialize LangChain: {exc}")

    # ------------------------------------------------------------------
    # Public explanation methods
    # ------------------------------------------------------------------

    def explain(self, chart_type: str, data: Dict) -> str:
        """Generate an explanation for a given chart type and data.

        Args:
            chart_type: One of the registered template keys
                (e.g. ``"lift_deciles"``, ``"qini_curve"``).
            data: Contextual data dict used to populate the explanation.

        Returns:
            A 2-4 sentence explanation string.
        """
        # Try LLM first (if within budget)
        if (
            self._langchain_available
            and self._explanation_count < self._max_explanations
        ):
            try:
                result = self._llm_explain(chart_type, data)
                self._explanation_count += 1
                if result:
                    return result
            except Exception as exc:
                print(f"  [EXPLAIN] LLM explanation failed ({exc}), using template")

        # Fallback to template
        template_fn = _TEMPLATE_EXPLANATIONS.get(chart_type)
        if template_fn:
            return template_fn(data)

        return (
            f"This chart visualizes **{chart_type.replace('_', ' ').title()}** "
            f"from the simulation output. Explore the interactive elements "
            f"for detailed data points."
        )

    def _llm_explain(self, chart_type: str, data: Dict) -> Optional[str]:
        """Generate explanation using LangChain + DeepSeek."""
        if not self._langchain_available or not self._llm:
            return None

        # Build a compact data summary for the prompt
        data_summary = self._summarize_data(chart_type, data)

        prompt_template = self._PromptTemplate.from_template(
            _LLM_PROMPTS.get(
                chart_type,
                _LLM_PROMPTS["_default"],
            )
        )

        try:
            response = self._llm.invoke(
                prompt_template.format(data_summary=data_summary)
            )
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            # Clean up: ensure it's 2-4 sentences
            sentences = re.split(r"(?<=[.!?])\s+", content.strip())
            if len(sentences) > 4:
                content = " ".join(sentences[:4])
            return content
        except Exception:
            return None

    @staticmethod
    def _summarize_data(chart_type: str, data: Dict) -> str:
        """Build a compact JSON summary of relevant data fields."""
        summary = {}

        if chart_type == "lift_deciles":
            deciles = data.get("uplift_by_decile", [])
            summary["n_deciles"] = len(deciles)
            if deciles:
                summary["top_decile_uplift"] = round(
                    max(d.get("avg_uplift", 0) for d in deciles), 4
                )
                summary["bottom_decile_uplift"] = round(
                    min(d.get("avg_uplift", 0) for d in deciles), 4
                )

        elif chart_type == "qini_curve":
            summary["qini_coefficient"] = round(data.get("qini_coefficient", 0), 4)
            summary["auuc"] = round(data.get("auuc", 0), 4)
            summary["lift_percent"] = round(data.get("lift_percent", 0), 2)

        elif chart_type == "lift_summary":
            summary["lift_percent"] = round(data.get("lift_percent", 0), 2)
            summary["p_value"] = round(data.get("p_value", 1.0), 4)
            summary["qini_coefficient"] = round(data.get("qini_coefficient", 0), 4)
            bench = data.get("criteo_benchmark_comparison", {})
            summary["benchmark_avg"] = round(bench.get("benchmark_average", 0) * 100, 2)

        elif chart_type == "causal_trajectories":
            summary["n_users"] = data.get("n_users", 0)
            summary["n_days"] = data.get("n_days", 0)
            summary["mean_affinity"] = round(data.get("mean_affinity", 0), 4)

        elif chart_type == "counterfactual_deltas":
            deltas = data.get("deltas", {})
            summary["scenarios"] = list(deltas.keys())
            for sname, sdata in deltas.items():
                summary[f"{sname}_deltas"] = {k: round(v, 2) for k, v in sdata.items()}

        elif chart_type == "causal_pathways":
            pathways = data.get("causal_pathways", [])
            summary["n_pathways"] = len(pathways)
            if pathways:
                strongest = max(
                    pathways,
                    key=lambda p: abs(
                        p.get("effect_sizes", {})
                        .get("total_edges", {})
                        .get("relative_change_pct", 0)
                    ),
                )
                summary["strongest_pathway"] = strongest.get("label", "Unknown")
                summary["strongest_effect_pct"] = round(
                    strongest.get("effect_sizes", {})
                    .get("total_edges", {})
                    .get("relative_change_pct", 0),
                    2,
                )

        elif chart_type == "device_stitching":
            metrics = data.get("metrics", {})
            summary["n_devices"] = metrics.get("n_devices", 0)
            summary["n_clusters"] = metrics.get("n_clusters", 0)
            summary["avg_cluster_size"] = round(metrics.get("avg_cluster_size", 0), 2)

        elif chart_type == "interaction_graphs":
            metrics = data.get("metrics", {})
            summary["n_users_with_graphs"] = metrics.get("n_users_with_graphs", 0)
            summary["total_edges"] = metrics.get("total_edges", 0)
            summary["avg_degree"] = round(metrics.get("avg_degree", 0), 4)

        elif chart_type == "ground_truth_summary":
            config = data.get("config", {})
            stats = data.get("statistics", {})
            summary["n_users"] = config.get("n_users", 0)
            summary["n_days"] = config.get("n_days", 0)
            summary["mean_interest"] = round(stats.get("mean_interest", 0), 4)
            summary["mean_treatment_effect"] = round(
                stats.get("mean_treatment_effect", 0), 4
            )

        elif chart_type == "cluster_size_distribution":
            metrics = data.get("metrics", {})
            summary["n_clusters"] = metrics.get("n_clusters", 0)
            summary["n_devices"] = metrics.get("n_devices", 0)
            summary["avg_cluster_size"] = round(metrics.get("avg_cluster_size", 0), 2)

        else:
            # Generic: include top-level numeric keys
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    summary[k] = round(v, 4) if isinstance(v, float) else v

        return json.dumps(summary, indent=2)

    # ------------------------------------------------------------------
    # HTML injection
    # ------------------------------------------------------------------

    _EXPLAIN_BUTTON_TPL: str = """\
<div style="margin-top:8px; text-align:right;">
  <button onclick="toggleExplanation(this)"
          style="background:#636EFA; color:white; border:none; border-radius:6px;
                 padding:6px 16px; cursor:pointer; font-size:13px;
                 transition:background 0.2s;"
          onmouseover="this.style.background='#4A54D6'"
          onmouseout="this.style.background='#636EFA'">
    __BUTTON_LABEL__
  </button>
  <div class="viz-explanation" style="display:none; margin-top:8px;
       background:#f0f4ff; border:1px solid #d0d8ff; border-radius:8px;
       padding:12px 16px; font-size:14px; line-height:1.6; color:#1a1a2e;
       text-align:left;">
    __EXPLANATION__
  </div>
</div>
<script>
function toggleExplanation(btn) {
  var panel = btn.nextElementSibling;
  if (panel.style.display === 'none' || panel.style.display === '') {
    panel.style.display = 'block';
    btn.textContent = '🙈 Hide explanation';
  } else {
    panel.style.display = 'none';
    btn.textContent = '__BUTTON_LABEL__';
  }
}
</script>
"""

    _EXPLAIN_PANEL_STATIC_TPL: str = """\
<div style="margin-top:8px; background:#f0f4ff; border:1px solid #d0d8ff;
            border-radius:8px; padding:12px 16px; font-size:14px; line-height:1.6;
            color:#1a1a2e;">
  <strong>💡 Insight</strong><br>
  __EXPLANATION__
</div>
"""

    @staticmethod
    def inject_explanation_button(
        fig_html: str,
        explanation: str,
        chart_title: str = "",
    ) -> str:
        """Wrap a Plotly figure HTML with a collapsible explanation panel.

        Args:
            fig_html: The raw HTML string from ``fig.to_html()``.
            explanation: The explanation text to embed.
            chart_title: Optional chart title for the button label.

        Returns:
            Modified HTML with the explanation button and panel injected
            just after the figure.
        """
        safe_explanation = VisualizationExplainer._sanitize_html(explanation)
        button_label = (
            "💡 Explain this chart" if not chart_title else f"💡 Explain: {chart_title}"
        )

        explain_html = VisualizationExplainer._EXPLAIN_BUTTON_TPL.replace(
            "__BUTTON_LABEL__", button_label
        ).replace("__EXPLANATION__", safe_explanation)

        if "</body>" in fig_html:
            return fig_html.replace("</body>", explain_html + "\n</body>")
        return fig_html + explain_html

    @staticmethod
    def inject_explanation_button_static(
        fig_html: str,
        explanation: str,
        chart_title: str = "",
    ) -> str:
        """Same as ``inject_explanation_button`` but with the explanation
        always visible (no toggle). Useful for print/PDF export."""
        safe_explanation = VisualizationExplainer._sanitize_html(explanation)

        panel_html = VisualizationExplainer._EXPLAIN_PANEL_STATIC_TPL.replace(
            "__EXPLANATION__", safe_explanation
        )

        if "</body>" in fig_html:
            return fig_html.replace("</body>", panel_html + "\n</body>")
        return fig_html + panel_html

    @staticmethod
    def _sanitize_html(text: str) -> str:
        """Sanitize text for safe embedding in HTML, and convert **bold** markers."""
        _amp = chr(38) + "amp;"  # &
        _lt = chr(38) + "lt;"  # <
        _gt = chr(38) + "gt;"  # >
        _quot = chr(38) + "quot;"  # "
        _apos = chr(38) + "#39;"  # '
        safe = (
            text.replace(chr(38), _amp)
            .replace(chr(60), _lt)
            .replace(chr(62), _gt)
            .replace(chr(34), _quot)
            .replace(chr(39), _apos)
            .replace("\n", "<br>")
        )
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        return safe


# ---------------------------------------------------------------------------
# LLM Prompt Templates
# ---------------------------------------------------------------------------

_LLM_PROMPTS: Dict[str, str] = {
    "_default": (
        "You are a data analyst explaining a machine learning visualization "
        "to a non-technical business stakeholder.\n\n"
        "Chart data summary:\n{data_summary}\n\n"
        "Provide a brief explanation (2-4 sentences) covering:\n"
        "1. What this chart shows\n"
        "2. The key insight or pattern\n"
        "3. A practical business implication\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "lift_deciles": (
        "You are a marketing analytics expert explaining an uplift decile chart.\n\n"
        "Decile data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. What deciles represent (users ranked by predicted uplift)\n"
        "2. How the top vs bottom deciles compare\n"
        "3. Which deciles should be targeted for maximum ROI\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "qini_curve": (
        "You are a data scientist explaining a Qini curve to a marketing director.\n\n"
        "Qini data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. What the Qini curve shows (model vs random targeting)\n"
        "2. Whether the model is effective (Qini coefficient interpretation)\n"
        "3. The business value of using this model over random targeting\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "lift_summary": (
        "You are a campaign analyst presenting results to stakeholders.\n\n"
        "Campaign data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. The overall campaign lift and whether it's statistically significant\n"
        "2. How it compares to industry benchmarks (Criteo)\n"
        "3. A recommendation based on the results\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "causal_trajectories": (
        "You are a behavioral analyst explaining user interest trajectories.\n\n"
        "Trajectory data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. How user interests evolve over time\n"
        "2. What the confidence bands tell us about user divergence\n"
        "3. What drives the trajectory patterns (persistence, treatment)\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "counterfactual_deltas": (
        "You are a causal inference expert explaining counterfactual analysis.\n\n"
        "Counterfactual data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. What counterfactual scenarios were tested\n"
        "2. Which intervention had the strongest effect on graph structure\n"
        "3. What this implies for campaign strategy\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "causal_pathways": (
        "You are a causal discovery researcher explaining pathway analysis.\n\n"
        "Pathway data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. What causal pathways were discovered\n"
        "2. Which pathway has the strongest effect\n"
        "3. How this helps understand the intervention mechanism\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "device_stitching": (
        "You are a privacy engineer explaining device stitching results.\n\n"
        "Stitching data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. What device stitching reveals about multi-device behavior\n"
        "2. The cluster size distribution and what it means\n"
        "3. Implications for cross-device identity resolution\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "interaction_graphs": (
        "You are a network analyst explaining user interaction graphs.\n\n"
        "Graph data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. What the interaction graphs represent\n"
        "2. Key network metrics and their interpretation\n"
        "3. How graph structure relates to user engagement\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "ground_truth_summary": (
        "You are a simulation engineer explaining the generated ground truth.\n\n"
        "Ground truth data:\n{data_summary}\n\n"
        "Explain in 2-4 sentences:\n"
        "1. The simulation configuration and scale\n"
        "2. Key statistics about the generated population\n"
        "3. How the data is structured for downstream analysis\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
}
