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
from typing import Dict, Optional, Callable

import numpy as np

from ecosystem_sim.intelligence.viz_objectives import get_objective

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
    deciles = data.get("deciles", data.get("uplift_by_decile", []))
    if not deciles:
        return (
            "This lift decile chart shows how campaign effectiveness varies "
            "across user segments. No decile data is available yet - run the "
            "lift analysis phase first."
        )
    best_decile = max(deciles, key=lambda d: d.get("avg_uplift", 0))
    worst_decile = min(deciles, key=lambda d: d.get("avg_uplift", 0))
    top3 = sorted(deciles, key=lambda d: d.get("avg_uplift", 0), reverse=True)[:3]
    top3_share = (
        sum(d.get("cumulative_uplift", 0) for d in top3)
        / max(sum(d.get("cumulative_uplift", 0) for d in deciles), 1e-10)
        * 100
    )
    positive = sum(1 for d in deciles if d.get("avg_uplift", 0) > 0)
    negative = sum(1 for d in deciles if d.get("avg_uplift", 0) < 0)
    return (
        f"This chart breaks users into 10 equal-sized groups (deciles) ranked "
        f"by their predicted uplift from the campaign. The **top decile** shows "
        f"an average uplift of **{best_decile.get('avg_uplift', 0):.4f}**, "
        f"while the **bottom decile** averages **{worst_decile.get('avg_uplift', 0):.4f}**. "
        f"**{positive} deciles** show positive uplift and **{negative}** show negative uplift. "
        f"The top 3 deciles capture **{top3_share:.1f}%** of total uplift, "
        f"meaning targeting these segments maximises return on ad spend."
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
        f"This dashboard summarises the campaign's overall performance. "
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
    personas = data.get("personas", [])
    affinity_trend = data.get("affinity_trend", "stable")
    trend_desc = {
        "increasing": "interest is growing over time, driven by positive persistence effects",
        "decreasing": "interest is declining over time",
        "stable": "interest remains relatively stable over the observation period",
    }.get(affinity_trend, "shows dynamic patterns")
    return (
        f"This chart shows how user interest (base affinity) evolves over "
        f"**{n_days} days** across **{n_users} users** and "
        f"{len(personas)} persona types ({', '.join(personas[:3])}{'...' if len(personas) > 3 else ''}). "
        f"The overall mean affinity is **{mean_affinity:.3f}** and the trend indicates "
        f"interest is **{trend_desc}**. "
        f"Widening confidence bands would indicate increasing divergence between users "
        f"over time, driven by the AR(1) persistence process and treatment effects."
    )


@_register_template("persona_trajectories")
def _template_persona_trajectories(data: Dict) -> str:
    """Template explanation for faceted persona trajectories."""
    personas = data.get("personas", [])
    n_days = data.get("n_days", 0)
    return (
        f"This faceted view separates interest trajectories by persona type "
        f"({', '.join(personas) if personas else 'N/A'}) over **{n_days} days**. "
        f"Each subplot shows individual users as thin translucent lines and the "
        f"persona average as a bold line. This reveals whether personas follow "
        f"distinct interest patterns - for example, whether 'Tech-Savvy Millennials' "
        f"show consistently higher affinity than 'Budget-Conscious Parents'."
    )


# ── Counterfactual Templates ───────────────────────────────────────────────


@_register_template("counterfactual_deltas")
def _template_counterfactual_deltas(data: Dict) -> str:
    """Template explanation for counterfactual delta chart."""
    scenarios = data.get("scenarios", [])
    deltas = data.get("deltas", {})
    if not scenarios:
        return (
            "This chart compares graph metrics across counterfactual scenarios. "
            "No delta data is available - enable counterfactual graph analysis "
            "in the configuration."
        )
    # Find the scenario with the largest absolute delta
    max_scenario = None
    max_abs_delta = 0
    for sname, sdeltas in deltas.items():
        for metric, val in sdeltas.items():
            if abs(val) > max_abs_delta:
                max_abs_delta = abs(val)
                max_scenario = sname
    max_desc = (
        f" The strongest effect is from **'{max_scenario.replace('_', ' ').title()}'** "
        f"with a **{max_abs_delta:+.1f}%** change in one or more graph metrics."
        if max_scenario
        else ""
    )
    return (
        f"This chart shows how different counterfactual interventions change "
        f"the user interaction graph relative to the factual (observed) world. "
        f"Scenarios analysed: **{', '.join(s.replace('_', ' ').title() for s in scenarios)}**. "
        f"Positive values mean the intervention increases graph connectivity "
        f"(more edges, higher engagement), while negative values indicate reduced "
        f"connectivity.{max_desc}"
    )


@_register_template("causal_pathways")
def _template_causal_pathways(data: Dict) -> str:
    """Template explanation for causal pathway discovery."""
    pathways = data.get("causal_pathways", [])
    if not pathways:
        return (
            "This section displays discovered causal pathways - the mechanisms "
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
        f"This analysis discovered **{len(pathways)} causal pathways** - the "
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
    n_personas = data.get("n_personas", 0)
    mean_te = stats.get("mean_treatment_effect", stats.get("mean_treatment", 0))
    mean_aff = stats.get("mean_interest", 0)
    return (
        f"This dashboard summarises the ground truth data generated by the "
        f"simulation. It covers **{n_users} users** over **{n_days} days** "
        f"across **{n_personas} persona types**. "
        f"The mean treatment effect is **{mean_te:.4f}** and mean base affinity "
        f"is **{mean_aff:.4f}**. "
        f"The persona pie chart shows how users are distributed "
        f"across types, while the box plots reveal how treatment effects and "
        f"base affinities vary within each persona group."
    )


@_register_template("interaction_graphs")
def _template_interaction_graphs(data: Dict) -> str:
    """Template explanation for interaction graph summary."""
    metrics = data.get("metrics", {})
    n_users = metrics.get("n_users_with_graphs", 0)
    n_edges = metrics.get("total_edges", 0)
    avg_degree = metrics.get("avg_degree", 0)
    return (
        f"These network graphs visualise user interaction patterns across "
        f"digital services. **{n_users} users** have interaction graphs with "
        f"a total of **{n_edges} edges** (average degree: **{avg_degree:.2f}**). "
        f"Each node represents a service (colored by type), and edge thickness "
        f"reflects interaction frequency. Denser graphs indicate users with "
        f"more diverse and connected digital behaviour."
    )


@_register_template("device_stitching")
def _template_device_stitching(data: Dict) -> str:
    """Template explanation for device stitching graph."""
    metrics = data.get("metrics", {})
    n_devices = metrics.get("n_devices", 0)
    n_clusters = metrics.get("n_clusters", 0)
    avg_size = metrics.get("avg_cluster_size", 0)
    return (
        f"This graph shows how devices are stitched together into user clusters. "
        f"**{n_devices} devices** were grouped into **{n_clusters} clusters** "
        f"(average size: **{avg_size:.1f}** devices per cluster). "
        f"Large blue nodes represent clusters; "
        f"small red nodes are individual devices. This cross-device identity "
        f"resolution is critical for understanding true user-level behaviour "
        f"across multiple touchpoints."
    )


@_register_template("propensity_distributions")
def _template_propensity_distributions(data: Dict) -> str:
    """Template explanation for propensity distributions."""
    n_personas = data.get("n_personas", 0)
    n_users = data.get("n_users", 0)
    return (
        f"These distributions show how treatment effects and base affinities "
        f"vary across the user population (**{n_users} users**, "
        f"**{n_personas} personas**). "
        f"The box plots reveal differences "
        f"between persona types - some personas may have inherently higher "
        f"affinity or be more responsive to treatment. The histograms show "
        f"the overall population distribution, which helps identify whether "
        f"the treatment effect is concentrated in a subset of users or "
        f"spread evenly."
    )


@_register_template("price_sensitivity")
def _template_price_sensitivity(data: Dict) -> str:
    """Template explanation for price sensitivity chart."""
    n_personas = data.get("n_personas", 0)
    return (
        f"This chart visualises persona-level metrics across "
        f"**{n_personas} persona types** as a proxy for price "
        f"sensitivity analysis. Personas with higher treatment effects are "
        f"more responsive to advertising interventions, which may correlate "
        f"with price sensitivity. The error bars show within-persona "
        f"variability - wide bars indicate heterogeneous responses within "
        f"a persona group."
    )


@_register_template("cluster_size_distribution")
def _template_cluster_size_distribution(data: Dict) -> str:
    """Template explanation for cluster size distribution."""
    metrics = data.get("metrics", {})
    avg_size = metrics.get("avg_cluster_size", 0)
    n_clusters = metrics.get("n_clusters", 0)
    return (
        f"This chart shows how **{n_clusters} device clusters** are distributed by size. "
        f"The average cluster size is **{avg_size:.1f} devices**. "
        f"Most users have 1-3 devices, but some clusters may be larger, "
        f"indicating users with many connected devices. The box plot "
        f"highlights outliers - unusually large clusters that may represent "
        f"shared devices (e.g., family tablets) or stitching errors."
    )


# ── Treatment Matrix Template ──────────────────────────────────────────────


@_register_template("treatment_matrix")
def _template_treatment_matrix(data: Dict) -> str:
    """Template explanation for treatment matrix."""
    n_users = data.get("n_users", 0)
    mean_te = data.get("mean_treatment_effect", 0)
    mean_ba = data.get("mean_base_affinity", 0)
    return (
        f"These histograms show the distribution of treatment effects and "
        f"base affinities across **{n_users} users**. "
        f"The mean treatment effect is **{mean_te:.4f}** and mean base affinity "
        f"is **{mean_ba:.4f}**. "
        f"Treatment effects represent how "
        f"strongly each user responds to advertising, while base affinity is "
        f"their inherent interest level. The spread of these distributions "
        f"determines how much uplift a campaign can achieve - more variation "
        f"means more opportunity for targeted treatment."
    )


# ── Interest Heatmap Template ──────────────────────────────────────────────


@_register_template("interest_heatmap")
def _template_interest_heatmap(data: Dict) -> str:
    """Template explanation for interest heatmap."""
    n_days = data.get("n_days", 0)
    n_categories = data.get("n_categories", 0)
    mean_interest = data.get("mean_interest", 0)
    return (
        f"This heatmap shows average user interest across **{n_days} days** (x-axis) and "
        f"**{n_categories} content categories** (y-axis). "
        f"The overall mean interest is **{mean_interest:.4f}**. "
        f"Brighter colours indicate higher interest. "
        f"Vertical patterns reveal temporal trends - for example, interest may "
        f"grow over time due to persistence effects. Horizontal patterns show "
        f"which categories are most engaging overall. This helps identify both "
        f"seasonal and structural interest patterns."
    )


# ── Counterfactual Table / User Comparison Templates ───────────────────────


@_register_template("user_counterfactual_comparison")
def _template_user_counterfactual_comparison(data: Dict) -> str:
    """Template explanation for per-user counterfactual comparison."""
    n_users = data.get("n_users", 0)
    n_days = data.get("n_days", 0)
    intervention = data.get("intervention_type", "unknown")
    return (
        f"This grid compares factual vs counterfactual interest trajectories "
        f"for **{n_users} users** over **{n_days} days** under the "
        f"**'{intervention}'** intervention. "
        f"Each subplot shows the factual trajectory (solid blue line) and the "
        f"counterfactual trajectory (dashed orange line), with orange shading "
        f"indicating the delta between them. Users whose counterfactual trajectory "
        f"lies above their factual one are positively affected by the intervention."
    )


@_register_template("counterfactual_table")
def _template_counterfactual_table(data: Dict) -> str:
    """Template explanation for counterfactual table."""
    n_users = data.get("n_users", 0)
    n_days = data.get("n_days", 0)
    intervention = data.get("intervention_type", "unknown")
    return (
        f"This colour-coded table shows per-user, per-day factual and "
        f"counterfactual base affinities for **{n_users} users** across "
        f"**{n_days} days** under the **'{intervention}'** intervention. "
        f"Green cells indicate positive deltas (intervention increases interest), "
        f"red cells indicate negative deltas (intervention decreases interest), "
        f"and grey cells indicate negligible change. "
        f"A predominantly green table means the intervention is broadly effective."
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

    @staticmethod
    def inject_explanation_button(
        html: str, explanation: str, title: str = "Explanation"
    ) -> str:
        """Inject an expandable explanation button into a Plotly HTML page.

        Args:
            html: The raw HTML content of the Plotly figure.
            explanation: The explanation text to inject.
            title: The title for the explanation section.

        Returns:
            Modified HTML with the explanation button injected.
        """
        button_html = f"""
        <div style="text-align:center; margin:10px 0;">
            <button onclick="toggleExplanation()"
                    style="background:#636EFA;color:white;border:none;
                           padding:8px 20px;border-radius:5px;cursor:pointer;
                           font-size:14px;">
                [chart] {title}
            </button>
        </div>
        <div id="explanation-box"
             style="display:none; margin:10px 20px; padding:15px;
                    background:#f8f9fa; border-left:4px solid #636EFA;
                    border-radius:4px; font-family:sans-serif; font-size:14px;
                    line-height:1.6;">
            {explanation}
        </div>
        <script>
        function toggleExplanation() {{
            var box = document.getElementById('explanation-box');
            box.style.display = box.style.display === 'none' ? 'block' : 'none';
        }}
        </script>
        """
        # Insert before the closing </body> tag, or append if not found
        if "</body>" in html:
            return html.replace("</body>", button_html + "\n</body>")
        return html + button_html

    def _init_langchain(self):
        """Attempt to initialize LangChain with DeepSeek (OpenAI-compatible)."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate

            self._llm = ChatOpenAI(
                model=self._model,
                api_key=self._api_key,
                temperature=self._temperature,
                base_url="https://api.deepseek.com/v1",
            )
            self._PromptTemplate = PromptTemplate
            self._langchain_available = True
            print(f"  [EXPLAIN] LangChain + DeepSeek initialized (model={self._model})")
        except ImportError:
            print(
                "  [EXPLAIN] LangChain not installed. "
                "Install with: pip install langchain langchain-community langchain-openai"
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
                Should contain the **full computed results** that the
                visualization renders, not just metadata.

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
            f"This chart visualises **{chart_type.replace('_', ' ').title()}** "
            f"from the simulation output. Explore the interactive elements "
            f"for detailed data points."
        )

    def generate_detailed_report(self, chart_type: str, data: Dict) -> str:
        """Generate a comprehensive, multi-paragraph report.

        Unlike ``explain()`` which returns 2-4 sentences, this method
        produces a full narrative report grounded in the actual data,
        suitable for inclusion in an automated analysis document.

        Args:
            chart_type: One of the registered template keys.
            data: Full computed results dict for the visualization.

        Returns:
            A multi-paragraph report string.
        """
        objective = get_objective(chart_type)
        data_summary = self._rich_summarize(chart_type, data)

        # Try LLM first
        if self._langchain_available:
            try:
                return self._llm_generate_report(chart_type, objective, data_summary)
            except Exception as exc:
                print(f"  [EXPLAIN] LLM report failed ({exc}), using template")

        # Fallback: build a structured report from template + data
        return self._template_generate_report(chart_type, objective, data, data_summary)

    def _llm_explain(self, chart_type: str, data: Dict) -> Optional[str]:
        """Generate explanation using LangChain + DeepSeek."""
        if not self._langchain_available or not self._llm:
            return None

        # Build a rich data summary for the prompt
        data_summary = self._rich_summarize(chart_type, data)
        objective = get_objective(chart_type)

        prompt_template = self._PromptTemplate.from_template(
            _LLM_PROMPTS.get(
                chart_type,
                _LLM_PROMPTS["_default"],
            )
        )

        try:
            response = self._llm.invoke(
                prompt_template.format(
                    data_summary=data_summary,
                    objective=objective.get("objective", ""),
                    key_questions="\n".join(
                        f"- {q}" for q in objective.get("key_questions", [])
                    ),
                )
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

    def _llm_generate_report(
        self, chart_type: str, objective: Dict, data_summary: str
    ) -> str:
        """Generate a detailed report using LangChain + DeepSeek."""
        if not self._langchain_available or not self._llm:
            return ""

        prompt = (
            "You are a senior data analyst writing an automated analysis report.\n\n"
            "CHART TITLE: {title}\n\n"
            "CHART OBJECTIVE: {objective}\n\n"
            "KEY QUESTIONS THIS CHART ANSWERS:\n{key_questions}\n\n"
            "DATA SUMMARY:\n{data_summary}\n\n"
            "Write a detailed analytical report (3-5 paragraphs) that:\n"
            "1. States the purpose of this visualisation and what it reveals\n"
            "2. Reports the key numbers from the data summary with context\n"
            "3. Identifies patterns, trends, or anomalies\n"
            "4. Provides actionable business or analytical implications\n"
            "5. Notes any caveats or limitations\n\n"
            "Use plain language suitable for a technical but non-specialist audience. "
            "Bold key numbers with **asterisks**."
        )

        prompt_template = self._PromptTemplate.from_template(prompt)
        try:
            response = self._llm.invoke(
                prompt_template.format(
                    title=objective.get("title", chart_type),
                    objective=objective.get("objective", ""),
                    key_questions="\n".join(
                        f"- {q}" for q in objective.get("key_questions", [])
                    ),
                    data_summary=data_summary,
                )
            )
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return ""

    @staticmethod
    def _template_generate_report(
        chart_type: str, objective: Dict, data: Dict, data_summary: str
    ) -> str:
        """Build a structured report from template explanation + data."""
        # Start with the template explanation
        template_fn = _TEMPLATE_EXPLANATIONS.get(chart_type)
        core_explanation = template_fn(data) if template_fn else ""

        # Parse the data summary for additional context
        try:
            summary_dict = json.loads(data_summary)
        except (json.JSONDecodeError, TypeError):
            summary_dict = {}

        # Build a structured report
        parts = [
            f"## {objective.get('title', chart_type.replace('_', ' ').title())}",
            "",
            f"**Objective:** {objective.get('objective', '')}",
            "",
            "### Key Findings",
            "",
            core_explanation,
            "",
        ]

        # Add data highlights if available
        highlights = []
        for k, v in summary_dict.items():
            if isinstance(v, (int, float)) and k not in (
                "chart_objective",
                "n_users",
                "n_days",
            ):
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    highlights.append(f"- **{label}**: {v:.4f}")
                else:
                    highlights.append(f"- **{label}**: {v}")
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)):
                        label = f"{k.replace('_', ' ').title()} - {sk.replace('_', ' ').title()}"
                        if isinstance(sv, float):
                            highlights.append(f"- **{label}**: {sv:.4f}")
                        else:
                            highlights.append(f"- **{label}**: {sv}")

        if highlights:
            parts.append("### Data Highlights")
            parts.append("")
            parts.extend(highlights)
            parts.append("")

        parts.append("---\n*Automated analysis generated from simulation output data.*")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Rich statistical summarization
    # ------------------------------------------------------------------

    @staticmethod
    def _rich_summarize(chart_type: str, data: Dict) -> str:
        """Build a rich, statistically-informed data summary.

        Unlike the previous ``_summarize_data`` which extracted only
        top-level metadata, this method computes meaningful derived
        statistics (ranges, distributions, concentrations) that give
        both the LLM and template engines enough context to produce
        grounded explanations.

        Args:
            chart_type: The registered chart type key.
            data: The full computed results dict.

        Returns:
            A JSON string with the enriched summary.
        """
        summary = {}
        objective = get_objective(chart_type)
        summary["chart_objective"] = objective.get("objective", "")

        # ── Lift Deciles ──────────────────────────────────────────────
        if chart_type == "lift_deciles":
            deciles = data.get("deciles", data.get("uplift_by_decile", []))
            if deciles:
                df_vals = [d.get("avg_uplift", 0) for d in deciles]
                cum_vals = [d.get("cumulative_uplift", 0) for d in deciles]
                summary["n_deciles"] = len(deciles)
                summary["uplift_min"] = round(min(df_vals), 4)
                summary["uplift_max"] = round(max(df_vals), 4)
                summary["uplift_mean"] = round(float(np.mean(df_vals)), 4)
                summary["uplift_std"] = round(float(np.std(df_vals)), 4)
                summary["positive_deciles"] = int(sum(1 for v in df_vals if v > 0))
                summary["negative_deciles"] = int(sum(1 for v in df_vals if v < 0))
                # Top 3 concentration
                sorted_vals = sorted(df_vals, reverse=True)
                top3 = sorted_vals[:3]
                total_cum = sum(cum_vals) if cum_vals else sum(df_vals)
                summary["top3_avg_uplift"] = round(float(np.mean(top3)), 4)
                summary["top3_concentration_pct"] = round(
                    sum(top3) / max(total_cum, 1e-10) * 100, 1
                )

        # ── Qini Curve ────────────────────────────────────────────────
        elif chart_type == "qini_curve":
            summary["qini_coefficient"] = round(data.get("qini_coefficient", 0), 4)
            summary["auuc"] = round(data.get("auuc", 0), 4)
            summary["lift_percent"] = round(data.get("lift_percent", 0), 2)
            qini = data.get("qini_coefficient", 0)
            summary["model_power"] = (
                "strong" if qini > 0.15 else "moderate" if qini > 0.05 else "weak"
            )

        # ── Lift Summary ──────────────────────────────────────────────
        elif chart_type == "lift_summary":
            summary["lift_percent"] = round(data.get("lift_percent", 0), 2)
            summary["p_value"] = round(data.get("p_value", 1.0), 4)
            summary["is_significant"] = data.get("p_value", 1.0) < 0.05
            summary["auuc"] = round(data.get("auuc", 0), 4)
            summary["qini_coefficient"] = round(data.get("qini_coefficient", 0), 4)
            bench = data.get("criteo_benchmark_comparison", {})
            summary["benchmark_avg_pct"] = round(
                bench.get("benchmark_average", 0) * 100, 2
            )
            summary["observed_uplift_pct"] = round(
                bench.get("observed_uplift", 0) * 100, 2
            )
            summary["above_benchmark"] = (
                summary["observed_uplift_pct"] > summary["benchmark_avg_pct"]
            )

        # ── Causal Trajectories ───────────────────────────────────────
        elif chart_type == "causal_trajectories":
            summary["n_users"] = data.get("n_users", 0)
            summary["n_days"] = data.get("n_days", 0)
            # Try to compute from DataFrame if passed
            df = data.get("df")
            if df is not None and hasattr(df, "groupby"):
                try:
                    summary["mean_affinity"] = round(float(df["affinity"].mean()), 4)
                    personas = sorted(df["persona"].unique())
                    summary["n_personas"] = len(personas)
                    summary["personas"] = list(personas[:5])
                    # Trend detection from daily means
                    daily = df.groupby("day")["affinity"].mean().sort_index()
                    affinity_by_day = daily.values
                    if len(affinity_by_day) >= 2:
                        mid = len(affinity_by_day) // 2
                        first_half = float(np.mean(affinity_by_day[:mid]))
                        second_half = float(np.mean(affinity_by_day[mid:]))
                        if second_half - first_half > 0.01:
                            summary["trend"] = "increasing"
                        elif first_half - second_half > 0.01:
                            summary["trend"] = "decreasing"
                        else:
                            summary["trend"] = "stable"
                except Exception:
                    pass
            else:
                # Fallback to pre-computed values
                summary["mean_affinity"] = round(data.get("mean_affinity", 0), 4)
                personas = data.get("personas", [])
                summary["n_personas"] = len(personas)
                summary["personas"] = personas[:5]
                affinity_by_day = data.get("affinity_by_day", [])
                if len(affinity_by_day) >= 2:
                    mid = len(affinity_by_day) // 2
                    first_half = float(np.mean(affinity_by_day[:mid]))
                    second_half = float(np.mean(affinity_by_day[mid:]))
                    if second_half - first_half > 0.01:
                        summary["trend"] = "increasing"
                    elif first_half - second_half > 0.01:
                        summary["trend"] = "decreasing"
                    else:
                        summary["trend"] = "stable"

        # ── Persona Trajectories ──────────────────────────────────────
        elif chart_type == "persona_trajectories":
            df = data.get("df")
            if df is not None and hasattr(df, "groupby"):
                try:
                    personas = sorted(df["persona"].unique())
                    summary["n_personas"] = len(personas)
                    summary["personas"] = list(personas[:5])
                    summary["n_days"] = int(df["day"].max())
                except Exception:
                    pass
            else:
                personas = data.get("personas", [])
                summary["n_personas"] = len(personas)
                summary["personas"] = personas[:5]
                summary["n_days"] = data.get("n_days", 0)

        # ── Counterfactual Deltas ─────────────────────────────────────
        elif chart_type == "counterfactual_deltas":
            deltas = data.get("deltas", {})
            scenarios = data.get("scenarios", list(deltas.keys()))
            summary["n_scenarios"] = len(scenarios)
            summary["scenarios"] = scenarios[:5]
            # Find max absolute delta
            max_abs = 0
            max_scenario = ""
            max_metric = ""
            for sname, sdeltas in deltas.items():
                for metric, val in sdeltas.items():
                    if abs(val) > max_abs:
                        max_abs = abs(val)
                        max_scenario = sname
                        max_metric = metric
            summary["max_delta_pct"] = round(max_abs, 2)
            summary["max_delta_scenario"] = max_scenario
            summary["max_delta_metric"] = max_metric

        # ── Causal Pathways ───────────────────────────────────────────
        elif chart_type == "causal_pathways":
            # Data may come as top-level 'pathways' key or nested 'causal_pathways'
            pathways = data.get("pathways", data.get("causal_pathways", []))
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

        # ── Ground Truth Summary ──────────────────────────────────────
        elif chart_type == "ground_truth_summary":
            # Data may come as top-level keys or nested in 'gt'
            gt = data.get("gt", data)
            config = data.get("config", gt.get("config", {}))
            stats = data.get("stats", data.get("statistics", gt.get("statistics", {})))
            users = data.get("users", gt.get("users", []))
            summary["n_users"] = len(users)
            summary["n_days"] = config.get("n_days", 0)
            # Compute persona count from users if available
            if users:
                try:
                    personas = set(u.get("persona", "") for u in users)
                    summary["n_personas"] = len(personas)
                except Exception:
                    summary["n_personas"] = data.get("n_personas", 0)
            else:
                summary["n_personas"] = data.get("n_personas", 0)
            summary["mean_interest"] = round(stats.get("mean_interest", 0), 4)
            summary["mean_treatment_effect"] = round(
                stats.get("mean_treatment_effect", stats.get("mean_treatment", 0)), 4
            )

        # ── Device Stitching ──────────────────────────────────────────
        elif chart_type == "device_stitching":
            metrics = data.get("metrics", {})
            summary["n_devices"] = metrics.get("n_devices", 0)
            summary["n_clusters"] = metrics.get("n_clusters", 0)
            summary["avg_cluster_size"] = round(metrics.get("avg_cluster_size", 0), 2)

        # ── Interaction Graphs ────────────────────────────────────────
        elif chart_type == "interaction_graphs":
            # Per-user data passed from plot_interaction_graphs
            metrics = data.get("metrics", {})
            if metrics:
                summary["n_users_with_graphs"] = metrics.get("n_users_with_graphs", 0)
                summary["total_edges"] = metrics.get("total_edges", 0)
                summary["avg_degree"] = round(metrics.get("avg_degree", 0), 4)
                summary["avg_events_per_user"] = round(
                    metrics.get("avg_events_per_user", 0), 2
                )
            else:
                # Per-user fallback: compute from edges
                edges = data.get("edges", [])
                summary["n_edges"] = len(edges)
                summary["n_events"] = data.get("n_events", 0)
                summary["user_id"] = data.get("user_id", "unknown")

        # ── Propensity Distributions ──────────────────────────────────
        elif chart_type == "propensity_distributions":
            users = data.get("users", [])
            summary["n_users"] = len(users)
            if users:
                try:
                    personas = set(u.get("persona", "") for u in users)
                    summary["n_personas"] = len(personas)
                except Exception:
                    summary["n_personas"] = data.get("n_personas", 0)
            else:
                summary["n_personas"] = data.get("n_personas", 0)

        # ── Price Sensitivity ─────────────────────────────────────────
        elif chart_type == "price_sensitivity":
            users = data.get("users", [])
            if users:
                try:
                    personas = set(u.get("persona", "") for u in users)
                    summary["n_personas"] = len(personas)
                except Exception:
                    summary["n_personas"] = data.get("n_personas", 0)
            else:
                summary["n_personas"] = data.get("n_personas", 0)

        # ── Cluster Size Distribution ─────────────────────────────────
        elif chart_type == "cluster_size_distribution":
            # Data may come as top-level keys or nested in 'metrics'
            metrics = data.get("metrics", {})
            clusters = data.get("clusters", {})
            if clusters:
                summary["n_clusters"] = len(clusters)
                cluster_sizes = [len(devices) for devices in clusters.values()]
                summary["n_devices"] = sum(cluster_sizes)
                summary["avg_cluster_size"] = (
                    round(float(np.mean(cluster_sizes)), 2) if cluster_sizes else 0
                )
            else:
                summary["n_clusters"] = metrics.get("n_clusters", 0)
                summary["n_devices"] = metrics.get("n_devices", 0)
                summary["avg_cluster_size"] = round(
                    metrics.get("avg_cluster_size", 0), 2
                )

        # ── Treatment Matrix ──────────────────────────────────────────
        elif chart_type == "treatment_matrix":
            # Data may come as pre-computed or raw lists
            treatment_effects = data.get("treatment_effects", [])
            base_affinities = data.get("base_affinities", [])
            if treatment_effects:
                summary["n_users"] = len(treatment_effects)
                summary["mean_treatment_effect"] = round(
                    float(np.mean(treatment_effects)), 4
                )
                summary["mean_base_affinity"] = (
                    round(float(np.mean(base_affinities)), 4) if base_affinities else 0
                )
            else:
                summary["n_users"] = data.get("n_users", 0)
                summary["mean_treatment_effect"] = round(
                    data.get("mean_treatment_effect", 0), 4
                )
                summary["mean_base_affinity"] = round(
                    data.get("mean_base_affinity", 0), 4
                )

        # ── Interest Heatmap ──────────────────────────────────────────
        elif chart_type == "interest_heatmap":
            summary["n_days"] = data.get("n_days", 0)
            summary["n_categories"] = data.get("n_categories", 0)
            summary["mean_interest"] = round(data.get("mean_interest", 0), 4)

        # ── User Counterfactual Comparison ────────────────────────────
        elif chart_type == "user_counterfactual_comparison":
            summary["n_users"] = data.get("n_users", 0)
            summary["n_days"] = data.get("n_days", 0)
            summary["intervention_type"] = data.get("intervention_type", "unknown")
            # Compute mean delta from DataFrame if available
            df = data.get("df")
            if df is not None and hasattr(df, "groupby"):
                try:
                    factual = (
                        df[df["scenario"] == "factual"]
                        .groupby("day")["affinity"]
                        .mean()
                    )
                    counterfactual = (
                        df[df["scenario"] == "counterfactual"]
                        .groupby("day")["affinity"]
                        .mean()
                    )
                    common_days = factual.index.intersection(counterfactual.index)
                    if len(common_days) > 0:
                        deltas = (
                            counterfactual[common_days].values
                            - factual[common_days].values
                        )
                        summary["mean_delta"] = round(float(np.mean(deltas)), 4)
                        summary["max_delta"] = round(float(np.max(np.abs(deltas))), 4)
                except Exception:
                    pass

        # ── Counterfactual Table ──────────────────────────────────────
        elif chart_type == "counterfactual_table":
            summary["n_users"] = data.get("n_users", 0)
            summary["n_days"] = data.get("n_days", 0)
            summary["intervention_type"] = data.get("intervention_type", "unknown")
            # Compute average delta from raw rows if available
            csv_rows = data.get("csv_rows", [])
            factual_keys = data.get("factual_keys", [])
            counterfactual_keys = data.get("counterfactual_keys", [])
            if csv_rows and factual_keys and counterfactual_keys:
                deltas = []
                for row in csv_rows:
                    for fk, cfk in zip(factual_keys, counterfactual_keys):
                        try:
                            deltas.append(
                                float(row.get(cfk, 0)) - float(row.get(fk, 0))
                            )
                        except (ValueError, TypeError):
                            pass
                if deltas:
                    summary["mean_delta"] = round(float(np.mean(deltas)), 4)
                    summary["positive_deltas"] = int(sum(1 for d in deltas if d > 0.01))
                    summary["negative_deltas"] = int(
                        sum(1 for d in deltas if d < -0.01)
                    )
                    summary["neutral_deltas"] = int(
                        sum(1 for d in deltas if -0.01 <= d <= 0.01)
                    )

        # ── Interest Heatmap ──────────────────────────────────────────
        elif chart_type == "interest_heatmap":
            summary["n_days"] = data.get("n_days", 0)
            summary["n_categories"] = data.get("n_categories", 0)
            summary["mean_interest"] = round(data.get("mean_interest", 0), 4)
            # Compute range from avg_interest matrix if available
            avg_interest = data.get("avg_interest")
            if avg_interest is not None and hasattr(avg_interest, "min"):
                try:
                    summary["interest_min"] = round(float(avg_interest.min()), 4)
                    summary["interest_max"] = round(float(avg_interest.max()), 4)
                except Exception:
                    pass

        # ── Generic fallback ──────────────────────────────────────────
        else:
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    summary[k] = round(v, 4) if isinstance(v, float) else v

        return json.dumps(summary, indent=2)


# ---------------------------------------------------------------------------
# LLM Prompt Templates (objective-driven)
# ---------------------------------------------------------------------------

_LLM_PROMPTS: Dict[str, str] = {
    "_default": (
        "You are a data analyst explaining a machine learning visualization "
        "to a non-technical business stakeholder.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "Chart data summary:\n{data_summary}\n\n"
        "Provide a brief explanation (2-4 sentences) covering:\n"
        "1. What this chart reveals about the data\n"
        "2. The key insight or pattern from the actual numbers\n"
        "3. A practical business implication\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "lift_deciles": (
        "You are a marketing analytics expert explaining an uplift decile chart "
        "to a campaign manager.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data above, provide a 2-4 sentence explanation:\n"
        "1. State the objective and whether the data supports it\n"
        "2. Report the key numbers (uplift range, top decile performance)\n"
        "3. Identify the actionable insight (which deciles to target/avoid)\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "qini_curve": (
        "You are a data scientist explaining a Qini curve to a marketing director.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. What the Qini curve shows (model vs random targeting)\n"
        "2. Whether the model is effective (Qini coefficient interpretation)\n"
        "3. The business value of using this model over random targeting\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "lift_summary": (
        "You are a campaign analyst presenting results to stakeholders.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. The overall campaign lift and whether it's statistically significant\n"
        "2. How it compares to industry benchmarks (Criteo)\n"
        "3. A recommendation based on the results\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "causal_trajectories": (
        "You are a behavioural analyst explaining user interest trajectories.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. How user interests evolve over time (trend direction)\n"
        "2. What the confidence bands tell us about user divergence\n"
        "3. What drives the trajectory patterns (persistence, treatment)\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "counterfactual_deltas": (
        "You are a causal inference expert explaining counterfactual analysis.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. What counterfactual scenarios were tested\n"
        "2. Which intervention had the strongest effect on graph structure\n"
        "3. What this implies for campaign strategy\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "causal_pathways": (
        "You are a causal discovery researcher explaining pathway analysis.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. What causal pathways were discovered\n"
        "2. Which pathway has the strongest effect\n"
        "3. How this helps understand the intervention mechanism\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "device_stitching": (
        "You are a privacy engineer explaining device stitching results.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. What device stitching reveals about multi-device behaviour\n"
        "2. The cluster size distribution and what it means\n"
        "3. Implications for cross-device identity resolution\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "interaction_graphs": (
        "You are a network analyst explaining user interaction graphs.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. What the interaction graphs represent\n"
        "2. Key network metrics and their interpretation\n"
        "3. How graph structure relates to user engagement\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
    "ground_truth_summary": (
        "You are a simulation engineer explaining the generated ground truth.\n\n"
        "CHART OBJECTIVE: {objective}\n\n"
        "KEY QUESTIONS:\n{key_questions}\n\n"
        "DATA SUMMARY:\n{data_summary}\n\n"
        "Based on the actual data, provide a 2-4 sentence explanation:\n"
        "1. The simulation configuration and scale\n"
        "2. Key statistics about the generated population\n"
        "3. How the data is structured for downstream analysis\n\n"
        "Use plain language. Bold key numbers with **asterisks**."
    ),
}
