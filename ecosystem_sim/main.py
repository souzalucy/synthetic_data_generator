"""Main orchestrator for ecosystem data simulator."""

import json
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from tqdm import tqdm

from ecosystem_sim.core import (
    UserGenerator,
    CausalConfig,
    CausalEngine,
    TimeManager,
    StateManager,
    detect_device,
    device_summary,
)
from ecosystem_sim.core.prior_loader import load_priors
from ecosystem_sim.streams import (
    SearchStream,
    CommerceStream,
    GeoStream,
    MediaStream,
    EmailStream,
    SocialStream,
    NetworkStream,
    UrbanStream,
)
from ecosystem_sim.intelligence import (
    GraphStitcher,
    CounterfactualGraphStitcher,
    PropensityModels,
    LiftAnalyzer,
)
from ecosystem_sim.visualization.dashboard import generate_all_visualizations


def _convert_to_native(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python types for JSON serialization.

    This handles numpy.bool_, numpy.integer, numpy.floating, numpy.ndarray,
    and any nested structures containing them.
    """
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_native(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_native(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_taxonomy(data_dir: str) -> Dict:
    """Load taxonomy.json.

    Args:
        data_dir: Path to data directory

    Returns:
        Taxonomy dictionary
    """
    taxonomy_path = Path(data_dir) / "taxonomy.json"
    if not taxonomy_path.exists():
        print(
            f"  [WARN] Taxonomy file not found at {taxonomy_path}, using empty taxonomy"
        )
        return {}
    with open(taxonomy_path, "r") as f:
        return json.load(f)


def _apply_cli_overrides(config: Dict, args: argparse.Namespace):
    """Overlay CLI arguments onto config dict. CLI takes precedence over YAML.

    Args:
        config: Configuration dictionary loaded from YAML
        args: Parsed CLI arguments
    """
    # Simulation parameters
    if args.n_users is not None:
        config["simulation"]["n_users"] = args.n_users
    if args.days is not None:
        config["simulation"]["simulation_days"] = args.days
    if args.start_date is not None:
        config["simulation"]["start_date"] = args.start_date
    if args.seed is not None:
        config["simulation"]["random_seed"] = args.seed

    # Persona control
    config.setdefault("persona", {})
    if args.n_personas is not None:
        config["persona"]["n_personas"] = args.n_personas
    if args.n_sectors is not None:
        config["persona"]["n_sectors"] = args.n_sectors

    # Causal parameters
    if args.categories is not None:
        config["causal"]["interest_categories"] = args.categories
    if args.treatment_range is not None:
        config["causal"]["treatment_effect_range"] = list(args.treatment_range)
    if args.persistence is not None:
        config["causal"]["persistence_factor"] = args.persistence

    # SCM
    if args.scm is not None:
        config.setdefault("scm", {})
        config["scm"]["enabled"] = args.scm
    if args.counterfactuals is not None:
        config.setdefault("scm", {}).setdefault("counterfactual_graphs", {})
        config["scm"]["counterfactual_graphs"]["enabled"] = args.counterfactuals

    # Stream toggles and rates
    for stream in [
        "search",
        "commerce",
        "geo",
        "media",
        "email",
        "social",
        "network",
        "urban",
    ]:
        enable_attr = f"stream_{stream}"
        if hasattr(args, enable_attr) and getattr(args, enable_attr) is not None:
            config["streams"][stream]["enabled"] = getattr(args, enable_attr)
        rate_attr = f"{stream}_rate"
        rate_val = getattr(args, rate_attr.replace("-", "_"), None)
        if rate_val is not None:
            config["streams"][stream]["event_rate"] = rate_val

    # Visualization
    if args.viz is not None:
        config.setdefault("visualization", {})
        config["visualization"]["enabled"] = args.viz

    # Intelligence layer parameters
    config.setdefault("intelligence", {})
    if args.confidence_threshold is not None:
        config["intelligence"]["confidence_threshold"] = args.confidence_threshold
    if args.campaign_day is not None:
        config["intelligence"]["campaign_day"] = args.campaign_day
    if args.campaign_id is not None:
        config["intelligence"]["campaign_id"] = args.campaign_id


def run_simulation(config: Dict, output_dir: str):
    """Run complete ecosystem simulation.

    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    print("=" * 80)
    print("ECOSYSTEM DATA SIMULATOR")
    print("=" * 80)

    # Device detection
    print("\n[DEVICE] Detecting compute hardware...")
    device_type = detect_device()
    summary = device_summary()
    print(f"  [DEVICE] Compute device: {device_type.upper()} ({summary})")
    if device_type == "cpu":
        print(
            "  [DEVICE] Tip: Install PyTorch with ROCm support to use your AMD GPU.\n"
            "           pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
        )

    # Setup
    print("\n[SETUP] Initializing simulation...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load taxonomy
    taxonomy = load_taxonomy("ecosystem_sim/data")

    # Load dataset priors
    priors = load_priors(config)

    # Phase 1: Generate users
    print("\n[PHASE 1] Generating user population...")
    user_gen = UserGenerator(
        config, random_seed=config["simulation"]["random_seed"], priors=priors
    )
    users = user_gen.generate_users(config["simulation"]["n_users"])
    print(f"  * Generated {len(users)} users with {len(user_gen.personas)} personas")

    # Phase 2: Generate causal trajectories using SCM
    print("\n[PHASE 2] Generating causal interest trajectories (SCM)...")
    causal_config = CausalConfig(
        n_categories=config["causal"]["interest_categories"],
        n_days=config["simulation"]["simulation_days"],
        treatment_effect_range=tuple(config["causal"]["treatment_effect_range"]),
        persistence_factor=config["causal"]["persistence_factor"],
        random_seed=config["simulation"]["random_seed"],
        use_scm=config.get("scm", {}).get("enabled", True),
    )
    causal_engine = CausalEngine(causal_config, users, priors=priors)
    interest_matrix = causal_engine.generate_causal_trajectories()
    print(f"  * Generated trajectories: shape {interest_matrix.shape}")
    print(
        f"  * SCM covariance matrix: {causal_engine.scm_generator.get_covariance_matrix().shape}"
    )

    # Export ground truth (JSON summary)
    ground_truth_path = output_path / "ground_truth.json"
    causal_engine.export_ground_truth(str(ground_truth_path), interest_matrix)
    print(f"  * Exported ground truth to {ground_truth_path}")

    # Export causal table (wide-format CSV with per-day base_affinity)
    causal_table_path = output_path / "causal_ground_truth.csv"
    causal_engine.export_causal_table(str(causal_table_path), interest_matrix)
    print(f"  * Exported causal ground truth table to {causal_table_path}")

    # Export causal table with counterfactual (dual columns: factual + counterfactual)
    # Uses time-localized intervention: treatment is zeroed from campaign_day onward
    campaign_day = config.get("intelligence", {}).get("campaign_day", 14)
    counterfactual_matrix = (
        causal_engine.generate_time_localized_no_treatment_counterfactual(
            interest_matrix, intervention_day=campaign_day
        )
    )
    causal_cf_table_path = output_path / "causal_table_with_counterfactual.csv"
    causal_engine.export_causal_table_with_counterfactual(
        str(causal_cf_table_path),
        interest_matrix,
        counterfactual_matrix,
        intervention_type="no_treatment",
        intervention_value=f"do(T=0 from day {campaign_day})",
    )
    print(f"  * Exported causal table with counterfactual to {causal_cf_table_path}")

    # Phase 3: Event generation loop
    print("\n[PHASE 3] Generating events (this may take a while)...")

    time_manager = TimeManager(config["simulation"]["start_date"], priors=priors)
    state_manager = StateManager(
        causal_engine, time_manager, interest_matrix, priors=priors
    )

    # Initialize streams
    streams = {
        "search": SearchStream(config, state_manager, taxonomy, priors=priors),
        "commerce": CommerceStream(config, state_manager, taxonomy, priors=priors),
        "geo": GeoStream(config, state_manager, taxonomy, priors=priors),
        "media": MediaStream(config, state_manager, taxonomy, priors=priors),
        "email": EmailStream(config, state_manager, taxonomy, priors=priors),
        "social": SocialStream(config, state_manager, taxonomy, priors=priors),
        "network": NetworkStream(config, state_manager, taxonomy, priors=priors),
        "urban": UrbanStream(config, state_manager, taxonomy, priors=priors),
    }

    total_events = 0

    for day in tqdm(
        range(config["simulation"]["simulation_days"]), desc="Generating events"
    ):
        for hour in range(24):
            timestamp = time_manager.get_timestamp(hour)

            for user in users:
                # Get action probabilities for this user
                probabilities = state_manager.get_daily_action_probabilities(
                    user, day, hour, users
                )

                # Generate events from enabled streams
                for stream_name, stream in streams.items():
                    if config["streams"][stream_name]["enabled"]:
                        events = stream.generate_event(user, timestamp, probabilities)
                        total_events += len(events)

        # Checkpoint every N days
        if (day + 1) % config["output"]["checkpoint_interval"] == 0:
            print(f"  [Checkpoint] Day {day + 1}: {total_events} events generated")

        # Advance time
        time_manager.advance_day()

    print(f"  * Generated {total_events} total events")

    # Phase 4: Export raw events
    print("\n[PHASE 4] Exporting raw event logs...")
    raw_logs_dir = output_path / "raw_logs"
    raw_logs_dir.mkdir(exist_ok=True)

    for stream_name, stream in streams.items():
        filepath = raw_logs_dir / f"{stream_name}_events.jsonl"
        stream.export_events(str(filepath))
        print(f"  * {stream_name}: {len(stream.events)} events")

    # Phase 5: Intelligence layer
    print("\n[PHASE 5] Running intelligence layer...")

    # Collect all events
    all_events = []
    for stream in tqdm(streams.values(), desc="  Collecting events", leave=False):
        all_events.extend(stream.events)

    print(f"  Total events: {len(all_events)}")

    # Read intelligence config (CLI overrides or defaults)
    intel_config = config.get("intelligence", {})
    confidence_threshold = intel_config.get("confidence_threshold", 0.7)
    campaign_day = intel_config.get("campaign_day", 14)
    campaign_id = intel_config.get("campaign_id", "campaign_001")

    # Graph stitching
    print("  [5a] Device stitching...")
    stitcher = GraphStitcher(confidence_threshold=confidence_threshold)
    stitched_data = stitcher.stitch_events(all_events)

    stitched_path = output_path / "stitched_data.json"
    with open(stitched_path, "w") as f:
        json.dump(
            _convert_to_native(
                {
                    "clusters": stitched_data["device_clusters"],
                    "metrics": stitched_data["metrics"],
                    "inferred_user_ids": stitched_data.get("inferred_user_ids", {}),
                }
            ),
            f,
            indent=2,
        )
    print(
        f"  * Stitched {stitched_data['metrics']['n_devices']} devices -> {len(stitched_data['device_clusters'])} clusters"
    )
    print(
        f"  * Inferred user_ids for {len(stitched_data.get('inferred_user_ids', {}))} unidentified events"
    )

    # Interaction graph construction (Layer 3)
    print("  [5b] Building interaction graphs...")
    interaction_graphs = stitcher.build_interaction_graphs(all_events)
    ig_metrics = interaction_graphs["metrics"]
    print(
        f"  * Built {ig_metrics['n_users_with_graphs']} user graphs: "
        f"{ig_metrics['total_edges']} edges across {ig_metrics['total_events']} events"
    )

    # Export interaction graphs
    ig_path = output_path / "interaction_graphs.json"
    stitcher.export_interaction_graphs(interaction_graphs, str(ig_path))
    print(f"  * Exported interaction graphs to {ig_path}")

    # Propensity models
    print("  [5c] Calculating propensity scores...")
    propensity = PropensityModels(priors=priors)
    propensity_scores = propensity.calculate_all(stitched_data)
    print(f"  * Calculated scores for all users")

    # Lift analysis
    print("  [5d] Analyzing campaign lift...")
    lift_analyzer = LiftAnalyzer(causal_engine, priors=priors)
    lift_report = lift_analyzer.calculate_lift(
        campaign_id=campaign_id,
        users=users,
        interest_matrix=interest_matrix,
        campaign_day=campaign_day,
    )

    lift_path = output_path / "lift_report.json"
    with open(lift_path, "w") as f:
        json.dump(_convert_to_native(lift_report), f, indent=2)

    print(
        f"  * Campaign lift: {lift_report['lift_percent']:.1f}% (p={lift_report['p_value']:.4f})"
    )

    # Phase 6: Counterfactual graph analysis (SCM-powered)
    cf_path = output_path / "counterfactual_graph_report.json"
    if config.get("scm", {}).get("counterfactual_graphs", {}).get("enabled", False):
        print("\n[PHASE 6] Generating counterfactual graph worlds...")

        cf_stitcher = CounterfactualGraphStitcher(
            confidence_threshold=confidence_threshold
        )

        cf_report = cf_stitcher.generate_all_scenario_graphs(
            causal_engine=causal_engine,
            users=users,
            factual_interest_matrix=interest_matrix,
            intervention_day=campaign_day,
        )

        # Export counterfactual report
        cf_path = output_path / "counterfactual_graph_report.json"
        cf_stitcher.export_counterfactual_report(cf_report, str(cf_path))
        print(f"  * Exported counterfactual graph report to {cf_path}")

        # Print summary of causal pathways
        print("\n  [Causal Pathway Discovery]")
        for pathway in cf_report.get("causal_pathways", []):
            print(f"    • {pathway.get('interpretation', 'N/A')}")

        # Print comparison deltas
        comparison = cf_report.get("comparison", {})
        deltas = comparison.get("deltas", {})
        if deltas:
            print("\n  [Graph-Level Treatment Effects]")
            for scenario, delta in deltas.items():
                edge_delta = delta.get("delta_total_edges", 0)
                density_delta = delta.get("delta_edge_density", 0)
                print(
                    f"    • {scenario}: edges={edge_delta:+.1f}%, "
                    f"density={density_delta:+.2f}%"
                )

        # Export individual scenario graphs
        cf_graphs_dir = output_path / "counterfactual_graphs"
        cf_graphs_dir.mkdir(exist_ok=True)

        for scenario_name, scenario_data in cf_report.get("scenarios", {}).items():
            scenario_path = cf_graphs_dir / f"{scenario_name}_metrics.json"
            with open(scenario_path, "w") as f:
                json.dump(_convert_to_native(scenario_data), f, indent=2)

        print(
            f"  * Exported {len(cf_report.get('scenarios', {}))} scenario graph metrics"
        )
    else:
        print("\n[PHASE 6] Counterfactual graph analysis disabled (enable in config)")

    # Phase 7: Generate visualizations
    if config.get("visualization", {}).get("enabled", False):
        print("\n[PHASE 7] Generating visualizations...")
        viz_output = config["visualization"].get("output_dir", str(output_path))
        viz_files = generate_all_visualizations(
            output_dir=str(output_path),
            ground_truth_path=str(ground_truth_path),
            causal_csv_path=str(causal_table_path),
            causal_cf_csv_path=str(causal_cf_table_path),
            interaction_graphs_path=str(ig_path),
            stitched_data_path=str(stitched_path),
            lift_report_path=str(lift_path),
            counterfactual_report_path=str(cf_path)
            if config.get("scm", {})
            .get("counterfactual_graphs", {})
            .get("enabled", False)
            else None,
            config=config,
        )
        print(f"  * Generated {len(viz_files)} visualization files")
        if "index" in viz_files:
            print(f"  * Index page: {viz_files['index']}")
    else:
        print("\n[PHASE 7] Visualizations disabled (enable in config)")

    # Summary report
    print("\n[SUMMARY]")
    print(
        f"  Simulation period: {config['simulation']['start_date']} to +{config['simulation']['simulation_days']} days"
    )
    print(f"  Users: {len(users)}")
    print(f"  Interest categories: {config['causal']['interest_categories']}")
    print(f"  SCM enabled: {config.get('scm', {}).get('enabled', True)}")
    print(f"  Total events: {total_events}")
    print(f"  Output directory: {output_path}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ecosystem Data Simulator")

    # Config / output
    parser.add_argument(
        "--config",
        type=str,
        default="ecosystem_sim/config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ecosystem_sim/outputs",
        help="Output directory",
    )

    # Simulation parameters
    parser.add_argument(
        "--n-users", type=int, default=None, help="Number of synthetic users"
    )
    parser.add_argument(
        "--days", type=int, default=None, help="Number of simulation days"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Simulation start date (ISO format)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Persona control
    parser.add_argument(
        "--n-personas",
        type=int,
        default=None,
        help="Number of persona clusters to use from priors",
    )
    parser.add_argument(
        "--n-sectors",
        type=int,
        default=None,
        help="Number of urban sectors to use",
    )

    # Causal / SCM
    parser.add_argument(
        "--categories",
        type=int,
        default=None,
        help="Number of interest categories",
    )
    parser.add_argument(
        "--treatment-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Treatment effect range (low high)",
    )
    parser.add_argument(
        "--persistence",
        type=float,
        default=None,
        help="Persistence factor for causal trajectories",
    )
    parser.add_argument(
        "--scm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable SCM",
    )
    parser.add_argument(
        "--counterfactuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable counterfactual graph analysis",
    )

    # Intelligence layer
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for device stitching (default: 0.7)",
    )
    parser.add_argument(
        "--campaign-day",
        type=int,
        default=None,
        help="Day of campaign for lift analysis (default: 14)",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Campaign identifier for lift analysis (default: campaign_001)",
    )

    # Stream toggles and rates
    for stream in [
        "search",
        "commerce",
        "geo",
        "media",
        "email",
        "social",
        "network",
        "urban",
    ]:
        parser.add_argument(
            f"--enable-{stream}",
            action="store_true",
            dest=f"stream_{stream}",
            default=None,
            help=f"Enable {stream} stream",
        )
        parser.add_argument(
            f"--disable-{stream}",
            action="store_false",
            dest=f"stream_{stream}",
            default=None,
            help=f"Disable {stream} stream",
        )
        parser.add_argument(
            f"--{stream}-rate",
            type=float,
            default=None,
            help=f"{stream} event rate (events per user per day)",
        )

    # Visualization
    parser.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable visualization generation",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides on top of YAML config
    _apply_cli_overrides(config, args)

    # Run simulation
    run_simulation(config, args.output)


if __name__ == "__main__":
    main()
