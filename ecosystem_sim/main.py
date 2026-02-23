"""Main orchestrator for ecosystem data simulator."""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from ecosystem_sim.core import (
    UserGenerator, CausalConfig, CausalEngine, TimeManager, StateManager
)
from ecosystem_sim.streams import (
    SearchStream, CommerceStream, GeoStream, MediaStream, EmailStream, SocialStream
)
from ecosystem_sim.intelligence import (
    GraphStitcher, PropensityModels, LiftAnalyzer
)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_taxonomy(data_dir: str) -> Dict:
    """Load taxonomy.json.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Taxonomy dictionary
    """
    with open(f"{data_dir}/taxonomy.json", 'r') as f:
        return json.load(f)


def run_simulation(config: Dict, output_dir: str):
    """Run complete ecosystem simulation.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    print("=" * 80)
    print("ECOSYSTEM DATA SIMULATOR")
    print("=" * 80)
    
    # Setup
    print("\n[SETUP] Initializing simulation...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load taxonomy
    taxonomy = load_taxonomy("ecosystem_sim/data")
    
    # Phase 1: Generate users
    print("\n[PHASE 1] Generating user population...")
    user_gen = UserGenerator(config, random_seed=config["simulation"]["random_seed"])
    users = user_gen.generate_users(config["simulation"]["n_users"])
    print(f"  ✓ Generated {len(users)} users with {len(user_gen.personas)} personas")
    
    # Phase 2: Generate causal trajectories
    print("\n[PHASE 2] Generating causal interest trajectories...")
    causal_config = CausalConfig(
        n_categories=config["causal"]["interest_categories"],
        n_days=config["simulation"]["simulation_days"],
        treatment_effect_range=tuple(config["causal"]["treatment_effect_range"]),
        persistence_factor=config["causal"]["persistence_factor"],
        random_seed=config["simulation"]["random_seed"],
    )
    causal_engine = CausalEngine(causal_config, users)
    interest_matrix = causal_engine.generate_causal_trajectories()
    print(f"  ✓ Generated trajectories: shape {interest_matrix.shape}")
    
    # Export ground truth
    ground_truth_path = output_path / "ground_truth.json"
    causal_engine.export_ground_truth(str(ground_truth_path), interest_matrix)
    print(f"  ✓ Exported ground truth to {ground_truth_path}")
    
    # Phase 3: Event generation loop
    print("\n[PHASE 3] Generating events (this may take a while)...")
    
    time_manager = TimeManager(config["simulation"]["start_date"])
    state_manager = StateManager(causal_engine, time_manager, interest_matrix)
    
    # Initialize streams
    streams = {
        "search": SearchStream(config, state_manager, taxonomy),
        "commerce": CommerceStream(config, state_manager, taxonomy),
        "geo": GeoStream(config, state_manager, taxonomy),
        "media": MediaStream(config, state_manager, taxonomy),
        "email": EmailStream(config, state_manager, taxonomy),
        "social": SocialStream(config, state_manager, taxonomy),
    }
    
    total_events = 0
    
    for day in tqdm(range(config["simulation"]["simulation_days"]), desc="Generating events"):
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
    
    print(f"  ✓ Generated {total_events} total events")
    
    # Phase 4: Export raw events
    print("\n[PHASE 4] Exporting raw event logs...")
    raw_logs_dir = output_path / "raw_logs"
    raw_logs_dir.mkdir(exist_ok=True)
    
    for stream_name, stream in streams.items():
        filepath = raw_logs_dir / f"{stream_name}_events.jsonl"
        stream.export_events(str(filepath))
        print(f"  ✓ {stream_name}: {len(stream.events)} events")
    
    # Phase 5: Intelligence layer
    print("\n[PHASE 5] Running intelligence layer...")
    
    # Collect all events
    all_events = []
    for stream in streams.values():
        all_events.extend(stream.events)
    
    print(f"  Total events: {len(all_events)}")
    
    # Graph stitching
    print("  [5a] Device stitching...")
    stitcher = GraphStitcher(confidence_threshold=0.7)
    stitched_data = stitcher.stitch_events(all_events)
    
    stitched_path = output_path / "stitched_data.json"
    with open(stitched_path, 'w') as f:
        json.dump({
            "clusters": stitched_data["device_clusters"],
            "metrics": stitched_data["metrics"],
        }, f, indent=2)
    print(f"  ✓ Stitched {stitched_data['metrics']['n_devices']} devices -> {len(stitched_data['device_clusters'])} clusters")
    
    # Propensity models
    print("  [5b] Calculating propensity scores...")
    propensity = PropensityModels()
    propensity_scores = propensity.calculate_all(stitched_data)
    print(f"  ✓ Calculated scores for all users")
    
    # Lift analysis
    print("  [5c] Analyzing campaign lift...")
    lift_analyzer = LiftAnalyzer(causal_engine)
    lift_report = lift_analyzer.calculate_lift(
        campaign_id="campaign_001",
        users=users,
        interest_matrix=interest_matrix,
        campaign_day=14,  # Mid-simulation
    )
    
    lift_path = output_path / "lift_report.json"
    with open(lift_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        report_copy = {
            "campaign_id": lift_report["campaign_id"],
            "lift_percent": float(lift_report["lift_percent"]),
            "p_value": float(lift_report["p_value"]),
            "auuc": float(lift_report["auuc"]),
            "n_exposed_users": len(lift_report["exposed_users"]),
        }
        json.dump(report_copy, f, indent=2)
    
    print(f"  ✓ Campaign lift: {lift_report['lift_percent']:.1f}% (p={lift_report['p_value']:.4f})")
    
    # Summary report
    print("\n[SUMMARY]")
    print(f"  Simulation period: {config['simulation']['start_date']} to +{config['simulation']['simulation_days']} days")
    print(f"  Users: {len(users)}")
    print(f"  Interest categories: {config['causal']['interest_categories']}")
    print(f"  Total events: {total_events}")
    print(f"  Output directory: {output_path}")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ecosystem Data Simulator")
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
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run simulation
    run_simulation(config, args.output)


if __name__ == "__main__":
    main()
