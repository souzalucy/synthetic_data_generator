# Ecosystem Data Simulator

A comprehensive synthetic data generator for multi-service user ecosystems with causal inference, event synthesis, and intelligent stitching.

## Architecture

The simulator is built in three layers:

1. **Causal Core** ("The Unconscious"): Generates ground-truth interest trajectories using causal inference
2. **Event Synthesizer** ("The Conscious"): Converts causal trends into discrete service events  
3. **Intelligence Graph** ("The Observer"): Stitches fragmented data and performs analytics

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run simulation with default config
python ecosystem_sim/main.py

# View results
ls ecosystem_sim/outputs/
```

## Project Structure

```
ecosystem_sim/
├── core/              # Causal inference engine
│   ├── user_agent.py
│   ├── causal_engine.py
│   ├── time_manager.py
│   └── state_manager.py
├── streams/           # Event generation
│   ├── base_stream.py
│   ├── search_stream.py
│   ├── commerce_stream.py
│   ├── geo_stream.py
│   ├── media_stream.py
│   ├── email_stream.py
│   └── social_stream.py
├── intelligence/      # Analytics & stitching
│   ├── graph_stitcher.py
│   ├── propensity.py
│   └── lift_analyzer.py
├── data/             # Taxonomies and provenance
├── config/           # Configuration files
├── tests/            # Unit and integration tests
├── main.py           # Orchestrator
└── outputs/          # Generated data (gitignored)
```

## Key Features

- **Causal Modeling**: AR(1) process with treatment effects
- **Multi-stream Events**: Search, Commerce, Geo, Media, Email, Social
- **Device Stitching**: Probabilistic multi-device linking
- **Propensity Models**: LTV, Churn, Conversion prediction
- **Lift Analysis**: Counterfactual reasoning for A/B tests
- **Realistic Behaviors**: Persona-driven, temporal patterns, device selection

## Documentation

See `plans/complete_implementation_plan.md` for detailed specifications.
