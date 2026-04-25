"""Utility for loading and accessing compiled dataset priors."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_priors(config: Dict) -> Optional[Dict]:
    priors_config = config.get("priors", {})
    if not priors_config.get("enabled", False):
        return None

    priors_path = priors_config.get("path", "ecosystem_sim/data/priors/compiled_priors.json")
    if not Path(priors_path).exists():
        print(f"  [WARN] Priors file not found at {priors_path}")
        print(f"         Run: python ecosystem_sim/scripts/extract_priors.py")
        return None

    with open(priors_path, "r") as f:
        priors = json.load(f)

    n_datasets = len(priors.get("metadata", {}).get("datasets_included", []))
    print(f"  [PRIORS] Loaded {n_datasets} dataset priors from {priors_path}")
    return priors


def get_prior(priors: Optional[Dict], key_path: str, default: Any = None) -> Any:
    if priors is None:
        return default
    current = priors
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
