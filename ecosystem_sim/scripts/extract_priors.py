"""CLI tool for extracting dataset priors."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ecosystem_sim.core.dataset_priors import DatasetPriorExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract statistical priors from real datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ecosystem_sim/data",
        help="Path to data directory containing datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ecosystem_sim/data/priors/compiled_priors.json",
        help="Output path for compiled_priors.json",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET PRIOR EXTRACTION")
    print("=" * 60)

    extractor = DatasetPriorExtractor(data_dir=args.data_dir)
    extractor.compile_all_priors()
    extractor.save_priors(args.output)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
