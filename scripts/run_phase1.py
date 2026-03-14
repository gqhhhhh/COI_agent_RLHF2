#!/usr/bin/env python3
"""Phase 1: Data preprocessing script.

Loads PersuasionForGood data (or generates demo data) and converts it
to the unified dialogue schema.

Usage:
    python scripts/run_phase1.py [--raw-dir <path>] [--output-dir <path>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import generate_demo_data, load_real_data
from src.utils import get_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data Preprocessing")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    args = parser.parse_args()

    config = get_config()
    raw_dir = args.raw_dir or config["dataset"]["raw_dir"]
    output_dir = args.output_dir or config["dataset"]["processed_dir"]
    seed = config["dataset"].get("random_seed", 42)

    print(f"[Phase 1] Data Preprocessing")
    print(f"  Raw dir: {raw_dir}")
    print(f"  Output dir: {output_dir}")

    if args.demo:
        stats = generate_demo_data(output_dir, seed)
    else:
        stats = load_real_data(raw_dir, output_dir, seed)

    print(f"\n[Phase 1] Data Statistics:")
    print(json.dumps(stats, indent=2))
    print(f"\n[Phase 1] Done! Output saved to {output_dir}/")


if __name__ == "__main__":
    main()
