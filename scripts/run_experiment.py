#!/usr/bin/env python3
"""Phase 9: Run the main experiment.

Evaluates and compares all three selection strategies.

Usage:
    python scripts/run_experiment.py [--selected-dir <path>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.runner import ExperimentRunner
from src.graph.coi_graph import CoIGraph
from src.intent.taxonomy import Taxonomy
from src.utils import get_config, load_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description="Phase 9: Main Experiment")
    parser.add_argument("--selected-dir", type=str, default="data/selected")
    parser.add_argument("--output", type=str, default="data/eval/experiment_results.json")
    args = parser.parse_args()

    config = get_config()
    taxonomy = Taxonomy()
    graph_dir = config["coi_graph"]["output_dir"]
    data_dir = config["dataset"]["processed_dir"]

    print("[Phase 9] Main Experiment: Strategy Comparison")

    # Load CoI graph and real data
    coi_graph = CoIGraph.load(graph_dir, taxonomy)
    real_train = load_jsonl(os.path.join(data_dir, "train.jsonl"))

    # Run experiment
    runner = ExperimentRunner(coi_graph, real_train, taxonomy)
    results = runner.run_main_experiment(args.selected_dir)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_json(results, args.output)

    # Print formatted table
    table = ExperimentRunner.format_results_table(results)
    print(table)

    # Save table as text
    table_path = args.output.replace(".json", "_table.txt")
    with open(table_path, "w") as f:
        f.write(table)

    print(f"\n[Phase 9] Results saved to {args.output}")
    print(f"[Phase 9] Table saved to {table_path}")


if __name__ == "__main__":
    main()
