#!/usr/bin/env python3
"""Phase 3: Route consistency validation.

Validates the route consistency module on real held-out data and synthetic
error cases.

Usage:
    python scripts/run_phase3.py [--graph-dir <path>] [--data-dir <path>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.coi_graph import CoIGraph
from src.graph.route_consistency import RouteConsistency
from src.intent.rule_labeler import RuleLabeler
from src.intent.taxonomy import Taxonomy
from src.utils import get_config, load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Route Consistency Validation")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    graph_dir = args.graph_dir or config["coi_graph"]["output_dir"]
    data_dir = args.data_dir or config["dataset"]["processed_dir"]

    print("[Phase 3] Route Consistency Validation")

    # Load CoI graph
    taxonomy = Taxonomy()
    coi_graph = CoIGraph.load(graph_dir, taxonomy)
    scorer = RouteConsistency(coi_graph)

    # Test on real held-out data (dev set)
    dev_path = os.path.join(data_dir, "dev.jsonl")
    if os.path.exists(dev_path):
        print("\n--- Test on real held-out data (dev set) ---")
        dev_data = load_jsonl(dev_path)
        labeler = RuleLabeler(taxonomy)
        intent_data = labeler.label_dataset(dev_data)
        scores = scorer.score_dataset(intent_data, dev_data)

        avg_scores = {}
        for key in ["edge_validity", "path_k_validity", "reachability", "prefix_validity", "route_score"]:
            vals = [s[key] for s in scores]
            avg_scores[key] = sum(vals) / len(vals) if vals else 0
        print(f"  Avg scores on real data: {json.dumps(avg_scores, indent=2)}")
    else:
        print(f"  [WARN] Dev data not found at {dev_path}")

    # Test on synthetic error cases
    print("\n--- Test on synthetic error cases ---")
    error_cases = [
        {
            "name": "valid_success_path",
            "sequence": ["Inquiry", "Positive", "Inquiry", "Action", "EndSuccess"],
            "outcome": "success",
            "expected": "high",
        },
        {
            "name": "valid_fail_path",
            "sequence": ["Inquiry", "Concern", "Reject"],
            "outcome": "fail",
            "expected": "medium-high",
        },
        {
            "name": "invalid_jump",
            "sequence": ["EndSuccess", "Reject", "Action", "Inquiry"],
            "outcome": "success",
            "expected": "low",
        },
        {
            "name": "outcome_mismatch",
            "sequence": ["Inquiry", "Reject", "Reject"],
            "outcome": "success",
            "expected": "low",
        },
        {
            "name": "single_turn",
            "sequence": ["Inquiry"],
            "outcome": "unknown",
            "expected": "neutral",
        },
    ]

    for case in error_cases:
        scores = scorer.compute_score(case["sequence"], case["outcome"])
        print(f"\n  Case: {case['name']} (expected: {case['expected']})")
        print(f"    Sequence: {case['sequence']}")
        print(f"    Scores: {json.dumps(scores, indent=4)}")

    print(f"\n[Phase 3] Done!")


if __name__ == "__main__":
    main()
