#!/usr/bin/env python3
"""Phase 2: Intent labeling and CoI graph construction.

Labels user turns with intent categories and builds the CoI transition graph.

Usage:
    python scripts/run_phase2.py [--data-dir <path>] [--output-dir <path>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent.rule_labeler import RuleLabeler
from src.intent.taxonomy import Taxonomy
from src.graph.coi_graph import CoIGraph
from src.utils import get_config, load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Intent Labeling + CoI Graph")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = get_config()
    data_dir = args.data_dir or config["dataset"]["processed_dir"]
    output_dir = args.output_dir or config["coi_graph"]["output_dir"]

    print(f"[Phase 2] Intent Labeling + CoI Graph Construction")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")

    # Load taxonomy and labeler
    taxonomy = Taxonomy()
    print(f"  Taxonomy: {taxonomy}")
    labeler = RuleLabeler(taxonomy)

    # Load training data
    train_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_path):
        print(f"[ERROR] Training data not found at {train_path}")
        print("[INFO] Run Phase 1 first: python scripts/run_phase1.py --demo")
        return

    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training dialogues")

    # 2.1: Label intents
    print("\n[Phase 2.1] Labeling user intents...")
    intent_data = labeler.label_dataset(train_data)

    # Save intent sequences
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(intent_data, os.path.join(output_dir, "intent_sequences.jsonl"))

    # Print some examples
    print("\n  Example intent sequences:")
    for item in intent_data[:3]:
        print(f"    {item['dialogue_id']}: {item['intent_sequence']}")

    # Intent distribution
    from collections import Counter
    all_intents = []
    for item in intent_data:
        all_intents.extend(item["intent_sequence"])
    intent_dist = Counter(all_intents)
    print(f"\n  Intent distribution: {dict(intent_dist)}")

    # 2.2: Build CoI graph
    print("\n[Phase 2.2] Building CoI graph...")
    coi_graph = CoIGraph(
        taxonomy=taxonomy,
        laplace_smoothing=config["coi_graph"]["laplace_smoothing"],
        ngram_sizes=config["coi_graph"]["ngram_sizes"],
    )
    coi_graph.build_from_sequences(intent_data, train_data)

    # Save graph
    coi_graph.save(output_dir)
    print(f"  Graph saved to {output_dir}/")

    # Generate heatmap
    heatmap_path = os.path.join(output_dir, "coi_heatmap.png")
    coi_graph.plot_heatmap(heatmap_path)

    # Print analysis
    report = coi_graph.get_analysis_report()
    print(f"\n{report}")

    # Save report
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.write(report)

    print(f"\n[Phase 2] Done!")


if __name__ == "__main__":
    main()
