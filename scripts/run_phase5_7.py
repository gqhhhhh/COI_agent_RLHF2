#!/usr/bin/env python3
"""Phases 5-7: Generate synthetic pool, evaluate, and select.

Runs the synthetic pool generation, instance/global evaluation, and
three selection strategies.

Usage:
    python scripts/run_phase5_7.py [--num-dialogues <N>] [--K <K>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.global_eval import GlobalEvaluator
from src.eval.instance_eval import InstanceEvaluator
from src.graph.coi_graph import CoIGraph
from src.graph.route_consistency import RouteConsistency
from src.intent.rule_labeler import RuleLabeler
from src.intent.taxonomy import Taxonomy
from src.select.selector import DataSelector
from src.simulator.rollout import generate_synthetic_pool
from src.utils import get_config, load_jsonl, save_json, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Phases 5-7: Pool Generation, Eval, Selection")
    parser.add_argument("--num-dialogues", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    config = get_config()
    taxonomy = Taxonomy()
    labeler = RuleLabeler(taxonomy)

    num_dialogues = args.num_dialogues or config["synthetic_pool"]["num_dialogues"]
    K = args.K or config["selection"]["K"]
    graph_dir = config["coi_graph"]["output_dir"]
    pool_dir = config["synthetic_pool"]["output_dir"]
    data_dir = config["dataset"]["processed_dir"]
    selected_dir = "data/selected"

    print(f"[Phases 5-7] Pool Generation → Evaluation → Selection")
    print(f"  Num synthetic dialogues: {num_dialogues}")
    print(f"  Selection K: {K}")

    # Phase 5: Generate synthetic pool
    if not args.skip_generation:
        print(f"\n[Phase 5] Generating synthetic pool...")
        pool = generate_synthetic_pool(
            num_dialogues=num_dialogues,
            max_turns=config["synthetic_pool"]["max_turns"],
            seed=config["dataset"]["random_seed"],
            output_dir=pool_dir,
        )
    else:
        pool_path = os.path.join(pool_dir, "synthetic_pool.jsonl")
        pool = load_jsonl(pool_path)
        print(f"  Loaded existing pool: {len(pool)} dialogues")

    # Prepare intent sequences
    # Convert pool items to dialogue format for labeling
    pool_dialogues = []
    for p in pool:
        pool_dialogues.append({
            "dialogue_id": p["dialogue_id"],
            "turns": p["turns"],
            "meta": {"outcome": p.get("outcome", "unknown")},
        })

    intent_data = labeler.label_dataset(pool_dialogues)
    intent_sequences = [d["intent_sequence"] for d in intent_data]

    # Load CoI graph and real data
    coi_graph = CoIGraph.load(graph_dir, taxonomy)
    route_scorer = RouteConsistency(coi_graph)

    real_train = load_jsonl(os.path.join(data_dir, "train.jsonl"))

    # Phase 6: Evaluation
    print(f"\n[Phase 6] Evaluating synthetic pool...")
    instance_eval = InstanceEvaluator(
        route_scorer=route_scorer,
        real_dialogues=real_train,
    )
    global_eval = GlobalEvaluator(
        real_coi_graph=coi_graph,
        real_dialogues=real_train,
        taxonomy=taxonomy,
    )

    # Instance-level scores
    instance_scores = instance_eval.evaluate_dataset(pool_dialogues, intent_data)

    # Save scores
    os.makedirs(os.path.join(pool_dir), exist_ok=True)
    save_jsonl(instance_scores, os.path.join(pool_dir, "scores.jsonl"))

    # Global-level summary of full pool
    global_metrics = global_eval.evaluate(pool_dialogues, intent_sequences)
    print(f"\n  Full pool global metrics:")
    print(json.dumps(global_metrics, indent=2))

    # Phase 7: Selection
    print(f"\n[Phase 7] Running selection strategies...")
    selector = DataSelector(
        instance_evaluator=instance_eval,
        global_evaluator=global_eval,
        K=K,
        M_ratio=config["selection"]["M_ratio"],
        mc_iterations=config["selection"]["monte_carlo_iterations"],
        greedy_iterations=config["selection"]["greedy_swap_iterations"],
        seed=config["dataset"]["random_seed"],
    )

    selections = selector.select_all(pool_dialogues, instance_scores, intent_sequences)

    # Save selections
    selector.save_selections(pool, selections, selected_dir)

    # Compare
    comparison = selector.compare_selections(
        pool_dialogues, selections, intent_sequences, instance_scores
    )
    save_json(comparison, os.path.join(selected_dir, "comparison.json"))

    print(f"\n[Phase 7] Selection comparison:")
    for strategy, metrics in comparison.items():
        print(f"\n  {strategy}:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")

    print(f"\n[Phases 5-7] Done! Selected data saved to {selected_dir}/")


if __name__ == "__main__":
    main()
