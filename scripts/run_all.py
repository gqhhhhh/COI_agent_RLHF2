#!/usr/bin/env python3
"""Run all phases end-to-end.

Usage:
    python scripts/run_all.py [--demo] [--num-dialogues <N>] [--K <K>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import generate_demo_data, load_real_data
from src.eval.global_eval import GlobalEvaluator
from src.eval.instance_eval import InstanceEvaluator
from src.experiments.runner import ExperimentRunner
from src.graph.coi_graph import CoIGraph
from src.graph.route_consistency import RouteConsistency
from src.intent.rule_labeler import RuleLabeler
from src.intent.taxonomy import Taxonomy
from src.rm.preference_builder import PreferencePairBuilder
from src.select.selector import DataSelector
from src.simulator.rollout import generate_synthetic_pool
from src.utils import get_config, load_jsonl, save_json, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Run full CoI-Select pipeline")
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    parser.add_argument("--num-dialogues", type=int, default=200,
                        help="Number of synthetic dialogues to generate")
    parser.add_argument("--K", type=int, default=50,
                        help="Number of dialogues to select (K)")
    args = parser.parse_args()

    config = get_config()
    taxonomy = Taxonomy()
    labeler = RuleLabeler(taxonomy)

    data_dir = config["dataset"]["processed_dir"]
    graph_dir = config["coi_graph"]["output_dir"]
    pool_dir = config["synthetic_pool"]["output_dir"]
    selected_dir = "data/selected"
    eval_dir = "data/eval"

    # =========================================================================
    # Phase 1: Data Preprocessing
    # =========================================================================
    print("=" * 60)
    print("Phase 1: Data Preprocessing")
    print("=" * 60)
    if args.demo:
        stats = generate_demo_data(data_dir, config["dataset"]["random_seed"])
    else:
        stats = load_real_data(
            config["dataset"]["raw_dir"], data_dir, config["dataset"]["random_seed"]
        )
    print(f"  Data stats: {json.dumps(stats, indent=2)}")

    # =========================================================================
    # Phase 2: Intent Labeling + CoI Graph
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Intent Labeling + CoI Graph")
    print("=" * 60)
    train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    intent_data = labeler.label_dataset(train_data)
    save_jsonl(intent_data, os.path.join(graph_dir, "intent_sequences.jsonl"))

    coi_graph = CoIGraph(
        taxonomy=taxonomy,
        laplace_smoothing=config["coi_graph"]["laplace_smoothing"],
        ngram_sizes=config["coi_graph"]["ngram_sizes"],
    )
    coi_graph.build_from_sequences(intent_data, train_data)
    coi_graph.save(graph_dir)
    print(f"  CoI graph built with {len(coi_graph.valid_edges)} valid edges")

    try:
        coi_graph.plot_heatmap(os.path.join(graph_dir, "coi_heatmap.png"))
    except Exception:
        pass

    # =========================================================================
    # Phase 3: Route Consistency Validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Route Consistency Validation")
    print("=" * 60)
    route_scorer = RouteConsistency(coi_graph)

    # Test on dev set
    dev_path = os.path.join(data_dir, "dev.jsonl")
    if os.path.exists(dev_path):
        dev_data = load_jsonl(dev_path)
        dev_intents = labeler.label_dataset(dev_data)
        dev_scores = route_scorer.score_dataset(dev_intents, dev_data)
        avg_route = sum(s["route_score"] for s in dev_scores) / len(dev_scores)
        print(f"  Avg route score on dev set: {avg_route:.4f}")

    # =========================================================================
    # Phase 5: Synthetic Pool Generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 5: Synthetic Pool Generation")
    print("=" * 60)
    pool = generate_synthetic_pool(
        num_dialogues=args.num_dialogues,
        max_turns=config["synthetic_pool"]["max_turns"],
        seed=config["dataset"]["random_seed"],
        output_dir=pool_dir,
    )

    # =========================================================================
    # Phase 6: Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 6: Instance + Global Evaluation")
    print("=" * 60)
    pool_dialogues = [
        {
            "dialogue_id": p["dialogue_id"],
            "turns": p["turns"],
            "meta": {"outcome": p.get("outcome", "unknown")},
        }
        for p in pool
    ]
    pool_intent_data = labeler.label_dataset(pool_dialogues)
    pool_sequences = [d["intent_sequence"] for d in pool_intent_data]

    instance_eval = InstanceEvaluator(
        route_scorer=route_scorer,
        real_dialogues=train_data,
    )
    global_eval = GlobalEvaluator(
        real_coi_graph=coi_graph,
        real_dialogues=train_data,
        taxonomy=taxonomy,
    )

    instance_scores = instance_eval.evaluate_dataset(pool_dialogues, pool_intent_data)
    global_metrics = global_eval.evaluate(pool_dialogues, pool_sequences)
    print(f"  Full pool global metrics:")
    for k, v in global_metrics.items():
        print(f"    {k}: {v:.4f}")

    save_jsonl(instance_scores, os.path.join(pool_dir, "scores.jsonl"))

    # =========================================================================
    # Phase 7: Selection
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 7: Data Selection")
    print("=" * 60)
    selector = DataSelector(
        instance_evaluator=instance_eval,
        global_evaluator=global_eval,
        K=args.K,
        M_ratio=config["selection"]["M_ratio"],
        mc_iterations=min(config["selection"]["monte_carlo_iterations"], 100),
        greedy_iterations=min(config["selection"]["greedy_swap_iterations"], 50),
        seed=config["dataset"]["random_seed"],
    )

    selections = selector.select_all(pool_dialogues, instance_scores, pool_sequences)
    selector.save_selections(pool, selections, selected_dir)

    comparison = selector.compare_selections(
        pool_dialogues, selections, pool_sequences, instance_scores
    )
    save_json(comparison, os.path.join(selected_dir, "comparison.json"))

    print(f"\n  Selection comparison:")
    for strategy, metrics in comparison.items():
        print(f"\n  {strategy}:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")

    # =========================================================================
    # Phase 8.2: Preference Pair Construction
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 8.2: Preference Pair Construction")
    print("=" * 60)
    pair_builder = PreferencePairBuilder(seed=config["dataset"]["random_seed"])
    pairs = pair_builder.build_pairs(
        pool_dialogues, instance_scores, pool_sequences,
        max_pairs=min(args.K * 2, 200),
    )
    os.makedirs(eval_dir, exist_ok=True)
    pair_builder.save_pairs(pairs, os.path.join(eval_dir, "preference_pairs.jsonl"))

    # =========================================================================
    # Phase 9: Experiment
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 9: Main Experiment")
    print("=" * 60)
    runner = ExperimentRunner(coi_graph, train_data, taxonomy)
    results = runner.run_main_experiment(selected_dir)
    save_json(results, os.path.join(eval_dir, "experiment_results.json"))

    table = ExperimentRunner.format_results_table(results)
    print(table)

    with open(os.path.join(eval_dir, "experiment_results_table.txt"), "w") as f:
        f.write(table)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nKey outputs:")
    print(f"  Processed data: {data_dir}/")
    print(f"  CoI graph: {graph_dir}/")
    print(f"  Synthetic pool: {pool_dir}/")
    print(f"  Selected data: {selected_dir}/")
    print(f"  Experiment results: {eval_dir}/")


if __name__ == "__main__":
    main()
