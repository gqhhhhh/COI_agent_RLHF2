"""Phase 9: Experiment runner and evaluator.

Runs the main experiment comparing Agent-Random, Agent-Instance, and Agent-CoI.
Computes both data-level and agent-level metrics.
Also runs ablation experiments.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.eval.global_eval import GlobalEvaluator
from src.eval.instance_eval import InstanceEvaluator
from src.graph.coi_graph import CoIGraph
from src.graph.route_consistency import RouteConsistency
from src.intent.rule_labeler import RuleLabeler
from src.intent.taxonomy import Taxonomy
from src.utils import load_jsonl, save_json


class ExperimentRunner:
    """Runs and evaluates the main comparative experiment."""

    def __init__(
        self,
        coi_graph: CoIGraph,
        real_dialogues: List[Dict[str, Any]],
        taxonomy: Taxonomy | None = None,
    ):
        self.taxonomy = taxonomy or Taxonomy()
        self.coi_graph = coi_graph
        self.real_dialogues = real_dialogues
        self.labeler = RuleLabeler(self.taxonomy)
        self.route_scorer = RouteConsistency(coi_graph)
        self.instance_eval = InstanceEvaluator(
            route_scorer=self.route_scorer,
            real_dialogues=real_dialogues,
        )
        self.global_eval = GlobalEvaluator(
            real_coi_graph=coi_graph,
            real_dialogues=real_dialogues,
            taxonomy=self.taxonomy,
        )

    def evaluate_data_quality(
        self,
        strategy_name: str,
        dialogues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate data-level quality metrics for a selection strategy.

        Returns comprehensive metrics dict.
        """
        # Label intents
        intent_data = self.labeler.label_dataset(dialogues)
        intent_sequences = [d["intent_sequence"] for d in intent_data]

        # Instance-level metrics
        instance_scores = self.instance_eval.evaluate_dataset(dialogues, intent_data)
        avg_instance = np.mean([s["instance_score"] for s in instance_scores])
        avg_route = np.mean([s["route_consistency"] for s in instance_scores])

        # Global-level metrics
        global_metrics = self.global_eval.evaluate(dialogues, intent_sequences)

        # Route scores
        route_scores = self.route_scorer.score_dataset(intent_data, dialogues)
        avg_route_detail = {
            "edge_validity": np.mean([s["edge_validity"] for s in route_scores]),
            "path_k_validity": np.mean([s["path_k_validity"] for s in route_scores]),
            "reachability": np.mean([s["reachability"] for s in route_scores]),
            "prefix_validity": np.mean([s["prefix_validity"] for s in route_scores]),
            "route_score": np.mean([s["route_score"] for s in route_scores]),
        }

        return {
            "strategy": strategy_name,
            "num_dialogues": len(dialogues),
            "avg_instance_score": float(avg_instance),
            "avg_route_score": float(avg_route),
            "route_detail": {k: float(v) for k, v in avg_route_detail.items()},
            "global_metrics": {k: float(v) for k, v in global_metrics.items()},
        }

    def evaluate_agent_multiturn(
        self,
        strategy_name: str,
        test_dialogues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate agent-level metrics on test dialogues.

        Computes multi-turn specific metrics:
        - Success Rate / Goal Completion
        - Average Turns
        - Route Validity
        - Prefix Validity
        - Result Consistency
        - Recovery after Concern/Reject
        - Efficient Success
        """
        intent_data = self.labeler.label_dataset(test_dialogues)

        success_count = 0
        total_turns = 0
        route_scores = []
        prefix_scores = []
        result_cons = []
        recovery_count = 0
        concern_reject_count = 0
        efficient_success = []

        for d, idata in zip(test_dialogues, intent_data):
            seq = idata["intent_sequence"]
            outcome = d.get("meta", {}).get("outcome", d.get("outcome", "unknown"))
            n_turns = len(d.get("turns", []))

            # Success rate
            if outcome == "success":
                success_count += 1

            total_turns += n_turns

            # Route metrics
            rs = self.route_scorer.compute_score(seq, outcome)
            route_scores.append(rs["route_score"])
            prefix_scores.append(rs["prefix_validity"])

            # Result consistency
            rc = self.instance_eval.result_consistency(seq, outcome)
            result_cons.append(rc)

            # Recovery after Concern/Reject
            for i in range(len(seq) - 1):
                if seq[i] in ("Concern", "Reject"):
                    concern_reject_count += 1
                    if seq[i + 1] in ("Positive", "Action", "EndSuccess"):
                        recovery_count += 1

            # Efficient success
            if outcome == "success":
                # Lower turns = more efficient (penalize > 10 turns)
                efficiency = max(0.0, 1.0 - (n_turns - 6) / 10)
                efficient_success.append(efficiency)

        n = len(test_dialogues)
        return {
            "strategy": strategy_name,
            "num_test_dialogues": n,
            "success_rate": success_count / n if n > 0 else 0,
            "avg_turns": total_turns / n if n > 0 else 0,
            "avg_route_validity": float(np.mean(route_scores)) if route_scores else 0,
            "avg_prefix_validity": float(np.mean(prefix_scores)) if prefix_scores else 0,
            "avg_result_consistency": float(np.mean(result_cons)) if result_cons else 0,
            "recovery_rate": recovery_count / concern_reject_count if concern_reject_count > 0 else 0,
            "avg_efficient_success": float(np.mean(efficient_success)) if efficient_success else 0,
        }

    def run_main_experiment(
        self,
        selected_dir: str,
        test_data_path: str | None = None,
    ) -> Dict[str, Any]:
        """Run the main comparative experiment.

        Evaluates all three strategies on both data-level and agent-level metrics.
        """
        strategies = ["random_k", "instance_top_k", "coi_selected_k"]
        results: Dict[str, Any] = {"data_quality": {}, "agent_quality": {}}

        for strategy in strategies:
            data_path = os.path.join(selected_dir, strategy, "selected.jsonl")
            if not os.path.exists(data_path):
                print(f"[WARN] Data for {strategy} not found at {data_path}")
                continue

            dialogues = load_jsonl(data_path)

            # Data-level evaluation
            data_metrics = self.evaluate_data_quality(strategy, dialogues)
            results["data_quality"][strategy] = data_metrics

            # Agent-level evaluation (use same data as proxy test set)
            test_data = dialogues
            if test_data_path and os.path.exists(test_data_path):
                test_data = load_jsonl(test_data_path)

            agent_metrics = self.evaluate_agent_multiturn(strategy, test_data)
            results["agent_quality"][strategy] = agent_metrics

        return results

    def run_ablation(
        self,
        pool: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
    ) -> Dict[str, Any]:
        """Run ablation experiments.

        Tests removing individual components to show their contribution:
        1. No route consistency
        2. No global KL/JS
        3. No diversity
        4. Instance-only (no global optimization)
        """
        # This is a simplified ablation framework
        # Full implementation would re-run selection with modified scoring

        ablation_results = {
            "description": "Ablation study comparing selection with/without key components",
            "ablations": {
                "no_route_consistency": {
                    "description": "Remove route consistency from instance scoring",
                    "status": "placeholder",
                },
                "no_kl_js": {
                    "description": "Remove KL/JS from global optimization",
                    "status": "placeholder",
                },
                "no_diversity": {
                    "description": "Remove diversity from global optimization",
                    "status": "placeholder",
                },
                "instance_only": {
                    "description": "Use only instance-level scores, skip global optimization",
                    "status": "placeholder",
                },
            },
        }

        return ablation_results

    @staticmethod
    def format_results_table(results: Dict[str, Any]) -> str:
        """Format experiment results as a readable comparison table."""
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT RESULTS: Data Selection Strategy Comparison")
        lines.append("=" * 80)

        # Data quality comparison
        lines.append("\n--- Data-Level Metrics ---")
        data_q = results.get("data_quality", {})
        if data_q:
            headers = ["Metric"]
            strategies = sorted(data_q.keys())
            headers.extend(strategies)
            lines.append(" | ".join(f"{h:>20}" for h in headers))
            lines.append("-" * (22 * len(headers)))

            # Key metrics
            key_metrics = [
                ("Avg Instance Score", lambda d: d.get("avg_instance_score", 0)),
                ("Avg Route Score", lambda d: d.get("avg_route_score", 0)),
                ("KL Divergence ↓", lambda d: d.get("global_metrics", {}).get("kl_divergence", 0)),
                ("JS Divergence ↓", lambda d: d.get("global_metrics", {}).get("js_divergence", 0)),
                ("Diversity ↑", lambda d: d.get("global_metrics", {}).get("diversity", 0)),
                ("Intent Coverage ↑", lambda d: d.get("global_metrics", {}).get("intent_coverage", 0)),
                ("Outcome Match ↑", lambda d: d.get("global_metrics", {}).get("outcome_ratio_match", 0)),
            ]

            for metric_name, getter in key_metrics:
                row = [f"{metric_name:>20}"]
                for s in strategies:
                    val = getter(data_q[s])
                    row.append(f"{val:>20.4f}")
                lines.append(" | ".join(row))

        # Agent quality comparison
        lines.append("\n--- Agent-Level Metrics ---")
        agent_q = results.get("agent_quality", {})
        if agent_q:
            strategies = sorted(agent_q.keys())
            agent_metrics = [
                ("Success Rate ↑", "success_rate"),
                ("Avg Turns", "avg_turns"),
                ("Route Validity ↑", "avg_route_validity"),
                ("Prefix Validity ↑", "avg_prefix_validity"),
                ("Result Consistency ↑", "avg_result_consistency"),
                ("Recovery Rate ↑", "recovery_rate"),
                ("Efficient Success ↑", "avg_efficient_success"),
            ]

            for metric_name, key in agent_metrics:
                row = [f"{metric_name:>20}"]
                for s in strategies:
                    val = agent_q[s].get(key, 0)
                    row.append(f"{val:>20.4f}")
                lines.append(" | ".join(row))

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
