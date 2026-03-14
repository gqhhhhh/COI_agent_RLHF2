"""Phase 7: Data selection strategies.

Implements three selection strategies:
A. Random-K – random sampling
B. InstanceTopK – top-K by instance-level score
C. CoI-Selected-K – two-stage: instance filter + global distribution optimization

The CoI-Selected-K strategy is the core contribution, implementing:
- Stage 1: Instance-level filtering (top M candidates)
- Stage 2: Global optimization via Monte Carlo search and greedy swap
"""

import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.eval.global_eval import GlobalEvaluator
from src.eval.instance_eval import InstanceEvaluator
from src.graph.coi_graph import CoIGraph
from src.utils import save_json, save_jsonl, set_seed


class DataSelector:
    """Implements all three selection strategies."""

    def __init__(
        self,
        instance_evaluator: InstanceEvaluator,
        global_evaluator: GlobalEvaluator,
        K: int = 500,
        M_ratio: float = 2.0,
        mc_iterations: int = 1000,
        greedy_iterations: int = 500,
        seed: int = 42,
        objective_weights: Dict[str, float] | None = None,
    ):
        self.instance_eval = instance_evaluator
        self.global_eval = global_evaluator
        self.K = K
        self.M = int(K * M_ratio)
        self.mc_iterations = mc_iterations
        self.greedy_iterations = greedy_iterations
        self.seed = seed
        self._obj_weights = objective_weights or {
            "kl_divergence": 1.0,
            "js_divergence": 1.0,
            "diversity": 2.0,
            "intent_coverage": 1.0,
            "transition_coverage": 1.0,
            "outcome_ratio_match": 1.0,
            "turn_length_match": 0.5,
        }

    def random_k(
        self,
        pool: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
    ) -> List[int]:
        """Strategy A: Random-K selection.

        Returns indices into the pool.
        """
        set_seed(self.seed)
        k = min(self.K, len(pool))
        indices = list(range(len(pool)))
        random.shuffle(indices)
        return sorted(indices[:k])

    def instance_top_k(
        self,
        pool: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
    ) -> List[int]:
        """Strategy B: InstanceTopK selection.

        Select K dialogues with highest instance-level scores.
        """
        scored = [(i, s.get("instance_score", 0.0)) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: -x[1])
        k = min(self.K, len(scored))
        return sorted([idx for idx, _ in scored[:k]])

    def coi_selected_k(
        self,
        pool: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
    ) -> List[int]:
        """Strategy C: CoI-Selected-K (two-stage selection).

        Stage 1: Instance-level filtering (top M)
        Stage 2: Global distribution optimization
        """
        # Stage 1: Instance-level filtering
        scored = [(i, s.get("instance_score", 0.0)) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: -x[1])
        m = min(self.M, len(scored))
        candidate_indices = [idx for idx, _ in scored[:m]]

        if len(candidate_indices) <= self.K:
            return sorted(candidate_indices)

        # Stage 2: Global optimization
        # Start with Monte Carlo search for initial solution
        best_indices = self._monte_carlo_search(
            pool, intent_sequences, candidate_indices
        )

        # Refine with greedy swap
        best_indices = self._greedy_swap(
            pool, intent_sequences, candidate_indices, best_indices
        )

        return sorted(best_indices)

    def _global_objective(
        self,
        pool: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
        indices: List[int],
    ) -> float:
        """Compute global objective score for a subset.

        Higher is better. Combines:
        - Lower KL/JS divergence
        - Higher diversity
        - Better outcome ratio match
        - Higher intent coverage
        """
        subset_dialogues = [pool[i] for i in indices]
        subset_sequences = [intent_sequences[i] for i in indices]

        metrics = self.global_eval.evaluate(subset_dialogues, subset_sequences)

        # Objective: maximize this score (weights configurable via constructor)
        score = (
            - self._obj_weights.get("kl_divergence", 1.0) * metrics["kl_divergence"]
            - self._obj_weights.get("js_divergence", 1.0) * metrics["js_divergence"]
            + self._obj_weights.get("diversity", 2.0) * metrics["diversity"]
            + self._obj_weights.get("intent_coverage", 1.0) * metrics["intent_coverage"]
            + self._obj_weights.get("transition_coverage", 1.0) * metrics["transition_coverage"]
            + self._obj_weights.get("outcome_ratio_match", 1.0) * metrics["outcome_ratio_match"]
            + self._obj_weights.get("turn_length_match", 0.5) * metrics["turn_length_match"]
        )
        return score

    def _monte_carlo_search(
        self,
        pool: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
        candidate_indices: List[int],
    ) -> List[int]:
        """Monte Carlo subset search for global optimization."""
        set_seed(self.seed)
        k = min(self.K, len(candidate_indices))

        best_score = float("-inf")
        best_subset: List[int] = []

        for _ in range(self.mc_iterations):
            subset = random.sample(candidate_indices, k)
            score = self._global_objective(pool, intent_sequences, subset)
            if score > best_score:
                best_score = score
                best_subset = subset

        return best_subset

    def _greedy_swap(
        self,
        pool: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
        candidate_indices: List[int],
        initial_subset: List[int],
    ) -> List[int]:
        """Greedy swap optimization for global metrics."""
        set_seed(self.seed + 1)
        current = list(initial_subset)
        current_set = set(current)
        remaining = [i for i in candidate_indices if i not in current_set]

        current_score = self._global_objective(pool, intent_sequences, current)

        for _ in range(self.greedy_iterations):
            if not remaining:
                break

            # Pick a random element to swap out and a random one to swap in
            out_idx = random.randint(0, len(current) - 1)
            in_idx = random.randint(0, len(remaining) - 1)

            # Try swap
            old_val = current[out_idx]
            new_val = remaining[in_idx]
            current[out_idx] = new_val

            new_score = self._global_objective(pool, intent_sequences, current)

            if new_score > current_score:
                # Accept swap
                remaining[in_idx] = old_val
                current_score = new_score
            else:
                # Revert
                current[out_idx] = old_val

        return current

    def select_all(
        self,
        pool: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
    ) -> Dict[str, List[int]]:
        """Run all three selection strategies.

        Returns dict mapping strategy name to selected indices.
        """
        return {
            "random_k": self.random_k(pool, scores),
            "instance_top_k": self.instance_top_k(pool, scores),
            "coi_selected_k": self.coi_selected_k(pool, scores, intent_sequences),
        }

    def save_selections(
        self,
        pool: List[Dict[str, Any]],
        selections: Dict[str, List[int]],
        output_dir: str,
    ) -> None:
        """Save selected data for each strategy."""
        os.makedirs(output_dir, exist_ok=True)

        for strategy, indices in selections.items():
            strategy_dir = os.path.join(output_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)

            selected_data = [pool[i] for i in indices]
            save_jsonl(selected_data, os.path.join(strategy_dir, "selected.jsonl"))
            save_json(
                {"indices": indices, "count": len(indices)},
                os.path.join(strategy_dir, "selection_info.json"),
            )

    def compare_selections(
        self,
        pool: List[Dict[str, Any]],
        selections: Dict[str, List[int]],
        intent_sequences: List[List[str]],
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Compare global metrics across all selection strategies.

        Returns a dict mapping strategy name to global metric dict.
        """
        comparison: Dict[str, Dict[str, float]] = {}

        for strategy, indices in selections.items():
            subset_dialogues = [pool[i] for i in indices]
            subset_sequences = [intent_sequences[i] for i in indices]
            subset_scores = [scores[i] for i in indices]

            # Global metrics
            global_metrics = self.global_eval.evaluate(
                subset_dialogues, subset_sequences
            )

            # Average instance metrics
            avg_instance = np.mean([s.get("instance_score", 0) for s in subset_scores])
            avg_route = np.mean([s.get("route_consistency", 0) for s in subset_scores])

            comparison[strategy] = {
                **global_metrics,
                "avg_instance_score": float(avg_instance),
                "avg_route_score": float(avg_route),
                "num_selected": len(indices),
            }

        return comparison
