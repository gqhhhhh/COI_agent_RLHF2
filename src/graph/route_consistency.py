"""Phase 3: Route Consistency scoring module.

Implements four soft-score route consistency metrics and a composite score.

Metrics:
1. Edge Validity – fraction of edges in a synthetic path found in the real graph
2. Path-k Validity – fraction of k-gram sub-paths found in the real path lexicon
3. Reachability – whether the path can reach the observed outcome in the real graph
4. Prefix Validity – average validity of all prefix sub-paths
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.graph.coi_graph import CoIGraph
from src.utils import get_config


class RouteConsistency:
    """Compute route consistency scores for intent sequences."""

    def __init__(
        self,
        coi_graph: CoIGraph,
        weights: Dict[str, float] | None = None,
    ):
        self.graph = coi_graph
        # Load weights from config if not provided
        if weights is None:
            try:
                cfg = get_config()
                weights = cfg["route_consistency"]["weights"]
            except Exception:
                weights = {
                    "edge_validity": 0.3,
                    "path_k_validity": 0.25,
                    "reachability": 0.25,
                    "prefix_validity": 0.2,
                }
        self.weights = weights

    def edge_validity(self, sequence: List[str]) -> float:
        """Fraction of edges in the sequence that appear in the real CoI graph."""
        if len(sequence) < 2:
            return 1.0
        valid = 0
        total = len(sequence) - 1
        for i in range(total):
            if (sequence[i], sequence[i + 1]) in self.graph.valid_edges:
                valid += 1
        return valid / total

    def path_k_validity(self, sequence: List[str], k: int = 3) -> float:
        """Fraction of k-gram sub-paths that appear in the real path lexicon."""
        if len(sequence) < k:
            return 1.0
        ngram_set = self.graph.path_ngrams.get(k, set())
        if not ngram_set:
            return 0.0
        valid = 0
        total = len(sequence) - k + 1
        for i in range(total):
            ngram = tuple(sequence[i:i + k])
            if ngram in ngram_set:
                valid += 1
        return valid / total

    def reachability(self, sequence: List[str], outcome: str = "unknown") -> float:
        """Check if the path is reachable for the given outcome.

        Returns a soft score:
        - 1.0 if the path's edges are consistent with the outcome
        - Fraction of edges consistent otherwise
        """
        if outcome == "unknown" or len(sequence) < 2:
            return 0.5  # neutral when outcome is unknown

        if outcome == "success":
            target_edges = self.graph.success_edges
        elif outcome == "fail":
            target_edges = self.graph.fail_edges
        else:
            return 0.5

        if not target_edges:
            return 0.5

        valid = 0
        total = len(sequence) - 1
        for i in range(total):
            if (sequence[i], sequence[i + 1]) in target_edges:
                valid += 1
        return valid / total if total > 0 else 0.5

    def prefix_validity(self, sequence: List[str]) -> float:
        """Average edge validity over all non-trivial prefix paths."""
        if len(sequence) < 2:
            return 1.0
        scores = []
        for length in range(2, len(sequence) + 1):
            prefix = sequence[:length]
            scores.append(self.edge_validity(prefix))
        return np.mean(scores).item()

    def compute_score(
        self,
        sequence: List[str],
        outcome: str = "unknown",
    ) -> Dict[str, float]:
        """Compute all route consistency metrics and the composite score.

        Returns a dict with individual scores and the weighted composite.
        """
        ev = self.edge_validity(sequence)
        pk = self.path_k_validity(sequence, k=3)
        reach = self.reachability(sequence, outcome)
        pv = self.prefix_validity(sequence)

        composite = (
            self.weights["edge_validity"] * ev
            + self.weights["path_k_validity"] * pk
            + self.weights["reachability"] * reach
            + self.weights["prefix_validity"] * pv
        )

        return {
            "edge_validity": ev,
            "path_k_validity": pk,
            "reachability": reach,
            "prefix_validity": pv,
            "route_score": composite,
        }

    def score_dataset(
        self,
        intent_data: List[Dict[str, Any]],
        dialogues: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Score all dialogues in a dataset.

        Returns a list of dicts with dialogue_id and route metrics.
        """
        outcome_map = {}
        if dialogues:
            for d in dialogues:
                outcome_map[d["dialogue_id"]] = d.get("meta", {}).get("outcome", "unknown")

        results = []
        for item in intent_data:
            seq = item["intent_sequence"]
            did = item["dialogue_id"]
            outcome = outcome_map.get(did, "unknown")
            scores = self.compute_score(seq, outcome)
            results.append({"dialogue_id": did, **scores})

        return results
