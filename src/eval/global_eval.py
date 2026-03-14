"""Phase 6.2: Global-level evaluation metrics.

Computes distribution-level quality metrics for a set of dialogues:
1. KL Divergence – real vs synthetic intent transition distributions
2. JS Divergence – symmetric version of KL
3. Intent Coverage – coverage of major intents and transitions
4. Diversity – intent-based diversity (Shannon entropy)
5. Turn-length Distribution Match
6. Outcome Ratio Match
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.graph.coi_graph import CoIGraph
from src.intent.taxonomy import Taxonomy


def _normalize_distribution(counts: np.ndarray, smoothing: float = 1e-10) -> np.ndarray:
    """Normalize a count array to a probability distribution."""
    dist = counts.astype(np.float64) + smoothing
    return dist / dist.sum()


class GlobalEvaluator:
    """Computes global distribution-level metrics for a subset of dialogues."""

    def __init__(
        self,
        real_coi_graph: CoIGraph,
        real_dialogues: List[Dict[str, Any]] | None = None,
        taxonomy: Taxonomy | None = None,
    ):
        self.coi_graph = real_coi_graph
        self.taxonomy = taxonomy or Taxonomy()

        # Real data distributions
        self.real_transition_dist = _normalize_distribution(
            real_coi_graph.count_matrix.flatten()
        )

        # Real turn-length distribution
        self.real_turn_lengths: List[int] = []
        self.real_outcomes: Dict[str, int] = Counter()

        if real_dialogues:
            for d in real_dialogues:
                self.real_turn_lengths.append(len(d["turns"]))
                outcome = d.get("meta", {}).get("outcome", "unknown")
                self.real_outcomes[outcome] += 1

    def _build_transition_matrix(
        self, intent_sequences: List[List[str]]
    ) -> np.ndarray:
        """Build a transition count matrix from intent sequences."""
        n = self.taxonomy.num_intents
        matrix = np.zeros((n, n), dtype=np.float64)
        for seq in intent_sequences:
            for i in range(len(seq) - 1):
                src = seq[i]
                dst = seq[i + 1]
                if self.taxonomy.is_valid_intent(src) and self.taxonomy.is_valid_intent(dst):
                    si = self.taxonomy.intent2id[src]
                    di = self.taxonomy.intent2id[dst]
                    matrix[si, di] += 1
        return matrix

    def kl_divergence(self, intent_sequences: List[List[str]]) -> float:
        """KL(real || synthetic) for intent transition distributions."""
        syn_matrix = self._build_transition_matrix(intent_sequences)
        syn_dist = _normalize_distribution(syn_matrix.flatten())
        # KL(P || Q) = sum(P * log(P / Q))
        return float(np.sum(self.real_transition_dist * np.log(
            self.real_transition_dist / syn_dist
        )))

    def js_divergence(self, intent_sequences: List[List[str]]) -> float:
        """Jensen-Shannon divergence between real and synthetic distributions."""
        syn_matrix = self._build_transition_matrix(intent_sequences)
        syn_dist = _normalize_distribution(syn_matrix.flatten())
        m = 0.5 * (self.real_transition_dist + syn_dist)
        kl_pm = float(np.sum(self.real_transition_dist * np.log(self.real_transition_dist / m)))
        kl_qm = float(np.sum(syn_dist * np.log(syn_dist / m)))
        return 0.5 * (kl_pm + kl_qm)

    def intent_coverage(self, intent_sequences: List[List[str]]) -> Dict[str, float]:
        """Compute intent and transition coverage.

        Returns:
            intent_coverage: fraction of intents present in synthetic data
            transition_coverage: fraction of real edges covered
        """
        syn_intents = set()
        syn_edges = set()
        for seq in intent_sequences:
            for intent in seq:
                syn_intents.add(intent)
            for i in range(len(seq) - 1):
                syn_edges.add((seq[i], seq[i + 1]))

        real_intents = set(self.taxonomy.intent_names)
        intent_cov = len(syn_intents & real_intents) / len(real_intents) if real_intents else 0

        real_edges = self.coi_graph.valid_edges
        edge_cov = len(syn_edges & real_edges) / len(real_edges) if real_edges else 0

        return {
            "intent_coverage": intent_cov,
            "transition_coverage": edge_cov,
        }

    def diversity(self, intent_sequences: List[List[str]]) -> float:
        """Shannon entropy of intent distribution as a diversity measure."""
        all_intents = []
        for seq in intent_sequences:
            all_intents.extend(seq)

        if not all_intents:
            return 0.0

        counts = Counter(all_intents)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by max possible entropy
        max_entropy = np.log(self.taxonomy.num_intents)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def turn_length_match(self, dialogues: List[Dict[str, Any]]) -> float:
        """Score how well the turn-length distribution matches real data.

        Uses Wasserstein-like distance normalized to [0, 1].
        """
        if not self.real_turn_lengths or not dialogues:
            return 0.5

        syn_lengths = [len(d.get("turns", [])) for d in dialogues]

        real_mean = np.mean(self.real_turn_lengths)
        real_std = max(np.std(self.real_turn_lengths), 1.0)
        syn_mean = np.mean(syn_lengths)
        syn_std = max(np.std(syn_lengths), 1.0)

        # Compare means and stds
        mean_diff = abs(real_mean - syn_mean) / real_mean if real_mean > 0 else 0
        std_diff = abs(real_std - syn_std) / real_std if real_std > 0 else 0

        score = 1.0 - min(0.5 * mean_diff + 0.5 * std_diff, 1.0)
        return float(score)

    def outcome_ratio_match(self, dialogues: List[Dict[str, Any]]) -> float:
        """Score how well outcome ratios match real data."""
        if not self.real_outcomes or not dialogues:
            return 0.5

        syn_outcomes: Dict[str, int] = Counter()
        for d in dialogues:
            outcome = d.get("meta", {}).get("outcome", d.get("outcome", "unknown"))
            syn_outcomes[outcome] += 1

        # Compare success/fail ratios
        real_total = sum(self.real_outcomes.values())
        syn_total = sum(syn_outcomes.values())

        if real_total == 0 or syn_total == 0:
            return 0.5

        total_diff = 0.0
        for key in set(list(self.real_outcomes.keys()) + list(syn_outcomes.keys())):
            real_ratio = self.real_outcomes.get(key, 0) / real_total
            syn_ratio = syn_outcomes.get(key, 0) / syn_total
            total_diff += abs(real_ratio - syn_ratio)

        return float(max(1.0 - total_diff, 0.0))

    def evaluate(
        self,
        dialogues: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
    ) -> Dict[str, float]:
        """Compute all global metrics for a set of dialogues."""
        results: Dict[str, float] = {}

        results["kl_divergence"] = self.kl_divergence(intent_sequences)
        results["js_divergence"] = self.js_divergence(intent_sequences)

        coverage = self.intent_coverage(intent_sequences)
        results.update(coverage)

        results["diversity"] = self.diversity(intent_sequences)
        results["turn_length_match"] = self.turn_length_match(dialogues)
        results["outcome_ratio_match"] = self.outcome_ratio_match(dialogues)

        return results
