"""Phase 6.1: Instance-level evaluation metrics.

Computes per-dialogue quality scores:
1. Style Similarity – embedding distance to real dialogues
2. Result Consistency – path/outcome agreement
3. Route Consistency – uses Phase 3 module
4. Repetition Penalty – penalizes repeated patterns
5. Length Sanity – penalizes abnormal dialogue lengths
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from src.graph.route_consistency import RouteConsistency


class InstanceEvaluator:
    """Computes instance-level quality scores for individual dialogues."""

    def __init__(
        self,
        route_scorer: RouteConsistency | None = None,
        real_dialogues: List[Dict[str, Any]] | None = None,
        real_avg_turns: float = 8.0,
        real_std_turns: float = 3.0,
    ):
        self.route_scorer = route_scorer
        self.real_avg_turns = real_avg_turns
        self.real_std_turns = real_std_turns

        # Pre-compute real dialogue embeddings if available
        self._real_embeddings = None
        if real_dialogues:
            self._precompute_real_stats(real_dialogues)

    def _precompute_real_stats(self, real_dialogues: List[Dict[str, Any]]) -> None:
        """Precompute statistics from real dialogues."""
        turn_counts = [len(d["turns"]) for d in real_dialogues]
        self.real_avg_turns = np.mean(turn_counts)
        self.real_std_turns = max(np.std(turn_counts), 1.0)

    def style_similarity(self, dialogue: Dict[str, Any]) -> float:
        """Compute style similarity score.

        Uses simple text statistics as a proxy for embedding-based similarity.
        Compares vocabulary diversity, avg sentence length, etc.
        """
        texts = [t["text"] for t in dialogue.get("turns", [])]
        if not texts:
            return 0.0

        # Vocabulary diversity
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())

        if not all_words:
            return 0.0

        vocab_diversity = len(set(all_words)) / len(all_words) if all_words else 0

        # Avg sentence length (in words)
        avg_len = np.mean([len(t.split()) for t in texts])

        # Normalize to [0, 1]
        # Typical dialogues have vocab diversity 0.3-0.7 and avg length 5-20
        div_score = min(vocab_diversity / 0.6, 1.0)
        len_score = 1.0 - min(abs(avg_len - 10) / 15, 1.0)

        return 0.5 * div_score + 0.5 * len_score

    def result_consistency(
        self,
        intent_sequence: List[str],
        outcome: str,
    ) -> float:
        """Check if the intent path is consistent with the outcome.

        Success paths should contain Action or EndSuccess.
        Fail paths should contain Reject and no Action.
        """
        if outcome == "unknown":
            return 0.5

        has_action = "Action" in intent_sequence or "EndSuccess" in intent_sequence
        has_reject = "Reject" in intent_sequence
        has_positive = "Positive" in intent_sequence

        if outcome == "success":
            if has_action:
                return 1.0
            elif has_positive and not has_reject:
                return 0.6
            elif has_reject:
                return 0.2
            else:
                return 0.4

        elif outcome == "fail":
            if has_reject and not has_action:
                return 1.0
            elif not has_action and not has_positive:
                return 0.7
            elif has_action:
                return 0.1
            else:
                return 0.4

        return 0.5

    def repetition_penalty(self, dialogue: Dict[str, Any]) -> float:
        """Compute repetition penalty score (higher = less repetition = better).

        Checks for:
        - Repeated exact utterances
        - Repeated n-grams
        """
        texts = [t["text"].strip().lower() for t in dialogue.get("turns", [])]
        if len(texts) <= 1:
            return 1.0

        # Exact repetition
        unique_ratio = len(set(texts)) / len(texts)

        # Bigram repetition across all turns
        all_bigrams = []
        for text in texts:
            words = text.split()
            for i in range(len(words) - 1):
                all_bigrams.append((words[i], words[i + 1]))

        if all_bigrams:
            bigram_unique_ratio = len(set(all_bigrams)) / len(all_bigrams)
        else:
            bigram_unique_ratio = 1.0

        return 0.6 * unique_ratio + 0.4 * bigram_unique_ratio

    def length_sanity(self, dialogue: Dict[str, Any]) -> float:
        """Score based on dialogue length normality.

        Penalizes dialogues that are too short or too long.
        """
        n_turns = len(dialogue.get("turns", []))
        if n_turns == 0:
            return 0.0

        # z-score based penalty
        z = abs(n_turns - self.real_avg_turns) / self.real_std_turns
        # Sigmoid-like: 1.0 for z=0, ~0.27 for z=2, ~0.05 for z=3
        score = 1.0 / (1.0 + 0.5 * z * z)
        return score

    def evaluate(
        self,
        dialogue: Dict[str, Any],
        intent_sequence: List[str],
        outcome: str = "unknown",
    ) -> Dict[str, float]:
        """Compute all instance-level metrics for a single dialogue.

        Returns dict with individual scores and total score.
        """
        scores: Dict[str, float] = {}

        scores["style_similarity"] = self.style_similarity(dialogue)
        scores["result_consistency"] = self.result_consistency(intent_sequence, outcome)
        scores["repetition_penalty"] = self.repetition_penalty(dialogue)
        scores["length_sanity"] = self.length_sanity(dialogue)

        # Route consistency (if scorer available)
        if self.route_scorer:
            route_scores = self.route_scorer.compute_score(intent_sequence, outcome)
            scores["route_consistency"] = route_scores["route_score"]
            scores.update({f"route_{k}": v for k, v in route_scores.items() if k != "route_score"})
        else:
            scores["route_consistency"] = 0.5

        # Total instance score (equal weights by default)
        metric_keys = [
            "style_similarity",
            "result_consistency",
            "route_consistency",
            "repetition_penalty",
            "length_sanity",
        ]
        scores["instance_score"] = np.mean([scores[k] for k in metric_keys]).item()

        return scores

    def evaluate_dataset(
        self,
        dialogues: List[Dict[str, Any]],
        intent_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate all dialogues in a dataset.

        Args:
            dialogues: List of dialogue dicts (with turns, meta, etc.)
            intent_data: List of intent annotation dicts (matching by dialogue_id)

        Returns:
            List of score dicts with dialogue_id and all metrics.
        """
        intent_map = {d["dialogue_id"]: d for d in intent_data}

        results = []
        for d in dialogues:
            did = d.get("dialogue_id", "")
            intent_info = intent_map.get(did, {})
            seq = intent_info.get("intent_sequence", d.get("intent_sequence", []))
            outcome = d.get("meta", {}).get("outcome", d.get("outcome", "unknown"))

            scores = self.evaluate(d, seq, outcome)
            scores["dialogue_id"] = did
            results.append(scores)

        return results
