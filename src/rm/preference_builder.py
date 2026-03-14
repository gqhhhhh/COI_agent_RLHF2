"""Phase 8.2: Reward Model - Preference pair construction.

Constructs preference pairs for reward model training:
- Positive samples: high style sim, high route consistency, high result consistency
- Negative samples: hallucination, route invalid, outcome/path conflict, repetitive
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from src.utils import save_jsonl, set_seed


class PreferencePairBuilder:
    """Constructs preference pairs from scored dialogue pool."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def build_pairs(
        self,
        pool: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
        intent_sequences: List[List[str]],
        max_pairs: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Build preference pairs from the dialogue pool.

        Pairs are constructed by matching dialogues with similar context
        but different quality scores.

        Returns list of preference pair dicts.
        """
        set_seed(self.seed)

        # Sort by instance score
        indexed = [(i, s.get("instance_score", 0)) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: -x[1])

        n = len(indexed)
        if n < 2:
            return []

        # Split into positive (top 40%) and negative (bottom 40%) pools
        top_cutoff = int(n * 0.4)
        bottom_cutoff = int(n * 0.6)

        positive_pool = [idx for idx, _ in indexed[:top_cutoff]]
        negative_pool = [idx for idx, _ in indexed[bottom_cutoff:]]

        if not positive_pool or not negative_pool:
            return []

        pairs = []
        for _ in range(max_pairs):
            pos_idx = random.choice(positive_pool)
            neg_idx = random.choice(negative_pool)

            pos_dialogue = pool[pos_idx]
            neg_dialogue = pool[neg_idx]
            pos_score = scores[pos_idx]
            neg_score = scores[neg_idx]

            pair = {
                "chosen": {
                    "dialogue_id": pos_dialogue.get("dialogue_id", ""),
                    "turns": pos_dialogue.get("turns", []),
                    "scores": {k: v for k, v in pos_score.items() if isinstance(v, (int, float))},
                },
                "rejected": {
                    "dialogue_id": neg_dialogue.get("dialogue_id", ""),
                    "turns": neg_dialogue.get("turns", []),
                    "scores": {k: v for k, v in neg_score.items() if isinstance(v, (int, float))},
                },
                "preference_reason": self._determine_reason(pos_score, neg_score),
            }
            pairs.append(pair)

        return pairs

    def _determine_reason(
        self,
        pos_score: Dict[str, Any],
        neg_score: Dict[str, Any],
    ) -> str:
        """Determine the main reason for preference."""
        reasons = []

        if pos_score.get("route_consistency", 0) > neg_score.get("route_consistency", 0) + 0.1:
            reasons.append("route_validity")
        if pos_score.get("result_consistency", 0) > neg_score.get("result_consistency", 0) + 0.1:
            reasons.append("result_consistency")
        if pos_score.get("repetition_penalty", 0) > neg_score.get("repetition_penalty", 0) + 0.1:
            reasons.append("less_repetitive")
        if pos_score.get("style_similarity", 0) > neg_score.get("style_similarity", 0) + 0.1:
            reasons.append("better_style")

        return ",".join(reasons) if reasons else "overall_quality"

    def save_pairs(
        self,
        pairs: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Save preference pairs to JSONL."""
        save_jsonl(pairs, output_path)
        print(f"[INFO] Saved {len(pairs)} preference pairs to {output_path}")
