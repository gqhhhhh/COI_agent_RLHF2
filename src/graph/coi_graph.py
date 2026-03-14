"""Phase 2.2: CoI graph construction from real intent sequences.

Builds:
- Transition matrix (with Laplace smoothing)
- Directed graph with valid edges
- N-gram path lexicon
- Success/fail reachable graphs
- Heatmap visualization
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.intent.taxonomy import Taxonomy
from src.utils import load_json, load_jsonl, save_json


class CoIGraph:
    """Chain-of-Intention graph built from real dialogue data."""

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        laplace_smoothing: float = 1.0,
        ngram_sizes: List[int] | None = None,
    ):
        self.taxonomy = taxonomy or Taxonomy()
        self.laplace_smoothing = laplace_smoothing
        self.ngram_sizes = ngram_sizes or [3, 4]
        self.n = self.taxonomy.num_intents
        self.intent_names = self.taxonomy.intent_names

        # Core data structures
        self.count_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        self.transition_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        self.valid_edges: Set[Tuple[str, str]] = set()
        self.path_ngrams: Dict[int, Set[Tuple[str, ...]]] = {k: set() for k in self.ngram_sizes}

        # Outcome-conditioned graphs
        self.success_edges: Set[Tuple[str, str]] = set()
        self.fail_edges: Set[Tuple[str, str]] = set()
        self.success_transitions = np.zeros((self.n, self.n), dtype=np.float64)
        self.fail_transitions = np.zeros((self.n, self.n), dtype=np.float64)

    def build_from_sequences(
        self,
        intent_data: List[Dict[str, Any]],
        dialogues: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Build the CoI graph from labeled intent sequences.

        Args:
            intent_data: List of intent annotation dicts (from labeler).
            dialogues: Optional list of original dialogues (for outcome info).
        """
        # Build outcome map
        outcome_map = {}
        if dialogues:
            for d in dialogues:
                outcome_map[d["dialogue_id"]] = d.get("meta", {}).get("outcome", "unknown")

        all_sequences = []
        for item in intent_data:
            seq = item["intent_sequence"]
            did = item["dialogue_id"]
            outcome = outcome_map.get(did, "unknown")
            all_sequences.append((seq, outcome))

        # Build count matrices
        for seq, outcome in all_sequences:
            for i in range(len(seq) - 1):
                src = seq[i]
                dst = seq[i + 1]
                if self.taxonomy.is_valid_intent(src) and self.taxonomy.is_valid_intent(dst):
                    si = self.taxonomy.intent2id[src]
                    di = self.taxonomy.intent2id[dst]
                    self.count_matrix[si, di] += 1
                    self.valid_edges.add((src, dst))

                    if outcome == "success":
                        self.success_transitions[si, di] += 1
                        self.success_edges.add((src, dst))
                    elif outcome == "fail":
                        self.fail_transitions[si, di] += 1
                        self.fail_edges.add((src, dst))

            # Build n-gram paths
            for k in self.ngram_sizes:
                for i in range(len(seq) - k + 1):
                    ngram = tuple(seq[i:i + k])
                    self.path_ngrams[k].add(ngram)

        # Compute smoothed transition matrix
        smoothed = self.count_matrix + self.laplace_smoothing
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_matrix = smoothed / row_sums

    def save(self, output_dir: str) -> None:
        """Save all graph artifacts to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Matrices
        np.save(os.path.join(output_dir, "coi_matrix_real.npy"), self.transition_matrix)
        np.save(os.path.join(output_dir, "count_matrix.npy"), self.count_matrix)

        # JSON versions
        save_json(
            {
                "intent_names": self.intent_names,
                "transition_matrix": self.transition_matrix.tolist(),
                "count_matrix": self.count_matrix.tolist(),
            },
            os.path.join(output_dir, "coi_matrix_real.json"),
        )

        # Valid edges
        save_json(
            {"edges": [list(e) for e in sorted(self.valid_edges)]},
            os.path.join(output_dir, "valid_edges.json"),
        )

        # N-gram paths
        ngram_data = {}
        for k, ngrams in self.path_ngrams.items():
            ngram_data[str(k)] = [list(ng) for ng in sorted(ngrams)]
        save_json(ngram_data, os.path.join(output_dir, "path_ngrams.json"))

        # Success/fail edges
        save_json(
            {
                "success_edges": [list(e) for e in sorted(self.success_edges)],
                "fail_edges": [list(e) for e in sorted(self.fail_edges)],
                "success_transitions": self.success_transitions.tolist(),
                "fail_transitions": self.fail_transitions.tolist(),
            },
            os.path.join(output_dir, "outcome_edges.json"),
        )

    @classmethod
    def load(cls, output_dir: str, taxonomy: Taxonomy | None = None) -> "CoIGraph":
        """Load a saved CoI graph from disk."""
        tax = taxonomy or Taxonomy()
        graph = cls(taxonomy=tax)

        graph.transition_matrix = np.load(os.path.join(output_dir, "coi_matrix_real.npy"))
        graph.count_matrix = np.load(os.path.join(output_dir, "count_matrix.npy"))

        edges_data = load_json(os.path.join(output_dir, "valid_edges.json"))
        graph.valid_edges = {tuple(e) for e in edges_data["edges"]}

        ngram_data = load_json(os.path.join(output_dir, "path_ngrams.json"))
        for k_str, ngrams in ngram_data.items():
            k = int(k_str)
            graph.path_ngrams[k] = {tuple(ng) for ng in ngrams}

        outcome_data = load_json(os.path.join(output_dir, "outcome_edges.json"))
        graph.success_edges = {tuple(e) for e in outcome_data["success_edges"]}
        graph.fail_edges = {tuple(e) for e in outcome_data["fail_edges"]}
        graph.success_transitions = np.array(outcome_data["success_transitions"])
        graph.fail_transitions = np.array(outcome_data["fail_transitions"])

        return graph

    def plot_heatmap(self, output_path: str) -> None:
        """Save a heatmap visualization of the transition matrix."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(1, 3, figsize=(24, 7))

            matrices = [
                (self.transition_matrix, "Transition Matrix (all)"),
                (self.success_transitions, "Success Transitions (counts)"),
                (self.fail_transitions, "Fail Transitions (counts)"),
            ]

            for ax, (mat, title) in zip(axes, matrices):
                sns.heatmap(
                    mat,
                    annot=True,
                    fmt=".2f",
                    xticklabels=self.intent_names,
                    yticklabels=self.intent_names,
                    cmap="YlOrRd",
                    ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("To Intent")
                ax.set_ylabel("From Intent")

            plt.tight_layout()
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Heatmap saved to {output_path}")
        except ImportError as e:
            print(f"[WARN] Cannot plot heatmap: {e}")

    def get_analysis_report(self) -> str:
        """Generate a text analysis report of the CoI graph."""
        lines = ["=" * 60, "CoI Graph Analysis Report", "=" * 60, ""]

        # Edge statistics
        lines.append(f"Total valid edges: {len(self.valid_edges)}")
        lines.append(f"Success-specific edges: {len(self.success_edges)}")
        lines.append(f"Fail-specific edges: {len(self.fail_edges)}")
        lines.append("")

        # Top transitions
        lines.append("Top 10 most common transitions:")
        flat = []
        for i in range(self.n):
            for j in range(self.n):
                if self.count_matrix[i, j] > 0:
                    flat.append((self.intent_names[i], self.intent_names[j], self.count_matrix[i, j]))
        flat.sort(key=lambda x: -x[2])
        for src, dst, cnt in flat[:10]:
            lines.append(f"  {src} -> {dst}: {cnt:.0f}")
        lines.append("")

        # N-gram stats
        for k, ngrams in self.path_ngrams.items():
            lines.append(f"{k}-gram paths: {len(ngrams)} unique")
        lines.append("")

        # Success vs fail pattern differences
        success_only = self.success_edges - self.fail_edges
        fail_only = self.fail_edges - self.success_edges
        lines.append(f"Edges only in success paths: {len(success_only)}")
        for e in sorted(success_only):
            lines.append(f"  {e[0]} -> {e[1]}")
        lines.append(f"Edges only in fail paths: {len(fail_only)}")
        for e in sorted(fail_only):
            lines.append(f"  {e[0]} -> {e[1]}")

        return "\n".join(lines)
