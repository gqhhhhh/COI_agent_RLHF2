"""Tests for core CoI-Select pipeline components."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Taxonomy tests
# ============================================================================
class TestTaxonomy:
    def test_load_taxonomy(self):
        from src.intent.taxonomy import Taxonomy
        tax = Taxonomy()
        assert tax.num_intents == 7
        assert "Inquiry" in tax.intent_names
        assert "EndSuccess" in tax.intent_names
        assert tax.is_valid_intent("Positive")
        assert not tax.is_valid_intent("FakeIntent")

    def test_intent_ids(self):
        from src.intent.taxonomy import Taxonomy
        tax = Taxonomy()
        for name, idx in tax.intent2id.items():
            assert tax.id2intent[idx] == name


# ============================================================================
# Rule labeler tests
# ============================================================================
class TestRuleLabeler:
    def test_label_inquiry(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        intent, conf = labeler.label_utterance("What do they do exactly?")
        assert intent == "Inquiry"

    def test_label_positive(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        intent, _ = labeler.label_utterance("That sounds great!")
        assert intent == "Positive"

    def test_label_concern(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        intent, _ = labeler.label_utterance("I'm not sure I can afford that.")
        assert intent == "Concern"

    def test_label_reject(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        intent, _ = labeler.label_utterance("No thanks, I'm not interested.")
        assert intent == "Reject"

    def test_label_action(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        intent, _ = labeler.label_utterance("I'll donate $1.00.")
        assert intent == "Action"

    def test_label_neutral(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        intent, _ = labeler.label_utterance("Okay.")
        assert intent == "Neutral"

    def test_label_dialogue(self):
        from src.intent.rule_labeler import RuleLabeler
        labeler = RuleLabeler()
        dialogue = {
            "dialogue_id": "test_001",
            "turns": [
                {"role": "agent", "text": "Hi there!"},
                {"role": "user", "text": "What is this about?"},
                {"role": "agent", "text": "It's about donating."},
                {"role": "user", "text": "That sounds good."},
            ],
        }
        result = labeler.label_dialogue(dialogue)
        assert result["dialogue_id"] == "test_001"
        assert len(result["user_turns"]) == 2
        assert len(result["intent_sequence"]) == 2


# ============================================================================
# CoI graph tests
# ============================================================================
class TestCoIGraph:
    def _get_sample_intent_data(self):
        return [
            {
                "dialogue_id": "d1",
                "user_turns": [],
                "intent_sequence": ["Inquiry", "Positive", "Action", "EndSuccess"],
            },
            {
                "dialogue_id": "d2",
                "user_turns": [],
                "intent_sequence": ["Inquiry", "Concern", "Positive", "Action"],
            },
            {
                "dialogue_id": "d3",
                "user_turns": [],
                "intent_sequence": ["Inquiry", "Concern", "Reject"],
            },
            {
                "dialogue_id": "d4",
                "user_turns": [],
                "intent_sequence": ["Neutral", "Inquiry", "Positive", "Action"],
            },
        ]

    def _get_sample_dialogues(self):
        return [
            {"dialogue_id": "d1", "meta": {"outcome": "success"}, "turns": []},
            {"dialogue_id": "d2", "meta": {"outcome": "success"}, "turns": []},
            {"dialogue_id": "d3", "meta": {"outcome": "fail"}, "turns": []},
            {"dialogue_id": "d4", "meta": {"outcome": "success"}, "turns": []},
        ]

    def test_build_graph(self):
        from src.graph.coi_graph import CoIGraph
        graph = CoIGraph()
        intent_data = self._get_sample_intent_data()
        dialogues = self._get_sample_dialogues()
        graph.build_from_sequences(intent_data, dialogues)

        assert len(graph.valid_edges) > 0
        assert ("Inquiry", "Positive") in graph.valid_edges
        assert ("Inquiry", "Concern") in graph.valid_edges
        assert graph.transition_matrix.shape == (7, 7)

    def test_save_load(self):
        from src.graph.coi_graph import CoIGraph
        graph = CoIGraph()
        intent_data = self._get_sample_intent_data()
        dialogues = self._get_sample_dialogues()
        graph.build_from_sequences(intent_data, dialogues)

        with tempfile.TemporaryDirectory() as tmpdir:
            graph.save(tmpdir)
            loaded = CoIGraph.load(tmpdir)
            assert loaded.valid_edges == graph.valid_edges
            np.testing.assert_array_almost_equal(
                loaded.transition_matrix, graph.transition_matrix
            )

    def test_ngram_paths(self):
        from src.graph.coi_graph import CoIGraph
        graph = CoIGraph(ngram_sizes=[3])
        intent_data = self._get_sample_intent_data()
        graph.build_from_sequences(intent_data)

        assert len(graph.path_ngrams[3]) > 0
        assert ("Inquiry", "Positive", "Action") in graph.path_ngrams[3]

    def test_success_fail_edges(self):
        from src.graph.coi_graph import CoIGraph
        graph = CoIGraph()
        intent_data = self._get_sample_intent_data()
        dialogues = self._get_sample_dialogues()
        graph.build_from_sequences(intent_data, dialogues)

        assert len(graph.success_edges) > 0
        assert len(graph.fail_edges) > 0
        assert ("Concern", "Reject") in graph.fail_edges


# ============================================================================
# Route consistency tests
# ============================================================================
class TestRouteConsistency:
    def _build_graph(self):
        from src.graph.coi_graph import CoIGraph
        graph = CoIGraph(ngram_sizes=[3])
        intent_data = [
            {"dialogue_id": "d1", "user_turns": [],
             "intent_sequence": ["Inquiry", "Positive", "Action", "EndSuccess"]},
            {"dialogue_id": "d2", "user_turns": [],
             "intent_sequence": ["Inquiry", "Concern", "Positive", "Action"]},
            {"dialogue_id": "d3", "user_turns": [],
             "intent_sequence": ["Inquiry", "Concern", "Reject"]},
        ]
        dialogues = [
            {"dialogue_id": "d1", "meta": {"outcome": "success"}, "turns": []},
            {"dialogue_id": "d2", "meta": {"outcome": "success"}, "turns": []},
            {"dialogue_id": "d3", "meta": {"outcome": "fail"}, "turns": []},
        ]
        graph.build_from_sequences(intent_data, dialogues)
        return graph

    def test_edge_validity_valid_path(self):
        from src.graph.route_consistency import RouteConsistency
        graph = self._build_graph()
        scorer = RouteConsistency(graph)
        score = scorer.edge_validity(["Inquiry", "Positive", "Action"])
        assert score == 1.0

    def test_edge_validity_invalid_path(self):
        from src.graph.route_consistency import RouteConsistency
        graph = self._build_graph()
        scorer = RouteConsistency(graph)
        score = scorer.edge_validity(["EndSuccess", "Reject", "Inquiry"])
        assert score < 1.0

    def test_path_k_validity(self):
        from src.graph.route_consistency import RouteConsistency
        graph = self._build_graph()
        scorer = RouteConsistency(graph)
        score = scorer.path_k_validity(["Inquiry", "Positive", "Action"], k=3)
        assert score == 1.0

    def test_composite_score(self):
        from src.graph.route_consistency import RouteConsistency
        graph = self._build_graph()
        scorer = RouteConsistency(graph)

        # Valid path should score higher
        valid_scores = scorer.compute_score(
            ["Inquiry", "Positive", "Action"], "success"
        )
        invalid_scores = scorer.compute_score(
            ["EndSuccess", "Reject", "Inquiry"], "success"
        )
        assert valid_scores["route_score"] > invalid_scores["route_score"]

    def test_single_intent(self):
        from src.graph.route_consistency import RouteConsistency
        graph = self._build_graph()
        scorer = RouteConsistency(graph)
        scores = scorer.compute_score(["Inquiry"], "unknown")
        assert scores["edge_validity"] == 1.0
        assert scores["prefix_validity"] == 1.0


# ============================================================================
# Instance evaluator tests
# ============================================================================
class TestInstanceEvaluator:
    def test_repetition_penalty(self):
        from src.eval.instance_eval import InstanceEvaluator
        evaluator = InstanceEvaluator()

        # No repetition
        d1 = {"turns": [
            {"role": "user", "text": "Hello there"},
            {"role": "agent", "text": "Hi!"},
            {"role": "user", "text": "How are you?"},
        ]}
        score1 = evaluator.repetition_penalty(d1)

        # With repetition
        d2 = {"turns": [
            {"role": "user", "text": "Hello there"},
            {"role": "agent", "text": "Hello there"},
            {"role": "user", "text": "Hello there"},
        ]}
        score2 = evaluator.repetition_penalty(d2)

        assert score1 > score2

    def test_result_consistency(self):
        from src.eval.instance_eval import InstanceEvaluator
        evaluator = InstanceEvaluator()

        # Good success path
        score1 = evaluator.result_consistency(["Inquiry", "Positive", "Action"], "success")
        # Bad: success path but has reject
        score2 = evaluator.result_consistency(["Inquiry", "Reject"], "success")
        assert score1 > score2

    def test_length_sanity(self):
        from src.eval.instance_eval import InstanceEvaluator
        evaluator = InstanceEvaluator(real_avg_turns=8.0, real_std_turns=3.0)

        # Normal length
        d1 = {"turns": [{"role": "user", "text": "hi"}] * 8}
        score1 = evaluator.length_sanity(d1)

        # Very long
        d2 = {"turns": [{"role": "user", "text": "hi"}] * 50}
        score2 = evaluator.length_sanity(d2)

        assert score1 > score2


# ============================================================================
# Global evaluator tests
# ============================================================================
class TestGlobalEvaluator:
    def _build_evaluator(self):
        from src.graph.coi_graph import CoIGraph
        from src.eval.global_eval import GlobalEvaluator

        graph = CoIGraph(ngram_sizes=[3])
        intent_data = [
            {"dialogue_id": "d1", "user_turns": [],
             "intent_sequence": ["Inquiry", "Positive", "Action"]},
            {"dialogue_id": "d2", "user_turns": [],
             "intent_sequence": ["Inquiry", "Concern", "Reject"]},
        ]
        real_dialogues = [
            {"dialogue_id": "d1", "meta": {"outcome": "success"},
             "turns": [{"role": "a", "text": "hi"}] * 8},
            {"dialogue_id": "d2", "meta": {"outcome": "fail"},
             "turns": [{"role": "a", "text": "hi"}] * 6},
        ]
        graph.build_from_sequences(intent_data, real_dialogues)

        return GlobalEvaluator(graph, real_dialogues)

    def test_kl_divergence(self):
        evaluator = self._build_evaluator()
        # Same distribution should have low KL
        same_seqs = [["Inquiry", "Positive", "Action"], ["Inquiry", "Concern", "Reject"]]
        kl = evaluator.kl_divergence(same_seqs)
        assert kl >= 0

    def test_js_divergence(self):
        evaluator = self._build_evaluator()
        seqs = [["Inquiry", "Positive", "Action"]]
        js = evaluator.js_divergence(seqs)
        assert 0 <= js

    def test_intent_coverage(self):
        evaluator = self._build_evaluator()
        seqs = [["Inquiry", "Positive", "Action", "Concern", "Reject", "Neutral", "EndSuccess"]]
        coverage = evaluator.intent_coverage(seqs)
        assert coverage["intent_coverage"] == 1.0

    def test_diversity(self):
        evaluator = self._build_evaluator()
        # High diversity
        seqs1 = [["Inquiry", "Positive", "Concern", "Action", "Neutral", "Reject", "EndSuccess"]]
        div1 = evaluator.diversity(seqs1)

        # Low diversity (all same)
        seqs2 = [["Inquiry", "Inquiry", "Inquiry"]]
        div2 = evaluator.diversity(seqs2)

        assert div1 > div2

    def test_outcome_ratio_match(self):
        evaluator = self._build_evaluator()
        # Similar outcome ratio
        d1 = [
            {"meta": {"outcome": "success"}, "turns": []},
            {"meta": {"outcome": "fail"}, "turns": []},
        ]
        score1 = evaluator.outcome_ratio_match(d1)

        # All same outcome
        d2 = [
            {"meta": {"outcome": "success"}, "turns": []},
            {"meta": {"outcome": "success"}, "turns": []},
        ]
        score2 = evaluator.outcome_ratio_match(d2)

        assert score1 >= score2


# ============================================================================
# Selector tests
# ============================================================================
class TestSelector:
    def _build_test_data(self):
        pool = []
        scores = []
        sequences = []
        for i in range(100):
            pool.append({
                "dialogue_id": f"test_{i:03d}",
                "turns": [{"role": "user", "text": f"turn {j}"} for j in range(6)],
                "meta": {"outcome": "success" if i % 3 != 0 else "fail"},
            })
            scores.append({
                "dialogue_id": f"test_{i:03d}",
                "instance_score": 0.5 + 0.005 * i,
                "route_consistency": 0.4 + 0.006 * i,
            })
            if i % 3 == 0:
                sequences.append(["Inquiry", "Concern", "Reject"])
            elif i % 3 == 1:
                sequences.append(["Inquiry", "Positive", "Action"])
            else:
                sequences.append(["Inquiry", "Positive", "Concern", "Action"])
        return pool, scores, sequences

    def test_random_k(self):
        from src.graph.coi_graph import CoIGraph
        from src.graph.route_consistency import RouteConsistency
        from src.eval.instance_eval import InstanceEvaluator
        from src.eval.global_eval import GlobalEvaluator
        from src.select.selector import DataSelector

        pool, scores, sequences = self._build_test_data()

        graph = CoIGraph(ngram_sizes=[3])
        intent_data = [{"dialogue_id": f"d{i}", "user_turns": [], "intent_sequence": s}
                       for i, s in enumerate(sequences[:10])]
        graph.build_from_sequences(intent_data)

        instance_eval = InstanceEvaluator()
        global_eval = GlobalEvaluator(graph, pool[:10])
        selector = DataSelector(instance_eval, global_eval, K=20, mc_iterations=10, greedy_iterations=5)

        selected = selector.random_k(pool, scores)
        assert len(selected) == 20

    def test_instance_top_k(self):
        from src.graph.coi_graph import CoIGraph
        from src.eval.instance_eval import InstanceEvaluator
        from src.eval.global_eval import GlobalEvaluator
        from src.select.selector import DataSelector

        pool, scores, sequences = self._build_test_data()
        graph = CoIGraph()
        graph.build_from_sequences([
            {"dialogue_id": "d1", "user_turns": [], "intent_sequence": ["Inquiry", "Positive", "Action"]}
        ])
        instance_eval = InstanceEvaluator()
        global_eval = GlobalEvaluator(graph, pool[:5])
        selector = DataSelector(instance_eval, global_eval, K=20, mc_iterations=5, greedy_iterations=3)

        selected = selector.instance_top_k(pool, scores)
        assert len(selected) == 20
        # Top-K should select highest scored
        selected_scores = [scores[i]["instance_score"] for i in selected]
        assert min(selected_scores) >= 0.89  # top 20 out of 100

    def test_coi_selected_k(self):
        from src.graph.coi_graph import CoIGraph
        from src.graph.route_consistency import RouteConsistency
        from src.eval.instance_eval import InstanceEvaluator
        from src.eval.global_eval import GlobalEvaluator
        from src.select.selector import DataSelector

        pool, scores, sequences = self._build_test_data()
        graph = CoIGraph(ngram_sizes=[3])
        intent_data = [{"dialogue_id": f"d{i}", "user_turns": [], "intent_sequence": s}
                       for i, s in enumerate(sequences[:10])]
        graph.build_from_sequences(intent_data)
        instance_eval = InstanceEvaluator()
        global_eval = GlobalEvaluator(graph, pool[:10])
        selector = DataSelector(instance_eval, global_eval, K=20, mc_iterations=10, greedy_iterations=5)

        selected = selector.coi_selected_k(pool, scores, sequences)
        assert len(selected) == 20


# ============================================================================
# Data preprocessing tests
# ============================================================================
class TestPreprocessing:
    def test_generate_demo_data(self):
        from src.data.preprocess import generate_demo_data
        from src.utils import load_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            stats = generate_demo_data(tmpdir)
            assert stats["total_dialogues"] > 0
            assert stats["train"] > 0

            train = load_jsonl(os.path.join(tmpdir, "train.jsonl"))
            assert len(train) == stats["train"]
            assert all("turns" in d for d in train)
            assert all("dialogue_id" in d for d in train)


# ============================================================================
# Preference builder tests
# ============================================================================
class TestPreferenceBuilder:
    def test_build_pairs(self):
        from src.rm.preference_builder import PreferencePairBuilder

        pool = [
            {"dialogue_id": f"d{i}", "turns": [{"role": "user", "text": f"hi {i}"}]}
            for i in range(20)
        ]
        scores = [
            {"instance_score": 0.1 * i, "route_consistency": 0.05 * i}
            for i in range(20)
        ]
        sequences = [["Inquiry", "Positive"] for _ in range(20)]

        builder = PreferencePairBuilder()
        pairs = builder.build_pairs(pool, scores, sequences, max_pairs=10)
        assert len(pairs) == 10
        assert "chosen" in pairs[0]
        assert "rejected" in pairs[0]


# ============================================================================
# End-to-end pipeline test
# ============================================================================
class TestPipeline:
    def test_mini_pipeline(self):
        """Test the full pipeline with minimal data."""
        from src.data.preprocess import generate_demo_data
        from src.eval.global_eval import GlobalEvaluator
        from src.eval.instance_eval import InstanceEvaluator
        from src.graph.coi_graph import CoIGraph
        from src.graph.route_consistency import RouteConsistency
        from src.intent.rule_labeler import RuleLabeler
        from src.intent.taxonomy import Taxonomy
        from src.select.selector import DataSelector
        from src.simulator.rollout import generate_synthetic_pool
        from src.utils import load_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            # Phase 1: Preprocess
            data_dir = os.path.join(tmpdir, "data")
            stats = generate_demo_data(data_dir)
            assert stats["total_dialogues"] > 0

            # Phase 2: Label + Graph
            taxonomy = Taxonomy()
            labeler = RuleLabeler(taxonomy)
            train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
            intent_data = labeler.label_dataset(train_data)
            assert len(intent_data) > 0

            coi_graph = CoIGraph(taxonomy=taxonomy, ngram_sizes=[3])
            coi_graph.build_from_sequences(intent_data, train_data)
            assert len(coi_graph.valid_edges) > 0

            graph_dir = os.path.join(tmpdir, "graph")
            coi_graph.save(graph_dir)

            # Phase 3: Route consistency
            route_scorer = RouteConsistency(coi_graph)
            dev_data = load_jsonl(os.path.join(data_dir, "dev.jsonl"))
            dev_intents = labeler.label_dataset(dev_data)
            dev_scores = route_scorer.score_dataset(dev_intents, dev_data)
            assert len(dev_scores) > 0
            assert all(0 <= s["route_score"] <= 1 for s in dev_scores)

            # Phase 5: Synthetic pool
            pool_dir = os.path.join(tmpdir, "pool")
            pool = generate_synthetic_pool(
                num_dialogues=30, max_turns=10, seed=42, output_dir=pool_dir
            )
            assert len(pool) == 30

            # Phase 6: Evaluation
            pool_dialogues = [
                {"dialogue_id": p["dialogue_id"], "turns": p["turns"],
                 "meta": {"outcome": p.get("outcome", "unknown")}}
                for p in pool
            ]
            pool_intent = labeler.label_dataset(pool_dialogues)
            pool_seqs = [d["intent_sequence"] for d in pool_intent]

            instance_eval = InstanceEvaluator(route_scorer=route_scorer, real_dialogues=train_data)
            global_eval = GlobalEvaluator(coi_graph, train_data, taxonomy)

            inst_scores = instance_eval.evaluate_dataset(pool_dialogues, pool_intent)
            assert len(inst_scores) == 30

            glob_metrics = global_eval.evaluate(pool_dialogues, pool_seqs)
            assert "kl_divergence" in glob_metrics
            assert "js_divergence" in glob_metrics

            # Phase 7: Selection
            selector = DataSelector(
                instance_eval, global_eval,
                K=10, mc_iterations=5, greedy_iterations=3, seed=42
            )
            selections = selector.select_all(pool_dialogues, inst_scores, pool_seqs)
            assert len(selections["random_k"]) == 10
            assert len(selections["instance_top_k"]) == 10
            assert len(selections["coi_selected_k"]) == 10

            # Compare
            comparison = selector.compare_selections(
                pool_dialogues, selections, pool_seqs, inst_scores
            )
            assert "random_k" in comparison
            assert "coi_selected_k" in comparison

            # Verify CoI-Selected has reasonable metrics
            coi_metrics = comparison["coi_selected_k"]
            assert coi_metrics["avg_instance_score"] >= 0
