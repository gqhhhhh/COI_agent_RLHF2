"""Phase 2A: Rule-based heuristic intent labeler for user turns.

This module applies keyword and pattern matching rules to classify each user
utterance into one of the CoI intent categories. It serves as a fast baseline
labeler for pipeline validation.
"""

import re
from typing import Any, Dict, List, Tuple

from src.intent.taxonomy import Taxonomy


class RuleLabeler:
    """Rule/heuristic based intent labeler for user utterances."""

    def __init__(self, taxonomy: Taxonomy | None = None):
        self.taxonomy = taxonomy or Taxonomy()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for each intent category."""
        self.patterns: List[Tuple[str, List[re.Pattern], float]] = [
            # (intent_name, compiled_patterns, confidence)
            # Order matters: first match wins. More specific patterns first.

            ("EndSuccess", [
                re.compile(r"\b(donated|done|completed|finished|glad\s+i?\s*could\s+help)\b", re.I),
                re.compile(r"\b(all\s+done|happy\s+to\s+(help|contribute))\b", re.I),
            ], 0.8),

            ("Action", [
                re.compile(r"\b(i('ll|will)\s+(donate|give|contribute))\b", re.I),
                re.compile(r"\b(let\s+me\s+(donate|give|sign\s+up))\b", re.I),
                re.compile(r"\b(i('ll|will)\s+give\s+\$?\d+)\b", re.I),
                re.compile(r"\b(donate\s+\$?\d+)\b", re.I),
                re.compile(r"\b(i\s+can\s+give)\b", re.I),
            ], 0.85),

            ("Reject", [
                re.compile(r"\b(no\s+thanks?|not\s+interested|i\s+won'?t|i\s+refuse)\b", re.I),
                re.compile(r"\b(don'?t\s+want\s+to\s+(donate|give))\b", re.I),
                re.compile(r"\b(i('ll)?\s+pass)\b", re.I),
                re.compile(r"\b(no,?\s+i\s+(really\s+)?can'?t)\b", re.I),
                re.compile(r"\b(please\s+stop)\b", re.I),
                re.compile(r"\b(not\s+going\s+to)\b", re.I),
            ], 0.85),

            ("Concern", [
                re.compile(r"\b(i'?m\s+not\s+sure)\b", re.I),
                re.compile(r"\b(worried|concern|hesitant|doubt)\b", re.I),
                re.compile(r"\b(can'?t\s+afford)\b", re.I),
                re.compile(r"\b(how\s+do\s+i\s+know)\b", re.I),
                re.compile(r"\b(i\s+don'?t\s+trust)\b", re.I),
                re.compile(r"\b(seems?\s+like\s+a\s+lot)\b", re.I),
                re.compile(r"\b(but\s+i)\b", re.I),
                re.compile(r"\b(i\s+do\s+worry)\b", re.I),
                re.compile(r"\b(i'?m\s+concerned)\b", re.I),
                re.compile(r"\b(already\s+(donate|give|stretch))\b", re.I),
                re.compile(r"\b(other\s+priorities)\b", re.I),
            ], 0.7),

            ("Positive", [
                re.compile(r"\b(sounds?\s+(great|good|nice|interesting|wonderful|worthwhile))\b", re.I),
                re.compile(r"\b(i\s+(think|believe)\s+i'?d?\s+like)\b", re.I),
                re.compile(r"\b(that'?s?\s+(great|good|impressive|reassuring|convincing))\b", re.I),
                re.compile(r"\b(i\s+agree|sure)\b", re.I),
                re.compile(r"\b(worth\s+supporting)\b", re.I),
                re.compile(r"\b(i\s+do\s+care)\b", re.I),
                re.compile(r"\b(good\s+cause)\b", re.I),
                re.compile(r"\b(you'?re\s+welcome)\b", re.I),
                re.compile(r"\b(my\s+pleasure)\b", re.I),
                re.compile(r"\b(glad\s+to)\b", re.I),
            ], 0.7),

            ("Inquiry", [
                re.compile(r"\?\s*$"),  # ends with question mark
                re.compile(r"\b(what|how|where|when|why|who|which|tell\s+me)\b", re.I),
                re.compile(r"\b(can\s+you\s+(tell|explain))\b", re.I),
                re.compile(r"\b(how\s+(much|many|long|does))\b", re.I),
            ], 0.65),
        ]

    def label_utterance(self, text: str) -> Tuple[str, float]:
        """Label a single user utterance.

        Returns (intent, confidence).
        """
        text = text.strip()
        if not text:
            return "Neutral", 0.5

        for intent_name, patterns, conf in self.patterns:
            for pat in patterns:
                if pat.search(text):
                    return intent_name, conf

        return "Neutral", 0.5

    def label_dialogue(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Label all user turns in a dialogue.

        Input: a unified dialogue dict.
        Returns: an intent annotation dict.
        """
        user_turns = []
        intent_sequence = []
        turn_id = 0

        for turn in dialogue["turns"]:
            if turn["role"] == "user":
                intent, conf = self.label_utterance(turn["text"])
                user_turns.append({
                    "turn_id": turn_id,
                    "text": turn["text"],
                    "intent": intent,
                    "confidence": conf,
                })
                intent_sequence.append(intent)
            turn_id += 1

        return {
            "dialogue_id": dialogue["dialogue_id"],
            "user_turns": user_turns,
            "intent_sequence": intent_sequence,
        }

    def label_dataset(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Label all dialogues in a dataset."""
        return [self.label_dialogue(d) for d in dialogues]
