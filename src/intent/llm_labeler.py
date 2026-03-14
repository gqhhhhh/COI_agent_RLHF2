"""Phase 2B: LLM-based intent labeler for user turns.

Uses an LLM (via transformers or API) to classify user utterances into CoI
intent categories. This provides higher quality labels than the rule-based
approach.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.intent.taxonomy import Taxonomy


class LLMLabeler:
    """LLM-based intent labeler for user utterances.

    Supports two backends:
    - 'api': Uses OpenAI-compatible API (requires OPENAI_API_KEY env var)
    - 'local': Uses a local transformers model
    """

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        backend: str = "api",
        model_name: str = "gpt-3.5-turbo",
    ):
        self.taxonomy = taxonomy or Taxonomy()
        self.backend = backend
        self.model_name = model_name
        self._client = None

    def _build_prompt(self, text: str, context: str = "") -> str:
        """Build classification prompt for the LLM."""
        intent_defs = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.taxonomy.intents.items()
        )

        prompt = f"""Classify the following user utterance into exactly one intent category.

Intent categories:
{intent_defs}

{f'Dialogue context: {context}' if context else ''}

User utterance: "{text}"

Respond with ONLY the intent label (one of: {', '.join(self.taxonomy.intent_names)}) and a confidence score between 0 and 1.
Format: INTENT|CONFIDENCE
Example: Inquiry|0.85"""
        return prompt

    def _parse_response(self, response: str) -> Tuple[str, float]:
        """Parse LLM response into (intent, confidence)."""
        response = response.strip()
        parts = response.split("|")
        if len(parts) >= 2:
            intent = parts[0].strip()
            try:
                conf = float(parts[1].strip())
            except ValueError:
                conf = 0.5
        else:
            intent = response.strip()
            conf = 0.5

        # Validate
        if not self.taxonomy.is_valid_intent(intent):
            # Try fuzzy match
            for name in self.taxonomy.intent_names:
                if name.lower() in intent.lower():
                    return name, conf
            return "Neutral", 0.3

        return intent, conf

    def label_utterance(
        self, text: str, context: str = ""
    ) -> Tuple[str, float]:
        """Label a single utterance. Falls back to Neutral if LLM unavailable."""
        try:
            prompt = self._build_prompt(text, context)
            response = self._call_llm(prompt)
            return self._parse_response(response)
        except Exception:
            return "Neutral", 0.3

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend."""
        if self.backend == "api":
            return self._call_api(prompt)
        else:
            return self._call_local(prompt)

    def _call_api(self, prompt: str) -> str:
        """Call OpenAI-compatible API."""
        try:
            import openai
            if self._client is None:
                self._client = openai.OpenAI()
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

    def _call_local(self, prompt: str) -> str:
        """Call local transformers model."""
        try:
            from transformers import pipeline
            if self._client is None:
                self._client = pipeline("text-generation", model=self.model_name)
            result = self._client(prompt, max_new_tokens=20, do_sample=False)
            return result[0]["generated_text"][len(prompt):]
        except Exception as e:
            raise RuntimeError(f"Local model call failed: {e}")

    def label_dialogue(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Label all user turns in a dialogue."""
        user_turns = []
        intent_sequence = []
        context_parts = []
        turn_id = 0

        for turn in dialogue["turns"]:
            if turn["role"] == "user":
                context = " | ".join(context_parts[-4:])
                intent, conf = self.label_utterance(turn["text"], context)
                user_turns.append({
                    "turn_id": turn_id,
                    "text": turn["text"],
                    "intent": intent,
                    "confidence": conf,
                })
                intent_sequence.append(intent)
            context_parts.append(f"{turn['role']}: {turn['text']}")
            turn_id += 1

        return {
            "dialogue_id": dialogue["dialogue_id"],
            "user_turns": user_turns,
            "intent_sequence": intent_sequence,
        }

    def label_dataset(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Label all dialogues in a dataset."""
        return [self.label_dialogue(d) for d in dialogues]
