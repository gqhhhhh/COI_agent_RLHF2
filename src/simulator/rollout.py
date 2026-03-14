"""Phase 4/5: Simulator rollout for synthetic dialogue generation.

Uses a trained (or template-based) user simulator together with a baseline agent
to generate synthetic candidate dialogues.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from src.intent.rule_labeler import RuleLabeler
from src.intent.taxonomy import Taxonomy
from src.utils import get_config, load_jsonl, save_jsonl, set_seed


class TemplateSimulator:
    """Template-based user simulator for pipeline testing.

    Generates user responses based on simple patterns. This is used as a
    fallback when the trained model is not available.
    """

    def __init__(self, seed: int = 42):
        set_seed(seed)
        self.responses = {
            "inquiry": [
                "What do they do exactly?",
                "How does the donation work?",
                "Can you tell me more about it?",
                "How much do people usually donate?",
                "Where does the money go?",
            ],
            "positive": [
                "That sounds interesting.",
                "I think that's a great cause.",
                "That's good to hear.",
                "I appreciate you sharing this.",
                "That makes sense.",
            ],
            "concern": [
                "I'm not sure I can afford that.",
                "How do I know it's legitimate?",
                "I have some concerns about where the money goes.",
                "That's a lot to think about.",
                "I'm a bit hesitant.",
            ],
            "reject": [
                "No, I'm not interested.",
                "I'll pass on this.",
                "I don't want to donate right now.",
                "No thanks.",
            ],
            "action": [
                "Okay, I'll donate $0.50.",
                "I'll give $0.25.",
                "Let me donate $1.00.",
                "Sure, I'll contribute $0.30.",
            ],
            "neutral": [
                "Okay.",
                "I see.",
                "Alright.",
                "Sure.",
                "Go on.",
            ],
            "end_success": [
                "Great, glad I could help!",
                "Happy to contribute.",
                "Done! Hope it helps.",
            ],
        }

    def respond(self, context: List[Dict[str, str]], turn_idx: int) -> str:
        """Generate a user response given the dialogue context."""
        max_turns = random.randint(6, 14)

        if turn_idx >= max_turns:
            if random.random() < 0.6:
                return random.choice(self.responses["action"])
            else:
                return random.choice(self.responses["reject"])

        # Simple state machine
        r = random.random()
        if turn_idx <= 2:
            if r < 0.6:
                return random.choice(self.responses["inquiry"])
            else:
                return random.choice(self.responses["neutral"])
        elif turn_idx <= 4:
            if r < 0.3:
                return random.choice(self.responses["positive"])
            elif r < 0.6:
                return random.choice(self.responses["concern"])
            else:
                return random.choice(self.responses["inquiry"])
        else:
            if r < 0.3:
                return random.choice(self.responses["positive"])
            elif r < 0.5:
                return random.choice(self.responses["concern"])
            elif r < 0.7:
                return random.choice(self.responses["action"])
            elif r < 0.85:
                return random.choice(self.responses["reject"])
            else:
                return random.choice(self.responses["neutral"])


class BaselineAgent:
    """Simple template-based baseline agent for synthetic pool generation."""

    def __init__(self):
        self.openers = [
            "Hi! Have you heard of Save the Children?",
            "Hello! I'd like to talk to you about a charity.",
            "Good day! Can I tell you about Save the Children?",
        ]
        self.info_responses = [
            "They help children around the world with education and healthcare.",
            "Save the Children provides food, education, and medical care to kids.",
            "They work in over 100 countries to improve children's lives.",
        ]
        self.persuade_responses = [
            "Even a small donation can make a big difference!",
            "Your contribution, no matter how small, helps children in need.",
            "87% of donations go directly to programs helping children.",
        ]
        self.address_concern = [
            "I understand your concern. They are very transparent about spending.",
            "That's a great question. They have excellent accountability ratings.",
            "I hear you. Even $0.10 can help provide a meal for a child.",
        ]
        self.closers = [
            "Thank you so much for your generosity!",
            "That's wonderful! Thank you!",
            "I really appreciate your contribution!",
        ]
        self.fail_closers = [
            "I understand. Thanks for your time.",
            "No problem. Have a great day!",
            "That's okay. Thank you for listening.",
        ]

    def respond(self, context: List[Dict[str, str]], turn_idx: int) -> str:
        """Generate an agent response."""
        if turn_idx == 0:
            return random.choice(self.openers)

        last_user = ""
        for turn in reversed(context):
            if turn["role"] == "user":
                last_user = turn["text"].lower()
                break

        # Simple rule-based response
        if any(w in last_user for w in ["what", "how", "tell", "?"]):
            return random.choice(self.info_responses)
        elif any(w in last_user for w in ["concern", "not sure", "worry", "afford", "trust"]):
            return random.choice(self.address_concern)
        elif any(w in last_user for w in ["donate", "give", "contribute", "i'll"]):
            return random.choice(self.closers)
        elif any(w in last_user for w in ["no", "pass", "not interested"]):
            return random.choice(self.fail_closers)
        else:
            return random.choice(self.persuade_responses)


def generate_synthetic_pool(
    num_dialogues: int = 100,
    max_turns: int = 20,
    seed: int = 42,
    output_dir: str = "data/synthetic_pool",
) -> List[Dict[str, Any]]:
    """Generate a pool of synthetic dialogues using simulator + agent.

    Returns list of synthetic dialogue dicts.
    """
    set_seed(seed)
    simulator = TemplateSimulator(seed)
    agent = BaselineAgent()
    labeler = RuleLabeler()

    dialogues = []

    for idx in range(num_dialogues):
        turns: List[Dict[str, str]] = []
        dialogue_id = f"syn_{idx:05d}"

        # Agent opens
        agent_text = agent.respond(turns, 0)
        turns.append({"role": "agent", "text": agent_text})

        for turn_i in range(1, max_turns):
            # User responds
            user_text = simulator.respond(turns, turn_i)
            turns.append({"role": "user", "text": user_text})

            # Check for termination
            user_lower = user_text.lower()
            if any(w in user_lower for w in ["donate", "give", "contribute", "i'll"]):
                agent_text = agent.respond(turns, turn_i + 1)
                turns.append({"role": "agent", "text": agent_text})
                break
            if any(w in user_lower for w in ["no thanks", "not interested", "i'll pass", "no,"]):
                agent_text = agent.respond(turns, turn_i + 1)
                turns.append({"role": "agent", "text": agent_text})
                break

            # Agent responds
            agent_text = agent.respond(turns, turn_i + 1)
            turns.append({"role": "agent", "text": agent_text})

        # Determine outcome from final turns
        last_user_text = ""
        for t in reversed(turns):
            if t["role"] == "user":
                last_user_text = t["text"].lower()
                break

        if any(w in last_user_text for w in ["donate", "give", "contribute"]):
            outcome = "success"
        elif any(w in last_user_text for w in ["no", "pass", "not interested"]):
            outcome = "fail"
        else:
            outcome = "unknown"

        # Label intents
        dialogue_dict = {
            "dialogue_id": dialogue_id,
            "dataset": "synthetic",
            "split": "synthetic",
            "meta": {"outcome": outcome},
            "turns": turns,
        }
        intent_data = labeler.label_dialogue(dialogue_dict)

        dialogues.append({
            "dialogue_id": dialogue_id,
            "source": "simulator_rollout",
            "turns": turns,
            "intent_sequence": intent_data["intent_sequence"],
            "outcome": outcome,
            "meta": {
                "num_turns": len(turns),
                "num_user_turns": sum(1 for t in turns if t["role"] == "user"),
            },
        })

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(dialogues, os.path.join(output_dir, "synthetic_pool.jsonl"))

    # Statistics
    stats = {
        "total": len(dialogues),
        "success": sum(1 for d in dialogues if d["outcome"] == "success"),
        "fail": sum(1 for d in dialogues if d["outcome"] == "fail"),
        "unknown": sum(1 for d in dialogues if d["outcome"] == "unknown"),
        "avg_turns": sum(d["meta"]["num_turns"] for d in dialogues) / len(dialogues),
        "avg_user_turns": sum(d["meta"]["num_user_turns"] for d in dialogues) / len(dialogues),
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[INFO] Generated {len(dialogues)} synthetic dialogues")
    print(f"[INFO] Stats: {json.dumps(stats, indent=2)}")

    return dialogues
