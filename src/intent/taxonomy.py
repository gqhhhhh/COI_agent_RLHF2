"""Taxonomy loader – reads intent definitions from YAML config.

The taxonomy is never hard-coded in Python; it is always read from the
configuration file (configs/taxonomy.yaml) so that adding or modifying
intents requires zero code changes.
"""

from typing import Any, Dict, List

from src.utils import load_yaml, project_root


class Taxonomy:
    """Dynamically loaded intent taxonomy."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = str(project_root() / "configs" / "taxonomy.yaml")
        raw = load_yaml(config_path)
        self.version: str = raw["version"]
        self.name: str = raw["name"]

        # Build user intent structures
        self.intents: Dict[str, Dict[str, Any]] = raw["intents"]
        self.intent_names: List[str] = list(self.intents.keys())
        self.intent2id: Dict[str, int] = {
            name: info["id"] for name, info in self.intents.items()
        }
        self.id2intent: Dict[int, str] = {v: k for k, v in self.intent2id.items()}
        self.num_intents: int = len(self.intent_names)

        # Agent actions (auxiliary)
        self.agent_actions: Dict[str, Dict[str, Any]] = raw.get("agent_actions", {})

    def is_valid_intent(self, intent: str) -> bool:
        return intent in self.intent2id

    def get_description(self, intent: str) -> str:
        return self.intents[intent]["description"]

    def __repr__(self) -> str:
        return f"Taxonomy(name={self.name}, version={self.version}, intents={self.intent_names})"
