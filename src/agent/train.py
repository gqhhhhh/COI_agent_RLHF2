"""Phase 8.1: Agent SFT training script.

Trains dialogue agents using supervised fine-tuning on selected data subsets.
Supports training with LoRA/QLoRA for efficient fine-tuning.
"""

import json
import os
from typing import Any, Dict, List

from src.utils import get_config, load_jsonl


def prepare_agent_training_data(
    dialogues: List[Dict[str, Any]],
    max_length: int = 512,
) -> List[Dict[str, str]]:
    """Convert dialogues into agent training examples.

    Each example has the dialogue history as input and the next agent turn as target.
    """
    examples = []
    for d in dialogues:
        turns = d.get("turns", [])
        for i, turn in enumerate(turns):
            if turn["role"] == "agent" and i > 0:
                # Build context from previous turns
                context_parts = []
                for t in turns[:i]:
                    context_parts.append(f"{t['role'].capitalize()}: {t['text']}")
                context = "\n".join(context_parts)
                target = turn["text"]

                examples.append({
                    "input": f"Generate the agent's persuasive response.\n\n{context}\n\nAgent:",
                    "output": f" {target}",
                })
    return examples


def train_agent(
    data_path: str,
    output_dir: str,
    config: Dict[str, Any] | None = None,
    strategy_name: str = "unknown",
) -> Dict[str, Any]:
    """Train an agent on a specific data selection.

    Args:
        data_path: Path to selected dialogues JSONL.
        output_dir: Where to save the trained model.
        config: Project config dict.
        strategy_name: Name of the selection strategy (for logging).

    Returns:
        Training results dict.
    """
    if config is None:
        config = get_config()

    agent_cfg = config["agent"]

    # Load data
    if not os.path.exists(data_path):
        print(f"[ERROR] Data not found at {data_path}")
        return {"error": f"Data not found: {data_path}"}

    dialogues = load_jsonl(data_path)
    examples = prepare_agent_training_data(dialogues, agent_cfg["max_length"])
    print(f"[INFO] Strategy '{strategy_name}': {len(examples)} training examples from {len(dialogues)} dialogues")

    os.makedirs(output_dir, exist_ok=True)

    # Save examples for inspection
    with open(os.path.join(output_dir, "train_examples_sample.json"), "w") as f:
        json.dump(examples[:3], f, indent=2, ensure_ascii=False)

    results = {
        "strategy": strategy_name,
        "num_dialogues": len(dialogues),
        "num_examples": len(examples),
        "output_dir": output_dir,
    }

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"[INFO] Loading base model: {agent_cfg['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(agent_cfg["base_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            agent_cfg["base_model"],
            torch_dtype=torch.float32,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=agent_cfg["lora_r"],
            lora_alpha=agent_cfg["lora_alpha"],
            lora_dropout=agent_cfg["lora_dropout"],
        )
        model = get_peft_model(model, lora_config)

        def tokenize(example):
            full_text = example["input"] + example["output"]
            return tokenizer(
                full_text,
                truncation=True,
                max_length=agent_cfg["max_length"],
                padding="max_length",
            )

        from datasets import Dataset
        dataset = Dataset.from_list(examples)
        tokenized = dataset.map(tokenize, remove_columns=["input", "output"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=agent_cfg["num_epochs"],
            per_device_train_batch_size=agent_cfg["batch_size"],
            learning_rate=agent_cfg["learning_rate"],
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
        )
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        results["status"] = "completed"
        print(f"[INFO] Agent ({strategy_name}) saved to {output_dir}")

    except ImportError as e:
        print(f"[WARN] Cannot train agent (missing deps): {e}")
        results["status"] = "skipped_missing_deps"
    except Exception as e:
        print(f"[WARN] Agent training failed: {e}")
        results["status"] = f"failed: {str(e)}"

    return results


def train_all_agents(
    selected_dir: str,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Train agents for all selection strategies.

    Args:
        selected_dir: Directory containing strategy subdirectories.
        config: Project config dict.

    Returns:
        Dict mapping strategy name to training results.
    """
    if config is None:
        config = get_config()

    strategies = ["random_k", "instance_top_k", "coi_selected_k"]
    all_results = {}

    for strategy in strategies:
        data_path = os.path.join(selected_dir, strategy, "selected.jsonl")
        output_dir = os.path.join(config["agent"]["output_dir"], strategy)
        results = train_agent(data_path, output_dir, config, strategy)
        all_results[strategy] = results

    return all_results
