"""Phase 4: User Simulator training script.

Trains an SFT-based user simulator using LoRA/QLoRA on a causal language model.
The simulator learns to generate user responses given dialogue context.
"""

import json
import os
from typing import Any, Dict, List

from src.utils import get_config, load_jsonl, project_root


def prepare_training_data(
    dialogues: List[Dict[str, Any]],
    max_length: int = 512,
) -> List[Dict[str, str]]:
    """Convert dialogues into simulator training examples.

    Each example has the dialogue history as input and the next user turn as
    the target output.
    """
    examples = []
    for d in dialogues:
        turns = d["turns"]
        for i, turn in enumerate(turns):
            if turn["role"] == "user" and i > 0:
                # Build context from previous turns
                context_parts = []
                for t in turns[:i]:
                    context_parts.append(f"{t['role'].capitalize()}: {t['text']}")
                context = "\n".join(context_parts)
                target = turn["text"]

                examples.append({
                    "input": f"Given the following dialogue, generate the user's response.\n\n{context}\n\nUser:",
                    "output": f" {target}",
                })

    return examples


def train_simulator(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Train the user simulator with LoRA.

    This function requires torch and transformers. It will gracefully degrade
    if GPU/model resources are not available.

    Returns:
        Training results dict with status and metadata.
    """
    if config is None:
        config = get_config()

    sim_cfg = config["simulator"]
    data_dir = config["dataset"]["processed_dir"]

    results: Dict[str, Any] = {
        "output_dir": sim_cfg["output_dir"],
        "base_model": sim_cfg["base_model"],
    }

    # Load training data
    train_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_path):
        print(f"[ERROR] Training data not found at {train_path}")
        print("[INFO] Run data preprocessing first: python scripts/run_phase1.py")
        results["status"] = "error_no_data"
        return results

    dialogues = load_jsonl(train_path)
    examples = prepare_training_data(dialogues, sim_cfg["max_length"])
    print(f"[INFO] Prepared {len(examples)} training examples for simulator")
    results["num_examples"] = len(examples)

    # Save prepared data
    output_dir = sim_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_examples.json"), "w") as f:
        json.dump(examples[:5], f, indent=2)  # save samples
    print(f"[INFO] Saved example samples to {output_dir}/train_examples.json")

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"[INFO] Loading base model: {sim_cfg['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(sim_cfg["base_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            sim_cfg["base_model"],
            torch_dtype=torch.float32,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=sim_cfg["lora_r"],
            lora_alpha=sim_cfg["lora_alpha"],
            lora_dropout=sim_cfg["lora_dropout"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Tokenize
        def tokenize(example):
            full_text = example["input"] + example["output"]
            return tokenizer(
                full_text,
                truncation=True,
                max_length=sim_cfg["max_length"],
                padding="max_length",
            )

        from datasets import Dataset
        dataset = Dataset.from_list(examples)
        tokenized = dataset.map(
            tokenize,
            remove_columns=["input", "output"],
        )
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

        # Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=sim_cfg["num_epochs"],
            per_device_train_batch_size=sim_cfg["batch_size"],
            learning_rate=sim_cfg["learning_rate"],
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
        print(f"[INFO] Simulator saved to {output_dir}")
        results["status"] = "completed"

    except ImportError as e:
        print(f"[WARN] Cannot train simulator (missing deps): {e}")
        print("[INFO] Skipping actual training. Data preparation completed.")
        results["status"] = "skipped_missing_deps"
    except Exception as e:
        print(f"[WARN] Simulator training failed: {e}")
        print("[INFO] Data preparation completed. Fix the error and retry.")
        results["status"] = f"failed: {str(e)}"

    return results
