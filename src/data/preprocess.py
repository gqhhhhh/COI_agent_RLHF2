"""PersuasionForGood dataset loader and preprocessor.

Converts raw PersuasionForGood data into the unified dialogue schema.

Dataset overview
----------------
PersuasionForGood (Wang et al., 2019) contains ~1,017 dialogues where one
participant (the persuader / agent) tries to convince the other (the
persuadee / user) to donate to Save the Children.

Each dialogue has an outcome: the amount the persuadee agreed to donate.
We treat donation > 0 as "success" and donation == 0 as "fail".
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import save_jsonl, set_seed


# ---------------------------------------------------------------------------
# Raw data parsing
# ---------------------------------------------------------------------------


def _parse_raw_csv(csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse the raw PersuasionForGood CSV into dialogue groups.

    Returns a dict mapping dialogue_id -> list of turn dicts.
    """
    dialogues: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row.get("B2", row.get("dialogue_id", "")).strip()
            if not did:
                continue
            dialogues[did].append(row)
    return dict(dialogues)


def _parse_annotation_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse the annotation file to get donation outcomes.

    Returns a dict mapping dialogue_id -> metadata dict.
    """
    meta: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row.get("B2", row.get("dialogue_id", "")).strip()
            if not did:
                continue
            # Try to extract donation amount
            donation_str = row.get("B7", row.get("donation", "0")).strip()
            try:
                donation = float(donation_str) if donation_str else 0.0
            except ValueError:
                donation = 0.0
            meta[did] = {
                "donation": donation,
                "outcome": "success" if donation > 0 else "fail",
            }
    return meta


# ---------------------------------------------------------------------------
# Unified format conversion
# ---------------------------------------------------------------------------


def convert_dialogue(
    did: str,
    rows: List[Dict[str, str]],
    meta: Optional[Dict[str, Any]] = None,
    split: str = "train",
) -> Optional[Dict[str, Any]]:
    """Convert a raw dialogue into the unified schema."""
    turns: List[Dict[str, str]] = []

    for row in rows:
        # PersuasionForGood uses columns like 'Unit' for turn index,
        # 'B4' or 'er_text' for persuader, 'B5' or 'ee_text' for persuadee
        er_text = row.get("B4", row.get("er_text", "")).strip()
        ee_text = row.get("B5", row.get("ee_text", "")).strip()

        if er_text:
            turns.append({"role": "agent", "text": er_text})
        if ee_text:
            turns.append({"role": "user", "text": ee_text})

    if not turns:
        return None

    outcome = "unknown"
    if meta:
        outcome = meta.get("outcome", "unknown")

    return {
        "dialogue_id": did,
        "dataset": "persuasionforgood",
        "split": split,
        "meta": {"outcome": outcome, **(meta or {})},
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Demo / synthetic data for testing the pipeline
# ---------------------------------------------------------------------------

_DEMO_DIALOGUES = [
    {
        "dialogue_id": "demo_001",
        "outcome": "success",
        "turns": [
            ("agent", "Hi! Have you heard of Save the Children? It's a wonderful charity."),
            ("user", "No, what do they do?"),
            ("agent", "They help children in need around the world with education and healthcare."),
            ("user", "That sounds interesting. How can I help?"),
            ("agent", "You can donate any amount from your task earnings. Even $0.10 helps!"),
            ("user", "Hmm, I'm not sure I can afford much right now."),
            ("agent", "Every little bit counts! Even a small donation can make a difference."),
            ("user", "Okay, I think I can give $0.50. That sounds reasonable."),
            ("agent", "That's wonderful! Thank you so much for your generosity."),
            ("user", "You're welcome. Glad I could help."),
        ],
    },
    {
        "dialogue_id": "demo_002",
        "outcome": "success",
        "turns": [
            ("agent", "Hello! Would you be interested in donating to Save the Children today?"),
            ("user", "I've heard of them. What specifically would my donation go to?"),
            ("agent", "Your donation helps provide food, education, and medical care to children."),
            ("user", "That's a good cause. I do worry about overhead costs though."),
            ("agent", "Save the Children has very low overhead - 87% goes directly to programs."),
            ("user", "That's reassuring. I'll donate $1.00."),
            ("agent", "Thank you! Your donation will make a real difference."),
            ("user", "Happy to do it!"),
        ],
    },
    {
        "dialogue_id": "demo_003",
        "outcome": "fail",
        "turns": [
            ("agent", "Hi there! I'm here to tell you about Save the Children."),
            ("user", "Okay, go ahead."),
            ("agent", "They work globally to ensure children's rights and provide aid."),
            ("user", "I see. I'm not really in a position to donate right now."),
            ("agent", "Even a small amount like $0.10 could help."),
            ("user", "No, I really can't. I have my own expenses to worry about."),
            ("agent", "I understand. Thanks for your time."),
            ("user", "No problem. Good luck."),
        ],
    },
    {
        "dialogue_id": "demo_004",
        "outcome": "fail",
        "turns": [
            ("agent", "Hello! Can I tell you about a charity called Save the Children?"),
            ("user", "Sure, what is it?"),
            ("agent", "It's an organization dedicated to improving children's lives worldwide."),
            ("user", "That sounds nice, but I don't trust online charities."),
            ("agent", "They're very reputable and have been around since 1919."),
            ("user", "I still don't want to donate. Sorry."),
            ("agent", "That's okay. Thank you for listening."),
            ("user", "Sure. Bye."),
        ],
    },
    {
        "dialogue_id": "demo_005",
        "outcome": "success",
        "turns": [
            ("agent", "Hi! Do you know about Save the Children's mission?"),
            ("user", "Not really. Tell me more."),
            ("agent", "They focus on giving children a healthy start, education, and protection."),
            ("user", "Education is really important to me. How does the donation work?"),
            ("agent", "You can choose any amount from $0 to your full payment of $2."),
            ("user", "I think education is worth supporting. I'll give $0.30."),
            ("agent", "Thank you! That's very kind of you."),
            ("user", "No problem at all."),
        ],
    },
    {
        "dialogue_id": "demo_006",
        "outcome": "success",
        "turns": [
            ("agent", "Good day! I'd like to discuss a donation opportunity with you."),
            ("user", "What kind of donation?"),
            ("agent", "It's for Save the Children - they help vulnerable kids worldwide."),
            ("user", "I've donated to charities before. What makes this one different?"),
            ("agent", "They have programs in over 100 countries and focus on lasting change."),
            ("user", "That's impressive. But I'm concerned about how much actually reaches the kids."),
            ("agent", "Great question! Over 85 cents of every dollar goes to programs and services."),
            ("user", "Alright, that's convincing. I'll donate $0.75."),
            ("agent", "Wonderful! Thank you for your support!"),
            ("user", "My pleasure. I hope it helps."),
        ],
    },
    {
        "dialogue_id": "demo_007",
        "outcome": "fail",
        "turns": [
            ("agent", "Hi! Have you ever considered donating to help children in need?"),
            ("user", "Not really, no."),
            ("agent", "Save the Children does amazing work for kids around the world."),
            ("user", "I'm sure they do, but I'm not interested."),
            ("agent", "Even $0.10 could help provide meals for a child."),
            ("user", "I understand, but I'll pass. I have other priorities."),
            ("agent", "Of course. Thanks for your time anyway."),
            ("user", "Thanks. Goodbye."),
        ],
    },
    {
        "dialogue_id": "demo_008",
        "outcome": "success",
        "turns": [
            ("agent", "Hello! Let me share some information about Save the Children with you."),
            ("user", "Okay, I'm listening."),
            ("agent", "They provide emergency relief, education, and healthcare to children globally."),
            ("user", "How long have they been operating?"),
            ("agent", "Since 1919 - over 100 years of helping children."),
            ("user", "That's a long track record. Do they work in my country?"),
            ("agent", "Yes, they operate in the US and in over 100 countries worldwide."),
            ("user", "Alright. I think I'll donate $0.25."),
            ("agent", "That's great! Every bit helps. Thank you!"),
            ("user", "You're welcome!"),
        ],
    },
    {
        "dialogue_id": "demo_009",
        "outcome": "fail",
        "turns": [
            ("agent", "Hi there! I wanted to talk to you about helping children through Save the Children."),
            ("user", "Go ahead."),
            ("agent", "Your donation can help provide food and shelter to children in crisis."),
            ("user", "That sounds sad, but I already donate to other causes."),
            ("agent", "Every additional contribution helps. Even a small amount?"),
            ("user", "No thanks. I'm already stretched thin with my current donations."),
            ("agent", "I understand completely. Thank you for being generous elsewhere."),
            ("user", "Thanks for understanding."),
        ],
    },
    {
        "dialogue_id": "demo_010",
        "outcome": "success",
        "turns": [
            ("agent", "Hello! Would you like to learn about Save the Children?"),
            ("user", "Sure, what's their focus?"),
            ("agent", "They help children around the world with education, nutrition, and protection from harm."),
            ("user", "That sounds very worthwhile. I do care about children's welfare."),
            ("agent", "It's a cause close to many people's hearts. Would you consider donating?"),
            ("user", "How much do most people give?"),
            ("agent", "The average donation is around $0.35 to $1.00 from earnings."),
            ("user", "I see. Let me think... Yes, I'll donate $0.50."),
            ("agent", "Thank you so much! That's very generous of you!"),
            ("user", "Happy to contribute to a good cause."),
        ],
    },
]

# Additional dialogues for richer demo data
_DEMO_DIALOGUES_EXTRA = [
    {
        "dialogue_id": f"demo_{i:03d}",
        "outcome": random.choice(["success", "fail"]),
        "turns": [
            ("agent", "Hello! I'd like to tell you about Save the Children."),
            ("user", "What do they do?"),
            ("agent", "They provide essential services to children in need worldwide."),
            ("user", "Interesting. How can I donate?"),
            ("agent", "You can donate any amount from your task bonus."),
            ("user", "I'll think about it." if i % 2 == 0 else "Sure, I'll give a little."),
            ("agent", "Thank you for considering!" if i % 2 == 0 else "Thank you so much!"),
            ("user", "No problem." if i % 2 == 0 else "Glad to help."),
        ],
    }
    for i in range(11, 61)
]


def generate_demo_data(output_dir: str, seed: int = 42) -> Dict[str, Any]:
    """Generate demo dialogues for pipeline testing.

    Creates train/dev/test splits from built-in demo dialogues.
    Returns statistics dict.
    """
    set_seed(seed)
    all_dialogues = _DEMO_DIALOGUES + _DEMO_DIALOGUES_EXTRA

    # Convert to unified format
    records = []
    for d in all_dialogues:
        turns = [{"role": role, "text": text} for role, text in d["turns"]]
        outcome = d["outcome"]
        records.append(
            {
                "dialogue_id": d["dialogue_id"],
                "dataset": "persuasionforgood",
                "split": "",  # assigned below
                "meta": {"outcome": outcome, "donation": 0.5 if outcome == "success" else 0.0},
                "turns": turns,
            }
        )

    # Shuffle and split
    random.shuffle(records)
    n = len(records)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)

    for i, rec in enumerate(records):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_dev:
            rec["split"] = "dev"
        else:
            rec["split"] = "test"

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "dev", "test"]:
        split_data = [r for r in records if r["split"] == split_name]
        save_jsonl(split_data, os.path.join(output_dir, f"{split_name}.jsonl"))

    save_jsonl(records, os.path.join(output_dir, "all.jsonl"))

    # Statistics
    stats = {
        "total_dialogues": n,
        "train": sum(1 for r in records if r["split"] == "train"),
        "dev": sum(1 for r in records if r["split"] == "dev"),
        "test": sum(1 for r in records if r["split"] == "test"),
        "success": sum(1 for r in records if r["meta"]["outcome"] == "success"),
        "fail": sum(1 for r in records if r["meta"]["outcome"] == "fail"),
        "avg_turns": sum(len(r["turns"]) for r in records) / n if n else 0,
    }

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def load_real_data(raw_dir: str, output_dir: str, seed: int = 42) -> Dict[str, Any]:
    """Load and convert real PersuasionForGood data if available.

    Falls back to demo data if raw files are not found.
    """
    csv_path = os.path.join(raw_dir, "300_dialog.csv")
    if not os.path.exists(csv_path):
        # Try alternate file names
        for fname in ["dialogue.csv", "persuasion_data.csv", "data.csv"]:
            alt = os.path.join(raw_dir, fname)
            if os.path.exists(alt):
                csv_path = alt
                break
        else:
            print(f"[INFO] Raw data not found at {raw_dir}. Generating demo data.")
            return generate_demo_data(output_dir, seed)

    print(f"[INFO] Loading real data from {csv_path}")
    set_seed(seed)

    dialogues = _parse_raw_csv(csv_path)
    # Try to load annotations
    ann_path = os.path.join(raw_dir, "annotations.csv")
    meta_map = _parse_annotation_csv(ann_path) if os.path.exists(ann_path) else {}

    records = []
    for did, rows in dialogues.items():
        meta = meta_map.get(did)
        rec = convert_dialogue(did, rows, meta)
        if rec:
            records.append(rec)

    if not records:
        print("[WARN] No valid dialogues parsed. Generating demo data.")
        return generate_demo_data(output_dir, seed)

    # Split
    random.shuffle(records)
    n = len(records)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    for i, rec in enumerate(records):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_dev:
            rec["split"] = "dev"
        else:
            rec["split"] = "test"

    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "dev", "test"]:
        split_data = [r for r in records if r["split"] == split_name]
        save_jsonl(split_data, os.path.join(output_dir, f"{split_name}.jsonl"))
    save_jsonl(records, os.path.join(output_dir, "all.jsonl"))

    stats = {
        "total_dialogues": n,
        "train": sum(1 for r in records if r["split"] == "train"),
        "dev": sum(1 for r in records if r["split"] == "dev"),
        "test": sum(1 for r in records if r["split"] == "test"),
        "success": sum(1 for r in records if r["meta"]["outcome"] == "success"),
        "fail": sum(1 for r in records if r["meta"]["outcome"] == "fail"),
        "avg_turns": sum(len(r["turns"]) for r in records) / n if n else 0,
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats
