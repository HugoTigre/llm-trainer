#!/usr/bin/env python3
"""
Convert raw examples into chat training format.

Reads:  data/training_data.json (or --raw-file) with system_prompt, optional schema,
        and user_assistant_messages. Schema is appended to system_prompt so training sees full prompt.
Writes: data/<run_id>/train.jsonl, data/<run_id>/val.jsonl  (run_id from config).

Run from project root:
  python scripts/prepare_data.py [--config path/to/config.yaml] [--raw-file path/to/file.json]

See docs/data-format.md for the raw and messages schema.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config
from util.log import setup_logging

# Default path when --raw-file not set; overridden by config training.training_data_file -> data.training_data_path
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "training_data.json"

# Fraction of examples to use for validation (e.g. 0.1 = 10%)
VAL_RATIO = 0.1
# Seed so the split is reproducible
RANDOM_SEED = 42


def load_raw_examples(path: Path) -> list[dict]:
    """
    Load from a single JSON: system_prompt, optional schema, user_assistant_messages.
    Schema (if present) is appended to system_prompt so each example gets full system content.
    Returns a list of dicts, each with system_prompt (merged), user, assistant.
    """
    log = logging.getLogger(__name__)
    if not path.exists():
        log.error("%s not found. Run from project root: python scripts/prepare_data.py", path)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "user_assistant_messages" not in data:
        log.error("Raw file must be JSON with 'system_prompt' and 'user_assistant_messages'.")
        sys.exit(1)

    system_prompt = (data.get("system_prompt") or "").strip()
    schema = (data.get("schema") or "").strip()
    if schema:
        system_prompt = system_prompt + "\n\n" + schema
    pairs = data.get("user_assistant_messages") or []
    return [
        {"system_prompt": system_prompt, "user": p.get("user", ""), "assistant": p.get("assistant", "")}
        for p in pairs
    ]


def raw_to_messages(raw: dict, system_content: str | None = None) -> dict:
    """
    Convert one raw example into the "messages" format for training/eval.
    Raw has user (full message) and assistant. system_content can override raw["system_prompt"] when passed (not used when prompt comes from file).
    """
    user_content = (raw["user"] or "").strip()
    assistant_content = (raw["assistant"] or "").strip()
    system_msg = (system_content or raw.get("system_prompt") or "").strip()
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare train/val JSONL from raw examples.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML (default: config/default.yaml)")
    parser.add_argument("--raw-file", type=Path, default=None, help="Raw JSON path (default: data/training_data.json)")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    raw_path = args.raw_file if args.raw_file is not None else cfg["data"]["training_data_path"]
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    train_path = cfg["data"]["train_path"]
    val_path = cfg["data"]["val_path"]
    run_id = cfg["run_id"]

    log.info("Run ID: %s", run_id)
    log.info("Loading raw examples from %s", raw_path)
    raw_examples = load_raw_examples(raw_path)
    if not raw_examples:
        log.error("No valid examples found.")
        return 1
    log.info("  Loaded %s examples (system prompt includes schema from file).", len(raw_examples))

    # Convert each raw example to messages format (no override; prompt comes from file)
    messages_list = [raw_to_messages(r, system_content=None) for r in raw_examples]

    # Shuffle and split into train / val
    random.seed(RANDOM_SEED)
    random.shuffle(messages_list)
    n_val = max(1, int(len(messages_list) * VAL_RATIO))  # at least 1 in val if we have enough
    n_train = len(messages_list) - n_val
    train_examples = messages_list[:n_train]
    val_examples = messages_list[n_train:]

    # Write under data/<run_id>/
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for obj in train_examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log.info("Wrote %s examples to %s", n_train, train_path)

    with open(val_path, "w", encoding="utf-8") as f:
        for obj in val_examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log.info("Wrote %s examples to %s", n_val, val_path)

    log.info("Done. Data is ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
