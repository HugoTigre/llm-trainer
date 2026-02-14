"""Helpers for loading training/eval data (JSONL with 'messages' per line)."""

import json
from pathlib import Path


def iter_examples(path: Path, max_n: int):
    """Yield up to max_n examples from a JSONL file (each line has 'messages'). File is closed when the generator is exhausted."""
    if not path.exists():
        return
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_n:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass


def iter_eval_examples(val_path: Path, train_path: Path, max_examples: int):
    """Yield up to max_examples from val, then from train if needed. One example in memory at a time."""
    n = 0
    for ex in iter_examples(val_path, max_examples):
        yield ex
        n += 1
    for ex in iter_examples(train_path, max_examples - n):
        yield ex
