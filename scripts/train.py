#!/usr/bin/env python3
"""
Fine-tune a causal LM with QLoRA (4-bit + LoRA) on chat-format data.

Reads:  config (model, paths, hyperparameters), data/train.jsonl (messages format)
Writes: output/<run_id>/adapter/  (LoRA weights + tokenizer), output/<run_id>/ checkpoints

Run from project root with venv activated:
  python train/train.py [--config path/to/config.yaml]

See docs/workflow.md for the full workflow and docs/concepts.md for training concepts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config
from util.log import setup_logging
from util.torch_utils import get_bitsandbytes_config, patch_torch_empty_for_mps

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _patch_checkpoint_use_reentrant():
    """PyTorch 2.x warns if use_reentrant is not passed to checkpoint; default to False."""
    orig_checkpoint = torch.utils.checkpoint.checkpoint
    if getattr(orig_checkpoint, "_llm_trainer_patched", False):
        return

    def patched_checkpoint(*args, **kwargs):
        if "use_reentrant" not in kwargs:
            kwargs["use_reentrant"] = False
        return orig_checkpoint(*args, **kwargs)

    patched_checkpoint._llm_trainer_patched = True
    torch.utils.checkpoint.checkpoint = patched_checkpoint


# -----------------------------------------------------------------------------
# 1. Load tokenizer
# -----------------------------------------------------------------------------
def load_tokenizer(model_id: str):
    """Load the model tokenizer. Used to turn messages into token IDs."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -----------------------------------------------------------------------------
# 2. Load training data from JSONL
# -----------------------------------------------------------------------------
def load_messages_from_jsonl(path: Path) -> list[dict]:
    """Read JSONL: one JSON per line, each with key 'messages'."""
    if not path.exists():
        return []
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data:
                    examples.append(data)
            except json.JSONDecodeError:
                pass
    return examples


# -----------------------------------------------------------------------------
# 3. Build dataset: messages → input_ids and labels (only assistant tokens count)
# -----------------------------------------------------------------------------
def build_dataset(examples: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    """
    Turn each example (list of messages) into token IDs and labels.
    Labels: -100 for system and user tokens; real token IDs for the assistant reply.
    """
    input_ids_list = []
    labels_list = []

    for idx, ex in enumerate(examples):
        messages = ex["messages"]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids_encoded = tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        full_ids = full_ids_encoded["input_ids"]
        if len(full_ids) > max_seq_length:
            print(
                f"[truncation] example {idx}: {len(full_ids)} tokens -> {max_seq_length} "
                f"(dropped {len(full_ids) - max_seq_length} tokens from the end)"
            )
            full_ids = full_ids[:max_seq_length]

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        reply_start = min(len(prompt_ids), len(full_ids))

        labels = [-100] * reply_start + full_ids[reply_start:]
        if len(labels) < len(full_ids):
            labels += [-100] * (len(full_ids) - len(labels))
        elif len(labels) > len(full_ids):
            labels = labels[: len(full_ids)]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
    })


# -----------------------------------------------------------------------------
# 4. Load model in 4-bit and attach LoRA
# -----------------------------------------------------------------------------
def load_model_and_lora(cfg: dict):
    """
    Load base model (optionally quantized), then attach LoRA adapters.
    Only the LoRA parameters are trained; the rest is frozen.
    """
    model_id = cfg["model_id"]
    lora = cfg["lora"]
    quant_bits = cfg.get("quantization_bits", 4)
    quantization_config = get_bitsandbytes_config(quant_bits)

    patch_torch_empty_for_mps()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        target_modules=lora["target_modules"],
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if cfg.get("training", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    return model


# -----------------------------------------------------------------------------
# 5. Data collator
# -----------------------------------------------------------------------------
def get_collator(tokenizer, max_seq_length: int):
    """Pad input_ids and labels to the same length in each batch."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate_fn(examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        max_len = ((max_len + 7) // 8) * 8
        max_len = min(max_len, max_seq_length)

        input_ids = []
        labels = []
        attention_mask = []
        for ex in examples:
            ids = ex["input_ids"]
            labs = ex["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate_fn


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with QLoRA.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/default.yaml)",
    )
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    run_id = cfg["run_id"]
    model_id = cfg["model_id"]
    train_path = cfg["data"]["train_path"]
    val_path = cfg["data"]["val_path"]
    adapter_dir = cfg["adapter_dir"]
    output_dir = cfg["output_dir"]
    tr = cfg["training"]
    max_seq_length = tr["max_seq_length"]

    log.info("Run ID: %s", run_id)
    log.info("Model: %s", model_id)
    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_id)

    log.info("Loading training data...")
    train_examples = load_messages_from_jsonl(train_path)
    if not train_examples:
        log.error("No examples in %s", train_path)
        return 1
    log.info("  %s examples", len(train_examples))
    train_dataset = build_dataset(train_examples, tokenizer, max_seq_length)

    quant_label = f"{cfg.get('quantization_bits', 4)}-bit" if cfg.get("quantization_bits") else "full precision"
    log.info("Loading model (%s) and attaching LoRA...", quant_label)
    _patch_checkpoint_use_reentrant()
    model = load_model_and_lora(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    num_training_steps = (
        len(train_dataset) // (tr["batch_size"] * tr["gradient_accumulation_steps"])
    ) * tr["num_epochs"]
    warmup_steps = max(1, int(num_training_steps * tr["warmup_ratio"]))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        num_train_epochs=tr["num_epochs"],
        learning_rate=tr["learning_rate"],
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_steps=tr["save_steps"],
        save_total_limit=2,
        bf16=False,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # MPS does not support pin_memory; avoids warning on Mac
        gradient_checkpointing=tr.get("gradient_checkpointing", False),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=get_collator(tokenizer, max_seq_length),
    )

    log.info("Starting training...")
    trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving LoRA adapter to %s", adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info("Done. Adapter saved at %s", adapter_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
