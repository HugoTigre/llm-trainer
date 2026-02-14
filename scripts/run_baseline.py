#!/usr/bin/env python3
"""
Baseline evaluation: run the base model (no fine-tuning) on examples from val/train JSONL.
Results are written to output/<run_id>/baseline_outputs.md for comparison with the
fine-tuned run (run_finetuned_eval.py).

Reads: config (model_id, data paths, eval.max_examples)
Writes: output/<run_id>/baseline_outputs.md

Run from project root with venv activated:
  python scripts/run_baseline.py [--config path/to/config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config
from util.data import iter_eval_examples
from util.log import setup_logging
from util.torch_utils import get_bitsandbytes_config, patch_torch_empty_for_mps


def run_baseline():
    """Load tokenizer and model (quantization from config), run on examples, write baseline_outputs.md."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="Run baseline (base model) evaluation.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    model_id = cfg["model_id"]
    val_path = cfg["data"]["val_path"]
    train_path = cfg["data"]["train_path"]
    max_examples = cfg["eval"]["max_examples"]
    output_dir = cfg["output_dir"]
    out_path = output_dir / "baseline_outputs.md"

    quant_bits = cfg.get("quantization_bits", 4)
    quant_label = f"{quant_bits}-bit" if quant_bits else "full precision"
    log.info("Processing up to %s examples (streaming from val then train). Loading tokenizer and model (%s)...", max_examples, quant_label)
    log.info("(First run may download the model; this can take a while.)")

    quantization_config = get_bitsandbytes_config(quant_bits)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    patch_torch_empty_for_mps()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    device = next(model.parameters()).device
    log.info("Model loaded on %s. Running inference...", device)

    results = []
    for i, ex in enumerate(iter_eval_examples(val_path, train_path, max_examples)):
        messages = ex["messages"]
        input_messages = [m for m in messages if m["role"] != "assistant"]
        expected = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        prompt = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        reply_ids = out[0][input_len:]
        actual = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        user_display = user_content[:200] + "..." if len(user_content) > 200 else user_content

        results.append({
            "index": i + 1,
            "user_prompt_display": user_display,
            "expected": expected,
            "actual": actual,
        })
        log.info("  Example %s done.", i + 1)

    if not results:
        log.error("No examples found in %s or %s.", val_path, train_path)
        return 1
    log.info("Processed %s examples.", len(results))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Baseline outputs (base model, no fine-tuning)\n\n")
        f.write(f"Run ID: {cfg['run_id']}. Compare with finetuned_outputs.md for the same run.\n\n")
        for r in results:
            f.write(f"## Example {r['index']}\n\n")
            f.write("**User (context + prompt, truncated):**\n```\n")
            f.write(r["user_prompt_display"] + "\n")
            f.write("```\n\n**Expected (gold):**\n```\n")
            f.write(r["expected"] + "\n")
            f.write("```\n\n**Actual (base model):**\n```\n")
            f.write(r["actual"] + "\n")
            f.write("```\n\n---\n\n")

    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(run_baseline())
