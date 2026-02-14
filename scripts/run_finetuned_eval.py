#!/usr/bin/env python3
"""
Post-train evaluation: run the fine-tuned model (base + LoRA adapter) on the same
examples as the baseline. Single-shot generation (one request → one response).
No constrained decoding; the exact raw model response is logged for analysis.

Reads:  config (model_id, adapter_dir, data paths, eval.max_examples)
Writes: output/<run_id>/finetuned_outputs.md

Run from project root with venv activated:
  python scripts/run_finetuned_eval.py [--config path/to/config.yaml]
"""

import argparse
import json
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


def run_finetuned_eval():
    """Load base + adapter, run on examples, write finetuned_outputs.md."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    parser = argparse.ArgumentParser(description="Run fine-tuned model evaluation.")
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
    adapter_dir = cfg["adapter_dir"]
    val_path = cfg["data"]["val_path"]
    train_path = cfg["data"]["train_path"]
    training_data_path = cfg["data"].get("training_data_path")
    max_examples = cfg["eval"]["max_examples"]
    include_schema_in_eval = cfg["eval"].get("include_schema_in_eval", True)
    output_dir = cfg["output_dir"]
    out_path = output_dir / "finetuned_outputs.md"

    # When include_schema_in_eval is False, use system_prompt only (no schema) from training_data.json
    eval_system_prompt_no_schema = None
    if not include_schema_in_eval and training_data_path and training_data_path.exists():
        try:
            data = json.loads(training_data_path.read_text(encoding="utf-8"))
            eval_system_prompt_no_schema = (data.get("system_prompt") or "").strip()
        except (json.JSONDecodeError, OSError):
            pass

    if include_schema_in_eval:
        log.info("Eval mode: using full system prompt from examples (includes schema).")
    else:
        log.info("Eval mode: using system prompt without schema (from training_data.json).")

    if not adapter_dir.exists():
        log.error("Adapter not found at %s. Run training (train/train.py) first.", adapter_dir)
        return 1

    log.info("Processing up to %s examples (streaming from val then train).", max_examples)
    log.info("Loading tokenizer from adapter dir...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)

    quant_bits = cfg.get("quantization_bits", 4)
    quant_label = f"{quant_bits}-bit" if quant_bits else "full precision"
    log.info("Loading base model (%s)...", quant_label)
    patch_torch_empty_for_mps()
    quantization_config = get_bitsandbytes_config(quant_bits)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    log.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    device = next(model.parameters()).device
    log.info("Model (base + adapter) loaded on %s. Running inference...", device)

    results = []
    for i, ex in enumerate(iter_eval_examples(val_path, train_path, max_examples)):
        messages = ex["messages"]
        if not include_schema_in_eval and eval_system_prompt_no_schema:
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            if user_msg is None:
                input_messages = [m for m in messages if m["role"] != "assistant"]
            else:
                input_messages = [
                    {"role": "system", "content": eval_system_prompt_no_schema},
                    user_msg,
                ]
        else:
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
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        reply_ids = out[0][input_len:]
        # Log exactly what the model outputs (no parsing or normalization)
        actual = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "\n\n" in user_content:
            user_question = user_content.split("\n\n")[-1].strip()
            user_context_display = user_content[:200] + "..." if len(user_content) > 200 else user_content
        else:
            user_question = user_content.strip() or "(empty)"
            user_context_display = user_content[:200] + "..." if len(user_content) > 200 else user_content

        results.append({
            "index": i + 1,
            "user_question": user_question,
            "user_context_display": user_context_display,
            "expected": expected,
            "actual": actual,
        })
        log.info("  Example %s done.", i + 1)

    if not results:
        log.error("No examples found in %s or %s.", val_path, train_path)
        return 1
    log.info("Processed %s examples (same as baseline).", len(results))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Fine-tuned outputs (base + LoRA adapter)\n\n")
        f.write(f"Run ID: {cfg['run_id']}. Single-shot generation; actual = exact raw model response (no constrained decoding).\n\n")
        f.write("Expected = gold from val/train (e.g. JSON steps for schema-agent). Actual = model output as-is.\n\n")
        for r in results:
            f.write(f"## Example {r['index']}\n\n")
            f.write("**User question:** " + r["user_question"].replace("\n", " ") + "\n\n")
            f.write("**User (context, truncated):**\n```\n")
            f.write(r["user_context_display"] + "\n")
            f.write("```\n\n**Expected (gold):**\n```\n")
            f.write(r["expected"] + "\n")
            f.write("```\n\n**Actual (exact model response):**\n```\n")
            f.write(r["actual"] + "\n")
            f.write("```\n\n---\n\n")

    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(run_finetuned_eval())
