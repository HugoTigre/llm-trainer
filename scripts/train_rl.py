#!/usr/bin/env python3
"""
RL training (REINFORCE): load SFT adapter, generate responses for prompts from train/val,
score with a rule-based reward, update LoRA, save to adapter_rl_dir.

Run after SFT: python scripts/train.py && python scripts/train_rl.py

Reads:  config (adapter_dir, adapter_rl_dir, model_id, data paths, rl.*)
Writes: output/<run_id>/adapter_rl/

Run from project root with venv activated:
  python scripts/train_rl.py [--config path/to/config.yaml]
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
from util.data import iter_examples
from util.log import setup_logging
from util.reward import reward_function_agent
from util.torch_utils import get_bitsandbytes_config, patch_torch_empty_for_mps


def load_prompts(train_path: Path, val_path: Path) -> list[list[dict]]:
    """Load prompts (input messages without assistant reply) from train and val JSONL."""
    prompts = []
    for path in (val_path, train_path):
        for ex in iter_examples(path, max_n=10**6):
            messages = ex.get("messages", [])
            if not messages:
                continue
            # Input = everything except the last (assistant) turn
            input_messages = [m for m in messages if m["role"] != "assistant"]
            if not input_messages:
                continue
            prompts.append(input_messages)
    return prompts


def run_train_rl():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    parser = argparse.ArgumentParser(description="Run RL (REINFORCE) training on SFT adapter.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    adapter_dir = cfg["adapter_dir"]
    adapter_rl_dir = cfg["adapter_rl_dir"]
    model_id = cfg["model_id"]
    train_path = cfg["data"]["train_path"]
    val_path = cfg["data"]["val_path"]
    rl = cfg["rl"]
    steps = rl["steps"]
    batch_size = rl["batch_size"]
    max_new_tokens = rl["max_new_tokens"]
    learning_rate = rl["learning_rate"]
    save_steps = rl["save_steps"]
    quant_bits = cfg.get("quantization_bits", 4)

    if not adapter_dir.exists():
        log.error("SFT adapter not found at %s. Run scripts/train.py first.", adapter_dir)
        return 1

    log.info("Loading prompts from %s and %s...", val_path, train_path)
    prompts = load_prompts(train_path, val_path)
    if not prompts:
        log.error("No prompts found in train/val JSONL.")
        return 1
    log.info("Loaded %s prompts.", len(prompts))

    log.info("Loading tokenizer from %s...", adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model and SFT adapter...")
    patch_torch_empty_for_mps()
    quantization_config = get_bitsandbytes_config(quant_bits)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    device = next(model.parameters()).device

    # Ensure LoRA adapter parameters are trainable (may be frozen when loading on top of quantized base)
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    # Only train LoRA parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        log.error("No trainable parameters found. LoRA adapter may not have been loaded correctly.")
        return 1
    log.info("Trainable parameters: %s", sum(p.numel() for p in trainable))
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    adapter_rl_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    for step in range(steps):
        model.train()
        # Sample batch of prompts
        batch_prompts = [random.choice(prompts) for _ in range(batch_size)]

        prompt_strings = []
        prompt_ids_list = []
        for input_messages in batch_prompts:
            prompt_str = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_strings.append(prompt_str)
            ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            prompt_ids_list.append(ids.to(device))

        # Generate (sample for exploration; fall back to greedy on numerical issues)
        responses = []
        response_ids_list = []
        with torch.no_grad():
            for prompt_ids in prompt_ids_list:
                input_ids = prompt_ids.unsqueeze(0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                try:
                    out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50, # greatly reduces the chance of hitting extreme probability values
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                except RuntimeError:
                    log.warning("Sampling failed (numerical instability), falling back to greedy for this sample.")
                    out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                response_ids = out[0][prompt_ids.shape[0] :]
                response_ids_list.append(response_ids)
                text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                responses.append(text)

        reward_values = [reward_function_agent(r) for r in responses]
        rewards = torch.tensor(reward_values, dtype=torch.float32, device=device)
        mean_reward = rewards.mean().item()

        # Log every response for the first 3 steps, then a sample every 10 steps
        if step < 3:
            for i, (r, rv) in enumerate(zip(responses, reward_values)):
                user_text = batch_prompts[i][-1].get("content", "")[:60]
                log.info("  [step %d sample %d] user=%r", step + 1, i, user_text)
                log.info("  [step %d sample %d] response=%r", step + 1, i, r[:200])
                log.info("  [step %d sample %d] reward=%.1f", step + 1, i, rv)
        elif (step + 1) % 10 == 0:
            user_text = batch_prompts[0][-1].get("content", "")[:60]
            log.info("  [sample] user=%r  response=%r  reward=%.1f", user_text, responses[0][:150], reward_values[0])

        # No positive rewards → no learning signal. Skip entirely to avoid
        # AdamW weight decay corrupting LoRA when gradients are zero.
        if mean_reward == 0.0:
            log.info("Step %s/%s  mean_reward=0.0 — no learning signal, skipping optimizer step.", step + 1, steps)
            continue

        # Compute log prob of generated responses (forward pass, need gradients)
        log_probs_list = []
        for prompt_ids, response_ids in zip(prompt_ids_list, response_ids_list):
            full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
            attention_mask = torch.ones_like(full_ids, dtype=torch.long, device=device)
            outputs = model(full_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prompt_len = prompt_ids.shape[0]
            resp_len = response_ids.shape[0]
            if resp_len == 0:
                log_probs_list.append(torch.tensor(0.0, device=device))
                continue
            log_probs = torch.log_softmax(logits[0], dim=-1)
            indices = full_ids[0, prompt_len : prompt_len + resp_len]
            positions = torch.arange(prompt_len - 1, prompt_len + resp_len - 1, device=device)
            lp = log_probs[positions, indices].sum()
            log_probs_list.append(lp)

        log_probs = torch.stack(log_probs_list)
        loss = -(rewards * log_probs).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("Step %s/%s  loss is NaN/Inf (mean_reward=%.4f), skipping optimizer step.", step + 1, steps, mean_reward)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        log.info("Step %s/%s  loss=%.4f  mean_reward=%.4f  min=%.1f max=%.1f nonzero=%.0f%%",
                 step + 1, steps, loss.item(), mean_reward,
                 rewards.min().item(), rewards.max().item(),
                 (rewards > 0).float().mean().item() * 100)

        if (step + 1) % save_steps == 0 or (step + 1) == steps:
            log.info("Saving adapter to %s...", adapter_rl_dir)
            model.save_pretrained(adapter_rl_dir)
            tokenizer.save_pretrained(adapter_rl_dir)

    log.info("RL training done. Adapter saved at %s", adapter_rl_dir)
    return 0


if __name__ == "__main__":
    sys.exit(run_train_rl())
