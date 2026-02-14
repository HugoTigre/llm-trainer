# Training concepts

This doc explains what happens during fine-tuning: what we're doing, how tokens and labels work, and what gets saved.

## 1. What we're doing in one sentence

We **fine-tune** a base causal LM so it learns to output our target replies when we give it the system message + context + user prompt. We do this **efficiently** by only training a small set of extra parameters (LoRA) while keeping the rest of the model frozen.

## 2. Fine-tuning vs training from scratch

- **Training from scratch**: Learn all parameters from data (huge data, huge compute).
- **Fine-tuning**: Start from a model that already knows language; **update** it on our dataset so it learns our task. We don't change all parameters, we use **LoRA** to add a small number of trainable parameters.

## 3. QLoRA = quantized base + LoRA

- **Quantization**: When we **load** the base model, we can load it in 4-bit or 8-bit form (config: `quantization_bits`) so it uses less RAM. The model weights are "compressed" in memory. We **don't train** these quantized weights. Default is 4-bit; set to `null` for full precision.
- **LoRA (Low-Rank Adaptation)**: We **add** a small set of new parameters (e.g. two small matrices per layer we adapt). Only these **LoRA parameters** are trained; the rest of the model is frozen. So we train a small fraction of the parameters, which is fast and uses little memory.
- **QLoRA** = load base in **Q**uantized form (default 4-bit) + train **LoRA** adapters.

## 4. Tokens, input_ids, and labels (plain language)

- **Tokens**: The tokenizer turns text into a list of integers (tokens). Roughly: one token ≈ one short word or part of a word.
- **input_ids**: The list of token IDs for the **full** conversation (system + user + assistant) is what we feed into the model.
- **Labels**: For training we need "at each position, what token **should** the model have predicted?" That's the **labels**. We set **labels** to -100 for every position that belongs to system or user. In PyTorch, **-100 means "ignore this position when computing the loss"**. So the model is only corrected on the assistant tokens.

**How we build input_ids and labels:**

1. Take the full conversation and turn it into one string with the tokenizer's chat template.
2. Tokenize that string → **input_ids**.
3. Find where the assistant reply starts (by tokenizing only system+user with generation prompt and measuring length).
4. Build **labels**: -100 for positions before the assistant reply; same token IDs as input_ids for the assistant part.

## 5. Loss and one training step (plain language)

- **Logits**: The model's raw scores (one per possible token) at each position.
- **Loss**: A single number measuring "how wrong" the model was, computed only on assistant tokens (where labels ≠ -100). Training minimizes this loss.
- **One step**: Forward (input_ids → logits), compute loss vs labels, backward (gradients), optimizer updates only the LoRA weights. Repeat.

## 6. After training: what we save

- We save **only the LoRA adapter** (small set of matrices), not the full model. That goes to `output/<run_id>/adapter/` (see [Workflow](workflow.md)).
- To use it: load the **same** base model again, then **apply** the saved adapter on top. Evaluation in this project does exactly that (base with same quantization + adapter). Merging the adapter into the base or exporting to other formats (e.g. GGUF) is left for future use.

## 7. max_seq_length (tokens, not characters)

- **max_seq_length** is in **tokens**, not characters. Roughly 1 token ≈ 3–4 characters for English/JSON.
- If an example tokenizes to more than this, it gets **truncated** (tail is dropped). Longer sequences use more memory (attention is quadratic in length), so we cap length to fit in RAM. You can set it in the config (`training.max_seq_length`). If you get out-of-memory errors, try lowering it or reducing `gradient_accumulation_steps`.

## 8. Key hyperparameters (in config)

- **max_seq_length**: Cap on token length per example.
- **batch_size**: Examples per step (often 1 or 2 to fit in RAM).
- **gradient_accumulation_steps**: Effective batch = batch_size × this.
- **num_epochs**: How many times we go through the training set (config: `training.num_epochs`).
- **LoRA r (rank)**: Size of the small matrices; higher = more capacity but more memory. Typical 8–64. Set in config under `lora.*`.

## 9. device_map="auto"

- **auto**: The library puts the model on GPU/MPS if available. On Mac, training and eval use MPS when possible; there is a small workaround for a known MPS/torch.empty issue during load.
