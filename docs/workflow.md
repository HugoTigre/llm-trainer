# Workflow

This document describes the steps this project supports: from data preparation through training and evaluation. All paths and model settings are driven by a **config file**; each run is identified by a **run ID** so outputs stay organized.

## Config and run ID

- **Config file**: `config/default.yaml` (or pass `--config path/to/other.yaml` to scripts). It defines:
  - `run_id`: string that names this run (e.g. `"default"`, `"some-model-id"`). **Everything for that run is scoped by this ID**: prepared data, adapter, and eval outputs.
  - `model_id`: Hugging Face model ID (e.g. `mistralai/Mistral-7B-Instruct-v0.2`).
  - By default, train/val paths are `data/<run_id>/train.jsonl` and `data/<run_id>/val.jsonl` (you can override in config if needed).
  - `quantization_bits`: 4, 8, or null (full precision); default 4.
  - `lora.*`, `training.*`, `eval.max_examples`: hyperparameters and eval size.
- **Layout per run** (all under the same `run_id`):
  - `data/<run_id>/train.jsonl`, `data/<run_id>/val.jsonl` — prepared data (from `prepare_data.py`).
  - `output/<run_id>/adapter/` — LoRA weights and tokenizer (from training).
  - `output/<run_id>/baseline_outputs.md` — base-model evaluation (optional step).
  - `output/<run_id>/finetuned_outputs.md` — fine-tuned model evaluation.
  - `output/<run_id>/` may also contain training checkpoints.

## Steps this project handles

### 1. Prepare data

- **Input**: `data/training_data.json` (default). Single JSON with `system_prompt`, optional `schema`, and `user_assistant_messages`. The schema (if present) is appended to the system prompt so training and baseline eval see the full prompt. See [Data format](data-format.md).
- **Script**: `python scripts/prepare_data.py [--config path/to/config.yaml] [--raw-file path/to/file.json]`
- **Output**: `data/<run_id>/train.jsonl` and `data/<run_id>/val.jsonl` in chat **messages** format. Training and eval scripts read these paths from config (derived from `run_id` by default).

### 2. (Optional) Baseline evaluation

- **Purpose**: Run the **base** model (no fine-tuning) on a few examples so you can compare later with the fine-tuned run.
- **Script**: `python scripts/run_baseline.py [--config path/to/config.yaml]`
- **Reads**: Config (model_id, data paths, eval.max_examples).
- **Writes**: `output/<run_id>/baseline_outputs.md` (user prompt, expected, actual for each example).

Use the same config (and thus same `run_id`) you will use for training and finetuned eval so baseline and finetuned outputs live in the same run folder.

### 3. Train

- **Purpose**: Fine-tune the model with QLoRA (quantized base, default 4-bit, + LoRA adapters). Only the adapter is trained; the rest of the model is frozen. Quantization is set in config (`quantization_bits`: 4, 8, or null for full precision).
- **Script**: `python scripts/train.py [--config path/to/config.yaml]`
- **Reads**: Config (model_id, data paths, lora.*, training.*) and the training JSONL.
- **Writes**:
  - `output/<run_id>/adapter/` — LoRA weights and tokenizer (this is what you use for evaluation).
  - Checkpoints under `output/<run_id>/` if configured.

Training time depends on dataset size, epochs, and hardware (CPU vs MPS/CUDA). See [Concepts](concepts.md) for how training works.

### 4. Post-train evaluation

- **Purpose**: Run the **fine-tuned** model (base with configurable quantization + LoRA adapter) on the **same** examples as the baseline.
- **Script**: `python scripts/run_finetuned_eval.py [--config path/to/config.yaml]`
- **Reads**: Config (model_id, adapter_dir derived from run_id, data paths, eval.max_examples). The adapter is loaded from `output/<run_id>/adapter/`.
- **Writes**: `output/<run_id>/finetuned_outputs.md` (same structure as baseline: user, expected, actual).

Compare `baseline_outputs.md` and `finetuned_outputs.md` in the same run folder to see whether the model improved (e.g. outputs closer to “expected”).

### 5. Iterate

- Add or edit raw examples, re-run **prepare data** (step 1).
- Optionally re-run **baseline** (step 2) if you changed data.
- Re-run **train** (step 3) and **finetuned eval** (step 4).

## Typical order

1. `scripts/prepare_data.py`
2. (Optional) `scripts/run_baseline.py`
3. `scripts/train.py`
4. `scripts/run_finetuned_eval.py`
5. Compare `output/<run_id>/baseline_outputs.md` and `output/<run_id>/finetuned_outputs.md`

All scripts are intended to be run from the **project root** with your virtual env activated.

## Future extensions

Possible additions (not implemented):

- **Merge and export**: Merge the LoRA adapter into the base model and save a single Hugging Face model; optionally convert to GGUF (e.g. via llama.cpp) for deployment. Testing is currently done with the HF stack (base + adapter) only.
- **Validation during training**: Use the validation set for evaluation during training (e.g. loss or a simple metric).
- **Other runtimes**: Integration with MLX, llama.cpp, or other backends for inference or deployment.
