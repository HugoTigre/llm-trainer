# Output folder

This folder is written by `scripts/train.py` after fine-tuning. Below is what each part is and what to use for the next step.

## `adapter/` (use this for the next step)

This is the **final LoRA adapter** saved by the script. Merge/export is not implemented yet (see [Workflow](workflow.md#future-extensions)).

| File | What it is |
|------|------------|
| **adapter_model.safetensors** | LoRA weights. The only trained parameters; load on top of the base model. |
| **adapter_config.json** | PEFT/LoRA config: base model id, rank (r), alpha, target modules, etc. Needed to load the adapter. |
| **tokenizer_config.json** | Tokenizer settings (same as base model). |
| **tokenizer.json** | Tokenizer vocabulary and merges. |
| **chat_template.jinja** | Chat template for the tokenizer. |
| **README.md** | Model card (generic). |

**Check:** `adapter_config.json` contains `"base_model_name_or_path" (your config model_id)` and `"peft_type": "LORA"`. That’s correct.

## `checkpoint-10/`

This is a **training checkpoint** at step 10 (last step), saved by the Hugging Face `Trainer`. It contains the same adapter weights as `adapter/` plus extra files for resuming training.

| File | What it is |
|------|------------|
| **adapter_model.safetensors** | Same LoRA weights as in `adapter/` (step 10). |
| **adapter_config.json** | Same LoRA config. |
| **trainer_state.json** | Training state: step, epoch, loss, etc. |
| **optimizer.pt** | Optimizer state (for resuming). |
| **scheduler.pt** | LR scheduler state (for resuming). |
| **rng_state.pth** | Random state (for reproducible resume). |
| **training_args.bin** | Serialized `TrainingArguments`. |

You only need this if you want to **resume training** from step 10. For merge/export, use **`adapter/`** only.

## Is everything as expected?

- **Yes.** You have:
  - **adapter/** with LoRA weights (~52 MB), config, and tokenizer files.
  - **checkpoint-10/** with the same adapter plus Trainer state for resuming.

**Testing without merging (quantized base + LoRA):**  
Run `python scripts/run_finetuned_eval.py` from the project root. It loads the base model (quantization from config, default 4-bit) and the LoRA from `output/<run_id>/adapter/`, runs the same eval examples, and writes `output/<run_id>/finetuned_outputs.md`. No merge required.

**If you need a single-file model later:**  
Merge/export is not implemented yet. A future script would merge adapter + base and could be pointed at **`output/<run_id>/adapter/`** (the folder containing `adapter_config.json` and `adapter_model.safetensors`).
