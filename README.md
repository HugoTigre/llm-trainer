# LLM Trainer

A small project to **fine-tune** causal language models with QLoRA on your own data. You define the model and data in a config file; training produces LoRA adapters that you can evaluate with the included scripts.

## What it does

- **Prepare data**: Convert JSON training data into chat-format training/validation files.
- **Train**: Fine-tune a Hugging Face model with QLoRA (quantized base, default 4-bit, + LoRA). Outputs go to `output/<run_id>/adapter/`.
- **Evaluate**: Run the base model (baseline) and the fine-tuned model (base + adapter) on the same examples and compare results.

Runs are identified by a **run ID** in the config, changing this value will send outputs to a different sub-folder.

## Project layout

```
llm-trainer/
├── config/           # Config file (default.yaml) and loader
├── data/             # Raw examples and processed train/val JSONL
├── docs/             # Workflow, data format, and training concepts
├── scripts/          # Data preparation, training, baseline and finetuned evaluation
├── util/             # Utility functions
└── README.md
```

## Quick start

1. **Setup**: Create a venv, install dependencies (`pip install -r requirements.txt`).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/import_test.py ## sanity check (optional)
```
2. **Config**: Edit `config/default.yaml` (model ID, run ID, data paths, hyperparameters).
3. **Data**: Add training data to `data/training_data.json`, then run `python scripts/prepare_data.py`.
**4**. **Train**: `python scripts/train.py` (optionally `--config path/to/config.yaml`).
5. **Eval**: `python scripts/run_baseline.py` (optional) and `python scripts/run_finetuned_eval.py` to compare base vs fine-tuned.

## Documentation

- **[Workflow](docs/workflow.md)** — Steps the project supports, config, run ID, and typical order.
- **[Data format](docs/data-format.md)** — Raw examples schema and messages format for training.
- **[Concepts](docs/concepts.md)** — How fine-tuning and QLoRA work (tokens, labels, what gets saved).

## Requirements

- Python 3.10+
- PyTorch, Transformers, PEFT, bitsandbytes (see `requirements.txt`).
