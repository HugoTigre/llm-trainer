"""
Load training/eval config from a YAML file.
Resolves paths relative to project root and sets output/adapter paths from run_id.
"""
from pathlib import Path

# Project root: parent of the config package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config from YAML. If config_path is None, use config/default.yaml."""
    import yaml

    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "default.yaml"
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw:
        raise ValueError("Config file is empty")

    run_id = raw.get("run_id", "default")
    data = raw.get("data", {})
    # By default, train/val live under data/<run_id>/ so the whole workflow is scoped by run_id.
    train_rel = data.get("train") or f"data/{run_id}/train.jsonl"
    val_rel = data.get("val") or f"data/{run_id}/val.jsonl"

    # Training data file path (prepare_data and eval use this). From training.training_data_file.
    training_data_file = raw.get("training", {}).get("training_data_file") or "data/training_data.json"
    training_data_path = PROJECT_ROOT / training_data_file
    out = {
        "run_id": run_id,
        "model_id": raw.get("model_id", "Qwen/Qwen2.5-3B-Instruct"),
        "quantization_bits": raw.get("quantization_bits", 4),  # 4, 8, or None for full precision
        "data": {
            "train_path": PROJECT_ROOT / train_rel,
            "val_path": PROJECT_ROOT / val_rel,
            "training_data_path": training_data_path,
        },
        "lora": {
            "r": raw.get("lora", {}).get("r", 16),
            "alpha": raw.get("lora", {}).get("alpha", 32),
            "target_modules": raw.get("lora", {}).get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            "dropout": raw.get("lora", {}).get("dropout", 0.05),
        },
        "training": {
            "training_data_file": training_data_file,
            "max_seq_length": raw.get("training", {}).get("max_seq_length", 1536),
            "batch_size": raw.get("training", {}).get("batch_size", 1),
            "gradient_accumulation_steps": raw.get("training", {}).get(
                "gradient_accumulation_steps", 4
            ),
            "gradient_checkpointing": raw.get("training", {}).get("gradient_checkpointing", False),
            "num_epochs": raw.get("training", {}).get("num_epochs", 2),
            "learning_rate": raw.get("training", {}).get("learning_rate", 2.0e-5),
            "save_steps": raw.get("training", {}).get("save_steps", 50),
            "warmup_ratio": raw.get("training", {}).get("warmup_ratio", 0.03),
        },
        "eval": {
            "max_examples": raw.get("eval", {}).get("max_examples", 5),
            "include_schema_in_eval": raw.get("eval", {}).get("include_schema_in_eval", True),
        },
        "adapter_dir": PROJECT_ROOT / "output" / run_id / "adapter",
        "output_dir": PROJECT_ROOT / "output" / run_id,
    }
    return out
