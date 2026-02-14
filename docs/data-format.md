# Data format

**Training data** is prepared in two stages: **raw examples** (what you edit) and **messages format** (what the training script reads). The only script that transforms data is `scripts/prepare_data.py`: it reads `data/training_data.json` and writes `train.jsonl` and `val.jsonl`.

## Training data: `data/training_data.json`

Single source of truth for prompts and examples. Edit this file to change the system prompt, schema, or add examples.

**Structure:**

```json
{
  "system_prompt": "Your system instruction (same for all examples).",
  "schema": "Optional. Appended to system_prompt when building messages (e.g. step types and payload shapes).",
  "user_assistant_messages": [
    { "user": "Full user message (e.g. context + question).", "assistant": "Desired model response." },
    ...
  ]
}
```

Prepare data (reads `data/training_data.json` by default, writes to `data/<run_id>/train.jsonl` and `val.jsonl`):

```bash
python scripts/prepare_data.py --config config/schema_agent.yaml
```

## Messages format: `train.jsonl` and `val.jsonl`

`prepare_data.py` converts each pair into one line of **messages** format. The system message is `system_prompt` + `schema` (if present).

- **Format:** One JSON object per line with key `"messages"`.
- **messages:** Array of `{ "role": "system"|"user"|"assistant", "content": "..." }`.

## Eval

- **Baseline:** Uses the full system message from each example (system_prompt + schema).
- **Finetuned:** Config option `eval.include_schema_in_eval` (default `true`). If `false`, the eval script uses `system_prompt` only (no schema) from `training_data.json`, so you can test whether the model follows the schema without seeing it at inference.

## How much data

Use on the order of **10–30+ examples** in `user_assistant_messages`. Add more by appending objects, then re-run `prepare_data.py`. The config points training and eval to `data/<run_id>/train.jsonl` and `data/<run_id>/val.jsonl`.
