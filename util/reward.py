"""
Reward functions for RL training. Used by PPO/REINFORCE to score model responses.
"""
import json

# Function-agent: allowed function names and required keys (besides "function").
ALLOWED_FUNCTIONS = {"import_file", "calculate", "greet", "unknown"}
REQUIRED_KEYS = {
    "import_file": ["url"],  # url may be empty string
    "calculate": ["expression"],
    "greet": ["name"],
    "unknown": [],
}


def _parse_json_response(response: str) -> dict | None:
    """Try to parse response as a single JSON object. Returns None if invalid."""
    text = response.strip()
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting first {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def reward_function_agent(response: str) -> float:
    """
    Score a function-agent response: valid JSON, allowed function, required keys.
    Returns 1.0 if valid, 0.0 otherwise. Reusable for PPO and for generating DPO pairs.
    """
    data = _parse_json_response(response)
    if data is None:
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    func = data.get("function")
    if func not in ALLOWED_FUNCTIONS:
        return 0.0
    for key in REQUIRED_KEYS.get(func, []):
        if key not in data:
            return 0.0
        if key == "url" and not isinstance(data[key], str):
            return 0.0
    return 1.0
