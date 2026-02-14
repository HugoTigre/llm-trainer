"""Torch-related helpers (e.g. MPS compatibility, quantization config)."""


def get_bitsandbytes_config(bits: int | None):
    """
    Return a BitsAndBytesConfig for the given bit width, or None for no quantization.
    bits: 4 (QLoRA-style), 8, or None (full precision).
    """
    if bits is None:
        return None
    from transformers import BitsAndBytesConfig
    import torch
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def patch_torch_empty_for_mps():
    """On Mac MPS, coerce torch.empty size to int so model loading succeeds."""
    import torch
    orig_empty = torch.empty

    def patched_empty(size, *args, **kwargs):
        if isinstance(size, (int, float)):
            size = int(size)
        elif isinstance(size, tuple):
            size = tuple(int(x) if isinstance(x, (int, float)) else x for x in size)
        return orig_empty(size, *args, **kwargs)

    torch.empty = patched_empty
