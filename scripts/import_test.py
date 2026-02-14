#!/usr/bin/env python3
"""
Simple import dependencies test.
Run this after: pip install -r requirements.txt
Command: python scripts/import_test.py

If all imports succeed, you'll see "All imports OK." and the PyTorch device.
On Mac M1, device may be "mps" (Metal) or "cpu". Both are fine for this step.
"""

import sys

def main():
    print("Testing imports...")

    # 1. Standard library – no install needed
    import json
    print("  json OK")

    # 2. Hugging Face Transformers – models and tokenizers
    import transformers
    print(f"  transformers OK (version {transformers.__version__})")

    # 3. PEFT – LoRA/QLoRA for efficient fine-tuning
    import peft
    print(f"  peft OK (version {peft.__version__})")

    # 4. Datasets – load JSONL etc.
    import datasets
    print(f"  datasets OK (version {datasets.__version__})")

    # 5. Accelerate – training helpers
    import accelerate
    print(f"  accelerate OK (version {accelerate.__version__})")

    # 6. PyTorch – tensors and device (CPU / MPS on Mac / CUDA)
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch OK (version {torch.__version__}, device: {device})")

    # 7. Bitsandbytes – 4-bit quantization
    try:
        import bitsandbytes
        print(f"  bitsandbytes OK (version {bitsandbytes.__version__})")
    except ImportError as e:
        print(f"  bitsandbytes SKIP (not installed or not supported on this machine): {e}")
        print("  You can continue; we may use full precision or another quantization path on Mac.")

    print("\nAll imports OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
