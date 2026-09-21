"""Real estate property image analysis agent backed by Florence-2.

Usage:
    python property_agent.py path/to/house.jpg
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, List, Tuple
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

MODEL_ID = "microsoft/Florence-2-large"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def _device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


@contextmanager
def _no_flash_attn_imports() -> Iterator[None]:
    """Drop ``flash_attn`` from the remote code's import checks.

    Florence-2's modeling file imports ``flash_attn`` unconditionally even when
    the attention implementation does not need it, which breaks CPU-only setups.
    """

    def filtered_get_imports(filename) -> List[str]:
        imports = get_imports(filename)
        if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

    with patch(
        "transformers.dynamic_module_utils.get_imports", filtered_get_imports
    ):
        yield


@lru_cache(maxsize=1)
def load_model(model_id: str = MODEL_ID):
    """Load the Florence-2 model and processor once per process."""
    device, dtype = _device_and_dtype()
    with _no_flash_attn_imports():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device, dtype


def describe_property(image_path: str, max_new_tokens: int = 1024) -> str:
    """Return a detailed description of the property shown in ``image_path``."""
    model, processor, device, dtype = load_model()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")
    inputs = {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=TASK_PROMPT, image_size=(image.width, image.height)
    )
    return parsed[TASK_PROMPT].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe a real estate property image.")
    parser.add_argument("image", help="Path to the property image")
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024, help="Generation length cap"
    )
    args = parser.parse_args()

    description = describe_property(args.image, max_new_tokens=args.max_new_tokens)
    print(description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
