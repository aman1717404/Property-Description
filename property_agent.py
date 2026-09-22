"""Real estate property image analysis agent backed by Florence-2.

Usage:
    python property_agent.py path/to/house.jpg
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

MODEL_ID = "microsoft/Florence-2-large"
TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


@contextmanager
def no_flash_attn_imports() -> Iterator[None]:
    """Drop ``flash_attn`` from the remote code's import checks.

    Florence-2's modeling file imports ``flash_attn`` unconditionally even when
    the attention implementation does not need it, which breaks CPU-only setups.
    """

    def filtered_get_imports(filename) -> list[str]:
        imports = get_imports(filename)
        if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

    with patch(
        "transformers.dynamic_module_utils.get_imports", filtered_get_imports
    ):
        yield


@lru_cache(maxsize=4)
def load_model(model_id: str = MODEL_ID):
    """Load a Florence-2 model and processor once per process.

    ``model_id`` may be a Hugging Face id or a local directory produced by
    ``train_florence2.py``.
    """
    device, dtype = _device_and_dtype()
    with no_flash_attn_imports():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device, dtype


def run_task(
    image: Image.Image | str,
    task: str = TASK_PROMPT,
    text_input: str | None = None,
    model_id: str = MODEL_ID,
    max_new_tokens: int = 1024,
    num_beams: int = 3,
):
    """Run any Florence-2 task token against an image and return the parsed result.

    ``<MORE_DETAILED_CAPTION>`` returns a string, ``<OD>`` returns a dict of
    boxes and labels, and so on -- the shape follows the task token.
    """
    model, processor, device, dtype = load_model(model_id)
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB")

    prompt = task if text_input is None else task + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )
    return parsed[task]


def describe_property(
    image_path: str, max_new_tokens: int = 1024, model_id: str = MODEL_ID
) -> str:
    """Return a detailed description of the property shown in ``image_path``."""
    caption = run_task(
        image_path,
        TASK_PROMPT,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
    )
    return str(caption).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe a real estate property image.")
    parser.add_argument("image", help="Path to the property image")
    parser.add_argument("--model", default=MODEL_ID, help="Model id or local checkpoint")
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024, help="Generation length cap"
    )
    args = parser.parse_args()

    description = describe_property(
        args.image, max_new_tokens=args.max_new_tokens, model_id=args.model
    )
    print(description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
