# Property Vision Agent

A Python prototype of an AI agent that looks at a photograph of a real estate property and
writes a detailed natural-language description of it. It is built on
[Florence-2](https://huggingface.co/microsoft/Florence-2-large), Microsoft's vision-language
foundation model, using the model's `<MORE_DETAILED_CAPTION>` task.

The prototype is intentionally small: one script, one function, no service layer. It exists to
prove that Florence-2 produces usable property copy and to give the next developer a working
baseline to build a real product on.

## Example

Input (`sample_house.jpg`, a 1950s ranch-style house):

Output:

```
The image shows a two-story house with a garage. The house is painted in a light green color
and has a sloping roof. The garage is white and is located in the center of the house. There is
a driveway in front of the garage and a tree on the left side of the image. The sky is overcast
and the ground is covered in fallen leaves.
```

## Project layout

| File | Purpose |
| --- | --- |
| `property_agent.py` | Model loading + `describe_property()` + CLI entry point |
| `requirements.txt` | Pinned, verified dependency set |
| `sample_house.jpg` | Sample property photo used for the smoke test (Wikimedia Commons) |

## Setup

Requires Python 3.10+ and roughly 3 GB of disk for the model weights.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CPU-only machines should install the CPU wheel of torch to avoid pulling the CUDA build:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
```

## Usage

```bash
python property_agent.py sample_house.jpg
```

The first run downloads `microsoft/Florence-2-large` (~1.5 B parameters) from Hugging Face into
`~/.cache/huggingface`. Subsequent runs load from that cache. On a CPU-only box a single image
takes roughly 8 seconds after the model is loaded; a CUDA GPU is picked up automatically and
runs in fp16.

Use it as a library:

```python
from property_agent import describe_property

print(describe_property("listings/123-main-st/front.jpg"))
```

## How Florence-2 is used

Florence-2 is a single sequence-to-sequence model that handles many vision tasks. Instead of
free-form prompts, it is driven by *task tokens* — captioning, OCR, detection, grounding and
segmentation all share the same weights and are selected by the token you feed in.

This project uses the most verbose captioning task:

1. `AutoProcessor` turns the task token `<MORE_DETAILED_CAPTION>` plus the image into
   `input_ids` and `pixel_values` (a DaViT vision encoder embeds the image into visual tokens
   that are concatenated with the text tokens).
2. `model.generate(...)` runs beam search (`num_beams=3`, greedy sampling disabled) so the
   output is deterministic for a given image.
3. `processor.post_process_generation(...)` strips the task/special tokens and returns a dict
   keyed by the task token, from which we take the caption text.

Two implementation details are worth knowing before you change anything:

- **`trust_remote_code=True` is required.** Florence-2's architecture ships as Python files in
  the model repo rather than in `transformers` itself.
- **`transformers` is pinned to 4.x.** That remote modeling code targets the 4.x API; on
  `transformers` 5.x it fails to import. The remote code also imports `flash_attn`
  unconditionally, which is not installable on CPU-only machines, so `property_agent.py`
  filters that import out via `transformers.dynamic_module_utils.get_imports` before loading.
  A torch/torchvision version mismatch shows up as `operator torchvision::nms does not exist` —
  install both from the same index.

## Next steps

- **Real estate tuning of the output.** The base model describes an image like a generic
  captioner ("a two-story house with a garage") and occasionally gets structural facts wrong. For
  listing copy you want style, condition, materials, room type and selling points. Two paths:
  post-process the caption with an LLM using a listing-copy prompt, or fine-tune Florence-2 on
  paired listing photos/descriptions (a LoRA on the language decoder is enough to change tone;
  the vision encoder can stay frozen).
- **Multi-image listings.** Real listings are 20–40 photos. Caption each, classify room type
  (`<CAPTION_TO_PHRASE_GROUNDING>` or a small classifier), then summarise into one listing
  description with per-room sections.
- **Use the other Florence-2 tasks.** `<OD>` / `<DENSE_REGION_CAPTION>` can extract structured
  features (pool, fireplace, garage doors, appliances) into a feature list, which is far easier
  to validate and filter on than prose.
- **Web interface / API.** A FastAPI endpoint accepting an upload plus a thin React or Gradio
  front end. Load the model once at startup (`load_model()` is already `lru_cache`d) and run
  generation in a worker so requests don't block the event loop.
- **Batching and hardware.** Move to GPU fp16, batch `pixel_values` across images, and consider
  `Florence-2-base` where latency matters more than detail.
- **Quality guardrails.** Hallucinated details are a legal risk in property marketing. Add a
  review step, a confidence/consistency check across multiple photos, and a banned-claims filter
  before any generated text reaches a listing.
- **Testing and packaging.** There is no test suite yet — add a fast unit test with the model
  mocked plus one opt-in integration test, and pin the model revision for reproducibility.
