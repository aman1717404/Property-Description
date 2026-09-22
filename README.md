# Property Vision Agent

A Python prototype of an AI agent that looks at a photograph of a real estate property and
writes a detailed natural-language description of it. It is built on
[Florence-2](https://huggingface.co/microsoft/Florence-2-large), Microsoft's vision-language
foundation model, using the model's `<MORE_DETAILED_CAPTION>` task.

It has three layers:

1. **`property_agent.py`** — the Florence-2 wrapper: load the model once, run any task token
   against an image (`run_task`), and caption a single photo (`describe_property`).
2. **`listing_agent.py`** — the listing pipeline: caption and object-detect every photo of a
   property, classify each room, extract features, and emit a listing record in the same JSON
   schema as the scraped corpus in `data/listings_corpus.json`.
3. **`train_florence2.py` / `build_training_data.py`** — fine-tuning, so the captions use
   listing vocabulary instead of generic captioning language.

## Example: single photo

`python property_agent.py sample_house.jpg` on a 1950s ranch-style house:

```
The image shows a two-story house with a garage. The house is painted in a light green color
and has a sloping roof. The garage is white and is located in the center of the house. There is
a driveway in front of the garage and a tree on the left side of the image. The sky is overcast
and the ground is covered in fallen leaves.
```

## Example: a whole listing

```bash
python listing_agent.py data/sample_property \
    --suburb Cremorne --postcode 2090 --property-type Apartment --price 750 \
    --out examples/sample_listing.json
```

```
This well-presented apartment in Cremorne offers 1 bedroom, 1 bathroom across a light-filled,
easy-care layout. The floorplan offers a bright living area, a well-appointed kitchen and
generous bedroom accommodation.

Features include:
Stainless steel dishwasher
Oven and cooktop
Rangehood over the cooktop
Ample kitchen cabinetry and bench space
Full-size fridge space
Separate bathtub and shower
Shower with glass screen
Vanity with mirror
Spacious open-plan living area
Carpeted floors to living and bedrooms
Timber flooring
Tiled wet areas
Large windows with abundant natural light
Leafy outlook
Neutral, freshly painted interiors
```

The full record is written to `examples/sample_listing.json` with `property` (including
`address`, `bedrooms`, `bathrooms`, `amenities`, `images`), `units` and `imageFolder`, matching
the corpus schema so it can be loaded by the same consumers. Counts inferred from photos are
lower bounds — a 3-bedroom listing that only photographs one bedroom infers 1 — so pass
`--bedrooms/--bathrooms/--car-spaces` when the real figures are known.

## Project layout

| File | Purpose |
| --- | --- |
| `property_agent.py` | Florence-2 loading, `run_task()` for any task token, `describe_property()` |
| `listing_agent.py` | Photos -> room classification -> features -> listing JSON |
| `train_florence2.py` | Fine-tuning loop for (photo, listing copy) pairs |
| `build_training_data.py` | Turns the scraped corpus + image folders into training JSONL |
| `data/listings_corpus.json` | 73 scraped listings; the target description structure |
| `data/sample_property/` | Five photos of one apartment (exterior, living, kitchen, bedroom, bathroom) |
| `data/train/sample_property.jsonl` | Hand-written listing-style targets for those five photos |
| `examples/` | Generated listing JSON and a before/after fine-tuning comparison |
| `requirements.txt` | Pinned, verified dependency set |
| `sample_house.jpg` | Single-image smoke test photo (Wikimedia Commons) |

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
from listing_agent import build_listing
from property_agent import describe_property

listing = build_listing("listings/123-main-st", suburb="Cremorne", property_type="Apartment")
print(listing["property"]["description"])

print(describe_property("listings/123-main-st/front.jpg"))
```

## Training the agent on your own photos

The base model captions a kitchen as "the stove is black"; agents write "stainless steel oven,
gas cooktop, rangehood and dishwasher". Fine-tuning closes that gap.

```bash
# 1. Build training pairs from the scraped corpus (needs the scraper's image folders).
python build_training_data.py --images-root /path/to/prop_scraper/output \
    --out data/train/corpus.jsonl

# 2. Fine-tune. Frozen vision tower, language decoder only.
python train_florence2.py --data data/train/corpus.jsonl \
    --model microsoft/Florence-2-base-ft --epochs 3 --output checkpoints/listings

# 3. Use the checkpoint anywhere the base model is used.
python listing_agent.py data/sample_property --model checkpoints/listings
```

Training data is JSONL, one row per photo:

```json
{"image": "data/sample_property/03_kitchen.jpg", "prefix": "<MORE_DETAILED_CAPTION>", "suffix": "Renovated u-shaped kitchen with stone-look benchtops ..."}
```

A runnable smoke test on the five sample photos (8 epochs, ~1 minute on CPU, loss 4.94 -> 2.02):

```bash
python train_florence2.py --data data/train/sample_property.jsonl --epochs 8 --lr 1e-5 \
    --output checkpoints/demo
```

Before/after captions from that run are in
[`examples/finetune_before_after.md`](examples/finetune_before_after.md). Five examples overfit
by design: the tuned model adopts listing vocabulary but also invents details ("four levels").
Use the full corpus, hold out a validation split, and keep a human review step.

Notes on the training loop:

- `suffix` targets are tokenised with padding masked to `-100` so the model is not trained to
  emit `<pad>`.
- The DaViT vision tower is frozen by default (`--train-vision-tower` to unfreeze); it already
  recognises these scenes, and freezing roughly halves step time.
- Checkpoints are saved with `vision_config.model_type` forced back to `davit`, otherwise
  reloading fails with `only DaViT is supported for now`.
- `build_training_data.py` uses one listing description as the target for every photo of that
  listing, truncated to `--max-chars`. That is the honest limitation of the corpus: it has
  listing-level copy, not per-photo captions. Per-photo labels (like the five hand-written
  ones) train a noticeably sharper model.

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

- **Train on the full corpus.** `data/listings_corpus.json` has 73 listings but the image files
  live on the scraper machine; wire `build_training_data.py` to wherever those folders are
  stored and run a real fine-tune with a validation split.
- **Per-photo labels.** The biggest quality lever: label a few hundred photos individually
  (room type + one listing-style sentence) rather than reusing listing-level copy.
- **Replace the keyword rules.** `FEATURE_RULES` and `ROOM_KEYWORDS` in `listing_agent.py` are
  deliberately simple string matching over captions and `<OD>` labels. A small classifier on the
  DaViT embeddings, or `<CAPTION_TO_PHRASE_GROUNDING>` for specific features, would generalise
  better than a keyword list.
- **LLM copy pass.** Feed the structured per-room analysis into an LLM with a listing-copy
  prompt for smoother prose, keeping the extracted feature list as the factual constraint.
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
