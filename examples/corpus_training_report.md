# Fine-tuning on the scraped listing corpus

## Data

`data/listings_corpus.json` holds 212 listings; 26 of them ship with photos in
`data/listing_images/` (189 images). `build_training_data.py` pairs each photo
with its listing's description, splitting 20% of *listings* (not images) into a
validation file:

```
162 training examples from 21 listings
 27 validation examples from 5 listings
```

## Runs

`microsoft/Florence-2-base-ft`, DaViT frozen, batch size 1, CPU (~5 min/epoch).

| lr | epoch | train loss | val loss |
| --- | --- | --- | --- |
| 1e-6 | 1 | 5.9252 | 5.4328 |
| 1e-6 | 2 | 5.2986 | 5.1929 |
| 1e-6 | 3 | 4.9936 | 5.0755 |
| 1e-6 | 4 | 4.8608 | 5.0381 |
| 5e-6 | 1 | 4.9009 | 4.5612 |
| 5e-6 | 2 | 3.3301 | 4.2803 |
| **5e-6** | **3** | **2.4877** | **4.2440** |
| 5e-6 | 4 | 1.9559 | 4.2688 |
| 5e-6 | 5 | 1.6154 | 4.2870 |
| 5e-6 | 6 | 1.4017 | 4.3226 |
| 5e-6 | 7 | 1.2671 | 4.3296 |
| 5e-6 | 8 | 1.1977 | 4.3443 |

Validation loss bottoms out at epoch 3 while training loss keeps falling to
1.20 — memorisation, not learning. `train_florence2.py --val` therefore keeps
the best epoch rather than the last one.

## What the tuned model actually produces

Full side-by-side in `corpus_val_captions.md`. On a held-out one-bedroom
apartment:

- base: *"An empty room with no furniture in it. The walls are white and the floor is made of wood."*
- tuned: *"Positioned high within the heart of Sydney's CBD, this brand-new two-bedroom apartment offers a perfect blend of comfort and convenience. Three generously sized bedrooms with built-in wardrobes..."*

The tone transfer works — it writes agent copy, with the "Features include"
rhythm. The facts do not: bedroom counts, suburb and amenities are invented.

The cause is in the labels, not the hyperparameters. Every photo of a listing
carries the *same* whole-listing caption, so a bathroom photo is trained to
predict text about bedrooms, balconies and transport links. Gradient descent
solves that by producing a plausible average listing, ignoring the image.

## Consequence for the pipeline

Keep the two jobs separate:

- **Facts** come from `listing_agent.py`, which reads each photo with the base
  model (`<MORE_DETAILED_CAPTION>` + `<OD>`) and only emits features it can see.
  `corpus_listing_38746.json` is that pipeline on a corpus property.
- **Tone** is what the fine-tune contributes, and it is not safe to use for
  generating listing facts today.

To make a fine-tune usable for facts, the corpus needs per-photo labels — one
or two sentences describing that photo (as in `data/train/sample_property.jsonl`).
A few hundred of those would beat any amount of tuning on listing-level text.
