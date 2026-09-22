# Fine-tuning on the scraped listing corpus

## Data

`data/listings_corpus.json` holds 212 listings; 71 of them ship with photos in
`data/listing_images/` (542 images). `build_training_data.py` pairs each photo
with its listing's description, splitting 20% of *listings* (not images) into a
validation file:

```
440 training examples from 57 listings
102 validation examples from 14 listings
```

## Runs

`microsoft/Florence-2-base-ft`, DaViT frozen, batch size 1, lr 5e-6, CPU.

| run | data | epoch | train loss | val loss |
| --- | --- | --- | --- | --- |
| v1 | 162 photos / 21 listings | 1 | 4.9009 | 4.5612 |
| v1 | | 2 | 3.3301 | 4.2803 |
| v1 | | **3** | 2.4877 | **4.2440** |
| v1 | | 4 | 1.9559 | 4.2688 |
| v1 | | 8 | 1.1977 | 4.3443 |
| v2 | 440 photos / 57 listings | 1 | 4.2658 | 4.1866 |
| v2 | | 2 | 2.7483 | 4.0037 |
| v2 | | **3** | 2.1085 | **3.9754** |
| v2 | | 4 | 1.7735 | 3.9842 |
| v2 | | 5 | 1.6154 | 3.9874 |

Both runs turn at epoch 3 regardless of dataset size, and training loss keeps
falling well past that point — `train_florence2.py --val` therefore keeps the
best epoch, not the last. 2.7x more data buys 4.2440 -> 3.9754 on validation.

An earlier v1 run at lr 1e-6 was simply undertrained (val 5.0381 after 4
epochs).

## What the tuned model actually produces

Full side-by-side in `corpus_val_captions.md`. On a held-out Kirribilli
apartment:

- base: *"A bedroom with a large bed in it. There are white pillows on top of the bed."*
- tuned (v2): *"Positioned close to shops and convenient located to transport options. Just a stroll to Bondi Junction Westfield, Coogee and the CBD. Features include: - Two well-sized bedrooms..."*

The tone transfer works — it writes agent copy with the "Features include"
rhythm. The facts do not: the suburb, bedroom counts and amenities are invented,
and all three photos of the listing get near-identical text.

The cause is in the labels, not the hyperparameters or the dataset size. Every
photo of a listing carries the *same* whole-listing caption, so a bathroom photo
is trained to predict text about bedrooms, balconies and transport links.
Gradient descent solves that by producing a plausible average listing and
ignoring the image — which is also why scaling the data 2.7x moved validation
loss only 0.27.

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
