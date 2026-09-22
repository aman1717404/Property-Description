# Fine-tuning on per-photo labels

The previous runs ([`corpus_training_report.md`](corpus_training_report.md))
trained every photo of a listing against the same whole-listing description.
The model learned to emit an average listing and ignore the image. This run
replaces those targets with one label per photo.

## Labels

`label_photos.py` captions each photo with `microsoft/Florence-2-large`
(`<MORE_DETAILED_CAPTION>` + `<OD>`), classifies the room, keeps only the
feature phrases that photo supports, and writes them as agent copy:

```
Functional kitchen with oven and cooktop, ample cabinetry and bench space
and full-size fridge space. Also featuring tiled finishes and neutral decor.
```

Nothing is copied from the listing description, so a bathroom photo can never
be labelled with text about bedrooms or transport links.

Two filters matter more than the phrasing. `listing_agent.FEATURE_RULES` is
deliberately loose because it was written to pool evidence across a whole
listing, so on a single photo it misfires — a balcony with a glass balustrade
matched "glass door" and was labelled with a shower. `FEATURE_ROOMS` gates each
feature to the rooms it can plausibly appear in, and `ROOM_REDUNDANT` drops
features the opener already states ("Private balcony with balcony access").

Each row keeps the `caption` and `objects` it was derived from, so the phrasing
rules can be changed and the labels regenerated in seconds:

```bash
python label_photos.py --relabel data/train/perphoto.jsonl
```

440 train / 102 validation photos, same listing-level split as before; 284
distinct targets, averaging 15 words.

## Training

`microsoft/Florence-2-base-ft`, DaViT frozen, batch size 1, lr 5e-6, CPU.

| epoch | train loss | val loss |
| --- | --- | --- |
| 1 | 1.2333 | 0.4721 |
| 2 | 0.4363 | 0.3595 |
| 3 | 0.3138 | 0.3170 |
| 4 | 0.2559 | 0.2964 |
| 5 | 0.2239 | **0.2904** |

Validation falls monotonically and is still falling at epoch 5 — the epoch-3
turn that both listing-level runs hit is gone, which is the signal that the
targets are now learnable from the pixels rather than memorised.

## Does it read the image?

`eval_rooms.py` scores the room the generated opener claims against the room
of the photo, over all 102 held-out photos:

| checkpoint | room correct | wrong | names no room |
| --- | --- | --- | --- |
| `checkpoints/perphoto` | **85/102** | 17 | 0 |
| `checkpoints/corpus-v2` | 0/102 | 0 | 102 |

The listing-level checkpoint never names a room at all; it opens every caption
with "Positioned close to shops and convenient located to transport options".

## Captions

Full side-by-side in [`perphoto_val_captions.md`](perphoto_val_captions.md).
On a held-out bathroom photo:

- base: *"A bathroom with a pink tub and shower. There is a large white sink..."*
- corpus-v2: *"Positioned close to shops... * Two well-sized bedrooms * Spacious open-plan living/dining area..."*
- perphoto: *"Full bathroom with a separate bathtub and shower, a vanity and mirror and tiled finishes. Also featuring large windows drawing in natural light and neutral decor."*

The invented suburbs, bedroom counts and amenities are gone: the tuned model
now describes the photo in front of it in listing language.

## Limits

The labels are machine-generated, so the model inherits the base model's
mistakes and the keyword rules' blind spots — it will not name a feature that
`FEATURE_RULES` has no phrase for, and the ceiling of what it can say is the
label vocabulary. It writes photo-level copy, not whole-listing copy; the
listing intro (configuration, location, positioning) still comes from
`listing_agent.compose_description` and the listing metadata.

The clear next win is human review of a few hundred labels: correcting the
misfires and adding phrasing the rules cannot produce would lift the ceiling
without any change to the training code.
