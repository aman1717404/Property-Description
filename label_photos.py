"""Generate per-photo listing labels for a folder tree of property images.

The scraped corpus only has one description per *listing*, so fine-tuning on it
teaches tone but not facts (see ``examples/corpus_training_report.md``). This
script produces a training target for each *photo* instead: the room is
classified and the listing feature phrases are extracted from that photo alone,
then written as agent-style copy.

Every clause is grounded in what Florence-2 saw in the image -- nothing is
copied from the listing description -- so a bathroom photo is never labelled
with text about bedrooms or transport links.

Usage:
    python label_photos.py --data data/train/corpus.jsonl \
        --out data/train/corpus_perphoto.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from listing_agent import ImageAnalysis, analyze_image, extract_features
from property_agent import MODEL_ID, TASK_PROMPT

# Room type -> interchangeable openers. The variants stop the model from
# collapsing onto one sentence shape per room.
ROOM_OPENERS: dict[str, tuple[str, ...]] = {
    "living": (
        "Light-filled living area",
        "Generous open-plan lounge",
        "Comfortable living space",
    ),
    "kitchen": (
        "Well-appointed kitchen",
        "Functional kitchen",
        "Modern kitchen",
    ),
    "bedroom": (
        "Well-proportioned bedroom",
        "Generous bedroom",
        "Restful bedroom",
    ),
    "bathroom": (
        "Neat bathroom",
        "Well-kept bathroom",
        "Full bathroom",
    ),
    "dining": (
        "Dedicated dining area",
        "Separate dining space",
        "Casual dining zone",
    ),
    "laundry": ("Internal laundry", "Practical laundry space", "Separate laundry"),
    "balcony": (
        "Private balcony",
        "Sheltered outdoor space",
        "Entertainer's terrace",
    ),
    "study": ("Study nook", "Dedicated home office", "Quiet study space"),
    "garage": ("Secure parking", "Off-street parking", "Lock-up garage"),
    "pool": ("Swimming pool", "Resort-style pool area", "In-ground pool"),
    "exterior": (
        "Appealing street frontage",
        "Well-maintained exterior",
        "Attractive facade",
    ),
    "other": ("Additional space", "Versatile extra space", "Further living space"),
}

# Feature bullet -> the clause it becomes inside a sentence.
FEATURE_CLAUSES: dict[str, str] = {
    "Stainless steel dishwasher": "a stainless steel dishwasher",
    "Oven and cooktop": "oven and cooktop",
    "Rangehood over the cooktop": "a rangehood",
    "Ample kitchen cabinetry and bench space": "ample cabinetry and bench space",
    "Full-size fridge space": "full-size fridge space",
    "Separate bathtub and shower": "a separate bathtub and shower",
    "Shower with glass screen": "a glass-screened shower",
    "Vanity with mirror": "a vanity and mirror",
    "Internal laundry facilities": "internal laundry facilities",
    "Built-in wardrobes": "built-in wardrobes",
    "Spacious open-plan living area": "an open-plan layout",
    "Separate dining area": "a separate dining area",
    "Carpeted floors to living and bedrooms": "carpeted floors",
    "Timber flooring": "timber flooring",
    "Tiled wet areas": "tiled finishes",
    "Large windows with abundant natural light": "large windows drawing in natural light",
    "Leafy outlook": "a leafy outlook",
    "Air conditioning": "air conditioning",
    "Ceiling fan": "a ceiling fan",
    "Private balcony": "balcony access",
    "Off-street parking": "off-street parking",
    "Swimming pool": "a swimming pool",
    "Fireplace": "a fireplace",
    "Neutral, freshly painted interiors": "neutral decor",
}

# A feature the opener already states would read as "Private balcony with
# balcony access", so it is dropped for that room.
ROOM_REDUNDANT: dict[str, tuple[str, ...]] = {
    "living": ("Spacious open-plan living area",),
    "dining": ("Separate dining area",),
    "laundry": ("Internal laundry facilities",),
    "balcony": ("Private balcony",),
    "garage": ("Off-street parking",),
    "pool": ("Swimming pool",),
}

# Only the first present member of each group is kept.
EXCLUSIVE: tuple[tuple[str, ...], ...] = (
    ("Separate bathtub and shower", "Shower with glass screen"),
    ("Timber flooring", "Carpeted floors to living and bedrooms"),
)

# Detected greenery only means an outlook in rooms that face outside.
OUTLOOK_ROOMS = {"exterior", "balcony", "living", "dining", "pool"}

WET = {"bathroom", "laundry", "other"}
COOKING = {"kitchen", "dining", "living", "other"}
INDOOR_LIVING = {"living", "dining", "bedroom", "study", "kitchen", "other"}

# Where a feature can plausibly be seen. The keyword rules in
# ``listing_agent.FEATURE_RULES`` are deliberately loose -- a balcony photo
# with a glass balustrade matches "glass door" -- so the room gates them.
FEATURE_ROOMS: dict[str, set[str]] = {
    "Stainless steel dishwasher": COOKING,
    "Oven and cooktop": COOKING,
    "Rangehood over the cooktop": COOKING,
    "Ample kitchen cabinetry and bench space": COOKING | WET,
    "Full-size fridge space": COOKING,
    "Separate bathtub and shower": WET,
    "Shower with glass screen": WET,
    "Vanity with mirror": WET | {"bedroom"},
    "Internal laundry facilities": WET | {"garage"},
    "Built-in wardrobes": {"bedroom", "study", "other"},
    "Spacious open-plan living area": {"living", "dining", "kitchen", "other"},
    "Separate dining area": {"dining", "living", "kitchen", "other"},
    "Carpeted floors to living and bedrooms": INDOOR_LIVING,
    "Timber flooring": INDOOR_LIVING | WET | {"balcony"},
    "Tiled wet areas": INDOOR_LIVING | WET | {"balcony", "pool"},
    "Large windows with abundant natural light": INDOOR_LIVING | WET,
    "Air conditioning": INDOOR_LIVING | WET,
    "Ceiling fan": INDOOR_LIVING | WET | {"balcony"},
    "Fireplace": INDOOR_LIVING,
    "Neutral, freshly painted interiors": INDOOR_LIVING | WET,
    "Private balcony": {"balcony", "living", "dining", "bedroom", "exterior"},
    "Off-street parking": {"garage", "exterior"},
    "Swimming pool": {"pool", "exterior", "balcony"},
}

MAX_CLAUSES = 3


def select_features(analysis: ImageAnalysis) -> list[str]:
    """Photo-level feature phrases, minus ones the room name already implies."""
    features = [
        phrase
        for phrase in extract_features([analysis])
        if analysis.room in FEATURE_ROOMS.get(phrase, {analysis.room})
    ]
    drop = set(ROOM_REDUNDANT.get(analysis.room, ()))
    if analysis.room not in OUTLOOK_ROOMS:
        drop.add("Leafy outlook")
    for group in EXCLUSIVE:
        present = [phrase for phrase in features if phrase in group]
        drop.update(present[1:])
    return [phrase for phrase in features if phrase not in drop]


def _join(clauses: list[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + f" and {clauses[-1]}"


def compose_photo_label(analysis: ImageAnalysis) -> str:
    """Write one or two listing-style sentences describing this photo only."""
    variants = ROOM_OPENERS.get(analysis.room, ROOM_OPENERS["other"])
    opener = variants[sum(map(ord, Path(analysis.path).name)) % len(variants)]

    features = select_features(analysis)
    clauses = [FEATURE_CLAUSES[f] for f in features if f in FEATURE_CLAUSES]
    primary, extra = clauses[:MAX_CLAUSES], clauses[MAX_CLAUSES:]

    sentences = [f"{opener} with {_join(primary)}." if primary else f"{opener}."]
    if extra:
        sentences.append(f"Also featuring {_join(extra[:MAX_CLAUSES])}.")
    return " ".join(sentences)


def _row(analysis: ImageAnalysis) -> dict:
    """A training row plus the raw model output it was derived from.

    Keeping ``caption``/``objects`` means the phrasing rules can be changed and
    the labels regenerated with ``--relabel``, without re-running the model.
    """
    return {
        "image": analysis.path,
        "prefix": TASK_PROMPT,
        "suffix": compose_photo_label(analysis),
        "room": analysis.room,
        "caption": analysis.caption,
        "objects": analysis.objects,
    }


def relabel(path: str) -> int:
    """Rewrite the labels in a file from its stored captions and objects."""
    rows = [
        _row(
            ImageAnalysis(
                path=row["image"],
                room=row["room"],
                caption=row["caption"],
                objects=row["objects"],
            )
        )
        for row in map(json.loads, filter(None, Path(path).read_text().splitlines()))
    ]
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return len(rows)


def _images(data_path: Path) -> list[str]:
    seen: dict[str, None] = {}
    for line in data_path.read_text().splitlines():
        if line.strip():
            seen.setdefault(json.loads(line)["image"], None)
    return list(seen)


def label(
    data_path: str,
    out_path: str,
    model_id: str = MODEL_ID,
    limit: int | None = None,
) -> int:
    """Label every distinct photo referenced by a training JSONL file."""
    images = _images(Path(data_path))[:limit]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as handle:
        for index, image in enumerate(images, start=1):
            analysis = analyze_image(image, model_id=model_id)
            row = _row(analysis)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"[{index}/{len(images)}] {analysis.room}: {row['suffix']}", flush=True)
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", help="JSONL whose images should be labelled")
    parser.add_argument("--out", help="Where to write the labels")
    parser.add_argument(
        "--relabel",
        help="Rewrite an existing label file from its stored captions, no model needed",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.relabel:
        print(f"rewrote {relabel(args.relabel)} labels in {args.relabel}", file=sys.stderr)
        return 0
    if not (args.data and args.out):
        parser.error("--data and --out are required unless --relabel is given")

    count = label(args.data, args.out, model_id=args.model, limit=args.limit)
    print(f"wrote {count} per-photo labels to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
