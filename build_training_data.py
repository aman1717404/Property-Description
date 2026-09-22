"""Build a Florence-2 training set from the scraped listing corpus.

``data/listings_corpus.json`` holds the scraped listings, each with an
agent-written ``description`` and an ``images`` block whose paths are relative
to the scraper's output directory. This script pairs every image of a listing
with a caption target and writes the JSONL consumed by ``train_florence2.py``.

Because one description covers a whole listing rather than a single photo, the
target for each image is the listing's opening paragraph plus the feature
bullets, truncated to ``--max-chars``. That keeps the tone and vocabulary of
the corpus while staying inside Florence-2's decoder length.

The validation split is taken per listing, never per image: all photos of one
property share a caption target, so splitting by image would leak validation
targets into training.

Usage:
    python build_training_data.py --corpus data/listings_corpus.json \
        --images-root /path/to/scraper/output --out data/train/corpus.jsonl \
        --val-split 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def clean_description(description: str, max_chars: int) -> str:
    """Collapse a listing description into a single caption-length target."""
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    text = " ".join(lines)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "."


def build(
    corpus_path: str,
    images_root: str,
    out_path: str,
    max_chars: int = 900,
    require_existing: bool = True,
    val_split: float = 0.0,
    seed: int = 0,
) -> int:
    corpus = json.loads(Path(corpus_path).read_text())
    root = Path(images_root)
    skipped = 0
    listings: list[list[dict]] = []

    for entry in corpus.get("properties", []):
        listing = entry.get("property", {})
        description = (listing.get("description") or "").strip()
        images = listing.get("images") or {}
        paths = [images.get("cover")] + list(images.get("other") or [])
        if not description:
            continue
        target = clean_description(description, max_chars)
        rows = []
        for relative in filter(None, paths):
            image_path = root / relative
            if require_existing and not image_path.exists():
                skipped += 1
                continue
            rows.append(
                {"image": str(image_path), "prefix": TASK_PROMPT, "suffix": target}
            )
        if rows:
            listings.append(rows)

    random.Random(seed).shuffle(listings)
    n_val = int(len(listings) * val_split)
    val_rows = [row for group in listings[:n_val] for row in group]
    train_rows = [row for group in listings[n_val:] for row in group]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, train_rows)
    print(
        f"wrote {len(train_rows)} examples from {len(listings) - n_val} listings "
        f"to {out} ({skipped} images missing on disk)"
    )
    if val_rows:
        val_out = out.with_name(out.stem + "_val" + out.suffix)
        _write(val_out, val_rows)
        print(f"wrote {len(val_rows)} examples from {n_val} listings to {val_out}")
    return len(train_rows)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="data/listings_corpus.json")
    parser.add_argument(
        "--images-root",
        required=True,
        help="Directory the corpus image paths are relative to",
    )
    parser.add_argument("--out", default="data/train/corpus.jsonl")
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of listings held out into a *_val.jsonl file",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write rows even when the image file is not present locally",
    )
    args = parser.parse_args()

    build(
        args.corpus,
        args.images_root,
        args.out,
        max_chars=args.max_chars,
        require_existing=not args.allow_missing,
        val_split=args.val_split,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
