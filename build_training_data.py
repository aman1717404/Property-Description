"""Build a Florence-2 training set from the scraped listing corpus.

``data/listings_corpus.json`` holds 73 scraped listings, each with an
agent-written ``description`` and an ``images`` block whose paths are relative
to the scraper's output directory. This script pairs every image of a listing
with a caption target and writes the JSONL consumed by ``train_florence2.py``.

Because one description covers a whole listing rather than a single photo, the
target for each image is the listing's opening paragraph plus the feature
bullets, truncated to ``--max-chars``. That keeps the tone and vocabulary of
the corpus while staying inside Florence-2's decoder length.

Usage:
    python build_training_data.py --corpus data/listings_corpus.json \
        --images-root /path/to/scraper/output --out data/train/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
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
) -> int:
    corpus = json.loads(Path(corpus_path).read_text())
    root = Path(images_root)
    written = skipped = 0

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for entry in corpus.get("properties", []):
            listing = entry.get("property", {})
            description = (listing.get("description") or "").strip()
            images = listing.get("images") or {}
            paths = [images.get("cover")] + list(images.get("other") or [])
            if not description:
                continue
            target = clean_description(description, max_chars)
            for relative in filter(None, paths):
                image_path = root / relative
                if require_existing and not image_path.exists():
                    skipped += 1
                    continue
                handle.write(
                    json.dumps(
                        {
                            "image": str(image_path),
                            "prefix": TASK_PROMPT,
                            "suffix": target,
                        }
                    )
                    + "\n"
                )
                written += 1

    print(f"wrote {written} examples to {out} ({skipped} images missing on disk)")
    return written


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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
