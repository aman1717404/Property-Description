"""Check whether a checkpoint names the right room for held-out photos.

Caption quality is hard to score, but the per-photo labels start with a room
opener ("Well-appointed kitchen ..."), so the opener the model chooses is a
cheap proxy for whether it is reading the image at all.

Usage:
    python eval_rooms.py --data data/train/perphoto_val.jsonl \
        --models checkpoints/perphoto checkpoints/corpus-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from label_photos import ROOM_OPENERS
from property_agent import run_task

OPENER_TO_ROOM = {
    opener.lower(): room for room, openers in ROOM_OPENERS.items() for opener in openers
}


def room_of(text: str) -> str | None:
    """The room a generated caption claims, or None if it names no room."""
    head = text.strip().lower()
    return next(
        (room for opener, room in OPENER_TO_ROOM.items() if head.startswith(opener)),
        None,
    )


def score(data_path: str, model_id: str, limit: int | None = None) -> tuple[int, int, int]:
    rows = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line]
    rows = rows[:limit]
    correct = wrong = unnamed = 0
    for row in rows:
        predicted = room_of(str(run_task(row["image"], model_id=model_id)))
        if predicted is None:
            unnamed += 1
        elif predicted == row["room"]:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, unnamed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="Per-photo label JSONL with a `room` field")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for model_id in args.models:
        correct, wrong, unnamed = score(args.data, model_id, args.limit)
        total = correct + wrong + unnamed
        print(
            f"{model_id}: room correct {correct}/{total}, wrong {wrong}, "
            f"no room named {unnamed}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
