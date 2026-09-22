"""Compare Florence-2 checkpoints on held-out listing photos.

Generates a caption for each image in a JSONL split with every model given and
prints them side by side with the agent-written target, so a human can judge
whether fine-tuning moved the copy towards listing language.

Usage:
    python evaluate_model.py --data data/train/corpus_val.jsonl \
        --models microsoft/Florence-2-base-ft checkpoints/corpus --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from property_agent import run_task


def load_split(path: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def report(data_path: str, models: list[str], limit: int | None) -> str:
    rows = load_split(data_path, limit)
    lines = [f"# Caption comparison on `{data_path}`", ""]
    for row in rows:
        lines += [f"## `{row['image']}`", "", f"**Target (agent copy):** {row['suffix']}", ""]
        for model_id in models:
            caption = run_task(row["image"], model_id=model_id)
            lines += [f"**{model_id}:** {caption}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/corpus_val.jsonl")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Florence-2-base-ft", "checkpoints/corpus"],
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default=None, help="Write the report to this file")
    args = parser.parse_args()

    text = report(args.data, args.models, args.limit)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
