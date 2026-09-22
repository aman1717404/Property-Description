"""Fine-tune Florence-2 on real estate photos.

The base model captions photos like a generic image captioner ("a two-story
house with a garage"). Fine-tuning on (photo, agent-written copy) pairs teaches
it the vocabulary and tone used in listings.

Training data is a JSONL file, one example per line:

    {"image": "data/sample_property/03_kitchen.jpg",
     "prefix": "<MORE_DETAILED_CAPTION>",
     "suffix": "Renovated u-shaped kitchen with stone benchtops ..."}

Usage:
    python train_florence2.py --data data/train/corpus.jsonl \
        --val data/train/corpus_val.jsonl --epochs 4 --output checkpoints/corpus

With ``--val`` the checkpoint written is the one with the lowest validation
loss, not the last epoch: on a few hundred photos the training loss keeps
falling long after the model starts memorising listings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

from property_agent import no_flash_attn_imports

DEFAULT_TRAIN_MODEL = "microsoft/Florence-2-base-ft"


@dataclass
class Example:
    image: str
    prefix: str
    suffix: str


class ListingCaptionDataset(Dataset):
    """(image, task prompt, target caption) triples read from JSONL."""

    def __init__(self, jsonl_path: str | Path, root: str | Path | None = None):
        self.root = Path(root) if root else Path(jsonl_path).parent.parent.parent
        self.examples: list[Example] = []
        with open(jsonl_path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self.examples.append(
                        Example(
                            image=record["image"],
                            prefix=record.get("prefix", "<MORE_DETAILED_CAPTION>"),
                            suffix=record["suffix"],
                        )
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def _resolve(self, image_path: str) -> Path:
        path = Path(image_path)
        return path if path.exists() else self.root / path

    def __getitem__(self, index: int) -> tuple[str, str, Image.Image]:
        example = self.examples[index]
        image = Image.open(self._resolve(example.image)).convert("RGB")
        return example.prefix, example.suffix, image


def _restore_vision_model_type(config_path: Path) -> None:
    """Write ``vision_config.model_type`` back into a saved config.

    Florence-2 loads its vision config into a plain ``PretrainedConfig``, whose
    ``to_dict`` serialises ``model_type`` as an empty string. Reloading such a
    checkpoint fails with ``only DaViT is supported for now``.
    """
    config = json.loads(config_path.read_text())
    vision_config = config.get("vision_config")
    if isinstance(vision_config, dict) and not vision_config.get("model_type"):
        vision_config["model_type"] = "davit"
        config_path.write_text(json.dumps(config, indent=2) + "\n")


def make_collate_fn(processor):
    def collate(batch):
        prefixes, suffixes, images = zip(*batch)
        inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt")
        labels = processor.tokenizer(
            text=list(suffixes),
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).input_ids
        # The loss must ignore padding, otherwise the model learns to emit <pad>.
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return inputs, labels

    return collate


@torch.inference_mode()
def evaluate(model, loader, device, dtype) -> float:
    model.eval()
    total = 0.0
    for inputs, labels in loader:
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device, dtype=dtype),
            labels=labels.to(device),
        )
        total += outputs.loss.item()
    model.train()
    return total / len(loader)


def train(
    data_path: str,
    model_id: str = DEFAULT_TRAIN_MODEL,
    output_dir: str = "checkpoints/florence2-listings",
    epochs: int = 6,
    batch_size: int = 1,
    lr: float = 1e-6,
    freeze_vision: bool = True,
    val_path: str | None = None,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # fp16 training is unstable without a GradScaler.

    with no_flash_attn_imports():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if freeze_vision:
        # The DaViT encoder already sees these scenes well; only the language
        # decoder needs to learn listing vocabulary, and freezing it roughly
        # halves the memory and step time.
        for parameter in model.vision_tower.parameters():
            parameter.requires_grad = False

    dataset = ListingCaptionDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
    )

    val_loader = None
    if val_path:
        val_loader = DataLoader(
            ListingCaptionDataset(val_path),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(processor),
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=epochs * len(loader)
    )

    output = Path(output_dir)
    best_val = float("inf")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in loader:
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += outputs.loss.item()

        message = f"epoch {epoch}/{epochs}  loss {epoch_loss / len(loader):.4f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, dtype)
            message += f"  val_loss {val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                _save(model, processor, output)
                message += "  (saved)"
        print(message, flush=True)

    if val_loader is None:
        _save(model, processor, output)
    print(f"saved fine-tuned model to {output}")
    return str(output)


def _save(model, processor, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    _restore_vision_model_type(output / "config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/sample_property.jsonl")
    parser.add_argument(
        "--val",
        default=None,
        help="Held-out JSONL; the best-scoring epoch is the one saved",
    )
    parser.add_argument("--model", default=DEFAULT_TRAIN_MODEL)
    parser.add_argument("--output", default="checkpoints/florence2-listings")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--train-vision-tower",
        action="store_true",
        help="Also update the DaViT image encoder (slower, needs more data)",
    )
    args = parser.parse_args()

    train(
        args.data,
        model_id=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_vision=not args.train_vision_tower,
        val_path=args.val,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
