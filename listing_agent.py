"""Turn a folder of property photos into a listing record.

The output follows the same schema as the scraped listing corpus
(``data/listings_corpus.json``): a ``property`` object holding the generated
``description`` plus ``units`` and ``imageFolder``.

Usage:
    python listing_agent.py data/sample_property \
        --suburb "Cremorne" --state NSW --postcode 2090 \
        --property-type Apartment --price 890 --out examples/sample_listing.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from property_agent import MODEL_ID, run_task

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Room type -> keywords looked for in the caption and in the detected objects.
ROOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "kitchen": ("kitchen", "dishwasher", "oven", "stove", "cooktop", "refrigerator"),
    "bathroom": ("bathroom", "bathtub", "toilet", "shower", "vanity", "cistern"),
    "bedroom": ("bedroom", "bed ", "nightstand", "wardrobe", "headboard"),
    "living": ("living room", "lounge", "sofa", "couch", "coffee table", "tv"),
    "dining": ("dining", "dining table"),
    "laundry": ("laundry", "washing machine", "dryer"),
    "balcony": ("balcony", "terrace", "patio", "deck"),
    "study": ("study", "home office", "desk and chair"),
    "garage": ("garage", "carport", "parking"),
    "pool": ("swimming pool", "pool"),
    "exterior": (
        "building",
        "house",
        "facade",
        "apartment block",
        "street",
        "garden",
        "front yard",
        "driveway",
    ),
}

# Listing feature phrase -> keywords that must appear in a caption/object label.
FEATURE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Stainless steel dishwasher", ("dishwasher",)),
    ("Oven and cooktop", ("oven", "cooktop", "gas stove", "stove")),
    ("Rangehood over the cooktop", ("fume hood", "rangehood", "range hood")),
    ("Ample kitchen cabinetry and bench space", ("cabinetry", "cabinets", "countertop", "benchtop")),
    ("Full-size fridge space", ("refrigerator", "fridge")),
    ("Separate bathtub and shower", ("bathtub", "bath tub")),
    ("Shower with glass screen", ("shower", "glass door")),
    ("Vanity with mirror", ("mirror", "vanity")),
    ("Internal laundry facilities", ("washing machine", "laundry", "dryer")),
    ("Built-in wardrobes", ("wardrobe", "built-in robe", "closet")),
    ("Spacious open-plan living area", ("living room", "lounge", "open plan", "sofa", "couch")),
    ("Separate dining area", ("dining table", "dining area", "dining room")),
    ("Carpeted floors to living and bedrooms", ("carpet",)),
    ("Timber flooring", ("wooden", "timber floor", "hardwood", "floorboards")),
    ("Tiled wet areas", ("tiles", "tiled")),
    ("Large windows with abundant natural light", ("large window", "natural light", "windows")),
    ("Leafy outlook", ("trees", "greenery", "garden", "shrubs")),
    ("Air conditioning", ("air conditioner", "air conditioning", "split system")),
    ("Ceiling fan", ("ceiling fan",)),
    ("Private balcony", ("balcony", "terrace")),
    ("Off-street parking", ("garage", "carport", "driveway", "parking")),
    ("Swimming pool", ("swimming pool",)),
    ("Fireplace", ("fireplace",)),
    ("Neutral, freshly painted interiors", ("white walls", "painted white", "neutral")),
)

ROOM_SENTENCES: dict[str, str] = {
    "living": "a bright living area",
    "kitchen": "a well-appointed kitchen",
    "bedroom": "generous bedroom accommodation",
    "bathroom": "a full bathroom",
    "dining": "a dedicated dining space",
    "balcony": "private outdoor space",
    "laundry": "internal laundry",
    "study": "a study nook",
    "garage": "secure parking",
    "pool": "a swimming pool",
}


@dataclass
class ImageAnalysis:
    """Florence-2 output for a single photo."""

    path: str
    room: str
    caption: str
    objects: list[str] = field(default_factory=list)


def find_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no images found in {folder}")
    return images


def classify_room(caption: str, objects: list[str]) -> str:
    """Pick the room type whose keywords best match the caption and objects."""
    haystack = (caption + " " + " ".join(objects)).lower()
    scores = {
        room: sum(haystack.count(keyword) for keyword in keywords)
        for room, keywords in ROOM_KEYWORDS.items()
    }
    # An exterior shot is only chosen when no interior room scored.
    interior = {room: score for room, score in scores.items() if room != "exterior"}
    best_room, best_score = max(interior.items(), key=lambda item: item[1])
    if best_score == 0:
        return "exterior" if scores["exterior"] else "other"
    return best_room


def analyze_image(path: str | Path, model_id: str = MODEL_ID) -> ImageAnalysis:
    """Caption one photo and detect the objects in it."""
    path = str(path)
    caption = str(run_task(path, "<MORE_DETAILED_CAPTION>", model_id=model_id)).strip()
    detection = run_task(path, "<OD>", model_id=model_id)
    objects = sorted(set(detection.get("labels", []))) if isinstance(detection, dict) else []
    return ImageAnalysis(
        path=path, room=classify_room(caption, objects), caption=caption, objects=objects
    )


def extract_features(analyses: list[ImageAnalysis]) -> list[str]:
    """Map captions and detected objects onto listing-style feature bullets."""
    haystack = " ".join(
        f"{item.caption} {' '.join(item.objects)}" for item in analyses
    ).lower()
    return [
        phrase
        for phrase, keywords in FEATURE_RULES
        if any(keyword in haystack for keyword in keywords)
    ]


def _count_rooms(analyses: list[ImageAnalysis]) -> Counter[str]:
    return Counter(item.room for item in analyses)


def compose_description(
    analyses: list[ImageAnalysis],
    features: list[str],
    property_type: str,
    suburb: str | None,
    bedrooms: int,
    bathrooms: int,
    car_spaces: int,
) -> str:
    """Write the listing copy: an intro paragraph plus a feature list.

    Mirrors the dominant shape of the scraped corpus -- a positioning sentence,
    a sentence on the layout, then ``Features include:`` bullets.
    """
    rooms = _count_rooms(analyses)
    location = f" in {suburb}" if suburb else ""
    config: list[str] = []
    if bedrooms:
        config.append(f"{bedrooms} bedroom{'s' if bedrooms != 1 else ''}")
    if bathrooms:
        config.append(f"{bathrooms} bathroom{'s' if bathrooms != 1 else ''}")
    if car_spaces:
        config.append(f"{car_spaces} car space{'s' if car_spaces != 1 else ''}")
    config_text = ", ".join(config) if config else "a comfortable layout"

    highlights = [
        ROOM_SENTENCES[room]
        for room, _ in rooms.most_common()
        if room in ROOM_SENTENCES
    ][:3]
    highlight_text = ""
    if highlights:
        if len(highlights) > 1:
            highlight_text = (
                " The floorplan offers "
                + ", ".join(highlights[:-1])
                + f" and {highlights[-1]}."
            )
        else:
            highlight_text = f" The floorplan offers {highlights[0]}."

    intro = (
        f"This well-presented {property_type.lower()}{location} offers {config_text} "
        f"across a light-filled, easy-care layout.{highlight_text}"
    )

    lines = [intro, "", "Features include:"]
    lines.extend(features)
    return "\n".join(lines).strip()


def build_listing(
    image_folder: str | Path,
    property_type: str = "Apartment",
    suburb: str | None = None,
    state: str = "NSW",
    postcode: str = "",
    street: str = "",
    country: str = "Australia",
    price: float | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    car_spaces: int | None = None,
    furnished: bool = False,
    listing_url: str = "",
    model_id: str = MODEL_ID,
) -> dict:
    """Analyze every photo in ``image_folder`` and build a listing record."""
    images = find_images(image_folder)
    analyses = [analyze_image(path, model_id=model_id) for path in images]
    features = extract_features(analyses)
    rooms = _count_rooms(analyses)

    bedrooms = rooms.get("bedroom", 0) if bedrooms is None else bedrooms
    bathrooms = rooms.get("bathroom", 0) if bathrooms is None else bathrooms
    if car_spaces is None:
        car_spaces = 1 if rooms.get("garage") else 0

    description = compose_description(
        analyses, features, property_type, suburb, bedrooms, bathrooms, car_spaces
    )

    folder = Path(image_folder)
    cover = next(
        (item for item in analyses if item.room == "exterior"), analyses[0]
    )
    others = [item.path for item in analyses if item.path != cover.path]

    return {
        "property": {
            "listing_url": listing_url,
            "description": description,
            "address": {
                "street": street,
                "suburb": suburb or "",
                "state": state,
                "postcode": postcode,
                "country": country,
                "city": suburb or "",
            },
            "location": {"city": suburb or "", "state": state, "country": country},
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "car_spaces": car_spaces,
            "furnished": furnished,
            "bills_included": False,
            "is_published": False,
            "amenities": features,
            "images": {"cover": cover.path, "other": others},
            "price": price,
        },
        "units": [
            {
                "listing_type": "room",
                "name": f"Room {index}",
                "price": round(price / bedrooms, 2) if price and bedrooms else None,
                "bond_amount": round(price / bedrooms * 4, 2) if price and bedrooms else None,
                "min_lease": 3,
                "max_lease": 12,
                "max_occupants": 1,
                "status": "active",
                "features": features,
            }
            for index in range(1, (bedrooms or 0) + 1)
        ],
        "imageFolder": folder.name,
        "image_analysis": [asdict(item) for item in analyses],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_folder", help="Folder of photos for one property")
    parser.add_argument("--property-type", default="Apartment")
    parser.add_argument("--street", default="")
    parser.add_argument("--suburb", default=None)
    parser.add_argument("--state", default="NSW")
    parser.add_argument("--postcode", default="")
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument("--bedrooms", type=int, default=None, help="Override the inferred count")
    parser.add_argument("--bathrooms", type=int, default=None, help="Override the inferred count")
    parser.add_argument("--car-spaces", type=int, default=None)
    parser.add_argument("--furnished", action="store_true")
    parser.add_argument("--listing-url", default="")
    parser.add_argument("--model", default=MODEL_ID, help="Model id or fine-tuned checkpoint")
    parser.add_argument("--out", default=None, help="Write the listing JSON here")
    args = parser.parse_args()

    listing = build_listing(
        args.image_folder,
        property_type=args.property_type,
        suburb=args.suburb,
        state=args.state,
        postcode=args.postcode,
        street=args.street,
        price=args.price,
        bedrooms=args.bedrooms,
        bathrooms=args.bathrooms,
        car_spaces=args.car_spaces,
        furnished=args.furnished,
        listing_url=args.listing_url,
        model_id=args.model,
    )

    print(listing["property"]["description"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(listing, indent=2) + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
