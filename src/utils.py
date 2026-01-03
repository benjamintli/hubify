import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


def load_coco_data(coco_path: Path):
    """Load COCO annotation file and return the data."""
    with coco_path.open() as f:
        return json.load(f)


def extract_categories(data: dict) -> dict:
    """Extract category mapping from COCO data (category_id - 1 -> name)."""
    return {cat["id"] - 1: cat["name"] for cat in data.get("categories", [])}


def find_instances_file(directory: Path) -> Path | None:
    """Find an instances*.json file in the given directory."""
    instances_pattern = re.compile(r"^instances.*\.json$")
    instances_files = [
        f
        for f in directory.iterdir()
        if f.is_file() and instances_pattern.match(f.name)
    ]
    return instances_files[0] if instances_files else None


def auto_detect_splits(data_dir: Path) -> dict[str, Path | None]:
    """Auto-detect train/test/validation directories and their annotations."""
    splits = {}

    for split_name in ["train", "test", "validation"]:
        split_dir = data_dir / split_name
        if split_dir.is_dir():
            instances_file = find_instances_file(split_dir)
            splits[split_name] = instances_file
        else:
            splits[split_name] = None

    return splits


def coco_to_metadata(coco_path: Path, out_path: Path):
    data = load_coco_data(coco_path)

    objects = defaultdict(lambda: {"bbox": [], "category": []})
    print(f"Processing annotations from {coco_path.name}...")
    for ann in tqdm(data.get("annotations", []), desc="Loading annotations"):
        img_id = ann["image_id"]
        objects[img_id]["bbox"].append([float(x) for x in ann["bbox"]])
        objects[img_id]["category"].append(int(ann["category_id"]) - 1)

    images = data.get("images", [])
    with out_path.open("w") as f:
        for img in tqdm(images, desc="Writing metadata"):
            img_id = img["id"]
            row = {
                "file_name": img["file_name"],
                "objects": objects.get(img_id, {"bbox": [], "category": []}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(images)} images)")

    return extract_categories(data)


def visualize_sample(
    metadata_path: Path, image_dir: Path, output_path: Path, categories: dict
):
    """Draw bounding boxes on a random sample image and save visualization."""
    # Read all metadata entries
    with metadata_path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print(f"No entries found in {metadata_path}")
        return

    # Pick a random entry with objects
    entries_with_objects = [e for e in entries if e["objects"]["bbox"]]
    if not entries_with_objects:
        print(f"No images with annotations found in {metadata_path}")
        return

    sample = random.choice(entries_with_objects)
    image_path = image_dir / sample["file_name"]

    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return

    # Load image and draw boxes
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Draw each bounding box
    for bbox, cat_id in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
        x, y, w, h = bbox
        # COCO format is [x, y, width, height], convert to corner coordinates
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # Draw category label if available
        if categories and cat_id in categories:
            cat_name = categories[cat_id]
            draw.text((x, y - 15), cat_name, fill="red")

    # Save visualization
    img.save(output_path)
    print(f"Saved visualization to {output_path}")
