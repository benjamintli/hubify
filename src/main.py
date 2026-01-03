import argparse
from pathlib import Path

from src.utils import auto_detect_splits, coco_to_metadata, visualize_sample


def main():
    """Main entry point for the COCO to HuggingFace format converter."""
    parser = argparse.ArgumentParser(
        description="Convert COCO format annotations to HuggingFace metadata JSONL files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing train/test/validation subdirectories",
    )
    parser.add_argument(
        "--train-annotations",
        type=Path,
        help="Optional: Path to training annotations JSON file (overrides auto-detection)",
    )
    parser.add_argument(
        "--test-annotations",
        type=Path,
        help="Optional: Path to test annotations JSON file (overrides auto-detection)",
    )
    parser.add_argument(
        "--validation-annotations",
        type=Path,
        help="Optional: Path to validation annotations JSON file (overrides auto-detection)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate sample visualization images with bounding boxes",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: {args.data_dir} is not a valid directory")
        exit(1)

    # Auto-detect splits
    detected_splits = auto_detect_splits(args.data_dir)

    # Override with manual paths if provided
    annotations = {
        "train": args.train_annotations or detected_splits["train"],
        "test": args.test_annotations or detected_splits["test"],
        "validation": args.validation_annotations or detected_splits["validation"],
    }

    # Process each split
    processed_count = 0
    for split_name, coco_path in annotations.items():
        if coco_path is None:
            continue

        if not coco_path.is_file():
            print(f"Warning: {split_name} annotations file not found: {coco_path}")
            continue

        # Write metadata.jsonl in the split directory under data_dir
        split_dir = args.data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "metadata.jsonl"
        categories = coco_to_metadata(coco_path, out_path)
        processed_count += 1

        # Optionally visualize a sample
        if args.visualize:
            vis_output = Path("sample_visualization.jpg")
            visualize_sample(out_path, split_dir, vis_output, categories)

    if processed_count == 0:
        warning_msg = f"""Warning: No annotation files found or processed in {args.data_dir}
Expected structure:
  data-dir/
    ├── train/instances*.json
    ├── validation/instances*.json
    └── test/instances*.json (optional)

Or use --train-annotations, --validation-annotations, --test-annotations to specify paths manually."""
        print(warning_msg)


if __name__ == "__main__":
    main()
