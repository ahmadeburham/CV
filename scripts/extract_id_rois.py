#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.id_roi_pipeline import process_id_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Egyptian ID ROIs using template alignment pipeline")
    parser.add_argument("--image", required=True, help="Path to scene image containing ID card")
    parser.add_argument("--template", required=True, help="Path to template image")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = process_id_image(args.image, args.template, args.output_dir)

    if result.get("card_detected"):
        print("Card detected.")
    else:
        print("Card detection failed.")

    print(f"Alignment success: {result.get('alignment_success')}")
    print(f"Results JSON: {Path(args.output_dir) / 'ocr' / 'results.json'}")


if __name__ == "__main__":
    main()
