from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import os

from .id_roi_pipeline import process_id_image


def _empty_result(message: str) -> Dict:
    return {
        "raw_text": [],
        "joined_text": "",
        "parsed_fields": {
            "full_name": None,
            "id_number": None,
            "date_of_birth": None,
            "expiry_date": None,
            "nationality": None,
            "address": None,
            "serial_number": None,
        },
        "language_detected": "unknown",
        "engine_used": "paddleocr_ar_template_pipeline",
        "confidence_summary": {"avg_confidence": 0.0, "num_segments": 0},
        "message": message,
        "template_validation": {
            "template_match": False,
            "template_score": 0.0,
            "method": "ssim_orb_masked",
        },
        "debug": {
            "rectified_saved_path": None,
            "field_crop_paths": {
                "name": None,
                "address": None,
                "id_number": None,
                "birth_date": None,
            },
        },
    }


def extract_text(image_path: str, template_path: Optional[str] = None, output_dir: str = "output") -> Dict:
    image = Path(image_path)
    if not image.exists():
        return _empty_result(f"File not found: {image_path}")

    template = template_path or os.getenv("ID_TEMPLATE_PATH")
    if not template:
        return _empty_result("Template image path is required. Pass template_path or set ID_TEMPLATE_PATH.")

    pipeline_result = process_id_image(str(image), str(template), output_dir)
    if not pipeline_result.get("card_detected"):
        return _empty_result(pipeline_result.get("error", "Card detection failed."))

    raw = pipeline_result.get("raw_ocr", {})
    fields = pipeline_result.get("fields", {})

    raw_text = [
        raw.get("name_line_1", ""),
        raw.get("name_line_2", ""),
        raw.get("birthplace", ""),
        raw.get("address", ""),
        raw.get("birth_date", ""),
        raw.get("id_number", ""),
    ]
    raw_text = [x.strip() for x in raw_text if x and x.strip()]

    validation = pipeline_result.get("template_validation", {})
    template_match = bool(validation.get("template_match", False))

    debug_base = Path(output_dir)
    return {
        "raw_text": raw_text,
        "joined_text": "\n".join(raw_text),
        "parsed_fields": {
            "full_name": fields.get("full_name"),
            "id_number": fields.get("id_number"),
            "date_of_birth": fields.get("birth_date"),
            "expiry_date": None,
            "nationality": None,
            "address": fields.get("address"),
            "serial_number": None,
        },
        "language_detected": "ar",
        "engine_used": "paddleocr_ar_template_pipeline",
        "confidence_summary": {
            "avg_confidence": 0.0,
            "num_segments": len(raw_text),
        },
        "template_validation": validation,
        "message": "Template-based OCR extraction completed." if template_match else "Template validation failed.",
        "debug": {
            "rectified_saved_path": str(debug_base / "card" / "rectified_card.jpg"),
            "field_crop_paths": {
                "name": str(debug_base / "crops" / "name_combined.jpg"),
                "address": str(debug_base / "crops" / "address.jpg"),
                "id_number": str(debug_base / "crops" / "id_number.jpg"),
                "birth_date": str(debug_base / "crops" / "birth_date.jpg"),
            },
        },
        "pipeline": pipeline_result,
    }


def extract_template_fields(image_path: str) -> Dict:
    return extract_text(image_path)
