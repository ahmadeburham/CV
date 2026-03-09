import json
import shutil
from pathlib import Path
from typing import Dict, List


def _load_image_safe(image_path: str):
    try:
        import cv2
    except Exception as exc:
        return None, f"OpenCV unavailable: {exc}"

    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        return None, f"Unreadable image: {image_path}"
    return image, None


def _ensure_dirs(base_dir: str) -> Dict[str, Path]:
    root = Path(base_dir)
    raw_dir = root / "raw"
    cleaned_dir = root / "cleaned"
    metadata_dir = root / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return {"raw": raw_dir, "cleaned": cleaned_dir, "metadata": metadata_dir}


def _align_card_region(image):
    """Best-effort card alignment/crop; falls back to original image."""
    try:
        import cv2
    except Exception:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    img_h, img_w = image.shape[:2]
    area_ratio = (w * h) / max(1, img_w * img_h)
    if area_ratio < 0.35:
        return image

    pad_x = int(0.01 * w)
    pad_y = int(0.01 * h)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return image[y1:y2, x1:x2]


def _expand_bbox(bbox: Dict[str, int], width: int, height: int, pad_ratio: float = 0.12) -> Dict[str, int]:
    w = max(1, bbox["x2"] - bbox["x1"])
    h = max(1, bbox["y2"] - bbox["y1"])
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    return {
        "x1": max(0, bbox["x1"] - pad_x),
        "y1": max(0, bbox["y1"] - pad_y),
        "x2": min(width - 1, bbox["x2"] + pad_x),
        "y2": min(height - 1, bbox["y2"] + pad_y),
    }


def _find_field_boxes(text_boxes: List[Dict], extracted_fields: Dict) -> Dict[str, Dict]:
    field_boxes = {}

    targets = {
        "full_name": extracted_fields.get("full_name"),
        "id_number": extracted_fields.get("id_number"),
        "birth_date": extracted_fields.get("birth_date"),
        "residence": extracted_fields.get("residence"),
    }

    for field_name, field_value in targets.items():
        if not field_value:
            continue

        value_norm = str(field_value).strip().lower()
        for box_entry in text_boxes:
            token = str(box_entry.get("text", "")).strip().lower()
            if not token:
                continue
            if token in value_norm or value_norm in token:
                bbox = box_entry.get("bbox")
                if bbox:
                    field_boxes[field_name] = bbox
                    break

    return field_boxes


def mask_personal_fields(image_path: str, ocr_results: Dict, output_path: str) -> Dict:
    try:
        import cv2
    except Exception as exc:
        return {"success": False, "masked_regions": [], "message": f"OpenCV unavailable: {exc}"}

    image, error = _load_image_safe(image_path)
    if error:
        return {"success": False, "masked_regions": [], "message": error}

    h, w = image.shape[:2]
    text_boxes = ocr_results.get("text_boxes", [])
    field_boxes = _find_field_boxes(text_boxes, ocr_results)

    masked_regions = []
    for field_name, bbox in field_boxes.items():
        b = _expand_bbox(bbox, w, h)
        cv2.rectangle(image, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (255, 255, 255), thickness=-1)
        masked_regions.append({"field": field_name, "bbox": b})

    if not masked_regions:
        return {
            "success": False,
            "masked_regions": [],
            "message": "No personal field regions were confidently located for masking.",
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(out), image)
    if not saved:
        return {"success": False, "masked_regions": [], "message": f"Failed to save cleaned template: {output_path}"}

    return {
        "success": True,
        "masked_regions": masked_regions,
        "message": "Personal fields masked and cleaned template saved.",
    }


def save_template_metadata(metadata_path: str, metadata: Dict) -> Dict:
    try:
        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"success": True, "message": "Template metadata saved."}
    except Exception as exc:
        return {"success": False, "message": f"Failed to save template metadata: {exc}"}


def build_template_from_id(image_path: str, output_dir: str = "templates") -> Dict:
    from .ocr_service import extract_template_fields

    input_path = Path(image_path)
    if not input_path.exists():
        return {"success": False, "message": f"Source template image not found: {input_path}"}

    dirs = _ensure_dirs(output_dir)

    raw_target = dirs["raw"] / input_path.name
    try:
        shutil.copyfile(str(input_path), str(raw_target))
    except Exception as exc:
        return {"success": False, "message": f"Failed to copy raw template image: {exc}"}

    image, error = _load_image_safe(str(raw_target))
    if error:
        return {"success": False, "message": error}

    # Best-effort alignment/crop for better OCR and future matching.
    try:
        import cv2

        aligned = _align_card_region(image)
        if aligned is not None and aligned.size > 0:
            cv2.imwrite(str(raw_target), aligned)
    except Exception:
        pass

    extracted = extract_template_fields(str(raw_target))

    cleaned_name = f"cleaned_{input_path.stem}.jpg"
    cleaned_target = dirs["cleaned"] / cleaned_name

    mask_result = mask_personal_fields(str(raw_target), extracted, str(cleaned_target))
    if not mask_result.get("success"):
        # Fallback: if no boxes were found, still store aligned raw image as cleaned candidate.
        try:
            shutil.copyfile(str(raw_target), str(cleaned_target))
            fallback_message = f"Masking fallback used: {mask_result.get('message')}"
        except Exception as exc:
            return {"success": False, "message": f"Failed to create cleaned template fallback: {exc}"}
    else:
        fallback_message = mask_result.get("message", "")

    metadata_name = f"{Path(cleaned_name).stem}.json"
    metadata_target = dirs["metadata"] / metadata_name

    metadata_payload = {
        "source_image": str(raw_target),
        "cleaned_template_image": str(cleaned_target),
        "fields": {
            "full_name": extracted.get("full_name"),
            "id_number": extracted.get("id_number"),
            "birth_date": extracted.get("birth_date"),
            "residence": extracted.get("residence"),
        },
        "engine_used": extracted.get("engine_used"),
        "language_detected": extracted.get("language_detected", "unknown"),
        "message": fallback_message or extracted.get("message", "Template built."),
        "raw_text": extracted.get("raw_text", []),
        "joined_text": extracted.get("joined_text", ""),
        "masked_regions": mask_result.get("masked_regions", []),
    }

    save_result = save_template_metadata(str(metadata_target), metadata_payload)

    return {
        "success": True,
        "source_image": str(raw_target),
        "cleaned_template_image": str(cleaned_target),
        "metadata_file": str(metadata_target),
        "fields": metadata_payload["fields"],
        "raw_text": extracted.get("raw_text", []),
        "joined_text": extracted.get("joined_text", ""),
        "engine_used": metadata_payload["engine_used"],
        "language_detected": metadata_payload["language_detected"],
        "masked_regions": metadata_payload["masked_regions"],
        "message": (
            f"Template build completed. {save_result.get('message', '')}".strip()
            if save_result.get("success")
            else f"Template built but metadata save failed: {save_result.get('message')}"
        ),
    }
