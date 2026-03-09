import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _default_response(message: str) -> Dict:
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
        },
        "language_detected": "unknown",
        "engine_used": None,
        "confidence_summary": {"avg_confidence": None, "num_segments": 0},
        "message": message,
        "text_boxes": [],
    }


def _load_image_safe(image_path: str):
    try:
        import cv2
    except Exception as exc:
        return None, f"OpenCV unavailable: {exc}"

    image = cv2.imread(image_path)
    if image is None:
        return None, f"Unreadable image: {image_path}"
    if image.size == 0:
        return None, f"Empty image: {image_path}"
    return image, None


def _build_text_roi(image):
    """Heuristic crop focusing on ID text area while keeping full image fallback."""
    h, w = image.shape[:2]
    # Keep lower 75% and right 85% where most ID text usually appears.
    y1 = int(h * 0.18)
    x1 = int(w * 0.10)
    return image[y1:h, x1:w]


def _preprocess_for_ocr(image) -> List[Tuple[str, object]]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 2.0 if max(h, w) < 1400 else 1.35
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    sharpened = cv2.filter2D(
        denoised,
        -1,
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    )

    roi = _build_text_roi(image)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_upscaled = cv2.resize(roi_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    roi_enhanced = clahe.apply(roi_upscaled)

    return [
        ("full_enhanced", enhanced),
        ("full_otsu", otsu),
        ("full_adaptive", adaptive),
        ("full_sharpened", sharpened),
        ("roi_enhanced", roi_enhanced),
    ]


def _clean_ocr_output(tokens: List[str]) -> List[str]:
    cleaned = []
    for token in tokens:
        # Preserve Arabic/Unicode text; only normalize whitespace.
        text = re.sub(r"\s+", " ", (token or "")).strip()
        if text:
            cleaned.append(text)

    seen = set()
    unique = []
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _detect_language(text_lines: List[str]) -> str:
    text = " ".join(text_lines)
    has_ar = bool(re.search(r"[\u0600-\u06FF]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))

    if has_ar and has_en:
        return "mixed"
    if has_ar:
        return "ar"
    if has_en:
        return "en"
    return "unknown"


def _score_text_quality(tokens: List[str], avg_confidence: float) -> int:
    if not tokens:
        return 0

    long_tokens = sum(1 for t in tokens if len(t) >= 3)
    chars = sum(len(t) for t in tokens)
    conf_bonus = int(max(0.0, min(1.0, avg_confidence)) * 8)
    mixed_bonus = 2 if _detect_language(tokens) == "mixed" else 0
    return (long_tokens * 3) + min(chars // 10, 10) + conf_bonus + mixed_bonus


def _is_weak_result(tokens: List[str], avg_confidence: float) -> bool:
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return True
    if sum(len(t) for t in tokens) < 10:
        return True
    if avg_confidence < 0.25:
        return True
    return False


def _normalize_bbox(points) -> Optional[Dict[str, int]]:
    try:
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
    except Exception:
        return None


def _run_easyocr_variants(image_variants: List[Tuple[str, object]]) -> Tuple[List[str], Optional[str], Optional[str], List[Dict], float]:
    try:
        import easyocr
    except Exception as exc:
        return [], None, f"EasyOCR unavailable: {exc}", [], 0.0

    try:
        reader = easyocr.Reader(["ar", "en"], gpu=False)
    except Exception as exc:
        return [], None, f"Failed to initialize EasyOCR: {exc}", [], 0.0

    best_tokens: List[str] = []
    best_variant = None
    best_score = -1
    best_boxes: List[Dict] = []
    best_avg_conf = 0.0

    for variant_name, img in image_variants:
        for paragraph_mode in (False, True):
            try:
                results = reader.readtext(img, detail=1, paragraph=paragraph_mode)
                tokens = []
                boxes = []
                confidences = []

                for entry in results:
                    if len(entry) < 3:
                        continue
                    bbox, text, conf = entry[0], str(entry[1]), float(entry[2])
                    if conf < 0.12 and len(text.strip()) <= 2:
                        continue
                    tokens.append(text)
                    confidences.append(conf)
                    norm_bbox = _normalize_bbox(bbox)
                    if norm_bbox is not None:
                        boxes.append({"text": text, "confidence": conf, "bbox": norm_bbox})

                cleaned = _clean_ocr_output(tokens)
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                score = _score_text_quality(cleaned, avg_conf)

                if score > best_score:
                    best_score = score
                    best_tokens = cleaned
                    best_boxes = boxes
                    best_avg_conf = avg_conf
                    mode_name = "paragraph" if paragraph_mode else "line"
                    best_variant = f"{variant_name}_{mode_name}"

                if score >= 16 and len(cleaned) >= 4 and avg_conf >= 0.35:
                    break
            except Exception:
                continue

    if not best_tokens:
        return [], None, "EasyOCR inference failed on all preprocessing variants.", [], 0.0

    return best_tokens, best_variant, None, best_boxes, best_avg_conf


def _extract_with_huggingface(image_path: str) -> Tuple[List[str], Optional[str]]:
    """Optional fallback only when EasyOCR output is weak or empty."""
    try:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        lines = _clean_ocr_output([line for line in text.split("\n") if line.strip()])
        if not lines and text.strip():
            lines = _clean_ocr_output([text.strip()])
        return lines, None
    except Exception as exc:
        return [], f"Hugging Face OCR fallback failed: {exc}"


def _search_value(patterns: List[str], lines: List[str]) -> Optional[str]:
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" :-")
                if value:
                    return value
    return None


def _parse_id_fields(text_lines: List[str]) -> Dict[str, Optional[str]]:
    normalized = [re.sub(r"\s+", " ", line).strip() for line in text_lines if line.strip()]

    fields = {
        "full_name": _search_value([r"(?:name|full\s*name|الاسم)\s*[:\-]?\s*(.+)$"], normalized),
        "id_number": _search_value(
            [r"(?:id\s*no|id\s*number|identity\s*no|رقم\s*الهوية)\s*[:\-]?\s*([A-Z0-9\-\u0660-\u0669]{5,})"],
            normalized,
        ),
        "date_of_birth": _search_value(
            [r"(?:dob|date\s*of\s*birth|birth\s*date|تاريخ\s*الميلاد)\s*[:\-]?\s*([0-9\u0660-\u0669]{1,4}[\/\-\.][0-9\u0660-\u0669]{1,2}[\/\-\.][0-9\u0660-\u0669]{1,4})"],
            normalized,
        ),
        "expiry_date": _search_value(
            [r"(?:expiry|exp\.?\s*date|date\s*of\s*expiry|تاريخ\s*الانتهاء)\s*[:\-]?\s*([0-9\u0660-\u0669]{1,4}[\/\-\.][0-9\u0660-\u0669]{1,2}[\/\-\.][0-9\u0660-\u0669]{1,4})"],
            normalized,
        ),
        "nationality": _search_value([r"(?:nationality|الجنسية)\s*[:\-]?\s*([\w\s\u0600-\u06FF]+)$"], normalized),
        "address": _search_value(
            [r"(?:address|العنوان|residence|محل\s*الإقامة|مكان\s*الإقامة)\s*[:\-]?\s*(.+)$"],
            normalized,
        ),
    }

    if fields["id_number"] is None:
        for line in normalized:
            matches = re.findall(r"\b[0-9\u0660-\u0669]{8,20}\b", line)
            if matches:
                fields["id_number"] = matches[0]
                break

    return fields


def extract_text(image_path: str) -> Dict:
    path = Path(image_path)
    if not path.exists():
        return _default_response(f"File not found: {path}")

    image, error = _load_image_safe(str(path))
    if error:
        return _default_response(error)

    variants = _preprocess_for_ocr(image)
    raw_text, variant_used, easyocr_error, text_boxes, avg_conf = _run_easyocr_variants(variants)
    engine_used = "easyocr"

    hf_error = None
    if _is_weak_result(raw_text, avg_conf):
        hf_text, hf_error = _extract_with_huggingface(str(path))
        # HF only replaces output if it clearly contains more usable text.
        if len(" ".join(hf_text)) > len(" ".join(raw_text)) + 6:
            raw_text = hf_text
            engine_used = "huggingface_trocr_fallback"
            variant_used = None
            text_boxes = []
            avg_conf = 0.0

    if not raw_text:
        msg_parts = ["OCR failed to extract readable text."]
        if easyocr_error:
            msg_parts.append(easyocr_error)
        if hf_error:
            msg_parts.append(hf_error)
        return _default_response(" ".join(msg_parts))

    parsed_fields = _parse_id_fields(raw_text)
    language = _detect_language(raw_text)
    joined_text = "\n".join(raw_text)

    message = "OCR extraction completed."
    if _is_weak_result(raw_text, avg_conf):
        message = "OCR extraction completed with weak text signal; try better lighting/straight crop."
    if variant_used:
        message = f"{message} Best preprocessing variant: {variant_used}."

    return {
        "raw_text": raw_text,
        "joined_text": joined_text,
        "parsed_fields": parsed_fields,
        "language_detected": language,
        "engine_used": engine_used,
        "confidence_summary": {
            "avg_confidence": round(avg_conf, 4) if avg_conf else 0.0,
            "num_segments": len(raw_text),
        },
        "message": message,
        "text_boxes": text_boxes,
    }


def extract_template_fields(image_path: str) -> Dict:
    """Template-builder helper that keeps compatibility with existing flow."""
    result = extract_text(image_path)
    parsed = result.get("parsed_fields", {})
    return {
        "full_name": parsed.get("full_name"),
        "id_number": parsed.get("id_number"),
        "birth_date": parsed.get("date_of_birth"),
        "residence": parsed.get("address"),
        "raw_text": result.get("raw_text", []),
        "joined_text": result.get("joined_text", ""),
        "language_detected": result.get("language_detected", "unknown"),
        "engine_used": result.get("engine_used"),
        "confidence_summary": result.get("confidence_summary", {"avg_confidence": None, "num_segments": 0}),
        "message": result.get("message", "OCR extraction completed."),
        "text_boxes": result.get("text_boxes", []),
    }
