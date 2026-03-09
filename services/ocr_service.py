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
        "message": message,
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


def _preprocess_for_ocr(image) -> List[Tuple[str, object]]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    scale = 2.0 if max(h, w) < 1400 else 1.4
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

    return [
        ("enhanced", enhanced),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("sharpened", sharpened),
    ]


def _clean_ocr_output(tokens: List[str]) -> List[str]:
    cleaned = []
    for token in tokens:
        text = re.sub(r"\s+", " ", token or "").strip()
        if not text:
            continue
        cleaned.append(text)

    seen = set()
    unique = []
    for item in cleaned:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _detect_language(text_lines: List[str]) -> str:
    text = " ".join(text_lines)
    has_ar = bool(re.search(r"[\u0600-\u06FF]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))

    if has_ar and has_en:
        return "ar+en"
    if has_ar:
        return "ar"
    if has_en:
        return "en"
    return "unknown"


def _score_text_quality(tokens: List[str]) -> int:
    if not tokens:
        return 0

    long_tokens = sum(1 for t in tokens if len(t) >= 3)
    chars = sum(len(t) for t in tokens)
    mixed_bonus = 2 if _detect_language(tokens) == "ar+en" else 0
    return (long_tokens * 3) + min(chars // 10, 10) + mixed_bonus


def _is_weak_result(tokens: List[str]) -> bool:
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return True
    if sum(len(t) for t in tokens) < 10:
        return True
    return False


def _run_easyocr_variants(image_variants: List[Tuple[str, object]]) -> Tuple[List[str], Optional[str], Optional[str]]:
    try:
        import easyocr
    except Exception as exc:
        return [], None, f"EasyOCR unavailable: {exc}"

    try:
        reader = easyocr.Reader(["ar", "en"], gpu=False)
    except Exception as exc:
        return [], None, f"Failed to initialize EasyOCR: {exc}"

    best_tokens: List[str] = []
    best_variant = None
    best_score = -1

    for variant_name, img in image_variants:
        for paragraph_mode in (False, True):
            try:
                results = reader.readtext(img, detail=1, paragraph=paragraph_mode)
                tokens = []
                for entry in results:
                    if len(entry) < 3:
                        continue
                    text = entry[1]
                    confidence = float(entry[2])
                    if confidence < 0.15 and len(text.strip()) <= 2:
                        continue
                    tokens.append(text)

                cleaned = _clean_ocr_output(tokens)
                score = _score_text_quality(cleaned)

                if score > best_score:
                    best_score = score
                    best_tokens = cleaned
                    mode_name = "paragraph" if paragraph_mode else "line"
                    best_variant = f"{variant_name}_{mode_name}"

                if score >= 16 and len(cleaned) >= 4:
                    break
            except Exception:
                continue

    if not best_tokens:
        return [], None, "EasyOCR inference failed on all preprocessing variants."

    return best_tokens, best_variant, None


def _extract_with_huggingface(image_path: str) -> Tuple[List[str], Optional[str]]:
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
        "full_name": _search_value(
            [r"(?:name|full\s*name|الاسم)\s*[:\-]?\s*(.+)$"],
            normalized,
        ),
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
        "nationality": _search_value(
            [r"(?:nationality|الجنسية)\s*[:\-]?\s*([\w\s\u0600-\u06FF]+)$"],
            normalized,
        ),
        "address": _search_value(
            [r"(?:address|العنوان)\s*[:\-]?\s*(.+)$"],
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

    raw_text, variant_used, easyocr_error = _run_easyocr_variants(variants)
    engine_used = "easyocr"

    hf_error = None
    if _is_weak_result(raw_text):
        hf_text, hf_error = _extract_with_huggingface(str(path))
        if _score_text_quality(hf_text) > _score_text_quality(raw_text):
            raw_text = hf_text
            engine_used = "huggingface_trocr_fallback"
            variant_used = None

    if not raw_text:
        msg_parts = ["OCR failed to extract readable text."]
        if easyocr_error:
            msg_parts.append(easyocr_error)
        if hf_error:
            msg_parts.append(hf_error)
        return _default_response(" ".join(msg_parts))

    joined_text = "\n".join(raw_text)
    language = _detect_language(raw_text)
    parsed_fields = _parse_id_fields(raw_text)

    message = "OCR extraction completed."
    if _is_weak_result(raw_text):
        message = "OCR extraction completed with weak text signal; verify image quality and orientation."

    if variant_used:
        message = f"{message} Best preprocessing variant: {variant_used}."

    return {
        "raw_text": raw_text,
        "joined_text": joined_text,
        "parsed_fields": parsed_fields,
        "language_detected": language,
        "engine_used": engine_used,
        "message": message,
    }
