from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import cv2
import numpy as np

from .document_rectifier import rectify_document
from .field_cropper import crop_egyptian_id_fields


try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None

try:
    import easyocr
except Exception:  # noqa: BLE001
    easyocr = None


_PADDLE_OCR = None
_EASY_OCR = None


def _normalize_digits(text: str) -> str:
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _clean_line(text: str) -> str:
    text = str(text).strip()
    text = text.replace("|", " ").replace("_", " ").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
        "engine_used": "paddleocr_ar_primary",
        "confidence_summary": {
            "avg_confidence": 0.0,
            "num_segments": 0,
        },
        "message": message,
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


def _get_paddle_ocr() -> Tuple[Optional[object], Optional[str]]:
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR, None

    if PaddleOCR is None:
        return None, "PaddleOCR is not installed or failed to import."

    try:
        _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="arabic", show_log=False)
        return _PADDLE_OCR, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to initialize PaddleOCR: {exc}"


def _get_easyocr_reader() -> Optional[object]:
    global _EASY_OCR
    if _EASY_OCR is not None:
        return _EASY_OCR
    if easyocr is None:
        return None
    try:
        _EASY_OCR = easyocr.Reader(["ar", "en"], gpu=False)
    except Exception:  # noqa: BLE001
        return None
    return _EASY_OCR


def _preprocess_variants(roi: np.ndarray) -> List[np.ndarray]:
    if roi is None or roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(upscaled)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    return [roi, upscaled, clahe, otsu, adaptive]


def _run_paddle_ocr_best(roi: np.ndarray) -> Tuple[List[str], List[float], bool]:
    ocr, _ = _get_paddle_ocr()
    if ocr is None:
        return [], [], False

    best_lines: List[str] = []
    best_scores: List[float] = []

    for variant in _preprocess_variants(roi):
        image_input = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR) if len(variant.shape) == 2 else variant
        try:
            result = ocr.ocr(image_input, cls=True)
        except Exception:  # noqa: BLE001
            continue

        lines: List[str] = []
        scores: List[float] = []

        if result and len(result) > 0:
            for block in result[0] or []:
                if len(block) < 2 or not block[1]:
                    continue
                text, score = block[1]
                text = _clean_line(text)
                if text:
                    lines.append(text)
                    scores.append(float(score))

        if len(" ".join(lines)) > len(" ".join(best_lines)):
            best_lines, best_scores = lines, scores

        if lines and (sum(scores) / len(scores)) >= 0.35:
            return lines, scores, True

    return best_lines, best_scores, bool(best_lines)


def _run_easyocr_fallback(roi: np.ndarray, allowlist: Optional[str] = None) -> Tuple[List[str], List[float]]:
    reader = _get_easyocr_reader()
    if reader is None or roi is None or roi.size == 0:
        return [], []

    best_lines: List[str] = []
    best_scores: List[float] = []

    for variant in _preprocess_variants(roi):
        try:
            output = reader.readtext(variant, detail=1, paragraph=False, allowlist=allowlist)
        except Exception:  # noqa: BLE001
            continue

        lines: List[str] = []
        scores: List[float] = []
        for entry in output:
            if len(entry) < 3:
                continue
            text = _clean_line(entry[1])
            if text:
                lines.append(text)
                scores.append(float(entry[2]))

        if len(" ".join(lines)) > len(" ".join(best_lines)):
            best_lines, best_scores = lines, scores

    return best_lines, best_scores


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    return ar / max(len(text), 1)


def _extract_id_number(lines: List[str]) -> Optional[str]:
    candidates: List[str] = []
    for line in lines:
        digits = re.sub(r"\D", "", _normalize_digits(line))
        candidates.extend(re.findall(r"\d{12,16}", digits))

    if not candidates:
        return None
    exact_14 = [x for x in candidates if len(x) == 14]
    if exact_14:
        return exact_14[0]
    return sorted(candidates, key=len, reverse=True)[0]


def _extract_date(lines: List[str]) -> Optional[str]:
    for line in lines:
        text = _normalize_digits(line)
        match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
        if match:
            return match.group(1)
        match = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})", text)
        if match:
            return match.group(1)
    return None


def _clean_name(lines: List[str]) -> Optional[str]:
    banned = {"الاسم", "جمهورية", "مصر", "العربية", "بطاقة", "تحقيق", "الشخصية", "الرقم", "القومي"}
    candidates = []
    for line in lines:
        if any(word in line for word in banned):
            continue
        cleaned = re.sub(r"[^\u0600-\u06FF\s]", " ", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= 4 and _arabic_ratio(cleaned) > 0.5:
            candidates.append(cleaned)

    if not candidates:
        return None

    best = max(candidates, key=lambda x: (len(x.split()), len(x)))
    return " ".join(best.split()[:6])


def _clean_address(lines: List[str]) -> Optional[str]:
    banned = {"العنوان", "محل", "الإقامة", "الاسم", "الرقم", "القومي", "بطاقة", "تحقيق", "الشخصية"}
    cleaned_lines = []
    for line in lines:
        if any(word in line for word in banned):
            continue
        line = re.sub(r"\s+", " ", line).strip(" -")
        if len(line) >= 4:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return None
    return " - ".join(cleaned_lines[:3])


def _detect_language(lines: List[str]) -> str:
    text = " ".join(lines)
    has_ar = bool(re.search(r"[\u0600-\u06FF]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_ar and has_en:
        return "mixed"
    if has_ar:
        return "ar"
    if has_en:
        return "en"
    return "unknown"


def _save_debug_images(debug_dir: Path, rectified: np.ndarray, fields: Dict[str, np.ndarray], stem: str) -> Tuple[str, Dict[str, str]]:
    debug_dir.mkdir(parents=True, exist_ok=True)

    rectified_path = debug_dir / f"{stem}_rectified.jpg"
    cv2.imwrite(str(rectified_path), rectified)

    mapping = {
        "name": "name",
        "address": "address",
        "id_number": "id_number",
        "birth_date": "birth_date",
    }

    crop_paths: Dict[str, str] = {k: None for k in mapping}
    for public_name, internal in mapping.items():
        roi = fields.get(internal)
        if roi is None or roi.size == 0:
            continue
        field_path = debug_dir / f"{stem}_{public_name}.jpg"
        cv2.imwrite(str(field_path), roi)
        crop_paths[public_name] = str(field_path)

    return str(rectified_path), crop_paths


def extract_text(image_path: str) -> Dict:
    path = Path(image_path)
    if not path.exists():
        return _empty_result(f"File not found: {path}")

    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return _empty_result(f"Unreadable image: {path}")

    rectified, rect_info = rectify_document(image)
    if rectified is None:
        return _empty_result(f"Document rectification failed: {rect_info.get('message')}")

    fields = crop_egyptian_id_fields(rectified)
    if not fields:
        return _empty_result("Field cropping failed on rectified document.")

    debug_dir = Path("debug")
    rectified_path, crop_paths = _save_debug_images(debug_dir, rectified, fields, path.stem)

    name_lines, name_scores, name_ok = _run_paddle_ocr_best(fields.get("name"))
    address_lines, addr_scores, address_ok = _run_paddle_ocr_best(fields.get("address"))
    id_lines, id_scores, id_ok = _run_paddle_ocr_best(fields.get("id_number"))
    birth_lines, birth_scores, birth_ok = _run_paddle_ocr_best(fields.get("birth_date"))
    full_lines, full_scores, full_ok = _run_paddle_ocr_best(fields.get("full_card_text"))


    if not name_ok:
        fb_lines, fb_scores = _run_easyocr_fallback(fields.get("name"))
        if len(" ".join(fb_lines)) > len(" ".join(name_lines)):
            name_lines, name_scores = fb_lines, fb_scores

    if not address_ok:
        fb_lines, fb_scores = _run_easyocr_fallback(fields.get("address"))
        if len(" ".join(fb_lines)) > len(" ".join(address_lines)):
            address_lines, addr_scores = fb_lines, fb_scores

    if not id_ok:
        fb_lines, fb_scores = _run_easyocr_fallback(fields.get("id_number"), allowlist="0123456789٠١٢٣٤٥٦٧٨٩")
        if len(" ".join(fb_lines)) > len(" ".join(id_lines)):
            id_lines, id_scores = fb_lines, fb_scores

    if not birth_ok:
        fb_lines, fb_scores = _run_easyocr_fallback(fields.get("birth_date"), allowlist="0123456789٠١٢٣٤٥٦٧٨٩/-")
        if len(" ".join(fb_lines)) > len(" ".join(birth_lines)):
            birth_lines, birth_scores = fb_lines, fb_scores


    if not full_ok:
        fb_lines, fb_scores = _run_easyocr_fallback(fields.get("full_card_text"))
        if len(" ".join(fb_lines)) > len(" ".join(full_lines)):
            full_lines, full_scores = fb_lines, fb_scores

    raw_text = []
    for bucket in [name_lines, address_lines, birth_lines, id_lines, full_lines]:
        for line in bucket:
            line = _clean_line(line)
            if line and line not in raw_text:
                raw_text.append(line)

    parsed_fields = {
        "full_name": _clean_name(name_lines),
        "id_number": _extract_id_number(id_lines),
        "date_of_birth": _extract_date(birth_lines),
        "expiry_date": _extract_date(full_lines),
        "nationality": "مصري" if any("جمهورية" in x and "مصر" in x for x in full_lines) else None,
        "address": _clean_address(address_lines),
        "serial_number": _extract_id_number(full_lines),
    }

    scores = name_scores + addr_scores + id_scores + birth_scores + full_scores
    avg_conf = float(np.mean(scores)) if scores else 0.0

    ocr_model, paddle_error = _get_paddle_ocr()
    engine_used = "paddleocr_ar_primary"
    message = "OCR extraction completed with Arabic-first field pipeline."
    if ocr_model is None:
        engine_used = "easyocr_fallback_only"
        message = f"PaddleOCR unavailable. Fallback used where possible. Reason: {paddle_error}"

    result = {
        "raw_text": raw_text,
        "joined_text": "\n".join(raw_text),
        "parsed_fields": parsed_fields,
        "language_detected": _detect_language(raw_text),
        "engine_used": engine_used,
        "confidence_summary": {
            "avg_confidence": round(avg_conf, 4),
            "num_segments": len(raw_text),
        },
        "message": message,
        "debug": {
            "rectified_saved_path": rectified_path,
            "field_crop_paths": crop_paths,
        },
    }

    return result


def extract_template_fields(image_path: str) -> Dict:
    return extract_text(image_path)
