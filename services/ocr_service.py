from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import cv2
import numpy as np
import easyocr


# Arabic-first OCR reader for Egyptian IDs
# English is still enabled because some IDs/passports may contain Latin text,
# but the parsing logic is built around Arabic-first extraction.
_READER = None


def _get_reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(["ar", "en"], gpu=False)
    return _READER


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
        },
        "language_detected": "unknown",
        "engine_used": "easyocr_ar_first",
        "confidence_summary": {
            "avg_confidence": 0.0,
            "num_segments": 0,
        },
        "message": message,
        "roi_text": {
            "name_lines": [],
            "address_lines": [],
            "id_number_line": [],
            "birth_date_line": [],
        },
    }


def _load_image(image_path: str) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        return None
    return img


def _preprocess_roi(roi: np.ndarray) -> List[np.ndarray]:
    variants = []

    if roi is None or roi.size == 0:
        return variants

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

    # Upscale for better OCR on phone photos
    upscaled = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(upscaled)

    # Sharpen
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    sharpened = cv2.addWeighted(clahe, 1.6, blurred, -0.6, 0)

    # Threshold variants
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    _, otsu = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.extend([gray, upscaled, clahe, sharpened, adaptive, otsu])
    return variants


def _run_easyocr_best(roi: np.ndarray, paragraph: bool = False) -> Tuple[List[str], float]:
    reader = _get_reader()
    best_lines: List[str] = []
    best_conf = 0.0

    for variant in _preprocess_roi(roi):
        results = reader.readtext(variant, paragraph=paragraph, detail=1)

        lines = []
        confs = []

        for entry in results:
            if len(entry) >= 3:
                text = str(entry[1]).strip()
                conf = float(entry[2])
                if text:
                    lines.append(text)
                    confs.append(conf)

        lines = _clean_text_lines(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        if len(" ".join(lines)) > len(" ".join(best_lines)):
            best_lines = lines
            best_conf = avg_conf

        if lines and avg_conf >= 0.25:
            return lines, avg_conf

    return best_lines, best_conf


def _clean_text_lines(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        line = line.replace("|", " ").replace("_", " ").strip()
        if len(line) == 1 and not re.search(r"[\u0600-\u06FFA-Za-z0-9]", line):
            continue
        cleaned.append(line)
    return cleaned


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


def _normalize_digits(text: str) -> str:
    eastern_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(eastern_to_western)


def _extract_id_number(lines: List[str]) -> Optional[str]:
    candidates = []
    for line in lines:
        norm = _normalize_digits(line)
        nums = re.findall(r"\d{10,20}", norm)
        candidates.extend(nums)

    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _extract_birth_date(lines: List[str]) -> Optional[str]:
    for line in lines:
        norm = _normalize_digits(line)
        match = re.search(r"(\d{1,4}[\/\-]\d{1,2}[\/\-]\d{1,4})", norm)
        if match:
            return match.group(1)
    return None


def _pick_name(name_lines: List[str]) -> Optional[str]:
    if not name_lines:
        return None

    filtered = []
    for line in name_lines:
        if re.search(r"(جمهورية|بطاقة|تحقيق|الشخصية|العنوان|محل|الإقامة)", line):
            continue
        filtered.append(line)

    if not filtered:
        return None

    return " ".join(filtered).strip()


def _pick_address(address_lines: List[str]) -> Optional[str]:
    if not address_lines:
        return None

    filtered = []
    for line in address_lines:
        if re.search(r"(رقم|الرقم|القومي|بطاقة|تحقيق الشخصية)", line):
            continue
        filtered.append(line)

    if not filtered:
        return None

    return " - ".join(filtered).strip()


def _parse_fields_from_egyptian_layout(
    name_lines: List[str],
    address_lines: List[str],
    id_lines: List[str],
    birth_lines: List[str],
    all_lines: List[str],
) -> Dict[str, Optional[str]]:
    full_name = _pick_name(name_lines)
    address = _pick_address(address_lines)
    id_number = _extract_id_number(id_lines + all_lines)
    birth_date = _extract_birth_date(birth_lines + all_lines)

    return {
        "full_name": full_name,
        "id_number": id_number,
        "date_of_birth": birth_date,
        "expiry_date": None,
        "nationality": "مصري" if any("جمهورية مصر العربية" in x for x in all_lines) else None,
        "address": address,
    }


def _get_egyptian_id_rois(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Layout tuned to the card form shown by the user:
    - Left side photo
    - Under photo: birth date
    - Right side:
      first two lines = name
      next two lines = residence/address
      last lower-right line = ID number
    """
    h, w = img.shape[:2]

    # Right-side text block
    x_right_start = int(w * 0.48)
    x_right_end = int(w * 0.96)

    # Name: upper-right, two lines
    name_roi = img[
        int(h * 0.22):int(h * 0.50),
        x_right_start:x_right_end
    ]

    # Address / residence: mid-right, next two lines
    address_roi = img[
        int(h * 0.46):int(h * 0.70),
        x_right_start:x_right_end
    ]

    # ID number: lower-right line
    id_roi = img[
        int(h * 0.68):int(h * 0.92),
        int(w * 0.50):int(w * 0.98)
    ]

    # Birth date: lower-left under photo
    birth_roi = img[
        int(h * 0.60):int(h * 0.88),
        int(w * 0.02):int(w * 0.42)
    ]

    # Full card fallback OCR
    full_roi = img[
        int(h * 0.10):int(h * 0.95),
        int(w * 0.02):int(w * 0.98)
    ]

    return {
        "name": name_roi,
        "address": address_roi,
        "id_number": id_roi,
        "birth_date": birth_roi,
        "full": full_roi,
    }


def extract_text(image_path: str) -> Dict:
    path = Path(image_path)
    if not path.exists():
        return _empty_result(f"File not found: {path}")

    img = _load_image(str(path))
    if img is None:
        return _empty_result(f"Unreadable image: {path}")

    rois = _get_egyptian_id_rois(img)

    try:
        name_lines, name_conf = _run_easyocr_best(rois["name"], paragraph=False)
        address_lines, address_conf = _run_easyocr_best(rois["address"], paragraph=False)
        id_lines, id_conf = _run_easyocr_best(rois["id_number"], paragraph=False)
        birth_lines, birth_conf = _run_easyocr_best(rois["birth_date"], paragraph=False)
        full_lines, full_conf = _run_easyocr_best(rois["full"], paragraph=False)
    except Exception as exc:
        return _empty_result(f"OCR failed: {exc}")

    name_lines = _clean_text_lines(name_lines)
    address_lines = _clean_text_lines(address_lines)
    id_lines = _clean_text_lines(id_lines)
    birth_lines = _clean_text_lines(birth_lines)
    full_lines = _clean_text_lines(full_lines)

    # Arabic-first aggregation:
    # prioritize targeted ROIs first, then use full-card OCR as fallback context
    raw_text = []
    raw_text.extend(name_lines)
    raw_text.extend(address_lines)
    raw_text.extend(birth_lines)
    raw_text.extend(id_lines)

    for line in full_lines:
        if line not in raw_text:
            raw_text.append(line)

    raw_text = _clean_text_lines(raw_text)

    parsed_fields = _parse_fields_from_egyptian_layout(
        name_lines=name_lines,
        address_lines=address_lines,
        id_lines=id_lines,
        birth_lines=birth_lines,
        all_lines=raw_text,
    )

    avg_conf = np.mean(
        [x for x in [name_conf, address_conf, id_conf, birth_conf, full_conf] if x > 0]
    ) if any(x > 0 for x in [name_conf, address_conf, id_conf, birth_conf, full_conf]) else 0.0

    return {
        "raw_text": raw_text,
        "joined_text": "\n".join(raw_text),
        "parsed_fields": parsed_fields,
        "language_detected": _detect_language(raw_text),
        "engine_used": "easyocr_ar_first_roi_layout",
        "confidence_summary": {
            "avg_confidence": round(float(avg_conf), 4),
            "num_segments": len(raw_text),
        },
        "message": "OCR extraction completed using Arabic-first ROI-based EasyOCR.",
        "roi_text": {
            "name_lines": name_lines,
            "address_lines": address_lines,
            "id_number_line": id_lines,
            "birth_date_line": birth_lines,
        },
    }


def extract_template_fields(image_path: str) -> Dict:
    return extract_text(image_path)