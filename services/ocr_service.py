from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import cv2
import numpy as np
import easyocr


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
        "engine_used": "easyocr_ar_first_roi_layout_v2",
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
            "serial_number_line": [],
            "full_lines": [],
        },
    }


def _load_image(image_path: str) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        return None
    return img


def _clean_text_lines(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = str(line).strip()
        if not line:
            continue
        line = line.replace("|", " ").replace("_", " ").replace("—", " ")
        line = re.sub(r"\s+", " ", line).strip()
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


def _preprocess_text_roi(roi: np.ndarray) -> List[np.ndarray]:
    variants = []
    if roi is None or roi.size == 0:
        return variants

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    upscaled = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(upscaled)

    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    sharpened = cv2.addWeighted(clahe, 1.8, blurred, -0.8, 0)

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    _, otsu = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.extend([gray, upscaled, clahe, sharpened, adaptive, otsu])
    return variants


def _preprocess_digits_roi(roi: np.ndarray) -> List[np.ndarray]:
    variants = []
    if roi is None or roi.size == 0:
        return variants

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(upscaled)

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        29,
        9,
    )

    variants.extend([gray, upscaled, clahe, otsu, adaptive])
    return variants


def _run_easyocr_best(
    roi: np.ndarray,
    preprocess_mode: str = "text",
    paragraph: bool = False,
    allowlist: Optional[str] = None,
) -> Tuple[List[str], float]:
    reader = _get_reader()
    best_lines: List[str] = []
    best_conf = 0.0

    variants = _preprocess_digits_roi(roi) if preprocess_mode == "digits" else _preprocess_text_roi(roi)

    for variant in variants:
        results = reader.readtext(
            variant,
            paragraph=paragraph,
            detail=1,
            allowlist=allowlist,
        )

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


def _extract_id_number(lines: List[str]) -> Optional[str]:
    candidates = []

    for line in lines:
        norm = _normalize_digits(line)
        norm = re.sub(r"[^\d]", "", norm)
        found = re.findall(r"\d{12,16}", norm)
        candidates.extend(found)

    if not candidates:
        return None

    exact_14 = [x for x in candidates if len(x) == 14]
    if exact_14:
        return exact_14[0]

    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _extract_birth_date(lines: List[str]) -> Optional[str]:
    for line in lines:
        norm = _normalize_digits(line)
        norm = norm.replace(" ", "")

        m = re.search(r"(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})", norm)
        if m:
            return m.group(1)

        m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})", norm)
        if m:
            return m.group(1)

    return None


def _filter_name_lines(name_lines: List[str]) -> List[str]:
    banned = [
        "جمهورية", "مصر", "العربية", "بطاقة", "تحقيق", "الشخصية",
        "الاسم", "العنوان", "الإقامة", "محل", "الرقم", "القومي"
    ]

    filtered = []
    for line in name_lines:
        if any(word in line for word in banned):
            continue
        if not re.search(r"[\u0600-\u06FF]", line):
            continue
        filtered.append(line)

    return filtered


def _pick_name(name_lines: List[str], full_lines: List[str]) -> Optional[str]:
    filtered = _filter_name_lines(name_lines)

    # if ROI name is weak, try to recover Arabic personal-name-like lines from full OCR
    if len(" ".join(filtered)) < 8:
        extra = []
        for line in full_lines:
            if any(word in line for word in ["جمهورية", "بطاقة", "تحقيق", "الشخصية", "العنوان", "الإقامة"]):
                continue
            if re.search(r"[\u0600-\u06FF]", line):
                extra.append(line)
        filtered.extend(extra)

    if not filtered:
        return None

    text = " ".join(filtered)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # keep only a reasonable number of Arabic tokens for a name
    tokens = text.split()
    if len(tokens) > 6:
        tokens = tokens[:6]

    text = " ".join(tokens).strip()
    return text or None


def _pick_address(address_lines: List[str]) -> Optional[str]:
    if not address_lines:
        return None

    filtered = []
    for line in address_lines:
        if any(word in line for word in ["رقم", "الرقم", "القومي", "بطاقة", "تحقيق", "الشخصية", "الاسم"]):
            continue
        filtered.append(line)

    if not filtered:
        return None

    text = " - ".join(filtered)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _parse_fields_from_egyptian_layout(
    name_lines: List[str],
    address_lines: List[str],
    id_lines: List[str],
    birth_lines: List[str],
    full_lines: List[str],
) -> Dict[str, Optional[str]]:
    full_name = _pick_name(name_lines, full_lines)
    address = _pick_address(address_lines)
    id_number = _extract_id_number(id_lines)
    birth_date = _extract_birth_date(birth_lines)

    nationality = None
    if any("جمهورية مصر العربية" in x or ("جمهورية" in x and "مصر" in x) for x in full_lines):
        nationality = "مصري"

    return {
        "full_name": full_name,
        "id_number": id_number,
        "date_of_birth": birth_date,
        "expiry_date": None,
        "nationality": nationality,
        "address": address,
    }


def _get_egyptian_id_rois(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Stricter layout for the card you showed:
    - photo on left
    - birth date below photo
    - name in upper-right
    - address in mid-right
    - national ID number in lower-right
    - blue strip serial number bottom-left
    """
    h, w = img.shape[:2]

    return {
        "name": img[
            int(h * 0.27):int(h * 0.49),
            int(w * 0.57):int(w * 0.95)
        ],

        "address": img[
            int(h * 0.47):int(h * 0.66),
            int(w * 0.58):int(w * 0.95)
        ],

        "id_number": img[
            int(h * 0.68):int(h * 0.83),
            int(w * 0.52):int(w * 0.96)
        ],

        "birth_date": img[
            int(h * 0.64):int(h * 0.81),
            int(w * 0.05):int(w * 0.39)
        ],

        "serial_number": img[
            int(h * 0.84):int(h * 0.97),
            int(w * 0.04):int(w * 0.33)
        ],

        "full": img[
            int(h * 0.20):int(h * 0.95),
            int(w * 0.03):int(w * 0.97)
        ],
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
        name_lines, name_conf = _run_easyocr_best(rois["name"], preprocess_mode="text", paragraph=False)
        address_lines, address_conf = _run_easyocr_best(rois["address"], preprocess_mode="text", paragraph=False)
        id_lines, id_conf = _run_easyocr_best(
            rois["id_number"],
            preprocess_mode="digits",
            paragraph=False,
            allowlist="0123456789٠١٢٣٤٥٦٧٨٩"
        )
        birth_lines, birth_conf = _run_easyocr_best(
            rois["birth_date"],
            preprocess_mode="digits",
            paragraph=False,
            allowlist="0123456789٠١٢٣٤٥٦٧٨٩/-"
        )
        serial_lines, serial_conf = _run_easyocr_best(
            rois["serial_number"],
            preprocess_mode="digits",
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        full_lines, full_conf = _run_easyocr_best(rois["full"], preprocess_mode="text", paragraph=False)
    except Exception as exc:
        return _empty_result(f"OCR failed: {exc}")

    name_lines = _clean_text_lines(name_lines)
    address_lines = _clean_text_lines(address_lines)
    id_lines = _clean_text_lines(id_lines)
    birth_lines = _clean_text_lines(birth_lines)
    serial_lines = _clean_text_lines(serial_lines)
    full_lines = _clean_text_lines(full_lines)

    # Build readable text, but keep parsing dependent mostly on ROI text
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
        full_lines=full_lines,
    )

    conf_values = [
        x for x in [name_conf, address_conf, id_conf, birth_conf, serial_conf, full_conf] if x > 0
    ]
    avg_conf = float(np.mean(conf_values)) if conf_values else 0.0

    return {
        "raw_text": raw_text,
        "joined_text": "\n".join(raw_text),
        "parsed_fields": parsed_fields,
        "language_detected": _detect_language(raw_text),
        "engine_used": "easyocr_ar_first_roi_layout_v2",
        "confidence_summary": {
            "avg_confidence": round(avg_conf, 4),
            "num_segments": len(raw_text),
        },
        "message": "OCR extraction completed using Arabic-first ROI-based EasyOCR.",
        "roi_text": {
            "name_lines": name_lines,
            "address_lines": address_lines,
            "id_number_line": id_lines,
            "birth_date_line": birth_lines,
            "serial_number_line": serial_lines,
            "full_lines": full_lines,
        },
    }


def extract_template_fields(image_path: str) -> Dict:
    return extract_text(image_path)
