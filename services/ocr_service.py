import re
from pathlib import Path
from typing import Dict, List, Optional


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


def _extract_with_huggingface(image_path: str) -> List[str]:
    """Best-effort OCR attempt with Hugging Face TrOCR."""
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines or ([text.strip()] if text.strip() else [])


def _extract_with_easyocr(image_path: str) -> List[str]:
    import easyocr

    reader = easyocr.Reader(["ar", "en"], gpu=False)
    results = reader.readtext(image_path)
    return [entry[1].strip() for entry in results if len(entry) > 1 and entry[1].strip()]


def _search_value(patterns: List[str], lines: List[str]) -> Optional[str]:
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
    return None


def _parse_fields(text_lines: List[str]) -> Dict[str, Optional[str]]:
    normalized = [line.replace("|", " ").replace("_", " ").strip() for line in text_lines]

    fields = {
        "full_name": _search_value(
            [r"(?:name|full\s*name|الاسم)\s*[:\-]?\s*(.+)$"],
            normalized,
        ),
        "id_number": _search_value(
            [r"(?:id\s*no|id\s*number|رقم\s*الهوية)\s*[:\-]?\s*([A-Z0-9\-]{5,})"],
            normalized,
        ),
        "date_of_birth": _search_value(
            [r"(?:dob|date\s*of\s*birth|تاريخ\s*الميلاد)\s*[:\-]?\s*([0-9]{2,4}[\/\-][0-9]{1,2}[\/\-][0-9]{1,4})"],
            normalized,
        ),
        "expiry_date": _search_value(
            [r"(?:expiry|exp\.?\s*date|تاريخ\s*الانتهاء)\s*[:\-]?\s*([0-9]{2,4}[\/\-][0-9]{1,2}[\/\-][0-9]{1,4})"],
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

    # Generic numeric fallback for ID number if label-based parse failed.
    if fields["id_number"] is None:
        for line in normalized:
            ids = re.findall(r"\b\d{8,20}\b", line)
            if ids:
                fields["id_number"] = ids[0]
                break

    return fields


def extract_text(image_path: str) -> Dict:
    path = Path(image_path)
    if not path.exists():
        return {
            "raw_text": [],
            "parsed_fields": {
                "full_name": None,
                "id_number": None,
                "date_of_birth": None,
                "expiry_date": None,
                "nationality": None,
                "address": None,
            },
            "language_detected": "unknown",
            "message": f"File not found: {path}",
        }

    raw_text: List[str] = []
    path_str = str(path)
    source = None

    # Step 1: Hugging Face OCR attempt.
    try:
        raw_text = _extract_with_huggingface(path_str)
        source = "huggingface_trocr"
    except Exception:
        raw_text = []

    # Step 2: EasyOCR fallback.
    if not raw_text:
        try:
            raw_text = _extract_with_easyocr(path_str)
            source = "easyocr"
        except Exception as exc:
            return {
                "raw_text": [],
                "parsed_fields": {
                    "full_name": None,
                    "id_number": None,
                    "date_of_birth": None,
                    "expiry_date": None,
                    "nationality": None,
                    "address": None,
                },
                "language_detected": "unknown",
                "message": f"OCR failed (HF + EasyOCR): {exc}",
            }

    parsed_fields = _parse_fields(raw_text)
    language = _detect_language(raw_text)

    return {
        "raw_text": raw_text,
        "parsed_fields": parsed_fields,
        "language_detected": language,
        "message": f"OCR extraction completed using {source}.",
    }
