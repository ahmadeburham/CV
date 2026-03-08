from pathlib import Path


def extract_text(image_path: str) -> dict:
    path = Path(image_path)

    if not path.exists():
        return {"text": [], "message": f"File not found: {path}"}

    try:
        import easyocr
    except Exception as exc:
        return {"text": [], "message": f"EasyOCR unavailable: {exc}"}

    try:
        reader = easyocr.Reader(["ar", "en"], gpu=False)
        results = reader.readtext(str(path))
    except Exception as exc:  # EasyOCR/model loading/read errors
        return {"text": [], "message": f"OCR failed: {exc}"}

    extracted_text = [entry[1] for entry in results]

    return {
        "text": extracted_text,
        "message": "OCR extraction completed.",
    }
